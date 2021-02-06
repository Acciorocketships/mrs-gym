import pickle
from helper.Reynolds import *
from helper.MRSAnalytics import *
from helper.Plotter import *
from helper.Trainer import *
from helper.DataGenerator import *
from torch.distributions import *
from mrsgym.Util import *
import os
from matplotlib import pyplot as plt



def main():
	# Parameters
	N = 12
	D = 6
	K = 1
	COMM_RANGE = 2.5
	datapoints = 1000
	episode_length = 200
	headless = True
	leader = True
	# File Paths
	dir_path = "data"
	evaldata_path = os.path.join(dir_path, "%s_data.pt")
	try:
		os.mkdir(dir_path)
	except FileExistsError:
		pass
	# Initialise Models
	reynolds = Reynolds(N, D, 1, 3)
	models = {"reynolds": reynolds, "random": RandomController(OUT_DIM=3)} # we will compare reynolds flocking to a random model
	# Create Environment
	z = Uniform(low=2.0*torch.ones(N,1), high=5.0*torch.ones(N,1))
	xy_normal = Normal(torch.zeros(N,2), 1.0)
	dist = CombinedDistribution([xy_normal, z], mixer='cat', dim=1) # create custom starting state distribution
	env = gym.make('mrs-v0', state_fn=state_fn, update_fn=update_fn, N_AGENTS=N, START_POS=dist, K_HOPS=1, COMM_RANGE=COMM_RANGE, ACTION_TYPE='set_target_vel', HEADLESS=headless)
	startpos = [env.generate_start_pos() for _ in range(int(datapoints/episode_length*2))]
	env.START_POS = StartPosGenerator(startpos) # use the same starting state for each model
	# Generate and Analyse Data
	analysers = {}
	for name, model in models.items():
		print(name)
		data = Trainer(K=K)
		is_data_loaded = data.load_trainer(path=evaldata_path % name) # load simulation data if it exists
		if not is_data_loaded: # generate data if it does not exist
			data.save_trainer_onexit(path=evaldata_path % name)
			simulate(env=env, model=model, trainer=data, datapoints=datapoints, episode_length=episode_length, leader=leader)
		analysers[name] = MRSAnalytics(data) # compute flocking metrics (separation, cohesion, leader dist)
		analysers[name].name = name
	# Draw Plots
	plot_separation(*analysers.values())
	plot_cohesion(*analysers.values())
	plot_leader_dist(*analysers.values())
	# Show
	show_plots()



def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])

def update_fn(env, action, **kwargs):
	for i, agent in enumerate(env.agents):
		env.add_line(start=[0.,0.,0.], end=action[i,:], parent=agent, name="line_%d" % i, colour=[0.,0.,1.])

def simulate(env, model, leader=False, **kwargs):
	if leader:
		leader_action_policy = RandomAction()
		action_fn = leader_action_policy.action_fn
		environment = env.get_env()
		leader_agent = environment.agents[0]
		environment.set_colour(leader_agent, [1.,0.,0.])
	else:
		action_fn = lambda action, state: action
	env.START_POS.reset()
	data = generate_mrs(env, model=model, action_fn=action_fn, **kwargs)
	return data

def plot_separation(*analysers):
	separations = [analyser.separation().permute(0,2,1).reshape(analyser.N*analyser.num_episodes, analyser.episode_length) for analyser in analysers]
	names = [analyser.name for analyser in analysers]
	plot_time_distribution(data=separations, labels=names, xlabel="Time Step", ylabel="Separation from Nearest Neighbor", title="", ignorenan=True)
	for name, separation in zip(names, separations):
		sep = separation.reshape(-1)
		mean = mean_ignorenan(sep.unsqueeze(1))[0]
		median = median_ignorenan(sep.unsqueeze(1))[0]
		std = std_ignorenan(sep.unsqueeze(1))[0]
		print("%s Separation: median=%g, mean=%g, std=%g" % (name, median, mean, std))

def plot_cohesion(*analysers):
	cohesions = [analyser.cohesion() for analyser in analysers]
	names = [analyser.name for analyser in analysers]
	plot_time_distribution(data=cohesions, labels=names, xlabel="Time Step", ylabel="Diameter of Smallest Sphere that Contains all Agents", title="", ignorenan=True)
	for name, cohesion in zip(names, cohesions):
		coh = cohesion.reshape(-1)
		mean = mean_ignorenan(coh.unsqueeze(1))[0]
		median = median_ignorenan(coh.unsqueeze(1))[0]
		std = std_ignorenan(coh.unsqueeze(1))[0]
		print("%s Cohesion: median=%g, mean=%g, std=%g" % (name, median, mean, std))

def plot_leader_dist(*analysers):
	dists = [analyser.dist_to_leader() for analyser in analysers]
	names = [analyser.name for analyser in analysers]
	plot_time_distribution(data=dists, labels=names, xlabel="Time Step", ylabel="Distance from the Center of the Swarm to the Leader", title="", ignorenan=True)
	for name, dist in zip(names, dists):
		leader_dist = dist.reshape(-1)
		mean = mean_ignorenan(leader_dist.unsqueeze(1))[0]
		median = median_ignorenan(leader_dist.unsqueeze(1))[0]
		std = std_ignorenan(leader_dist.unsqueeze(1))[0]
		print("%s Leader Dist: median=%g, mean=%g, std=%g" % (name, median, mean, std))

def loss_fn(output, target):
	loss = torch.nn.MSELoss()(output, target.double())
	loss_dict = {"Loss": loss}
	return loss_dict

# Generates an off-policy random action (used for creating a leader)
class RandomAction:

	def __init__(self):
		torch.manual_seed(0)
		self.target_vel = torch.zeros(3)
		self.sigma = 0.05
		self.maxspeed = 1.0
		self.dist = Normal(self.target_vel, self.sigma)

	def action_fn(self, action, state):
		# Update target_vel
		self.dist = Normal(self.target_vel, self.sigma)
		self.target_vel = self.dist.sample()
		mag = self.target_vel.norm()
		self.target_vel *= self.maxspeed / max(mag, self.maxspeed)
		# Set Leader Velocity
		action[0,:] = self.target_vel
		return action

# Stores the starting states in simulation so they can be reused with other models for consistency
class StartPosGenerator:

	def __init__(self, pos_list):
		self.pos_list = pos_list
		self.i = 0

	def sample(self):
		startpos = self.pos_list[self.i]
		self.i += 1
		return startpos

	def reset(self):
		self.i = 0

# Random model
class RandomController:

	def __init__(self, OUT_DIM):
		self.OUT_DIM = OUT_DIM

	def forward(self, A, X):
		batch, N, D, K = X.shape
		return torch.randn(batch, N, self.OUT_DIM)



if __name__ == '__main__':
	main()