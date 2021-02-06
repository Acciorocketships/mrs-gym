from helper.Trainer import *
from helper.DataGenerator import *
from helper.Reynolds import *
from helper.Util import *
import mrsgym
from torch.distributions import *
import torch
import gym
import os


def main():

	N = 12
	COMM_RANGE = 2.5

	headless = False
	leader = True

	dir_name = "data"
	try:
		os.mkdir(dir_name)
	except FileExistsError:
		pass

	data_name = 'flocking_N=%d' % N
	data_path = os.path.join(dir_name, '%s.pt' % data_name)
	data = Trainer(data_path=data_path)

	z = Uniform(low=2.0*torch.ones(N,1), high=5.0*torch.ones(N,1)) # uniform between 0 and 1 in the z direction
	xy_normal = Normal(torch.zeros(N,2), 1.25) # gaussian in the xy direction
	dist = CombinedDistribution([xy_normal, z], mixer='cat', dim=1)

	model = Reynolds(N=N, D=6, K=1, OUT_DIM=3)

	env = gym.make('mrs-v0', state_fn=state_fn, update_fn=update_fn, done_fn=done_fn, N_AGENTS=N, START_POS=dist, K_HOPS=1, COMM_RANGE=COMM_RANGE, ACTION_TYPE='set_target_vel', HEADLESS=headless)

	if leader:
		leader_action_policy = RandomAction()
		action_fn = leader_action_policy.action_fn
		environment = env.get_env()
		leader_agent = environment.agents[0]
		environment.set_colour(leader_agent, [1.,0.,0.])
	else:
		action_fn = lambda action, state: action

	data.save_trainer_onexit()
	data = generate_mrs(env=env, model=model, action_fn=action_fn, trainer=data, datapoints=1000, episode_length=200)


# add action_fn input to generate_mrs for leader


def done_fn(A, **kwargs):
	degree = A[:,:,0].sum(dim=1)
	if torch.any(degree == 0):
		return True
	return False


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])


def update_fn(env, action, **kwargs):
	for i, agent in enumerate(env.agents):
		env.add_line(start=[0.,0.,0.], end=action[i,:], parent=agent, name="line_%d" % i, colour=[0.,0.,1.])


class RandomAction:

	def __init__(self):
		# torch.manual_seed(1)
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



if __name__ == '__main__':
	main()