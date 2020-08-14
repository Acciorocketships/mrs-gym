from mrsgym.Quadcopter import *
from mrsgym.Object import *
import torch

class Environment:

	def __init__(self):
		self.agents = []
		self.objects = []


	def init_agents(self, N):
		self.agents = [Quadcopter() for _ in range(N)]


	def add_object(self, obj):
		self.objects.append(obj)


	def remove_object(self, obj):
		if isinstance(obj, int):
			del self.objects[obj]
		elif isinstance(obj, Object):
			idx = self.objects.index(obj)
			del self.objects[idx]


	def add_agent(self, agent=None):
		if agent is None:
			agent = Quadcopter()
		self.agents.append(agent)
		return agent


	def remove_agent(self, agent):
		if isinstance(agent, int):
			del self.agents[agent]
		elif isinstance(obj, Quadcopter):
			idx = self.agents.index(agent)
			del self.agents[idx]


	def get_X(self, state_fn):
		X = list(map(state_fn, self.agents))
		X = torch.stack(X, dim=0)
		return X


	def set_actions(self, actions, behaviour='set_controls'):
		for i, agent in enumerate(self.agents):
			getattr(agent, behaviour)(actions[i,:])


	def get_done(self):
		return torch.tensor([agent.collision() for agent in self.agents])


	def set_state(self, pos, ori, vel, angvel):
		for i, agent in enumerate(self.agents):
			agent.set_state(pos=pos[i,:], ori=ori[i,:], vel=vel[i,:], angvel=angvel[i,:])

	def get_pos(self):
		return torch.stack([agent.get_pos() for agent in self.agents], dim=0)

	def get_vel(self):
		return torch.stack([agent.get_vel() for agent in self.agents], dim=0)

	def get_ori(self):
		return torch.stack([agent.get_ori() for agent in self.agents], dim=0)

	def get_angvel(self):
		return torch.stack([agent.get_angvel() for agent in self.agents], dim=0)


# add/read debug parameter, add debug text

# enable/disable collisions (single and pairwise)


# simple environment generator
def env_generator(N=1, envtype='simple'):
	env = Environment()
	env.init_agents(N)
	if envtype == 'simple':
		ground = Object("plane.urdf", pos=[0,0,0], ori=[0,0,0])
		env.add_object(ground)
	return env