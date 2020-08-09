from mrsgym.Quadcopter import *
from mrsgym.Object import *
import torch

class Environment:

	def __init__(self):
		self.agents = []
		self.objects = []


	def init_agents(self, N):
		self.agents = [Quadcopter() for _ in range(N)]


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


# add/read debug parameter, add debug text

# enable/disable collisions (single and pairwise)


# simple environment generator
def env_generator(N=1, envtype='simple'):
	env = Environment()
	env.init_agents(N)
	if envtype == 'simple':
		ground = Object("plane.urdf", pos=[0,0,0], ori=[0,0,0])
		env.objects.append(ground)
	return env