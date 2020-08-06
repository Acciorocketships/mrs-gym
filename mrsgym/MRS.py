from mrsgym.BulletSim import *
from mrsgym.Environment import *
import gym
from gym import spaces
from collections import deque
from torch.distributions.normal import Normal
import torch
import time
import math

class MRS(gym.Env):

	metadata = {'render.modes': ['headless', 'bullet']}

	# state_fn:: input: Quadcopter, output: size D tensor
	# reward_fn:: input: Environment, output: scalar
	# info_fn:: input: Environment, output: dict
	def __init__(self, state_fn=None, reward_fn=None, info_fn=None, env='simple', **kwargs):
		super(MRS, self).__init__()
		# Inputs
		self.state_fn = state_fn
		self.reward_fn = reward_fn
		self.info_fn = info_fn
		self.env = env
		# Constants
		self.N_AGENTS = 1
		self.K_HOPS = 0
		self.AGENT_RADIUS = 0.25
		self.COMM_RANGE = float('inf')
		self.RETURN_A = True
		self.ACTION_TYPE = "set_target_vel"
		self.set_constants(kwargs)
		# Constants that depend on other constants
		self.START_DISTRIBUTION = Normal(torch.tensor([0.,0.,2.]), 1.0) # must have sample() method implemented. can generate size (N,3) or (3,)
		self.START_ORI = torch.tensor([-math.pi/2,0,0,math.pi/2,0,0]) # shape (N,6) or (N,3) or (6,) or (3,).
		if len(self.START_ORI.shape)==1:
			self.START_ORI = self.START_ORI.expand(self.N_AGENTS, -1)
		self.set_constants(kwargs)
		# Data
		self.X = deque([])
		self.A = deque([])
		self.steps_since_reset = 0
		self.last_loop_time = time.monotonic()
		# Simulation
		BulletSim.setup()
		if isinstance(env, str):
			self.env = env_generator(self.N_AGENTS, env)
		# Setup
		self.reset()


	def set_constants(self, kwargs):
		for name, val in kwargs.items():
			if name in self.__dict__:
				self.__dict__[name] = val
			elif hasattr(BulletSim, name):
				setattr(BulletSim, name, val)


	def calc_A(self, X):
		if self.COMM_RANGE == float('inf'):
			return torch.ones(self.N_AGENTS, self.N_AGENTS)
		copos = self.get_relative_position(X[:,:3])
		codist = copos.norm(dim=2)
		codist.diagonal().fill_(float('inf'))
		A = (codist < self.COMM_RANGE).float()
		return A


	def generate_start_pos(self):
		startpos = self.START_DISTRIBUTION.sample()
		sample_one_agent = False
		if len(startpos.shape)==1:
			startpos = torch.stack([self.START_DISTRIBUTION.sample() for _ in range(self.N_AGENTS)], dim=0)
			sample_one_agent = True
		codist = self.get_relative_position(startpos).norm(dim=2)
		codist.diagonal().fill_(float('inf'))
		while torch.any(codist < 2*self.AGENT_RADIUS):
			idxs = torch.where(codist < 2*self.AGENT_RADIUS)[0]
			if sample_one_agent:
				for idx in idxs:
					startpos[idx,:] = self.START_DISTRIBUTION.sample()
			else:
				startposnew = self.START_DISTRIBUTION.sample()
				startpos[idxs,:] = startposnew[idxs,:]
			codist = self.get_relative_position(startpos).norm(dim=2)
			codist.diagonal().fill_(float('inf'))
		return startpos


	def generate_start_ori(self):
		if self.START_ORI.shape[1] == 3:
			return self.START_ORI
		else:
			return randrange(self.START_ORI[:,:3],self.START_ORI[:,3:])


	# Input: N x 3
	# Output: N x N x 3
	def get_relative_position(self, X):
		N = X.shape[0]
		posi = X.unsqueeze(1).expand(-1,N,-1)
		posj = X.unsqueeze(0).expand(N,-1,-1)
		return posi-posj


	def reset(self, pos=None, ori=None, vel=None, angvel=None):
		if pos is None:
			pos = self.generate_start_pos()
		if ori is None:
			ori = self.generate_start_ori()
		if vel is None:
			vel = torch.zeros(self.N_AGENTS, 3)
		if angvel is None:
			angvel = torch.zeros(self.N_AGENTS, 3)
		self.env.set_state(pos=pos, ori=ori, vel=vel, angvel=angvel)


	def close(self):
		pass


	def render(self, mode='bullet', close=False):
		pass

	# waits for a given dt. If dt is not given, it uses the value from the simulator
	def wait(self, dt=None):
		if dt is None:
			dt = BulletSim.DT
		diff = time.monotonic() - self.last_loop_time
		waittime = max(dt-diff, 0)
		time.sleep(waittime)


	def step(self, actions):
		## Set Action and Step Environment ##
		# actions: N x ACTION_DIM
		if not isinstance(actions, torch.Tensor):
			actions = torch.tensor(actions)
		self.env.set_actions(actions, behaviour=self.ACTION_TYPE)
		BulletSim.step_sim()
		## Collect Data ##
		# X: N x D
		X = self.env.get_X(self.state_fn)
		self.X.appendleft(X)
		if len(self.X) > self.K_HOPS+1:
			self.X.pop()
		for _ in range(self.K_HOPS+1 - len(self.X)):
			self.X.append(X)
		Xk = torch.stack(list(self.X), dim=2).squeeze(2) # Xk: N x D x K+1
		# A: N x N
		if self.RETURN_A:
			A = self.calc_A(X)
			self.A.appendleft(A)
			if len(self.A) > self.K_HOPS+1:
				self.A.pop()
			for _ in range(self.K_HOPS+1 - len(self.A)):
				self.A.append(torch.zeros(self.N_AGENTS, self.N_AGENTS))
			Ak = torch.stack(list(self.A), dim=2).squeeze(2) # Ak: N x N x K+1
		# reward: scalar
		reward = None
		if self.reward_fn is not None:
			reward = self.reward_fn(self.env)
		# info: dict
		info = {}
		if self.info_fn is not None:
			info = self.info_fn(self.env)
		if self.RETURN_A:
			info["A"] = Ak
		# done: N (bool tensor)
		done = self.env.get_done()
		# time
		self.last_loop_time = time.monotonic()
		# return
		return X, reward, done, info


def randrange(low, high):
	if not isinstance(low, torch.Tensor):
		low = torch.tensor(low)
		high = torch.tensor(high)
	x = torch.rand(low.shape)
	x *= (high - low)
	x += low
	return x
