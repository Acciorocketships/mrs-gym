from mrsgym.BulletSim import *
from mrsgym.Util import *
from mrsgym.EnvCreator import *
import gym
from gym.spaces import Box
from collections import deque
from torch.distributions import *
import torch
import time
import numpy as np

class MRS(gym.Env):

	metadata = {'render.modes': ['headless', 'bullet']}

	# state_fn:: input: Quadcopter; output: size D tensor
	# reward_fn:: input: Environment; output: scalar
	# done_fn:: input: Environment, steps_since_reset; output: bool
	# info_fn:: input: Environment; output: dict
	def __init__(self, state_fn, reward_fn=None, done_fn=None, info_fn=None, update_fn=None, env='simple', **kwargs):
		super(MRS, self).__init__()
		# Constants
		self.N_AGENTS = 1
		self.K_HOPS = 0
		self.STATE_SIZE = 0
		self.ACTION_DIM = 3
		self.AGENT_RADIUS = 0.25
		self.COMM_RANGE = float('inf')
		self.RETURN_A = True
		self.RETURN_EVENTS = True
		self.ACTION_TYPE = "set_target_vel"
		self.HEADLESS = False
		self.MAX_TIMESTEPS = 1000
		self.set_constants(kwargs)
		# Inputs
		self.state_fn = state_fn
		self.reward_fn = reward_fn if (reward_fn is not None) else (lambda env: 0.0)
		self.done_fn = done_fn if (done_fn is not None) else (lambda env, steps: steps >= self.MAX_TIMESTEPS)
		self.info_fn = info_fn if (info_fn is not None) else (lambda env: {})
		self.update_fn = update_fn
		self.env = env
		# Constants that depend on other constants
		self.observation_space = Box(np.zeros((self.N_AGENTS,self.STATE_SIZE,self.K_HOPS+1)), np.ones((self.N_AGENTS,self.STATE_SIZE,self.K_HOPS+1)), dtype=np.float64)
		self.action_space = Box(np.zeros((self.N_AGENTS,self.ACTION_DIM)), np.ones((self.N_AGENTS,self.ACTION_DIM)), dtype=np.float64)
		self.START_POS = Normal(torch.tensor([0.,0.,2.]), 1.0) # must have sample() method implemented. can generate size (N,3) or (3,)
		self.START_ORI = torch.tensor([0,0,-np.pi/2,0,0,np.pi/2]) # shape (N,6) or (N,3) or (6,) or (3,).
		if len(self.START_ORI.shape)==1:
			self.START_ORI = self.START_ORI.expand(self.N_AGENTS, -1)
		self.set_constants(kwargs)
		# Data
		self.X = deque([])
		self.A = deque([])
		self.steps_since_reset = 0
		self.last_loop_time = time.monotonic()
		# Simulation
		self.sim = BulletSim(**kwargs)
		if isinstance(env, str):
			self.env = env_generator(envtype=env, N=self.N_AGENTS, sim=self.sim)
		# Setup
		self.reset()


	def set_constants(self, kwargs):
		for name, val in kwargs.items():
			if name in self.__dict__:
				self.__dict__[name] = val


	def calc_Xk(self):
		X = self.env.get_X(self.state_fn) # X: N x D
		self.X.appendleft(X)
		if len(self.X) > self.K_HOPS+1:
			self.X.pop()
		for _ in range(self.K_HOPS+1 - len(self.X)):
			self.X.append(X)
		Xk = torch.stack(list(self.X), dim=2).squeeze(2) # Xk: N x D x K+1
		return Xk


	def calc_Ak(self):
		A = self.calc_A(self.X[-1]) # A: N x N
		self.A.appendleft(A)
		if len(self.A) > self.K_HOPS+1:
			self.A.pop()
		for _ in range(self.K_HOPS+1 - len(self.A)):
			self.A.append(torch.zeros(self.N_AGENTS, self.N_AGENTS))
		Ak = torch.stack(list(self.A), dim=2).squeeze(2) # Ak: N x N x K+1
		return Ak


	def calc_A(self, X):
		if self.COMM_RANGE == float('inf'):
			return torch.ones(self.N_AGENTS, self.N_AGENTS)
		copos = self.get_relative_position(X[:,:3])
		codist = copos.norm(dim=2)
		codist.diagonal().fill_(float('inf'))
		A = (codist < self.COMM_RANGE).float()
		return A


	def generate_start_pos(self):
		if isinstance(self.START_POS, torch.Tensor):
			return self.START_POS
		startpos = self.START_POS.sample()
		sample_one_agent = False
		if len(startpos.shape)==1:
			startpos = torch.stack([self.START_POS.sample() for _ in range(self.N_AGENTS)], dim=0)
			sample_one_agent = True
		codist = self.get_relative_position(startpos).norm(dim=2)
		codist.diagonal().fill_(float('inf'))
		while torch.any(codist < 2*self.AGENT_RADIUS):
			idxs = torch.where(codist < 2*self.AGENT_RADIUS)[0]
			if sample_one_agent:
				for idx in idxs:
					startpos[idx,:] = self.START_POS.sample()
			else:
				startposnew = self.START_POS.sample()
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


	# Resets the agents (defaults will be used for parameters that aren't given)
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
		self.X = deque([])
		self.A = deque([])
		self.steps_since_reset = 0


	# Assigns a given state to the agents without "resetting" (the state for parameters that aren't given wont be changed)
	def set(self, pos=None, ori=None, vel=None, angvel=None):
		pass


	def close(self):
		pass


	def render(self, mode='bullet', close=False):
		pass

	# waits for a given dt. If dt is not given, it uses the value from the simulator
	def wait(self, dt=None):
		if dt is None:
			dt = self.sim.DT
		diff = time.monotonic() - self.last_loop_time
		waittime = max(dt-diff, 0)
		time.sleep(waittime)


	def step(self, actions, ACTION_TYPE=None):
		## Set Action and Step Environment ##
		# actions: N x ACTION_DIM
		if actions is not None:
			if not isinstance(actions, torch.Tensor):
				actions = torch.tensor(actions)
			if ACTION_TYPE is None:
				ACTION_TYPE = self.ACTION_TYPE
			self.env.set_actions(actions, behaviour=ACTION_TYPE)
		self.env.update_controlled()
		if self.update_fn is not None:
			self.update_fn(self.env)
		self.sim.step_sim()
		## Collect Data ##
		Xk = self.calc_Xk()
		if self.RETURN_A:
			Ak = self.calc_Ak()
		# reward: scalar
		reward = self.reward_fn(self.env)
		# info: dict
		info = self.info_fn(self.env)
		if self.RETURN_A:
			info["A"] = Ak
		if self.RETURN_EVENTS:
			info["keyboard_events"] = self.env.get_keyboard_events()
			info["mouse_events"] = self.env.get_mouse_events()
		# done: N (bool tensor)
		done = self.done_fn(self.env, self.steps_since_reset)
		# time
		self.last_loop_time = time.monotonic()
		self.steps_since_reset += 1
		# return
		return Xk, reward, done, info


	def get_env(self):
		return self.env

	def get_objects(self):
		return self.env.objects

	def get_agents(self):
		return self.env.agents

	def get_controlled(self):
		return self.env.controlled

	def get_object_dict(self):
		return self.env.object_dict

		
