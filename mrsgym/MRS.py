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
	# reward_fn:: input: Environment, X, A, Xlast, action, steps_since_reset; output: scalar
	# done_fn:: input: Environment, X, A, Xlast, action, steps_since_reset; output: bool
	# info_fn:: input: Environment, X, A, Xlast, action, steps_since_reset; output: dict
	# update_fn:: input: Environment, X, A, Xlast, action, steps_since_reset
	def __init__(self, state_fn=None, reward_fn=None, done_fn=None, info_fn=None, update_fn=None, start_fn=None, env='simple', **kwargs):
		super(MRS, self).__init__()
		# Constants
		self.N_AGENTS = 1
		self.K_HOPS = 0
		self.STATE_SIZE = 0
		self.ACTION_DIM = 0
		self.AGENT_RADIUS = 0.3
		self.COMM_RANGE = float('inf')
		self.RETURN_A = False
		self.RETURN_EVENTS = False
		self.ACTION_TYPE = "set_target_vel"
		self.HEADLESS = False
		self.MAX_TIMESTEPS = float('inf')
		self.set_constants(kwargs)
		# Inputs
		self.state_fn = state_fn
		self.reward_fn = reward_fn if (reward_fn is not None) else (lambda **kwargs: 0.0)
		self.done_fn = done_fn if (done_fn is not None) else (lambda **kwargs: kwargs["steps_since_reset"] >= self.MAX_TIMESTEPS)
		self.info_fn = info_fn if (info_fn is not None) else (lambda **kwargs: {})
		self.update_fn = update_fn
		self.start_fn = start_fn
		if isinstance(env, Environment):
			self.N_AGENTS = len(env.agents)
			self.env = env
			self.sim = self.env.sim
		elif isinstance(env, str):
			self.sim = BulletSim(**kwargs)
			self.env = env_generator(envtype=env, N=self.N_AGENTS, sim=self.sim)
		# Constants that depend on other constants
		if self.ACTION_DIM == 0:
			if "set_target" in self.ACTION_TYPE:
				self.ACTION_DIM = 3
			else:
				self.ACTION_DIM = 4
		self.observation_space = Box(np.full((self.N_AGENTS,self.STATE_SIZE,self.K_HOPS+1), -np.inf, dtype=np.float32), np.full((self.N_AGENTS,self.STATE_SIZE,self.K_HOPS+1), np.inf, dtype=np.float32))
		self.action_space = Box(np.full((self.N_AGENTS,self.ACTION_DIM), -np.inf, dtype=np.float32), np.full((self.N_AGENTS,self.ACTION_DIM), np.inf, dtype=np.float32))
		self.START_POS = Normal(torch.tensor([0.,0.,2.]), 1.0) # must have sample() method implemented. can generate size (N,3) or (3,)
		self.START_ORI = torch.tensor([0,0,-np.pi/2,0,0,np.pi/2]) # shape (N,6) or (N,3) or (6,) or (3,).
		if len(self.START_ORI.shape)==1:
			self.START_ORI = self.START_ORI.expand(self.N_AGENTS, -1)
		self.set_constants(kwargs)
		# Data
		self.X = deque([])
		self.A = deque([])
		self.steps_since_reset = 0
		self.last_action = None
		self.last_obs = None
		self.last_loop_time = time.monotonic()
		self.is_initialised = False


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
		Xk = self.get_Xk()
		return Xk


	def get_Xk(self):
		return torch.stack(list(self.X), dim=0) # Xk: K+1 x N x D


	def calc_Ak(self):
		A = self.calc_A() # A: N x N
		self.A.appendleft(A)
		if len(self.A) > self.K_HOPS+1:
			self.A.pop()
		for _ in range(self.K_HOPS+1 - len(self.A)):
			self.A.append(torch.zeros(self.N_AGENTS, self.N_AGENTS))
		Ak = self.get_Ak()
		return Ak


	def get_Ak(self):
		return torch.stack(list(self.A), dim=0) # Ak: K+1 x N x N


	def calc_A(self):
		if self.COMM_RANGE == float('inf'):
			return torch.ones(self.N_AGENTS, self.N_AGENTS) - torch.eye(self.N_AGENTS)
		copos = self.get_relative_position(self.env.get_pos())
		codist = copos.norm(dim=2)
		codist.diagonal().fill_(float('inf'))
		A = (codist <= self.COMM_RANGE).float()
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
			collisions = codist < 2*self.AGENT_RADIUS
			idxs = []
			while torch.sum(collisions) != 0:
				idx = torch.mode(torch.where(collisions)[0])[0]
				idxs.append(idx)
				collisions[idx,:] = 0
				collisions[:,idx] = 0
			idxs = torch.tensor(idxs)
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
	def get_relative_position(self, pos):
		N = pos.shape[0]
		posi = pos.unsqueeze(1).expand(-1,N,-1)
		posj = pos.unsqueeze(0).expand(N,-1,-1)
		return posi-posj


	# Resets the agents (defaults will be used for parameters that aren't given)
	def reset(self, pos=None, ori=None, vel=None, angvel=None):
		self.is_initialised = True
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
		if self.start_fn is not None:
			self.start_fn(self)
		Xk = self.calc_Xk()
		self.last_obs = Xk
		return Xk


	# Like reset, except the values that aren't given will be left unchanged (instead of using defaults)
	def set(self, pos=None, ori=None, vel=None, angvel=None):
		self.env.set_state(pos=pos, ori=ori, vel=vel, angvel=angvel)
		self.X = deque([])
		self.A = deque([])
		self.steps_since_reset = 0
		if self.start_fn is not None:
			self.start_fn(self)
		Xk = self.calc_Xk()
		self.last_obs = Xk
		return Xk


	def set_data(self, name, val):
		self.env.set_data(name, val)


	def get_data(self, name):
		return self.env.get_data(name)


	def __del__(self):
		self.close()


	def close(self):
		self.sim.stop()


	def render(self, mode='bullet', close=False):
		if close:
			self.close()

	# waits for a given dt. If dt is not given, it uses the value from the simulator
	def wait(self, dt=None):
		if dt is None:
			dt = self.sim.DT
		diff = time.monotonic() - self.last_loop_time
		waittime = max(dt-diff, 0)
		time.sleep(waittime)


	def step(self, actions, ACTION_TYPE=None):
		## Set Action and Step Environment ##
		# init
		if not self.is_initialised:
			self.reset()
		# actions: N x ACTION_DIM
		if actions is not None:
			actions = totensor(actions).detach()
			if torch.any(np.isnan(actions)):
				raise Exception('The given action contains NaN:\n %s' % str(actions))
			self.last_action = actions
			if ACTION_TYPE is None:
				ACTION_TYPE = self.ACTION_TYPE
			self.env.set_actions(actions, behaviour=ACTION_TYPE)
		self.env.update_controlled()
		self.sim.step_sim()
		## Collect Data ##
		Xk = self.calc_Xk()
		Ak = self.calc_Ak()
		# update function
		self.env.draw_links(Ak[:,:,0])
		if self.update_fn is not None:
			self.update_fn(env=self.env, X=Xk, A=Ak, Xlast=self.last_obs, action=self.last_action, steps_since_reset=self.steps_since_reset)
		# reward: scalar
		reward = self.reward_fn(env=self.env, X=Xk, A=Ak, Xlast=self.last_obs, action=self.last_action, steps_since_reset=self.steps_since_reset)
		self.last_obs = Xk
		# info: dict
		info = self.info_fn(env=self.env, X=Xk, A=Ak, Xlast=self.last_obs, action=self.last_action, steps_since_reset=self.steps_since_reset)
		info["A"] = Ak
		if self.RETURN_EVENTS:
			info["keyboard_events"] = self.env.get_keyboard_events()
			info["mouse_events"] = self.env.get_mouse_events()
		# done: N (bool tensor)
		done = self.done_fn(env=self.env, X=Xk, A=Ak, Xlast=self.last_obs, action=self.last_action, steps_since_reset=self.steps_since_reset)
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

		
