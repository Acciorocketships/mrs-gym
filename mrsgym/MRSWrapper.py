from mrsgym.MRS import *
import gym
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

defaults = {
	"state_fn": lambda quad: torch.tensor([]),
	"reward_fn": lambda env: 0.0,
	"done_fn": lambda env, steps: torch.any(torch.tensor([agent.collision() for agent in self.agents]))
	"info_fn": lambda env: {},
	"env": "simple",
	"ACTION_TRANSFORM": lambda action: action,
	"STATE_SIZE": 0,
	"ACTION_DIM": 3,
	"N_AGENTS": 1,
}

class MRS_RLlib_MultiAgent(MRS, MultiAgentEnv):

	def __init__(self, config=defaults):
		MRS.__init__(**config)
		self.action_space = spaces.Box(np.zeros(self.ACTION_DIM), np.ones(self.ACTION_DIM))
		self.observation_space = spaces.Box(np.zeros(self.STATE_SIZE), np.ones(self.STATE_SIZE))
		self.convert_agents_to_dict()

	def convert_agents_to_dict(self, env):
		num_agents = len(env.agents)
		agent_names = [("agent%d" % i) for i in range(num_agents)]
		agent_dict = {agent_names[i]: env.agents[i] for i in range(num_agents)}
		env.agents = agent_dict
		self.agent_names = agent_names
		env.agent_names = agent_names

	def switch_function_names(self, names):
		for name in names:
			setattr(self, name+"_mrs", getattr(self, name))
			setattr(self, name, getattr(self, "_"+name))

	def calc_obs(self):
		return {agent_name: self.state_fn(self.env.agents[agent_name]) for agent_name in self.agent_names}

	def calc_reward(self):
		return self.reward_fn(self.env)

	def calc_done(self):
		return 

	def calc_obs(self):
		return {agent_name: self.state_fn(self.env.agents[agent_name]) for agent_name in self.agent_names}

	
	def reset(self):
		self.N_AGENTS = len(self.env.agents)
		pos = self.generate_start_pos()
		ori = self.generate_start_ori()
		vel = torch.zeros(self.N_AGENTS, 3)
		angvel = torch.zeros(self.N_AGENTS, 3)
		for i, agent_name in enumerate(self.agent_names):
			self.env.agents[agent_name].set_state(pos=pos[i,:], ori=ori[i,:], vel=vel[i,:], angvel=angvel[i,:])
		return self.calc_obs()

	def step(self, actions):
		# Set Actions
		for agent_name, action in actions.items():
			getattr(self.env.agents[agent_name], self.ACTION_TYPE)(action)
		# Step Simulation
		BulletSim.step_sim()
		# Return Values
		obs = self.calc_obs()
		reward = self.calc_reward()
		done = self.calc_done()
		info = self.calc_info()
		# Time
		self.last_loop_time = time.monotonic()
		self.steps_since_reset += 1
		# Return
		return obs, reward, done, info

