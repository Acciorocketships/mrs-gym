from mrsgym.MRS import *
import gym
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# RLlib Notes
# • make sure to give STATE_SIZE so that the observation_space is set to the proper size
# • by default, the action_space is between 0 and 1. you can specify an action_fn to transform given actions (or give your own action_space)


class MRS_RLlib_MultiAgent(MRS, MultiAgentEnv):

	defaults = {
		"state_fn": lambda quad: torch.tensor([]),
		"reward_fn": lambda env: 0.0,
		"done_fn": lambda env, steps: torch.any(torch.tensor([agent.collision() for agent in env.agents.values()])),
		"info_fn": lambda env: {},
		"update_fn": None,
		"env": "simple",
		"ACTION_TYPE": "set_target_pos",
		"action_fn": lambda action: action,
		"STATE_SIZE": 0,
		"ACTION_DIM": 3,
		"N_AGENTS": 1,
	}

	def __init__(self, config={}):
		parameters = MRS_RLlib_MultiAgent.defaults; parameters.update(config)
		super(MRS_RLlib_MultiAgent, self).__init__(**parameters)
		self.action_space = spaces.Box(np.zeros(self.ACTION_DIM), np.ones(self.ACTION_DIM),dtype=np.float64)
		self.observation_space = spaces.Box(np.zeros(self.STATE_SIZE), np.ones(self.STATE_SIZE),dtype=np.float64)
		self.convert_agents_to_dict(self.env)
		self.reset()

	def convert_agents_to_dict(self, env):
		num_agents = len(env.agents)
		agent_names = [("agent%d" % i) for i in range(num_agents)]
		agent_dict = {agent_names[i]: env.agents[i] for i in range(num_agents)}
		env.agents = agent_dict
		env.agent_names = agent_names

	def switch_function_names(self, names):
		for name in names:
			setattr(self, name+"_mrs", getattr(self, name))
			setattr(self, name, getattr(self, "_"+name))

	def calc_obs(self):
		return {agent_name: self.state_fn(self.env.agents[agent_name]) for agent_name in self.env.agent_names}

	def reset(self):
		if not isinstance(self.env.agents, dict):
			return
		self.steps_since_reset = 0
		self.N_AGENTS = len(self.env.agents)
		pos = self.generate_start_pos()
		ori = self.generate_start_ori()
		vel = torch.zeros(self.N_AGENTS, 3)
		angvel = torch.zeros(self.N_AGENTS, 3)
		for i, agent_name in enumerate(self.env.agent_names):
			self.env.agents[agent_name].set_state(pos=pos[i,:], ori=ori[i,:], vel=vel[i,:], angvel=angvel[i,:])
		obs = self.calc_obs()
		self.last_obs = obs
		return obs

	def step(self, actions):
		# Set Actions
		self.last_action = actions
		for agent_name, action in actions.items():
			getattr(self.env.agents[agent_name], self.ACTION_TYPE)(self.action_fn(action))
		# Custom Updates
		self.env.update_controlled()
		if self.update_fn is not None:
			self.update_fn(self.env)
		# Step Simulation
		self.sim.step_sim()
		# Return Values
		obs = self.calc_obs()
		reward = self.reward_fn(self.env, self.last_obs, self.last_action, obs)
		done = self.done_fn(self.env, obs, self.steps_since_reset)
		info = self.info_fn(self.env)
		self.last_obs = obs
		# Time
		self.last_loop_time = time.monotonic()
		self.steps_since_reset += 1
		# Return
		return obs, reward, done, info



class MRS_RLlib(MRS):

	defaults = {
		"state_fn": lambda quad: torch.tensor([]),
		"reward_fn": lambda env: 0.0,
		"done_fn": lambda env, steps: torch.any(torch.tensor([agent.collision() for agent in env.agents.values()])),
		"info_fn": lambda env: {},
		"update_fn": None,
		"env": "simple",
		"ACTION_TYPE": "set_target_pos",
		"action_fn": lambda action: action,
		"STATE_SIZE": 0,
		"ACTION_DIM": 3,
		"N_AGENTS": 1,
	}

	def __init__(self, config={}):
		parameters = MRS_RLlib_MultiAgent.defaults; parameters.update(config)
		super(MRS_RLlib_MultiAgent, self).__init__(**parameters)
		self.action_fn = parameters.get("action_fn", lambda action: action)
		self.switch_function_names(["step"])


	def switch_function_names(self, names):
		for name in names:
			setattr(self, name+"_mrs", getattr(self, name))
			setattr(self, name, getattr(self, "_"+name))


	def _step(self, actions):
		actions = self.action_fn(actions)
		return self.step_mrs(actions)
