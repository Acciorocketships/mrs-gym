from mrsgym.MRS import *
import gym
from gym.spaces import Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# RLlib Notes
# • make sure to give STATE_SIZE so that the observation_space is set to the proper size
# • by default, the action_space is between 0 and 1. you can specify an action_fn to transform given actions (or give your own action_space)


class MRS_RLlib(MRS):

	def __init__(self, config={}):
		super(MRS_RLlib, self).__init__(**config)
		self.action_fn = config.get("action_fn", lambda action: action)

	def step(self, actions):
		actions = self.action_fn(actions)
		return super(MRS_RLlib, self).step(actions)



class MRS_RLlib_MultiAgent(MRS_RLlib, MultiAgentEnv):

	defaults = {
		"reward_fn": lambda env, obs, action, obs_next: {name: 0.0 for name in obs.keys()},
		"done_fn": lambda env, obs, steps: {"__all__": False},
		"K_HOPS": 0,
	}

	def __init__(self, config={}):
		params = MRS_RLlib_MultiAgent.defaults.copy()
		params.update(config)
		super(MRS_RLlib_MultiAgent, self).__init__(params)
		import pdb; pdb.set_trace()
		self.observation_space = Box(self.observation_space.low[0,:,0], self.observation_space.high[0,:,0], dtype=np.float32)
		self.action_space = Box(self.action_space.low[0,:], self.action_space.high[0,:], dtype=np.float32)
		agent_names = [("agent%d" % (idx+1)) for idx in range(len(self.env.agents))]
		self.names_dict = {agent_names[idx]: idx for idx in range(len(agent_names))}
		self.env.names_dict = self.names_dict

	def calc_Xk(self):
		X = self.env.get_X(self.state_fn)
		self.X.appendleft(X)
		return {name: np.array(X[idx,:]) for name, idx in self.names_dict.items()}

	def step(self, actions):
		N = len(self.env.agents)
		actions_array = np.zeros((N, self.ACTION_DIM))
		for name, action in actions.items():
			idx = self.names_dict[name]
			actions_array[idx,:] = action
		return super(MRS_RLlib_MultiAgent, self).step(actions_array)