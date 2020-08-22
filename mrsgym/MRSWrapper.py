from mrsgym.MRS import *
import gym
from gym import spaces
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

	def __init__(self, config={}):
		config.update({"K_HOPS": 0})
		super(MRS_RLlib_MultiAgent, self).__init__(config)
		agent_names = [("agent%d" % idx) for idx in range(len(self.env.agents))]
		self.names_dict = {agent_names[idx]: idx for idx in range(len(agent_names))}

	def calc_Xk(self):
		X = self.env.get_X(self.state_fn)
		self.X.appendleft(X)
		return {name: X[idx,:] for name, idx in self.names_dict.items()}

	def step(self, actions):
		N = len(self.env.agents)
		actions_tensor = torch.zeros(N, self.ACTION_DIM)
		for name, action in actions.items():
			idx = self.names_dict[name]
			actions_tensor[idx,:] = action
		return super(MRS_RLlib_MultiAgent, self).step(actions_tensor)