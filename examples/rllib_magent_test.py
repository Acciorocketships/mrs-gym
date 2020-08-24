from mrsgym import *
import gym
import torch
import torch.nn as nn

import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.agents.callbacks import DefaultCallbacks


def main():
	ray.init()
	ModelCatalog.register_custom_model('CustomModel', MLPmodel)
	# CustomDistribution = rllib_distribution(torch.distributions.Normal)
	# ModelCatalog.register_custom_action_dist("CustomDistribution", CustomDistribution)
	register_env("mrsgym", lambda cfg: MRS_RLlib_MultiAgent(config=cfg))
	Policy = PPOTorchPolicy.with_updates(name="PPOPolicy")
	Trainer = PPOTrainer.with_updates(name="PPOTrainer", default_policy=Policy)

	N = 3
	low = np.array([0.,-1.,-1.,-1.])
	high = np.array([10.,1.,1.,1.,])
	action_space = gym.spaces.Box(np.tile(low,(N,1)), np.tile(high, (N,1)), dtype=np.float64)

	env_config = {
					'state_fn': state_fn,
					'reward_fn': reward_fn,
					'done_fn': done_fn,
					'start_fn': start_fn,
					'N_AGENTS': N,
					'STATE_SIZE': 6,
					'ACTION_TYPE': "set_control",
					'action_space': action_space,
				 }
	config = {
				'env': "mrsgym",
				'env_config': env_config,
				'num_workers': 1,
				'model' : {
					# 'custom_action_dist': 'CustomDistribution',
					'custom_model': 'CustomModel',
				},
				'framework': 'torch',
			 }
	stop_config = {
					'training_iteration': 100,
				  }

	# trainer = Trainer(config=config, env='mrsgym')
	# for _ in range(1000):
	# 	trainer.train()
	ray.tune.run(PPOTrainer, config=config, stop=stop_config)



class MLPmodel(TorchModelV2, nn.Module):

	def __init__(self, obs_space, action_space, num_outputs, model_config, name):
		TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
		nn.Module.__init__(self)
		input_dim = obs_space.shape[-1]
		# Actor
		self.L1a = nn.Linear(input_dim, 16)
		self.L2a = nn.Linear(16, 32)
		self.L3a = nn.Linear(32, num_outputs)
		# Critic
		self.L1c = nn.Linear(num_outputs+input_dim, 16)
		self.L2c = nn.Linear(16, 8)
		self.L3c = nn.Linear(8, 1)

	def forward(self, input_dict, state, seq_lens):
		x = input_dict["obs"]
		# Actor
		y1 = self.L1a(x)
		y1 = nn.functional.relu(y1)
		y1 = self.L2a(y1)
		y1 = nn.functional.relu(y1)
		y1 = self.L3a(y1)
		logits = nn.functional.relu(y1)
		# Critic
		y2 = self.L1c(torch.cat([x,logits], dim=-1))
		y2 = nn.functional.relu(y2)
		y2 = self.L2c(y2)
		y2 = nn.functional.relu(y2)
		value = self.L3c(y2)
		self.value = value.view(-1)
		# Return
		return logits, []

	def value_function(self):
		return self.value



def start_fn(gymenv):
	target_vel = torch.randn(3)
	gymenv.set_data("target_vel", target_vel)


def reward_fn(env, obs, action, obs_next):
	reward = {}
	for name, s in obs.items():
		reward[name] = -np.linalg.norm(s[:3])
	return reward


def state_fn(quad):
	target_vel = quad.get_data("target_vel")
	return np.array(torch.cat([target_vel - quad.get_vel(), quad.get_ori()]))



def done_fn(env, obs, step):
	for name in obs.keys():
		agent = env.agents[env.names_dict[name]]
		if agent.collision():
			return {"__all__": True}
	return {"__all__": False}


if __name__ == '__main__':
	main()

