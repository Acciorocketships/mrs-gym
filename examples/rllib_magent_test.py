from mrsgym import *
import gym
import torch
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.callbacks import DefaultCallbacks

class SetGoal(DefaultCallbacks):
	def on_episode_start(self, worker, base_env, policies, episode, **kwargs):
		target_vel = torch.randn(3)
		base_env.set_data("target_vel", target_vel)


def train():
	ray.init()
	ModelCatalog.register_custom_model('CustomModel', MLPmodel)
	Policy = PPOTorchPolicy.with_updates(name="PPOPolicy")
	Trainer = PPOTrainer.with_updates(name="PPOTrainer", default_policy=policy)
	env_config = {
					'state_fn': state_fn,
					'N_AGENTS': 3,
					'STATE_SIZE': 6,
				 }
	config = {
				'env': 'mrs-rllib-multiagent-v0',
				'env_config': env_config,
				'num_workers': 1,
				'callbacks': SetGoal,
				'model' : {
					'custom_model': 'CustomModel',
				},
				'framework': 'torch',
			 }
	stop_config = {
					'training_iteration': 1000,
				  }
	ray.tune.run(Trainer, config=config, stop=stop_config)


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])


def run_env():
	N = 3
	env = gym.make('mrs-rllib-multiagent-v0', config={"state_fn": state_fn, "N_AGENTS":N})
	while True:
		actions = {'agent0': torch.tensor([1.0,0.0,1.0]), 'agent1': torch.tensor([0.0,1.0,1.0]), 'agent2': torch.tensor([-1.0,0.0,1.0])}
		X, reward, done, info = env.step(actions)
		env.wait()


if __name__ == '__main__':
	train()