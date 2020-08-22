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

	N = 3
	low = np.array([0.,-1.,-1.,-1.])
	high = np.array([10.,1.,1.,1.,])
	action_space = gym.spaces.Box(np.tile(low,(N,1)), np.tile(high, (N,1)), dtype=np.float64)

	env_config = {
					'state_fn': state_fn,
					'reward_fn': reward_fn,
					'N_AGENTS': N,
					'STATE_SIZE': 6,
					'ACTION_TYPE': "set_control",
					'action_space': action_space,
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


def reward_fn(env, obs, action, obs_next):
	reward = {}
	for name, s in obs.items():
		reward[name] = -s[:3].norm()
	return reward


def state_fn(quad):
	target_vel = quad.get_data("target_vel")
	return torch.cat([target_vel - quad.get_vel(), quad.get_ori()])


def run_env():
	N = 3
	low = np.array([0.,-1.,-1.,-1.])
	high = np.array([10.,1.,1.,1.,])
	action_space = gym.spaces.Box(np.tile(low,(N,1)), np.tile(high, (N,1)), dtype=np.float64)
	env_config = {
					'state_fn': state_fn,
					'reward_fn': reward_fn,
					'N_AGENTS': N,
					'STATE_SIZE': 6,
					'ACTION_TYPE': "set_control",
					'action_space': action_space,
				 }
	env = gym.make('mrs-rllib-multiagent-v0', config=env_config)
	env.set_data("target_vel", torch.randn(3))
	while True:
		actions = {'agent0': torch.tensor([5.,0.,0.,0.]), 'agent1': torch.tensor([5.,0.,0.,0.]), 'agent2': torch.tensor([5.,0.,0.,0.])}
		X, reward, done, info = env.step(actions)
		import pdb; pdb.set_trace()
		env.wait()


if __name__ == '__main__':
	run_env()