from mrsgym import *
import gym
import torch
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.agents.callbacks import DefaultCallbacks


def train():
	ray.init()
	# ModelCatalog.register_custom_model('CustomModel', MLPmodel)
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
				# 'model' : {
				# 	'custom_model': 'CustomModel',
				# },
				'framework': 'torch',
			 }
	stop_config = {
					'training_iteration': 5000,
				  }

	# trainer = Trainer(config=config, env='mrsgym')
	# for _ in range(1000):
	# 	trainer.train()
	ray.tune.run(PPOTrainer, config=config, stop=stop_config)


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
	env = MRS_RLlib_MultiAgent(config=env_config)
	env.set_data("target_vel", torch.randn(3))
	while True:
		actions = {'agent0': torch.tensor([5.,0.,0.,0.]), 'agent1': torch.tensor([5.,0.,0.,0.]), 'agent2': torch.tensor([5.,0.,0.,0.])}
		obs, reward, done, info = env.step(actions)
		env.wait()


if __name__ == '__main__':
	train()

