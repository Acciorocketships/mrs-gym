from mrsgym import *
import torch
import gym
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune.integration.wandb import WandbLoggerCallback
from ray import tune
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


project = "quadcontroller"
training_iter = 100


def run():

	ray.init(local_mode=True)

	register_env("mrs", MRS_RLlib)

	env_config = {
		"N_AGENTS": 1,
		"ACTION_TYPE": "set_control",
		"MAX_TIMESTEPS": 100,
		"start_fn": start_fn,
		"state_fn": state_fn,
		"reward_fn": reward_fn,
		"STATE_SIZE": 9,
		"ACTION_DIM": 4,
		}

	tune.run(
		PPOTrainer,
		checkpoint_freq = 1,
		stop = {"training_iteration": training_iter},
		config = {
			"framework": "torch",
			"env": "mrs",
			"env_config": env_config,
			"num_workers": 1,
		},
		callbacks=[WandbLoggerCallback(
			project = project,
			group = "ppo",
			api_key = "c872091b231c8ae89224e04844d538c14423bdb2",
			log_config = False)]
	)


def state_fn(quad):
	return torch.cat([quad.get_data("target_vel") - quad.get_vel(), quad.get_ori(), quad.get_angvel()])


def start_fn(env):
	agent = env.env.agents[0]
	target_vel = 3 * torch.rand(3)
	agent.set_data("target_vel", target_vel)


def reward_fn(**kwargs):
	x = kwargs["X"][0,0,:]
	vel_e = x[:3]
	angvel = x[6:9]
	reward = -5 * vel_e.norm() + -1 * angvel.norm()
	return reward.item()


class CustomTorchModel(TorchModelV2):
	def __init__(self, obs_space, action_space, num_outputs, model_config, name):
		pass
	def forward(self, input_dict, state, seq_lens):
		pass
	def value_function(self):
		pass


if __name__ == '__main__':
	run()