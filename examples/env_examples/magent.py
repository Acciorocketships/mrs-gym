from mrsgym import *
import gym
import torch

def main():
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
					'STATE_SIZE': 3,
					'ACTION_TYPE': "set_control",
					'action_space': action_space,
				 }
	env = MRS_RLlib_MultiAgent(config=env_config)
	while True:
		actions = {'agent1': torch.tensor([5.,0.,0.,0.]), 'agent2': torch.tensor([5.,0.,0.,0.]), 'agent3': torch.tensor([5.,0.,0.,0.])}
		obs, reward, done, info = env.step(actions)
		env.wait()


def start_fn(gymenv):
	target_vel = torch.randn(gymenv.N_AGENTS,3)
	gymenv.set_data("target_vel", target_vel)


def state_fn(quad):
	target_vel = quad.get_data("target_vel")[quad.get_idx(),:]
	return torch.cat([target_vel - quad.get_vel()])


def reward_fn(**kwargs):
	env = kwargs['env']
	reward = {}
	for name, agent_idx in env.names_dict.items():
		agent = env.agents[agent_idx]
		reward[name] = -1 if agent.collision() else 0
	return reward


def done_fn(**kwargs):
	return {"__all__": kwargs['steps_since_reset'] > 250}


if __name__ == '__main__':
	main()