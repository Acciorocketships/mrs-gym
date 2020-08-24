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
					'STATE_SIZE': 6,
					'ACTION_TYPE': "set_control",
					'action_space': action_space,
				 }
	env = MRS_RLlib_MultiAgent(config=env_config)
	env.set_data("target_vel", torch.randn(3))
	while True:
		actions = {'agent1': torch.tensor([5.,0.,0.,0.]), 'agent2': torch.tensor([5.,0.,0.,0.]), 'agent3': torch.tensor([5.,0.,0.,0.])}
		obs, reward, done, info = env.step(actions)
		env.wait()


if __name__ == '__main__':
	main()