from mrsgym import *
import gym

def main():
	N = 3
	env = gym.make('mrs-rllib-multiagent-v0', config={"state_fn": state_fn, "N_AGENTS":N})
	while True:
		actions = {'agent0': torch.tensor([1.0,0.0,1.0]), 'agent1': torch.tensor([0.0,1.0,1.0]), 'agent2': torch.tensor([-1.0,0.0,1.0])}
		X, reward, done, info = env.step(actions)
		env.wait()


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])

if __name__ == '__main__':
	main()