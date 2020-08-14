from mrsgym import *
import gym

def main():
	N = 3
	env = gym.make('mrs-v0', state_fn=state_fn, N_AGENTS=N)
	while True:
		actions = torch.tensor([[1.0,0.0,1.0], [0.0,1.0,1.0], [-1.0,0.0,1.0]])
		X, reward, done, info = env.step(actions, ACTION_TYPE='set_target_pos')
		env.wait()


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])

if __name__ == '__main__':
	main()