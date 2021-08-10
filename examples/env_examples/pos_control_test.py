from mrsgym import *
import gym

def main():
	N = 4
	env = gym.make('mrs-v0', state_fn=state_fn, N_AGENTS=N, ACTION_TYPE='set_target_pos', RETURN_EVENTS=True)
	formation = torch.tensor([[1.0,0.0,2.0], [0.0,1.0,2.0], [-1.0,0.0,2.0], [0.0,-1.0,2.0]])
	normal_offset = 1.0
	actions = formation
	while True:
		X, reward, done, info = env.step(actions)
		env.wait()


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])

if __name__ == '__main__':
	main()