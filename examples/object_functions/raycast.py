from mrsgym import *
import gym

def main():
	N = 3
	env = gym.make('mrs-v0', state_fn=state_fn, N_AGENTS=N, ACTION_TYPE='set_target_pos')
	actions = torch.stack([quad.get_pos() for quad in env.get_agents()], dim=0)
	while True:
		X, reward, done, info = env.step(actions)
		print(X[0,:].view(4,3))
		env.wait()


def state_fn(quad):
	offset = torch.tensor([0.,0.,-0.1])
	directions = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,-1.]])
	rays = quad.raycast(offset=offset, directions=directions, body=True)
	return rays["pos"].reshape(directions.shape[0]*3)

if __name__ == '__main__':
	main()