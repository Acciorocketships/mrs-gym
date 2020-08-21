from mrsgym import *
import gym

def main():
	N = 2
	env = gym.make('mrs-v0', state_fn=state_fn, N_AGENTS=N, ACTION_TYPE='set_target_pos', update_fn=update)
	actions = torch.stack([quad.get_pos() for quad in env.get_agents()], dim=0)
	while True:
		X, reward, done, info = env.step(actions)
		env.wait()


def update(env):
	contact = env.agents[0].get_contact_points(env.agents[1], body=False)
	dist = env.agents[0].get_dist(env.agents[1], body=False)
	print(dict2str(dist))
	if contact['distance'].shape[0] != 0:
		print(dict2str(contact))


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])

if __name__ == '__main__':
	main()