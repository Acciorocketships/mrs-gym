from mrsgym import *
import gym

def main():
	N = 3
	env = gym.make('mrs-v0', state_fn=state_fn, N_AGENTS=N, ACTION_TYPE='set_target_pos', update_fn=update)
	actions = torch.stack([quad.get_pos() for quad in env.get_agents()], dim=0)
	while True:
		X, reward, done, info = env.step(actions)
		env.wait()


def update(env, obs, action):
	objects = env.agents[0].get_closest_objects(radius=3.0)
	object_dists = {obj: env.agents[0].get_dist(obj)['distance'] for obj in objects}
	print(dict2str(object_dists))


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])

if __name__ == '__main__':
	main()