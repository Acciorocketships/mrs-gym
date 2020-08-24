from torch.distributions import *
from mrsgym import *
import torch


def main():
	# Distribution
	N = 12
	z = Uniform(low=torch.zeros(N,1), high=torch.ones(N,1)) # uniform between 0 and 1 in the z direction
	xy_normal = Normal(torch.zeros(N,2), 1.0)
	sphere_transform = SphereTransform(radius=1.0, within=True)
	xy_circle = TransformedDistribution(xy_normal, [sphere_transform]) # gaussian with sigma=1 bounded by a cirle of radius=1 in the xy direction
	dist = CombinedDistribution([xy_circle, z], mixer='cat', dim=1)
	import pdb; pdb.set_trace()
	print(dist.sample())
	# Environment
	env = gym.make('mrs-v0', state_fn=state_fn, N_AGENTS=N, ACTION_TYPE='set_target_pos', START_POS=dist)
	actions = torch.stack([quad.get_pos() for quad in env.get_agents()], dim=0)
	while True:
		X, reward, done, info = env.step(actions)
		env.wait()


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])


if __name__ == '__main__':
	main()