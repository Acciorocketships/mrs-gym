from mrsgym import *
import torch
import gym

def main():
	env = gym.make('mrs-v0', state_fn=state_fn, env=generate_environment(), ACTION_TYPE='set_target_vel', RETURN_EVENTS=True)
	N = env.N_AGENTS
	actions = torch.zeros(N,3)
	for t in range(100000):
		X, reward, done, info = env.step(actions)
		actions = 3 * key_to_action(info["keyboard_events"]).expand(N,-1)
		if t % 100 == 0:
			add_random_object(env.env)
		env.wait()


def generate_environment():
	N = 1
	sim = BulletSim()
	environment = Environment(sim=sim)
	environment.init_agents(N)
	ground = Object(uid="plane.urdf", env=environment, sim=sim, pos=torch.tensor([0,0,0]), ori=torch.tensor([0,0,0]))
	environment.add_object(ground)
	return environment


def add_random_object(env):
	mass = 2 ** torch.randn(1)
	pos = 2 * torch.randn(3); pos[2] = 2 * torch.rand(1) + 1
	vel = 3 * torch.randn(3)
	ori = 180 * torch.rand(3)
	if torch.rand(1) > 0.5:
		dim = torch.rand(3)
		obj = create_box(dim=dim, pos=pos, ori=ori, mass=mass, collisions=True, env=env, sim=env.sim)
	else:
		rad = torch.rand(1)
		obj = create_sphere(radius=rad, pos=pos, mass=mass, collisions=True, env=env, sim=env.sim)
	obj.set_state(vel=vel)
	env.add_object(obj)


def key_to_action(keys):
	action = torch.zeros(3)
	if Key.up in keys:
		action[0] += 1
	if Key.down in keys:
		action[0] -= 1
	if Key.left in keys:
		action[1] += 1
	if Key.right in keys:
		action[1] -= 1
	if Key.space in keys:
		action[2] += 1
	if Key.option in keys:
		action[2] -= 1
	return action


def state_fn(quad):
	return quad.get_vel()


if __name__ == '__main__':
	main()