from mrsgym import *
import torch
import gym

def main():
	env = gym.make('mrs-v0', state_fn=state_fn, env=generate_environment(), ACTION_TYPE='set_target_vel', RETURN_EVENTS=True)
	N = env.N_AGENTS
	actions = torch.zeros(N,3)
	while True:
		X, reward, done, info = env.step(actions)
		actions = key_to_action(info["keyboard_events"]).expand(N,-1)
		env.wait()


def generate_environment():
	N = 1
	sim = BulletSim()
	environment = Environment(sim=sim)
	environment.init_agents(N)
	ground = Object(uid="plane.urdf", env=environment, sim=sim, pos=torch.tensor([0,0,0]), ori=torch.tensor([0,0,0]))
	box = create_box(dim=torch.tensor([0.5,1.,0.5]), pos=torch.tensor([3.,0.,1.]), ori=torch.tensor([0.,0.,0.]), mass=0., collisions=True, env=environment, sim=sim)
	ball = create_sphere(radius=0.5, pos=torch.tensor([-3.,0,3.]), mass=0.1, collisions=True, env=environment, sim=sim)
	environment.add_object(ground)
	environment.add_object(box)
	environment.add_object(ball)
	return environment


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
	imgs = quad.get_image(forward=torch.tensor([1.,0.,0.]), up=torch.tensor([0.,0.,1.]), offset=torch.tensor([0.35,0.,0.]), body=True, fov=90., aspect=4/3, height=225)
	return imgs["img"].view(-1)


if __name__ == '__main__':
	main()