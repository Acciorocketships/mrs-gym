from mrsgym import *
import gym

def main():
	N = 4
	env = gym.make('mrs-v0', state_fn=state_fn, N_AGENTS=N, ACTION_TYPE='set_target_pos', RETURN_EVENTS=True)
	formation = torch.tensor([[1.0,0.0,0.0], [0.0,1.0,0.0], [-1.0,0.0,0.0], [0.0,-1.0,0.0]])
	normal_offset = 1.0
	actions = torch.stack([quad.get_pos() for quad in env.get_agents()], dim=0)
	while True:
		X, reward, done, info = env.step(actions)
		if info["mouse_events"] is not None:
			target = info["mouse_events"]["target_pos"]
			target_normal = info["mouse_events"]["target_normal"]
			actions = mouse_to_setpos(target, target_normal, formation, normal_offset)
		env.wait()


def mouse_to_setpos(target, target_normal, formation, normal_offset):
	return target.expand(formation.shape[0], -1) + formation + (target_normal[None,:] * normal_offset)


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])

if __name__ == '__main__':
	main()