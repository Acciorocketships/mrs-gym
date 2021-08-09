from mrsgym import *
from mrsgym.Util import *
import gym

def main():
	N = 3
	env = gym.make('mrs-v0', state_fn=state_fn, N_AGENTS=1, ACTION_TYPE='set_target_ori', RETURN_EVENTS=True)
	actions = torch.zeros(N,3)
	while True:
		X, reward, done, info = env.step(actions)
		actions = key_to_action(info["keyboard_events"]).expand(N,-1)
		set_camera(env.env)
		env.wait()

def key_to_action(keys):
	action = torch.zeros(3)
	if Key.up in keys:
		action[0] += .7
	if Key.down in keys:
		action[0] -= .7
	if Key.left in keys:
		action[1] += .7
	if Key.right in keys:
		action[1] -= .7
	if Key.space in keys:
		action[2] += 1
	if Key.option in keys:
		action[2] -= 1
	return action


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])

def set_camera(env):
	agent_pos = env.agents[0].get_pos()
	env.set_camera(pos=agent_pos + torch.tensor([2,0,1]), target=agent_pos)

if __name__ == '__main__':
	main()