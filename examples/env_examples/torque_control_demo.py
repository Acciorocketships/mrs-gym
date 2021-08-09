from mrsgym import *
from mrsgym.Util import *
import gym

def main():
	N = 3
	env = gym.make('mrs-v0', state_fn=state_fn, N_AGENTS=N, ACTION_TYPE='set_control', RETURN_EVENTS=True)
	actions = action_transform(np.zeros((N,4)))
	while True:
		X, reward, done, info = env.step(actions)
		actions = action_transform(key_to_action(info["keyboard_events"]).expand(N,-1))
		env.wait()

def key_to_action(keys):
	action = torch.zeros(4)
	if Key.up in keys:
		action[2] += 1
	if Key.down in keys:
		action[2] -= 1
	if Key.left in keys:
		action[1] += 1
	if Key.right in keys:
		action[1] -= 1
	if Key.space in keys:
		action[0] += 1
	if Key.option in keys:
		action[0] -= 1
	if Key.period in keys:
		action[3] += 1
	if Key.comma in keys:
		action[3] -= 1
	return action


def action_transform(action):
	action = np.array(action)
	thrustC = 0.1
	torqueC = 0.1
	action[:,0] = action[:,0] * thrustC + 9.81
	action[:,1] = torqueC * action[:,1]
	action[:,2] = torqueC * action[:,2]
	action[:,3] = torqueC * action[:,3]
	return action


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])

if __name__ == '__main__':
	main()