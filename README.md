# Multi Robot Systems Gym

```python
from mrsgym import *
import gym
import torch

def main():
	N = 3
	env = gym.make('mrs-v0', N_AGENTS=N, state_fn=state_fn, ACTION_TYPE='set_target_vel')
	for _ in range(10000):
		actions = torch.tensor([0.5,0.0,0.0]).expand(N,-1)
		X, reward, done, info = env.step(actions)
		env.wait()


def state_fn(quad):
	return torch.cat([quad.get_pos(), quad.get_vel()])

if __name__ == '__main__':
	main()
```
