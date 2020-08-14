# Multi Robot Systems Gym

## Installation
```bash
pip3 install -e .
```

## Example
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

## Documentation

### MRS
1. `__init__`
	
	Example Usage: `mrsenv = gym.make('mrs-v0', **kwargs)`
	
	Description: creates the MRS gym environment
	
	Arguments:
	- state_fn: (function) [required] specifies the content of the obs output of mrsenv.step(actions). Takes an instance of the Quadcopter class as an input, and produces a one-dimensional (STATE_SIZE) tensor as an output
	- reward_fn: (function) specifies the content of the reward output of mrsenv.step(actions). Takes an instance of the Environment class as an input, and produces an output of any type (default is 0.0)
	- info_fn: (function) specifies the content of the info output of mrsenv.step(actions). Takes an instance of the Environment class as an input, and produces a dict as an output
	- env: (Environment OR str) defines the environment, which contains all objects and agents. Choose from:
		1. Environment: uses a user-defined environment. See the Environment class for how to construct it manually
		2. "simple": creates a world with N_AGENTS agents and a plane at z=0. This is the default
	- N_AGENTS: (int) sets the number of agents in the environment
	- K_HOPS: (int) sets the number of consecutive time steps to return in the observation, which is of size (N_AGENTS x STATE_SIZE x K_HOPS+1)
	- AGENT_RADIUS: (float) used purely for start position generation, this prevents agents from overlapping
	- RETURN_A: (bool) specifies whether or not to compute the adjacency matrix. It is returned in the info dict under key "A"
	- COMM_RANGE: (float) sets the communication range for use when computing the adjacency matrix
	- ACTION_TYPE: (str) specifies how to interpret the given action. Choose from:
		1. "set_target_vel": PID control with given target velocity (m/s) [vx, vy, vz]
		2. "set_target_pos": PID control with given target position (m) [x, y, z]
		3. "set_force": PID control to emulate a desired applied force vector (N) [fx, fy, fz]
		4. "set_control": thrust force (N) and roll, pitch, and yaw torques (Nm) [Fthrust, Troll, Tpitch, Tyaw]
		5. "set_speeds": sets the speeds of the individual propellors (rad/s) [wfront, wleft, wback, wright]
	- START_POS: ((N_AGENTS x 3) tensor OR torch.distributions.distribution.Distribution) sets the starting positions or the distribution from which the starting positions are sampled.
		1. (N_AGENTS x 3) tensor: specifies fixed starting positions for the agents
		2. (3) distribution: a distribution whose sample() method yields a (3) tensor is used to independently set the starting positions for each agent. If there are any overlapping agents, they will be re-sampled.
		3. (N_AGENTS x 3) distribution: a joint distribution whose sample() method yields a (N_AGENTS x 3) tensor is used to set the starting positions for the whole swarm. This can be useful for using a generator to output a fixed suite of starting positions to compare different algorithms with the same initial conditions.
	- START_ORI ((3) tensor OR (6) tensor OR (N_AGENTS x 3) tensor OR (N_AGENTS x 6) tensor): specifies the starting orientations or the distribution of starting orientations
		1. ((3) tensor OR (N_AGENTS x 3) tensor: sets the exact euler angles (rad) in [roll, pitch, yaw] format. If a (3) tensor is given, then all of the orientations are the same.
		2. ((6) tensor OR (N_AGENTS x 6) tensor: sets the range of possible euler angles, to be sampled from a uniform distribution [roll_low, pitch_low, yaw_low, roll_high, pitch_high, yaw_high]
	- DT: (float) sets the timestep for the environment. The default is 0.01s
	- REAL_TIME: (bool) If true, the environment runs asynchronously. If false, one time step elapses whenever env.step(actions) is called.
	- HEADLESS: (bool) If true, the simulation runs without a GUI component.
	
2. `reset`
	
	Example Usage: `mrsenv.reset(vel=torch.tensor([1.,0.,0.]).expand(N_AGENTS,-1)`
	
	Description: uses START_POS and START_ORI to reset the states of all agents. In addition, pos, vel, ori, and angvel can be given as optional arguments. This will override the default reset value
	
	Arguments:
	- pos: ((N_AGENTS x 3) tensor) overrides START_POS to specify starting positions [x, y, z] for all agents
	- vel: ((N_AGENTS x 3) tensor) sets starting velocities [vx, vy, vz] for all agents instead of the default value of [0, 0, 0]
	- ori: ((N_AGENTS x 3) tensor) overrides START_ORI to specify starting euler angles [roll, pitch, yaw] for all agents. All euler angles in this library are given as extrinsic rotations in the order 'xyz' (or equivalently intrinsic rotations in the order 'ZYX')
	- angvel: ((N_AGENTS x 3) tensor) sets starting angular velocities [wx, wy, yz] for all agents instead of the default value of [0, 0, 0]
		
3. `wait`

	Example Usage: `mrsenv.wait()`
	
	Description: times the loop time of execution and pauses for the appropriate amount of time so the simulation runs with a period of DT and plays at 1x speed.
	
	Arguments:
	- dt: (float) sets the desired loop time. If none is given, then BulletSim.DT is used
		
4. `step`

	Example Usage: `mrsenv.step(torch.zeros(N_AGENTS,3), ACTION_TYPE="set_target_pos")`
	
	Description: sets the actions and steps the simulation by one timestep
	
	Arguments:
	- actions: ((N_AGENTS x 3) tensor) [required] sets the actions for the agents
	- ACTION_TYPE: (str) overrides the action type specified in the initialisation of the environment
	
	Outputs: (obs, reward, done, info)
	- obs: ((N_AGENTS x STATE_SIZE) tensor) a matrix of the states returned by state_fn(agent) for all agents
	- reward: (float) the reward calculated by reward_fn(self.env). It can have any type, but the default is 0.0 if a reward_fn is not given.
	- done: ((N_AGENTS) bool tensor) an array indicating which agents are experiencing a collision. A custom done function can be used by overriding mrsenv.env.get_done()
	- info: (dict) a dict of extra information that is calculated by info_fn(self.env). The adjacency matrix is also stored in info["A"] if RETURN_A is true
		
		
