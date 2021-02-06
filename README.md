# Multi Robot Systems Gym

## Installation
```bash
pip3 install -e .
```

## Demo
![Position Control Demo](demo.gif)

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
	
	Example Usage: 
	```python
	mrsenv = gym.make('mrs-v0', **kwargs)
	```
	
	Description: creates the MRS gym environment
	
	Arguments:
	- state_fn: (function) [required] specifies the content of the obs output of mrsenv.step(actions). Takes an agent (Quadcopter) class as an input, and produces a one-dimensional (STATE_SIZE) tensor as an output.
	- reward_fn: (function) specifies the content of the reward output of mrsenv.step(actions). The input is env (an instance of the Environment object), X (the joint observation with the last K_HOPS+1 timesteps of data), A (the adjacency matrix, computed using COMM_RANGE), Xlast (the X value from the last timestep), action (the action that was used in the last timestep), and steps_since_reset (the number of timesteps since the simulation was set/reset). The output can be a scalar or a tensor.
	- done_fn: (function) done_fn(env, s, t) specifies the termination conditions. The input is env (an instance of the Environment object), X (the joint observation with the last K_HOPS+1 timesteps of data), A (the adjacency matrix, computed using COMM_RANGE), Xlast (the X value from the last timestep), action (the action that was used in the last timestep), and steps_since_reset (the number of timesteps since the simulation was set/reset). The output is a bool.
	- info_fn: (function) specifies the content of the info output of mrsenv.step(actions). The input is env (an instance of the Environment object), X (the joint observation with the last K_HOPS+1 timesteps of data), A (the adjacency matrix, computed using COMM_RANGE), Xlast (the X value from the last timestep), action (the action that was used in the last timestep), and steps_since_reset (the number of timesteps since the simulation was set/reset). The output is a dict containing any data that might be useful.
	- update_fn: (function) This function allows the user to specify any extra behaviour they wish to run every time the simulation is stepped. The update_fn could be used to add aerodynamics disturbances to the objects in the environment, change the position and orientation of the camera view, add/remove objects or agents, etc. Takes the env (Environment) as an input. The input is env (an instance of the Environment object), X (the joint observation with the last K_HOPS+1 timesteps of data), A (the adjacency matrix, computed using COMM_RANGE), Xlast (the X value from the last timestep), action (the action that was used in the last timestep), and steps_since_reset (the number of timesteps since the simulation was set/reset).
	- env: (Environment OR str) defines the environment, which contains all objects and agents. Choose from:
		1. Environment: uses a user-defined environment. See the Environment class for how to construct it manually
		2. "simple": creates a world with N_AGENTS agents and a plane at z=0. This is the default
	- N_AGENTS: (int) sets the number of agents in the environment.
	- K_HOPS: (int) sets the number of consecutive time steps to return in the observation, which is of size (N_AGENTS x STATE_SIZE x K_HOPS+1).
	- RETURN_A: (bool) specifies whether or not to compute the adjacency matrix. It is returned in the info dict under key "A".
	- RETURN_EVENTS: (bool) specifies whether or note to include "keyboard_events" and "mouse_events" in the info dict. "keyboard_events" is a dict which maps a key (str) to a value (1 for pressed, 0 for held, -1 for released). "mouse_events" is not None when there is a mouse click, and it returns a dict of:
		1. "camera_pos": ((3) tensor) the world coordinates of the camera
		2. "target_pos" ((3) tensor) the world coordinates of the location that was clicked
		3. "target_normal" ((3) tensor) the normal of the surface that was clicked.
		4. "target_obj" (Object) the object that was clicked
	- COMM_RANGE: (float) sets the communication range for use when computing the adjacency matrix.
	- ACTION_TYPE: (str) specifies how to interpret the given action. Choose from:
		1. "set_target_vel": PID control with given target velocity (m/s) [vx, vy, vz]
		2. "set_target_pos": PID control with given target position (m) [x, y, z]
		3. "set_force": PID control to emulate a desired applied force vector (N) [fx, fy, fz]
		4. "set_control": thrust force (N) and roll, pitch, and yaw torques (Nm) [Fthrust, Troll, Tpitch, Tyaw]
		5. "set_speeds": sets the speeds of the individual propellors (rad/s) [wfront, wleft, wback, wright]
	- START_POS: ((N_AGENTS x 3) tensor OR torch.distributions.distribution.Distribution) sets the starting positions or the distribution from which the starting positions are sampled.
		1. (N_AGENTS x 3) tensor: specifies fixed starting positions for the agents
		2. (3) distribution: a distribution whose sample() method yields a (3) tensor is used to independently set the starting positions for each agent. If there are any overlapping agents (given the specified AGENT_RADIUS), they will be re-sampled.
		3. (N_AGENTS x 3) distribution: a joint distribution whose sample() method yields a (N_AGENTS x 3) tensor is used to set the starting positions for the whole swarm. This can be useful for using a generator to output a fixed suite of starting positions to compare different algorithms with the same initial conditions.
	- START_ORI ((3) tensor OR (6) tensor OR (N_AGENTS x 3) tensor OR (N_AGENTS x 6) tensor): specifies the starting orientations or the distribution of starting orientations
		1. ((3) tensor OR (N_AGENTS x 3) tensor: sets the exact euler angles (rad) in [roll, pitch, yaw] format. If a (3) tensor is given, then all of the orientations are the same.
		2. ((6) tensor OR (N_AGENTS x 6) tensor: sets the range of possible euler angles, to be sampled from a uniform distribution [roll_low, pitch_low, yaw_low, roll_high, pitch_high, yaw_high]
	- DT: (float) sets the timestep for the environment. The default is 0.01s
	- REAL_TIME: (bool) If True, the environment runs asynchronously. If False, one time step elapses whenever env.step(actions) is called.
	- HEADLESS: (bool) If True, the simulation runs without a GUI component.
	
2. `reset`
	
	Example Usage: 
	```python
	mrsenv.reset(vel=torch.tensor([1.,0.,0.].expand(N_AGENTS,-1))
	```
	
	Description: uses START_POS and START_ORI to reset the states of all agents. In addition, pos, vel, ori, and angvel can be given as optional arguments. This will override the default reset value
	
	Arguments:
	- pos: ((N_AGENTS x 3) tensor) overrides START_POS to specify starting positions [x, y, z] for all agents.
	- vel: ((N_AGENTS x 3) tensor) sets starting velocities [vx, vy, vz] for all agents instead of the default value of [0, 0, 0].
	- ori: ((N_AGENTS x 3) tensor) overrides START_ORI to specify starting euler angles [roll, pitch, yaw] for all agents. All euler angles in this library are given as extrinsic rotations in the order 'xyz' (or equivalently intrinsic rotations in the order 'ZYX').
	- angvel: ((N_AGENTS x 3) tensor) sets starting angular velocities [wx, wy, yz] for all agents instead of the default value of [0, 0, 0].
	
3. `set`

	Example Usage: 
	```python
	mrsenv.set(ori=torch.tensor([0.,0.,0.].expand(N_AGENTS,-1))
	```
	Description: The same as reset, except if the optional arguments are not given, then those components of the state will not be changed. When called with no arguments, reset() sets all of the agents to their default values, while set() does nothing.
	
	Arguments:
	- pos: ((N_AGENTS x 3) tensor) sets starting positions [x, y, z] for all agents
	- vel: ((N_AGENTS x 3) tensor) sets starting velocities [vx, vy, vz] for all agents.
	- ori: ((N_AGENTS x 3) tensor) sets euler angles [roll, pitch, yaw] for all agents.
	- angvel: ((N_AGENTS x 3) tensor) sets starting angular velocities [wx, wy, yz] for all agents.
		
4. `wait`

	Example Usage: 
	```python
	mrsenv.wait()
	```
	
	Description: measures the loop time of execution and pauses for the appropriate amount of time so the simulation runs with a period of DT and plays at 1x speed.
	
	Arguments:
	- dt: (float) sets the desired loop time. If None is given, then BulletSim.DT is used.
		
5. `step`

	Example Usage: 
	```python
	mrsenv.step(torch.zeros(N_AGENTS,3), ACTION_TYPE="set_target_vel")
	```
	
	Description: sets the actions and steps the simulation by one timestep.
	
	Arguments:
	- actions: ((N_AGENTS x 3) tensor) [required] sets the actions for the agents
	- ACTION_TYPE: (str) overrides the action type specified in the initialisation of the environment
	
	Outputs: (obs, reward, done, info)
	- obs: ((N_AGENTS x STATE_SIZE x K_HOPS+1) tensor) a matrix of the states returned by state_fn(agent) for all agents
	- reward: (float) the reward calculated by the given reward_fn(env, s, a, s'). It can have any type, but the default is 0.0 if a reward_fn is not given.
	- done: (bool) the termination conditions calculated by the given done_fn(env, s, t). By default, it is True when the number of timesteps since the last reset reaches MAX_TIMESTEPS.
	- info: (dict) a dict of extra information that is calculated by info_fn(env). The adjacency matrix (N_AGENTS x N_AGENTS x K_HOPS+1) is also stored in info["A"] if RETURN_A is True
		
		
6. `set_data` and `get_data`

	Example Usage:
	```python
	mrsenv.set_data("target_vel", torch.randn(N,3))
	
	def state_fn(quad):
		target_vel = quad.get_data("target_vel")[quad.get_idx(),:]
		return torch.cat([target_vel - quad.get_vel(), quad.get_ori()])
	```
	
	Description: Sets and gets user data to be stored in the environment. This data can be accessed from the base mrsenv, or from the env Environment() object which is available in the reward_fn, info_fn, done_fn, and update_fn, or from the quad Object() which is available in the state_fn.
	
	Arguments:
	- name: name of the variable you want to set/get
	- val (only for set_data): value of the variable you want to set
	
	Outputs:
	- val (only for get_data): value of the variable you want to get
