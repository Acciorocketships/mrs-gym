from mrsgym.Environment import *

def env_generator(N=1, envtype='simple'):
	env = Environment()
	env.init_agents(N)
	if envtype == 'simple':
		ground = Object("plane.urdf", pos=[0,0,0], ori=[0,0,0])
		env.add_object(ground)
	return env