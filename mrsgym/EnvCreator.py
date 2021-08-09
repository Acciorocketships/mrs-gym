from mrsgym.BulletSim import *
from mrsgym.Environment import *
from mrsgym.Object import *
import torch
import pybullet as p

def env_generator(envtype='simple', N=1, sim=DefaultSim()):
	env = Environment(sim=sim)
	env.init_agents(N)
	if envtype == 'simple':
		ground = Object(uid="plane.urdf", env=env, sim=sim, pos=[0,0,0], ori=[0,0,0])
		env.add_object(ground)
	return env


def create_box(dim=[1,1,1], pos=[0,0,0], ori=[0,0,0], mass=0, collisions=True, env=None, sim=DefaultSim()):
	if isinstance(dim, torch.Tensor):
		dim = dim.tolist()
	if collisions:
		uid = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim, physicsClientId=sim.id)
		uid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=uid, physicsClientId=sim.id)
	else:
		uid = p.createVisualShape(p.GEOM_BOX, halfExtents=dim, physicsClientId=sim.id)
		uid = p.createMultiBody(baseMass=mass, baseVisualShapeIndex=uid, physicsClientId=sim.id)
	obj = Object(uid=uid, env=env, sim=sim, pos=pos, ori=ori)
	return obj


def create_sphere(radius=0.5, pos=[0,0,0], mass=0, collisions=True, env=None, sim=DefaultSim()):
	if collisions:
		uid = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=sim.id)
		uid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=uid, physicsClientId=sim.id)
	else:
		uid = p.createVisualShape(p.GEOM_SPHERE, radius=radius, physicsClientId=sim.id)
		uid = p.createMultiBody(baseMass=mass, baseVisualShapeIndex=uid, physicsClientId=sim.id)
	obj = Object(uid=uid, env=env, sim=sim, pos=pos)
	return obj


def load_urdf(path, pos=[0,0,0], ori=[0,0,0], inpackage=True, sim=DefaultSim()):
	# position
	if isinstance(pos, torch.Tensor):
		pos = pos.tolist()
	# orientation
	if isinstance(ori, list):
		ori = torch.tensor(ori)
	if ori.shape == (3,3):
		r = R.from_matrix(ori)
	elif ori.shape == (3,):
		r = R.from_euler('XYZ', ori, degrees=True)
	elif ori.shape == (4,):
		r = R.from_quat(ori)
	ori = r.as_quat()
	ori = ori.tolist()
	# path
	if inpackage:
		directory = os.path.join(os.path.dirname(mrsgym.__file__), 'models')
		path = os.path.join(directory, path)
	# load
	model = p.loadURDF(fileName=path, basePosition=pos, baseOrientation=ori, physicsClientId=sim.id)
	return model

	