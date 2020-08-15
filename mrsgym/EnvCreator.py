from mrsgym.Environment import *
from mrsgym.Object import *

def env_generator(N=1, envtype='simple'):
	env = Environment()
	env.init_agents(N)
	if envtype == 'simple':
		ground = Object("plane.urdf", pos=[0,0,0], ori=[0,0,0])
		env.add_object(ground)
	return env


def create_box(dim=[1,1,1], pos=[0,0,0], ori=[0,0,0], mass=0, collisions=True):
	if isinstance(dim, torch.Tensor):
		dim = dim.tolist()
	if collisions:
		uid = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim)
		if mass != 0:
			uid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=uid)
	else:
		uid = p.createVisualShape(p.GEOM_BOX, halfExtents=dim)
		if mass != 0:
			uid = p.createMultiBody(baseMass=mass, baseVisualShapeIndex=uid)
	obj = Object(uid=uid, pos=pos, ori=ori)
	return obj


def create_sphere(radius=0.5, pos=[0,0,0], mass=0, collisions=True):
	if collisions:
		uid = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
		if mass != 0:
			uid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=uid)
	else:
		uid = p.createVisualShape(p.GEOM_SPHERE, radius=radius)
		if mass != 0:
			uid = p.createMultiBody(baseMass=mass, baseVisualShapeIndex=uid)
	obj = Object(uid=uid, pos=pos)
	return obj


def load_urdf(path, pos=[0,0,0], ori=[0,0,0], inpackage=True):
	# position
	if isinstance(pos, torch.Tensor):
		pos = pos.tolist()
	# orientation
	if isinstance(ori, list):
		ori = torch.tensor(ori)
	if ori.shape == (3,3):
		r = R.from_matrix(ori)
	elif ori.shape == (3,):
		r = R.from_euler('xyz', ori, degrees=True)
	elif ori.shape == (4,):
		r = R.from_quat(ori)
	ori = r.as_quat()
	ori = ori.tolist()
	# path
	if inpackage:
		directory = os.path.join(os.path.dirname(mrsgym.__file__), 'models')
		path = os.path.join(directory, path)
	# load
	model = p.loadURDF(path, pos, ori)
	return model