import numpy as np
from scipy.special import gamma
import torch
import pybullet as p

def wrap_angle(angle, margin=np.pi):
	if isinstance(angle, np.ndarray):
		angle[angle<-margin] += 2*margin
		angle[angle>margin] -= 2*margin
	else:
		if angle < -margin:
			angle += 2*margin
		elif angle > margin:
			angle -= 2*margin
	return angle


def randrange(low, high):
	if not isinstance(low, torch.Tensor):
		low = torch.tensor(low)
		high = torch.tensor(high)
	x = torch.rand(low.shape)
	x *= (high - low)
	x += low
	return x


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


def combine_decorator(method, methodtype='return'):
	def decorate(cls):
		def newmethod(self, *args, **kwargs):
			atomic_methods = [getattr(dist, method) for dist in self.dist]
			if methodtype == 'return':
				outputs = [atomic_method(*args, **kwargs) for atomic_method in atomic_methods]
				if self.mixer == 'stack':
					return torch.stack(outputs, dim=self.dim)
				elif self.mixer == 'cat':
					return torch.cat(outputs, dim=self.dim)
			elif methodtype == 'apply':
				for atomic_method in atomic_methods:
					atomic_method(*args, **kwargs)
		setattr(cls, method, newmethod)
		return cls
	return decorate


@combine_decorator('sample', 'return')
@combine_decorator('rsample', 'return')
@combine_decorator('expand', 'apply')
class CombinedDistribution(torch.distributions.distribution.Distribution):

	def __init__(self, dist, mixer='stack', dim=0):
		super(CombinedDistribution, self).__init__()
		self.dist = dist
		self.mixer = mixer
		self.dim = dim

	def apply(self, func):
		return getattr(torch, self.mixer)(list(map(func, self.dist)), dim=self.dim)

	@property
	def batch_shape(self):
		return self.apply(lambda d: d.sample()).shape

	@property
	def mean(self):
		return self.apply(lambda d: d._mean)

	@property
	def variance(self):
		return self.apply(lambda d: d._variance)

	@property
	def stddev(self):
		return self.apply(lambda d: d._stddev)

	def __str__(self):
		return self.__class__.__name__ + "(shape: " + str(self.batch_shape) + ") : " + str(list(map(lambda d: str(d), self.dist))).replace("'","")

	def __repr__(self):
		return self.__str__()

	def __getitem__(self, idx):
		return self.dist[idx]

	def __setitem__(self, idx, item):
		if idx < len(self.dist):
			self.dist[idx] = item
		else:
			self.dist.append(item)

	def __len__(self):
		return len(self.dist)



class nSphere(torch.distributions.constraints.Constraint):

	def __init__(self, centre=None, radius=1.0, within=True):
		self.centre = centre
		self.radius = radius
		self.within = within

	def check(self, value):
		centre = self.centre if (self.centre is not None) else torch.zeros(value.shape[:-1])
		size = (value - centre.unsqueeze(-1)).norm(dim=-1)
		if within:
			return torch.all(size <= self.radius)
		else:
			return torch.all((size - self.radius).abs() < 1e-6)


class NormaliseTransform(torch.distributions.transforms.Transform):

	domain = torch.distributions.constraints.real
	codomain = nSphere(radius=1.0, within=False)
	event_dim = 1

	def _call(self, x):
		return x / x.norm(dim=-1).unsqueeze(dim=-1)


class CutoffTransform(torch.distributions.transforms.Transform):
	domain = torch.distributions.constraints.real
	codomain = torch.distributions.constraints.real

	def __init__(self, low=0., high=float('inf')):
		self.low = low
		self.high = high

	def _call(self, x):
		x[x<low] = low
		x[x>high] = high
		return x


class SphereTransform(torch.distributions.transforms.Transform):
	domain = torch.distributions.constraints.real
	codomain = torch.distributions.constraints.real
	event_dim = 1

	def __init__(self, centre=None, radius=1.0, within=True):
		super(SphereTransform, self).__init__()
		self.centre = centre
		self.radius = radius
		self.within = within
		self.codomain = nSphere(centre, radius, within)

	def _call(self, x):
		centre = (self.centre if (self.centre is not None) else torch.zeros(x.shape[:-1])).unsqueeze(-1)
		shifted = x - centre
		mag = shifted.norm(dim=-1).unsqueeze(dim=-1)
		if self.within:
			mag[mag<self.radius] = self.radius
		normalised = (shifted / mag * self.radius) + centre
		return normalised






