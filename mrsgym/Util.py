import numpy as np
import torch
from enum import IntEnum

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


# returns a unit vector representing the direction of the target from the camera position (in world coordinates)
def pix2world(vpix, hpix, height, aspect, fov, forward, up, pos):
	if not isinstance(forward, torch.Tensor):
		forward = torch.tensor(forward)
	if not isinstance(up, torch.Tensor):
		up = torch.tensor(up)
	if not isinstance(pos, torch.Tensor):
		pos = torch.tensor(pos)
	width = int(height * aspect)
	fov_scaler = np.tan(np.pi/180 * fov/2)
	pos_view = torch.tensor([1.0, fov*aspect*(1.0-(hpix/width)*2), fov*(1.0-(vpix/height)*2)])
	camera_forward = forward / forward.norm()
	camera_up = up / up.norm()
	camera_left = torch.cross(camera_up, camera_forward)
	R = torch.stack([camera_forward, camera_left, camera_up], dim=1)
	pos_world = R @ pos_view + pos
	vec = pos_world - pos; vec = vec / vec.norm()
	return vec


def dict2str(dictionary, spaces=0):
	string = ""
	if type(dictionary) == dict:
		for (key,value) in dictionary.items():
			if type(value) != dict and type(value) != list:
				string += (" " * spaces*4) + str(key) + " - " + str(value) + "\n"
			else:
				string += (" " * spaces*4) + str(key) + " - \n"
				string += dict2str(value,spaces+1)
	elif type(dictionary) == list:
		for item in dictionary:
			string += dict2str(item,spaces+1)
	else:
		string += (" " * spaces*4) + str(dictionary) + "\n"
	return string


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



class Key(IntEnum):
	# special
	left = 65295
	right = 65296
	up = 65297
	down = 65298
	shift = 65306
	ctrl = 65307
	option = 65308
	enter = 65309
	space = 32
	tab = 9
	delete = 8
	# other
	minus = 45
	plus = 61
	left_bracket = 91
	right_bracket = 93
	semicolon = 59
	quote = 39
	comma = 44
	period = 46
	slash = 47
	tilde = 96
	backslash = 92
	# alphanumeric
	a = 97
	b = 98
	c = 99
	d = 100
	e = 101
	f = 102
	g = 103
	h = 104
	i = 105
	j = 106
	k = 107
	l = 108
	m = 109
	n = 110
	o = 111
	p = 112
	q = 113
	r = 114
	s = 115
	t = 116
	u = 117
	v = 118
	w = 119
	x = 120
	y = 121
	z = 122
	zero = 48
	one = 49
	two = 50
	three = 51
	four = 52
	five = 53
	six = 54
	seven = 55
	eight = 56
	nine = 57











