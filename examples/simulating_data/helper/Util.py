import atexit
import torch
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_model(model, path):
	try:
		fp = open(path,'rb')
	except Exception as exception:
		print("No Model to Load: %s" % exception)
		return
	model_dict = torch.load(fp, map_location=device)
	model.load_state_dict(model_dict)
	print("Loaded Model " + path)
	return model


def save_model(model, path):
	model_dict = model.state_dict()
	with open(path, 'wb') as fp:
		torch.save(model_dict, fp)
	print("Saved Model " + path)


def save_model_onexit(model, path):
	atexit.register(save_model, model=model, path=path)


def totensor(x):
	if isinstance(x, torch.Tensor):
		return x
	else:
		return torch.tensor(x)


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
	import signal

	class TimeoutError(Exception):
		pass

	def handler(signum, frame):
		raise TimeoutError()

	# set the timeout handler
	signal.signal(signal.SIGALRM, handler) 
	signal.alarm(timeout_duration)
	try:
		result = func(*args, **kwargs)
	except TimeoutError as exc:
		result = default
	finally:
		signal.alarm(0)

	return result


def get_abs_path(dirname="Run"):
	curr_dir = os.path.abspath("")
	idx = curr_dir.find(dirname)
	new_dir = curr_dir[:idx] + dirname
	return new_dir


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

	