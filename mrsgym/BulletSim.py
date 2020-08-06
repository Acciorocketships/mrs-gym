import pybullet as p
from scipy.spatial.transform import Rotation as R
import mrsgym
import torch
import os

class BulletSim:

	def __init__(self, **kwargs):
		# Constants
		self.GRAVITY = 9.81
		self.DT = 0.01
		self.REAL_TIME = False # Async
		self.set_constants(kwargs)
		# Setup
		self.client = p.connect(p.GUI)
		p.setGravity(0,0,-self.GRAVITY)
		p.setTimeStep(self.DT)
		p.setRealTimeSimulation(1 if self.REAL_TIME else 0)


	def set_constants(self, kwargs):
		for name, val in kwargs.items():
			if name in self.__dict__:
				self.__dict__[name] = val

	def step_sim(self):
		p.stepSimulation()


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
		r = R.from_euler('zyx', ori, degrees=True)
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

