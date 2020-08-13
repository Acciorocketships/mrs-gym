import pybullet as p
from scipy.spatial.transform import Rotation as R
import mrsgym
import torch
import os

class BulletSim:

	# Constants
	GRAVITY = 9.81
	DT = 0.01
	REAL_TIME = False # Async

	@staticmethod
	def setup():
		client = p.connect(p.GUI)
		p.setGravity(0,0,-BulletSim.GRAVITY)
		p.setTimeStep(BulletSim.DT)
		p.setRealTimeSimulation(1 if BulletSim.REAL_TIME else 0)

	@staticmethod
	def step_sim():
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

