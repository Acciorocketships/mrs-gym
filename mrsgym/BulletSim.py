import pybullet as p
from scipy.spatial.transform import Rotation as R
import mrsgym
import torch
import numpy as np
import os

class BulletSim:
	
	def __init__(self, **kwargs):
		self.REAL_TIME = False # Async
		self.HEADLESS = False
		self.GRAVITY = 9.81
		self.DT = 0.01
		self.set_constants(kwargs)
		self.setup()


	def set_constants(self, kwargs):
		for name, val in kwargs.items():
			if name in self.__dict__:
				self.__dict__[name] = val
			elif hasattr(BulletSim, name):
				setattr(BulletSim, name, val)


	def setup(self):
		if self.HEADLESS:
			self.id = p.connect(p.DIRECT)
		else:
			self.id = p.connect(p.GUI)
		p.setGravity(gravX=0, gravY=0, gravZ=-self.GRAVITY, physicsClientId=self.id)
		p.setTimeStep(timeStep=self.DT, physicsClientId=self.id)
		p.setRealTimeSimulation(1 if self.REAL_TIME else 0, physicsClientId=self.id)

	
	def step_sim(self):
		p.stepSimulation(physicsClientId=self.id)


	def set_camera(self, pos, target=torch.zeros(3)):
		if not isinstance(pos, torch.Tensor):
			pos = torch.tensor(pos)
		if not isinstance(target, torch.Tensor):
			target = torch.tensor(target)
		disp = target - pos
		dist = disp.norm()
		yaw = np.arctan2(-disp[0],disp[1]) * 180/np.pi
		pitch = np.arctan2(disp[2],np.sqrt(disp[0]**2+disp[1]**2)) * 180/np.pi
		p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target.tolist(), physicsClientId=self.id)


	def add_line(start, end, name="line", width=0.1, lifetime=0., colour=torch.tensor([1.,0.,0.])):
		if not isinstance(start, torch.Tensor):
			start = torch.tensor(start)
		if not isinstance(end, torch.Tensor):
			end = torch.tensor(end)
		if not isinstance(colour, torch.Tensor):
			colour = torch.tensor(colour)
		if not hasattr(self, "line_names"):
			self.line_names = {}
		uid = self.line_names.get(name, -1)
		new_uid = p.addUserDebugLine(lineFromXYZ=start.tolist(), lineToXYZ=end.tolist(), lineColorRGB=colour.tolist(), lineWidth=width, lifeTime=lifetime, replaceItemUniqueId=uid, physicsClientId=self.id)
		self.line_names[name] = new_uid


class DefaultSim:

	def __init__(self):
		self.id = 0
		self.GRAVITY = 9.81
		self.DT = 0.01


	# add/read debug parameter, add debug text

	# enable/disable collisions (single and pairwise)

