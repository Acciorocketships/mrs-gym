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


	def stop(self):
		try:
			p.resetSimulation(self.id)
			p.disconnect(self.id)
		except:
			pass

	
	def step_sim(self):
		p.stepSimulation(physicsClientId=self.id)




class DefaultSim:

	def __init__(self):
		self.id = 0
		self.GRAVITY = 9.81
		self.DT = 0.01


	# add/read debug parameter, add debug text

	# enable/disable collisions (single and pairwise)

