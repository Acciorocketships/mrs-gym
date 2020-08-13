from mrsgym.BulletSim import *
from mrsgym.QuadControl import *
from mrsgym.Object import *
import pybullet as p
import torch

class Quadcopter(Object):

	ARM_LENGTH = 0.175 # TODO: read this from getLinkState
	MODEL_PATH = "quadcopter.urdf"

	def __init__(self, pos=[0,0,0], ori=[0,0,0]):
		super(Quadcopter, self).__init__(Quadcopter.MODEL_PATH, pos, ori)
		self.controller = QuadControl()

	# Input: [F0, F1, F2, F3, Tq]
	def set_motorforces(self, forces=[0,0,0,0,0]):
		p.applyExternalForce( self.model,-1, forceObj=[0.,0.,forces[0]], posObj=[Quadcopter.ARM_LENGTH,0.,0.], flags=p.LINK_FRAME)
		p.applyExternalForce( self.model,-1, forceObj=[0.,0.,forces[1]], posObj=[0.,Quadcopter.ARM_LENGTH,0.], flags=p.LINK_FRAME)
		p.applyExternalForce( self.model,-1, forceObj=[0.,0.,forces[2]], posObj=[-Quadcopter.ARM_LENGTH,0.,0.], flags=p.LINK_FRAME)
		p.applyExternalForce( self.model,-1, forceObj=[0.,0.,forces[3]], posObj=[0.,-Quadcopter.ARM_LENGTH,0.], flags=p.LINK_FRAME)
		p.applyExternalTorque(self.model,-1, torqueObj=[0.,0.,forces[4]], flags=p.LINK_FRAME)

	# Input: [w0, w1, w2, w3]
	def set_speeds(self, speeds=[0,0,0,0]):
		self.set_motorforces(self.controller.speed_to_motorforce(speeds))

	# Input: [Fthrust, Tyaw, Tpitch, Troll]
	def set_control(self, controls=[0,0,0,0]):
		self.set_motorforces(self.controller.control_to_motorforce(controls))


	def set_force(self, forces=[0,0,0]):
		controls = self.controller.force_control(ori=self.get_ori(), angvel=self.get_angvel(), target_force=forces)
		self.set_controls(controls)


	def set_target_pos(self, pos=[0,0,0]):
		controls = self.controller.pid_control(pos=self.get_pos(), vel=self.get_vel(), ori=self.get_ori(), angvel=self.get_angvel(), target_pos=pos)
		self.set_controls(controls)

