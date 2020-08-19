from mrsgym.BulletSim import *
from mrsgym.QuadControl import *
from mrsgym.Object import *
import pybullet as p
import torch

class Quadcopter(Object):

	MODEL_PATH = "quadcopter.urdf"

	def __init__(self, pos=[0,0,0], ori=[0,0,0], **kwargs):
		super(Quadcopter, self).__init__(uid=Quadcopter.MODEL_PATH, pos=pos, ori=ori, **kwargs)
		self.attributes = self.read_attributes()
		self.controller = QuadControl(self.attributes)

	def read_attributes(self):
		attributes = {}
		attributes['MASS'] = p.getDynamicsInfo(self.uid, linkIndex=-1, physicsClientId=self.sim.id)[0]
		Ixx, Iyy, Izz = p.getDynamicsInfo(self.uid, linkIndex=-1, physicsClientId=self.sim.id)[2]
		attributes['Ixx'] = Ixx; attributes["Iyy"] = Iyy; attributes["Izz"] = Izz
		attributes['ARM_LENGTH'] = 0.175
		attributes["THRUST_COEFF"] = 1.5
		attributes["DRAG_COEFF"] = 1.3
		return attributes

	# Input: [F0, F1, F2, F3, Tq]
	def set_motorforces(self, forces=[0,0,0,0,0]):
		p.applyExternalForce( self.uid, linkIndex=-1, forceObj=[0.,0.,forces[0]], posObj=[self.attributes["ARM_LENGTH"],0.,0.], flags=p.LINK_FRAME, physicsClientId=self.sim.id)
		p.applyExternalForce( self.uid, linkIndex=-1, forceObj=[0.,0.,forces[1]], posObj=[0.,self.attributes["ARM_LENGTH"],0.], flags=p.LINK_FRAME, physicsClientId=self.sim.id)
		p.applyExternalForce( self.uid, linkIndex=-1, forceObj=[0.,0.,forces[2]], posObj=[-self.attributes["ARM_LENGTH"],0.,0.], flags=p.LINK_FRAME, physicsClientId=self.sim.id)
		p.applyExternalForce( self.uid, linkIndex=-1, forceObj=[0.,0.,forces[3]], posObj=[0.,-self.attributes["ARM_LENGTH"],0.], flags=p.LINK_FRAME, physicsClientId=self.sim.id)
		p.applyExternalTorque(self.uid, linkIndex=-1, torqueObj=[0.,0.,forces[4]], flags=p.LINK_FRAME, physicsClientId=self.sim.id)

	# Input: [w0, w1, w2, w3]
	def set_speeds(self, speeds=[0,0,0,0]):
		self.set_motorforces(self.controller.speed_to_motorforce(speeds))

	# Input: [Fthrust, Troll, Tpitch, Tyaw]
	def set_control(self, control=[0,0,0,0]):
		self.set_motorforces(self.controller.control_to_motorforce(control))


	def set_force(self, forces=[0,0,0]):
		control = self.controller.force_control(ori=self.get_ori(), angvel=self.get_angvel(), target_force=forces)
		self.set_control(control)


	def set_target_pos(self, pos=[0,0,0]):
		control = self.controller.pos_control(pos=self.get_pos(), vel=self.get_vel(), ori=self.get_ori(), angvel=self.get_angvel(), target_pos=pos)
		self.set_control(control)


	def set_target_vel(self, vel=[0,0,0]):
		control = self.controller.vel_control(vel=self.get_vel(), ori=self.get_ori(), angvel=self.get_angvel(), target_vel=vel)
		self.set_control(control)

