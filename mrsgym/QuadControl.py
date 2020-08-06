import torch
import math
from mrsgym.BulletSim import *

# Dyanmics and Parameter Defaults:
# https://support.dce.felk.cvut.cz/mediawiki/images/5/5e/Dp_2017_gopalakrishnan_eswarmurthi.pdf

class QuadControl:

	def __init__(self, **kwargs):
		# Model Parameters
		self.THRUST_COEFF = 1.5
		self.DRAG_COEFF = 1.3
		self.ARM_LENGTH = 0.175
		self.MASS = 0.5
		self.set_constants(kwargs)
		# Controller Paremeters
		self.DT = BulletSim.DT
		self.I_VEL_MAX = float('inf')
		self.I_ANG_MAX = math.pi/2
		self.MAX_PITCH = math.pi/6
		self.Pv = 1.0
		self.Iv = 0.0
		self.Dv = 0.5
		self.Pa = 1.0
		self.Ia = 0.0
		self.Da = 0.5
		# Controller Variables
		self.vel_last = None
		self.vel_int = torch.tensor([0,0,4.0])
		self.ang_int = torch.zeros(3)

	def set_constants(self, kwargs):
		for name, val in kwargs.items():
			if name in self.__dict__:
				self.__dict__[name] = val

	# Input: angular velocity of each motor (4,)
	# Output: force at each motor and the torque in the Z direction (5,)
	# speed units: rad/s, force units: N
	def speed_to_force(self, speed=[0,0,0,0]):
		if not isinstance(speed, torch.Tensor):
			speed = torch.tensor(speed)
		w2 = speed ** 2
		forces = torch.zeros(5)
		forces[:4] = self.THRUST_COEFF * w2
		forces[4] = self.DRAG_COEFF * (w2[0]+w2[2]-w2[1]-w2[3])
		return forces

	# Input: [Fthrust, Tyaw, Tpitch, Troll]
	# Fthrust units: N, Typr units: Nm, output units: rad/s
	def control_to_speed(self, control=[0,0,0,0]):
		if not isinstance(control, torch.Tensor):
			control = torch.tensor(control)
		wrdiff = control[3] / self.THRUST_COEFF / self.ARM_LENGTH # = w1^2 - w3^2
		wpdiff = control[2] / self.THRUST_COEFF / self.ARM_LENGTH # = w0^2 - w2^2
		wydiff = control[1] / self.DRAG_COEFF # = w0^2 - w1^2 + w2^2 - w3^2
		wtsum = control[0] / self.THRUST_COEFF # = w0^2 + w1^2 + w2^2 + w3^2
		w0s = (((wydiff + wtsum) / 2) + wpdiff) / 2
		w1s = (((wtsum - wydiff) / 2) + wrdiff) / 2
		w2s = w0s - wpdiff
		w3s = w1s - wrdiff
		speed = torch.sqrt(torch.clamp(torch.tensor([w0s,w1s,w2s,w3s]), 0))
		return speed

	def control_to_force(self, control=[0,0,0,0]):
		return self.speed_to_force(self.control_to_speed(control))


	def pid_velocity(self, vel=[0,0,0], ori=[0,0,0], angvel=[0,0,0], target_vel=[0,0,0], target_ori=None):
		# Inputs
		if not isinstance(vel, torch.Tensor):
			vel = torch.tensor(vel)
		if not isinstance(ori, torch.Tensor):
			ori = torch.tensor(ori)
		if not isinstance(angvel, torch.Tensor):
			angvel = torch.tensor(angvel)
		if not isinstance(target_vel, torch.Tensor):
			target_vel = torch.tensor(target_vel)
		if target_ori is None:
			target_ori = torch.tensor([math.atan2(target_vel[1], target_vel[0]), 0, 0])
		elif not isinstance(target_ori, torch.Tensor):
			target_ori = torch.tensor(target_ori)
		# Errors
		# P vel
		vel_err = target_vel - vel
		# I vel
		self.vel_int = torch.clamp(self.vel_int + vel_err * self.DT, -self.I_VEL_MAX, self.I_VEL_MAX)
		vel_int_err = self.vel_int
		# D vel
		if self.vel_last is None:
			self.vel_last = torch.zeros(3)
		vel_dot_err = -(vel-self.vel_last) / self.DT
		self.vel_last = vel
		# P ang
		ang_err = torch.zeros(3)
		ang_err[0] = wrap(ori[0] - target_ori[0])
		# I ang
		self.ang_int = torch.clamp(wrap(self.ang_int + ang_err * self.DT), -self.I_ANG_MAX, self.I_ANG_MAX)
		ang_int_err = self.ang_int
		# D ang
		ang_dot_err = torch.zeros(3)
		ang_dot_err[0] = -angvel[0]
		# Controls (world coordinates)
		linear = self.Pv * vel_err + self.Iv * vel_int_err + self.Dv * vel_dot_err
		angular = self.Pa * ang_err + self.Ia * ang_int_err + self.Da * ang_dot_err
		# Controls (body coordinates)
		rot = torch.tensor(R.from_euler('zyx', ori, degrees=False).as_matrix()).float()
		linear_body = rot.T @ linear
		control = torch.tensor([max(linear_body[2],0), angular[0], linear_body[0], -linear_body[1]])
		return control


def wrap(angle):
	if isinstance(angle, torch.Tensor):
		angle[angle<-math.pi/2] += math.pi
		angle[angle>math.pi/2] -= math.pi
	else:
		if angle < -math.pi/2:
			angle += math.pi
		elif angle > math.pi/2:
			angle -= math.pi
	return angle





