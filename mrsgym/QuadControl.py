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
		self.MAX_CONTROL = 0.5
		self.P_cutoff = 1.0
		self.Pv = 0.5
		self.Iv = 0.1
		self.Dv = 1.0
		self.Pa = 4.0
		self.Ia = 0.0
		self.Da = 0.8
		# Controller Variables
		self.vel_last = None
		self.vel_int = torch.tensor([0, 0, BulletSim.GRAVITY * self.MASS / self.Iv])
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
		# Values
		if not isinstance(control, torch.Tensor):
			control = torch.tensor(control)
		wrdiff = control[3] / self.THRUST_COEFF / self.ARM_LENGTH # = w1^2 - w3^2
		wpdiff = control[2] / self.THRUST_COEFF / self.ARM_LENGTH # = w2^2 - w0^2
		wydiff = control[1] / self.DRAG_COEFF # = w0^2 - w1^2 + w2^2 - w3^2
		wtsum = control[0] / self.THRUST_COEFF # = w0^2 + w1^2 + w2^2 + w3^2
		# Limits
		wtsum = torch.clamp(wtsum, 0)
		wpdiff = torch.clamp(wpdiff, -wydiff/2 - wtsum/2, wydiff/2 + wtsum/2)  # wpdiff = -wydiff/2 - wtsum/2    wpdiff = wydiff/2 + wtsum/2
		wrdiff = torch.clamp(wrdiff, wydiff/2 - wtsum/2, -wydiff/2 + wtsum/2)  # wrdiff = wydiff/2 - wtsum/2     wrdiff = -wydiff/2 + wtsum/2
		wydiff = torch.clamp(wydiff, -2*abs(wpdiff)-wtsum, 2*abs(wrdiff)+wtsum) # (+/-)2*wpdiff - wtsum = wydiff   (+/-)2*wrdiff + wtsum = wydiff
		wtsum = torch.clamp(wtsum, abs(wpdiff) + abs(wrdiff) + abs(wydiff))
		# Motor Speed Squared Calculation
		w2s = (((wydiff + wtsum) / 2) + wpdiff) / 2
		w1s = (((wtsum - wydiff) / 2) + wrdiff) / 2
		w0s = w2s - wpdiff
		w3s = w1s - wrdiff
		speed2 = torch.tensor([w0s,w1s,w2s,w3s])
		if torch.sum(speed2<0) > 0:
			speed2 -= speed2.min() # fix rounding error
		# Motor Speeds
		speed = torch.sqrt(speed2)
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
		ang_err[0] = wrap(target_ori[0] - ori[0])
		# I ang
		self.ang_int = torch.clamp(self.ang_int + ang_err * self.DT, -self.I_ANG_MAX, self.I_ANG_MAX)
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
		control = torch.tensor([linear_body[2], angular[0], linear_body[0], -linear_body[1]])
		# Pitch/Roll Cutoffs
		pitch_diff = 0
		if abs(ori[1]) > self.MAX_PITCH:
			pitch_diff = (self.MAX_PITCH - ori[1]) if (ori[1] > 0) else (self.MAX_PITCH + ori[1])
		roll_diff = 0
		if abs(ori[2]) > self.MAX_PITCH:
			roll_diff = (self.MAX_PITCH - ori[2]) if (ori[2] > 0) else (self.MAX_PITCH + ori[2])
		control[2] += self.P_cutoff * pitch_diff
		control[3] += self.P_cutoff * roll_diff
		# Control Cutoff
		control[1:] = torch.clamp(control[1:], -self.MAX_CONTROL, self.MAX_CONTROL)
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





