import torch
import numpy as np
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
		self.MAX_ROLL_PITCH = np.pi/6

	def set_constants(self, kwargs):
		for name, val in kwargs.items():
			if name in self.__dict__:
				self.__dict__[name] = val

	# Input: angular velocity of each motor (4,)
	# Output: force at each motor and the torque in the Z direction (5,)
	# speed units: rad/s, force units: N
	def speed_to_motorforce(self, speed=[0,0,0,0]):
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


	def control_to_motorforce(self, control=[0,0,0,0]):
		return self.speed_to_motorforce(self.control_to_speed(control))


	def pid_control(self, pos, vel, ori, angvel, target_pos=None, target_vel=None):
		pos = np.array(pos)
		vel = np.array(vel)
		target_pos = np.array(target_pos)
		pos_e = target_pos - pos
		if not hasattr(self, 'integral_pos_e'):
			self.integral_pos_e = np.zeros(3)
		d_pos_e = -vel # TODO: add velocity control
		self.integral_pos_e = self.integral_pos_e + pos_e*BulletSim.DT
		P_COEFF_FOR = np.array([.3, .3, .6]); I_COEFF_FOR = np.array([.0001, .0001, .0001]); D_COEFF_FOR = np.array([.6, .6, .8])
		target_force = np.array([0,0,BulletSim.GRAVITY*self.MASS]) + np.multiply(P_COEFF_FOR,pos_e) + np.multiply(I_COEFF_FOR,self.integral_pos_e) + np.multiply(D_COEFF_FOR,d_pos_e)
		return self.force_control(ori, angvel, target_force)



	def force_control(self, ori, angvel, target_force):
		cur_rpy = np.flip(np.array(ori))
		cur_angvel = np.flip(np.array(angvel))
		cur_rotation = torch.tensor(R.from_euler('zyx', ori, degrees=False).as_matrix()).float()
		if not hasattr(self, 'integral_rpy_e'):
			self.integral_rpy_e = np.zeros(3)
		computed_target_rpy = np.zeros(3)
		sign_z =  np.sign(target_force[2])
		if sign_z == 0: sign_z = 1 
		computed_target_rpy[0] = np.arcsin(-sign_z*target_force[1] / np.linalg.norm(target_force))
		computed_target_rpy[1] = np.arctan2(sign_z*target_force[0],sign_z*target_force[2])
		computed_target_rpy[2] = 0. # TODO: add yaw control
		computed_target_rpy[0] = np.clip(computed_target_rpy[0], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
		computed_target_rpy[1] = np.clip(computed_target_rpy[1], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
		rpy_e = wrap(computed_target_rpy - cur_rpy)
		d_rpy_e = -cur_angvel
		self.integral_rpy_e = self.integral_rpy_e + rpy_e*BulletSim.DT
		P_COEFF_TOR = np.array([.3, .3, .05]); I_COEFF_TOR = np.array([.0001, .0001, .0001]); D_COEFF_TOR = np.array([.3, .3, .5])
		target_torques = np.multiply(P_COEFF_TOR,rpy_e) + np.multiply(I_COEFF_TOR,self.integral_rpy_e) + np.multiply(D_COEFF_TOR,d_rpy_e)
		target_force = np.dot(cur_rotation, target_force)
		control = torch.tensor([target_force[2], target_torques[2], target_torques[1], target_torques[0]])
		return control



def wrap(angle):
	if isinstance(angle, np.ndarray):
		angle[angle<-np.pi] += 2*np.pi
		angle[angle>np.pi] -= 2*np.pi
	else:
		if angle < -np.pi:
			angle += 2*np.pi
		elif angle > np.pi:
			angle -= 2*np.pi
	return angle





