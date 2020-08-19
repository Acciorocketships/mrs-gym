import torch
import numpy as np
from mrsgym.BulletSim import *
from mrsgym.Util import wrap_angle

# Dyanmics and Parameter Defaults:
# https://support.dce.felk.cvut.cz/mediawiki/images/5/5e/Dp_2017_gopalakrishnan_eswarmurthi.pdf

class QuadControl:

	def __init__(self, attributes, sim=DefaultSim(), **kwargs):
		# Model Parameters
		self.attributes = attributes
		self.sim = sim
		# Controller Paremeters
		self.MAX_ROLL_PITCH = np.pi/6


	# Input: angular velocity of each motor (4,)
	# Output: force at each motor and the torque in the Z direction (5,)
	# speed units: rad/s, force units: N
	def speed_to_motorforce(self, speed=[0,0,0,0]):
		if not isinstance(speed, torch.Tensor):
			speed = torch.tensor(speed)
		w2 = speed ** 2
		forces = torch.zeros(5)
		forces[:4] = self.attributes["THRUST_COEFF"] * w2
		forces[4] = self.attributes["DRAG_COEFF"] * (w2[0]+w2[2]-w2[1]-w2[3])
		return forces

	# Input: [Fthrust, Troll, Tpitch, Tyaw]
	# Fthrust units: N, Typr units: Nm, output units: rad/s
	def control_to_speed(self, control=[0,0,0,0]):
		# Values
		if not isinstance(control, torch.Tensor):
			control = torch.tensor(control)
		wrdiff = control[1] / self.attributes["THRUST_COEFF"] / self.attributes["ARM_LENGTH"] # = w1^2 - w3^2
		wpdiff = control[2] / self.attributes["THRUST_COEFF"] / self.attributes["ARM_LENGTH"] # = w2^2 - w0^2
		wydiff = control[3] / self.attributes["DRAG_COEFF"] # = w0^2 - w1^2 + w2^2 - w3^2
		wtsum = control[0] / self.attributes["THRUST_COEFF"] # = w0^2 + w1^2 + w2^2 + w3^2
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


	def pos_control(self, pos, vel, ori, angvel, target_pos=None, target_ori=None, target_vel=np.zeros(3)):
		pos = np.array(pos)
		vel = np.array(vel)
		target_pos = np.array(target_pos)
		target_vel = np.array(target_vel)
		pos_e = target_pos - pos
		if not hasattr(self, 'integral_pos_e'):
			self.integral_pos_e = np.zeros(3)
		d_pos_e = target_vel - vel
		self.integral_pos_e = self.integral_pos_e + pos_e*self.sim.DT
		P_COEFF_FOR = np.array([.4, .4, .6]); I_COEFF_FOR = np.array([.0001, .0001, .0001]); D_COEFF_FOR = np.array([1.0, 1.0, 1.0])
		target_force = np.array([0,0,self.sim.GRAVITY*self.attributes["MASS"]]) + np.multiply(P_COEFF_FOR,pos_e) + np.multiply(I_COEFF_FOR,self.integral_pos_e) + np.multiply(D_COEFF_FOR,d_pos_e)
		if target_ori is None:
			target_ori = np.zeros(3)
		return self.force_control(ori=ori, angvel=angvel, target_force=target_force, target_ori=target_ori)


	def vel_control(self, vel, ori, angvel, target_vel=np.zeros(3), target_ori=None):
		vel = np.array(vel)
		target_vel = np.array(target_vel)
		vel_e = target_vel - vel
		if not hasattr(self, 'last_vel_e'):
			self.last_vel_e = vel_e
			self.d_vel_e = np.zeros(3)
		if not hasattr(self, 'integral_vel_e'):
			self.integral_vel_e = np.zeros(3)
		self.d_vel_e = ((vel_e - self.last_vel_e) / self.sim.DT) * 0.5 + self.d_vel_e * 0.5
		self.last_vel_e = vel_e
		self.integral_vel_e = self.integral_vel_e + vel_e*self.sim.DT
		P_COEFF_FOR = np.array([2.0, 2.0, 2.0]); I_COEFF_FOR = np.array([.001, .001, .001]); D_COEFF_FOR = np.array([0.3, 0.3, 0.3])
		target_force = np.array([0,0,self.sim.GRAVITY*self.attributes["MASS"]]) + np.multiply(P_COEFF_FOR,vel_e) + np.multiply(I_COEFF_FOR,self.integral_vel_e) + np.multiply(D_COEFF_FOR,self.d_vel_e)
		if target_ori is None:
			target_ori = wrap_angle(np.array([0, 0, np.arctan2(vel[0], vel[1])]), margin=np.pi/2)
		# print("P: %s, I: %s, D: %s" % (vel_e, self.integral_vel_e, d_vel_e))
		return self.force_control(ori=ori, angvel=angvel, target_force=target_force, target_ori=target_ori)


	def force_control(self, ori, angvel, target_force, target_ori=None):
		ori = np.array(ori)
		angvel = np.array(angvel)
		target_force = np.array(target_force)
		cur_rotation = torch.tensor(R.from_euler('xyz', ori, degrees=False).as_matrix()).float()
		if target_ori is None:
			target_yaw = np.arctan2(target_force[1],target_force[0])
		else:
			target_yaw = target_ori[2]
		if not hasattr(self, 'integral_rpy_e'):
			self.integral_rpy_e = np.zeros(3)
		target_ori = np.zeros(3)
		sign_z =  np.sign(target_force[2])
		if sign_z == 0: sign_z = 1 
		target_ori[0] = np.arcsin(-sign_z*target_force[1] / np.linalg.norm(target_force))
		target_ori[1] = np.arctan2(sign_z*target_force[0],sign_z*target_force[2])
		target_ori[2] = wrap_angle(target_yaw, margin=np.pi/2)
		target_ori[0] = np.clip(target_ori[0], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
		target_ori[1] = np.clip(target_ori[1], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
		rpy_e = wrap_angle(target_ori - ori)
		d_rpy_e = -angvel
		self.integral_rpy_e = self.integral_rpy_e + rpy_e*self.sim.DT
		P_COEFF_TOR = np.array([.3, .3, .1]); I_COEFF_TOR = np.array([.0001, .0001, .0001]); D_COEFF_TOR = np.array([.3, .3, .5])
		target_torques = np.multiply(P_COEFF_TOR,rpy_e) + np.multiply(I_COEFF_TOR,self.integral_rpy_e) + np.multiply(D_COEFF_TOR,d_rpy_e)
		target_force = np.dot(cur_rotation, target_force)
		control = torch.tensor([target_force[2], target_torques[0], target_torques[1], target_torques[2]])
		return control





