import torch
import numpy as np
import math
from mrsgym.BulletSim import *
from mrsgym.Util import wrap_angle
from scipy.spatial.transform import Rotation

class QuadControl:

	def __init__(self, attributes, sim=DefaultSim(), **kwargs):
		# Model Parameters
		self.attributes = attributes
		self.sim = sim
		self.control_params = {
			"PosCtrl_P": np.array([4., 4., 4.]),
			"PosCtrl_I": np.array([.1, .1, .1]),
			"PosCtrl_D": np.array([2., 2., 2.]),
			"VelCtrl_P": np.array([4., 4., 4.]),
			"VelCtrl_I": np.array([.1, .1, .1]),
			"VelCtrl_D": np.array([2., 2., 2.]),
			"AccelCtrl_P": np.array([10., 10., 10.]),
			"AccelCtrl_I": np.array([.01, .01, .01]),
			"AccelCtrl_D": np.array([5., 5., 5.]),
			"OriCtrl_P": np.array([70000., 70000., 60000.]),
			"OriCtrl_I": np.array([.0, .0, 500.]),
			"OriCtrl_D": np.array([20000., 20000., 12000.]),
			"MixerMatrix": np.array([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ]),
			"MinPWM": 20000,
			"MaxPWM": 65535,
			"PWMtoRPM_A": 0.2685,
			"PWMtoRPM_B": 4070.3,
		}


	def pos_control(self, pos, vel, ori, angvel, target_pos, target_yaw=None, target_vel=np.zeros(3)):
		pos = np.array(pos)
		vel = np.array(vel)
		target_pos = np.array(target_pos)
		target_vel = np.array(target_vel)
		pos_e = target_pos - pos
		if not hasattr(self, 'integral_pos_e'):
			self.integral_pos_e = np.zeros(3)
		d_pos_e = target_vel - vel
		self.integral_pos_e = self.integral_pos_e + pos_e * self.sim.DT
		target_accel =  np.multiply(self.control_params["PosCtrl_P"], pos_e) + \
						np.multiply(self.control_params["PosCtrl_I"], self.integral_pos_e) + \
						np.multiply(self.control_params["PosCtrl_D"], d_pos_e)
		return self.accel_control(ori=ori, angvel=angvel, target_accel=target_accel, target_yaw=target_yaw)


	def vel_control(self, vel, ori, angvel, target_vel=np.zeros(3), target_yaw=None):
		vel = np.array(vel)
		target_vel = np.array(target_vel)
		vel_e = target_vel - vel
		if not hasattr(self, 'last_vel_e'):
			self.last_vel_e = vel_e
			self.d_vel_e = np.zeros(3)
		if not hasattr(self, 'last_target_vel'):
			self.last_target_vel = target_vel
		if not hasattr(self, 'integral_vel_e'):
			self.integral_vel_e = np.zeros(3)
		self.d_vel_e = (((vel_e - self.last_vel_e) - (target_vel - self.last_target_vel)) / self.sim.DT) * 0.5 + self.d_vel_e * 0.5
		# self.d_vel_e = ((vel_e - self.last_vel_e) / self.sim.DT)
		self.last_vel_e = vel_e
		self.last_target_vel = target_vel
		self.integral_vel_e = self.integral_vel_e + vel_e * self.sim.DT
		target_accel = np.multiply(self.control_params["VelCtrl_P"], vel_e) + \
						np.multiply(self.control_params["VelCtrl_I"], self.integral_vel_e) + \
						np.multiply(self.control_params["VelCtrl_D"], self.d_vel_e)
		return self.accel_control(ori=ori, angvel=angvel, target_accel=target_accel, target_yaw=target_yaw)


	def accel_control(self, target_accel, ori, angvel, target_yaw=None):
		ori = np.array(ori)
		angvel = np.array(angvel)
		target_accel = np.array(target_accel) + np.array([0., 0., self.sim.GRAVITY])
		rotation = torch.tensor(R.from_euler('xyz', ori, degrees=False).as_matrix()).float()
		target_rotation_z = target_accel / np.linalg.norm(target_accel)
		if np.any(np.isnan(target_rotation_z)):
			target_rotation_z = np.array([0.,0.,1.])
		if target_yaw is None:
			target_rotation_x = np.cross(rotation[:,1], target_rotation_z)
			target_rotation_y = np.cross(target_rotation_z, target_rotation_x)
		else:
			dir_unit_vec = np.array([np.cos(target_yaw), np.sin(target_yaw), 0])
			target_rotation_y = np.cross(target_rotation_z, dir_unit_vec)
			target_rotation_x = np.cross(target_rotation_y, target_rotation_z)
		target_rotation = np.stack([target_rotation_x, target_rotation_y, target_rotation_z], axis=1)
		target_ori = R.from_matrix(target_rotation).as_euler('xyz', degrees=False)
		return self.attitude_control(target_accel=target_accel, target_ori=target_ori, ori=ori, angvel=angvel)


	def attitude_control(self, target_accel, target_ori, ori, angvel, target_angvel=[0,0,0]):
		ori = np.array(ori)
		target_ori = np.array(target_ori)
		target_angvel = np.array(target_angvel)
		rotation = R.from_euler('xyz', ori, degrees=False).as_matrix()
		target_rotation = R.from_euler('xyz', target_ori, degrees=False).as_matrix()
		rot_matrix_e = np.dot((target_rotation.transpose()), rotation) - np.dot(rotation.transpose(), target_rotation)
		rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
		if not hasattr(self, 'last_ori'):
			self.last_ori = ori
			self.integral_ori_e = np.zeros(3)
		angvel_e = target_angvel - angvel
		self.last_ori = ori
		self.integral_ori_e = self.integral_ori_e - rot_e * self.sim.DT
		self.integral_ori_e = np.clip(self.integral_ori_e, -1500., 1500.)
		self.integral_ori_e[0:2] = np.clip(self.integral_ori_e[0:2], -1., 1.)
		#### PID target torques ####################################
		target_torques = - np.multiply(self.control_params["OriCtrl_P"], rot_e) \
						 + np.multiply(self.control_params["OriCtrl_I"], self.integral_ori_e) \
						 + np.multiply(self.control_params["OriCtrl_D"], angvel_e)
		target_torques = np.clip(target_torques, -3200, 3200)
		scalar_thrust = max(0., np.dot(target_accel*self.attributes["Mass"], rotation[:,2]))
		thrust = (math.sqrt(scalar_thrust / (4*self.attributes["Kf"])) - self.control_params["PWMtoRPM_B"]) / self.control_params["PWMtoRPM_A"]
		pwm = thrust + np.dot(self.control_params["MixerMatrix"], target_torques)
		pwm = np.clip(pwm, self.control_params["MinPWM"], self.control_params["MaxPWM"])
		rpm = self.control_params["PWMtoRPM_A"] * pwm + self.control_params["PWMtoRPM_B"]
		return rpm



