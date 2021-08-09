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
			"PosCtrl_P": np.array([10., 10., 10.]),
			"PosCtrl_I": np.array([.01, .01, .01]),
			"PosCtrl_D": np.array([5., 5., 5.]),
			"VelCtrl_P": np.array([2., 2., 1.]),
			"VelCtrl_I": np.array([.001, .001, .001]),
			"VelCtrl_D": np.array([2., 2., 1.]),
			"AccelCtrl_P": np.array([10., 10., 10.]),
			"AccelCtrl_I": np.array([.01, .01, .01]),
			"AccelCtrl_D": np.array([5., 5., 5.]),
			"OriCtrl_P": np.array([3., 3., 3.]),
			"OriCtrl_I": np.array([.01, .01, .01]),
			"OriCtrl_D": np.array([3., 3., 3.]),
			"MaxRollPitch": np.pi/6,
		}


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
		target_accel =  np.multiply(self.control_params["PosCtrl_P"], pos_e) + \
						np.multiply(self.control_params["PosCtrl_I"], self.integral_pos_e) + \
						np.multiply(self.control_params["PosCtrl_D"], d_pos_e)
		if target_ori is None:
			target_ori = np.zeros(3)
		return self.accel_control(ori=ori, angvel=angvel, target_accel=target_accel, target_ori=target_ori)


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
		self.d_vel_e = (((vel_e - self.last_vel_e) - (target_vel - self.last_target_vel)) / self.sim.DT) * 1 + self.d_vel_e * 0
		self.last_vel_e = vel_e
		self.last_target_vel = target_vel
		self.integral_vel_e = self.integral_vel_e + vel_e * self.sim.DT
		target_accel = np.multiply(self.control_params["VelCtrl_P"], vel_e) + \
						np.multiply(self.control_params["VelCtrl_I"], self.integral_vel_e) + \
						np.multiply(self.control_params["VelCtrl_D"], self.d_vel_e)
		# print("P: %s, I: %s, D: %s" % (vel_e, self.integral_vel_e, d_vel_e))
		# import pdb; pdb.set_trace()
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
		return self.attitude_control(target_ori=target_ori, ori=ori, angvel=angvel, target_thrust=np.linalg.norm(target_accel))


	def attitude_control(self, target_ori, ori, angvel, target_thrust=None, target_angvel=np.zeros(3)):
		ori = np.array(ori)
		angvel = np.array(angvel)
		target_ori = np.array(target_ori)
		target_angvel = np.array(target_angvel)
		rotation = R.from_euler('xyz', ori, degrees=False).as_matrix()
		target_rotation = R.from_euler('xyz', target_ori, degrees=False).as_matrix()
		rot_matrix_e = np.dot((target_rotation.transpose()), rotation) - np.dot(rotation.transpose(), target_rotation)
		ori_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
		angvel_e = target_angvel - angvel
		if not hasattr(self, 'integral_ori_e'):
			self.integral_ori_e = np.zeros(3)
		self.integral_ori_e = self.integral_ori_e + ori_e * self.sim.DT
		target_torques = - np.multiply(self.control_params["OriCtrl_P"], ori_e) \
						 - np.multiply(self.control_params["OriCtrl_I"], self.integral_ori_e) \
						 + np.multiply(self.control_params["OriCtrl_D"], angvel_e)
		if target_thrust is None:
			# C = 1.0
			# scaler = (1 + C) / (rotation[2,2] + C) # dot product of z-axis with [0,0,1] = cos(phi), where phi is the angle from vertical
			scaler = 1.0
			target_thrust = scaler * self.sim.GRAVITY
		# import pdb; pdb.set_trace()
		target_thrust = np.minimum(target_thrust, self.attributes["MaxThrust"]/self.attributes["Mass"])
		roll_torque = np.clip(target_torques[0], -self.attributes["MaxXYTorque"]/self.attributes["Ixx"], self.attributes["MaxXYTorque"]/self.attributes["Ixx"])
		pitch_torque = np.clip(target_torques[1], -self.attributes["MaxXYTorque"]/self.attributes["Iyy"], self.attributes["MaxXYTorque"]/self.attributes["Iyy"])
		yaw_torque = np.clip(target_torques[2], -self.attributes["MaxZTorque"]/self.attributes["Izz"], self.attributes["MaxZTorque"]/self.attributes["Izz"])
		control = np.array([target_thrust, roll_torque, pitch_torque, yaw_torque])
		return control



	# def accel_control(self, ori, angvel, target_accel, target_yaw=None):
	# 	ori = np.array(ori)
	# 	angvel = np.array(angvel)
	# 	target_accel = np.array(target_accel)
	# 	cur_rotation = torch.tensor(R.from_euler('xyz', ori, degrees=False).as_matrix()).float()
	# 	if target_yaw is None:
	# 		target_yaw = ori[2]
	# 	if not hasattr(self, 'integral_rpy_e'):
	# 		self.integral_rpy_e = np.zeros(3)
	# 	sign_z =  np.sign(target_accel[2])
	# 	if sign_z == 0: sign_z = 1 
	# 	target_ori = np.zeros(3)
	# 	target_ori[0] = np.arctan2(-sign_z*target_accel[1], sign_z*target_accel[2])
	# 	target_ori[1] = np.arctan2(sign_z*target_accel[0], sign_z*target_accel[2])
	# 	target_ori[2] = wrap_angle(target_yaw, margin=np.pi/2)
	# 	target_ori[0] = np.clip(target_ori[0], -self.control_params["MaxRollPitch"], self.control_params["MaxRollPitch"])
	# 	target_ori[1] = np.clip(target_ori[1], -self.control_params["MaxRollPitch"], self.control_params["MaxRollPitch"])
	# 	rpy_e = wrap_angle(target_ori - ori)
	# 	d_rpy_e = -angvel
	# 	self.integral_rpy_e = self.integral_rpy_e + rpy_e*self.sim.DT
	# 	target_torques = np.multiply(self.control_params["AccelCtrl_P"], rpy_e) + \
	# 					np.multiply(self.control_params["AccelCtrl_I"], self.integral_rpy_e) + \
	# 					np.multiply(self.control_params["AccelCtrl_D"], d_rpy_e)
	# 	target_accel = np.dot(cur_rotation, target_accel + np.array([0., 0., self.sim.GRAVITY]))
	# 	control = torch.tensor([target_accel[2], target_torques[0], target_torques[1], target_torques[2]])
	# 	print("target: ", target_accel)
	# 	return control


	# def pos_control(self, pos, vel, ori, angvel, target_pos, target_ori=np.zeros(3), target_vel=np.zeros(3)):
	# 	pos = np.array(pos)
	# 	vel = np.array(vel)
	# 	target_pos = np.array(target_pos)
	# 	target_vel = np.array(target_vel)
	# 	rotation = R.from_euler('xyz', ori, degrees=False).as_matrix()
	# 	pos_e = target_pos - pos
	# 	if not hasattr(self, 'integral_pos_e'):
	# 		self.integral_pos_e = np.zeros(3)
	# 	vel_e = target_vel - vel
	# 	self.integral_pos_e = self.integral_pos_e + pos_e*self.sim.DT
	# 	self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
	# 	self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)
	# 	#### PID target thrust #####################################
	# 	target_thrust = np.multiply(self.control_params["PosCtrl_P"], pos_e) \
	# 					+ np.multiply(self.control_params["PosCtrl_I"], self.integral_pos_e) \
	# 					+ np.multiply(self.control_params["PosCtrl_D"], vel_e) \
	# 					+ np.array([0, 0, self.attributes["GravityForce"]])
	# 	scalar_thrust = max(0., np.dot(target_thrust, rotation[:,2]))
	# 	thrust = (math.sqrt(scalar_thrust / (4*self.attributes["Kf"])) - self.control_params["PWMtoRPM_B"]) / self.control_params["PWMtoRPM_A"]
	# 	target_z_ax = target_thrust / np.linalg.norm(target_thrust)
	# 	target_x_c = np.array([math.cos(target_ori[2]), math.sin(target_ori[2]), 0])
	# 	target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
	# 	target_x_ax = np.cross(target_y_ax, target_z_ax)
	# 	target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
	# 	#### Target rotation #######################################
	# 	target_ori_new = (Rotation.from_matrix(target_rotation)).as_euler('xyz', degrees=False)
	# 	return self.attitude_control(thrust=thrust, ori=ori, target_ori=target_ori_new)


	# def attitude_control(self, thrust, ori, target_ori, target_angvel=[0,0,0]):
	# 	ori = np.array(ori)
	# 	target_ori = np.array(target_ori)
	# 	target_angvel = np.array(target_angvel)
	# 	rotation = R.from_euler('xyz', ori, degrees=False).as_matrix()
	# 	target_rotation = R.from_euler('xyz', target_ori, degrees=False).as_matrix()
	# 	rot_matrix_e = np.dot((target_rotation.transpose()), rotation) - np.dot(rotation.transpose(), target_rotation)
	# 	rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
	# 	if not hasattr(self, 'last_ori'):
	# 		self.last_ori = ori
	# 		self.integral_ori_e = np.zeros(3)
	# 	angvel_e = target_angvel - (ori - self.last_ori) / self.sim.DT
	# 	self.last_ori = ori
	# 	self.integral_ori_e = self.integral_ori_e - rot_e * self.sim.DT
	# 	self.integral_ori_e = np.clip(self.integral_ori_e, -1500., 1500.)
	# 	self.integral_ori_e[0:2] = np.clip(self.integral_ori_e[0:2], -1., 1.)
	# 	#### PID target torques ####################################
	# 	target_torques = - np.multiply(self.control_params["OriCtrl_P"], rot_e) \
	# 					 + np.multiply(self.control_params["OriCtrl_I"], self.integral_ori_e) \
	# 					 + np.multiply(self.control_params["OriCtrl_D"], angvel_e)
	# 	target_torques = np.clip(target_torques, -3200, 3200)
	# 	pwm = thrust + np.dot(self.control_params["MixerMatrix"], target_torques)
	# 	pwm = np.clip(pwm, self.control_params["MinPWM"], self.control_params["MaxPWM"])
	# 	control_force = self.control_params["PWMtoRPM_A"] * pwm + self.control_params["PWMtoRPM_B"]
	# 	# control_accel = np.array([control_force[0]/self.attributes["Mass"], control_force[1]/self.attributes["Ixx"], control_force[2]/self.attributes["Iyy"], control_force[3]/self.attributes["Izz"]])
	# 	return control_force

	# self.control_params = {
	# 	"PosCtrl_P": np.array([.4, .4, 1.25]),
	# 	"PosCtrl_I": np.array([.05, .05, .05]),
	# 	"PosCtrl_D": np.array([.2, .2, .5]),
	# 	"VelCtrl_P": np.array([.4, .4, 1.25]),
	# 	"VelCtrl_I": np.array([.05, .05, .05]),
	# 	"VelCtrl_D": np.array([.2, .2, .5]),
	# 	"OriCtrl_P": np.array([70000., 70000., 60000.]),
	# 	"OriCtrl_I": np.array([.0, .0, 500.]),
	# 	"OriCtrl_D": np.array([20000., 20000., 12000.]),
	# 	"MaxRollPitch": np.pi/6,
	# 	"MixerMatrix": np.array([ [.5, -.5,  -1], [.5, .5, 1], [-.5,  .5,  -1], [-.5, -.5, 1] ]),
	# 	"MinPWM": 20000,
	# 	"MaxPWM": 65535,
	# 	"PWMtoRPM_A": 0.2685,
	# 	"PWMtoRPM_B": 4070.3,
	# }





