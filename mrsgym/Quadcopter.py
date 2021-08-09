from mrsgym.BulletSim import *
from mrsgym.QuadControl import *
from mrsgym.Object import *
import pybullet as p
import numpy as np

import xml.etree.ElementTree as etxml
from scipy.optimize import nnls

class Quadcopter(Object):

	MODEL_PATH = "cf2x.urdf"

	def __init__(self, **kwargs):
		super(Quadcopter, self).__init__(uid=Quadcopter.MODEL_PATH, **kwargs)
		self.attributes = self.read_attributes()
		self.attributes.update(self.calculate_parameters())
		self.controller = QuadControl(self.attributes)

	def get_idx(self):
		return self.env.agent_idxs[self]


	# Input: [Fthrust, Troll, Tpitch, Tyaw]
	def set_control(self, control=[0,0,0,0]):
		thrust = control[0] * self.attributes["Mass"]
		roll = control[1] * self.attributes["Ixx"]
		pitch = control[2] * self.attributes["Iyy"]
		yaw = control[3] * self.attributes["Izz"]
		speeds = nnlsRPM(thrust=thrust, x_torque=roll, y_torque=pitch, z_torque=yaw, max_thrust=self.attributes['MaxThrust'], \
						max_xy_torque=self.attributes['MaxXYTorque'], max_z_torque=self.attributes['MaxZTorque'], 
						a=self.attributes['A'], inv_a=self.attributes['Ainv'], b_coeff=self.attributes['Bcoeff'])
		self.set_speeds(speeds)


	# Input: [Motor0RPM, Motor1RPM, Motor2RPM, Motor3RPM]
	def set_speeds(self, speeds=[0,0,0,0]):
		speeds = np.array(speeds)
		forces = np.array(speeds**2) * self.attributes["Kf"]
		torques = np.array(speeds**2) * self.attributes["Km"]
		z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
		for i in range(4):
			p.applyExternalForce(self.uid, linkIndex=i, forceObj=[0,0,forces[i]], posObj=[0,0,0], flags=p.LINK_FRAME, physicsClientId=self.sim.id)
		p.applyExternalTorque(self.uid, linkIndex=4, torqueObj=[0,0,z_torque], flags=p.LINK_FRAME, physicsClientId=self.sim.id)


	def set_target_accel(self, accel=[0,0,0]):
		control = self.controller.accel_control(ori=self.get_ori(), angvel=self.get_angvel(), target_accel=accel)
		self.set_control(control)


	def set_target_pos(self, pos=[0,0,0]):
		control = self.controller.pos_control(pos=self.get_pos(), vel=self.get_vel(), ori=self.get_ori(), angvel=self.get_angvel(), target_pos=pos)
		self.set_control(control)


	def set_target_vel(self, vel=[0,0,0]):
		control = self.controller.vel_control(vel=self.get_vel(), ori=self.get_ori(), angvel=self.get_angvel(), target_vel=vel)
		self.set_control(control)


	def set_target_ori(self, ori=[0,0,0]):
		speeds = self.controller.attitude_control(target_ori=ori, ori=self.get_ori(), angvel=self.get_angvel())
		self.set_control(speeds)


	def read_attributes(self):
		attributes = {}

		directory = os.path.join(os.path.dirname(mrsgym.__file__), 'models')
		path = os.path.join(directory, Quadcopter.MODEL_PATH)
		URDF_TREE = etxml.parse(path).getroot()

		attributes['Mass'] = float(URDF_TREE[1][0][1].attrib['value'])
		attributes['ArmLength'] = float(URDF_TREE[0].attrib['arm'])
		attributes['ThrustToWeightRatio'] = float(URDF_TREE[0].attrib['thrust2weight'])
		attributes['Ixx'] = float(URDF_TREE[1][0][2].attrib['ixx'])
		attributes['Iyy'] = float(URDF_TREE[1][0][2].attrib['iyy'])
		attributes['Izz'] = float(URDF_TREE[1][0][2].attrib['izz'])
		attributes['J'] = np.diag([attributes['Ixx'], attributes['Iyy'], attributes['Izz']])
		attributes['Jinv'] = np.linalg.inv(attributes['J'])
		attributes['Kf'] = float(URDF_TREE[0].attrib['kf'])
		attributes['Km'] = float(URDF_TREE[0].attrib['km'])
		attributes['CollisionH'] = float(URDF_TREE[1][2][1][0].attrib['length'])
		attributes['CollisionR'] = float(URDF_TREE[1][2][1][0].attrib['radius'])
		attributes['CollisionShapeOffsets'] = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
		attributes['CollisionZOffset'] = attributes['CollisionShapeOffsets'][2]
		attributes['MaxSpeedKMH'] = float(URDF_TREE[0].attrib['max_speed_kmh'])
		attributes['GroundEffectCoeff'] = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
		attributes['PropRadius'] = float(URDF_TREE[0].attrib['prop_radius'])
		attributes['DragCoeffXY'] = float(URDF_TREE[0].attrib['drag_coeff_xy'])
		attributes['DragCoeffZ'] = float(URDF_TREE[0].attrib['drag_coeff_z'])
		attributes['DragCoeff'] = np.array([attributes['DragCoeffXY'], attributes['DragCoeffXY'], attributes['DragCoeffZ']])
		attributes['DwCoeff1'] = float(URDF_TREE[0].attrib['dw_coeff_1'])
		attributes['DwCoeff2'] = float(URDF_TREE[0].attrib['dw_coeff_2'])
		attributes['DwCoeff3'] = float(URDF_TREE[0].attrib['dw_coeff_3'])

		return attributes


	def calculate_parameters(self):
		params = {}

		params["GravityForce"] = self.sim.GRAVITY * self.attributes["Mass"]
		params["HoverRPM"] = np.sqrt(params["GravityForce"] / (4 * self.attributes["Kf"]))
		params["MaxRPM"] = np.sqrt((self.attributes['ThrustToWeightRatio'] * params["GravityForce"]) / (4 * self.attributes["Kf"]))
		params["MaxThrust"] = (4. * self.attributes["Kf"] * params["MaxRPM"]**2)
		params["MaxXYTorque"] = np.sqrt(2) * self.attributes['ArmLength'] * self.attributes['Kf'] * params["MaxRPM"]**2 # remove sqrt(2) for the '+' configuration instead of 'x'
		params["MaxZTorque"] = 2. * self.attributes['Km'] * params["MaxRPM"]**2
		params["GroundEffectHClip"] = 0.25 * self.attributes['PropRadius'] * np.sqrt((15 * params["MaxRPM"]**2 * self.attributes["Kf"] * self.attributes['GroundEffectCoeff']) / params["MaxThrust"])

		params['A'] = np.array([ [1, 1, 1, 1], [1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2), -1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2)], [-1, 1, -1, 1] ])
		params['Ainv'] = np.linalg.inv(params['A'])
		params['Bcoeff'] = np.array([1/self.attributes["Kf"], 1/(self.attributes["Kf"]*self.attributes["ArmLength"]), 1/(self.attributes["Kf"]*self.attributes["ArmLength"]), 1/self.attributes["Km"]])

		return params



def nnlsRPM(thrust, x_torque, y_torque, z_torque, max_thrust, max_xy_torque, max_z_torque, a, inv_a, b_coeff):
	"""Non-negative Least Squares (NNLS) RPMs from desired thrust and torques.
	This function uses the NNLS implementation in `scipy.optimize`.
	Parameters
	----------
	thrust : float
		Desired thrust along the drone's z-axis.
	x_torque : float
		Desired drone's x-axis torque.
	y_torque : float
		Desired drone's y-axis torque.
	z_torque : float
		Desired drone's z-axis torque.
	max_thrust : float
		Maximum thrust of the quadcopter.
	max_xy_torque : float
		Maximum torque around the x and y axes of the quadcopter.
	max_z_torque : float
		Maximum torque around the z axis of the quadcopter.
	a : ndarray
		(4, 4)-shaped array of floats containing the motors configuration.
	inv_a : ndarray
		(4, 4)-shaped array of floats, inverse of a.
	b_coeff : ndarray
		(4,1)-shaped array of floats containing the coefficients to re-scale thrust and torques. 
	Returns
	-------
	ndarray
		(4,)-shaped array of ints containing the desired RPMs of each propeller.
	"""
	B = np.multiply(np.array([thrust, x_torque, y_torque, z_torque]), b_coeff)
	sq_rpm = np.dot(inv_a, B)
	#### NNLS if any of the desired ang vel is negative ########
	if np.min(sq_rpm) < 0:
		sol, res = nnls(a, B, maxiter=3*a.shape[1])
		sq_rpm = sol
	return np.sqrt(sq_rpm)

