from mrsgym.BulletSim import *
import pybullet as p
from scipy.spatial.transform import Rotation as R
import torch


class Object:

	def __init__(self, uid=None, pos=[0,0,0], ori=[0,0,0], inpackage=True):
		if isinstance(uid, str):
			self.uid = load_urdf(uid, pos=pos, ori=ori, inpackage=inpackage)
		else:
			self.uid = uid
			self.set_state(pos=pos, ori=ori)


	def __del__(self):
		p.removeBody(self.uid)


	# resets the given inputs, and keeps the others the same
	def set_state(self, pos=None, ori=None, vel=None, angvel=None):
		# pos
		if pos is None:
			pos = self.get_pos()
		if isinstance(pos, torch.Tensor):
			pos = pos.tolist()
		# ori
		if ori is None:
			ori = self.get_ori()
		if isinstance(ori, list):
			ori = torch.tensor(ori)
		if ori.shape == (3,3):
			r = R.from_matrix(ori)
			ori = torch.tensor(r.as_quat())
		elif ori.shape == (3,):
			r = R.from_euler('xyz', ori, degrees=False)
			ori = torch.tensor(r.as_quat())
		ori = ori.tolist()
		# vel
		if vel is None:
			vel = self.get_vel()
		if isinstance(vel, torch.Tensor):
			vel = vel.tolist()
		# angvel
		if angvel is None:
			angvel = self.get_angvel()
		if isinstance(angvel, torch.Tensor):
			angvel = angvel.tolist()
		# reset
		p.resetBasePositionAndOrientation(self.uid, posObj=pos, ornObj=ori)
		p.resetBaseVelocity(self.uid, linearVelocity=vel, angularVelocity=angvel)


	def get_joint_info(self):
		numJoints = p.getNumJoints(self.uid)
		return {n: p.getJointInfo(self.uid, jointIndex=n) for n in range(numJoints)}


	def get_joint_state(self):
		numJoints = p.getNumJoints(self.uid)
		return {n: p.getJointState(self.uid, jointIndex=n) for n in range(numJoints)}


	def get_vel(self):
		return torch.tensor(p.getBaseVelocity(self.uid)[0])


	def get_angvel(self, mat=False):
		return torch.tensor(p.getBaseVelocity(self.uid)[1])
		

	def get_pos(self):
		return torch.tensor(p.getBasePositionAndOrientation(self.uid)[0]).float()


	def get_ori(self, mat=False):
		quat = torch.tensor(p.getBasePositionAndOrientation(self.uid)[1])
		r = R.from_quat(quat)
		if mat:
			return torch.tensor(r.as_matrix()).float()
		else:
			return torch.tensor(r.as_euler('xyz', degrees=False).copy()).float()


	def get_contact_points(self, other=None):
		if other is None: # filter by specific ID of another object
			contactObjList = p.getContactPoints(self.uid)
		else:
			contactObjList = p.getContactPoints(self.uid, filterBodyUniqueIdB=other)
		contacts = []
		for contactObj in contactObjList:
			contact = {}
			contact['id'] = contactObj[2]
			contact['pos'] = torch.tensor(contactObj[5]) # world coordinates
			contact['normal'] = contactObj[9] * torch.tensor(contactObj[7]) # direction and magnitude of normal force
			contact['distance'] = contactObj[8]
			contacts.append(contact)
		return contacts


	def collision(self):
		return any(map(lambda contact: contact['distance']<0.02, self.get_contact_points())) # TODO: use getPhysicsEngineParameters for 0.02


	def get_nearby_objects(self):
		pass


	def raycast(self):
		pass


	def get_image(self):
		pass

