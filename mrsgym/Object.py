import pybullet as p
from mrsgym.BulletSim import *
from scipy.spatial.transform import Rotation as R
import torch


class Object:

	def __init__(self, model=None, pos=[0,0,0], ori=[0,0,0], inpackage=True):
		if isinstance(model, str):
			self.model = load_urdf(model, pos=pos, ori=ori, inpackage=inpackage)
		else:
			self.model = model
			self.set_state(pos=pos, ori=ori)

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
			r = R.from_euler('zyx', ori, degrees=False)
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
		p.resetBasePositionAndOrientation(self.model, posObj=pos, ornObj=ori)
		p.resetBaseVelocity(self.model, linearVelocity=vel, angularVelocity=angvel)


	def get_joint_info(self):
		numJoints = p.getNumJoints(self.model)
		return {n: p.getJointInfo(self.model, jointIndex=n) for n in range(numJoints)}


	def get_joint_state(self):
		numJoints = p.getNumJoints(self.model)
		return {n: p.getJointState(self.model, jointIndex=n) for n in range(numJoints)}


	def get_vel(self):
		return torch.tensor(p.getBaseVelocity(self.model)[0])


	def get_angvel(self, mat=False):
		return torch.tensor(p.getBaseVelocity(self.model)[1])
		

	def get_pos(self):
		return torch.tensor(p.getBasePositionAndOrientation(self.model)[0]).float()


	def get_ori(self, mat=False):
		quat = torch.tensor(p.getBasePositionAndOrientation(self.model)[1])
		r = R.from_quat(quat)
		if mat:
			return torch.tensor(r.as_matrix()).float()
		else:
			return torch.tensor(r.as_euler('zyx', degrees=False).copy()).float()


	def get_contact_points(self, other=None):
		if other is None: # filter by specific ID of another object
			contactObjList = p.getContactPoints(self.model)
		else:
			contactObjList = p.getContactPoints(self.model, filterBodyUniqueIdB=other)
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


def create_box(dim=[1,1,1], pos=[0,0,0], ori=[0,0,0], mass=0, collisions=True):
	if isinstance(dim, torch.Tensor):
		dim = dim.tolist()
	if collisions:
		model = p.createCollisionShape(p.GEOM_BOX, halfExtents=dim)
		if mass != 0:
			model = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=model)
	else:
		model = p.createVisualShape(p.GEOM_BOX, halfExtents=dim)
		if mass != 0:
			model = p.createMultiBody(baseMass=mass, baseVisualShapeIndex=model)
	obj = Object(model=model, pos=pos, ori=ori)
	return obj

def create_sphere(radius=0.5, pos=[0,0,0], mass=0, collisions=True):
	if collisions:
		model = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
		if mass != 0:
			model = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=model)
	else:
		model = p.createVisualShape(p.GEOM_SPHERE, radius=radius)
		if mass != 0:
			model = p.createMultiBody(baseMass=mass, baseVisualShapeIndex=model)
	obj = Object(model=model, pos=pos)
	return obj