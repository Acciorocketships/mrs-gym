from mrsgym.BulletSim import *
import mrsgym
import pybullet as p
from scipy.spatial.transform import Rotation as R
import torch


class Object:

	def __init__(self, uid=None, env=None, sim=DefaultSim(), pos=[0,0,0], ori=[0,0,0], inpackage=True):
		self.env = env
		self.sim = sim
		if isinstance(uid, str):
			self.uid = mrsgym.EnvCreator.load_urdf(uid, pos=pos, ori=ori, inpackage=inpackage, sim=sim)
		else:
			self.uid = uid
			self.set_state(pos=pos, ori=ori)


	def __del__(self):
		p.removeBody(self.uid, physicsClientId=self.sim.id)


	@property
	def name(self):
		return (p.getBodyInfo(self.uid, physicsClientId=self.sim.id)[1]).decode('utf-8')


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
		p.resetBasePositionAndOrientation(self.uid, posObj=pos, ornObj=ori, physicsClientId=self.sim.id)
		p.resetBaseVelocity(self.uid, linearVelocity=vel, angularVelocity=angvel, physicsClientId=self.sim.id)


	def get_joint_info(self):
		numJoints = p.getNumJoints(self.uid, physicsClientId=self.sim.id)
		return {n: p.getJointInfo(self.uid, jointIndex=n, physicsClientId=self.sim.id) for n in range(numJoints)}


	def get_joint_state(self):
		numJoints = p.getNumJoints(self.uid, physicsClientId=self.sim.id)
		return {n: p.getJointState(self.uid, jointIndex=n, physicsClientId=self.sim.id) for n in range(numJoints)}


	def get_vel(self):
		return torch.tensor(p.getBaseVelocity(self.uid, physicsClientId=self.sim.id)[0])


	def get_angvel(self, mat=False):
		return torch.tensor(p.getBaseVelocity(self.uid, physicsClientId=self.sim.id)[1])
		

	def get_pos(self):
		return torch.tensor(p.getBasePositionAndOrientation(self.uid, physicsClientId=self.sim.id)[0]).float()


	def get_ori(self, mat=False):
		quat = torch.tensor(p.getBasePositionAndOrientation(self.uid, physicsClientId=self.sim.id)[1])
		r = R.from_quat(quat)
		if mat:
			return torch.tensor(r.as_matrix()).float()
		else:
			return torch.tensor(r.as_euler('xyz', degrees=False).copy()).float()


	def get_contact_points(self, other=None, body=False):
		if other is None: # filter by specific ID of another object
			contactObjList = p.getContactPoints(self.uid, physicsClientId=self.sim.id)
		else:
			contactObjList = p.getContactPoints(self.uid, filterBodyUniqueIdB=other, physicsClientId=self.sim.id)
		if len(contactObjList)==0:
			return {'object':[], 'pos':torch.zeros(0,3), 'normal':torch.zeros(0,3), 'distance':torch.zeros(0)}
		contacts = {}
		contacts['object'] = [self.env.object_dict.get(contactObj[2], None) for contactObj in contactObjList]
		contacts['pos'] = torch.stack([torch.tensor(contactObj[5]) for contactObj in contactObjList], dim=0) # world coordinates
		contacts['normal'] = torch.stack([contactObj[9] * torch.tensor(contactObj[7]) for contactObj in contactObjList], dim=0) # direction and magnitude of normal force
		contacts['distance'] = torch.tensor([contactObj[8] for contactObj in contactObjList])
		if body:
			R = self.get_ori(mat=True)
			pos = self.get_pos()
			contacts['pos'] = (R.T @ contacts['pos'].T - R.T @ pos).T
			contacts['normal'] = (R.T @ contacts['normal'].T).T
		return contacts


	def get_closest_points(self, other=None, body=False):
		pass


	def get_dist(self, other=None):
		pass


	def collision(self):
		return any(map(lambda dist: dist<0.04, self.get_contact_points()['distance']))


	def get_nearby_objects(self):
		pass


	def raycast(self, offset=torch.zeros(3), directions=torch.tensor([100.,0.,0.]), body=True):
		if len(directions.shape)==1:
			directions = directions.unsqueeze(0)
		if len(offset.shape)==1:
			offset = offset.expand(directions.shape[0])
		R = self.get_ori(mat=True)
		pos = self.get_pos()
		if body:
			offset = R @ offset
			directions = (R @ directions.T).T
		start = offset + pos
		start = start.expand(directions.shape[0], -1)
		end = directions + start
		rays = p.rayTestBatch(rayFromPositions=start.tolist(), rayToPositions=end.tolist(), parentObjectUniqueId=self.uid, physicsClientId=self.sim.id)
		ray_dict = {}
		ray_dict["object"] = [self.env.object_dict.get(ray[0], None) for ray in rays]
		ray_dict["pos_world"] = torch.stack([torch.tensor(ray[3]) for ray in rays])
		ray_dict["pos"] = (R.T @ ray_dict['pos_world'].T - R.T @ pos).T
		ray_dict["dist"] = (ray_dict['pos'] - offset).norm(dim=-1)
		return ray_dict


	def get_image(self, forward=torch.tensor([1.,0.,0.]), up=torch.tensor([0.,0.,1.]), offset=torch.zeros(3), body=True, fov=90., aspect=4/3, height=720):
		if not isinstance(forward, torch.Tensor):
			forward = torch.tensor(forward)
		if not isinstance(up, torch.Tensor) and up is not None:
			up = torch.tensor(up)
		if not isinstance(offset, torch.Tensor):
			offset = torch.tensor(offset)
		if body:
			R = self.get_ori(mat=True)
			forward = R @ forward
			up = R @ up
			offset = R @ offset
		pos = self.get_pos() + offset
		view_mat = p.computeViewMatrix(cameraEyePosition=pos.tolist(), cameraTargetPosition=forward.tolist(), cameraUpVector=up.tolist(), physicsClientId=self.sim.id)
		NEAR_PLANE = 0.01
		FAR_PLANE = 1000.0
		proj_mat = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=NEAR_PLANE, farVal=FAR_PLANE, physicsClientId=self.sim.id)
		img = p.getCameraImage(width=int(aspect * height), height=height, viewMatrix=view_mat, projectionMatrix=proj_mat, physicsClientId=self.sim.id)
		return img




class ControlledObject(Object):

	# Either give update funciton in init or extend class and implement update()
	def __init__(self, update_fn=None, *args, **kwargs):
		super(ControlledObject, self).__init__(*args, **kwargs)
		if update_fn is not None:
			self.update = update_fn



