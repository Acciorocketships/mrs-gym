from mrsgym.BulletSim import *
from mrsgym.Quadcopter import *
from mrsgym.Object import *
from mrsgym.Util import *
import pybullet as p
import torch

class Environment:

	def __init__(self, sim=DefaultSim()):
		self.sim = sim
		self.agents = []
		self.objects = []
		self.controlled = []
		self.object_dict = {} # dict of uid to all types of objects in env (agents, objects, controlled)


	def init_agents(self, N):
		for _ in range(N):
			self.add_agent(Quadcopter(env=self, sim=self.sim))


	def add_object(self, obj):
		self.objects.append(obj)
		self.object_dict[obj.uid] = obj


	def remove_object(self, obj):
		if isinstance(obj, int):
			uid = self.objects[obj].uid
			del self.objects[obj]
		elif isinstance(obj, Object):
			uid = obj.uid
			idx = self.objects.index(obj)
			del self.objects[idx]
		del self.object_dict[uid]


	def add_controlled(self, obj):
		self.controlled.append(obj)
		self.object_dict[obj.uid] = obj


	def remove_controlled(self, obj):
		if isinstance(obj, int):
			uid = self.objects[obj].uid
			del self.controlled[obj]
		elif isinstance(obj, ControlledObject):
			uid = obj.uid
			idx = self.controlled.index(obj)
			del self.controlled[idx]
		del self.object_dict[uid]


	def add_agent(self, agent):
		self.agents.append(agent)
		self.object_dict[agent.uid] = agent


	def remove_agent(self, agent):
		if isinstance(agent, int):
			uid = self.objects[obj].uid
			del self.agents[agent]
		elif isinstance(obj, Quadcopter):
			uid = obj.uid
			idx = self.agents.index(agent)
			del self.agents[idx]
		del self.object_dict[uid]


	def get_X(self, state_fn):
		X = list(map(state_fn, self.agents))
		X = torch.stack(X, dim=0)
		return X


	def set_actions(self, actions, behaviour='set_controls'):
		for i, agent in enumerate(self.agents):
			getattr(agent, behaviour)(actions[i,:])


	def set_state(self, pos, ori, vel, angvel):
		for i, agent in enumerate(self.agents):
			agent.set_state(pos=pos[i,:], ori=ori[i,:], vel=vel[i,:], angvel=angvel[i,:])


	def update_controlled(self):
		for controlled_obj in self.controlled:
			controlled_obj.update(self)


	def get_pos(self):
		return torch.stack([agent.get_pos() for agent in self.agents], dim=0)


	def get_vel(self):
		return torch.stack([agent.get_vel() for agent in self.agents], dim=0)


	def get_ori(self):
		return torch.stack([agent.get_ori() for agent in self.agents], dim=0)


	def get_angvel(self):
		return torch.stack([agent.get_angvel() for agent in self.agents], dim=0)


	def set_collisions(obj1, obj2, collisions=False):
		p.setCollisionFilterPair(bodyUniqueIdA=obj1.uid, bodyUniqueIdB=obj2.uid, linkIndexA=-1, linkIndexB=-1, enableCollision=(1 if collisions else 0), physicsClientId=self.sim.id)


	def get_image(self, pos, forward, up=None, fov=90., aspect=4/3, height=720):
		if not isinstance(pos, torch.Tensor):
			pos = torch.tensor(pos)
		if not isinstance(forward, torch.Tensor):
			forward = torch.tensor(forward)
		if not isinstance(up, torch.Tensor) and up is not None:
			up = torch.tensor(up)
		if up is None:
			left = torch.cross(torch.tensor([0.,0.,1.]), forward)
			up = torch.cross(forward, left)
		view_mat = p.computeViewMatrix(cameraEyePosition=pos.tolist(), cameraTargetPosition=forward.tolist(), cameraUpVector=up.tolist(), physicsClientId=self.sim.id)
		NEAR_PLANE = 0.01
		FAR_PLANE = 1000.0
		proj_mat = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=NEAR_PLANE, farVal=FAR_PLANE, physicsClientId=self.sim.id)
		img = p.getCameraImage(width=int(aspect * height), height=height, viewMatrix=view_mat, projectionMatrix=proj_mat, physicsClientId=self.sim.id)
		return img


	def get_keyboard_events(self):
		events = p.getKeyboardEvents(physicsClientId=self.sim.id)
		keys = {}
		for keycode, valcode in events.items():
			if valcode == 3:
				val = 1
			elif valcode == 1:
				val = 0
			elif valcode == 4:
				val = -1
			key = Key(keycode)
			keys[key] = val
		return keys


	def get_mouse_events(self):
		events = p.getMouseEvents(physicsClientId=self.sim.id)
		for event in events:
			if event[4] == 3:
				_, x, y, _, _ = event
				camera_params = p.getDebugVisualizerCamera(physicsClientId=self.sim.id)
				width = camera_params[0]
				height = camera_params[1]
				aspect = width/height
				pos_view = torch.tensor([1.0, aspect*(1.0-(x/width)*2), 1.0-(y/height)*2])
				camera_forward = torch.tensor(camera_params[5])
				camera_left = -torch.tensor(camera_params[6]); camera_left /= camera_left.norm()
				camera_up = torch.tensor(camera_params[7]); camera_up /= camera_up.norm()
				R = torch.stack([camera_forward, camera_left, camera_up], dim=1)
				camera_target = torch.tensor(camera_params[-1])
				target_dist = camera_params[-2]
				camera_pos_world = camera_target - target_dist * camera_forward
				pos_world = R @ pos_view + camera_pos_world
				vec = pos_world - camera_pos_world; vec = vec / vec.norm()
				ray = p.rayTest(rayFromPosition=camera_pos_world.tolist(), rayToPosition=(100.0*vec+camera_pos_world).tolist(), physicsClientId=self.sim.id)
				hit_pos = torch.tensor(ray[0][3])
				results = {}
				results['camera_pos'] = camera_pos_world
				results['target_pos'] = torch.tensor(ray[0][3])
				results['target_obj'] = self.object_dict.get(ray[0][0], None)
				results['target_normal'] = torch.tensor(ray[0][4])
				return results



