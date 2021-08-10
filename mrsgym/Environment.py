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
		self.agent_idxs = {} # dict of agent objects to their idx in self.agents
		self.data = {}
		self.debug_names = {}


	def init_agents(self, N):
		for _ in range(N):
			self.add_agent(Quadcopter(env=self, sim=self.sim))


	def set_data(self, name, val):
		self.data[name] = val


	def get_data(self, name):
		return self.data.get(name, None)


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
			uid = self.controlled[obj].uid
			del self.controlled[obj]
		elif isinstance(obj, ControlledObject):
			uid = obj.uid
			idx = self.controlled.index(obj)
			del self.controlled[idx]
		del self.object_dict[uid]


	def add_agent(self, agent):
		self.agents.append(agent)
		self.object_dict[agent.uid] = agent
		self.agent_idxs[agent] = len(self.agents)-1


	def remove_agent(self, agent):
		if isinstance(agent, int):
			uid = self.agents[obj].uid
			del self.agents[agent]
		elif isinstance(obj, Quadcopter):
			uid = obj.uid
			idx = self.agent_idxs[agent]
			del self.agents[idx]
		del self.agent_idxs[self.object_dict[uid]]
		del self.object_dict[uid]


	def get_X(self, state_fn):
		X = [totensor(state_fn(agent)) for agent in self.agents]
		X = torch.stack(X, dim=0)
		return X


	def set_actions(self, actions, behaviour='set_controls'):
		for i, agent in enumerate(self.agents):
			getattr(agent, behaviour)(actions[i,:])
		for agent in self.agents:
			agent.dynamics()


	def set_state(self, pos, ori, vel, angvel):
		for i, agent in enumerate(self.agents):
			posval = pos[i,:] if (pos is not None) else None
			orival = ori[i,:] if (ori is not None) else None
			velval = vel[i,:] if (vel is not None) else None
			angvelval = angvel[i,:] if (angvel is not None) else None
			agent.set_state(pos=posval, ori=orival, vel=velval, angvel=angvelval)


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


	def set_collisions(self, obj1, obj2, collisions=False):
		p.setCollisionFilterPair(bodyUniqueIdA=obj1.uid, bodyUniqueIdB=obj2.uid, linkIndexA=-1, linkIndexB=-1, enableCollision=(1 if collisions else 0), physicsClientId=self.sim.id)


	def draw_links(self, A):
		N = A.shape[0]
		for i in range(N):
			for j in range(i):
				name = "CommLink({},{})".format(i,j)
				if A[i,j] > 0:
					self.add_line(start=self.agents[i], end=self.agents[j], name=name, lifetime=0., colour=[1.,1.,0.])
				else:
					self.remove_debug(name)


	def get_image(self, pos, forward, up=None, fov=90., aspect=4/3, height=720):
		pos = totensor(pos)
		forward = totensor(forward)
		if up is not None:
			up = totensor(up)
		else:
			left = torch.cross(torch.tensor([0.,0.,1.]), forward)
			up = torch.cross(forward, left)
		view_mat = p.computeViewMatrix(cameraEyePosition=pos.tolist(), cameraTargetPosition=forward.tolist(), cameraUpVector=up.tolist(), physicsClientId=self.sim.id)
		NEAR_PLANE = 0.01
		FAR_PLANE = 1000.0
		proj_mat = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=NEAR_PLANE, farVal=FAR_PLANE, physicsClientId=self.sim.id)
		_, _, img, depth, mask = p.getCameraImage(width=int(aspect * height), height=height, viewMatrix=view_mat, projectionMatrix=proj_mat, physicsClientId=self.sim.id)
		img = torch.tensor(img)
		depth = FAR_PLANE * NEAR_PLANE / (FAR_PLANE - (FAR_PLANE-NEAR_PLANE)*torch.tensor(depth))
		mask = torch.tensor(mask)
		return {"img": img, "depth": depth, "mask": mask}


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
			else:
				continue
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


	def get_camera_pos(self):
		camera_params = p.getDebugVisualizerCamera(physicsClientId=self.sim.id)
		camera_forward = torch.tensor(camera_params[5])
		camera_left = -torch.tensor(camera_params[6]); camera_left /= camera_left.norm()
		camera_up = torch.tensor(camera_params[7]); camera_up /= camera_up.norm()
		R = torch.stack([camera_forward, camera_left, camera_up], dim=1)
		camera_target = torch.tensor(camera_params[-1])
		target_dist = camera_params[-2]
		camera_pos_world = camera_target - target_dist * camera_forward
		return camera_pos_world, R


	def set_camera(self, pos=None, target=torch.zeros(3), dist=None):
		if pos is None:
			camera_params = p.getDebugVisualizerCamera(physicsClientId=self.sim.id)
			camera_forward = torch.tensor(camera_params[5])
			camera_target = torch.tensor(camera_params[-1])
			if dist is None:
				target_dist = camera_params[-2]
			else:
				target_dist = dist
			pos = camera_target - target_dist * camera_forward
		if isinstance(target, Object):
			target = target.get_pos()
		pos = totensor(pos)
		target = totensor(target)
		disp = target - pos
		dist = disp.norm()
		yaw = np.arctan2(-disp[0],disp[1]) * 180/np.pi
		pitch = np.arctan2(disp[2],np.sqrt(disp[0]**2+disp[1]**2)) * 180/np.pi
		p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target.tolist(), physicsClientId=self.sim.id)


	def add_line(self, start, end, parent=None, name="line", width=1.0, lifetime=0., colour=[0.,0.,0.]):
		if isinstance(start, Object):
			start = start.get_pos()
		if isinstance(end, Object):
			end = end.get_pos()
		start = totensor(start)
		end = totensor(end)
		colour = totensor(colour)
		uid = self.debug_names.get(name, -1)
		if parent is not None:
			new_uid = p.addUserDebugLine(lineFromXYZ=start.tolist(), lineToXYZ=end.tolist(), lineColorRGB=colour.tolist(), lineWidth=width, lifeTime=lifetime, parentObjectUniqueId=parent.uid, replaceItemUniqueId=uid, physicsClientId=self.sim.id)
		else:
			new_uid = p.addUserDebugLine(lineFromXYZ=start.tolist(), lineToXYZ=end.tolist(), lineColorRGB=colour.tolist(), lineWidth=width, lifeTime=lifetime, replaceItemUniqueId=uid, physicsClientId=self.sim.id)
		self.debug_names[name] = new_uid


	def add_text(self, text, pos=None, poscam=[1,1.25,-0.95], parent=None, name="text", size=1.0, lifetime=0., colour=[0.,0.,0.]):
		# TODO: set pos to right in front of the camera if it is None
		if pos is None:
			camera_pos, R = self.get_camera_pos()
			forward = R[:,0]
			left = R[:,1]
			up = R[:,2]
			pos = camera_pos + poscam[0] * forward + poscam[1] * left + poscam[2] * up
		else:
			pos = totensor(pos)
		colour = totensor(colour)
		uid = self.debug_names.get(name, -1)
		if parent is not None:
			new_uid = p.addUserDebugText(text=text, textPosition=pos.tolist(), textColorRGB=colour.tolist(), textSize=size, lifeTime=lifetime, parentObjectUniqueId=parent.uid, replaceItemUniqueId=uid, physicsClientId=self.sim.id)
		else:
			new_uid = p.addUserDebugText(text=text, textPosition=pos.tolist(), textColorRGB=colour.tolist(), textSize=size, lifeTime=lifetime, replaceItemUniqueId=uid, physicsClientId=self.sim.id)
		self.debug_names[name] = new_uid


	def add_param(self, name, low=0., high=1., start=None):
		if start is None:
			start = low
		new_uid = p.addUserDebugParameter(paramName=name, rangeMin=low, rangeMax=high, startValue=start, physicsClientId=self.sim.id)
		self.debug_names[name] = new_uid


	def read_param(self, name):
		uid = self.debug_names[name]
		value = p.readUserDebugParameter(itemUniqueId=uid, physicsClientId=self.sim.id)
		return value


	def remove_debug(self, name):
		if name in self.debug_names:
			uid = self.debug_names[name]
			p.removeUserDebugItem(itemUniqueId=uid, physicsClientId=self.sim.id)
			del self.debug_names[name]
			return True
		return False


	def set_colour(self, obj, colour=None):
		if colour is not None:
			colour = totensor(colour)
			p.setDebugObjectColor(objectUniqueId=obj.uid, linkIndex=-1, objectDebugColorRGB=colour.tolist(), physicsClientId=self.sim.id)
		else: # if colour is none, it is reset to the default
			p.setDebugObjectColor(objectUniqueId=obj.uid, linkIndex=-1, physicsClientId=self.sim.id)


	def record(self, filename="demo.mp4"):
		p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4, fileName=filename, physicsClientId=self.sim.id)

