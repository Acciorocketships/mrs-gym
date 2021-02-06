import torch
import gym

class MRSAnalytics:

	def __init__(self, data):
		self.data = data
		self.X = self.data.get_episodes()["X"]
		self.num_episodes = self.X.shape[0]
		self.episode_length = self.X.shape[1]
		self.N = self.X.shape[2]


	def vel_leader_alignment_avg(self):
		return torch.mean(self.vel_leader_alignment())


	# Output size: num_episodes x episode_length x N
	def vel_leader_alignment(self):
		vel = self.velocity()
		N = vel.shape[2]
		veldiff = (vel - vel[:,:,0,:][:,:,None,:])[:,:,1:,:]
		velmag = veldiff.norm(dim=3)
		return velmag


	# The magnitude of the velocity
	# Output size: 1
	def vel_mag_avg(self):
		return self.vel_mag().mean()
		
	# The magnitude of the velocity
	# Output size: num_episodes x episode_length x N
	def vel_mag(self):
		return self.velocity().norm(dim=3)

	# The average variance in velocity over all time steps
	# Ouput size: 1
	def vel_stddev_avg(self):
		return torch.mean(self.vel_stddev())

	# The variance in velocity as a function of time for each episode
	# Output size: num_episodes x episode_length
	def vel_stddev(self):
		vel = self.velocity()
		N = vel.shape[2]
		vel = vel.view(self.num_episodes*self.episode_length,N,3)
		vel_avg = torch.mean(vel, dim=1, keepdim=True)
		vel_diff = vel - vel_avg
		covariance = torch.bmm(vel_diff.permute(0,2,1), vel_diff)
		stddev = torch.sqrt(covariance.det()).view(self.num_episodes,self.episode_length)
		return stddev

	# The average separation from the closest neighbour over all time steps
	# Output size: 1
	def separation_avg(self):
		return torch.mean(self.separation())

	# The separation of each agent from its closest neighbour as a function of time for each episode
	# Output size: num_episodes x episode_length x N
	def separation(self):
		pos = self.position()
		pos = pos.view(self.num_episodes*self.episode_length,self.N,3)
		posi = pos.unsqueeze(1).expand(-1,pos.shape[1],-1,-1)
		posj = pos.unsqueeze(2).expand(-1,-1,pos.shape[1],-1)
		codisp = posi-posj # num_episodes*episode_length x N x N x 3
		codist = codisp.norm(dim=3) # num_episodes*episode_length x N x N
		codist[codist==0] = float('inf')
		separation = codist.min(dim=2)[0] # min returns 
		separation = separation.view(self.num_episodes, self.episode_length, self.N)
		return separation


	# The average diameter sphere that contains all agents over all time steps
	# Output size: 1
	def cohesion_avg(self):
		return torch.mean(self.cohesion())

	# The diameter of the sphere that contains all agents as a function of time for each episode
	# Output size: num_episodes x episode_length
	def cohesion(self, exclude_leader=False):
		pos = self.position()
		if exclude_leader:
			pos = pos[:,:,1:,:]
		pos = pos.view(self.num_episodes*self.episode_length, -1, 3)
		posi = pos.unsqueeze(1).expand(-1,pos.shape[1],-1,-1)
		posj = pos.unsqueeze(2).expand(-1,-1,pos.shape[1],-1)
		codisp = posi-posj # num_episodes*episode_length x N x N x 3
		codist = codisp.norm(dim=3) # num_episodes*episode_length x N x N
		bounding_diameter = codist.max(dim=2)[0].max(dim=1)[0]
		bounding_diameter = bounding_diameter.view(self.num_episodes, self.episode_length)
		return bounding_diameter

	def dist_to_leader(self):
		pos = self.position()
		pos = pos.view(self.num_episodes*self.episode_length, self.N, 3)
		swarm_pos = pos[:,1:,:]
		leader_pos = pos[:,0,:]
		disp = torch.mean(swarm_pos, dim=1) - leader_pos
		dist = disp.norm(dim=1).view(self.num_episodes, self.episode_length)
		return dist

	# The velocity of each agent as a function of time for each episode
	# Output size: num_episodes x episode_length x N x 3
	def velocity(self):
		return self.X[:,:,:,3:]

	# The position of each agent as a function of time for each episode
	# Output size: num_episodes x episode_length x N x 3
	def position(self):
		return self.X[:,:,:,:3]


