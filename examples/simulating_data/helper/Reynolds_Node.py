import torch.nn as nn
import torch


class Reynolds_Node:

	def __init__(self):
		self.controller = Reynolds_Controller()


class Reynolds_Controller():

	def __init__(self):
		self.MAX_ACCEL = 1.0

	def forward(self, X):
		neighbours = X[1]
		rel_pos = torch.sum(neighbours[:,:3] * torch.norm(neighbours[:,:3], dim=1)[:,None], dim=0)
		rel_pos_inv = - torch.sum(neighbours[:,:3] / (torch.norm(neighbours[:,:3], dim=1) ** 3)[:,None], dim=0)
		rel_vel = torch.sum(neighbours[:,3:6] * torch.norm(neighbours[:,3:6], dim=1)[:,None], dim=0)
		X_out = 0.25 * (1.0 * rel_pos + 7.0 * rel_pos_inv + 1.0 * rel_vel)
		mag = torch.clamp(torch.norm(X_out), max=self.MAX_ACCEL)
		X_out = (X_out / torch.norm(X_out)) * mag
		return X_out

	def forward_batch(self, Z, A):
		# Z: batch x N x N x D x K+1
		# A: batch x N x N
		batch, N, _, D, _ = Z.shape
		neighbours = (Z[:,:,:,:,1] - Z[:,:,:,:,0]).view(batch*N*N,D)
		A = A.view(batch*N*N)
		rel_pos = torch.sum((neighbours[:,:3] * torch.norm(neighbours[:,:3], dim=1).unsqueeze(1)).view(batch*N,N,3), dim=1)
		rel_pos_inv = - torch.sum((A.unsqueeze(1) * (neighbours[:,:3] / (A-1 + torch.norm(neighbours[:,:3], dim=1) ** 3).unsqueeze(1))).view(batch*N,N,3), dim=1)
		rel_vel = torch.sum((neighbours[:,3:6] * torch.norm(neighbours[:,3:6], dim=1).unsqueeze(1)).view(batch*N,N,3), dim=1)
		X_out = 0.5 * (1.0 * rel_pos + 3.0 * rel_pos_inv + 3.0 * rel_vel)
		mag = torch.clamp(torch.norm(X_out, dim=1), max=self.MAX_ACCEL) / torch.norm(X_out, dim=1)
		X_out = X_out * mag[:,None]
		return X_out.view(batch,N,3)



