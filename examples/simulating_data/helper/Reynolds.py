import torch
import torch.nn as nn
from helper.Reynolds_Node import *
import itertools

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Reynolds(nn.Module):

	def __init__(self, N, D, K=1, OUT_DIM=3):
		super().__init__()
		self.N = N
		self.D = D
		self.K = K
		self.rel_state_size = 6
		self.OUT_DIM = OUT_DIM
		self.network = Reynolds_Node()
		self.init_data()
		self.forward = self.forward_batch

	def init_data(self):
		self.data = [self.init_node_data() for _ in range(self.N)]


	def init_node_data(self):
		neighbour_set = {"Y%d" % i: torch.zeros(0, self.D, device=device) for i in range(1,self.K+1)}
		neighbour_agg = {"y%d" % i: torch.zeros(1, self.D, device=device) for i in range(self.K)}
		return merge_dicts(neighbour_set, neighbour_agg)


	def forward_single(self, As, Xs):
		# Xs = [X(t), X(t-1), ..., X(t-K)]
		# As = [A(t), A(t-1), ..., A(t-K)] (A(t-(K)) is not used. X(t-K) is dispersed with A(t-(K-1)))
		As = As.to(device)
		Xs = Xs.to(device)
		self.init_data()
		self.step(None, Xs[:,:,-1], 0)
		for k in range(1,self.K+1): # from 1 to K
			#actions = self.step(As[:,:,-k-1], Xs[:,:,-k-1], k)
			actions = self.step(As[:,:,-k], Xs[:,:,-k], k) # -k instead of -k-1 so the transform is calculated with the last time step's data (because the received data is one time step old)
		return actions


	def step(self, A, X, up_to_K=None):
		if A is not None:
			A = A.to(device)
		X = X.to(device)
		K = self.K if (up_to_K is None) else up_to_K
		for k in range(K+1):
			for i in range(self.N):
				# Base Case
				if k == 0:
					self.data[i]["y0"] = torch.cat([torch.zeros(self.rel_state_size, device=device), X[i,self.rel_state_size:]])
					continue
				# Sum y^(k-1) of neighbours and add rij
				neighbour_idx = torch.nonzero(A[i,:]).view(-1)
				Ni = neighbour_idx.shape[0]
				Yk = torch.zeros((Ni, self.D), device=device)
				for idx, j in enumerate(neighbour_idx):
					ysum = self.data[j]["y%d"%(k-1)]
					yij = ysum + self.rij(X[i,:],X[j,:]) # relative position
					Yk[idx,:] = yij
				self.data[i]["Y%d"%k] = Yk
				self.data[i]["y%d"%k] = sum(Yk)
		if self.K == K:
			# Local Computation
			Xout = torch.zeros((self.N,self.OUT_DIM), device=device)
			for i in range(self.N):
				# TODO: apply rotation here
				Xout[i,:] = self.network.controller.forward([self.data[i]["y0"].view(1,X.shape[1])] + [self.data[i]["Y%d" % j] for j in range(1,K+1)])
			return Xout


	def rij(self,xi,xj):
		r = torch.zeros(xi.shape, device=device)
		r[:self.rel_state_size] = xj[:self.rel_state_size] - xi[:self.rel_state_size]
		return r


	def forward_batch(self, As, Xs):
		# Xs = batch x N x D x K+1
		batch, N, D, K = Xs.shape; K-=1
		As = (torch.ones(N, N, device=device) - torch.eye(N, device=device)).unsqueeze(0).unsqueeze(3).repeat(batch,1,1,K+1)
		# As = As.to(device)
		Xs = Xs.to(device)
		# Calculate weighted average Y
		Y = {} # batch x N x D. aggregated k-hop neighbourhood of data from time step t-k
		Y[1] = Xs[:,:,:,1] # starts at t-1 because we don't want to aggregate the last hop
		A = As[:,:,:,1]
		Lagg = self.laplacian(A)
		Aagg = A
		for k in range(2,self.K+1):
			rel = torch.bmm(Lagg, Xs[:,:,:self.rel_state_size,k])
			nonrel = torch.bmm(Aagg, Xs[:,:,self.rel_state_size:,k])
			Y[k] = torch.cat([rel, nonrel], dim=2)
			if k != self.K: # no need to calculate A for the last step, it's unused
				A = As[:,:,:,k]
				Lagg = torch.bmm(self.laplacian(A),Lagg)
				Aagg = torch.bmm(A,Aagg)
		# Calculate controller input Z
		Z = torch.zeros((batch, N, N, D, self.K+1), device=device)
		Z[:,:,:,self.rel_state_size:,0] = torch.diag_embed(Xs[:,:,self.rel_state_size:,0].permute(0,2,1), dim1=1, dim2=2)
		A = As[:,:,:,0] #self.calc_A(As[:,:,:,0]) # maybe don't apply attention at the last step?
		for k in range(1,self.K+1):
			Z[:,:,:,:,k] = (A[:,:,:,None] * Y[k][:,:,None,:]).permute(0,2,1,3)
			Z[:,:,:,:self.rel_state_size,k] -= A[:,:,:,None] * Xs[:,:,None,:self.rel_state_size,1]
		# Apply the controller
		actions = self.network.controller.forward_batch(Z, A)
		actions[torch.isnan(actions)] = 0
		return actions


	def laplacian(self, A):
		degree = torch.sum(A, dim=2)[:,:,None] * torch.eye(A.shape[-1], device=device)[None,:,:]
		laplacian = A - degree
		return laplacian


def merge_dicts(*dicts):
	merged = dicts[0]
	for d in dicts[1:]:
		merged.update(d)
	return merged
