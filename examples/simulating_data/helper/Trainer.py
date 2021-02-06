from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import atexit
from helper.Util import *

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

class Trainer:

	def __init__(self, model=None, optimiser=None, tensorboard_path=None, data_path=None, model_path=None, K=0):
		self.history = {"A": deque([]), "X": deque([]), "done": deque([]), "expert": deque([]), "context": deque([])}
		self.iter = 0
		self.model = model
		self.optimiser = optimiser
		self.tensorboard_path = tensorboard_path
		self.data_path = data_path
		self.model_path = model_path
		self.K = K
		self.valid_idxs = []
		self.start_idxs = []
		self.sample_weights = torch.tensor([])
		self.sample_idxs = []
		self.steps_since_start = -1
		self.loss = None
		if self.tensorboard_path is not None:
			self.tensorboard = SummaryWriter(self.tensorboard_path)
		if self.model_path is not None:
			load_model(self.model, self.model_path)
		if self.data_path is not None:
			self.load_trainer()


	def save_trainer_onexit(self, path=None):
		atexit.register(self.save_trainer, path=path)


	def save_trainer(self, path=None):
		# Create state dict
		if len(self.history["done"]) >= 1:
			self.history["done"][-1] = True
		data = self.__dict__.copy()
		if data["model"] is not None:
			save_model(self.model, self.model_path)
		if data["optimiser"] is not None:
			data["optimiser_state"] = self.optimiser.state_dict()
		for name in ["model", "optimiser", "K", "tensorboard_path", "data_path", "model_path", "valid_idxs", "start_idxs", "steps_since_start", "tensorboard", "loss", "loss_fn", "tensorboard_fn"]:
			if name in data:
				del data[name]
		# Save file
		if path is None:
			path = self.data_path
		with open(path, 'wb') as fp:
			torch.save(data, fp)
		print("Saved Trainer " + path)
		return data


	def load_trainer(self, path=None):
		if path is None:
			path = self.data_path
		try:
			fp = open(path,'rb')
		except:
			print("No Trainer to Load")
			return False
		data = torch.load(fp, map_location=device)
		self.load_trainer_dict(data)
		self.update_valid_idxs()
		print("Loaded Trainer " + path)
		return True


	def load_trainer_dict(self, data):
		if "optimiser_state" in data:
			self.optimiser.load_state_dict(data["optimiser_state"])
		for name in ["model", "optimiser", "K", "tensorboard_path", "data_path", "model_path", "valid_idxs", "start_idxs", "steps_since_start", "tensorboard", "loss", "loss_fn", "tensorboard_fn"]:
			if name in data:
				del data[name]
		for name, val in data.items():
			setattr(self, name, val)


	def set_state(self, A, X, done=False, expert=None, context={}):
		# Update data
		self.history["done"].append(done)
		self.history["A"].append(A.to(device))
		self.history["X"].append(X.to(device))
		self.history["context"].append(context)
		if expert is not None:
			self.history["expert"].append(expert.to(device))
		else:
			self.history["expert"].append(expert)
		# Update valid idxs
		if done:
			self.steps_since_start = -1 # steps since first episode step
		else:
			self.steps_since_start += 1
		idx = len(self.history["X"]) - 1
		if self.steps_since_start >= self.K:
			self.valid_idxs.append(idx)
		if self.steps_since_start == 0:
			self.start_idxs.append(idx)


	def get_state(self, idx=-1, K=None):
		if K is None:
			K = self.K
		idx = self.valid_idxs[idx]
		# Compile data
		batch = {}
		X = torch.stack([self.history["X"][idx-k] for k in range(K+1)], dim=2)
		batch["X"] = X
		A = torch.stack([self.history["A"][idx-k] for k in range(K+1)], dim=2)
		batch["A"] = A
		expert = torch.stack([self.history["expert"][idx-k] for k in range(K+1)], dim=2)
		batch["expert"] = expert
		context = [self.history["context"][idx-k] for k in range(K+1)]
		batch["context"] = context
		return batch


	def update_valid_idxs(self):
		self.valid_idxs = []
		self.start_idxs = []
		for idx in range(len(self.history["done"])):
			if self.history["done"][idx]:
				self.steps_since_start = -1
			else:
				self.steps_since_start += 1
			if self.steps_since_start >= self.K:
				self.valid_idxs.append(idx)
			if self.steps_since_start == 0:
				self.start_idxs.append(idx)


	def train(self, loss):
		loss.backward()
		self.optimiser.step()
		self.optimiser.zero_grad()
		# self.iter = list(self.optimiser.state_dict()["state"].values())[0]['step']
		self.iter += 1


	def get_batch(self, batch_size=16, data_split=1.0, weighting=False):
		# data_split = 0.9 for train set, data_split = -0.1 for test set
		# Init weighting
		if weighting and self.sample_weights.shape[0] != len(self.valid_idxs):
			self.sample_weights = torch.ones(len(self.valid_idxs))
		# Get indices
		size = min(batch_size, len(self.valid_idxs)-1)
		idxs = self.get_random_idxs(num=size, data_split=data_split, weighting=weighting)
		self.sample_idxs = idxs
		# Compile data
		batch = {}
		X = torch.stack([torch.stack([self.history["X"][idx-k] for k in range(self.K+1)], dim=2) for idx in idxs], dim=0)
		batch["X"] = X
		A = torch.stack([torch.stack([self.history["A"][idx-k] for k in range(self.K+1)], dim=2) for idx in idxs], dim=0)
		batch["A"] = A
		expert = torch.stack([self.history["expert"][idx] for idx in idxs], dim=0)
		batch["expert"] = expert
		context = [self.history["context"][idx] for idx in idxs]
		batch["context"] = context
		return batch


	def get_random_idxs(self, num=1, data_split=1.0, weighting=False):
		N = len(self.valid_idxs)
		N_set = int(N*abs(data_split))
		if weighting:
			weights = self.sample_weights[:N_set]
		else:
			weights = torch.ones(N_set)
		if data_split > 0:
			idx_idxs = torch.multinomial(weights,num,replacement=False)
		else:
			idx_idxs = (N-N_set) + torch.multinomial(weights,num,replacement=False)
		return [self.valid_idxs[i] for i in idx_idxs]


	def get_episodes(self):
		episodes = [self.get_episode(idx) for idx in self.start_idxs]
		Xs = [episode["X"] for episode in episodes]
		As = [episode["A"] for episode in episodes]
		experts = [episode["expert"] for episode in episodes]
		contexts = [episode["context"] for episode in episodes]
		max_length = max(map(lambda episode: episode["X"].shape[0], episodes))
		N = Xs[0].shape[1]
		D = Xs[0].shape[2]
		Xs = [torch.cat([X.float(), torch.full((max_length-X.shape[0],N,D), np.nan, device=device)], dim=0) for X in Xs]
		As = [torch.cat([A.float(), torch.full((max_length-A.shape[0],N,N), np.nan, device=device)], dim=0) for A in As]
		if len(experts[0].shape)==3:
			OUT_DIM = experts[0].shape[2]
			experts = [torch.cat([expert.float(), torch.full((max_length-expert.shape[0],N,OUT_DIM), np.nan, device=device)], dim=0) for expert in experts]
		elif len(experts[0].shape)==2:
			experts = [torch.cat([expert.float(), torch.full((max_length-expert.shape[0],N), np.nan, device=device)], dim=0) for expert in experts]
		contexts = [np.concatenate([context, np.array([{} for _ in range(max_length-context.shape[0])])], axis=0) for context in contexts]
		Xs = torch.stack(Xs, dim=0)
		As = torch.stack(As, dim=0)
		experts = torch.stack(experts, dim=0)
		contexts = np.array(contexts)
		data = {"X": Xs, "A": As, "expert": experts, "context": contexts}
		return data # num_episodes x episode_length x N x D


	def get_episode(self, idx=None):
		# if idx is given, returns the rest of that episode
		# if idx is None, returns an entire random episode
		if idx is None:
			N = len(self.start_idxs)
			idx = self.start_idxs[torch.randint(N,size=())]
		A = []
		X = []
		expert = []
		context = []
		k = 0
		while idx+k != len(self.history["done"]) and (not self.history["done"][idx+k] or k==0):
			A.append(self.history["A"][idx+k])
			X.append(self.history["X"][idx+k])
			expert.append(self.history["expert"][idx+k])
			context.append(self.history["context"][idx+k])
			k += 1
		A = torch.stack(A, dim=0)
		X = torch.stack(X, dim=0)
		expert = torch.stack(expert, dim=0)
		context = np.array(context)
		return {"A": A, "X": X, "expert": expert, "context": context}


	def get_X(self):
		X = torch.tensor(self.history["X"])
		X = X[self.valid_idxs]
		return X


	def update_tensorboard(self, value, t=None, datatype=None):
		if not hasattr(self, "tensorboard"):
			print("No tensorboard object")
			return
		if t is None:
			t = self.iter
		if datatype is None:
			for name, val in value.items():
				self.update_tensorboard(val, t=t, datatype=name)
		else:
			if datatype is 'scalar': # given as a dict {name: value}
				for name, val in value.items():
					if isinstance(val, dict):
						self.tensorboard.add_scalars(name, val, global_step=t) # adds to the same plot
					else:
						self.tensorboard.add_scalar(name, val, global_step=t)
			elif datatype is 'graph': # given as a tuple (net, inputs)
				if t == 0:
					net = value[0]
					inputs = value[1]
					self.tensorboard.add_graph(net, inputs)
			elif datatype is 'embedding': # given as a tuple (features, labels)
				features = value[0] # NxD, each row is the feature vector of a data point
				labels = value[1]  # N, vector of int labels
				self.tensorboard.add_embedding(features, metadata=labels, global_step=t)
			elif datatype is 'hyperparameter': # given as a tuple (hparam_dict, metric_dict)
				hparam_dict = value[0] # {hyperparameter name: hyperparameter value}
				metric_dict = value[1] # {metric name: metric value}
				self.tensorboard.add_hparams(hparam_dict, metric_dict)
			elif datatype is 'histogram':
				for name, val in value.items():
					self.tensorboard.add_histogram(name, val, global_step=t)



	def show_tensorboard(self, openbrowser=True):
		from tensorboard import program
		tb = program.TensorBoard()
		tb.configure(argv=[None, '--logdir', self.tensorboard.log_dir])
		url = tb.launch()
		if openbrowser:
			import webbrowser
			webbrowser.open(url)


def stack_k(X, tdim=0, K=0):
	orig_shape = list(X.shape)
	shape = list(X.shape)
	T = shape[tdim]
	shape[tdim] = T - (K)
	batch_size = int(torch.prod(torch.tensor(shape[:tdim])).item())
	shape = [batch_size] + shape[tdim:]
	shape.append(K+1)
	Y = torch.zeros(shape)
	new_X_shape = [batch_size] + orig_shape[tdim:]
	X = X.view(*new_X_shape)
	for i_batch in range(batch_size):
		for t in range(K,T):
			for k in range(K+1):
				Y[i_batch,t-K,:,:,k] = X[i_batch,t-k,:,:]
	end_shape = orig_shape[:tdim] + shape[1:]
	return Y.view(*end_shape)
