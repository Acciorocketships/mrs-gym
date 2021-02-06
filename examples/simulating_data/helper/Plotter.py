import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

colours = ['b','g','r','c','m','y',(1.0,0.5,0.0),(0.6,0.4,0.3)]

def plot_time_distribution(data=[], labels=[], idxs=None, xlabel="", ylabel="", title="", newfigure=True, ignorenan=True):
	if len(data)==0 or data[0].shape[0]==0:
		raise Exception("data is empty")
	if not isinstance(data, list):
		data = [data]
	if not isinstance(labels, list):
		labels = [labels]
	if idxs is None:
		idxs = torch.arange(0,data[0].shape[-1])
	if newfigure:
		new_figure(size=(10,5))
	lines = []
	for i in range(len(data)):
		# Generate Distribution Stats
		if ignorenan:
			mean = mean_ignorenan(data[i])
			median = median_ignorenan(data[i])
			std = std_ignorenan(data[i])
		else:
			mean = torch.mean(data[i], dim=0)
			median = torch.median(data[i], dim=0)[0]
			std = torch.std(data[i], dim=0)
		# To Numpy
		median = median.cpu().detach().numpy()
		mean = mean.cpu().detach().numpy()
		std = std.cpu().detach().numpy()
		# Plot
		line, = plt.plot(idxs[:median.shape[0]], median, color=colours[i], linewidth=3, linestyle="-", label=labels[i] + " median")
		plt.fill_between(idxs[:mean.shape[0]], mean-std, mean+std, color=colours[i], alpha=0.2, label=labels[i] + " mean +/- 1 std dev")
		lines.append(line)
	if len(title) > 0:
		plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(loc='lower left', handles=lines, labels=labels)
	plt.tight_layout()


def plot_time_series(data=[], labels=[], idxs=None, xlabel="", ylabel="", title="", newfigure=True):
	if len(data)==0 or data[0].shape[0]==0:
		raise Exception("data is empty")
	if not isinstance(data, list):
		data = [data]
	for i in range(len(data)):
		data[i] = data[i].cpu().detach().numpy()
	if not isinstance(labels, list):
		labels = [labels]
	if idxs is None:
		idxs = torch.arange(0,data[0].shape[-1])
	if newfigure:
		new_figure()
	for i in range(len(data)):
		label = labels[i] if i < len(labels) else None
		h = plt.plot(idxs[:data[i].shape[0]], data[i], color=colours[i%len(labels)], linewidth=3, linestyle="-", label=label)[0]
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(loc='upper right')


def plot_distribution(data=[], labels=[], xlabel="", ylabel="", title="", newfigure=True):
	if len(data)==0 or data[0].shape[0]==0:
		raise Exception("data is empty")
	if isinstance(data, list):
		if isinstance(data[0], torch.Tensor):
			data = torch.stack(data, dim=1)
			data = data.cpu().detach().numpy()
		elif isinstance(data[0], np.ndarray):
			data = np.stack(data, axis=1)
	if not isinstance(labels, list):
		labels = [labels]
	if newfigure:
		new_figure()
	ax = plt.gca()
	ax.violinplot(data, showmedians=True, showextrema=False, points=10000)
	ax.set_xticks([i for i in range(1,len(labels)+1)])
	ax.set_xticklabels(labels)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)


def subplot(num, i, projection=None):
	width = num // np.sqrt(num)
	height = num / width
	plt.subplot(height, width, i+1, projection=projection)


# data: episode_length x N x 3
def plot_spacetime(data=torch.zeros(0,0,3), xlabel="", ylabel="", title="", newfigure=True):
	if new_figure:
		fig = new_figure()
		fig.add_subplot(111, projection='3d')
	for i in range(data.shape[1]):
		plt.plot(data[:,i,0],data[:,i,1],data[:,i,2], linestyle='--', color=colours[i])
		plt.plot([data[0,i,0]],[data[0,i,1]],[data[0,i,2]], 'x', color=colours[i])
		plt.plot([data[-1,i,0]],[data[-1,i,1]],[data[-1,i,2]], 'o', color=colours[i])
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)



def new_figure(size=(8,8)):
	return plt.figure(figsize=size)


def show_plots():
	plt.show()


def median_ignorenan(x):
	return torch.tensor(np.nanmedian(x.detach().cpu().numpy(),axis=0))

def mean_ignorenan(x):
	x = x.clone()
	n = (~torch.isnan(x)).sum(dim=0)
	x[torch.isnan(x)] = 0
	mean = x.sum(dim=0) / n[None,:]
	return mean[0,:]

def std_ignorenan(x):
	x = x.clone()
	n = (~torch.isnan(x)).sum(dim=0)
	mean = mean_ignorenan(x)
	x[torch.isnan(x)] = mean.expand(x.shape[0],-1)[torch.isnan(x)]
	var = torch.sqrt(x.std(dim=0) ** 2 * (x.shape[0]-1) / (n-1)[None,:])
	return var[0,:]