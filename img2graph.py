# from graphviz import view
import numpy as np
from torch_geometric.data import Data
import torch
from skimage.util import view_as_blocks

def loader(dataset):
	# handle datasets 
	# will output dim=3, even for grayscale (0.5 = [0.5,0.5,0.5])
	if dataset == 'bcv':
		# f = open
		# change to blob to for organized file management


# load image and mask from file (will change to accomodate multiple files)
img = np.load("bcv/img.npy")
# img looks like this: [ [0.5, 0.1, 0.9, 1], [.........], [............]]
label = np.load("bcv/label.npy")

n = img.shape[0] # 256

# [0.        , 0.        , 0.        , ..., 0.08953857, 0.18035889, 0.2607422 ],
# [0.        , 0.        , 0.        , ..., 0.04266357, 0.08770752, 0.17474365],
# [0.        , 0.        , 0.        , ..., 0.04364014, 0.04266357, 0.07977295],
# patches are square

patch_dim = 1 # for pixel-level use patch_dim = 1, otherwise take a factor of 256 

if patch_dim > 1:
	img = view_as_blocks(img,(patch_dim,patch_dim))
	label = view_as_blocks(label,(patch_dim,patch_dim))

# make node features and adjacency 
# node features are just values of pixels, 1d for bcv, handling for 1d currently
# [ [0.5, 0.1, 0.9, 1], [.........], [............]]

x = torch.Tensor(img.reshape(-1,1)) # [num_nodes, num_node_features]
# [[0.5] ,[0.1], [0.9] [1]]
y = torch.Tensor(label.reshape(-1,1)) # same
# make edges connect with neighbours at distance of 2

k = 2
alpha, beta = 1, 1
edges = [[],[]]
edge_weights = []
num_nodes = n**2

def similarity(i,j):
	# needs to be vectorized/numpy usage
	return -(alpha*abs(x[i] - x[j]) + beta*abs(y[i] - y[j]))

def addedge(a, b):
	if a < 0 or a > num_nodes or b < 0 or b > num_nodes:
		return
	edges[0].append(a)
	edges[1].append(b)
	edge_weights.append([-(alpha*abs(x[i] - x[j]) + beta*abs(y[i] - y[j]))])
	# edges.append([a,b])

# def addweight():

for i in range(n**2):
	for j in range(1,k):
		# todo: vectorize these calls
		addedge(i, i-j)
		# edge_weights.append()
		addedge(i, i-n*j)
		addedge(i, i+j)
		addedge(i, i+n*j)
