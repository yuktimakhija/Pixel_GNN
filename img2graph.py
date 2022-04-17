# from graphviz import view
import numpy as np
from torch_geometric.data import Data
import torch_geometric.utils
import torch
from skimage.util import view_as_blocks
from config import config
from tqdm import tqdm
# def loader(dataset):
# 	# handle datasets 
# 	# will output dim=3, even for grayscale (0.5 = [0.5,0.5,0.5])
# 	if dataset == 'bcv':
# 		# f = open
# 		# change to blob to for organized file management


# load image and mask from file (will change to accomodate multiple files)
# img = np.load("bcv/img.npy")
# img looks like this: [ [0.5, 0.1, 0.9, 1], [.........], [............]]
# label = np.load("bcv/label.npy")

# config = json.load(open("config.py"))

def img2graph(img, label=None):
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
	# 1D node features 			vs   3D node features
	# [ [0.5, 0.1, 0.9, 1], 	|	[[ [0.1,0.5,0.3], [0.2,0.6,0.7],...]
	#   [................],		|	 [ [...], [...], ...]
	#   [................] ]	|	 [ [...], [...], ...] ]
	dataset = config['dataset']
	# num_node_features = 1
	num_node_features = 3 if dataset in ['coco'] else 1
	x = torch.tensor(img.reshape(-1,num_node_features), dtype = torch.float) # [num_nodes, num_node_features]

	# [[0.5] ,[0.1], [0.9] [1]]
	y = None
	if label is not None:
		y = torch.tensor(label.reshape(-1,1), dtype = torch.float) # same
	# make edges connect with neighbours at distance of 2

	k = 2
	alpha, beta = 1, 1
	edges = [[],[]]
	edge_weights = []
	num_nodes = n**2

	def similarity(i,j):
		# needs to be vectorized/numpy usage ??
		return -(alpha*abs(x[i] - x[j]) + beta*abs(y[i] - y[j]))

	def addedge(a, b):
		if a < 0 or a >= num_nodes or b < 0 or b >= num_nodes:
			return
		edges[0].append(a)
		edges[1].append(b)
		if label is not None:
			edge_weights.append([-(alpha*abs((x[a] - x[b]).mean()) + beta*abs(y[a] - y[b]))])
		else:
			edge_weights.append([-(alpha*abs((x[a] - x[b]).mean()) )])
		# edges.append([a,b])
		

	# def addweight():

	for i in tqdm(range(n**2)):
		for j in range(1,k):
			# todo: vectorize these calls ?
			addedge(i, i-j) #left
			addedge(i, i-n*j) #top
			addedge(i, i+j) #right
			addedge(i, i+n*j) #bottom
			for l in range(1,j+1): #diagonal
				addedge(i, i-n*j -l) #top left
				addedge(i, i-n*j +l) #top right
				addedge(i, i+n*j -l) #bottom left
				addedge(i, i+n*j +l) #bottom right

	edges = torch.tensor(edges, dtype = torch.long)
	edge_weights = torch.tensor(edge_weights, dtype = torch.float)
	# if y is not None:
	return Data(x=x, y=y, edge_index=edges, edge_attr=edge_weights)
	# else return Data(x=x, edge_index=edges, edge_attr=edge_weights)

def visualise_graph(data):
	import matplotlib.pyplot as plt
	import igraph as ig
	adj = torch_geometric.utils.to_dense_adj(data['edge_index'], edge_attr=data['edge_attr'])
	g = ig.Graph.Weighted_Adjacency(adj.tolist()[0])
	l = g.layout('lgl')
	ig.plot(g, layout=l)
	# plt.show()

if __name__ == "__main__":
	img = np.load("sample_bcv/img.npy")
	label = np.load("sample_bcv/label.npy")
	data = img2graph(img,label)
	visualise_graph(data)