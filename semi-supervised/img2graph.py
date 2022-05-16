import random
import numpy as np
from torch_geometric.data import Data
import torch_geometric.utils
import torch
from skimage.util import view_as_blocks
from config import config
from tqdm import tqdm

device = torch.device('cuda:0'if torch.cuda.is_available() else "cpu")

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
	x = torch.tensor(img.reshape(-1,num_node_features), dtype = torch.float).to(device) # [num_nodes, num_node_features]

	# [[0.5] ,[0.1], [0.9] [1]]
	y = None
	if label is not None:
		y = torch.tensor(label.reshape(-1,1), dtype = torch.float).to(device) # same
	# make edges connect with neighbours at distance of 2

	num_neighbors = config['num_neighbors']
	alpha, beta = config['alpha'], config['beta']
	edges = [[],[]]
	edgelist = {}
	edge_weights = []
	num_nodes = n**2

	def addedge(a, b):
		if a < 0 or a >= num_nodes or b < 0 or b >= num_nodes or (a,b) in edgelist:
			return
		edgelist[a].append(b)
		edges[0].append(a)
		edges[1].append(b)
		if label is not None:
			edge_weights.append([-(alpha*abs((x[a] - x[b]).mean()) + beta*abs(y[a] - y[b]))])
		else:
			edge_weights.append([-(alpha*abs((x[a] - x[b]).mean()) )])
		# print(f"Edge b/w {a} & {b}")
		# edges.append([a,b])

	for i in tqdm(range(n**2)):
		for j in range(1,num_neighbors+1):
			# todo: vectorize these calls ?
			if (i%n != 0):
				addedge(i, i-j) #left
			addedge(i, i-n*j) #top
			if (i%n != n-1):
				addedge(i, i+j) #right
			addedge(i, i+n*j) #bottom
			for l in range(1,j+1): #diagonal
				if ((i-l)%n < n-l):
					addedge(i, i-n*j -l) #top left
					addedge(i, i+n*j -l) #bottom left
				if ((i+l)%n > l-1):
					addedge(i, i-n*j +l) #top right
					addedge(i, i+n*j +l) #bottom right
				if ((i+j)%n > j-1):
					addedge(i, i+j-n*l)
					addedge(i, i+j+n*l)
				if ((i-j)%n < n-j):
					addedge(i, i-j-n*l)
					addedge(i, i-j+n*l)
		# print(f"Edges of {i} = {edges[0]}")

	edges = torch.tensor(edges, dtype = torch.long).to(device)
	edge_weights = torch.tensor(edge_weights, dtype = torch.float).to(device)
	# if y is not None:
	return Data(x=x, y=y, edge_index=edges, edge_attr=edge_weights)
	# else return Data(x=x, edge_index=edges, edge_attr=edge_weights)

def support_graph(support_images, support_labels):
	num_images = len(support_images)
	n = support_images[0].shape[0]
	num_neighbors = 1 #number of neighbours (between graphs)	
	nn = n**2;
	alpha, beta = config['alpha'], config['beta']
	dataset = config['dataset']
	num_node_features = 3 if dataset in ['coco'] else 1
	edges = [[],[]]
	edgelist = []
	edge_weights = []
	x = torch.cat([torch.tensor(img.reshape(-1,num_node_features), dtype = torch.float) for img in support_images]).to(device)
	y = None
	if support_labels is not None:
		y = torch.cat([torch.tensor(label.reshape(-1,1), dtype = torch.float) for label in support_labels]).to(device) # same
	num_node_features = 3 if dataset in ['coco'] else 1
	# x = torch.tensor(img.reshape(-1,num_node_features), dtype = torch.float) # [num_nodes, num_node_features]

	def addedge(a, b):
		if a < 0 or a >= nn or b < 0 or b >= nn or (a,b) in edgelist:
			return
		edgelist.append((a,b))
		edges[0].append(a)
		edges[1].append(b)
		if support_labels is not None:
			edge_weights.append([-(alpha*abs((x[a] - x[b]).mean()) + beta*abs(y[a] - y[b]))])
		else:
			edge_weights.append([-(alpha*abs((x[a] - x[b]).mean()) )])
		# print(f"Edge b/w {a} & {b}")

	for k in range(num_images):
		for i in tqdm(range(nn)):
			for j in range(1,num_neighbors+1):
				# Intra-graph connections
				if (i%n != 0):
					addedge(k*nn+i, i-j+k*nn) #left
				addedge(i+k*nn, i-n*j+k*nn) #top
				if (i%n != n-1):
					addedge(i+k*nn, i+j+k*nn) #right
				addedge(i+k*nn, i+n*j+k*nn) #bottom
				for l in range(1,j+1): #diagonal
					if ((i-l)%n < n-l):
						addedge(i+k*nn, i-n*j -l+k*nn) #top left
						addedge(i+k*nn, i+n*j -l+k*nn) #bottom left
					if ((i+l)%n > l-1):
						addedge(i+k*nn, i-n*j +l+k*nn) #top right
						addedge(i+k*nn, i+n*j +l+k*nn) #bottom right
					if ((i+j)%n > j-1):
						addedge(i+k*nn, i+j-n*l+k*nn)
						addedge(i+k*nn, i+j+n*l+k*nn)
					if ((i-j)%n < n-j):
						addedge(i+k*nn, i-j-n*l+k*nn)
						addedge(i+k*nn, i-j+n*l+k*nn)
				# Inter-graph connections
				if (k != 0):
					addedge(k*nn +i, i + (k-1)*nn) # directly below
					if ((i-1)%n < n-1):
						addedge(k*nn+i, i-1+(k-1)*nn) #left
						addedge(k*nn+i, i-1+(k-1)*nn-n) # top left
						addedge(k*nn+i, i-1+(k-1)*nn+n) # bottom left
					addedge(i+k*nn, i-n+(k-1)*nn) #top
					if ((i+1)%n > 0):
						addedge(i+k*nn, i+1+(k-1)*nn) #right
						addedge(i+k*nn, i+1+(k-1)*nn -n) # top right
						addedge(i+k*nn, i+1+(k-1)*nn +n) # bottom right
					addedge(i+k*nn, i+n+(k-1)*nn) #bottom
				if (k != num_images-1):
					addedge(k*nn +i, i + (k+1)*nn) # directly above
					if ((i-1)%n < n-1):
						addedge(k*nn+i, i-1+(k+1)*nn) #left
						addedge(k*nn+i, i-1+(k+1)*nn-n) # top left
						addedge(k*nn+i, i-1+(k+1)*nn+n) # bottom left
					addedge(i+k*nn, i-n+(k+1)*nn) #top
					if ((i+1)%n > 0):
						addedge(i+k*nn, i+1+(k+1)*nn) #right
						addedge(i+k*nn, i+1+(k+1)*nn -n) # top right
						addedge(i+k*nn, i+1+(k+1)*nn +n) # bottom right
					addedge(i+k*nn, i+n+(k+1)*nn) #bottom
	edges = torch.tensor(edges, dtype = torch.long).to(device)
	edge_weights = torch.tensor(edge_weights, dtype = torch.float).to(device)
	# if y is not None:
	return Data(x=x, y=y, edge_index=edges, edge_attr=edge_weights)	

def support_graph_matrix(labelled_images, labels, unlabeled_images, query_images):
	num_label = len(labels) # N*K
	M = len(unlabeled_images) 
	Q = len(query_images)
	n = labelled_images[0].shape[0]
	num_neighbors_inter = 1 #number of neighbours (between graphs)	
	num_neighbors = 2 # max distance of neighbors
	nn = n**2;
	alpha, beta = config['alpha'], config['beta']
	dataset = config['dataset']
	num_node_features = 3 if dataset in ['COCO'] else 1
	edges = [[],[]]
	edgelist = []
	edge_weights = []
	edge_func = config['edge_weight_function']
	if edge_func == 'mean-abs':
		f = lambda x1,x2: torch.mean(torch.abs(x1-x2), axis=2).reshape(-1)
	elif edge_func == 'euclidean':
		f = lambda x1,x2: torch.linalg.norm(x1-x2, axis=2)
	# x_labelled = torch.cat([torch.tensor(img.reshape(-1,num_node_features), dtype = torch.float) for img in labelled_images]).to(device)
	# x_unlabeled = torch.cat([torch.tensor(img.reshape(-1,num_node_features),
	# 	dtype = torch.float) for img in unlabeled_images]).to(device)
	x_labelled = torch.tensor([])
	x_unlabeled = torch.tensor([])
	x_task = torch.tensor([])
	# y_unlabeled = None
	# y_labeled = torch.cat([torch.tensor(label.reshape(-1,1), dtype = torch.float) for label in labels]).to(device) # same
	y_labelled = torch.tensor([])
	# y_unlabelled = torch.tensor([])
	y_task = torch.tensor([])
	num_node_features = 3 if dataset in ['x'] else 1

	index = np.arange(num_label+M+Q) # order in which images appear in graph
	random.shuffle(index) # for i in index: if i<num_label then labelled else unlabeled 
	# [2,4,1,5,3,0] these values correspond to positions in the final array/combined graph
	
	
	
	def intra_graph_connections(a, img_num, img_index):
		indices = np.arange(nn).reshape(a.shape[:-1]) + img_num*nn
		lab = torch.zeros(a.shape[:-1])
		edges = [[],[]]
		edge_weights = []
		if img_index<num_label:
			lab = labels[img_index] 
		for i in range(1,num_neighbors+1):
			# right
			edge_weights += (alpha*f(a[:, :-i], a[:, i:]) + beta*torch.abs(lab[:, :-i] - lab[:, i:]).reshape(-1)).tolist()
			edges[0] += indices[:, :-i].reshape(-1).tolist()
			edges[1] += indices[:, i:].reshape(-1).tolist()
			# left
			edge_weights += (alpha*f(a[:, i:], a[:, :-i]) + beta*torch.abs(lab[:, i:] - lab[:, :-i]).reshape(-1)).tolist()
			edges[0] += indices[:, i:].reshape(-1).tolist()
			edges[1] += indices[:, :-i].reshape(-1).tolist()
			# down
			edge_weights += (alpha*f(a[:-i, :], a[i:, :]) + beta*torch.abs(lab[:-i, :] - lab[i:, :]).reshape(-1)).tolist()
			edges[0] += indices[:-i, :].reshape(-1).tolist()
			edges[1] += indices[i:, :].reshape(-1).tolist()
			# up
			edge_weights += (alpha*f(a[i:, :], a[:-i, :]) + beta*torch.abs(lab[i:, :] - lab[:-i, :]).reshape(-1)).tolist()
			edges[0] += indices[i:, :].reshape(-1).tolist()
			edges[1] += indices[:-i, :].reshape(-1).tolist()
			# top right (diagonally)
			edge_weights += (alpha*f(a[i:, :-i], a[:-i, i:]) + beta*torch.abs(lab[i:, :-i] - lab[:-i, i:]).reshape(-1)).tolist()
			edges[0] += indices[i:, :-i].reshape(-1).tolist()
			edges[1] += indices[:-i, i:].reshape(-1).tolist()
			# bottom left (diagonally)
			edge_weights += (alpha*f(a[i:, :-i], a[:-i, i:]) + beta*torch.abs(lab[i:, :-i] - lab[:-i, i:]).reshape(-1)).tolist()
			edges[0] += indices[:-i, i:].reshape(-1).tolist()
			edges[1] += indices[i:, :-i].reshape(-1).tolist()
			# top left (diagonally)
			edge_weights += (alpha*f(a[i:, i:], a[:-i, :-i]) + beta*torch.abs(lab[i:, i:] - lab[:-i, :-i]).reshape(-1)).tolist()
			edges[0] += indices[i:, i:].reshape(-1).tolist()
			edges[1] += indices[:-i, :-i].reshape(-1).tolist()
			# bottom right (diagonally)
			edge_weights += (alpha*f(a[i:, i:], a[:-i, :-i]) + beta*torch.abs(lab[i:, i:] - lab[:-i, :-i]).reshape(-1)).tolist()
			edges[0] += indices[:-i, :-i].reshape(-1).tolist()
			edges[1] += indices[i:, i:].reshape(-1).tolist()
		
		# remaining connections 
		# 2 up and 1 right
		edge_weights += (alpha*f(a[2:, :-1], a[:-2, 1:]) + beta*torch.abs(lab[2:, :-1] - lab[:-2, 1:]).reshape(-1)).tolist()
		edges[0] += indices[2:, :-1].reshape(-1).tolist()
		edges[1] += indices[:-2, 1:].reshape(-1).tolist()
		# 2 down and 1 left
		edge_weights += (alpha*f(a[2:, :-1], a[:-2, 1:]) + beta*torch.abs(lab[2:, :-1] - lab[:-2, 1:]).reshape(-1)).tolist()
		edges[0] += indices[:-2, 1:].reshape(-1).tolist()
		edges[1] += indices[2:, :-1].reshape(-1).tolist()
		# 2 up and 1 left
		edge_weights += (alpha*f(a[2:, 1:], a[:-2, :-1]) + beta*torch.abs(lab[2:, 1:] - lab[:-2, :-1]).reshape(-1)).tolist()
		edges[0] += indices[2:, 1:].reshape(-1).tolist()
		edges[1] += indices[:-2, :-1].reshape(-1).tolist()
		# 2 down and 1 right
		edge_weights += (alpha*f(a[2:, 1:], a[:-2, :-1]) + beta*torch.abs(lab[2:, 1:] - lab[:-2, :-1]).reshape(-1)).tolist()
		edges[0] += indices[:-2, :-1].reshape(-1).tolist()
		edges[1] += indices[2:, 1:].reshape(-1).tolist()
		# 1 up and 2 right
		edge_weights += (alpha*f(a[1:, :-2], a[:-1, 2:]) + beta*torch.abs(lab[1:, :-2] - lab[:-1, 2:]).reshape(-1)).tolist()
		edges[0] += indices[1:, :-2].reshape(-1).tolist()
		edges[1] += indices[:-1, 2:].reshape(-1).tolist()
		# 1 down and 2 left
		edge_weights += (alpha*f(a[1:, :-2], a[:-1, 2:]) + beta*torch.abs(lab[1:, :-2] - lab[:-1, 2:]).reshape(-1)).tolist()
		edges[1] += indices[1:, :-2].reshape(-1).tolist()
		edges[0] += indices[:-1, 2:].reshape(-1).tolist()
		# 1 up and 2 left
		edge_weights += (alpha*f(a[1:, 2:], a[:-1, :-2]) + beta*torch.abs(lab[1:, 2:] - lab[:-1, :-2]).reshape(-1)).tolist()
		edges[0] += indices[1:, 2:].reshape(-1).tolist()
		edges[1] += indices[:-1, :-2].reshape(-1).tolist()
		# 1 down and 2 right	
		edge_weights += (alpha*f(a[1:, 2:], a[:-1, :-2]) + beta*torch.abs(lab[1:, 2:] - lab[:-1, :-2]).reshape(-1)).tolist()
		edges[1] += indices[1:, 2:].reshape(-1).tolist()
		edges[0] += indices[:-1, :-2].reshape(-1).tolist()

		return edge_weights,edges
	
	def inter_graph_connections(a, b, img_num, img_index, prev_img_index):
		indices = np.arange(nn).reshape(a.shape[:-1]) + img_num*nn
		lab = torch.zeros(a.shape[:-1])
		prev_img_lab = torch.zeros(b.shape[:-1])
		edges = [[],[]]
		edge_weights = []
		if img_index<num_label and prev_img_index<num_label:
			lab = labels[img_index] 
			prev_img_lab = labels[prev_img_index]
		# directly above and directly below
		edge_weights += (alpha*f(a, b) + beta*torch.abs(lab - prev_img_lab).reshape(-1)).tolist()
		edges[0] += indices.reshape(-1).tolist()
		edges[1] += (indices-nn).reshape(-1).tolist()
		edge_weights += (alpha*f(a, b) + beta*torch.abs(lab - prev_img_lab).reshape(-1)).tolist()
		edges[1] += indices.reshape(-1).tolist()
		edges[0] += (indices-nn).reshape(-1).tolist()
		# top-right and bottom-left
		edge_weights += (alpha*f(a[:, :-1], b[:, 1:]) + beta*torch.abs(lab[:, :-1] - prev_img_lab[:, 1:]).reshape(-1)).tolist()
		edges[0] += indices[:, :-1].reshape(-1).tolist()
		edges[1] += (indices-nn)[:, 1:].reshape(-1).tolist()
		edge_weights += (alpha*f(a[:, :-1], b[:, 1:]) + beta*torch.abs(lab[:, :-1] - prev_img_lab[:, 1:]).reshape(-1)).tolist()
		edges[1] += indices[:, :-1].reshape(-1).tolist()
		edges[0] += (indices-nn)[:, 1:].reshape(-1).tolist()
		# top-left and bottom right
		edge_weights += (alpha*f(a[:, 1:], b[:, :-1]) + beta*torch.abs(lab[:, 1:] - prev_img_lab[:, :-1]).reshape(-1)).tolist()
		edges[0] += indices[:, 1:].reshape(-1).tolist()
		edges[1] += (indices-nn)[:, :-1].reshape(-1).tolist()
		edge_weights += (alpha*f(a[:, 1:], b[:, :-1]) + beta*torch.abs(lab[:, 1:] - prev_img_lab[:, :-1]).reshape(-1)).tolist()
		edges[1] += indices[:, 1:].reshape(-1).tolist()
		edges[0] += (indices-nn)[:, :-1].reshape(-1).tolist()
		# top-up and bottom down
		edge_weights += (alpha*f(a[1:, :], b[:-1, :]) + beta*torch.abs(lab[1:, :] - prev_img_lab[:-1, :]).reshape(-1)).tolist()
		edges[0] += indices[1:, :].reshape(-1).tolist()
		edges[1] += (indices-nn)[:-1, :].reshape(-1).tolist()
		edge_weights += (alpha*f(a[1:, :], b[:-1, :]) + beta*torch.abs(lab[1:, :] - prev_img_lab[:-1, :]).reshape(-1)).tolist()
		edges[1] += indices[1:, :].reshape(-1).tolist()
		edges[0] += (indices-nn)[:-1, :].reshape(-1).tolist()
		# top down and bottom up
		edge_weights += (alpha*f(a[:-1, :], b[1:, :]) + beta*torch.abs(lab[:-1, :] - prev_img_lab[1:, :]).reshape(-1)).tolist()
		edges[0] += indices[:-1, :].reshape(-1).tolist()
		edges[1] += (indices-nn)[1:, :].reshape(-1).tolist()
		edge_weights += (alpha*f(a[:-1, :], b[1:, :]) + beta*torch.abs(lab[:-1, :] - prev_img_lab[1:, :]).reshape(-1)).tolist()
		edges[0] += indices[:-1, :].reshape(-1).tolist()
		edges[1] += (indices-nn)[1:, :].reshape(-1).tolist()
		# top: down right and down: up left
		edge_weights += (alpha*f(a[:-1, :-1], b[1:, 1:]) + beta*torch.abs(lab[:-1, :-1] - prev_img_lab[1:, 1:]).reshape(-1)).tolist()
		edges[0] += indices[:-1, :-1].reshape(-1).tolist()
		edges[1] += (indices-nn)[1:, 1:].reshape(-1).tolist()
		edge_weights += (alpha*f(a[:-1, :-1], b[1:, 1:]) + beta*torch.abs(lab[:-1, :-1] - prev_img_lab[1:, 1:]).reshape(-1)).tolist()
		edges[1] += indices[:-1, :-1].reshape(-1).tolist()
		edges[0] += (indices-nn)[1:, 1:].reshape(-1).tolist()
		# top: up right and down: bottom left
		edge_weights += (alpha*f(a[1:, :-1], b[:-1, 1:]) + beta*torch.abs(lab[1:, :-1] - prev_img_lab[:-1, 1:]).reshape(-1)).tolist()
		edges[0] += indices[1:, :-1].reshape(-1).tolist()
		edges[1] += (indices-nn)[:-1, 1:].reshape(-1).tolist()
		edge_weights += (alpha*f(a[1:, :-1], b[:-1, 1:]) + beta*torch.abs(lab[1:, :-1] - prev_img_lab[:-1, 1:]).reshape(-1)).tolist()
		edges[1] += indices[1:, :-1].reshape(-1).tolist()
		edges[0] += (indices-nn)[:-1, 1:].reshape(-1).tolist()
		# top: up left and bottom: down right
		edge_weights += (alpha*f(a[1:, 1:], b[:-1, :-1]) + beta*torch.abs(lab[1:, 1:] - prev_img_lab[:-1, :-1]).reshape(-1)).tolist()
		edges[0] += indices[1:, 1:].reshape(-1).tolist()
		edges[1] += (indices-nn)[:-1, :-1].reshape(-1).tolist()
		edge_weights += (alpha*f(a[1:, 1:], b[:-1, :-1]) + beta*torch.abs(lab[1:, 1:] - prev_img_lab[:-1, :-1]).reshape(-1)).tolist()
		edges[1] += indices[1:, 1:].reshape(-1).tolist()
		edges[0] += (indices-nn)[:-1, :-1].reshape(-1).tolist()
		# top: down left and bottom: up right
		edge_weights += (alpha*f(a[:-1, 1:], b[1:, :-1]) + beta*torch.abs(lab[:-1, 1:] - prev_img_lab[1:, :-1]).reshape(-1)).tolist()
		edges[0] += indices[:-1, 1:].reshape(-1).tolist()
		edges[1] += (indices-nn)[1:, :-1].reshape(-1).tolist()
		edge_weights += (alpha*f(a[:-1, 1:], b[1:, :-1]) + beta*torch.abs(lab[:-1, 1:] - prev_img_lab[1:, :-1]).reshape(-1)).tolist()
		edges[1] += indices[:-1, 1:].reshape(-1).tolist()
		edges[0] += (indices-nn)[1:, :-1].reshape(-1).tolist()
		return edge_weights,edges

	edges_labelled = [[],[]]
	edge_weights_labelled = []	
	edges_unlabeled = [[],[]]
	edge_weights_unlabeled = []
	edges_combined = [[],[]]
	edge_weights_combined = []
	labelled_num, unlabeled_num = 0,0
	prev_lab_index, prev_unlab_index = 0,0
	query_index = []
	labelled_index = []
	for i in range(len(index)):
		if index[i]<num_label: #labeled 
			labelled_index.append(i)
			x_labelled = torch.cat((x_labelled, labelled_images[index[i]].reshape(-1,num_node_features).clone().detach()))
			x_task = torch.cat((x_task, labelled_images[index[i]].reshape(-1,num_node_features).clone().detach()))
			y_labelled = torch.cat((y_labelled, labels[index[i]].reshape(-1).clone().detach()))
			y_task = torch.cat((y_task, labels[index[i]].reshape(-1).clone().detach()))
			ew,e = intra_graph_connections(labelled_images[index[i]],i,index[i])
			edge_weights_combined += ew
			edge_weights_labelled += ew
			edges_combined[0] += e[0]
			edges_combined[1] += e[1]
			if (i - labelled_num >0):
				edges_labelled[0] += (np.array(e[0])-(i - labelled_num)*nn).tolist()
				edges_labelled[1] += (np.array(e[1])-(i - labelled_num)*nn).tolist()
			else:
				edges_labelled[0] += (np.array(e[0])).tolist()
				edges_labelled[1] += (np.array(e[1])).tolist()
			if labelled_num != 0:
				ew,e = inter_graph_connections(labelled_images[index[i]],labelled_images[prev_lab_index],labelled_num,index[i],prev_lab_index)
				edge_weights_labelled += ew
				edges_labelled[0] += e[0]
				edges_labelled[1] += e[1]
			if i!=0:
				if index[i-1] < num_label:
					ew,e = inter_graph_connections(labelled_images[index[i]],labelled_images[prev_lab_index],i,index[i],prev_lab_index)
				elif index[i-1] < num_label +M:
					ew,e = inter_graph_connections(labelled_images[index[i]],unlabeled_images[prev_unlab_index-num_label],i,index[i],prev_unlab_index)
				else:
					ew,e = inter_graph_connections(labelled_images[index[i]],query_images[index[query_index[-1]]-num_label-M],i,index[i],index[query_index[-1]])
				edge_weights_combined += ew
				edges_combined[0] += e[0]
				edges_combined[1] += e[1]
			labelled_num += 1
			prev_lab_index = index[i]
		elif index[i]< (M+num_label) : #unlabeled
			x_unlabeled = torch.cat((x_unlabeled, unlabeled_images[index[i]-num_label].reshape(-1,num_node_features).clone().detach()))
			x_task = torch.cat((x_task, unlabeled_images[index[i]-num_label].reshape(-1,num_node_features).clone().detach()))
			y_task = torch.cat((y_task, -torch.ones(unlabeled_images[index[i]-num_label].reshape(-1).shape)))
			ew,e = intra_graph_connections(unlabeled_images[index[i]-num_label],i,index[i])
			edge_weights_combined += ew
			edge_weights_unlabeled += ew
			edges_combined[0] += e[0]
			edges_combined[1] += e[1]
			if (i - unlabeled_num >0):
				edges_unlabeled[0] += (np.array(e[0])-(i - unlabeled_num)*nn).tolist()
				edges_unlabeled[1] += (np.array(e[1])-(i - unlabeled_num)*nn).tolist()
			else:
				edges_unlabeled[0] += (np.array(e[0])).tolist()
				edges_unlabeled[1] += (np.array(e[1])).tolist()
			if unlabeled_num != 0:
				ew,e = inter_graph_connections(unlabeled_images[index[i]-num_label],unlabeled_images[prev_unlab_index-num_label],unlabeled_num,index[i],prev_unlab_index)
				edge_weights_unlabeled += ew
				edges_unlabeled[0] += e[0]
				edges_unlabeled[1] += e[1]
			if i!=0:
				if index[i-1] < num_label:
					ew,e = inter_graph_connections(unlabeled_images[index[i]-num_label],labelled_images[prev_lab_index],i,index[i],prev_lab_index)
				elif index[i-1] < num_label +M:
					ew,e = inter_graph_connections(unlabeled_images[index[i]-num_label],unlabeled_images[prev_unlab_index-num_label],i,index[i],prev_unlab_index)
				else:
					ew,e = inter_graph_connections(unlabeled_images[index[i]-num_label],query_images[index[query_index[-1]]-num_label-M],i,index[i],index[query_index[-1]])
				edge_weights_combined += ew
				edges_combined[0] += e[0]
				edges_combined[1] += e[1]
			unlabeled_num += 1
			prev_unlab_index = index[i]
		else: #query
			x_task = torch.cat((x_task, query_images[index[i]-num_label-M].reshape(-1,num_node_features).clone().detach()))
			y_task = torch.cat((y_task, -torch.ones(query_images[index[i]-num_label-M].reshape(-1).shape)))
			query_index.append(i)
			ew,e = intra_graph_connections(query_images[index[i]-num_label-M],i,index[i])
			edge_weights_combined += ew
			edges_combined[0] += e[0]
			edges_combined[1] += e[1]
			if i!=0:
				if index[i-1] < num_label:
					ew,e = inter_graph_connections(query_images[index[i]-num_label-M],labelled_images[prev_lab_index],i,index[i],prev_lab_index)
				elif index[i-1] < num_label +M:
					ew,e = inter_graph_connections(query_images[index[i]-num_label-M],unlabeled_images[prev_unlab_index-num_label],i,index[i],prev_unlab_index)
				else:
					ew,e = inter_graph_connections(query_images[index[i]-num_label-M],query_images[index[query_index[-1]]-num_label-M],i,index[i],index[query_index[-1]])
				edge_weights_combined += ew
				edges_combined[0] += e[0]
				edges_combined[1] += e[1]
	edges_lab = torch.tensor(edges_labelled, dtype=torch.long)
	edges_unlab = torch.tensor(edges_unlabeled, dtype=torch.long)
	edges_task = torch.tensor(edges_combined, dtype=torch.long)
	ew_lab = torch.tensor(edge_weights_labelled).reshape(-1,1)
	ew_unlab = torch.tensor(edge_weights_unlabeled).reshape(-1,1)
	ew_tasks = torch.tensor(edge_weights_combined).reshape(-1,1)

	sup_graph = Data(x=x_labelled, y=y_labelled, edge_index=edges_lab, edge_attr=ew_lab).to(device)
	unsup_graph = Data(x=x_unlabeled, edge_index=edges_unlab, edge_attr=ew_unlab).to(device)
	task_graph = Data(x=x_task, y=y_task, edge_index=edges_task, edge_attr=ew_tasks).to(device)
	return query_index, labelled_index, sup_graph, unsup_graph, task_graph 
# def visualise_graph(data):
# 	import matplotlib.pyplot as plt
# 	import igraph as ig
# 	adj = torch_geometric.utils.to_dense_adj(data['edge_index'], edge_attr=data['edge_attr'])
# 	g = ig.Graph.Weighted_Adjacency(adj.tolist()[0])
# 	l = g.layout('lgl')
# 	ig.plot(g, layout=l)
# 	# plt.show()

# if __name__ == "__main__":
# 	img = np.load("sample_bcv/img.npy")
# 	label = np.load("sample_bcv/label.npy")
# 	data = img2graph(img,label)
# 	visualise_graph(data)

