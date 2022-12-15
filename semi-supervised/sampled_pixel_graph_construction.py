import numpy as np
from torch_geometric.data import Data
import torch_geometric.utils
import torch
from skimage.util import view_as_blocks
from config import config
from tqdm import tqdm

rng = np.random.default_rng()
device = torch.device('cuda:0'if torch.cuda.is_available() else "cpu")

def support_graph_matrix(labelled_images, labels, unlabeled_images, query_images):
	# support, task and unlabeled graph construction using only sampled pixels

	# few-shot setting
	num_label = len(labels) # N*K
	M = len(unlabeled_images) 
	Q = len(query_images)

	# image-related variables
	n = labelled_images[0].shape[0]
	nn = n**2;

	# Edge related variables
	alpha, beta = config['alpha'], config['beta']
	dataset = config['dataset']
	num_node_features = 3 if dataset in ['COCO'] else 1
	edge_func = config['edge_weight_function']
	if edge_func == 'mean-abs':
		f = lambda x1,x2: torch.mean(torch.abs(x1-x2), axis=2).reshape(-1)
	elif edge_func == 'euclidean':
		f = lambda x1,x2: torch.linalg.norm(x1-x2, axis=2)

	# Support, task and unlabeled graph tensors
	x_labelled = torch.tensor([])
	x_unlabeled = torch.tensor([])
	x_task = torch.tensor([])
	y_labelled = torch.tensor([])
	y_unlabelled = torch.tensor([])
	y_task = torch.tensor([])

	# sampling fg pixels from labeled images 
	fgs_chosen_in_imgs = []
	max_fg_per_img = config['max_fg_per_img']
	fg_indices_in_imgs = [np.where(x == 1) for x in labels]
	num_fg_in_imgs = [len(x) for x in fg_indices_in_imgs]
	# sampling bg pixels (fg + bg) from labeled images
	bgs_chosen_in_imgs = []
	total_pix_per_img = config['total_pixel_per_image']
	num_bg_reqd = [(total_pix_per_img - len(x)) for x in fgs_chosen_in_imgs]
	bg_indices_in_imgs = [np.where(x == 0) for x in labels]	
	num_bg_in_imgs = [len(x) for x in bg_indices_in_imgs]
	for i in range(num_label):
		num_fgs = num_fg_in_imgs[i]
		if num_fgs>max_fg_per_img:
			fg_sampled_idx = rng.integers(0, num_fgs, (max_fg_per_img))
			fgs_chosen_in_imgs.append(fg_indices_in_imgs[i][fg_sampled_idx])
		else:
			fgs_chosen_in_imgs.append(fg_indices_in_imgs[i])
		num_bgs = num_bg_in_imgs[i]
		bg_sampled_idx = rng.integers(0, num_bgs, num_bg_reqd[i])
		bgs_chosen_in_imgs.append(bg_indices_in_imgs[i][bg_sampled_idx])
	
	
	
	