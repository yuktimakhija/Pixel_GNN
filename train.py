from config import config
import torch
import numpy as np
from tqdm import tqdm
from model import GNN
import dataloader
import json

dataset = config['dataset']
img_dim = 3 # for COCO/any color datasets
batch_size = config['batch_size']
dataset_type = 'general'
if dataset in ['BCV', "CT_ORG", "DECATHLON"]:
	img_dim = 1
	dataset_type = 'medical'

model = GNN(img_dim)

if dataset_type == 'general':
	lists_path = config['lists_path'] + '/coco/'
	train_list = json.load(open(lists_path + 'train_list.json'))
	train_data = dataloader.GeneralDataLoader(train_list)
	trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True, 
												pin_memory = True, drop_last=True) 


elif dataset_type == 'medical':


model.train()

# iterate over dataloader and get a batch
for episode in range(config['n_episodes']):
	print(f'Episode {episode}')

	for 
	# phase 1: CL
	# make augmentations from images (are they needed if we are sampling)
	# pass all through model and get embeddings
	# once embeddings are here, make projection heads from a simple MLP?
	# call the loss function on task graph augs (query??) and obtain contrastive loss
	# backprop through optimizer
	
	# phase 2: node classification
	# get embeddings for task graphs and query graphs (all batches/parallel)
	# append one-hot label to all node's embeddings in task graph and
	# [1/k,1/k,....,1/k] (uniform distribution/k-simplex) to query graph
	# join the query graph to task graph pixel-by-pixel?