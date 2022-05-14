from config import config
import torch
import numpy as np
from tqdm import tqdm
from model import GNN
import dataloader
import json
import loss

device = torch.device('cuda:0'if torch.cuda.is_available() else "cpu")

dataset = config['dataset']
img_dim = 3 # for COCO/any color datasets
batch_size = config['batch_size']
n_episodes = config['n_episodes']
dataset_type = 'general'
if dataset in ['BCV', 'CT_ORG', 'DECATHLON']:
	img_dim = 1
	dataset_type = 'medical'

emb_dim = config['embedding_dim']
num_classes = config['classes'][dataset]
GNN_Encoder = GNN(img_dim, emb_dim)
GNN_Decoder = GNN(num_classes)

CL_fn = loss.Node2NodeSupConLoss()
unsup_weight = config['unsup_weight']
loss_fn = loss.QueryClassificationLoss()
encoder_optimizer = torch.optim.Adam(GNN_Encoder.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
decoder_optimizer = torch.optim.Adam(GNN_Decoder.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

if dataset_type == 'general':
	lists_path = config['lists_path'] + '/coco/'
	train_sup_list = lists_path + 'train_sup_list.json'
	train_unsup_list = lists_path + 'train_unsup_list.json'
	train_data = dataloader.GeneralDataLoader(train_sup_list, train_unsup_list)
	trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True, 
												pin_memory = True, drop_last=True) 


# elif dataset_type == 'medical':


GNN_Encoder.train()
GNN_Decoder.train()

# iterate over dataloader and get a batch
for episode in range(n_episodes):
	print(f'Episode {episode}')
	
	for i, (q_graph, sup_graph, unsup_graph, task_graph, subcls_lists) in enumerate(trainloader):
		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad())
		# all tensors are on device already
		# phase 1: CL
		# make augmentations from images (are they needed if we are sampling)
		# pass all through GNN_Encoder and get embeddings
		# q_embs = GNN_Encoder(q_graph)
		sup_embs = GNN_Encoder(sup_graph)
		unsup_embs = GNN_Encoder(unsup_graph)
		# once embeddings are here, make projection heads from a simple MLP?
		# ?
		# call the loss function on task graph augs (query??) and obtain contrastive loss
		sup_CL, unsup_CL = CL_fn(sup_embs, unsup_embs)
		contrastive_loss = (1-unsup_weight)*sup_CL + unsup_weight*unsup_CL
		# backprop through optimizer
		contrastive_loss.backward()
		encoder_optimizer.step()
		# phase 2: node classification
		# GNN_Encoder.no_grad()
		# get embeddings for task graphs and query graphs (all batches/parallel)
		# append one-hot label to all node's embeddings in task graph and
		# [1/k,1/k,....,1/k] (uniform distribution/k-simplex) to query graph
		with torch.no_grad():
			task_embs = GNN_Encoder(task_graph)
		task_embs = GNN_Decoder(task_embs)
		loss = loss_fn(task_embs)
		loss.backward()
		decoder_optimizer.step()

	if i%10 == 0:
		print(f'Episode {i} complete, CL:{contrastive_loss.item()}\t(S:{sup_CL.item()},\tU:{unsup_CL.item()})')
		print(f'Classification Loss:{loss.item()}')