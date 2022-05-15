from config import config
import torch
import numpy as np
from tqdm import tqdm
from model import GNN_Encoder, GNN_Decoder
import dataloader
import loss
import time
import GCL.augmentors as A
from GCL.models import DualBranchContrast
import GCL.losses as L
from torch_geometric.data import Data

def augment(unsup_graph):
	aug1, aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
					A.FeatureMasking(pf=0.1),
					A.EdgeRemoving(pe=0.1)],
					num_choices=2)
	x1, edges1, edge_weights1 = aug1(unsup_graph['x'], unsup_graph['edge_index'], unsup_graph['edge_attr'])
	x2, edges2, edge_weights2 = aug2(unsup_graph['x'], unsup_graph['edge_index'], unsup_graph['edge_attr'])
	return Data(x=x1, edge_index=edges1, edge_attr=edge_weights1), Data(x=x2, edge_index=edges2, edge_attr=edge_weights2)

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
GNN_Encoder = GNN_Encoder(img_dim, emb_dim).to(device)
GNN_Decoder = GNN_Decoder(emb_dim, num_classes).to(device)

supCL_fn = loss.Node2NodeSupConLoss()
unsupCL_fn = contraster = DualBranchContrast(loss=L.InfoNCE(tau=config['temp']), mode='L2L', intraview_negs=True).to(device)
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
run = config['run'] + 1
config['run'] = run # updating run number
print(f'PixelGNN on {dataset}. Run {run} started at {time.strftime("%H:%M:%S")}')
starttime = time.time()

# iterate over dataloader and get a batch
for episode in range(n_episodes):
	print(f'Episode {episode}')
	
	for i, (q_index, sup_index, sup_graph, unsup_graph, task_graph, subcls_lists) in enumerate(trainloader):
		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()
		# all tensors are on device already
		# phase 1: CL
		# make augmentations from images (are they needed if we are sampling)
		# pass all through GNN_Encoder and get embeddings
		# q_embs = GNN_Encoder(q_graph)
		sup_embs = GNN_Encoder(sup_graph)
		unsup_graph_aug1, unsup_graph_aug2 = augment(unsup_graph)
		unsup_embs1 = GNN_Encoder(unsup_graph_aug1)
		unsup_embs2 = GNN_Encoder(unsup_graph_aug2)
		# once embeddings are here, make projection heads from a simple MLP?
		# ?
		# call the loss function on task graph augs (query??) and obtain contrastive loss
		sup_CL, unsup_CL = supCL_fn(sup_embs.x, sup_embs.y), unsupCL_fn(unsup_embs1.x, unsup_embs2.x)
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
		task_embs = task_embs.detach()
		task_embs = GNN_Decoder(task_embs)
		loss = loss_fn(task_embs[sup_index], task_graph[sup_index].y)
		loss.backward()
		decoder_optimizer.step()
		# task ends

	if i%10 == 0:
		print(f'Episode {i} complete, CL:{contrastive_loss.item()}\t(S:{sup_CL.item()},\tU:{unsup_CL.item()})')
		print(f'Classification Loss:{loss.item()}')


