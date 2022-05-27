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
from torch_geometric.data import Data, DataLoader
import json
rng = np.random.default_rng()
from torch_geometric.profile import count_parameters
import os, sys
import pandas as pd
from torch_geometric import utils

def augment(x, edge_index, edge_attr):
	aug1 = A.RandomChoice([A.FeatureMasking(pf=0.2),
					A.NodeDropping(pn=0.2),
					A.EdgeRemoving(pe=0.2)],
					num_choices=2)
	aug2 = A.RandomChoice([A.FeatureMasking(pf=0.2),
					A.NodeDropping(pn=0.2),
					A.EdgeRemoving(pe=0.2)],
					num_choices=2)
	x1, edges1, edge_weights1 = aug1(x, edge_index, edge_attr)
	x2, edges2, edge_weights2 = aug2(x, edge_index, edge_attr)
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
# num_classes = config['classes'][dataset]
num_classes = config['ways'] + 1
GNN_Encoder = GNN_Encoder(img_dim, emb_dim).to(device)
GNN_Decoder = GNN_Decoder(emb_dim, num_classes).to(device)

supCL_fn = loss.Node2NodeSupConLoss()
unsupCL_fn = contraster = DualBranchContrast(loss=L.InfoNCE(tau=config['temp']), mode='L2L', intraview_negs=True).to(device)
unsup_weight = config['unsup_weight']
# loss_fn = loss.QueryClassificationLoss()
loss_fn = torch.nn.CrossEntropyLoss()
encoder_optimizer = torch.optim.Adadelta(GNN_Encoder.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
decoder_optimizer = torch.optim.Adadelta(GNN_Decoder.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

if len(sys.argv)>1:
	split = int(sys.argv[1])
else:
	split = config['split']
	config['split'] = split # only a local change, wont be saved to file

if dataset_type == 'general':
	lists_path = config['lists_path'] + '/coco/'
	train_sup_list = lists_path + 'train_sup_list.json'
	train_unsup_list = lists_path + 'train_unsup_list.json'
	train_data = dataloader.GeneralDataLoader(train_sup_list, train_unsup_list, split)
	trainloader = DataLoader(train_data, batch_size=batch_size, shuffle = True, 
												drop_last=True) 


# elif dataset_type == 'medical':


GNN_Encoder.train()
GNN_Decoder.train()
run_dict = json.load(open('runs.json'))
run = run_dict['last_run'] + 1
print(f'PixelGNN on {dataset}. Run {run} started at {time.strftime("%H:%M:%S")}')

dirname = f"./weights/{dataset}/{config['ways']}way_{config['shot']}shot/{run}/"
os.makedirs(dirname, exist_ok=True)

outfile = open(dirname+f'summary_split{split}.txt', 'w')
outcsv = open(dirname+f'summary_split{split}.csv', 'w')
outcsv.write('Episode, CL, SupCL, UnsupCL, ClassificationLoss\n')
json.dump(config, open(dirname+f'config_split{split}.json', 'w'), indent=4)
run_dict['list'][run] = dirname # updating run number
run_dict['last_run'] = run
json.dump(run_dict, open('runs.json', 'w'), indent=4)

master = pd.read_csv('all_runs.csv', index_col=[0])
master.loc[run] = [f"{config['ways']}-{config['shot']}-{config['unlabelled']}", f"{dataset}-{split}", n_episodes, batch_size, 'No']
master.to_csv('all_runs.csv')

# master.write(f"\n{run},{config['ways']}-{config['shot']}-{config['unlabelled']},{dataset}-{split},{n_episodes},{batch_size},")
# master.close() 
# this line was causing desync issues (like YesYesYes appearing in the csv when multiple models are run together)

print("Encoder Params",count_parameters(GNN_Encoder))
print("Decoder Params",count_parameters(GNN_Decoder))
starttime = time.time()
# iterate over dataloader and get a batch
# for episode in tqdm(range(n_episodes)):
	# tqdm.write(f'Episode {episode}')
for i, (q_index, sup_index, sup_graph, unsup_graph, task_graph, q_label) in tqdm(enumerate(trainloader), total=n_episodes):
	# print(sup_graph.x.min(), sup_graph.x.max())
	episode_losses = [0,0,0,0,0] # CL, supCL, unsupCL, nodeClassification, only query nodeClossification
	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()
	# all tensors are on device already
	# phase 1: CL
	# make augmentations from images (are they needed if we are sampling)
	# pass all through GNN_Encoder and get embeddings
	# q_embs = GNN_Encoder(q_graph)
	sup_embs = GNN_Encoder(sup_graph)
	selected_anchors = rng.integers(low=0, high=unsup_graph.x.shape[0], size=config['unsup_anchors']*config['unlabelled'])
	# take the subgraph (the function returns edge_index and edge_attr)
	sub_e, sub_ew = utils.subgraph(torch.tensor(selected_anchors), unsup_graph.edge_index, unsup_graph.edge_attr, relabel_nodes=True)
	unsup_graph_aug1, unsup_graph_aug2 = augment(unsup_graph.x[selected_anchors], sub_e, sub_ew)
	unsup_embs1 = GNN_Encoder(unsup_graph_aug1)
	unsup_embs2 = GNN_Encoder(unsup_graph_aug2)
	# once embeddings are here, make projection heads from a simple MLP?
	# ?
	# call the loss function on task graph augs (query??) and obtain contrastive loss
	sup_CL = supCL_fn(sup_embs[0], sup_graph.y)
	unsup_CL = unsupCL_fn(unsup_embs1[0], unsup_embs2[0])
	contrastive_loss = (1-unsup_weight)*sup_CL + unsup_weight*unsup_CL
	episode_losses[0] += contrastive_loss.item()
	episode_losses[1] += sup_CL.item()
	episode_losses[2] += unsup_CL.item()
	# backprop through optimizer
	contrastive_loss.backward()
	encoder_optimizer.step()
	# phase 2: node classification
	# GNN_Encoder.no_grad()
	# get embeddings for task graphs and query graphs (all batches/parallel)
	with torch.no_grad():
		task_embs = GNN_Encoder(task_graph)
	task_embs_d = Data(x=task_embs[0].detach(), edge_index=task_embs[1][0].detach(), edge_attr=task_embs[1][1].detach())
	task_embs = GNN_Decoder(task_embs_d)
	# sup_index += q_index
	loss_labels = torch.tensor([])
	calcd_indices = torch.tensor([], dtype=torch.long)
	# print(sup_index, q_index)
	# sup_index is [tensor([1, 1, 0, 1]), tensor([3, 3, 3, 3])]
	n = q_label.shape[-1]
	for j in sup_index[0].tolist():
		# now iterating inside batch (batch_size # of iterations)
		calcd_indices = torch.cat((calcd_indices,torch.arange(j*n,(j+1)*n)))
	# print(f't{task_graph}')
	# print(f't{task_graph.y[calcd_indices].shape}')
	# print(f'q{q_label.shape}')
	# print(f'c{calcd_indices.shape}')
	# print(task_graph.y[0], task_graph.y[n], task_graph.y[2*n])
	# print(task_graph.y.min(), task_graph.y.max())
	# print(task_graph.y[calcd_indices].min(), task_graph.y[calcd_indices].max())
	# if task_graph.y[calcd_indices].min() == -1:
	# 	continue
	loss_labels = torch.cat((task_graph.y[calcd_indices], q_label.reshape(-1).to(device))).type(torch.LongTensor)
	# print(q_label.min(), q_label.max())
	loss_labels = loss_labels.to(device)
	# print(loss_labels.min(), loss_labels.max())
	# only one query
	q_calcd = torch.arange(q_index[0].item()*n,(q_index[0].item()+1)*n)
	calcd_indices = torch.cat((calcd_indices, q_calcd))
	# print(f'ls {loss_labels.shape}')
	# print(f'ts {task_embs[0][calcd_indices].shape}')
	loss = loss_fn(task_embs[0][calcd_indices], loss_labels)
	episode_losses[3] += loss.item()
	loss.backward()
	decoder_optimizer.step()
	# q_loss = loss_fn(task_embs[0][q_calcd].unsqueeze(0), q_label.to(device))
	# episode_losses[4] += q_loss.item()
	# task ends

	if i/n_episodes in [0.25,0.5,0.75,1.0]:
		torch.save(GNN_Encoder.state_dict(), dirname+f'enc_split{split}_{i/n_episodes}%.pt')
		torch.save(GNN_Decoder.state_dict(), dirname+f'dec_split{split}_{i/n_episodes}%.pt')
	tqdm.write(f'Episode {i} complete, CL:{episode_losses[0]}\t(S:{episode_losses[1]},\tU:{episode_losses[2]})')
	tqdm.write(f'Classification Loss:{episode_losses[3]}')
	outfile.write(f'Episode {i} complete, CL:{episode_losses[0]}\t(S:{episode_losses[1]},\tU:{episode_losses[2]})\n')
	outfile.write(f'Classification Loss:{episode_losses[3]}\n')
	outcsv.write(f'{i},{episode_losses[0]},{episode_losses[1]},{episode_losses[2]},{episode_losses[3]}\n')
	if i+1 > n_episodes:
		break

torch.save(GNN_Encoder.state_dict(), dirname+f'enc_split{split}_final.pt')
torch.save(GNN_Decoder.state_dict(), dirname+f'dec_split{split}_final.pt')

# master = pd.read_csv('all_runs.csv', index_cols=[0])
master.loc[run, 'Completed?'] = "Yes"
master.to_csv('all_runs.csv')
# master.close()

print(f'PixelGNN training on {dataset} finished. Run {run} ended at {time.strftime("%H:%M:%S")}')
endtime = time.time()
h = (endtime-starttime)//3600
m = (endtime-starttime)//60 - h*60
s = (endtime-starttime)%60
print(f"Run took {h} hrs, {m} mins, {s} seconds.")