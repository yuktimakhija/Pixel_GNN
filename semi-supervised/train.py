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

def augment(unsup_graph):
	aug1 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
					A.FeatureMasking(pf=0.1),
					A.EdgeRemoving(pe=0.1)],
					num_choices=1)
	aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
					A.FeatureMasking(pf=0.1),
					A.EdgeRemoving(pe=0.1)],
					num_choices=1)
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
# loss_fn = loss.QueryClassificationLoss()
loss_fn = torch.nn.CrossEntropyLoss()
encoder_optimizer = torch.optim.Adam(GNN_Encoder.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
decoder_optimizer = torch.optim.Adam(GNN_Decoder.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

if dataset_type == 'general':
	lists_path = config['lists_path'] + '/coco/'
	train_sup_list = lists_path + 'train_sup_list.json'
	train_unsup_list = lists_path + 'train_unsup_list.json'
	train_data = dataloader.GeneralDataLoader(train_sup_list, train_unsup_list)
	trainloader = DataLoader(train_data, batch_size=batch_size, shuffle = True, 
												drop_last=True) 


# elif dataset_type == 'medical':


GNN_Encoder.train()
GNN_Decoder.train()
run_dict = json.load(open('runs.json'))
run = run_dict['last_run'] + 1
print(f'PixelGNN on {dataset}. Run {run} started at {time.strftime("%H:%M:%S")}')
starttime = time.time()

dirname = f"./weights/{dataset}/{config['ways']}way_{config['shot']}shot/{run}/"
run_dict['list'][run] = dirname # updating run number
json.dump(run_dict, open('runs.json', 'w'), indent=4)

# iterate over dataloader and get a batch
for episode in tqdm(range(n_episodes)):
	# tqdm.write(f'Episode {episode}')
	episode_losses = [0,0,0,0,0] # CL, supCL, unsupCL, nodeClassification, only query nodeClossification
	for i, (q_index, sup_index, sup_graph, unsup_graph, task_graph, q_label) in tqdm(enumerate(trainloader), leave=False):
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
		sup_CL = supCL_fn(sup_embs[0], sup_graph.y)
		selected_anchors = rng.integers(low=0, high=unsup_embs1[0].shape[0], size=config['num_anchors'])
		unsup_CL = unsupCL_fn(unsup_embs1[0][selected_anchors], unsup_embs2[0][selected_anchors])
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
		loss = loss_fn(task_embs[0][sup_index + q_index], torch.cat((task_graph.y[sup_index], q_label.to(device))))
		episode_losses[3] += loss.item()
		loss.backward()
		decoder_optimizer.step()
		q_loss = loss_fn(task_embs[0][q_index], q_label)
		episode_losses[4] += q_loss.item()
		# task ends

	if i%20 == 0:
		torch.save(GNN_Encoder.state_dict(), dirname+f'enc_split{config["split"]}_{i/n_episodes}%.pt')
		torch.save(GNN_Decoder.state_dict(), dirname+f'dec_split{config["split"]}_{i/n_episodes}%.pt')

	tqdm.write(f'Episode {i} complete, CL:{episode_losses[0]}\t(S:{episode_losses[1]},\tU:{episode_losses[2]})')
	tqdm.write(f'Classification Loss:{episode_losses[3]}, \tQuery Loss:{episode_losses[4]}')

print(f'PixelGNN training on {dataset} finished. Run {run} ended at {time.strftime("%H:%M:%S")}')
endtime = time.time()
h = (endtime-starttime)//3600
m = (endtime-starttime)//60 - h*60
s = (endtime-starttime)%60
print(f"Run took {h} hrs, {m} mins, {s} seconds.")