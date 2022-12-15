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
import torch.nn.functional as F
import pandas as pd


assert config['mode'] == 'val'

device = torch.device('cuda:0'if torch.cuda.is_available() else "cpu")
# print(device)
dataset = config['dataset']
img_dim = 3 # for COCO/any color datasets
batch_size = config['batch_size']
dataset_type = 'general'

emb_dim = config['embedding_dim']
num_classes = config['ways'] + 1
GNN_Encoder = GNN_Encoder(img_dim, emb_dim)
GNN_Decoder = GNN_Decoder(emb_dim, num_classes)

if len(sys.argv)>1:
	split = int(sys.argv[1])
	config['split'] = split # only a local change, wont be saved to file
else:
	split = config['split']

if dataset_type == 'general':
	lists_path = config['lists_path'] + '/coco/'
	val_sup_list = lists_path + 'val_sup_list.json'
	val_unsup_list = lists_path + 'val_unsup_list.json'
	val_data = dataloader.GeneralDataLoader(val_sup_list, val_unsup_list, split)
	testloader = DataLoader(val_data, batch_size=batch_size, shuffle = True, 
												drop_last=True) 

run_to_test = config['run_id_to_test']
dirname = f"./weights/{dataset}/{config['ways']}way_{config['shot']}shot/{run_to_test}/"
enc_path = dirname+f'enc_split{split}_0.25%.pt'
dec_path = dirname+f'dec_split{split}_0.25%.pt'
GNN_Encoder.load_state_dict(torch.load(enc_path))
GNN_Decoder.load_state_dict(torch.load(dec_path))
GNN_Encoder.to(device)
GNN_Decoder.to(device)
GNN_Encoder.eval()
GNN_Decoder.eval()

results_file = open(dirname+f'test_results.txt', 'w')
total_miou = []

starttime = time.time()

with torch.no_grad():
	for i, (q_index, sup_index, sup_graph, unsup_graph, task_graph, q_label) in tqdm(enumerate(testloader), total=len(testloader)):
		if q_index == -1 and sup_index == -1:
			# check dataloader (line 114)
			continue
		n = q_label.shape[-1] #65536
		task_embs = GNN_Encoder(task_graph) #returns tuple
		torch.save(task_embs[0], 'task_embs_enc.pt')
		torch.save(task_graph.y, 'task_embs_gt.pt')
		# print(task_embs[0][rng.integers(0,task_embs[0].shape[0],20)])
		task_embs_d = Data(x=task_embs[0].detach(), edge_index=task_embs[1][0].detach(), edge_attr=task_embs[1][1].detach())
		task_embs = GNN_Decoder(task_embs_d)
		q_label = q_label.to(device)
		q_index_fg = torch.where(q_label==1)[0]
		q_calcd = torch.arange(q_index[0].item()*n,(q_index[0].item()+1)*n)
		q_preds = task_embs[0][q_calcd]
		# print(task_embs[0][rng.integers(0,task_embs[0].shape[0],20)])
		torch.save(task_embs[0], 'task_embs_dec.pt')
		# print(q_preds, q_preds.shape)
		q_preds = F.softmax(q_preds,dim=1)
		q_preds_fg = q_preds[q_index_fg]
		# print(q_preds_fg, q_preds_fg.shape)
		q_preds = torch.argmax(q_preds, dim=1)
		# print(q_preds, q_preds.shape)
		# print(torch.where(q_preds==1)[0].shape) #check how many indices f prediction are fg
		# q_preds = q_preds.to(torch.int32)
		i, u, _ = loss.intersectionAndUnionGPU(q_preds, q_label[0], num_classes)
		# i and u are 1D tensors with num_classes elements (one value for each class)
		# miou = (i/u).mean()
		miou = (i/u)
		total_miou.append(miou.cpu().numpy())
		# print(q_preds)
		# print(q_label[0])
		tqdm.write(f'Latest image mIOU = {miou}')
	
# find mean of miou's of all images in the testloader (each image is a query once)
print(total_miou)
total_miou = torch.tensor(total_miou)
mean_miou = total_miou.mean(dim=-2)
std_miou = total_miou.std(dim=-2)

all_results = pd.read_csv('all_results.csv',index_col=[0])
all_results.loc[run_to_test] = [f"{dataset}-{split}", f"{config['ways']}-{config['shot']}-{config['unlabelled']}", mean_miou, std_miou]
all_results.to_csv('all_results.csv')

results_file.write(f"Results on run {run_to_test}(split{split})[N={config['ways']},K={config['shot']},M={config['unlabelled']}]:\n")
results_file.write(f"mIOU = {mean_miou} ± {std_miou}")

endtime = time.time()
h = (endtime-starttime)//3600
m = (endtime-starttime)//60 - h*60
s = (endtime-starttime)%60
print(f"Testing took {h} hrs, {m} mins, {s} seconds.")
# all_results.write(f"{run_id_to_test},{config['ways']}-{config['shot']}-{config['unlabelled']},{mean_miou},{std_miou}\n")
print(f"Results on run {run_to_test}(split{split})[N={config['ways']},K={config['shot']},M={config['unlabelled']}]:")
print(f"mIOU = {mean_miou} ± {std_miou}")
print("Testing ended.")
