from tqdm import tqdm
import time
from config import config
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import dataloader

dataset = config['dataset']
img_dim = 3 # for COCO/any color datasets
batch_size = config['batch_size']
n_episodes = config['n_episodes']
dataset_type = 'general'
if dataset in ['BCV', 'CT_ORG', 'DECATHLON']:
	img_dim = 1
	dataset_type = 'medical'
split = 0

if dataset_type == 'general':
	lists_path = config['lists_path'] + '/coco/'
	train_sup_list = lists_path + 'train_sup_list.json'
	train_unsup_list = lists_path + 'train_unsup_list.json'
	train_data = dataloader.GeneralDataLoader(train_sup_list, train_unsup_list, split)
	trainloader = DataLoader(train_data, batch_size=batch_size, shuffle = True, 
												drop_last=True) 

for i,j in tqdm(enumerate(trainloader), total=n_episodes, position=0, leave=True):
	tqdm.write(str(i))
