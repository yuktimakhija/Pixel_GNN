import imp
# from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
import torch
import torch.nn as nn
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.distances import CosineSimilarity
from config import config

from numpy.random import default_rng
rng = default_rng()

device = torch.device('cuda:0'if torch.cuda.is_available() else "cpu")

class Node2NodeSupConLoss(nn.Module):
	def check_valid(self, y, sampled):
		# get unique classes and counts per class
		classes, counts = y[sampled].unique(return_counts=True)
		# ideal should be (total samples)/(total classes) [all classes have same number of images]
		ideal = y[sampled].shape[0]/classes.shape[0]
		print(counts)
		# normalize counts array by ideal value
		counts = counts/ideal
		# we can allow upto 20% deviation in the ratio, i.e. for 2 classes
		# it can be 0.8 for class A => class A has 0.8*ideal images (so B has 1.2*ideal), which translates to
		# a 40-60 division of images b/w class A and B
		# make this a config parameter?
		return (counts >= 0.8).all().item() and (counts <= 1.2).all().item()

		# for count in counts:
		# 	if count < 0.8 or count > 1.2:
		# 		return False
		# return True

	def forward(self, x, y):
		# x = torch.cat([graph['x'] for graph in graphs])
		# y = torch.cat([graph['y'] for graph in graphs])
		# x = sup_graph.x
		# y = sup_graph.y
		# total number of nodes in the batch
		# n = sum(graph['x'].shape[0] for graph in graphs)
		n = x.shape[0]
		self.n = n
		# pos_loss, neg_loss = 0,0
		
		# randomly sample anchors from all graphs
		selected_anchors = rng.integers(low=0, high=n, size=config['num_anchors'])
		# do we check_valid here?

		total_loss = 0
		for anchor in selected_anchors:
			sampled_nodes = rng.integers(low=0, high=n, size=config['num_samples'])
			# while the sample is not valid, sample again.
			# while(not self.check_valid(y, sampled_nodes)):
			# 	sampled_nodes = rng.integers(low=0, high=n, size=config['num_samples'])
			positive_samples = torch.where(y[sampled_nodes] == y[anchor])
			# make the negative_samples indices by making an array of ones and set the positive_samples to 0
			# negative_samples = torch.ones_like(positive_samples)
			# negative_samples[positive_samples] = 0
			pos_sim = CosineSimilarity(x[anchor], x[positive_samples])
			all_sim = CosineSimilarity(x[anchor], x[sampled_nodes])
			numerator = torch.sum(torch.exp(pos_sim/config['temp']))
			denominator = torch.sum(torch.exp(all_sim/config['temp']))
			# normalize by number of positive samples per anchor according to formula
			total_loss += (-1/positive_samples.shape[0])(torch.log(numerator/denominator))
		
		return total_loss

# class Node2NodeUnsupConLoss(nn.Module):
# 	def forward(self, unsup_graph_aug1, unsup_graph_aug2):
		
	
# class QueryClassificationLoss(nn.Module):
# 	def __init__(self):
# 		self.CE_fn = nn.CrossEntropyLoss()

# 	def forward(self, x, y):
# 		return self.CE_fn(x, y)