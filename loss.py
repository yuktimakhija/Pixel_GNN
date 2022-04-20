from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
import torch
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.distances import CosineSimilarity

class Node2NodeSupConLoss(GenericPairLoss):
	def __init__(self, **kwargs):
		super.__init__(mat_based_loss=False, **kwargs)
	
	def _compute_loss(self, graphs, indices_tuple):
		# graphs is a list of all 2k Data objects in a task (k support images -> 2k augmentations)
		# we assume it is binary currently (either class A or not class A) 
		labels = [graph['y'] for graph in graphs]
		pos_loss, neg_loss = 0,0

		for graph in graphs:
			for node in range(graph['x'].shape[0]):
				current_label= graph['y'][node]
				pos_indices = torch.where(graph['y'] == current_label)
				neg_indices = torch.where(graph['y'] != current_label)
				graph['x'][pos_indices]
				pos_loss = torch.sum(torch.exp())

		self.pos_pairs = torch.where(graphs == )
		self.neg_pairs = torch.where(graphs != )
		# pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
		# neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)

		pos_loss = torch.sum(torch.exp())
		neg_loss = 

		return {
			"pos_loss": {
				"losses": pos_loss,
				"indices": self.pos_pairs,
				"reduction_type": "pos_pair",
			},
			"neg_loss": {
				"losses": neg_loss,
				"indices": self.neg_pairs,
				"reduction_type": "neg_pair",
			},
		}

	def pairwise_loss(self, option):
		# option tells whether to take 