# from torch_geometric.nn import GCN
# from torch_geometric.nn import GCN2Conv
# from torch_geometric.nn import GAT
# from torch_geometric.nn import GraphConv
# # from torch_geometric import SAGEConv # NO EDGE FEATURES
# from torch_geometric.nn import GINEConv
# from torch_geometric.nn import DeepGraphInfomax
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GNN_Encoder(torch.nn.Module):
	def __init__(self, img_in_dim, emb_dep):
		super().__init__()
		self.layer1 = GATv2Conv(img_in_dim, 8) # img_dim is 3 [R,G,B]
		self.layer2 = GATv2Conv(8, 16)
		self.layer3 = GATv2Conv(16, emb_dep)

	def forward(self, data:Data):
		x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
		x, (edge_index, edge_attr) = self.layer1(x, edge_index, edge_attr, return_attention_weights = True)		
		x = F.relu(x)
		x = F.dropout(x, training= self.training)
		x, (edge_index, edge_attr) = self.layer2(x, edge_index, edge_attr, return_attention_weights = True)
		x = F.relu(x)
		x = F.dropout(x, training= self.training)
		x, (edge_index, edge_attr) = self.layer3(x, edge_index, edge_attr, return_attention_weights = True)
		return x, (edge_index, edge_attr)

class GNN_Decoder(torch.nn.Module):
	def __init__(self, out_dim):
		super().__init__()
		self.layer1 = GATv2Conv(32, 16) # img_dim is 3 [R,G,B]
		self.layer2 = GATv2Conv(16, out_dim)

	def forward(self, data:Data):
		x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
		x, (edge_index, edge_attr) = self.layer1(x, edge_index, edge_attr, return_attention_weights = True)		
		x = F.relu(x)
		x = F.dropout(x, training= self.training)
		x, (edge_index, edge_attr) = self.layer2(x, edge_index, edge_attr, return_attention_weights = True)
		return x, (edge_index, edge_attr)

		
# def load_model(model_name, params):
# 	if 'model_name' == 'GCN':
# 		return GCN(params)
# 	elif 'model_name' == 'GCN2':
# 		return GCN2Conv(params)
# 	elif 'model_name' == 'GAT':
# 		return GAT(params) # need to put v2 = True 
# 	elif 'model_name' == 'GraphConv':
# 		return GraphConv(params)
# 	elif 'model_name' == 'GIN':
# 		return GINEConv(params)
# 	elif 'model_name' == 'DGI':
# 		return DeepGraphInfomax(params)
# 	else:
# 		raise ValueError("Wrong model")

