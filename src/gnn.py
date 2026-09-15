import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNNodeClassifier(nn.Module):
	"""Multi-layer weighted GCN for node-level binary classification."""

	def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
		super().__init__()
		self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=True, normalize=True)
		self.output = GCNConv(hidden_dim , 1, add_self_loops=True, normalize=True)
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(dropout)

	def forward(self, data):
		edge_weight = data.edge_weight.float() if data.edge_weight is not None else None
		x = data.x.float()
		x = self.conv1(x, data.edge_index, edge_weight)
		x = self.activation(x)
		x = self.dropout(x)
		logits = self.output(x, data.edge_index, edge_weight)
		return torch.sigmoid(logits).squeeze(-1)

