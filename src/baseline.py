import torch
from torch import nn


class LinearBaseline(nn.Module):
	"""Single-layer binary classifier for node-level baseline features."""

	def __init__(self, input_dim):
		super().__init__()
		self.linear1 = nn.Linear(input_dim, 100)
		self.linear2 = nn.Linear(100, 1)
		
		self.gelu = nn.GELU()
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		x = x.float().flatten(start_dim=1)
		x = self.gelu(self.linear1(x))
		x = self.linear2(x)
		return self.sigmoid(x)

