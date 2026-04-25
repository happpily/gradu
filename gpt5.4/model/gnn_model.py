import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class GATPowerAllocator(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        p_max: float,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.p_max = p_max
        self.gat1 = GATConv(
            in_channels=node_dim,
            out_channels=64,
            heads=4,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            in_channels=64 * 4,
            out_channels=32,
            heads=1,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(32, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.gat1(data.x, data.edge_index, data.edge_attr)
        x = F.elu(x)
        x = self.gat2(x, data.edge_index, data.edge_attr)
        x = F.elu(x)
        power = torch.sigmoid(self.output_layer(x)).squeeze(-1) * self.p_max
        return power
