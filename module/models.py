import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from module.config import Config


class CGCNetLayer(MessagePassing):
    def __init__(self, node_dim: int, edge_dim: int, hidden1: int, hidden2: int):
        super().__init__(aggr="max")
        self.mlp1 = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(node_dim + hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, node_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        combined = torch.cat([x, aggr_out], dim=-1)
        out = self.mlp2(combined)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.mlp1(msg_input)


class CGCNet(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.p_max = config.p_max

        node_dim = 3
        edge_dim = 2

        self.shared_layer = CGCNetLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden1=config.hidden_dim1,
            hidden2=config.hidden_dim2,
        )
        self.output_layer = nn.Linear(node_dim * 2, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_init = x

        for _ in range(self.config.num_layers):
            residual = self.shared_layer(x, edge_index, edge_attr)
            if residual.shape != x.shape:
                raise ValueError(f"Residual shape mismatch: x={x.shape}, residual={residual.shape}")
            x = x + residual

        x_out = torch.cat([x, x_init], dim=-1)
        power_raw = self.output_layer(x_out).squeeze(-1)
        power = torch.sigmoid(power_raw) * self.p_max
        return power
