from typing import Dict

import torch
from torch_geometric.data import Data


def graph_summary(data: Data) -> Dict[str, float]:
    # 为调试和论文记录提供简单图统计信息。
    direct_gain = torch.diag(data.channel_matrix)
    interference_gain = data.channel_matrix.sum() - direct_gain.sum()
    return {
        "num_nodes": float(data.num_nodes),
        "num_edges": float(data.edge_index.size(1)),
        "avg_direct_gain": direct_gain.mean().item(),
        "avg_interference_gain": interference_gain.item() / max(data.edge_index.size(1), 1),
    }
