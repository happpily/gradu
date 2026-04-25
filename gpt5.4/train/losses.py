from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from eval.metrics import compute_metrics, sinr_and_rates


def unsupervised_loss(
    data: Data,
    power: torch.Tensor,
    noise_power: float,
    rate_threshold: float,
    qos_lambda: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    _, rates = sinr_and_rates(data.channel_matrix, power, noise_power)
    sum_rate = rates.sum()

    # 软约束项用于提高可靠性，避免模型把弱链路完全牺牲掉。
    qos_penalty = F.relu(rate_threshold - rates).mean()
    loss = -sum_rate + qos_lambda * qos_penalty

    metrics = compute_metrics(data.channel_matrix, power, noise_power, rate_threshold)
    metrics["loss"] = float(loss.item())
    return loss, metrics
