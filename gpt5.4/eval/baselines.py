from typing import Optional

import numpy as np
import torch


def max_power_allocation(num_links: int, p_max: float, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.full((num_links,), p_max, dtype=torch.float32, device=device)


def equal_power_allocation(num_links: int, p_max: float, device: Optional[torch.device] = None) -> torch.Tensor:
    return torch.full((num_links,), 0.5 * p_max, dtype=torch.float32, device=device)


def random_power_allocation(
    num_links: int,
    p_max: float,
    rng: np.random.Generator,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    power = rng.uniform(0.0, p_max, size=num_links).astype(np.float32)
    return torch.tensor(power, dtype=torch.float32, device=device)


def wmmse_power_allocation(
    channel_matrix: torch.Tensor,
    p_max: float,
    noise_power: float,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> torch.Tensor:
    # 基于 SISO 干扰网络的经典 WMMSE 迭代。
    device = channel_matrix.device
    num_links = channel_matrix.size(0)
    h_amp = torch.sqrt(torch.clamp(channel_matrix, min=1e-12))
    v = torch.full((num_links,), np.sqrt(0.5 * p_max), dtype=torch.float32, device=device)
    v_max = np.sqrt(p_max)

    for _ in range(max_iters):
        v_prev = v.clone()

        total_received_power = (h_amp.pow(2) @ (v.pow(2))) + noise_power
        u = torch.diag(h_amp) * v / (total_received_power + 1e-12)

        mse = 1.0 - 2.0 * u * torch.diag(h_amp) * v + (u.pow(2) * total_received_power)
        w = 1.0 / torch.clamp(mse, min=1e-12)

        for tx_idx in range(num_links):
            numerator = w[tx_idx] * u[tx_idx] * h_amp[tx_idx, tx_idx]
            denominator = torch.sum(w * (u.pow(2)) * (h_amp[:, tx_idx].pow(2)))
            v[tx_idx] = torch.clamp(numerator / (denominator + 1e-12), min=0.0, max=v_max)

        if torch.max(torch.abs(v - v_prev)).item() < tol:
            break

    return v.pow(2)
