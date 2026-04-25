import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from module.config import Config


def _sample_tx_positions(n: int, area_size: float, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, area_size, size=(n, 2))


def _sample_rx_positions(
    tx_pos: np.ndarray, min_dist: float, max_dist: float, area_size: float, rng: np.random.Generator
) -> np.ndarray:
    n = tx_pos.shape[0]
    radius = rng.uniform(min_dist, max_dist, size=(n, 1))
    theta = rng.uniform(0.0, 2.0 * math.pi, size=(n, 1))
    offset = np.concatenate([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    rx_pos = np.clip(tx_pos + offset, 0.0, area_size)
    return rx_pos


def _compute_dist_matrix(tx_pos: np.ndarray, rx_pos: np.ndarray) -> np.ndarray:
    diff = tx_pos[:, None, :] - rx_pos[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _sample_rayleigh_channel(n: int, rng: np.random.Generator) -> np.ndarray:
    real = rng.normal(0.0, math.sqrt(0.5), size=(n, n))
    imag = rng.normal(0.0, math.sqrt(0.5), size=(n, n))
    return real + 1j * imag


def _compute_gain_matrix(h_tilde: np.ndarray, dist: np.ndarray, alpha: float) -> np.ndarray:
    return (np.abs(h_tilde) ** 2) / (np.power(dist, alpha) + 1e-6)


def _compute_external_interference(
    rx_pos: np.ndarray,
    area_size: float,
    alpha: float,
    config: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    n = rx_pos.shape[0]
    if not config.external_interference_enabled:
        return np.zeros(n, dtype=np.float32)

    ext_tx_pos = rng.uniform(0.0, area_size, size=(config.ext_interferer_count, 2))
    diff = ext_tx_pos[:, None, :] - rx_pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    h_real = rng.normal(0.0, math.sqrt(0.5), size=(config.ext_interferer_count, n))
    h_imag = rng.normal(0.0, math.sqrt(0.5), size=(config.ext_interferer_count, n))
    h_abs2 = h_real ** 2 + h_imag ** 2
    path_gain = h_abs2 / (np.power(dist, alpha) + 1e-6)

    active_mask = (rng.random(config.ext_interferer_count) < config.ext_activity_prob).astype(np.float32)
    active_power = config.ext_tx_power * active_mask[:, None]
    ext_interference = np.sum(active_power * path_gain, axis=0)

    burst_mask = (rng.random(n) < config.ext_burst_prob).astype(np.float32)
    burst_amp = rng.uniform(1.0, config.ext_burst_scale, size=n).astype(np.float32)
    ext_interference += burst_mask * burst_amp * config.noise_power

    return ext_interference.astype(np.float32)


def generate_network_data(
    n: int,
    area_size: float = 100.0,
    min_rx_dist: float = 2.0,
    max_rx_dist: float = 10.0,
    noise_power: float = 1e-3,
    alpha: float = 2.0,
    external_interference_enabled: bool = True,
    ext_interferer_count: int = 3,
    ext_tx_power: float = 0.8,
    ext_activity_prob: float = 0.65,
    ext_burst_prob: float = 0.15,
    ext_burst_scale: float = 3.0,
    seed: Optional[int] = None,
) -> Tuple[Data, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)

    tx_pos = _sample_tx_positions(n, area_size, rng)
    rx_pos = _sample_rx_positions(tx_pos, min_rx_dist, max_rx_dist, area_size, rng)
    dist_matrix = _compute_dist_matrix(tx_pos, rx_pos)
    h_tilde = _sample_rayleigh_channel(n, rng)
    g_matrix = _compute_gain_matrix(h_tilde, dist_matrix, alpha)

    cfg = Config(
        external_interference_enabled=external_interference_enabled,
        ext_interferer_count=ext_interferer_count,
        ext_tx_power=ext_tx_power,
        ext_activity_prob=ext_activity_prob,
        ext_burst_prob=ext_burst_prob,
        ext_burst_scale=ext_burst_scale,
    )
    ext_interference = _compute_external_interference(rx_pos, area_size, alpha, cfg, rng)

    diag_dist = np.diag(dist_matrix)
    alpha_i = 1.0 / (diag_dist + 1e-6)
    alpha_i = alpha_i / np.mean(alpha_i)

    g_diag = np.diag(g_matrix)
    x = np.stack([
        np.log1p(g_diag).astype(np.float32),
        alpha_i.astype(np.float32),
        np.full(n, np.log10(noise_power + 1e-12), dtype=np.float32),
    ], axis=1)

    src_nodes, dst_nodes = [], []
    edge_features = []
    for j in range(n):
        for i in range(n):
            if j == i:
                continue
            src_nodes.append(j)
            dst_nodes.append(i)
            edge_features.append([
                float(np.log1p(g_matrix[j, i])),
                float(np.log1p(g_matrix[i, j])),
            ])

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        g_matrix=torch.tensor(g_matrix, dtype=torch.float32),
        alpha_i=torch.tensor(alpha_i, dtype=torch.float32),
        ext_interference=torch.tensor(ext_interference, dtype=torch.float32),
        noise_power=torch.tensor(noise_power, dtype=torch.float32),
    )

    extra_info = {
        "tx_pos": tx_pos,
        "rx_pos": rx_pos,
        "distance_matrix": dist_matrix,
        "g_diag": g_diag,
        "alpha_i": alpha_i,
    }
    return data, extra_info
