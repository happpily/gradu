from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from delet.config import SimulationConfig


def _build_edge_index(num_links: int) -> torch.Tensor:
    # 构造全连接有向图，去掉自环。
    senders = []
    receivers = []
    for src in range(num_links):
        for dst in range(num_links):
            if src == dst:
                continue
            senders.append(src)
            receivers.append(dst)
    return torch.tensor([senders, receivers], dtype=torch.long)


def _sample_positions(cfg: SimulationConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tx_pos = rng.uniform(0.0, cfg.area_size, size=(cfg.num_links, 2))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=cfg.num_links)
    link_distances = rng.uniform(cfg.d_min, cfg.d_max, size=cfg.num_links)

    rx_pos = np.zeros_like(tx_pos)
    rx_pos[:, 0] = np.clip(tx_pos[:, 0] + link_distances * np.cos(angles), 0.0, cfg.area_size)
    rx_pos[:, 1] = np.clip(tx_pos[:, 1] + link_distances * np.sin(angles), 0.0, cfg.area_size)
    return tx_pos, rx_pos, link_distances


def _sample_channel_gain(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    # distance_matrix[i, j] 表示发射机 j 到接收机 i 的距离。
    diff = rx_pos[:, None, :] - tx_pos[None, :, :]
    distance_matrix = np.linalg.norm(diff, axis=-1)

    # 复高斯瑞利衰落：实部和虚部均服从 N(0, 0.5)。
    fading_real = rng.normal(0.0, np.sqrt(0.5), size=(cfg.num_links, cfg.num_links))
    fading_imag = rng.normal(0.0, np.sqrt(0.5), size=(cfg.num_links, cfg.num_links))
    fading_power = fading_real ** 2 + fading_imag ** 2

    channel_gain = fading_power / (np.power(distance_matrix, cfg.pathloss_exp) + 1e-6)
    return channel_gain, distance_matrix


def _build_node_features(
    channel_gain: np.ndarray,
    link_distances: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    # 节点特征做对数压缩，提升训练稳定性。
    direct_gain = np.diag(channel_gain)
    neighbor_ratio = np.full(cfg.num_links, (cfg.num_links - 1) / max(cfg.num_links, 1), dtype=np.float32)
    node_features = np.stack(
        [
            np.log10(direct_gain + 1e-12),
            np.log10(link_distances + 1e-12),
            np.full(cfg.num_links, np.log10(cfg.noise_power + 1e-12), dtype=np.float32),
            neighbor_ratio,
        ],
        axis=1,
    )
    return node_features.astype(np.float32)


def _build_edge_features(
    channel_gain: np.ndarray,
    distance_matrix: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    edge_features = []
    for src in range(cfg.num_links):
        for dst in range(cfg.num_links):
            if src == dst:
                continue
            edge_features.append(
                [
                    np.log10(channel_gain[dst, src] + 1e-12),
                    distance_matrix[dst, src] / cfg.area_size,
                ]
            )
    return np.asarray(edge_features, dtype=np.float32)


def generate_network_data(
    num_links: int,
    cfg: Optional[SimulationConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Data, Dict[str, torch.Tensor]]:
    # 对外暴露的单图生成接口，与任务书中的函数命名保持一致。
    cfg = cfg or SimulationConfig(num_links=num_links)
    cfg.num_links = num_links
    rng = rng or np.random.default_rng()

    tx_pos, rx_pos, link_distances = _sample_positions(cfg, rng)
    channel_gain, distance_matrix = _sample_channel_gain(tx_pos, rx_pos, cfg, rng)

    edge_index = _build_edge_index(cfg.num_links)
    node_features = _build_node_features(channel_gain, link_distances, cfg)
    edge_features = _build_edge_features(channel_gain, distance_matrix, cfg)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_features, dtype=torch.float32),
        channel_matrix=torch.tensor(channel_gain, dtype=torch.float32),
        distance_matrix=torch.tensor(distance_matrix, dtype=torch.float32),
        direct_distances=torch.tensor(link_distances, dtype=torch.float32),
        tx_pos=torch.tensor(tx_pos, dtype=torch.float32),
        rx_pos=torch.tensor(rx_pos, dtype=torch.float32),
        num_links=torch.tensor([cfg.num_links], dtype=torch.long),
    )

    extras = {
        "h_diag": torch.tensor(np.diag(channel_gain), dtype=torch.float32),
        "h_off_diag": torch.tensor(channel_gain - np.diag(np.diag(channel_gain)), dtype=torch.float32),
        "distance_matrix": torch.tensor(distance_matrix, dtype=torch.float32),
    }
    return data, extras


def generate_fixed_dataset(
    num_samples: int,
    cfg: SimulationConfig,
    seed: int,
) -> List[Data]:
    rng = np.random.default_rng(seed)
    dataset: List[Data] = []
    for _ in range(num_samples):
        data, _ = generate_network_data(cfg.num_links, cfg=cfg, rng=rng)
        dataset.append(data)
    return dataset
