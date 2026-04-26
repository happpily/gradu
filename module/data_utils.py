"""
data_utils.py — 无线网络数据生成模块
=====================================
本模块负责随机生成无线网络拓扑数据，并将其转化为图神经网络所需的
PyG（PyTorch Geometric）图数据格式。

主要功能：
    1. 随机部署发射机和接收机位置（均匀分布）
    2. 计算距离矩阵和 Rayleigh 衰落信道矩阵
    3. 建模来自网络外部的干扰（如邻区基站、未授权用户等）
    4. 构建有向干扰图：每条有向边 (j→i) 表示发射机 j 对接收机 i 的干扰
    5. 提取节点特征和边特征，封装为 PyG Data 对象

信道模型说明：
    · 大尺度路径损耗：g_ij ∝ d_ij^{-α}，其中 α 为路径损耗指数
    · 小尺度衰落：Rayleigh 衰落，信道系数 h ~ CN(0,1)
    · 综合信道增益：G_ij = |h_ij|² / (d_ij^α + ε)

图数据格式：
    节点特征 x_i = [log(1+G_ii), α_i, log10(σ²)]
    边特征 e_{j→i} = [log(1+G_{ji}), log(1+G_{ij})]
    其中 G_ii 为直传信道增益，G_{ji} 为干扰信道增益
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from module.config import Config


def _sample_tx_positions(n: int, area_size: float, rng: np.random.Generator) -> np.ndarray:
    """
    在正方形区域内均匀随机部署 n 个发射机。
    Uniformly deploy n transmitters in the square deployment area.

    Args:
        n: 发射机数量 (Number of transmitters)
        area_size: 区域边长 [m] (Area side length)
        rng: NumPy 随机数生成器 (Random number generator)

    Returns:
        tx_pos: [n, 2] 发射机坐标矩阵 (Transmitter positions)
    """
    return rng.uniform(0.0, area_size, size=(n, 2))


def _sample_rx_positions(
    tx_pos: np.ndarray,
    min_dist: float,
    max_dist: float,
    area_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    在各发射机周围随机部署对应的接收机（距离约束在 [min_dist, max_dist]）。
    Deploy receivers near each transmitter with constrained distance.

    每个接收机的位置由随机极坐标偏移确定：
        rx_i = tx_i + r_i·[cos(θ_i), sin(θ_i)]
        其中 r_i ~ Uniform(min_dist, max_dist)，θ_i ~ Uniform(0, 2π)
    接收机坐标被裁剪到区域边界内。

    Args:
        tx_pos   : [n, 2] 发射机坐标
        min_dist : 最小收发距离 [m]
        max_dist : 最大收发距离 [m]
        area_size: 区域边长 [m]
        rng      : 随机数生成器

    Returns:
        rx_pos: [n, 2] 接收机坐标矩阵
    """
    n = tx_pos.shape[0]
    radius = rng.uniform(min_dist, max_dist, size=(n, 1))       # 随机距离
    theta = rng.uniform(0.0, 2.0 * math.pi, size=(n, 1))       # 随机角度
    offset = np.concatenate([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    rx_pos = np.clip(tx_pos + offset, 0.0, area_size)           # 裁剪到边界
    return rx_pos


def _compute_dist_matrix(tx_pos: np.ndarray, rx_pos: np.ndarray) -> np.ndarray:
    """
    计算所有发射机到所有接收机的欧氏距离矩阵。
    Compute Euclidean distance matrix between all TX-RX pairs.

    D[j, i] = ||tx_j - rx_i||_2  表示发射机 j 到接收机 i 的距离

    Args:
        tx_pos: [n, 2] 发射机坐标
        rx_pos: [n, 2] 接收机坐标

    Returns:
        dist_matrix: [n, n] 距离矩阵，dist_matrix[j, i] 为 TX_j → RX_i 的距离
    """
    diff = tx_pos[:, None, :] - rx_pos[None, :, :]  # [n, n, 2]
    return np.linalg.norm(diff, axis=-1)             # [n, n]


def _sample_rayleigh_channel(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    生成 n×n 复数 Rayleigh 衰落信道矩阵。
    Generate n×n complex Rayleigh fading channel matrix.

    复高斯信道系数：h_{j,i} = (X + jY) / √2，其中 X,Y ~ N(0, 0.5)
    由此保证 |h_{j,i}|² ~ Exp(1)（单位功率 Rayleigh 衰落）

    Args:
        n  : 链路数量
        rng: 随机数生成器

    Returns:
        h_tilde: [n, n] 复数信道矩阵，h_tilde[j, i] 为 TX_j → RX_i 的信道系数
    """
    real = rng.normal(0.0, math.sqrt(0.5), size=(n, n))
    imag = rng.normal(0.0, math.sqrt(0.5), size=(n, n))
    return real + 1j * imag


def _compute_gain_matrix(h_tilde: np.ndarray, dist: np.ndarray, alpha: float) -> np.ndarray:
    """
    计算综合信道增益矩阵（路径损耗 + 小尺度衰落）。
    Compute composite channel gain matrix (path loss + small-scale fading).

    G[j, i] = |h_{j,i}|² / (d_{j,i}^α + ε)

    其中：
        |h_{j,i}|² : Rayleigh 衰落功率增益（指数分布）
        d_{j,i}^α  : 路径损耗（α 为路径损耗指数）
        ε = 1e-6   : 数值稳定常数，避免近距离除零

    Args:
        h_tilde: [n, n] 复数 Rayleigh 信道矩阵
        dist   : [n, n] 距离矩阵
        alpha  : 路径损耗指数（典型值 2.0~3.5）

    Returns:
        g_matrix: [n, n] 实数信道增益矩阵
    """
    return (np.abs(h_tilde) ** 2) / (np.power(dist, alpha) + 1e-6)


def _compute_external_interference(
    rx_pos: np.ndarray,
    area_size: float,
    alpha: float,
    config: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    建模外部干扰功率（来自网络外部的非协作干扰源）。
    Model external interference power from non-cooperative interferers.

    干扰模型分为两部分：
    ① 持续干扰：ext_interferer_count 个干扰源，每个以概率 ext_activity_prob 活跃
       ext_i = Σ_k p_k · G_{k,i}，其中 p_k ∈ {0, ext_tx_power}
    ② 突发干扰：模拟突然出现的强干扰（如雷达、工业设备）
       以概率 ext_burst_prob 触发，强度为 Uniform(1, ext_burst_scale) × σ²

    Args:
        rx_pos   : [n, 2] 接收机坐标
        area_size: 区域边长 [m]
        alpha    : 路径损耗指数
        config   : 全局配置（含外部干扰参数）
        rng      : 随机数生成器

    Returns:
        ext_interference: [n] 各接收机承受的外部干扰功率 [W]
    """
    n = rx_pos.shape[0]
    if not config.external_interference_enabled:
        return np.zeros(n, dtype=np.float32)  # 禁用外部干扰时返回零向量

    # ── 随机部署外部干扰源 ────────────────────────────────────────────────
    ext_tx_pos = rng.uniform(0.0, area_size, size=(config.ext_interferer_count, 2))

    # 计算外部干扰源到各接收机的信道增益
    diff = ext_tx_pos[:, None, :] - rx_pos[None, :, :]          # [K, n, 2]
    dist = np.linalg.norm(diff, axis=-1)                         # [K, n]
    h_real = rng.normal(0.0, math.sqrt(0.5), size=(config.ext_interferer_count, n))
    h_imag = rng.normal(0.0, math.sqrt(0.5), size=(config.ext_interferer_count, n))
    h_abs2 = h_real ** 2 + h_imag ** 2                          # Rayleigh 衰落功率
    path_gain = h_abs2 / (np.power(dist, alpha) + 1e-6)         # 路径损耗修正

    # ① 持续干扰：依据活跃概率决定各干扰源是否发射
    active_mask = (rng.random(config.ext_interferer_count) < config.ext_activity_prob).astype(np.float32)
    active_power = config.ext_tx_power * active_mask[:, None]    # [K, 1]
    ext_interference = np.sum(active_power * path_gain, axis=0)  # [n]

    # ② 突发干扰：随机叠加强脉冲干扰
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
    """
    生成一个随机无线网络拓扑，返回 PyG 图数据和辅助信息。
    Generate a random wireless network topology as a PyG graph.

    生成流程：
        1. 随机部署 n 对收发节点
        2. 计算路径损耗 + Rayleigh 衰落综合信道增益矩阵 G
        3. 建模外部干扰
        4. 提取节点特征、构建有向干扰图（全连接，去自环）
        5. 封装为 PyG Data 对象

    Args:
        n                 : 链路对数
        area_size         : 部署区域边长 [m]
        min_rx_dist       : 最小收发距离 [m]
        max_rx_dist       : 最大收发距离 [m]
        noise_power       : 热噪声功率 [W]
        alpha             : 路径损耗指数
        external_interference_enabled: 是否启用外部干扰
        ext_interferer_count         : 外部干扰源数量
        ext_tx_power                 : 外部干扰源发射功率 [W]
        ext_activity_prob            : 干扰源活跃概率
        ext_burst_prob               : 突发干扰概率
        ext_burst_scale              : 突发干扰强度倍数
        seed              : 随机种子（None 表示不固定）

    Returns:
        data     : PyG Data 对象（图神经网络输入）
        extra_info: 包含位置、距离矩阵等辅助信息的字典
    """
    rng = np.random.default_rng(seed)

    # ── Step 1: 生成网络拓扑 ─────────────────────────────────────────────
    tx_pos = _sample_tx_positions(n, area_size, rng)
    rx_pos = _sample_rx_positions(tx_pos, min_rx_dist, max_rx_dist, area_size, rng)

    # ── Step 2: 计算信道矩阵 ─────────────────────────────────────────────
    dist_matrix = _compute_dist_matrix(tx_pos, rx_pos)  # [n, n]，D[j,i]=TX_j→RX_i
    h_tilde = _sample_rayleigh_channel(n, rng)           # Rayleigh 衰落信道
    g_matrix = _compute_gain_matrix(h_tilde, dist_matrix, alpha)  # 综合信道增益 [n, n]

    # ── Step 3: 计算外部干扰 ─────────────────────────────────────────────
    cfg_ext = Config(
        external_interference_enabled=external_interference_enabled,
        ext_interferer_count=ext_interferer_count,
        ext_tx_power=ext_tx_power,
        ext_activity_prob=ext_activity_prob,
        ext_burst_prob=ext_burst_prob,
        ext_burst_scale=ext_burst_scale,
    )
    ext_interference = _compute_external_interference(rx_pos, area_size, alpha, cfg_ext, rng)

    # ── Step 4: 计算节点权重 α_i（基于链路距离的反比权重，归一化） ──────
    # α_i 越大表示该链路距离越短、路径损耗越小，可分配更多功率资源
    diag_dist = np.diag(dist_matrix)             # 各链路收发距离 d_ii
    alpha_i = 1.0 / (diag_dist + 1e-6)          # 距离反比权重
    alpha_i = alpha_i / np.mean(alpha_i)         # 归一化，均值为 1

    # ── Step 5: 构建节点特征矩阵 x ─────────────────────────────────────
    # 节点特征：[对数直传信道增益, 链路权重, 对数噪声功率]
    # Node features: [log direct gain, link weight, log noise power]
    g_diag = np.diag(g_matrix)  # 各链路直传信道增益 G_ii
    x = np.stack([
        np.log1p(g_diag).astype(np.float32),                               # log(1 + G_ii)
        alpha_i.astype(np.float32),                                          # 链路权重
        np.full(n, np.log10(noise_power + 1e-12), dtype=np.float32),        # log10(σ²)
    ], axis=1)  # [n, 3]

    # ── Step 6: 构建有向干扰图（完全图，去自环） ────────────────────────
    # 有向边 (j→i) 表示发射机 j 对接收机 i 造成的干扰
    # Directed edge (j→i): TX_j interferes with RX_i
    src_nodes, dst_nodes = [], []
    edge_features = []
    for j in range(n):
        for i in range(n):
            if j == i:
                continue  # 跳过自环（直传链路不是干扰） (Skip self-loops)
            src_nodes.append(j)
            dst_nodes.append(i)
            edge_features.append([
                float(np.log1p(g_matrix[j, i])),  # 干扰信道增益 log(1+G_{j,i})
                float(np.log1p(g_matrix[i, j])),  # 反向信道增益 log(1+G_{i,j})（互易性）
            ])

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)  # [2, E]
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)          # [E, 2]

    # ── Step 7: 封装为 PyG Data 对象 ────────────────────────────────────
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),                               # 节点特征 [N, 3]
        edge_index=edge_index,                                                  # 边索引 [2, E]
        edge_attr=edge_attr,                                                    # 边特征 [E, 2]
        g_matrix=torch.tensor(g_matrix, dtype=torch.float32),                  # 信道增益矩阵 [N, N]
        alpha_i=torch.tensor(alpha_i, dtype=torch.float32),                    # 链路权重 [N]
        ext_interference=torch.tensor(ext_interference, dtype=torch.float32),  # 外部干扰 [N]
        noise_power=torch.tensor(noise_power, dtype=torch.float32),             # 噪声功率（标量）
    )

    # 辅助信息（用于可视化和分析，不参与训练）
    # Auxiliary info (for visualization/analysis, not used in training)
    extra_info = {
        "tx_pos": tx_pos,
        "rx_pos": rx_pos,
        "distance_matrix": dist_matrix,
        "g_diag": g_diag,
        "alpha_i": alpha_i,
    }
    return data, extra_info
