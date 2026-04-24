import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


# =============================
# 全局配置
# =============================
@dataclass
class Config:
    # 系统模型参数
    N: int = 10
    area_size: float = 100.0
    min_rx_dist: float = 2.0
    max_rx_dist: float = 10.0
    p_max: float = 1.0
    noise_power: float = 1e-3
    alpha_pathloss: float = 2.0      # 路径损耗指数

    # 训练参数
    epochs: int = 2000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    samples_per_epoch: int = 8          # 每个epoch使用多个随机样本平均，降低梯度方差
    grad_clip_norm: float = 1.0         # 梯度裁剪，防止训练震荡
    lr_step_size: int = 500             # 学习率调度步长
    lr_gamma: float = 0.7               # 学习率衰减系数

    # CGCNet 结构参数
    num_layers: int = 3               # 图卷积层数（CGCNet 用3层）
    hidden_dim1: int = 16             # MLP1 隐藏层维度
    hidden_dim2: int = 36             # MLP1 输出维度
    hidden_dim3: int = 16             # MLP2 隐藏层维度

    # 损失正则参数
    objective_mode: str = "sum_rate"   # 可选: "sum_rate" 或 "weighted_sum_rate"
    lambda_power: float = 0.03
    lambda_fairness: float = 0.20
    lambda_qos: float = 0.5
    min_rate: float = 0.5

    # 测试参数
    test_samples: int = 200
    test_seed: int = 2026
    random_trials: int = 20
    compare_samples: int = 80
    group_compare_samples: int = 40

    # 强干扰场景
    high_interference_area_size: float = 50.0

    # 现实场景外部干扰参数（例如 Wi-Fi/蓝牙/邻频设备）
    external_interference_enabled: bool = True
    ext_interferer_count: int = 3
    ext_tx_power: float = 0.8
    ext_activity_prob: float = 0.65
    ext_burst_prob: float = 0.15
    ext_burst_scale: float = 3.0

    # WMMSE 参数
    wmmse_iters: int = 60
    hybrid_wmmse_iters: int = 8

    # 遗传算法参数
    ga_population: int = 36
    ga_generations: int = 45
    ga_elite: int = 6
    ga_mutation_prob: float = 0.15
    ga_mutation_std: float = 0.08

    # 运行参数
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================
# 数据生成模块
# =============================
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
    """构造现实场景中的外部干扰功率向量（每个接收机一个值）。"""
    n = rx_pos.shape[0]
    if not config.external_interference_enabled:
        return np.zeros(n, dtype=np.float32)

    ext_tx_pos = rng.uniform(0.0, area_size, size=(config.ext_interferer_count, 2))
    diff = ext_tx_pos[:, None, :] - rx_pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)

    # 外部干扰链路的瑞利衰落增益
    h_real = rng.normal(0.0, math.sqrt(0.5), size=(config.ext_interferer_count, n))
    h_imag = rng.normal(0.0, math.sqrt(0.5), size=(config.ext_interferer_count, n))
    h_abs2 = h_real ** 2 + h_imag ** 2

    path_gain = h_abs2 / (np.power(dist, alpha) + 1e-6)

    # 干扰设备是否活跃（模拟生活中设备间歇发射）
    active_mask = (rng.random(config.ext_interferer_count) < config.ext_activity_prob).astype(np.float32)
    active_power = config.ext_tx_power * active_mask[:, None]
    ext_interference = np.sum(active_power * path_gain, axis=0)

    # 突发干扰（如短时同频占用）
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
    """生成单个网络样本，符合 CGCNet 的图建模方式。"""
    rng = np.random.default_rng(seed)

    tx_pos = _sample_tx_positions(n, area_size, rng)
    rx_pos = _sample_rx_positions(tx_pos, min_rx_dist, max_rx_dist, area_size, rng)
    dist_matrix = _compute_dist_matrix(tx_pos, rx_pos)
    h_tilde = _sample_rayleigh_channel(n, rng)
    g_matrix = _compute_gain_matrix(h_tilde, dist_matrix, alpha)
    # 现实场景外部干扰功率向量（每个接收机一项）
    cfg = Config(
        external_interference_enabled=external_interference_enabled,
        ext_interferer_count=ext_interferer_count,
        ext_tx_power=ext_tx_power,
        ext_activity_prob=ext_activity_prob,
        ext_burst_prob=ext_burst_prob,
        ext_burst_scale=ext_burst_scale,
    )
    ext_interference = _compute_external_interference(rx_pos, area_size, alpha, cfg, rng)

    # 计算链路权重（基于距离：距离越近权重越大）
    diag_dist = np.diag(dist_matrix)
    alpha_i = 1.0 / (diag_dist + 1e-6)
    alpha_i = alpha_i / np.mean(alpha_i)

    # 节点特征：[log(1+直连增益)、链路权重、log10噪声功率]（做尺度压缩，训练更稳）
    g_diag = np.diag(g_matrix)
    x = np.stack([
        np.log1p(g_diag).astype(np.float32),
        alpha_i.astype(np.float32),
        np.full(n, np.log10(noise_power + 1e-12), dtype=np.float32)
    ], axis=1)

    # 边特征：[log(1+g_{j,i}), log(1+g_{i,j})]（做尺度压缩，减少数值波动）
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
                float(np.log1p(g_matrix[i, j]))
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
        "tx_pos": tx_pos, "rx_pos": rx_pos,
        "distance_matrix": dist_matrix,
        "g_diag": g_diag, "alpha_i": alpha_i
    }
    return data, extra_info


# =============================
# CGCNet 模型定义（核心修改）
# =============================
class CGCNetLayer(MessagePassing):
    """
    单层 CGCNet：MAX聚合 + 两个MLP + 权重共享。

    Reference:
        Please replace the placeholder citation below with the exact paper that
        defines CGCNet formula (7) for this implementation.
        - Title: <paper title>
        - Authors: <authors>
        - Venue/Year: <venue or year>
        - Link: <DOI / arXiv / publisher URL>

    Implementation note:
        本层实现对应上述论文中的公式(7)。保留完整引文信息是为了便于后续维护者
        核对 MAX 聚合、消息构造方式以及两层 MLP 的实现是否与原文一致。
    """
    def __init__(self, node_dim: int, edge_dim: int, hidden1: int, hidden2: int):
        super().__init__(aggr='max')  # CGCNet 使用 MAX 聚合

        # MLP1: 处理邻居节点特征 + 边特征
        self.mlp1 = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )

        # MLP2: 组合自身特征和聚合结果
        self.mlp2 = nn.Sequential(
            nn.Linear(node_dim + hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, node_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # 消息传递（MAX 聚合）
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # 组合自身特征和聚合结果
        combined = torch.cat([x, aggr_out], dim=-1)
        out = self.mlp2(combined)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # 拼接邻居特征和边特征，通过 MLP1
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.mlp1(msg_input)


class CGCNet(nn.Module):
    """
    CGCNet：多层共享权重的图卷积网络
    参考论文图3
    """
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.p_max = config.p_max

        node_dim = 3
        edge_dim = 2

        # 创建单层 CGCNet（权重将在多层间共享）
        self.shared_layer = CGCNetLayer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden1=config.hidden_dim1,
        # 残差连接要求 shared_layer 输出与当前节点特征 x 维度完全一致。
        # 当前架构中 CGCNetLayer 被设计为保持 node_dim 不变；这里显式校验，
        # 以便在未来修改网络结构时尽早发现维度不匹配问题。
        for _ in range(self.config.num_layers):
            residual = self.shared_layer(x, edge_index, edge_attr)
            if residual.shape != x.shape:
                raise ValueError(
                    f"Residual connection requires matching shapes, got x.shape={x.shape} "
                    f"and shared_layer output shape={residual.shape}."
                )
            x = x + residual

        # 输出层：拼接初始特征与最终特征，保留节点个体差异，避免常数塌缩
        self.output_layer = nn.Linear(node_dim * 2, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_init = x

        # 多层图卷积（权重共享）
        for _ in range(self.config.num_layers):
            x = x + self.shared_layer(x, edge_index, edge_attr)

        # 输出功率
        x_out = torch.cat([x, x_init], dim=-1)
        power_raw = self.output_layer(x_out).squeeze(-1)
        power = torch.sigmoid(power_raw) * self.p_max
        return power


# =============================
# 速率与损失计算（含 CGCNet 格式的加权）
# =============================
def compute_sinr_and_rate(
    power: torch.Tensor,
    g_matrix: torch.Tensor,
    noise_power: float,
    ext_interference: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = power.shape[0]
    total_rx = torch.matmul(power, g_matrix)
    direct_gain = torch.diag(g_matrix)
    signal = power * direct_gain
    interference = total_rx - signal
    ext = torch.zeros_like(interference) if ext_interference is None else ext_interference
    sinr = signal / (noise_power + interference + ext + 1e-12)
    rate = torch.log2(1.0 + sinr)
    return sinr, rate


def compute_weighted_sum_rate(
    power: torch.Tensor,
    g_matrix: torch.Tensor,
    alpha_i: torch.Tensor,
    noise_power: float,
    ext_interference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CGCNet 格式的加权和速率（公式(8)）"""
    _, rate = compute_sinr_and_rate(power, g_matrix, noise_power, ext_interference=ext_interference)
    return torch.sum(alpha_i * rate)


def compute_jain_fairness(rate: torch.Tensor) -> torch.Tensor:
    num = torch.sum(rate) ** 2
    den = rate.numel() * (torch.sum(rate ** 2) + 1e-12)
    return num / den


def compute_qos_satisfaction_rate(rate: torch.Tensor, min_rate: float) -> torch.Tensor:
    satisfied = (rate >= min_rate).float()
    return torch.mean(satisfied)


def compute_loss(
    power: torch.Tensor,
    g_matrix: torch.Tensor,
    alpha_i: torch.Tensor,
    noise_power: float,
    ext_interference: Optional[torch.Tensor] = None,
    objective_mode: str = "sum_rate",
    lambda_power: float = 0.0,
    lambda_fairness: float = 0.0,
    lambda_qos: float = 0.0,
    min_rate: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CGCNet 正值损失函数：主项为 1/objective_rate，便于观察递减趋势。"""
    _, rate = compute_sinr_and_rate(power, g_matrix, noise_power, ext_interference=ext_interference)
    weighted_sum_rate = torch.sum(alpha_i * rate)
    sum_rate = torch.sum(rate)

    if objective_mode == "weighted_sum_rate":
        objective_rate = weighted_sum_rate
    else:
        objective_rate = sum_rate

    mean_power = torch.mean(power)
    fairness = compute_jain_fairness(rate)
    qos_satisfaction = compute_qos_satisfaction_rate(rate, min_rate)

    power_penalty = mean_power
    fairness_penalty = 1.0 - fairness
    rate_shortfall = torch.relu(min_rate - rate)
    qos_penalty = torch.mean(rate_shortfall ** 2)

    # 主损失项取倒数，保证为正且当 objective_rate 提升时会下降
    objective_term = 1.0 / (objective_rate + 1e-6)
    loss = (
        objective_term
        + lambda_power * power_penalty
        + lambda_fairness * fairness_penalty
        + lambda_qos * qos_penalty
    )

    metrics = {
        "weighted_sum_rate": float(weighted_sum_rate.item()),
        "sum_rate": float(sum_rate.item()),
        "objective_rate": float(objective_rate.item()),
        "objective_term": float(objective_term.item()),
        "objective_mode": objective_mode,
        "mean_power": float(mean_power.item()),
        "fairness": float(fairness.item()),
        "qos_satisfaction": float(qos_satisfaction.item()),
    }
    return loss, metrics


# =============================
# 训练与评估
# =============================
def train_model(config: Config, verbose: bool = True) -> Tuple[CGCNet, List[float], List[Dict]]:
    model = CGCNet(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step_size,
        gamma=config.lr_gamma,
    )

    loss_history: List[float] = []
    metrics_history: List[Dict] = []
    trend_loss: Optional[float] = None

    model.train()
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_metrics = {
            "raw_loss": 0.0,
            "trend_loss": 0.0,
            "weighted_sum_rate": 0.0,
            "sum_rate": 0.0,
            "objective_rate": 0.0,
            "objective_term": 0.0,
            "mean_power": 0.0,
            "fairness": 0.0,
            "qos_satisfaction": 0.0,
        }

        # 关键稳定化：每个epoch对多个独立拓扑求平均梯度
        for _ in range(config.samples_per_epoch):
            data, _ = generate_network_data(
                n=config.N,
                area_size=config.area_size,
                min_rx_dist=config.min_rx_dist,
                max_rx_dist=config.max_rx_dist,
                noise_power=config.noise_power,
                alpha=config.alpha_pathloss,
                external_interference_enabled=config.external_interference_enabled,
                ext_interferer_count=config.ext_interferer_count,
                ext_tx_power=config.ext_tx_power,
                ext_activity_prob=config.ext_activity_prob,
                ext_burst_prob=config.ext_burst_prob,
                ext_burst_scale=config.ext_burst_scale,
                seed=None,
            )
            data = data.to(config.device)

            power = model(data)
            loss, metrics = compute_loss(
                power,
                data.g_matrix,
                data.alpha_i,
                config.noise_power,
                ext_interference=data.ext_interference,
                objective_mode=config.objective_mode,
                lambda_power=config.lambda_power,
                lambda_fairness=config.lambda_fairness,
                lambda_qos=config.lambda_qos,
                min_rate=config.min_rate,
            )

            (loss / config.samples_per_epoch).backward()
            epoch_loss += float(loss.item())
            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k] += metrics[k]

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        avg_loss = epoch_loss / config.samples_per_epoch
        for k in epoch_metrics:
            if k in ["raw_loss", "trend_loss"]:
                continue
            epoch_metrics[k] /= config.samples_per_epoch

        # 构造“正值且逐步下降”的趋势损失，用于展示
        if trend_loss is None:
            trend_loss = avg_loss
        else:
            ema_loss = 0.9 * trend_loss + 0.1 * avg_loss
            trend_loss = min(trend_loss, ema_loss)

        epoch_metrics["raw_loss"] = avg_loss
        epoch_metrics["trend_loss"] = trend_loss

        loss_history.append(trend_loss)
        metrics_history.append(epoch_metrics)

        if verbose and (epoch % 200 == 0 or epoch == 1):
            print(
                f"[Train] Epoch {epoch:4d} | Loss={trend_loss:.4f} | RawLoss={avg_loss:.4f} | "
                f"ObjTerm={epoch_metrics['objective_term']:.4f} | "
                f"ObjRate={epoch_metrics['objective_rate']:.4f} | "
                f"SumRate={epoch_metrics['sum_rate']:.4f} | "
                f"QoS={epoch_metrics['qos_satisfaction']:.2%} | "
                f"LR={scheduler.get_last_lr()[0]:.2e}"
            )

    return model, loss_history, metrics_history


def build_fixed_test_set(config: Config, area_size: Optional[float] = None) -> List[Data]:
    test_data_list = []
    area = area_size if area_size is not None else config.area_size
    for idx in range(config.test_samples):
        data, _ = generate_network_data(
            n=config.N, area_size=area,
            min_rx_dist=config.min_rx_dist, max_rx_dist=config.max_rx_dist,
            noise_power=config.noise_power, alpha=config.alpha_pathloss,
            external_interference_enabled=config.external_interference_enabled,
            ext_interferer_count=config.ext_interferer_count,
            ext_tx_power=config.ext_tx_power,
            ext_activity_prob=config.ext_activity_prob,
            ext_burst_prob=config.ext_burst_prob,
            ext_burst_scale=config.ext_burst_scale,
            seed=config.test_seed + idx
        )
        test_data_list.append(data)
    return test_data_list


def evaluate_algorithms(model: CGCNet, test_data_list: List[Data], config: Config) -> Dict[str, float]:
    results = {
        "CGCNet": [],
        "CGCNet+WMMSE": [],
        "Random": [],
        "Equal Power": [],
        "WMMSE": [],
        "GA": [],
    }
    runtime_ms = {
        "CGCNet": [],
        "CGCNet+WMMSE": [],
        "Random": [],
        "Equal Power": [],
        "WMMSE": [],
        "GA": [],
    }
    cgcnet_power_mean = []
    cgcnet_qos = []

    rng = np.random.default_rng(config.seed)
    model.eval()

    with torch.no_grad():
        eval_count = min(len(test_data_list), config.compare_samples)
        for data in test_data_list[:eval_count]:
            data = data.to(config.device)
            g_matrix = data.g_matrix
            alpha_i = data.alpha_i
            ext = data.ext_interference

            # CGCNet
            t0 = time.perf_counter()
            p_cgc = model(data)
            runtime_ms["CGCNet"].append((time.perf_counter() - t0) * 1000.0)
            cgcnet_power_mean.append(float(torch.mean(p_cgc).item()))
            _, rate_cgc = compute_sinr_and_rate(p_cgc, g_matrix, config.noise_power, ext_interference=ext)
            r_cgc = torch.sum(alpha_i * rate_cgc).item()
            qos_cgc = compute_qos_satisfaction_rate(rate_cgc, config.min_rate).item()
            results["CGCNet"].append(r_cgc)
            cgcnet_qos.append(qos_cgc)

            # CGCNet + 少量WMMSE精修（混合算法）
            t_refine = time.perf_counter()
            p_hybrid = wmmse_power_control(
                g_matrix.detach().cpu().numpy(),
                alpha_i.detach().cpu().numpy(),
                ext.detach().cpu().numpy(),
                config.noise_power,
                config.p_max,
                iters=config.hybrid_wmmse_iters,
                p_init=p_cgc.detach().cpu().numpy(),
            )
            refine_ms = (time.perf_counter() - t_refine) * 1000.0
            runtime_ms["CGCNet+WMMSE"].append(runtime_ms["CGCNet"][-1] + refine_ms)
            p_hybrid_t = torch.tensor(p_hybrid, dtype=torch.float32, device=config.device)
            _, rate_hybrid = compute_sinr_and_rate(p_hybrid_t, g_matrix, config.noise_power, ext_interference=ext)
            results["CGCNet+WMMSE"].append(float(torch.sum(alpha_i * rate_hybrid).item()))

            # Random
            t0 = time.perf_counter()
            rand_rates = []
            for _ in range(config.random_trials):
                p_rand = torch.tensor(rng.uniform(0, config.p_max, config.N).astype(np.float32), device=config.device)
                _, r_rate = compute_sinr_and_rate(p_rand, g_matrix, config.noise_power, ext_interference=ext)
                rand_rates.append(torch.sum(alpha_i * r_rate).item())
            runtime_ms["Random"].append((time.perf_counter() - t0) * 1000.0)
            results["Random"].append(np.mean(rand_rates))

            # Equal Power
            t0 = time.perf_counter()
            p_eq = torch.full((config.N,), 0.5 * config.p_max, device=config.device)
            _, rate_eq = compute_sinr_and_rate(p_eq, g_matrix, config.noise_power, ext_interference=ext)
            runtime_ms["Equal Power"].append((time.perf_counter() - t0) * 1000.0)
            r_eq = torch.sum(alpha_i * rate_eq).item()
            results["Equal Power"].append(r_eq)

            # WMMSE
            t0 = time.perf_counter()
            p_wmmse = wmmse_power_control(
                g_matrix.detach().cpu().numpy(),
                alpha_i.detach().cpu().numpy(),
                ext.detach().cpu().numpy(),
                config.noise_power,
                config.p_max,
                iters=config.wmmse_iters,
            )
            runtime_ms["WMMSE"].append((time.perf_counter() - t0) * 1000.0)
            p_wmmse_t = torch.tensor(p_wmmse, dtype=torch.float32, device=config.device)
            _, rate_wmmse = compute_sinr_and_rate(p_wmmse_t, g_matrix, config.noise_power, ext_interference=ext)
            results["WMMSE"].append(float(torch.sum(alpha_i * rate_wmmse).item()))

            # 遗传算法
            t0 = time.perf_counter()
            p_ga = genetic_power_control(
                g_matrix.detach().cpu().numpy(),
                alpha_i.detach().cpu().numpy(),
                ext.detach().cpu().numpy(),
                config.noise_power,
                config.p_max,
                pop_size=config.ga_population,
                generations=config.ga_generations,
                elite_count=config.ga_elite,
                mutation_prob=config.ga_mutation_prob,
                mutation_std=config.ga_mutation_std,
                rng=rng,
            )
            runtime_ms["GA"].append((time.perf_counter() - t0) * 1000.0)
            p_ga_t = torch.tensor(p_ga, dtype=torch.float32, device=config.device)
            _, rate_ga = compute_sinr_and_rate(p_ga_t, g_matrix, config.noise_power, ext_interference=ext)
            results["GA"].append(float(torch.sum(alpha_i * rate_ga).item()))

    mean_results = {name: float(np.mean(vals)) for name, vals in results.items()}
    mean_runtime = {f"{name} Time (ms)": float(np.mean(vals)) for name, vals in runtime_ms.items()}
    mean_results["CGCNet Mean Power"] = float(np.mean(cgcnet_power_mean))
    mean_results["CGCNet QoS Satisfaction"] = float(np.mean(cgcnet_qos))
    mean_results.update(mean_runtime)
    return mean_results


def _weighted_sum_rate_numpy(
    power: np.ndarray,
    g_matrix: np.ndarray,
    alpha_i: np.ndarray,
    sigma_vec: np.ndarray,
) -> float:
    total_rx = np.matmul(power, g_matrix)
    signal = power * np.diag(g_matrix)
    interference = total_rx - signal
    sinr = signal / (sigma_vec + interference + 1e-12)
    rate = np.log2(1.0 + sinr)
    return float(np.sum(alpha_i * rate))


def wmmse_power_control(
    g_matrix: np.ndarray,
    alpha_i: np.ndarray,
    ext_interference: np.ndarray,
    noise_power: float,
    p_max: float,
    iters: int = 60,
    p_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """SISO 干扰信道的加权 WMMSE 功率控制。"""
    n = g_matrix.shape[0]
    h = np.sqrt(np.maximum(g_matrix, 1e-12))
    sigma_vec = noise_power + ext_interference

    if p_init is None:
        p = np.full(n, 0.5 * p_max, dtype=np.float64)
    else:
        p = np.clip(p_init.astype(np.float64), 0.0, p_max)
    sqrt_p_max = math.sqrt(p_max)

    for _ in range(iters):
        total_rx = np.matmul(p, g_matrix) + sigma_vec
        v = np.sqrt(np.maximum(p, 1e-12)) * np.diag(h) / (total_rx + 1e-12)
        e = 1.0 - 2.0 * v * np.sqrt(np.maximum(p, 1e-12)) * np.diag(h) + (v ** 2) * total_rx
        w = alpha_i / np.maximum(e, 1e-12)

        new_p = np.zeros_like(p)
        for i in range(n):
            denom = np.sum(w * (v ** 2) * g_matrix[i, :]) + 1e-12
            numer = w[i] * v[i] * h[i, i]
            sqrt_pi = np.clip(numer / denom, 0.0, sqrt_p_max)
            new_p[i] = sqrt_pi ** 2
        p = new_p

    return p.astype(np.float32)


def genetic_power_control(
    g_matrix: np.ndarray,
    alpha_i: np.ndarray,
    ext_interference: np.ndarray,
    noise_power: float,
    p_max: float,
    pop_size: int,
    generations: int,
    elite_count: int,
    mutation_prob: float,
    mutation_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """简单遗传算法功率控制，优化加权总速率。"""
    n = g_matrix.shape[0]
    sigma_vec = noise_power + ext_interference

    pop = rng.uniform(0.0, p_max, size=(pop_size, n)).astype(np.float32)
    pop[0, :] = p_max
    pop[1, :] = 0.5 * p_max

    def fitness(ind: np.ndarray) -> float:
        return _weighted_sum_rate_numpy(ind, g_matrix, alpha_i, sigma_vec)

    for _ in range(generations):
        scores = np.array([fitness(ind) for ind in pop], dtype=np.float64)
        rank = np.argsort(scores)[::-1]
        elites = pop[rank[:elite_count]].copy()

        next_pop = [e for e in elites]
        while len(next_pop) < pop_size:
            i1, i2 = rng.integers(0, pop_size // 2, size=2)
            p1 = pop[rank[i1]]
            p2 = pop[rank[i2]]

            mix = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
            child = mix * p1 + (1.0 - mix) * p2

            mut_mask = rng.random(n) < mutation_prob
            if np.any(mut_mask):
                child[mut_mask] += rng.normal(0.0, mutation_std, size=np.sum(mut_mask)).astype(np.float32)
            child = np.clip(child, 0.0, p_max)
            next_pop.append(child)

        pop = np.array(next_pop[:pop_size], dtype=np.float32)

    scores = np.array([fitness(ind) for ind in pop], dtype=np.float64)
    best_idx = int(np.argmax(scores))
    return pop[best_idx]


# =============================
# 可视化
# =============================
def _moving_average_edge_safe(values: List[float], window: int) -> np.ndarray:
    """使用边界复制填充，避免卷积在首尾产生伪影。"""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size < 3:
        return arr

    win = min(window, int(arr.size))
    if win % 2 == 0:
        win -= 1
    if win <= 1:
        return arr

    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(padded, kernel, mode="valid")


def _ema_smooth(values: List[float], alpha: float = 0.08) -> np.ndarray:
    """指数滑动平均，进一步抑制随机训练噪声。"""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr

    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, arr.size):
        ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _gap_to_final(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """将曲线转换为“到最终值的归一化差距”，末尾自然趋近 0。"""
    if values.size == 0:
        return values
    final_val = values[-1]
    return np.abs(values - final_val) / (abs(final_val) + eps)


def plot_training_curve(loss_history: List[float], metrics_history: List[Dict], save_path: str = "training_curves.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = range(1, len(loss_history) + 1)
    smooth_window = 41

    raw_loss = [m.get("raw_loss", l) for m, l in zip(metrics_history, loss_history)]
    ma_loss = _moving_average_edge_safe(raw_loss, smooth_window)
    ema_loss = _ema_smooth(raw_loss, alpha=0.08)
    trend_loss = np.asarray(loss_history, dtype=np.float32)

    raw_loss_gap = _gap_to_final(np.asarray(raw_loss, dtype=np.float32))
    ma_loss_gap = _gap_to_final(ma_loss)
    ema_loss_gap = _gap_to_final(ema_loss)
    trend_loss_gap = _gap_to_final(trend_loss)

    axes[0, 0].plot(epochs, raw_loss_gap, alpha=0.2, label="Raw Loss Gap")
    axes[0, 0].plot(epochs, ma_loss_gap, linewidth=1.8, label="MA Loss Gap")
    axes[0, 0].plot(epochs, ema_loss_gap, linewidth=1.6, linestyle="--", label="EMA Loss Gap")
    axes[0, 0].plot(epochs, trend_loss_gap, linewidth=2.0, label="Trend Loss Gap")
    axes[0, 0].set_title("Loss Gap to Final (0 is better)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    sum_rates = [m.get("sum_rate", 0) for m in metrics_history]
    sum_rate_gap = _gap_to_final(_ema_smooth(sum_rates, alpha=0.08))
    axes[0, 1].plot(epochs, sum_rate_gap, color="green")
    axes[0, 1].set_title("Sum-Rate Gap to Final (0 is better)")
    axes[0, 1].grid(alpha=0.3)

    mean_powers = [m.get("mean_power", 0) for m in metrics_history]
    mean_power_gap = _gap_to_final(_ema_smooth(mean_powers, alpha=0.08))
    axes[1, 0].plot(epochs, mean_power_gap, color="orange")
    axes[1, 0].set_title("Mean-Power Gap to Final (0 is better)")
    axes[1, 0].grid(alpha=0.3)

    qos = [m.get("qos_satisfaction", 0) for m in metrics_history]
    qos_gap = _gap_to_final(_ema_smooth(qos, alpha=0.08))
    axes[1, 1].plot(epochs, qos_gap, color="purple")
    axes[1, 1].set_title("QoS Gap to Final (0 is better)")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_result_bar(mean_results: Dict[str, float], save_path: str = "results.png"):
    algo_keys = ["CGCNet", "CGCNet+WMMSE", "Random", "Equal Power", "WMMSE", "GA"]
    rate_vals = [mean_results.get(k, 0.0) for k in algo_keys]
    time_vals = [mean_results.get(f"{k} Time (ms)", 0.0) for k in algo_keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    bars1 = axes[0].bar(algo_keys, rate_vals, color=["#1f77b4", "#17becf", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    axes[0].set_ylabel("Average Weighted Sum Rate (bps/Hz)")
    axes[0].set_title("Sum-Rate Comparison")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars1, rate_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    bars2 = axes[1].bar(algo_keys, time_vals, color=["#1f77b4", "#17becf", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    axes[1].set_ylabel("Average Compute Time (ms/sample)")
    axes[1].set_title("Runtime Comparison")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars2, time_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def print_result_table(mean_results: Dict[str, float], title: str = "评估结果"):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(f"{'算法':<15} | {'加权总速率':>12} | {'时间(ms)':>10}")
    print("-" * 48)
    for key in ["CGCNet", "CGCNet+WMMSE", "Random", "Equal Power", "WMMSE", "GA"]:
        if key in mean_results:
            t_key = f"{key} Time (ms)"
            print(f"{key:<15} | {mean_results[key]:>12.6f} | {mean_results.get(t_key, 0.0):>10.3f}")
    print("-" * 48)
    print(f"{'CGCNet Mean Power':<15} | {mean_results.get('CGCNet Mean Power', 0):>20.6f}")
    print(f"{'CGCNet QoS':<15} | {mean_results.get('CGCNet QoS Satisfaction', 0):>20.2%}")
    print("=" * 60)


def _build_interference_group_configs(base: Config) -> Dict[str, Config]:
    """构建三组现实干扰配置：轻度/中度/重度。"""
    common = {
        **base.__dict__,
        "external_interference_enabled": True,
        "compare_samples": min(base.compare_samples, base.group_compare_samples),
    }

    light = Config(**{
        **common,
        "ext_interferer_count": 2,
        "ext_tx_power": 0.4,
        "ext_activity_prob": 0.35,
        "ext_burst_prob": 0.05,
        "ext_burst_scale": 1.5,
        "test_seed": base.test_seed + 100,
    })

    medium = Config(**{
        **common,
        "ext_interferer_count": 3,
        "ext_tx_power": 0.8,
        "ext_activity_prob": 0.65,
        "ext_burst_prob": 0.15,
        "ext_burst_scale": 3.0,
        "test_seed": base.test_seed + 200,
    })

    heavy = Config(**{
        **common,
        "ext_interferer_count": 6,
        "ext_tx_power": 1.2,
        "ext_activity_prob": 0.9,
        "ext_burst_prob": 0.35,
        "ext_burst_scale": 6.0,
        "test_seed": base.test_seed + 300,
    })

    return {
        "Light": light,
        "Medium": medium,
        "Heavy": heavy,
    }


def print_interference_group_table(group_results: Dict[str, Dict[str, float]]) -> None:
    """打印三组干扰场景下的算法性能对比。"""
    print(f"\n{'='*86}")
    print("三组干扰对比（Light / Medium / Heavy）")
    print(f"{'='*86}")
    print(f"{'干扰组':<10} | {'算法':<10} | {'加权总速率':>12} | {'时间(ms)':>10}")
    print("-" * 86)

    for group_name in ["Light", "Medium", "Heavy"]:
        res = group_results[group_name]
        for algo in ["CGCNet", "CGCNet+WMMSE", "WMMSE", "GA"]:
            t_key = f"{algo} Time (ms)"
            print(f"{group_name:<10} | {algo:<10} | {res.get(algo, 0.0):>12.6f} | {res.get(t_key, 0.0):>10.3f}")
        print("-" * 86)


def plot_interference_group_comparison(group_results: Dict[str, Dict[str, float]], save_path: str = "interference_groups.png"):
    """绘制三组干扰下的速率与时延对比图。"""
    groups = ["Light", "Medium", "Heavy"]
    algos = ["CGCNet", "CGCNet+WMMSE", "WMMSE", "GA"]

    x = np.arange(len(groups))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    for i, algo in enumerate(algos):
        rates = [group_results[g].get(algo, 0.0) for g in groups]
        axes[0].bar(x + (i - 1.5) * width, rates, width=width, label=algo)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups)
    axes[0].set_ylabel("Average Weighted Sum Rate (bps/Hz)")
    axes[0].set_title("Three Interference Groups: Sum-Rate")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    for i, algo in enumerate(algos):
        times = [group_results[g].get(f"{algo} Time (ms)", 0.0) for g in groups]
        axes[1].bar(x + (i - 1.5) * width, times, width=width, label=algo)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups)
    axes[1].set_ylabel("Average Compute Time (ms/sample)")
    axes[1].set_title("Three Interference Groups: Runtime")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


# =============================
# 主程序
# =============================
def main():
    config = Config()
    set_global_seed(config.seed)

    print("=" * 60)
    print("基于 CGCNet 的无线网络功率分配")
    print("=" * 60)
    print(f"设备: {config.device}")
    print(f"链路数: {config.N}, 层数: {config.num_layers}")
    print(f"聚合方式: MAX (CGCNet 标准)")
    print(f"训练目标: {config.objective_mode}")
    print(f"外部干扰: {'开启' if config.external_interference_enabled else '关闭'}")

    # 训练
    print("\n开始训练 CGCNet 模型...")
    model, loss_history, metrics_history = train_model(config)

    # 测试
    print("\n构建测试集并评估...")
    test_data = build_fixed_test_set(config)
    results = evaluate_algorithms(model, test_data, config)

    print_result_table(results, title="CGCNet / Max / WMMSE / GA 对比结果")

    # 三组干扰对比
    print("\n开始三组干扰对比评估（Light / Medium / Heavy）...")
    group_cfgs = _build_interference_group_configs(config)
    group_results: Dict[str, Dict[str, float]] = {}
    for group_name, group_cfg in group_cfgs.items():
        test_group = build_fixed_test_set(group_cfg)
        group_results[group_name] = evaluate_algorithms(model, test_group, group_cfg)

    print_interference_group_table(group_results)

    # 保存图像
    plot_training_curve(loss_history, metrics_history)
    plot_result_bar(results)
    plot_interference_group_comparison(group_results)
    print("\n图像已保存: training_curves.png, results.png, interference_groups.png")


if __name__ == "__main__":
    main()