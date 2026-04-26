"""
config.py — 全局配置模块
======================
集中管理所有超参数，包括：
  · 无线网络物理层参数（链路数、区域大小、最大功率等）
  · GATNet 模型结构参数（注意力头数、隐藏维度、层数等）
  · 训练优化参数（学习率、批量大小、正则化权重等）
  · 基线算法参数（WMMSE 迭代次数、遗传算法种群大小等）
  · 外部干扰模型参数（干扰源数量、活跃概率、突发干扰等）

使用方法：
    from module.config import Config, set_global_seed
    config = Config()           # 使用默认参数
    set_global_seed(config.seed)  # 固定随机种子，保证实验可复现
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Config:
    # ── 无线网络物理层参数 ──────────────────────────────────────────────────
    # Physical Layer Parameters
    N: int = 10                    # 发送-接收链路对数 (Number of TX-RX pairs)
    area_size: float = 100.0       # 网络部署区域边长 [m]  (Area side length in meters)
    min_rx_dist: float = 2.0       # 收发两端最小间距 [m]  (Min TX-RX separation)
    max_rx_dist: float = 10.0      # 收发两端最大间距 [m]  (Max TX-RX separation)
    p_max: float = 1.0             # 单链路最大发射功率 [W] (Max transmit power per link)
    noise_power: float = 1e-3      # 热噪声功率 [W]         (Thermal noise power)
    alpha_pathloss: float = 2.0    # 路径损耗指数            (Path loss exponent)

    # ── 训练超参数 ─────────────────────────────────────────────────────────
    # Training Hyperparameters
    epochs: int = 3000             # 总训练轮数              (Total training epochs)
    lr: float = 3e-4               # Adam 初始学习率          (Initial learning rate)
    weight_decay: float = 1e-4     # L2 权重衰减系数          (L2 weight decay)
    samples_per_epoch: int = 16    # 每轮随机网络拓扑采样数   (Random topologies per epoch)
    grad_clip_norm: float = 1.0    # 梯度裁剪范数上限         (Gradient clip max norm)
    lr_step_size: int = 600        # 学习率衰减步长 (epochs)  (LR scheduler step size)
    lr_gamma: float = 0.65         # 学习率衰减倍率           (LR decay factor)

    # ── GATNet 模型结构参数 ────────────────────────────────────────────────
    # GATNet Architecture Parameters
    # 注意力头数；多头注意力使模型从多角度建模信道状态信息
    gat_heads: int = 4             # Number of attention heads (多头注意力)
    # 每个注意力头的隐藏特征维度；总维度 = gat_heads × gat_head_dim
    gat_head_dim: int = 32         # Hidden dim per attention head
    # GAT 消息传递层数；更多层可捕获更远邻居的干扰关系
    gat_layers: int = 4            # Number of GAT message-passing layers
    # Dropout 防止过拟合（注意力权重上）
    gat_dropout: float = 0.10      # Dropout rate on attention coefficients
    # 输出 MLP 中间层维度
    gat_mlp_hidden: int = 64       # Hidden dim in output MLP

    # ── 保留 CGCNet 参数（作为对比基线） ──────────────────────────────────
    # Legacy CGCNet Parameters (kept for comparison baseline)
    num_layers: int = 3
    hidden_dim1: int = 16
    hidden_dim2: int = 36
    hidden_dim3: int = 16

    # ── 损失函数与正则化 ───────────────────────────────────────────────────
    # Loss Function & Regularization Weights
    # 优化目标模式：'sum_rate' 直接最大化加权和速率；
    #              'weighted_sum_rate' 使用距离反比权重 α_i
    objective_mode: str = "sum_rate"
    # 功率消耗惩罚权重（鼓励节能）
    lambda_power: float = 0.01     # Power consumption penalty weight
    # 公平性惩罚权重（减小 Jain 不公平指数）；
    # 已从 0.20 降低以减少对和速率的影响
    lambda_fairness: float = 0.05  # Fairness penalty weight (reduced from 0.20)
    # QoS 惩罚权重（惩罚低于最低速率门限的链路）；
    # 已从 0.50 降低以兼顾和速率与服务质量
    lambda_qos: float = 0.15       # QoS penalty weight (reduced from 0.50)
    min_rate: float = 0.5          # 最低速率门限 [bps/Hz]    (Min rate threshold)
    # SINR 惩罚权重（惩罚低于 SINR 门限的链路）
    lambda_sinr: float = 0.20      # SINR penalty weight
    min_sinr_db: float = 0.0       # 最低 SINR 门限 [dB]      (Min SINR threshold)

    # ── 测试与评估参数 ────────────────────────────────────────────────────
    # Test & Evaluation Parameters
    test_samples: int = 200        # 固定测试集样本数         (Fixed test set size)
    test_seed: int = 2026          # 测试集随机种子           (Test set seed)
    random_trials: int = 20        # 随机功率基线试验次数     (Random baseline trials)
    compare_samples: int = 80      # 实际参与对比评估的样本数 (Eval samples)
    group_compare_samples: int = 40 # 分组对比评估的样本数   (Group eval samples)

    # ── 高干扰场景参数 ────────────────────────────────────────────────────
    # High-Interference Scenario Parameters
    high_interference_area_size: float = 50.0  # 密集部署区域边长 (Dense deployment area)

    # ── 外部干扰模型参数 ──────────────────────────────────────────────────
    # External Interference Model Parameters
    # 建模来自网络外部的非协作干扰源（如邻区基站、WiFi 等）
    external_interference_enabled: bool = True  # 是否启用外部干扰 (Enable ext. interference)
    ext_interferer_count: int = 3               # 外部干扰源数量
    ext_tx_power: float = 0.8                   # 外部干扰源发射功率 [W]
    ext_activity_prob: float = 0.65             # 干扰源活跃概率
    ext_burst_prob: float = 0.15                # 突发干扰发生概率
    ext_burst_scale: float = 3.0                # 突发干扰强度倍数

    # ── WMMSE 基线算法参数 ────────────────────────────────────────────────
    # WMMSE Baseline Parameters
    # WMMSE 是经典的凸优化迭代算法，作为性能上界参考
    wmmse_iters: int = 60          # WMMSE 完整迭代次数       (Full WMMSE iterations)
    hybrid_wmmse_iters: int = 8    # 混合方案精调迭代次数     (Hybrid refinement iters)

    # ── 遗传算法基线参数 ──────────────────────────────────────────────────
    # Genetic Algorithm (GA) Baseline Parameters
    ga_population: int = 36        # 种群大小                 (Population size)
    ga_generations: int = 45       # 进化代数                 (Number of generations)
    ga_elite: int = 6              # 精英保留数量             (Elite individuals kept)
    ga_mutation_prob: float = 0.15 # 变异概率                 (Mutation probability)
    ga_mutation_std: float = 0.08  # 变异标准差               (Mutation std deviation)

    # ── 运行时参数 ────────────────────────────────────────────────────────
    # Runtime Parameters
    seed: int = 42                 # 全局随机种子             (Global random seed)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 运算设备


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        # 如果需要完全可复现（牺牲一点速度），可启用下一行并设 benchmark=False
        # torch.backends.cudnn.deterministic = True
