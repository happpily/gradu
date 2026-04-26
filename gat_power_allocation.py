"""
gat_power_allocation.py — 主程序入口
=======================================
基于图注意力网络（GAT）的无线网络功率分配系统
Wireless Power Allocation via Graph Attention Network (GAT)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
论文课题：基于图神经网络的无线网络功率分配研究
面向专业：通信工程 / 电子信息工程

研究内容：
    本程序实现了一套基于图注意力网络（GAT）的端到端无线网络功率
    分配优化系统，主要包括：

    1. 网络信道建模
       · Rayleigh 衰落 + 路径损耗综合信道模型
       · 外部干扰建模（持续干扰 + 突发干扰）
       · 图结构表示：节点=链路对，有向边=干扰关系

    2. GATNet 模型设计
       · 多头图注意力机制：自适应权衡不同干扰源的影响
       · 残差连接 + Layer Normalization：稳定深层网络训练
       · 端到端无监督训练：以频谱效率为优化目标

    3. 对比实验（满足论文 3 组以上对比实验要求）
       · 6 种算法对比：GATNet / GATNet+WMMSE / Random / Equal / WMMSE / GA
       · 3 组干扰场景：轻度 / 中度 / 重度外部干扰
       · 5 类性能指标：和速率 / SINR / BER / QoS / 能量效率

模块结构：
    module/
    ├── config.py       — 超参数配置
    ├── data_utils.py   — 随机网络数据生成
    ├── models.py       — GATNet / CGCNet 模型定义
    ├── metrics.py      — 性能指标与损失函数
    ├── baselines.py    — WMMSE / GA 基线算法
    ├── trainer.py      — 训练与评估流程
    ├── visualization.py— 实验结果可视化
    └── pipeline.py     — 主实验流水线

运行方式：
    python gat_power_allocation.py
    或：python -m module.pipeline

依赖：
    torch, torch-geometric, numpy, matplotlib
    （见 requirements.txt）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── 公共接口（向后兼容旧导入路径）────────────────────────────────────────
# Public API (backward-compatible import path)
from module.baselines import genetic_power_control, wmmse_power_control
from module.config import Config, set_global_seed
from module.data_utils import generate_network_data
from module.metrics import (
    _compute_link_ber_from_sinr,
    _compute_thesis_metrics,
    compute_jain_fairness,
    compute_loss,
    compute_qos_satisfaction_rate,
    compute_sinr_and_rate,
    compute_weighted_sum_rate,
)
from module.models import CGCNet, CGCNetLayer, GATNet
from module.pipeline import main
from module.trainer import build_fixed_test_set, evaluate_algorithms, train_model
from module.visualization import (
    _ema_smooth,
    _gap_to_final,
    _moving_average_edge_safe,
    build_interference_group_configs,
    plot_interference_group_comparison,
    plot_result_bar,
    plot_thesis_metrics,
    plot_training_curve,
    print_interference_group_table,
    print_result_table,
)

# 别名，保持旧版兼容性
_build_interference_group_configs = build_interference_group_configs

__all__ = [
    "Config", "set_global_seed",
    "generate_network_data",
    # 模型（GATNet 为主方案）
    "GATNet", "CGCNet", "CGCNetLayer",
    # 指标
    "compute_sinr_and_rate", "compute_weighted_sum_rate",
    "compute_jain_fairness", "compute_qos_satisfaction_rate",
    "_compute_link_ber_from_sinr", "_compute_thesis_metrics",
    "compute_loss",
    # 训练
    "train_model", "build_fixed_test_set", "evaluate_algorithms",
    # 基线
    "wmmse_power_control", "genetic_power_control",
    # 可视化
    "_moving_average_edge_safe", "_ema_smooth", "_gap_to_final",
    "plot_training_curve", "plot_result_bar", "plot_thesis_metrics",
    "print_result_table", "build_interference_group_configs",
    "_build_interference_group_configs",
    "print_interference_group_table", "plot_interference_group_comparison",
    "main",
]


if __name__ == "__main__":
    main()
