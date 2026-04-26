"""
module/__init__.py — 模块公共接口
===================================
统一导出 module 包的所有公共符号，方便外部代码直接从 `module` 导入。

主要组件：
    · Config / set_global_seed   — 配置与随机种子
    · GATNet / CGCNet            — 图注意力网络 / 图卷积网络模型
    · generate_network_data      — 随机网络拓扑生成
    · compute_* / _compute_*     — 性能指标计算函数
    · compute_loss               — 训练损失函数
    · train_model                — 模型训练
    · build_fixed_test_set       — 固定测试集构建
    · evaluate_algorithms        — 多算法对比评估
    · wmmse_power_control        — WMMSE 基线算法
    · genetic_power_control      — 遗传算法基线
    · plot_* / print_*           — 可视化与结果打印
    · main                       — 主实验流水线入口
"""

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

__all__ = [
    # 配置
    "Config",
    "set_global_seed",
    # 数据生成
    "generate_network_data",
    # 模型（GATNet 为主要方案，CGCNet 保留为对比基线）
    "GATNet",
    "CGCNet",
    "CGCNetLayer",
    # 性能指标
    "compute_sinr_and_rate",
    "compute_weighted_sum_rate",
    "compute_jain_fairness",
    "compute_qos_satisfaction_rate",
    "_compute_link_ber_from_sinr",
    "_compute_thesis_metrics",
    "compute_loss",
    # 训练与评估
    "train_model",
    "build_fixed_test_set",
    "evaluate_algorithms",
    # 基线算法
    "wmmse_power_control",
    "genetic_power_control",
    # 可视化工具
    "_moving_average_edge_safe",
    "_ema_smooth",
    "_gap_to_final",
    "plot_training_curve",
    "plot_result_bar",
    "plot_thesis_metrics",
    "print_result_table",
    "build_interference_group_configs",
    "print_interference_group_table",
    "plot_interference_group_comparison",
    # 主流水线
    "main",
]
