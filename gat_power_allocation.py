"""Compatibility entry point.

This file keeps the original import path while the project is split into modules.
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
from module.models import CGCNet, CGCNetLayer
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

_build_interference_group_configs = build_interference_group_configs

__all__ = [
    "Config",
    "set_global_seed",
    "generate_network_data",
    "CGCNetLayer",
    "CGCNet",
    "compute_sinr_and_rate",
    "compute_weighted_sum_rate",
    "compute_jain_fairness",
    "compute_qos_satisfaction_rate",
    "_compute_link_ber_from_sinr",
    "_compute_thesis_metrics",
    "compute_loss",
    "train_model",
    "build_fixed_test_set",
    "evaluate_algorithms",
    "wmmse_power_control",
    "genetic_power_control",
    "_moving_average_edge_safe",
    "_ema_smooth",
    "_gap_to_final",
    "plot_training_curve",
    "plot_result_bar",
    "plot_thesis_metrics",
    "print_result_table",
    "_build_interference_group_configs",
    "build_interference_group_configs",
    "print_interference_group_table",
    "plot_interference_group_comparison",
    "main",
]


if __name__ == "__main__":
    main()
