import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Config:
    # System parameters
    N: int = 10
    area_size: float = 100.0
    min_rx_dist: float = 2.0
    max_rx_dist: float = 10.0
    p_max: float = 1.0
    noise_power: float = 1e-3
    alpha_pathloss: float = 2.0

    # Training parameters
    epochs: int = 2000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    samples_per_epoch: int = 8
    grad_clip_norm: float = 1.0
    lr_step_size: int = 500
    lr_gamma: float = 0.7

    # CGCNet parameters
    num_layers: int = 3
    hidden_dim1: int = 16
    hidden_dim2: int = 36
    hidden_dim3: int = 16

    # Loss regularization
    objective_mode: str = "sum_rate"
    lambda_power: float = 0.03
    lambda_fairness: float = 0.20
    lambda_qos: float = 0.5
    min_rate: float = 0.5
    lambda_sinr: float = 0.35
    min_sinr_db: float = 0.0

    # Test parameters
    test_samples: int = 200
    test_seed: int = 2026
    random_trials: int = 20
    compare_samples: int = 80
    group_compare_samples: int = 40

    # High-interference setting
    high_interference_area_size: float = 50.0

    # External interference model
    external_interference_enabled: bool = True
    ext_interferer_count: int = 3
    ext_tx_power: float = 0.8
    ext_activity_prob: float = 0.65
    ext_burst_prob: float = 0.15
    ext_burst_scale: float = 3.0

    # WMMSE parameters
    wmmse_iters: int = 60
    hybrid_wmmse_iters: int = 8

    # GA parameters
    ga_population: int = 36
    ga_generations: int = 45
    ga_elite: int = 6
    ga_mutation_prob: float = 0.15
    ga_mutation_std: float = 0.08

    # Runtime
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
