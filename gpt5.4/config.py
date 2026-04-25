from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # Wireless network parameters
    num_links: int = 10
    area_size: float = 100.0
    d_min: float = 2.0
    d_max: float = 10.0
    pathloss_exp: float = 2.0
    noise_power: float = 1e-3
    p_max: float = 1.0

    # Training and evaluation
    rate_threshold: float = 1.0
    train_epochs: int = 2000
    val_size: int = 100
    test_size: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    qos_lambda: float = 0.2
    seed: int = 42

    @property
    def node_feature_dim(self) -> int:
        return 4

    @property
    def edge_feature_dim(self) -> int:
        return 2
