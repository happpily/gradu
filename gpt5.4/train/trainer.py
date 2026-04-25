import argparse
import copy
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from delet.config import SimulationConfig
from data.simulator import generate_fixed_dataset, generate_network_data
from eval.baselines import (
    equal_power_allocation,
    max_power_allocation,
    random_power_allocation,
    wmmse_power_allocation,
)
from eval.metrics import (
    add_improvement_metrics,
    average_metrics,
    compute_metrics,
    format_improvement_summary,
    format_result_table,
)
from eval.visualization import plot_training_and_results
from model.gnn_model import GATPowerAllocator
from train.losses import unsupervised_loss
from utils import ensure_dir, get_numpy_rng, set_seed


def _evaluate_model(
    model: GATPowerAllocator,
    dataset: List[torch.Tensor],
    cfg: SimulationConfig,
    device: torch.device,
) -> Dict[str, float]:
    metrics = []
    model.eval()
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            power = model(data)
            metrics.append(compute_metrics(data.channel_matrix, power, cfg.noise_power, cfg.rate_threshold))
    return average_metrics(metrics)


def _full_evaluation(
    model: GATPowerAllocator,
    dataset: List[torch.Tensor],
    cfg: SimulationConfig,
    device: torch.device,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    results: Dict[str, List[Dict[str, float]]] = {
        "GAT (Ours)": [],
        "Max Power": [],
        "Equal Power": [],
        "Random": [],
        "WMMSE": [],
    }

    model.eval()
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            power_gat = model(data)
            results["GAT (Ours)"].append(
                compute_metrics(data.channel_matrix, power_gat, cfg.noise_power, cfg.rate_threshold)
            )

            power_max = max_power_allocation(cfg.num_links, cfg.p_max, device=device)
            results["Max Power"].append(
                compute_metrics(data.channel_matrix, power_max, cfg.noise_power, cfg.rate_threshold)
            )

            power_equal = equal_power_allocation(cfg.num_links, cfg.p_max, device=device)
            results["Equal Power"].append(
                compute_metrics(data.channel_matrix, power_equal, cfg.noise_power, cfg.rate_threshold)
            )

            power_random = random_power_allocation(cfg.num_links, cfg.p_max, rng, device=device)
            results["Random"].append(
                compute_metrics(data.channel_matrix, power_random, cfg.noise_power, cfg.rate_threshold)
            )

            power_wmmse = wmmse_power_allocation(data.channel_matrix, cfg.p_max, cfg.noise_power)
            results["WMMSE"].append(
                compute_metrics(data.channel_matrix, power_wmmse, cfg.noise_power, cfg.rate_threshold)
            )

    averaged = {name: average_metrics(metric_list) for name, metric_list in results.items()}
    return add_improvement_metrics(averaged)


def _scenario_configs(base_cfg: SimulationConfig) -> Dict[str, SimulationConfig]:
    return {
        "Light": SimulationConfig(
            **{**base_cfg.__dict__, "area_size": 120.0, "noise_power": base_cfg.noise_power * 0.8}
        ),
        "Medium": SimulationConfig(
            **{**base_cfg.__dict__, "area_size": base_cfg.area_size, "noise_power": base_cfg.noise_power}
        ),
        "Heavy": SimulationConfig(
            **{**base_cfg.__dict__, "area_size": 70.0, "noise_power": base_cfg.noise_power * 1.5}
        ),
    }


def _print_scenario_summary(
    model: GATPowerAllocator,
    base_cfg: SimulationConfig,
    device: torch.device,
    seed: int,
) -> None:
    print("\nScenario comparison (Light / Medium / Heavy):")
    print("Scenario".ljust(10) + "Method".ljust(16) + "SumRate".rjust(12) + "AvgBER".rjust(12) + "QoS".rjust(10))
    print("-" * 60)

    scenario_cfgs = _scenario_configs(base_cfg)
    for idx, (name, cfg) in enumerate(scenario_cfgs.items()):
        dataset = generate_fixed_dataset(min(base_cfg.test_size, 80), cfg, seed=seed + 100 + idx)
        results = _full_evaluation(model, dataset, cfg, device, seed=seed + 200 + idx)
        for method in ["GAT (Ours)", "WMMSE", "Equal Power", "Random"]:
            metric = results[method]
            print(
                name.ljust(10)
                + method.ljust(16)
                + f"{metric['sum_rate']:>12.4f}"
                + f"{metric['avg_ber']:>12.4e}"
                + f"{metric['success_ratio']:>10.4f}"
            )
        print("-" * 60)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    cfg = SimulationConfig(
        num_links=args.num_links,
        area_size=args.area_size,
        d_min=args.d_min,
        d_max=args.d_max,
        pathloss_exp=args.pathloss_exp,
        noise_power=args.noise_power,
        p_max=args.p_max,
        rate_threshold=args.rate_threshold,
        train_epochs=args.epochs,
        val_size=args.val_size,
        test_size=args.test_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        qos_lambda=args.qos_lambda,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ensure_dir(args.output_dir)

    train_rng = get_numpy_rng(args.seed)
    val_dataset = generate_fixed_dataset(cfg.val_size, cfg, seed=args.seed + 1)
    test_dataset = generate_fixed_dataset(cfg.test_size, cfg, seed=args.seed + 2)

    model = GATPowerAllocator(
        node_dim=cfg.node_feature_dim,
        edge_dim=cfg.edge_feature_dim,
        p_max=cfg.p_max,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    history: Dict[str, List[float]] = {"train_loss": [], "train_sum_rate": [], "val_sum_rate": []}
    best_state = copy.deepcopy(model.state_dict())
    best_val_sum_rate = float("-inf")

    for epoch in range(1, cfg.train_epochs + 1):
        model.train()
        data, _ = generate_network_data(cfg.num_links, cfg=cfg, rng=train_rng)
        data = data.to(device)

        power = model(data)
        loss, metrics = unsupervised_loss(
            data=data,
            power=power,
            noise_power=cfg.noise_power,
            rate_threshold=cfg.rate_threshold,
            qos_lambda=cfg.qos_lambda,
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        history["train_loss"].append(metrics["loss"])
        history["train_sum_rate"].append(metrics["sum_rate"])

        if epoch % args.val_every == 0 or epoch == 1 or epoch == cfg.train_epochs:
            val_metrics = _evaluate_model(model, val_dataset, cfg, device)
            val_sum_rate = val_metrics["sum_rate"]
            if val_sum_rate > best_val_sum_rate:
                best_val_sum_rate = val_sum_rate
                best_state = copy.deepcopy(model.state_dict())
            current_val = val_sum_rate
        else:
            current_val = history["val_sum_rate"][-1] if history["val_sum_rate"] else metrics["sum_rate"]
        history["val_sum_rate"].append(current_val)

        if epoch % args.log_every == 0 or epoch == 1 or epoch == cfg.train_epochs:
            print(
                f"Epoch {epoch:04d}/{cfg.train_epochs} | "
                f"Loss {metrics['loss']:.4f} | "
                f"Train Sum-Rate {metrics['sum_rate']:.4f} | "
                f"Train BER {metrics['avg_ber']:.4e} | "
                f"Train QoS {metrics['success_ratio']:.4f} | "
                f"Val Sum-Rate {current_val:.4f}"
            )

    model.load_state_dict(best_state)

    checkpoint_path = os.path.join(args.output_dir, "best_gat_power_allocator.pt")
    torch.save(model.state_dict(), checkpoint_path)

    final_results = _full_evaluation(model, test_dataset, cfg, device, seed=args.seed + 3)
    print("\nAverage performance on the fixed test set:")
    print(format_result_table(final_results))
    print()
    print(format_improvement_summary(final_results))

    print("\nMetrics covered for the thesis requirement:")
    print("- Spectral efficiency: `sum_rate`, `avg_spectral_efficiency`")
    print("- Reliability/QoS: `success_ratio`, `outage_ratio`, `min_rate`, `p5_rate`")
    print("- Communication quality: `avg_sinr_db`, `min_sinr_db`, `avg_ber`, `max_ber`")
    print("- Resource usage: `mean_power`, `energy_efficiency`")
    print("- Fairness: `jain_fairness`")

    _print_scenario_summary(model, cfg, device, seed=args.seed + 10)

    figure_path = os.path.join(args.output_dir, "power_allocation_results.png")
    plot_training_and_results(history, final_results, figure_path)

    print(f"\nModel checkpoint saved to: {checkpoint_path}")
    print(f"Result figure saved to: {figure_path}")
