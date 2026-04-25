from typing import Dict, Optional, Tuple

import numpy as np
import torch


def compute_sinr_and_rate(
    power: torch.Tensor,
    g_matrix: torch.Tensor,
    noise_power: float,
    ext_interference: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    _, rate = compute_sinr_and_rate(power, g_matrix, noise_power, ext_interference=ext_interference)
    return torch.sum(alpha_i * rate)


def compute_jain_fairness(rate: torch.Tensor) -> torch.Tensor:
    num = torch.sum(rate) ** 2
    den = rate.numel() * (torch.sum(rate ** 2) + 1e-12)
    return num / den


def compute_qos_satisfaction_rate(rate: torch.Tensor, min_rate: float) -> torch.Tensor:
    satisfied = (rate >= min_rate).float()
    return torch.mean(satisfied)


def _compute_link_ber_from_sinr(sinr: torch.Tensor) -> torch.Tensor:
    sinr_safe = torch.clamp(sinr, min=0.0)
    return 0.5 * torch.erfc(torch.sqrt(sinr_safe + 1e-12))


def _compute_thesis_metrics(
    power: torch.Tensor,
    sinr: torch.Tensor,
    rate: torch.Tensor,
    rate_threshold: float,
) -> Dict[str, float]:
    sum_rate = torch.sum(rate)
    avg_spectral_efficiency = torch.mean(rate)

    success_ratio = torch.mean((rate >= rate_threshold).float())
    outage_ratio = 1.0 - success_ratio
    min_rate = torch.min(rate)

    rate_np = rate.detach().cpu().numpy()
    p5_rate = float(np.percentile(rate_np, 5))

    sinr_db = 10.0 * torch.log10(sinr + 1e-12)
    avg_sinr_db = torch.mean(sinr_db)
    min_sinr_db = torch.min(sinr_db)

    ber = _compute_link_ber_from_sinr(sinr)
    avg_ber = torch.mean(ber)
    max_ber = torch.max(ber)

    energy_efficiency = sum_rate / (torch.sum(power) + 1e-12)
    jain_fairness = compute_jain_fairness(rate)

    return {
        "sum_rate": float(sum_rate.item()),
        "avg_spectral_efficiency": float(avg_spectral_efficiency.item()),
        "success_ratio": float(success_ratio.item()),
        "outage_ratio": float(outage_ratio.item()),
        "min_rate": float(min_rate.item()),
        "p5_rate": p5_rate,
        "avg_sinr_db": float(avg_sinr_db.item()),
        "min_sinr_db": float(min_sinr_db.item()),
        "avg_ber": float(avg_ber.item()),
        "max_ber": float(max_ber.item()),
        "energy_efficiency": float(energy_efficiency.item()),
        "jain_fairness": float(jain_fairness.item()),
    }


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
    lambda_sinr: float = 0.0,
    min_sinr_db: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    sinr, rate = compute_sinr_and_rate(power, g_matrix, noise_power, ext_interference=ext_interference)
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
    sinr_target = 10.0 ** (min_sinr_db / 10.0)
    sinr_shortfall = torch.relu(sinr_target - sinr)
    sinr_penalty = torch.mean(sinr_shortfall ** 2)

    objective_term = 1.0 / (objective_rate + 1e-6)
    loss = (
        objective_term
        + lambda_power * power_penalty
        + lambda_fairness * fairness_penalty
        + lambda_qos * qos_penalty
        + lambda_sinr * sinr_penalty
    )

    avg_sinr_db = 10.0 * torch.mean(torch.log10(sinr + 1e-12))

    metrics = {
        "weighted_sum_rate": float(weighted_sum_rate.item()),
        "sum_rate": float(sum_rate.item()),
        "objective_rate": float(objective_rate.item()),
        "objective_term": float(objective_term.item()),
        "objective_mode": objective_mode,
        "mean_power": float(mean_power.item()),
        "fairness": float(fairness.item()),
        "qos_satisfaction": float(qos_satisfaction.item()),
        "sinr_penalty": float(sinr_penalty.item()),
        "avg_sinr_db": float(avg_sinr_db.item()),
    }
    return loss, metrics
