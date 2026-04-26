"""
metrics.py — Performance Metrics and Loss Functions
=====================================================
Single-sample and BATCHED metric/loss computation.

CRITICAL FIX in compute_loss_batched:
    WRONG: torch.bmm(g, p.unsqueeze(-1))
           → result[b,j] = sum_i G[b,j,i]*p[b,i]   (sums over RX — wrong)
    RIGHT: torch.bmm(p.unsqueeze(1), g).squeeze(1)
           → result[b,i] = sum_j p[b,j]*G[b,j,i]   (sums over TX — correct)

    G[b, j, i] = gain from TX_j to RX_i (row=TX, col=RX convention).
    Total received power at RX_i = sum_j p_j * G[j,i], i.e. p @ G, not G @ p.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Single-sample (used during evaluation)
# ═══════════════════════════════════════════════════════════════════════════

def compute_sinr_and_rate(
    power: torch.Tensor,
    g_matrix: torch.Tensor,
    noise_power: float,
    ext_interference: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-sample SINR and Shannon rate.

    SINR_i = p_i*G_ii / (sigma^2 + sum_{j!=i} p_j*G_{j,i} + ext_i)
    Rate_i = log2(1 + SINR_i)  [bps/Hz]

    Args:
        power          : [N]
        g_matrix       : [N, N],  G[j, i] = TX_j → RX_i gain
        noise_power    : scalar
        ext_interference: [N] optional

    Returns:
        sinr [N], rate [N]
    """
    total_rx    = torch.matmul(power, g_matrix)   # [N]  total_rx[i]=sum_j p_j*G[j,i]
    direct_gain = torch.diag(g_matrix)             # [N]
    signal      = power * direct_gain              # [N]
    interference= total_rx - signal                # [N]
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
    _, rate = compute_sinr_and_rate(power, g_matrix, noise_power, ext_interference)
    return torch.sum(alpha_i * rate)


def compute_jain_fairness(rate: torch.Tensor) -> torch.Tensor:
    """Jain's Fairness Index = (sum R_i)^2 / (N * sum R_i^2)."""
    num = torch.sum(rate) ** 2
    den = rate.numel() * (torch.sum(rate ** 2) + 1e-12)
    return num / den


def compute_qos_satisfaction_rate(rate: torch.Tensor, min_rate: float) -> torch.Tensor:
    """Fraction of links with rate >= min_rate."""
    return torch.mean((rate >= min_rate).float())


def _compute_link_ber_from_sinr(sinr: torch.Tensor) -> torch.Tensor:
    """Approximate BPSK BER: 0.5 * erfc(sqrt(SINR))."""
    return 0.5 * torch.erfc(torch.sqrt(torch.clamp(sinr, min=0.0) + 1e-12))


def _compute_thesis_metrics(
    power: torch.Tensor,
    sinr: torch.Tensor,
    rate: torch.Tensor,
    rate_threshold: float,
) -> Dict[str, float]:
    """Full thesis metric dictionary for one evaluation sample."""
    sum_rate               = torch.sum(rate)
    avg_spectral_efficiency= torch.mean(rate)
    success_ratio          = torch.mean((rate >= rate_threshold).float())
    outage_ratio           = 1.0 - success_ratio
    min_rate_val           = torch.min(rate)
    p5_rate                = float(np.percentile(rate.detach().cpu().numpy(), 5))
    sinr_db                = 10.0 * torch.log10(sinr + 1e-12)
    avg_sinr_db            = torch.mean(sinr_db)
    min_sinr_db            = torch.min(sinr_db)
    ber                    = _compute_link_ber_from_sinr(sinr)
    avg_ber                = torch.mean(ber)
    max_ber                = torch.max(ber)
    energy_efficiency      = sum_rate / (torch.sum(power) + 1e-12)
    jain_fairness          = compute_jain_fairness(rate)
    return {
        "sum_rate":               float(sum_rate.item()),
        "avg_spectral_efficiency":float(avg_spectral_efficiency.item()),
        "success_ratio":          float(success_ratio.item()),
        "outage_ratio":           float(outage_ratio.item()),
        "min_rate":               float(min_rate_val.item()),
        "p5_rate":                p5_rate,
        "avg_sinr_db":            float(avg_sinr_db.item()),
        "min_sinr_db":            float(min_sinr_db.item()),
        "avg_ber":                float(avg_ber.item()),
        "max_ber":                float(max_ber.item()),
        "energy_efficiency":      float(energy_efficiency.item()),
        "jain_fairness":          float(jain_fairness.item()),
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
    """Single-sample multi-objective loss (kept for compatibility)."""
    sinr, rate        = compute_sinr_and_rate(power, g_matrix, noise_power, ext_interference)
    weighted_sum_rate = torch.sum(alpha_i * rate)
    sum_rate          = torch.sum(rate)
    objective_rate    = weighted_sum_rate if objective_mode == "weighted_sum_rate" else sum_rate

    mean_power     = torch.mean(power)
    fairness       = compute_jain_fairness(rate)
    qos_sat        = compute_qos_satisfaction_rate(rate, min_rate)
    qos_penalty    = torch.mean(torch.relu(min_rate - rate) ** 2)
    sinr_target    = 10.0 ** (min_sinr_db / 10.0)
    sinr_penalty   = torch.mean(torch.relu(sinr_target - sinr) ** 2)
    objective_term = 1.0 / (objective_rate + 1e-6)

    loss = (objective_term
            + lambda_power    * mean_power
            + lambda_fairness * (1.0 - fairness)
            + lambda_qos      * qos_penalty
            + lambda_sinr     * sinr_penalty)

    avg_sinr_db = 10.0 * torch.mean(torch.log10(sinr + 1e-12))
    return loss, {
        "weighted_sum_rate": float(weighted_sum_rate.item()),
        "sum_rate":          float(sum_rate.item()),
        "objective_rate":    float(objective_rate.item()),
        "objective_term":    float(objective_term.item()),
        "objective_mode":    objective_mode,
        "mean_power":        float(mean_power.item()),
        "fairness":          float(fairness.item()),
        "qos_satisfaction":  float(qos_sat.item()),
        "sinr_penalty":      float(sinr_penalty.item()),
        "avg_sinr_db":       float(avg_sinr_db.item()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# BATCHED variants — GPU-efficient training
# ═══════════════════════════════════════════════════════════════════════════

def compute_loss_batched(power, g, alpha, noise_power, ext_interference,
                         objective_mode, lambda_power, lambda_fairness,
                         lambda_qos, min_rate, lambda_sinr, min_sinr_db):
    """Batched loss for GPU parallel execution."""
    B, N, _ = g.shape
    p = power.view(B, N)
    
    # ✅ 修正矩阵方向：p @ g，得到每个接收机收到的总功率
    total_received = torch.bmm(p.unsqueeze(1), g).squeeze(1)  # [B, N]
    
    g_diag = torch.diagonal(g, dim1=1, dim2=2)  # [B, N]
    signal = g_diag * p  # [B, N]
    interference = total_received - signal + ext_interference + noise_power  # [B, N]
    
    sinr = signal / (interference + 1e-12)  # [B, N]
    rate = torch.log2(1.0 + sinr)  # [B, N]
    
    weighted_sum_rate = torch.sum(alpha * rate, dim=1)  # [B]
    sum_rate = torch.sum(rate, dim=1)  # [B]
    objective_rate = weighted_sum_rate.mean() if objective_mode == "weighted_sum_rate" else sum_rate.mean()
    
    power_penalty = torch.mean(p)
    
    sum_r = torch.sum(rate, dim=1)
    sum_r2 = torch.sum(rate ** 2, dim=1)
    jain_indices = (sum_r ** 2) / (N * sum_r2 + 1e-12)
    fairness_penalty = 1.0 - jain_indices.mean()
    
    qos_penalty = torch.mean(torch.relu(min_rate - rate) ** 2)
    
    sinr_target = 10.0 ** (min_sinr_db / 10.0)
    sinr_shortfall = torch.relu(sinr_target - sinr)
    sinr_penalty = torch.mean(sinr_shortfall ** 2)
    
    loss = (1.0 / (objective_rate + 1e-6)
            + lambda_power * power_penalty
            + lambda_fairness * fairness_penalty
            + lambda_qos * qos_penalty
            + lambda_sinr * sinr_penalty)
            
    avg_sinr_db = 10.0 * torch.mean(torch.log10(sinr + 1e-12))
    
    # ✅ 补全 metrics，确保可视化能取到数据
    metrics = {
        "weighted_sum_rate": float(weighted_sum_rate.mean().item()),
        "sum_rate": float(sum_rate.mean().item()),
        "mean_power": float(power_penalty.item()),          # ← 左下图需要
        "fairness": float(jain_indices.mean().item()),
        "qos_satisfaction": float(torch.mean((rate >= min_rate).float()).item()),
        "avg_sinr_db": float(avg_sinr_db.item()),           # ← 方便调试
    }
    return loss, metrics