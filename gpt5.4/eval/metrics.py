from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


def sinr_and_rates(
    channel_matrix: torch.Tensor,
    power: torch.Tensor,
    noise_power: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = channel_matrix.device
    num_links = channel_matrix.size(0)
    eye = torch.eye(num_links, device=device)

    signal = torch.diag(channel_matrix) * power
    interference_matrix = channel_matrix * (1.0 - eye)
    interference = interference_matrix @ power
    sinr = signal / (interference + noise_power + 1e-12)
    rates = torch.log2(1.0 + sinr)
    return sinr, rates


def jain_fairness(values: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum(values) ** 2
    denominator = values.numel() * (torch.sum(values ** 2) + 1e-12)
    return numerator / denominator


def approximate_ber_from_sinr(sinr: torch.Tensor) -> torch.Tensor:
    # Uncoded BPSK BER approximation: Pb = 0.5 * erfc(sqrt(SINR))
    return 0.5 * torch.erfc(torch.sqrt(torch.clamp(sinr, min=1e-12)))


def compute_metrics(
    channel_matrix: torch.Tensor,
    power: torch.Tensor,
    noise_power: float,
    rate_threshold: float,
) -> Dict[str, float]:
    sinr, rates = sinr_and_rates(channel_matrix, power, noise_power)
    ber = approximate_ber_from_sinr(sinr)
    rates_np = rates.detach().cpu().numpy()
    sinr_db = 10.0 * torch.log10(torch.clamp(sinr, min=1e-12))
    total_power = torch.sum(power) + 1e-12

    sum_rate = rates.sum()
    avg_rate = rates.mean()

    return {
        "sum_rate": float(sum_rate.item()),
        "spectral_efficiency": float(sum_rate.item()),
        "avg_spectral_efficiency": float(avg_rate.item()),
        "avg_rate": float(avg_rate.item()),
        "min_rate": float(rates.min().item()),
        "p5_rate": float(np.percentile(rates_np, 5)),
        "outage_ratio": float((rates < rate_threshold).float().mean().item()),
        "success_ratio": float((rates >= rate_threshold).float().mean().item()),
        "avg_sinr_db": float(sinr_db.mean().item()),
        "min_sinr_db": float(sinr_db.min().item()),
        "avg_ber": float(ber.mean().item()),
        "max_ber": float(ber.max().item()),
        "energy_efficiency": float((sum_rate / total_power).item()),
        "jain_fairness": float(jain_fairness(rates).item()),
        "mean_power": float(power.mean().item()),
    }


def average_metrics(metrics_list: Iterable[Dict[str, float]]) -> Dict[str, float]:
    metrics_list = list(metrics_list)
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {key: float(np.mean([item[key] for item in metrics_list])) for key in keys}


def add_improvement_metrics(results: Dict[str, Dict[str, float]], ours_key: str = "GAT (Ours)") -> Dict[str, Dict[str, float]]:
    if ours_key not in results:
        return results

    ours_sum_rate = results[ours_key]["sum_rate"]
    enriched: Dict[str, Dict[str, float]] = {}
    for method, metrics in results.items():
        metrics_copy = dict(metrics)
        if method == ours_key:
            metrics_copy["improvement_vs_self_pct"] = 0.0
        else:
            base = max(metrics["sum_rate"], 1e-12)
            metrics_copy["improvement_vs_ours_pct"] = (ours_sum_rate - base) / base * 100.0
        enriched[method] = metrics_copy
    return enriched


def format_result_table(results: Dict[str, Dict[str, float]]) -> str:
    header = (
        "Method".ljust(16)
        + "SumRate".rjust(12)
        + "AvgSE".rjust(10)
        + "AvgSINR(dB)".rjust(13)
        + "AvgBER".rjust(12)
        + "QoS".rjust(8)
        + "Fair".rjust(8)
        + "EE".rjust(10)
    )
    lines: List[str] = [header, "-" * len(header)]
    for method, metric in results.items():
        lines.append(
            method.ljust(16)
            + f"{metric['sum_rate']:>12.4f}"
            + f"{metric['avg_spectral_efficiency']:>10.4f}"
            + f"{metric['avg_sinr_db']:>13.4f}"
            + f"{metric['avg_ber']:>12.4e}"
            + f"{metric['success_ratio']:>8.4f}"
            + f"{metric['jain_fairness']:>8.4f}"
            + f"{metric['energy_efficiency']:>10.4f}"
        )
    return "\n".join(lines)


def format_improvement_summary(results: Dict[str, Dict[str, float]], ours_key: str = "GAT (Ours)") -> str:
    if ours_key not in results:
        return ""

    ours = results[ours_key]
    lines = ["Improvement of GAT over baselines (Sum-Rate / Spectral Efficiency):"]
    for method, metric in results.items():
        if method == ours_key:
            continue
        base = max(metric["sum_rate"], 1e-12)
        improvement = (ours["sum_rate"] - base) / base * 100.0
        lines.append(f"- vs {method}: {improvement:.2f}%")
    return "\n".join(lines)
