from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from module.config import Config


def _moving_average_edge_safe(values: List[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size < 3:
        return arr

    win = min(window, int(arr.size))
    if win % 2 == 0:
        win -= 1
    if win <= 1:
        return arr

    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(padded, kernel, mode="valid")


def _ema_smooth(values: List[float], alpha: float = 0.08) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr

    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, arr.size):
        ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _gap_to_final(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if values.size == 0:
        return values
    final_val = values[-1]
    return np.abs(values - final_val) / (abs(final_val) + eps)


def plot_training_curve(loss_history: List[float], metrics_history: List[Dict], save_path: str = "training_curves.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = range(1, len(loss_history) + 1)
    smooth_window = 41

    raw_loss = [m.get("raw_loss", l) for m, l in zip(metrics_history, loss_history)]
    ma_loss = _moving_average_edge_safe(raw_loss, smooth_window)
    ema_loss = _ema_smooth(raw_loss, alpha=0.08)
    trend_loss = np.asarray(loss_history, dtype=np.float32)

    raw_loss_gap = _gap_to_final(np.asarray(raw_loss, dtype=np.float32))
    ma_loss_gap = _gap_to_final(ma_loss)
    ema_loss_gap = _gap_to_final(ema_loss)
    trend_loss_gap = _gap_to_final(trend_loss)

    axes[0, 0].plot(epochs, raw_loss_gap, alpha=0.2, label="Raw Loss Gap")
    axes[0, 0].plot(epochs, ma_loss_gap, linewidth=1.8, label="MA Loss Gap")
    axes[0, 0].plot(epochs, ema_loss_gap, linewidth=1.6, linestyle="--", label="EMA Loss Gap")
    axes[0, 0].plot(epochs, trend_loss_gap, linewidth=2.0, label="Trend Loss Gap")
    axes[0, 0].set_title("Loss Gap to Final (0 is better)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    sum_rates = [m.get("sum_rate", 0) for m in metrics_history]
    sum_rate_gap = _gap_to_final(_ema_smooth(sum_rates, alpha=0.08))
    axes[0, 1].plot(epochs, sum_rate_gap, color="green")
    axes[0, 1].set_title("Sum-Rate Gap to Final (0 is better)")
    axes[0, 1].grid(alpha=0.3)

    mean_powers = [m.get("mean_power", 0) for m in metrics_history]
    mean_power_gap = _gap_to_final(_ema_smooth(mean_powers, alpha=0.08))
    axes[1, 0].plot(epochs, mean_power_gap, color="orange")
    axes[1, 0].set_title("Mean-Power Gap to Final (0 is better)")
    axes[1, 0].grid(alpha=0.3)

    qos = [m.get("qos_satisfaction", 0) for m in metrics_history]
    qos_gap = _gap_to_final(_ema_smooth(qos, alpha=0.08))
    axes[1, 1].plot(epochs, qos_gap, color="purple")
    axes[1, 1].set_title("QoS Gap to Final (0 is better)")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_result_bar(mean_results: Dict[str, float], save_path: str = "results.png"):
    algo_keys = ["CGCNet", "CGCNet+WMMSE", "Random", "Equal Power", "WMMSE", "GA"]
    rate_vals = [mean_results.get(k, 0.0) for k in algo_keys]
    time_vals = [mean_results.get(f"{k} Time (ms)", 0.0) for k in algo_keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    bars1 = axes[0].bar(algo_keys, rate_vals, color=["#1f77b4", "#17becf", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    axes[0].set_ylabel("Average Weighted Sum Rate (bps/Hz)")
    axes[0].set_title("Sum-Rate Comparison")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars1, rate_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    bars2 = axes[1].bar(algo_keys, time_vals, color=["#1f77b4", "#17becf", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    axes[1].set_ylabel("Average Compute Time (ms/sample)")
    axes[1].set_title("Runtime Comparison")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars2, time_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_thesis_metrics(mean_results: Dict[str, float], save_path: str = "thesis_metrics.png"):
    algo_keys = ["CGCNet", "CGCNet+WMMSE", "Random", "Equal Power", "WMMSE", "GA"]
    colors = ["#1f77b4", "#17becf", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    sinr_vals = [mean_results.get(f"{k} avg_sinr_db", mean_results.get("avg_sinr_db", 0.0)) for k in algo_keys]
    ber_vals = [mean_results.get(f"{k} avg_ber", mean_results.get("avg_ber", 0.0)) for k in algo_keys]
    qos_vals = [mean_results.get(f"{k} success_ratio", mean_results.get("success_ratio", 0.0)) for k in algo_keys]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    bars1 = axes[0].bar(algo_keys, sinr_vals, color=colors)
    axes[0].set_title("Average SINR (dB)")
    axes[0].set_ylabel("dB")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, sinr_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    bars2 = axes[1].bar(algo_keys, ber_vals, color=colors)
    axes[1].set_title("Average BER")
    axes[1].set_ylabel("BER")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, ber_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2e}", ha="center", va="bottom", fontsize=9)

    bars3 = axes[2].bar(algo_keys, qos_vals, color=colors)
    axes[2].set_title("QoS Success Ratio")
    axes[2].set_ylabel("Ratio")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, qos_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def print_result_table(mean_results: Dict[str, float], title: str = "Evaluation Results"):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(f"{'Algorithm':<15} | {'Sum-Rate':>12} | {'Time(ms)':>10}")
    print("-" * 48)
    for key in ["CGCNet", "CGCNet+WMMSE", "Random", "Equal Power", "WMMSE", "GA"]:
        if key in mean_results:
            t_key = f"{key} Time (ms)"
            print(f"{key:<15} | {mean_results[key]:>12.6f} | {mean_results.get(t_key, 0.0):>10.3f}")
    print("-" * 48)
    print(f"{'CGCNet Mean Power':<15} | {mean_results.get('CGCNet Mean Power', 0):>20.6f}")
    print(f"{'CGCNet QoS':<15} | {mean_results.get('CGCNet QoS Satisfaction', 0):>20.2%}")
    print("-" * 48)
    print("Thesis Metrics (CGCNet):")
    print(f"{'sum_rate':<20}: {mean_results.get('sum_rate', 0.0):.6f}")
    print(f"{'avg_spectral_efficiency':<20}: {mean_results.get('avg_spectral_efficiency', 0.0):.6f}")
    print(f"{'success_ratio':<20}: {mean_results.get('success_ratio', 0.0):.4f}")
    print(f"{'outage_ratio':<20}: {mean_results.get('outage_ratio', 0.0):.4f}")
    print(f"{'min_rate':<20}: {mean_results.get('min_rate', 0.0):.6f}")
    print(f"{'p5_rate':<20}: {mean_results.get('p5_rate', 0.0):.6f}")
    print(f"{'avg_sinr_db':<20}: {mean_results.get('avg_sinr_db', 0.0):.4f}")
    print(f"{'min_sinr_db':<20}: {mean_results.get('min_sinr_db', 0.0):.4f}")
    print(f"{'avg_ber':<20}: {mean_results.get('avg_ber', 0.0):.6e}")
    print(f"{'max_ber':<20}: {mean_results.get('max_ber', 0.0):.6e}")
    print(f"{'energy_efficiency':<20}: {mean_results.get('energy_efficiency', 0.0):.6f}")
    print(f"{'jain_fairness':<20}: {mean_results.get('jain_fairness', 0.0):.6f}")
    print("=" * 60)


def build_interference_group_configs(base: Config) -> Dict[str, Config]:
    common = {
        **base.__dict__,
        "external_interference_enabled": True,
        "compare_samples": min(base.compare_samples, base.group_compare_samples),
    }

    light = Config(**{
        **common,
        "ext_interferer_count": 2,
        "ext_tx_power": 0.4,
        "ext_activity_prob": 0.35,
        "ext_burst_prob": 0.05,
        "ext_burst_scale": 1.5,
        "test_seed": base.test_seed + 100,
    })

    medium = Config(**{
        **common,
        "ext_interferer_count": 3,
        "ext_tx_power": 0.8,
        "ext_activity_prob": 0.65,
        "ext_burst_prob": 0.15,
        "ext_burst_scale": 3.0,
        "test_seed": base.test_seed + 200,
    })

    heavy = Config(**{
        **common,
        "ext_interferer_count": 6,
        "ext_tx_power": 1.2,
        "ext_activity_prob": 0.9,
        "ext_burst_prob": 0.35,
        "ext_burst_scale": 6.0,
        "test_seed": base.test_seed + 300,
    })

    return {
        "Light": light,
        "Medium": medium,
        "Heavy": heavy,
    }


def print_interference_group_table(group_results: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{'='*86}")
    print("Three Interference Groups (Light / Medium / Heavy)")
    print(f"{'='*86}")
    print(f"{'Group':<10} | {'Algorithm':<12} | {'Sum-Rate':>12} | {'Time(ms)':>10}")
    print("-" * 86)

    for group_name in ["Light", "Medium", "Heavy"]:
        res = group_results[group_name]
        for algo in ["CGCNet", "CGCNet+WMMSE", "WMMSE", "GA"]:
            t_key = f"{algo} Time (ms)"
            print(f"{group_name:<10} | {algo:<12} | {res.get(algo, 0.0):>12.6f} | {res.get(t_key, 0.0):>10.3f}")
        print("-" * 86)


def plot_interference_group_comparison(group_results: Dict[str, Dict[str, float]], save_path: str = "interference_groups.png"):
    groups = ["Light", "Medium", "Heavy"]
    algos = ["CGCNet", "CGCNet+WMMSE", "WMMSE", "GA"]

    x = np.arange(len(groups))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    for i, algo in enumerate(algos):
        rates = [group_results[g].get(algo, 0.0) for g in groups]
        axes[0].bar(x + (i - 1.5) * width, rates, width=width, label=algo)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups)
    axes[0].set_ylabel("Average Weighted Sum Rate (bps/Hz)")
    axes[0].set_title("Three Interference Groups: Sum-Rate")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    for i, algo in enumerate(algos):
        times = [group_results[g].get(f"{algo} Time (ms)", 0.0) for g in groups]
        axes[1].bar(x + (i - 1.5) * width, times, width=width, label=algo)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups)
    axes[1].set_ylabel("Average Compute Time (ms/sample)")
    axes[1].set_title("Three Interference Groups: Runtime")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
