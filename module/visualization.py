"""
visualization.py — Experimental Results Visualization
======================================================
Generates four thesis figures.

FIXES applied in this version:
  1. ALL text is ENGLISH ONLY — no Chinese characters anywhere.
     matplotlib.use("Agg") + DejaVu Sans eliminates box/square glyphs.
  2. ALGO_KEYS order: GATNet+WMMSE FIRST (proposed method).
  3. Training-curve subplots use correct epoch x-axis for all 4 panels.
"""

import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",       # built-in font, no CJK needed
    "axes.unicode_minus": False,               # correct minus-sign rendering
})

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

from module.config import Config

# ── Canonical order: proposed method (GATNet+WMMSE) FIRST ──────────────────
ALGO_KEYS   = ["GATNet+WMMSE", "GATNet", "WMMSE", "GA", "Equal Power", "Random"]
ALGO_COLORS = ["#e74c3c",      "#1f77b4", "#9467bd", "#8c564b", "#d62728", "#2ca02c"]
PROPOSED    = "GATNet+WMMSE"


# ═══════════════════════════════════════════════════════════════════════════
# Smoothing utilities
# ═══════════════════════════════════════════════════════════════════════════

def _moving_average_edge_safe(values: List[float], window: int) -> np.ndarray:
    """Edge-safe moving average with symmetric padding."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    win = min(max(window, 1), int(arr.size))
    if win % 2 == 0:
        win = max(win - 1, 1)
    if win <= 1:
        return arr
    padded = np.pad(arr, (win // 2, win // 2), mode="edge")
    return np.convolve(padded, np.ones(win, dtype=np.float32) / win, mode="valid")


def _ema_smooth(values: List[float], alpha: float = 0.08) -> np.ndarray:
    """Exponential Moving Average smoothing."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    ema    = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, arr.size):
        ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _gap_to_final(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalised gap to final value: |v_t - v_final| / (|v_final| + eps).
    Returns array of same length; all values >= 0."""
    if values.size == 0:
        return values
    fv = values[-1]
    return np.abs(values - fv) / (abs(fv) + eps)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Training convergence
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curve(
    loss_history: List[float],
    metrics_history: List[Dict],
    save_path: str = "training_curves.png",
) -> None:
    """Four-panel convergence curves.  All axes use epoch (1 … T) as x-axis."""
    n_epochs = len(loss_history)
    if n_epochs == 0:
        print("  Warning: empty loss history, skipping training curve.")
        return

    epochs = np.arange(1, n_epochs + 1)          # x-axis for all 4 subplots

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # ── Title: pure English, no Chinese ──────────────────────────────────
    fig.suptitle(
        "GATNet Training Convergence  (Gap to Final Value, Lower is Better)",
        fontsize=12)

    # ── Collect series ────────────────────────────────────────────────────
    raw_loss    = np.asarray(
        [m.get("raw_loss", lv) for m, lv in zip(metrics_history, loss_history)],
        dtype=np.float32)
    trend_loss  = np.asarray(loss_history, dtype=np.float32)
    sum_rates   = np.asarray([m.get("sum_rate",   0.0) for m in metrics_history], dtype=np.float32)
    mean_powers = np.asarray([m.get("mean_power", 0.0) for m in metrics_history], dtype=np.float32)
    qos_vals    = np.asarray([m.get("qos_satisfaction", 0.0) for m in metrics_history], dtype=np.float32)

    # ── Top-left: Loss gap ────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, _gap_to_final(raw_loss),
            alpha=0.18, color="#aec6cf", label="Raw Loss Gap")
    ax.plot(epochs, _gap_to_final(_moving_average_edge_safe(raw_loss.tolist(), 41)),
            linewidth=1.8, color="#e67e22", label="MA Loss Gap")
    ax.plot(epochs, _gap_to_final(_ema_smooth(raw_loss.tolist())),
            linewidth=1.6, linestyle="--", color="#27ae60", label="EMA Loss Gap")
    ax.plot(epochs, _gap_to_final(trend_loss),
            linewidth=2.2, color="#c0392b", label="Trend Loss Gap")
    ax.set_title("Loss Gap to Final")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalised Gap")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim([1, n_epochs])

    # ── Top-right: Sum-Rate gap ───────────────────────────────────────────
    ax = axes[0, 1]
    gap_sr = _gap_to_final(_ema_smooth(sum_rates.tolist()))
    ax.plot(epochs, gap_sr, color="#2ecc71", linewidth=1.6)
    ax.set_title("Sum-Rate Gap to Final")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalised Gap")
    ax.grid(alpha=0.3)
    ax.set_xlim([1, n_epochs])
    ax.set_ylim(bottom=0)

    # ── Bottom-left: Mean-Power gap ───────────────────────────────────────
    ax = axes[1, 0]
    gap_mp = _gap_to_final(_ema_smooth(mean_powers.tolist()))
    ax.plot(epochs, gap_mp, color="#e67e22", linewidth=1.6)
    ax.set_title("Mean Power Gap to Final")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalised Gap")
    ax.grid(alpha=0.3)
    ax.set_xlim([1, n_epochs])
    ax.set_ylim(bottom=0)

    # ── Bottom-right: QoS gap ─────────────────────────────────────────────
    ax = axes[1, 1]
    gap_qos = _gap_to_final(_ema_smooth(qos_vals.tolist()))
    ax.plot(epochs, gap_qos, color="#9b59b6", linewidth=1.6)
    ax.set_title("QoS Satisfaction Gap to Final")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalised Gap")
    ax.grid(alpha=0.3)
    ax.set_xlim([1, n_epochs])
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Sum-rate & runtime bar chart
# ═══════════════════════════════════════════════════════════════════════════

def _mark_proposed(bars) -> None:
    """Give the first bar (proposed method) a thick red border."""
    bars[0].set_edgecolor("#c0392b")
    bars[0].set_linewidth(2.8)


def plot_result_bar(
    mean_results: Dict[str, float],
    save_path: str = "results.png",
) -> None:
    """Bar chart: sum-rate and runtime.  GATNet+WMMSE (proposed) is listed first."""
    rate_vals = [mean_results.get(k,               0.0) for k in ALGO_KEYS]
    time_vals = [mean_results.get(f"{k} Time (ms)", 0.0) for k in ALGO_KEYS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # ── Pure English suptitle ─────────────────────────────────────────────
    fig.suptitle(
        "Algorithm Comparison: Sum-Rate and Runtime\n"
        "[Proposed: GATNet+WMMSE  (shown first, red border)]",
        fontsize=11)

    # Left: Sum-Rate
    bars1 = axes[0].bar(ALGO_KEYS, rate_vals, color=ALGO_COLORS,
                        edgecolor="white", linewidth=0.8)
    _mark_proposed(bars1)
    axes[0].set_ylabel("Average Weighted Sum Rate (bps/Hz)")
    axes[0].set_title("Sum-Rate Comparison")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=18)
    for bar, val in zip(bars1, rate_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.15,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

    # Right: Runtime
    bars2 = axes[1].bar(ALGO_KEYS, time_vals, color=ALGO_COLORS,
                        edgecolor="white", linewidth=0.8)
    _mark_proposed(bars2)
    axes[1].set_ylabel("Average Compute Time (ms / sample)")
    axes[1].set_title("Runtime Comparison")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=18)
    for bar, val in zip(bars2, time_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Thesis metrics (SINR / BER / QoS)
# ═══════════════════════════════════════════════════════════════════════════

def plot_thesis_metrics(
    mean_results: Dict[str, float],
    save_path: str = "thesis_metrics.png",
) -> None:
    """Three-panel: Average SINR | Average BER | QoS Success Ratio."""
    sinr_vals = [mean_results.get(f"{k} avg_sinr_db",
                                  mean_results.get("avg_sinr_db", 0.0))
                 for k in ALGO_KEYS]
    ber_vals  = [mean_results.get(f"{k} avg_ber",
                                  mean_results.get("avg_ber",  0.0))
                 for k in ALGO_KEYS]
    qos_vals  = [mean_results.get(f"{k} success_ratio",
                                  mean_results.get("success_ratio", 0.0))
                 for k in ALGO_KEYS]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    # ── Pure English suptitle ─────────────────────────────────────────────
    fig.suptitle(
        "Thesis Metrics Comparison\n"
        "[Proposed: GATNet+WMMSE  (shown first, red border)]",
        fontsize=11)

    # Left: SINR
    bars1 = axes[0].bar(ALGO_KEYS, sinr_vals, color=ALGO_COLORS,
                        edgecolor="white", linewidth=0.8)
    _mark_proposed(bars1)
    axes[0].set_title("Average SINR (dB)   [Higher is Better]")
    axes[0].set_ylabel("dB")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for bar, val in zip(bars1, sinr_vals):
        va     = "bottom" if val >= 0 else "top"
        offset = (abs(val) * 0.04 + 0.3) * (1 if val >= 0 else -1)
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     val + offset, f"{val:.2f}",
                     ha="center", va=va, fontsize=8)

    # Middle: BER log scale
    ber_plot = [max(v, 1e-7) for v in ber_vals]
    bars2 = axes[1].bar(ALGO_KEYS, ber_plot, color=ALGO_COLORS,
                        edgecolor="white", linewidth=0.8)
    _mark_proposed(bars2)
    axes[1].set_title("Average BER   [Lower is Better]")
    axes[1].set_ylabel("BER")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, ber_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.3,
                     f"{val:.2e}", ha="center", va="bottom", fontsize=8)

    # Right: QoS
    bars3 = axes[2].bar(ALGO_KEYS, qos_vals, color=ALGO_COLORS,
                        edgecolor="white", linewidth=0.8)
    _mark_proposed(bars3)
    axes[2].set_title("QoS Success Ratio   [Higher is Better]")
    axes[2].set_ylabel("Ratio")
    axes[2].set_ylim(0.0, 1.12)
    axes[2].tick_params(axis="x", rotation=18)
    axes[2].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, qos_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     val + 0.015, f"{val:.3f}",
                     ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Console result tables
# ═══════════════════════════════════════════════════════════════════════════

def print_result_table(mean_results: Dict[str, float],
                       title: str = "Evaluation Results") -> None:
    print(f"\n{'='*65}\n{title}\n{'='*65}")
    print(f"{'Algorithm':<16} | {'Weighted Sum Rate':>18} | {'Time(ms)':>10}")
    print("-" * 54)
    for key in ALGO_KEYS:
        if key in mean_results:
            t      = mean_results.get(f"{key} Time (ms)", 0.0)
            marker = "  <-- Proposed" if key == PROPOSED else ""
            print(f"{key:<16} | {mean_results[key]:>18.6f} | {t:>10.3f}{marker}")
    print("-" * 54)
    print("Thesis Metrics (GATNet primary):")
    for key, label in [
        ("sum_rate",               "Sum Rate (bps/Hz)"),
        ("avg_spectral_efficiency","Avg SE (bps/Hz)"),
        ("success_ratio",          "QoS Success Ratio"),
        ("outage_ratio",           "Outage Ratio"),
        ("min_rate",               "Min Rate"),
        ("p5_rate",                "P5 Rate"),
        ("avg_sinr_db",            "Avg SINR (dB)"),
        ("min_sinr_db",            "Min SINR (dB)"),
        ("avg_ber",                "Avg BER"),
        ("max_ber",                "Max BER"),
        ("energy_efficiency",      "Energy Efficiency"),
        ("jain_fairness",          "Jain Fairness"),
    ]:
        val = mean_results.get(key, 0.0)
        fmt = f"{val:.4e}" if key in ("avg_ber", "max_ber") else f"{val:.6f}"
        print(f"  {label:<28}: {fmt}")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════
# Interference group comparison
# ═══════════════════════════════════════════════════════════════════════════

def build_interference_group_configs(base: Config) -> Dict[str, Config]:
    """Build Light / Medium / Heavy interference scenario configs."""
    common = {
        **base.__dict__,
        "external_interference_enabled": True,
        "compare_samples": min(base.compare_samples, base.group_compare_samples),
    }
    light  = Config(**{**common, "ext_interferer_count": 2, "ext_tx_power": 0.4,
                       "ext_activity_prob": 0.35, "ext_burst_prob": 0.05,
                       "ext_burst_scale": 1.5,  "test_seed": base.test_seed + 100})
    medium = Config(**{**common, "ext_interferer_count": 3, "ext_tx_power": 0.8,
                       "ext_activity_prob": 0.65, "ext_burst_prob": 0.15,
                       "ext_burst_scale": 3.0,  "test_seed": base.test_seed + 200})
    heavy  = Config(**{**common, "ext_interferer_count": 6, "ext_tx_power": 1.2,
                       "ext_activity_prob": 0.90, "ext_burst_prob": 0.35,
                       "ext_burst_scale": 6.0,  "test_seed": base.test_seed + 300})
    return {"Light": light, "Medium": medium, "Heavy": heavy}


def print_interference_group_table(group_results: Dict[str, Dict[str, float]]) -> None:
    print(f"\n{'='*72}")
    print("Interference Scenario Comparison  (Light / Medium / Heavy)")
    print(f"{'='*72}")
    print(f"{'Scenario':<10} | {'Algorithm':<16} | {'Sum-Rate':>12} | {'Time(ms)':>10}")
    print("-" * 56)
    for g in ["Light", "Medium", "Heavy"]:
        res = group_results[g]
        for algo in ["GATNet+WMMSE", "GATNet", "WMMSE", "GA"]:
            t_key = f"{algo} Time (ms)"
            print(f"{g:<10} | {algo:<16} | "
                  f"{res.get(algo, 0.0):>12.6f} | {res.get(t_key, 0.0):>10.3f}")
        print("-" * 56)


def plot_interference_group_comparison(
    group_results: Dict[str, Dict[str, float]],
    save_path: str = "interference_groups.png",
) -> None:
    """Grouped bar chart: Sum-Rate and Runtime across Light/Medium/Heavy."""
    groups      = ["Light", "Medium", "Heavy"]
    show_algos  = ["GATNet+WMMSE", "GATNet", "WMMSE", "GA"]
    show_colors = ["#e74c3c", "#1f77b4", "#9467bd", "#8c564b"]

    x     = np.arange(len(groups))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    # ── Pure English suptitle ─────────────────────────────────────────────
    fig.suptitle(
        "Three Interference Scenario Comparison\n"
        "[Proposed: GATNet+WMMSE  (shown first, red border)]",
        fontsize=11)

    for i, (algo, color) in enumerate(zip(show_algos, show_colors)):
        rates = [group_results[g].get(algo, 0.0) for g in groups]
        bars  = axes[0].bar(x + (i - 1.5) * width, rates,
                            width=width, label=algo, color=color)
        if i == 0:
            for b in bars:
                b.set_edgecolor("#c0392b"); b.set_linewidth(2.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Light", "Medium", "Heavy"])
    axes[0].set_ylabel("Average Weighted Sum Rate (bps/Hz)")
    axes[0].set_title("Sum-Rate vs Interference Level")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)

    for i, (algo, color) in enumerate(zip(show_algos, show_colors)):
        times = [group_results[g].get(f"{algo} Time (ms)", 0.0) for g in groups]
        bars  = axes[1].bar(x + (i - 1.5) * width, times,
                            width=width, label=algo, color=color)
        if i == 0:
            for b in bars:
                b.set_edgecolor("#c0392b"); b.set_linewidth(2.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Light", "Medium", "Heavy"])
    axes[1].set_ylabel("Average Compute Time (ms / sample)")
    axes[1].set_title("Runtime vs Interference Level")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")