from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training_and_results(
    history: Dict[str, List[float]],
    results: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    methods = list(results.keys())
    sum_rates = [results[name]["sum_rate"] for name in methods]
    success_rates = [results[name]["success_ratio"] for name in methods]
    avg_bers = [results[name]["avg_ber"] for name in methods]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#c0392b", linewidth=1.5)
    axes[0].plot(history["val_sum_rate"], label="Val Sum-Rate", color="#2980b9", linewidth=1.5)
    axes[0].set_title("Training Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    bars = axes[1].bar(methods, sum_rates, color=["#34495e", "#7f8c8d", "#27ae60", "#8e44ad", "#d35400"])
    axes[1].set_title("Spectral Efficiency Comparison")
    axes[1].set_ylabel("Sum-Rate / bps/Hz")
    axes[1].tick_params(axis="x", rotation=15)
    axes1_twin = axes[1].twinx()
    axes1_twin.plot(methods, success_rates, color="#e74c3c", marker="o", linewidth=1.8, label="QoS Success")
    axes1_twin.set_ylabel("QoS Success Ratio")
    axes1_twin.set_ylim(0.0, 1.05)
    for bar, value in zip(bars, sum_rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    axes[2].bar(methods, avg_bers, color=["#1f77b4", "#95a5a6", "#2ecc71", "#9b59b6", "#e67e22"])
    axes[2].set_title("Average BER Comparison")
    axes[2].set_ylabel("BER")
    axes[2].set_yscale("log")
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
