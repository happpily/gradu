"""
pipeline.py — Main Experiment Pipeline
=======================================
Orchestrates the full experiment:
    1. Train GATNet
    2. Evaluate 6 algorithms on fixed test set
    3. Run three interference scenario comparison experiments
    4. Save four thesis figures
"""

from typing import Dict

from module.config import Config, set_global_seed
from module.models import GATNet
from module.trainer import build_fixed_test_set, evaluate_algorithms, train_model
from module.visualization import (
    build_interference_group_configs,
    plot_interference_group_comparison,
    plot_result_bar,
    plot_thesis_metrics,
    plot_training_curve,
    print_interference_group_table,
    print_result_table,
)


def main() -> None:
    config = Config()
    set_global_seed(config.seed)

    print("=" * 65)
    print("  Wireless Power Allocation via Graph Attention Network (GAT)")
    print("=" * 65)
    print(f"  Device         : {config.device}")
    print(f"  Links (N)      : {config.N}")
    print(f"  GAT layers     : {config.gat_layers}")
    print(f"  Attention heads: {config.gat_heads}")
    print(f"  Head dim       : {config.gat_head_dim}")
    print(f"  Total hidden   : {config.gat_heads * config.gat_head_dim}")
    print(f"  Epochs         : {config.epochs}")
    print(f"  Batch per epoch: {config.samples_per_epoch}")
    print(f"  Objective      : {config.objective_mode}")
    print(f"  Ext. interference: {'ON' if config.external_interference_enabled else 'OFF'}")
    print("=" * 65)

    # 1. Train
    print("\n[1/4] Training GATNet (batched GPU mode)...")
    model, loss_history, metrics_history = train_model(
        config, model_class=GATNet, verbose=True)

    # 2. Evaluate
    print("\n[2/4] Evaluating all algorithms on fixed test set...")
    test_data = build_fixed_test_set(config)
    results   = evaluate_algorithms(model, test_data, config)
    print_result_table(results, title="GATNet+WMMSE vs. All Baselines")

    # 3. Interference scenario comparison
    print("\n[3/4] Three interference scenario experiments...")
    group_cfgs = build_interference_group_configs(config)
    group_results: Dict[str, Dict[str, float]] = {}
    for name, gcfg in group_cfgs.items():
        print(f"  Evaluating: {name} interference...")
        group_results[name] = evaluate_algorithms(model, build_fixed_test_set(gcfg), gcfg)
    print_interference_group_table(group_results)

    # 4. Plot
    print("\n[4/4] Saving thesis figures...")
    plot_training_curve(loss_history, metrics_history)
    plot_result_bar(results)
    plot_thesis_metrics(results)
    plot_interference_group_comparison(group_results)

    print("\nDone.  Figures saved:")
    print("  training_curves.png   — convergence curves")
    print("  results.png           — sum-rate & runtime")
    print("  thesis_metrics.png    — SINR / BER / QoS")
    print("  interference_groups.png — scenario comparison")


if __name__ == "__main__":
    main()