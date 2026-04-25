from typing import Dict

from module.config import Config, set_global_seed
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

    print("=" * 60)
    print("CGCNet-based Wireless Power Allocation")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Links: {config.N}, Layers: {config.num_layers}")
    print("Aggregation: MAX")
    print(f"Training objective: {config.objective_mode}")
    print(f"External interference: {'ON' if config.external_interference_enabled else 'OFF'}")

    print("\nTraining model...")
    model, loss_history, metrics_history = train_model(config)

    print("\nEvaluating on fixed test set...")
    test_data = build_fixed_test_set(config)
    results = evaluate_algorithms(model, test_data, config)

    print_result_table(results, title="CGCNet / Hybrid / Baselines")

    print("\nEvaluating three interference groups (Light/Medium/Heavy)...")
    group_cfgs = build_interference_group_configs(config)
    group_results: Dict[str, Dict[str, float]] = {}
    for group_name, group_cfg in group_cfgs.items():
        test_group = build_fixed_test_set(group_cfg)
        group_results[group_name] = evaluate_algorithms(model, test_group, group_cfg)

    print_interference_group_table(group_results)

    plot_training_curve(loss_history, metrics_history)
    plot_result_bar(results)
    plot_thesis_metrics(results)
    plot_interference_group_comparison(group_results)
    print("\nSaved images: training_curves.png, results.png, thesis_metrics.png, interference_groups.png")


if __name__ == "__main__":
    main()
