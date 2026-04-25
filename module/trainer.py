import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from module.baselines import genetic_power_control, wmmse_power_control
from module.config import Config
from module.data_utils import generate_network_data
from module.metrics import (
    _compute_thesis_metrics,
    compute_loss,
    compute_qos_satisfaction_rate,
    compute_sinr_and_rate,
)
from module.models import CGCNet


def train_model(config: Config, verbose: bool = True):
    model = CGCNet(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step_size,
        gamma=config.lr_gamma,
    )

    loss_history: List[float] = []
    metrics_history: List[Dict] = []
    trend_loss: Optional[float] = None

    model.train()
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_metrics = {
            "raw_loss": 0.0,
            "trend_loss": 0.0,
            "weighted_sum_rate": 0.0,
            "sum_rate": 0.0,
            "objective_rate": 0.0,
            "objective_term": 0.0,
            "mean_power": 0.0,
            "fairness": 0.0,
            "qos_satisfaction": 0.0,
            "sinr_penalty": 0.0,
            "avg_sinr_db": 0.0,
        }

        for _ in range(config.samples_per_epoch):
            data, _ = generate_network_data(
                n=config.N,
                area_size=config.area_size,
                min_rx_dist=config.min_rx_dist,
                max_rx_dist=config.max_rx_dist,
                noise_power=config.noise_power,
                alpha=config.alpha_pathloss,
                external_interference_enabled=config.external_interference_enabled,
                ext_interferer_count=config.ext_interferer_count,
                ext_tx_power=config.ext_tx_power,
                ext_activity_prob=config.ext_activity_prob,
                ext_burst_prob=config.ext_burst_prob,
                ext_burst_scale=config.ext_burst_scale,
                seed=None,
            )
            data = data.to(config.device)

            power = model(data)
            loss, metrics = compute_loss(
                power,
                data.g_matrix,
                data.alpha_i,
                config.noise_power,
                ext_interference=data.ext_interference,
                objective_mode=config.objective_mode,
                lambda_power=config.lambda_power,
                lambda_fairness=config.lambda_fairness,
                lambda_qos=config.lambda_qos,
                min_rate=config.min_rate,
                lambda_sinr=config.lambda_sinr,
                min_sinr_db=config.min_sinr_db,
            )

            (loss / config.samples_per_epoch).backward()
            epoch_loss += float(loss.item())
            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k] += metrics[k]

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        avg_loss = epoch_loss / config.samples_per_epoch
        for k in epoch_metrics:
            if k in ["raw_loss", "trend_loss"]:
                continue
            epoch_metrics[k] /= config.samples_per_epoch

        if trend_loss is None:
            trend_loss = avg_loss
        else:
            ema_loss = 0.9 * trend_loss + 0.1 * avg_loss
            trend_loss = min(trend_loss, ema_loss)

        epoch_metrics["raw_loss"] = avg_loss
        epoch_metrics["trend_loss"] = trend_loss

        loss_history.append(trend_loss)
        metrics_history.append(epoch_metrics)

        if verbose and (epoch % 200 == 0 or epoch == 1):
            print(
                f"[Train] Epoch {epoch:4d} | Loss={trend_loss:.4f} | RawLoss={avg_loss:.4f} | "
                f"ObjTerm={epoch_metrics['objective_term']:.4f} | "
                f"ObjRate={epoch_metrics['objective_rate']:.4f} | "
                f"SumRate={epoch_metrics['sum_rate']:.4f} | "
                f"QoS={epoch_metrics['qos_satisfaction']:.2%} | "
                f"LR={scheduler.get_last_lr()[0]:.2e}"
            )

    return model, loss_history, metrics_history


def build_fixed_test_set(config: Config, area_size: Optional[float] = None) -> List[Data]:
    test_data_list = []
    area = area_size if area_size is not None else config.area_size
    for idx in range(config.test_samples):
        data, _ = generate_network_data(
            n=config.N,
            area_size=area,
            min_rx_dist=config.min_rx_dist,
            max_rx_dist=config.max_rx_dist,
            noise_power=config.noise_power,
            alpha=config.alpha_pathloss,
            external_interference_enabled=config.external_interference_enabled,
            ext_interferer_count=config.ext_interferer_count,
            ext_tx_power=config.ext_tx_power,
            ext_activity_prob=config.ext_activity_prob,
            ext_burst_prob=config.ext_burst_prob,
            ext_burst_scale=config.ext_burst_scale,
            seed=config.test_seed + idx,
        )
        test_data_list.append(data)
    return test_data_list


def evaluate_algorithms(model: CGCNet, test_data_list: List[Data], config: Config) -> Dict[str, float]:
    results = {
        "CGCNet": [],
        "CGCNet+WMMSE": [],
        "Random": [],
        "Equal Power": [],
        "WMMSE": [],
        "GA": [],
    }
    runtime_ms = {
        "CGCNet": [],
        "CGCNet+WMMSE": [],
        "Random": [],
        "Equal Power": [],
        "WMMSE": [],
        "GA": [],
    }
    cgcnet_power_mean = []
    cgcnet_qos = []
    thesis_metrics_acc = {
        "sum_rate": [],
        "avg_spectral_efficiency": [],
        "success_ratio": [],
        "outage_ratio": [],
        "min_rate": [],
        "p5_rate": [],
        "avg_sinr_db": [],
        "min_sinr_db": [],
        "avg_ber": [],
        "max_ber": [],
        "energy_efficiency": [],
        "jain_fairness": [],
    }
    algo_thesis_metrics_acc = {
        name: {
            "success_ratio": [],
            "avg_sinr_db": [],
            "avg_ber": [],
        }
        for name in results.keys()
    }

    rng = np.random.default_rng(config.seed)
    model.eval()

    with torch.no_grad():
        eval_count = min(len(test_data_list), config.compare_samples)
        for data in test_data_list[:eval_count]:
            data = data.to(config.device)
            g_matrix = data.g_matrix
            alpha_i = data.alpha_i
            ext = data.ext_interference

            t0 = time.perf_counter()
            p_cgc = model(data)
            runtime_ms["CGCNet"].append((time.perf_counter() - t0) * 1000.0)
            cgcnet_power_mean.append(float(torch.mean(p_cgc).item()))
            sinr_cgc, rate_cgc = compute_sinr_and_rate(p_cgc, g_matrix, config.noise_power, ext_interference=ext)
            r_cgc = torch.sum(alpha_i * rate_cgc).item()
            qos_cgc = compute_qos_satisfaction_rate(rate_cgc, config.min_rate).item()
            results["CGCNet"].append(r_cgc)
            cgcnet_qos.append(qos_cgc)

            tm = _compute_thesis_metrics(power=p_cgc, sinr=sinr_cgc, rate=rate_cgc, rate_threshold=config.min_rate)
            for k in thesis_metrics_acc:
                thesis_metrics_acc[k].append(tm[k])
            for k in algo_thesis_metrics_acc["CGCNet"]:
                algo_thesis_metrics_acc["CGCNet"][k].append(tm[k])

            t_refine = time.perf_counter()
            p_hybrid = wmmse_power_control(
                g_matrix.detach().cpu().numpy(),
                alpha_i.detach().cpu().numpy(),
                ext.detach().cpu().numpy(),
                config.noise_power,
                config.p_max,
                iters=config.hybrid_wmmse_iters,
                p_init=p_cgc.detach().cpu().numpy(),
            )
            refine_ms = (time.perf_counter() - t_refine) * 1000.0
            runtime_ms["CGCNet+WMMSE"].append(runtime_ms["CGCNet"][-1] + refine_ms)
            p_hybrid_t = torch.tensor(p_hybrid, dtype=torch.float32, device=config.device)
            sinr_hybrid, rate_hybrid = compute_sinr_and_rate(p_hybrid_t, g_matrix, config.noise_power, ext_interference=ext)
            results["CGCNet+WMMSE"].append(float(torch.sum(alpha_i * rate_hybrid).item()))
            tm_hybrid = _compute_thesis_metrics(power=p_hybrid_t, sinr=sinr_hybrid, rate=rate_hybrid, rate_threshold=config.min_rate)
            for k in algo_thesis_metrics_acc["CGCNet+WMMSE"]:
                algo_thesis_metrics_acc["CGCNet+WMMSE"][k].append(tm_hybrid[k])

            t0 = time.perf_counter()
            rand_rates = []
            rand_tms = []
            for _ in range(config.random_trials):
                p_rand = torch.tensor(rng.uniform(0, config.p_max, config.N).astype(np.float32), device=config.device)
                sinr_rand, r_rate = compute_sinr_and_rate(p_rand, g_matrix, config.noise_power, ext_interference=ext)
                rand_rates.append(torch.sum(alpha_i * r_rate).item())
                rand_tms.append(_compute_thesis_metrics(power=p_rand, sinr=sinr_rand, rate=r_rate, rate_threshold=config.min_rate))
            runtime_ms["Random"].append((time.perf_counter() - t0) * 1000.0)
            results["Random"].append(np.mean(rand_rates))
            for k in algo_thesis_metrics_acc["Random"]:
                algo_thesis_metrics_acc["Random"][k].append(float(np.mean([tm[k] for tm in rand_tms])))

            t0 = time.perf_counter()
            p_eq = torch.full((config.N,), 0.5 * config.p_max, device=config.device)
            sinr_eq, rate_eq = compute_sinr_and_rate(p_eq, g_matrix, config.noise_power, ext_interference=ext)
            runtime_ms["Equal Power"].append((time.perf_counter() - t0) * 1000.0)
            r_eq = torch.sum(alpha_i * rate_eq).item()
            results["Equal Power"].append(r_eq)
            tm_eq = _compute_thesis_metrics(power=p_eq, sinr=sinr_eq, rate=rate_eq, rate_threshold=config.min_rate)
            for k in algo_thesis_metrics_acc["Equal Power"]:
                algo_thesis_metrics_acc["Equal Power"][k].append(tm_eq[k])

            t0 = time.perf_counter()
            p_wmmse = wmmse_power_control(
                g_matrix.detach().cpu().numpy(),
                alpha_i.detach().cpu().numpy(),
                ext.detach().cpu().numpy(),
                config.noise_power,
                config.p_max,
                iters=config.wmmse_iters,
            )
            runtime_ms["WMMSE"].append((time.perf_counter() - t0) * 1000.0)
            p_wmmse_t = torch.tensor(p_wmmse, dtype=torch.float32, device=config.device)
            sinr_wmmse, rate_wmmse = compute_sinr_and_rate(p_wmmse_t, g_matrix, config.noise_power, ext_interference=ext)
            results["WMMSE"].append(float(torch.sum(alpha_i * rate_wmmse).item()))
            tm_wmmse = _compute_thesis_metrics(power=p_wmmse_t, sinr=sinr_wmmse, rate=rate_wmmse, rate_threshold=config.min_rate)
            for k in algo_thesis_metrics_acc["WMMSE"]:
                algo_thesis_metrics_acc["WMMSE"][k].append(tm_wmmse[k])

            t0 = time.perf_counter()
            p_ga = genetic_power_control(
                g_matrix.detach().cpu().numpy(),
                alpha_i.detach().cpu().numpy(),
                ext.detach().cpu().numpy(),
                config.noise_power,
                config.p_max,
                pop_size=config.ga_population,
                generations=config.ga_generations,
                elite_count=config.ga_elite,
                mutation_prob=config.ga_mutation_prob,
                mutation_std=config.ga_mutation_std,
                rng=rng,
            )
            runtime_ms["GA"].append((time.perf_counter() - t0) * 1000.0)
            p_ga_t = torch.tensor(p_ga, dtype=torch.float32, device=config.device)
            sinr_ga, rate_ga = compute_sinr_and_rate(p_ga_t, g_matrix, config.noise_power, ext_interference=ext)
            results["GA"].append(float(torch.sum(alpha_i * rate_ga).item()))
            tm_ga = _compute_thesis_metrics(power=p_ga_t, sinr=sinr_ga, rate=rate_ga, rate_threshold=config.min_rate)
            for k in algo_thesis_metrics_acc["GA"]:
                algo_thesis_metrics_acc["GA"][k].append(tm_ga[k])

    mean_results = {name: float(np.mean(vals)) for name, vals in results.items()}
    mean_runtime = {f"{name} Time (ms)": float(np.mean(vals)) for name, vals in runtime_ms.items()}
    mean_results["CGCNet Mean Power"] = float(np.mean(cgcnet_power_mean))
    mean_results["CGCNet QoS Satisfaction"] = float(np.mean(cgcnet_qos))
    for k, vals in thesis_metrics_acc.items():
        mean_results[k] = float(np.mean(vals))
    for algo_name, metric_map in algo_thesis_metrics_acc.items():
        for metric_name, vals in metric_map.items():
            mean_results[f"{algo_name} {metric_name}"] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    mean_results.update(mean_runtime)
    return mean_results
