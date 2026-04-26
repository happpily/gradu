"""
trainer.py — GPU-Optimised Model Training and Evaluation
"""

import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from module.baselines import genetic_power_control, wmmse_power_control
from module.config import Config
from module.data_utils import generate_network_data
from module.metrics import (
    _compute_thesis_metrics,
    compute_loss,
    compute_loss_batched,  # 这里导入了 batched 函数
    compute_qos_satisfaction_rate,
    compute_sinr_and_rate,
)
from module.models import CGCNet, GATNet

def _stack_aux(data_list: List[Data], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g  = torch.stack([d.g_matrix        for d in data_list], dim=0).to(device)  # [B,N,N]
    a  = torch.stack([d.alpha_i         for d in data_list], dim=0).to(device)  # [B,N]
    ei = torch.stack([d.ext_interference for d in data_list], dim=0).to(device) # [B,N]
    return g, a, ei

def train_model(config: Config, model_class=GATNet, verbose: bool = True) -> Tuple[Union[GATNet, CGCNet], List[float], List[Dict]]:
    model = model_class(config).to(config.device)
    model_name  = model_class.__name__
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

    use_amp   = (config.device == "cuda")
    # 修复警告 1
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    if verbose:
        print(f"\n[Train] Model: {model_name}  |  Params: {total_params:,}  |  Device: {config.device}  |  AMP: {use_amp}")
        print(f"        Batch size per epoch: {config.samples_per_epoch} graphs")

    loss_history:    List[float] = []
    metrics_history: List[Dict]  = []
    trend_loss: Optional[float]  = None

    model.train()
    for epoch in range(1, config.epochs + 1):
        samples = []
        for _ in range(config.samples_per_epoch):
            d, _ = generate_network_data(
                n=config.N, area_size=config.area_size, min_rx_dist=config.min_rx_dist, max_rx_dist=config.max_rx_dist,
                noise_power=config.noise_power, alpha=config.alpha_pathloss,
                external_interference_enabled=config.external_interference_enabled,
                ext_interferer_count=config.ext_interferer_count, ext_tx_power=config.ext_tx_power,
                ext_activity_prob=config.ext_activity_prob, ext_burst_prob=config.ext_burst_prob,
                ext_burst_scale=config.ext_burst_scale, seed=None,
            )
            samples.append(d)

        batch = Batch.from_data_list(samples).to(config.device)
        B, N = config.samples_per_epoch, config.N
        g_mat, alpha, ext = _stack_aux(samples, torch.device(config.device))

        optimizer.zero_grad()

        # 修复警告 2
        with torch.amp.autocast('cuda', enabled=use_amp):
            power_flat = model(batch)                  # [B*N]
            
            # 使用导入的 compute_loss_batched
            loss, metrics = compute_loss_batched(
                power_flat, g_mat, alpha, config.noise_power, ext,
                config.objective_mode, config.lambda_power, config.lambda_fairness,
                config.lambda_qos, config.min_rate, config.lambda_sinr, config.min_sinr_db
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        avg_loss = float(loss.item())
        if trend_loss is None: trend_loss = avg_loss
        else: trend_loss = min(trend_loss, 0.9 * trend_loss + 0.1 * avg_loss)

        metrics["raw_loss"]   = avg_loss
        metrics["trend_loss"] = trend_loss
        loss_history.append(trend_loss)
        metrics_history.append(metrics)

        if verbose and (epoch % 200 == 0 or epoch == 1):
            print(f"[Train] Epoch {epoch:4d}/{config.epochs} | Loss={trend_loss:.4f} | SumRate={metrics['sum_rate']:.4f} | QoS={metrics['qos_satisfaction']:.2%} | LR={scheduler.get_last_lr()[0]:.2e}")

    return model, loss_history, metrics_history

def build_fixed_test_set(config: Config, area_size: Optional[float] = None) -> List[Data]:
    area = area_size if area_size is not None else config.area_size
    test_data_list = []
    for idx in range(config.test_samples):
        d, _ = generate_network_data(
            n=config.N, area_size=area, min_rx_dist=config.min_rx_dist, max_rx_dist=config.max_rx_dist,
            noise_power=config.noise_power, alpha=config.alpha_pathloss,
            external_interference_enabled=config.external_interference_enabled,
            ext_interferer_count=config.ext_interferer_count, ext_tx_power=config.ext_tx_power,
            ext_activity_prob=config.ext_activity_prob, ext_burst_prob=config.ext_burst_prob,
            ext_burst_scale=config.ext_burst_scale, seed=config.test_seed + idx,
        )
        test_data_list.append(d)
    return test_data_list

def evaluate_algorithms(model: Union[GATNet, CGCNet], test_data_list: List[Data], config: Config) -> Dict[str, float]:
    algo_names = ["GATNet+WMMSE", "GATNet", "WMMSE", "GA", "Equal Power", "Random"]
    results, runtime_ms = {n: [] for n in algo_names}, {n: [] for n in algo_names}
    gatnet_power_mean, gatnet_qos = [], []
    thesis_acc = {k: [] for k in ["sum_rate", "avg_spectral_efficiency", "success_ratio", "outage_ratio", "min_rate", "p5_rate", "avg_sinr_db", "min_sinr_db", "avg_ber", "max_ber", "energy_efficiency", "jain_fairness"]}
    algo_acc = {n: {"success_ratio": [], "avg_sinr_db": [], "avg_ber": []} for n in algo_names}
    rng = np.random.default_rng(config.seed)
    model.eval()

    with torch.no_grad():
        eval_count = min(len(test_data_list), config.compare_samples)
        for data in test_data_list[:eval_count]:
            data = data.to(config.device)
            g, ai, ext = data.g_matrix, data.alpha_i, data.ext_interference

            t0 = time.perf_counter()
            p_gat = model(data)
            runtime_ms["GATNet"].append((time.perf_counter() - t0) * 1000.0)
            gatnet_power_mean.append(float(torch.mean(p_gat).item()))
            sinr_g, rate_g = compute_sinr_and_rate(p_gat, g, config.noise_power, ext)
            results["GATNet"].append(float(torch.sum(ai * rate_g).item()))
            gatnet_qos.append(float(compute_qos_satisfaction_rate(rate_g, config.min_rate).item()))
            tm = _compute_thesis_metrics(p_gat, sinr_g, rate_g, config.min_rate)
            for k in thesis_acc: thesis_acc[k].append(tm[k])
            for k in algo_acc["GATNet"]: algo_acc["GATNet"][k].append(tm[k])

            t_ref = time.perf_counter()
            p_hw = wmmse_power_control(g.cpu().numpy(), ai.cpu().numpy(), ext.cpu().numpy(), config.noise_power, config.p_max, iters=config.hybrid_wmmse_iters, p_init=p_gat.cpu().numpy())
            refine_ms = (time.perf_counter() - t_ref) * 1000.0
            runtime_ms["GATNet+WMMSE"].append(runtime_ms["GATNet"][-1] + refine_ms)
            p_hw_t = torch.tensor(p_hw, dtype=torch.float32, device=config.device)
            sinr_hw, rate_hw = compute_sinr_and_rate(p_hw_t, g, config.noise_power, ext)
            results["GATNet+WMMSE"].append(float(torch.sum(ai * rate_hw).item()))
            tm_hw = _compute_thesis_metrics(p_hw_t, sinr_hw, rate_hw, config.min_rate)
            for k in algo_acc["GATNet+WMMSE"]: algo_acc["GATNet+WMMSE"][k].append(tm_hw[k])

            t0 = time.perf_counter()
            p_wm = wmmse_power_control(g.cpu().numpy(), ai.cpu().numpy(), ext.cpu().numpy(), config.noise_power, config.p_max, iters=config.wmmse_iters)
            runtime_ms["WMMSE"].append((time.perf_counter() - t0) * 1000.0)
            p_wm_t = torch.tensor(p_wm, dtype=torch.float32, device=config.device)
            sinr_wm, rate_wm = compute_sinr_and_rate(p_wm_t, g, config.noise_power, ext)
            results["WMMSE"].append(float(torch.sum(ai * rate_wm).item()))
            tm_wm = _compute_thesis_metrics(p_wm_t, sinr_wm, rate_wm, config.min_rate)
            for k in algo_acc["WMMSE"]: algo_acc["WMMSE"][k].append(tm_wm[k])

            t0 = time.perf_counter()
            p_ga = genetic_power_control(g.cpu().numpy(), ai.cpu().numpy(), ext.cpu().numpy(), config.noise_power, config.p_max, pop_size=config.ga_population, generations=config.ga_generations, elite_count=config.ga_elite, mutation_prob=config.ga_mutation_prob, mutation_std=config.ga_mutation_std, rng=rng)
            runtime_ms["GA"].append((time.perf_counter() - t0) * 1000.0)
            p_ga_t = torch.tensor(p_ga, dtype=torch.float32, device=config.device)
            sinr_ga, rate_ga = compute_sinr_and_rate(p_ga_t, g, config.noise_power, ext)
            results["GA"].append(float(torch.sum(ai * rate_ga).item()))
            tm_ga = _compute_thesis_metrics(p_ga_t, sinr_ga, rate_ga, config.min_rate)
            for k in algo_acc["GA"]: algo_acc["GA"][k].append(tm_ga[k])

            t0 = time.perf_counter()
            p_eq = torch.full((config.N,), 0.5 * config.p_max, device=config.device)
            sinr_eq, rate_eq = compute_sinr_and_rate(p_eq, g, config.noise_power, ext)
            runtime_ms["Equal Power"].append((time.perf_counter() - t0) * 1000.0)
            results["Equal Power"].append(float(torch.sum(ai * rate_eq).item()))
            tm_eq = _compute_thesis_metrics(p_eq, sinr_eq, rate_eq, config.min_rate)
            for k in algo_acc["Equal Power"]: algo_acc["Equal Power"][k].append(tm_eq[k])

            t0 = time.perf_counter()
            rand_rates, rand_tms = [], []
            for _ in range(config.random_trials):
                p_r = torch.tensor(rng.uniform(0, config.p_max, config.N).astype(np.float32), device=config.device)
                s_r, r_r = compute_sinr_and_rate(p_r, g, config.noise_power, ext)
                rand_rates.append(float(torch.sum(ai * r_r).item()))
                rand_tms.append(_compute_thesis_metrics(p_r, s_r, r_r, config.min_rate))
            runtime_ms["Random"].append((time.perf_counter() - t0) * 1000.0)
            results["Random"].append(float(np.mean(rand_rates)))
            for k in algo_acc["Random"]: algo_acc["Random"][k].append(float(np.mean([t[k] for t in rand_tms])))

    mean_results = {n: float(np.mean(v)) for n, v in results.items()}
    mean_results.update({f"{n} Time (ms)": float(np.mean(v)) for n, v in runtime_ms.items()})
    mean_results["GATNet Mean Power"], mean_results["GATNet QoS Satisfaction"] = float(np.mean(gatnet_power_mean)), float(np.mean(gatnet_qos))
    for k, v in thesis_acc.items(): mean_results[k] = float(np.mean(v))
    for algo, mmap in algo_acc.items():
        for metric, vals in mmap.items(): mean_results[f"{algo} {metric}"] = float(np.mean(vals)) if vals else 0.0

    return mean_results