"""
baselines.py — Traditional Power Allocation Baseline Algorithms
===============================================================
1. WMMSE — Weighted Minimum Mean Square Error (iterative convex optimisation)
2. GA    — Genetic Algorithm (evolutionary meta-heuristic)

Both algorithms include a minimum power floor (p_min = p_min_frac * p_max).

WHY p_min_frac IS CRITICAL:
    Standard WMMSE uses water-filling which can set weak links' power to
    exactly zero.  A zeroed link has SINR → 0, i.e. SINR_dB → −∞.
    Even one such link drags the MEAN SINR far below 0 dB and makes the
    "Average SINR" bar in the thesis plot appear extremely negative
    despite most links performing well.

    Setting p_min = 0.02 * p_max ensures every link transmits at least
    at 2 % of peak power, preserving a reasonable SINR floor while having
    negligible impact on the sum-rate optimisation.
"""

import math
from typing import Optional

import numpy as np


def _weighted_sum_rate_numpy(
    power: np.ndarray,
    g_matrix: np.ndarray,
    alpha_i: np.ndarray,
    sigma_vec: np.ndarray,
) -> float:
    """Weighted sum rate in NumPy (GA fitness function)."""
    total_rx  = np.matmul(power, g_matrix)
    signal    = power * np.diag(g_matrix)
    interfere = total_rx - signal
    sinr      = signal / (sigma_vec + interfere + 1e-12)
    rate      = np.log2(1.0 + sinr)
    return float(np.sum(alpha_i * rate))


def wmmse_power_control(
    g_matrix: np.ndarray,
    alpha_i: np.ndarray,
    ext_interference: np.ndarray,
    noise_power: float,
    p_max: float,
    iters: int = 60,
    p_init: Optional[np.ndarray] = None,
    p_min_frac: float = 0.02,
) -> np.ndarray:
    """WMMSE with minimum power floor to prevent SINR collapse.

    BCD iteration:
        v_i ← sqrt(p_i)*h_ii / total_rx_i
        e_i ← 1 - 2*v_i*sqrt(p_i)*h_ii + v_i^2 * total_rx_i
        w_i ← alpha_i / e_i
        p_i ← clip([w_i*v_i*h_ii / sum_j(w_j*v_j^2*G_{i,j})]^2, p_min, p_max)

    Args:
        p_min_frac: minimum power as fraction of p_max (default 2 %).
                    Prevents any link being silenced, which would cause
                    SINR_dB → −∞ and bias the average SINR metric.
    """
    n         = g_matrix.shape[0]
    h         = np.sqrt(np.maximum(g_matrix, 1e-12))
    sigma_vec = noise_power + ext_interference
    p_min     = p_min_frac * p_max

    if p_init is None:
        p = np.full(n, 0.5 * p_max, dtype=np.float64)
    else:
        p = np.clip(p_init.astype(np.float64), p_min, p_max)

    sqrt_p_max = math.sqrt(p_max)
    sqrt_p_min = math.sqrt(p_min)

    for _ in range(iters):
        total_rx = np.matmul(p, g_matrix) + sigma_vec        # [N]

        # MMSE receiver
        v = np.sqrt(np.maximum(p, 1e-12)) * np.diag(h) / (total_rx + 1e-12)

        # Minimum MSE
        e = (1.0
             - 2.0 * v * np.sqrt(np.maximum(p, 1e-12)) * np.diag(h)
             + (v ** 2) * total_rx)

        # MSE weight
        w = alpha_i / np.maximum(e, 1e-12)

        # Optimal power (closed form), clipped to [p_min, p_max]
        new_p = np.zeros_like(p)
        for i in range(n):
            denom   = np.sum(w * (v ** 2) * g_matrix[i, :]) + 1e-12
            numer   = w[i] * v[i] * h[i, i]
            sqrt_pi = np.clip(numer / denom, sqrt_p_min, sqrt_p_max)
            new_p[i] = sqrt_pi ** 2
        p = new_p

    return p.astype(np.float32)


def genetic_power_control(
    g_matrix: np.ndarray,
    alpha_i: np.ndarray,
    ext_interference: np.ndarray,
    noise_power: float,
    p_max: float,
    pop_size: int,
    generations: int,
    elite_count: int,
    mutation_prob: float,
    mutation_std: float,
    rng: np.random.Generator,
    p_min_frac: float = 0.02,
) -> np.ndarray:
    """Genetic algorithm: elitism + uniform crossover + Gaussian mutation.

    Args:
        p_min_frac: same as in wmmse_power_control — prevents zero-power links.
    """
    n         = g_matrix.shape[0]
    sigma_vec = noise_power + ext_interference
    p_min     = p_min_frac * p_max

    # Initialise: random in [p_min, p_max] + two heuristic seeds
    pop       = rng.uniform(p_min, p_max, size=(pop_size, n)).astype(np.float32)
    pop[0, :] = p_max
    pop[1, :] = 0.5 * p_max

    def fitness(ind: np.ndarray) -> float:
        return _weighted_sum_rate_numpy(ind, g_matrix, alpha_i, sigma_vec)

    for _ in range(generations):
        scores = np.array([fitness(ind) for ind in pop], dtype=np.float64)
        rank   = np.argsort(scores)[::-1]
        elites = pop[rank[:elite_count]].copy()

        next_pop = list(elites)
        while len(next_pop) < pop_size:
            i1, i2 = rng.integers(0, pop_size // 2, size=2)
            p1, p2 = pop[rank[i1]], pop[rank[i2]]

            mix   = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
            child = mix * p1 + (1.0 - mix) * p2

            mut_mask = rng.random(n) < mutation_prob
            if np.any(mut_mask):
                child[mut_mask] += rng.normal(
                    0.0, mutation_std, size=int(np.sum(mut_mask))).astype(np.float32)

            child = np.clip(child, p_min, p_max)
            next_pop.append(child)

        pop = np.array(next_pop[:pop_size], dtype=np.float32)

    scores   = np.array([fitness(ind) for ind in pop], dtype=np.float64)
    best_idx = int(np.argmax(scores))
    return pop[best_idx]