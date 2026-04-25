import math
from typing import Optional

import numpy as np


def _weighted_sum_rate_numpy(
    power: np.ndarray,
    g_matrix: np.ndarray,
    alpha_i: np.ndarray,
    sigma_vec: np.ndarray,
) -> float:
    total_rx = np.matmul(power, g_matrix)
    signal = power * np.diag(g_matrix)
    interference = total_rx - signal
    sinr = signal / (sigma_vec + interference + 1e-12)
    rate = np.log2(1.0 + sinr)
    return float(np.sum(alpha_i * rate))


def wmmse_power_control(
    g_matrix: np.ndarray,
    alpha_i: np.ndarray,
    ext_interference: np.ndarray,
    noise_power: float,
    p_max: float,
    iters: int = 60,
    p_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = g_matrix.shape[0]
    h = np.sqrt(np.maximum(g_matrix, 1e-12))
    sigma_vec = noise_power + ext_interference

    if p_init is None:
        p = np.full(n, 0.5 * p_max, dtype=np.float64)
    else:
        p = np.clip(p_init.astype(np.float64), 0.0, p_max)
    sqrt_p_max = math.sqrt(p_max)

    for _ in range(iters):
        total_rx = np.matmul(p, g_matrix) + sigma_vec
        v = np.sqrt(np.maximum(p, 1e-12)) * np.diag(h) / (total_rx + 1e-12)
        e = 1.0 - 2.0 * v * np.sqrt(np.maximum(p, 1e-12)) * np.diag(h) + (v ** 2) * total_rx
        w = alpha_i / np.maximum(e, 1e-12)

        new_p = np.zeros_like(p)
        for i in range(n):
            denom = np.sum(w * (v ** 2) * g_matrix[i, :]) + 1e-12
            numer = w[i] * v[i] * h[i, i]
            sqrt_pi = np.clip(numer / denom, 0.0, sqrt_p_max)
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
) -> np.ndarray:
    n = g_matrix.shape[0]
    sigma_vec = noise_power + ext_interference

    pop = rng.uniform(0.0, p_max, size=(pop_size, n)).astype(np.float32)
    pop[0, :] = p_max
    pop[1, :] = 0.5 * p_max

    def fitness(ind: np.ndarray) -> float:
        return _weighted_sum_rate_numpy(ind, g_matrix, alpha_i, sigma_vec)

    for _ in range(generations):
        scores = np.array([fitness(ind) for ind in pop], dtype=np.float64)
        rank = np.argsort(scores)[::-1]
        elites = pop[rank[:elite_count]].copy()

        next_pop = [e for e in elites]
        while len(next_pop) < pop_size:
            i1, i2 = rng.integers(0, pop_size // 2, size=2)
            p1 = pop[rank[i1]]
            p2 = pop[rank[i2]]

            mix = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
            child = mix * p1 + (1.0 - mix) * p2

            mut_mask = rng.random(n) < mutation_prob
            if np.any(mut_mask):
                child[mut_mask] += rng.normal(0.0, mutation_std, size=np.sum(mut_mask)).astype(np.float32)
            child = np.clip(child, 0.0, p_max)
            next_pop.append(child)

        pop = np.array(next_pop[:pop_size], dtype=np.float32)

    scores = np.array([fitness(ind) for ind in pop], dtype=np.float64)
    best_idx = int(np.argmax(scores))
    return pop[best_idx]
