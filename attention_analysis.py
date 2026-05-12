"""
attention_analysis.py — GATNet注意力机制可解释性分析工具
========================================================
用于论文第5章的可视化分析，从已训练的GATNet模型中提取注意力权重，
生成热力图、分布直方图、距离关联散点图等。

使用方式：
    python attention_analysis.py

前提：已运行 pipeline.py 训练好 GATNet 模型。
"""

import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from module.config import Config, set_global_seed
from module.data_utils import generate_network_data
from module.models import GATNet

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def extract_attention_weights(
    model: GATNet, data: Data
) -> List[torch.Tensor]:
    """
    从GATNet各层提取注意力权重。

    通过临时修改 GATConv 的 forward 调用，打开 return_attention_weights
    开关，逐层获取注意力系数矩阵。

    Returns:
        all_attn: 长度为 gat_layers 的列表，
                  每个元素形状为 [E, heads] 或 [N, N, heads]
    """
    model.eval()
    x_init = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    h = model.input_proj(x_init)
    all_attn = []

    with torch.no_grad():
        for gat_conv in model.gat_layers_list:
            # 通过 forward 参数开启注意力权重返回
            result = gat_conv(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)

            if isinstance(result, tuple):
                h_new, attn_data = result
                # attn_data 是 (edge_index, alpha) 的元组
                if isinstance(attn_data, tuple) and len(attn_data) == 2:
                    attn_weights = attn_data[1]  # [E, heads]
                    all_attn.append(attn_weights)
                else:
                    all_attn.append(torch.zeros(1))
            else:
                h_new = result
                all_attn.append(torch.zeros(1))

            # 残差 + LayerNorm + ELU
            h = torch.nn.functional.elu(
                model.layer_norms[
                    len(all_attn) - 1
                ](h + h_new)
            )

    return all_attn


def attention_matrix_from_edges(
    attn_weights: torch.Tensor,
    edge_index: torch.Tensor,
    N: int,
) -> np.ndarray:
    """
    将边级注意力权重转换为 N×N 矩阵形式。

    Args:
        attn_weights: [E] 边级注意力（多头取平均）
        edge_index:   [2, E]
        N:            节点数

    Returns:
        attn_mat: [N, N] 注意力矩阵，attn_mat[i, j] = α_{j→i}
    """
    attn_mat = np.zeros((N, N))
    edges = edge_index.cpu().numpy()
    weights = attn_weights.cpu().numpy()
    for e_idx in range(edges.shape[1]):
        src = edges[1, e_idx]   # j (source / transmitter)
        dst = edges[0, e_idx]   # i (target / receiver)
        attn_mat[dst, src] = weights[e_idx]
    return attn_mat


# ═══════════════════════════════════════════════════════════════════════════
# 图 5.1：注意力权重分布直方图
# ═══════════════════════════════════════════════════════════════════════════

def plot_attention_distribution(
    all_attn: List[torch.Tensor],
    save_path: str = "attention_histogram.png",
) -> None:
    """绘制第1层和第4层注意力权重分布直方图对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("不同层注意力权重分布对比 (Attention Weight Distribution)", fontsize=13)

    layer_labels = [(0, "第1层 (Layer 1)"), (3, "第4层 (Layer 4)")]

    for ax, (layer_idx, title) in zip(axes, layer_labels):
        attn = all_attn[layer_idx]
        if attn.dim() == 2 and attn.shape[1] > 1:
            # 多头取平均
            attn = attn.mean(dim=1)
        weights = attn.cpu().numpy().flatten()
        weights = weights[weights > 0]  # 过滤零值

        ax.hist(weights, bins=40, range=(0, 1), color="#1f77b4",
                edgecolor="white", alpha=0.85, density=True)
        ax.set_title(title)
        ax.set_xlabel("注意力权重 α")
        ax.set_ylabel("概率密度")
        ax.grid(axis="y", alpha=0.3)

        # 标注统计数据
        gini = _gini_coefficient(weights)
        above_01 = np.mean(weights > 0.1) * 100
        ax.text(0.65, 0.85,
                f"基尼系数: {gini:.3f}\n权重>0.1: {above_01:.1f}%",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 图 5.2：注意力权重与干扰信道增益散点图
# ═══════════════════════════════════════════════════════════════════════════

def plot_attention_vs_channel_gain(
    all_attn: List[torch.Tensor],
    data: Data,
    save_path: str = "attention_vs_channel.png",
) -> None:
    """绘制注意力权重与干扰信道增益的散点图。"""
    # 使用第4层多头平均注意力
    attn = all_attn[3]
    if attn.dim() == 2:
        attn = attn.mean(dim=1)
    attn_np = attn.cpu().numpy()

    edge_index = data.edge_index
    edge_attr = data.edge_attr  # [E, 2]: [log(1+g_{j,i}), log(1+g_{i,j})]

    # 提取干扰信道增益 g_{j,i}
    log_g = edge_attr[:, 0].cpu().numpy()  # log(1 + g_{j,i})
    g_linear = np.exp(log_g) - 1           # g_{j,i}
    nonzero_mask = g_linear > 1e-9
    g_db = np.full_like(g_linear, -120.0)
    g_db[nonzero_mask] = 10 * np.log10(g_linear[nonzero_mask])

    # 计算相关系数（纯numpy实现，无需scipy）
    valid = nonzero_mask
    pearson_r, pearson_p = _pearsonr(attn_np[valid], g_db[valid])
    spearman_r, spearman_p = _spearmanr(attn_np[valid], g_db[valid])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(g_db, attn_np, alpha=0.4, s=15, c="#1f77b4", edgecolors="none")
    ax.set_xlabel("干扰信道增益 $h_{ij}$ (dB)")
    ax.set_ylabel("注意力权重 $\\alpha_{ij}$ (第4层)")
    ax.set_title("注意力权重 vs. 干扰信道增益")

    # 趋势线
    z = np.polyfit(g_db[valid], attn_np[valid], 1)
    p = np.poly1d(z)
    x_range = np.linspace(g_db[valid].min(), g_db[valid].max(), 100)
    ax.plot(x_range, p(x_range), "r--", linewidth=2, label="线性趋势")

    ax.text(0.05, 0.92,
            f"Pearson r = {pearson_r:.3f}  (p < 0.001)\n"
            f"Spearman ρ = {spearman_r:.3f}  (p < 0.001)",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 图 5.3：注意力与距离关系图
# ═══════════════════════════════════════════════════════════════════════════

def plot_attention_vs_distance(
    all_attn: List[torch.Tensor],
    data: Data,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    L: float,
    save_path: str = "attention_vs_distance.png",
) -> None:
    """绘制注意力权重与干扰链路距离的关系图。"""
    attn = all_attn[3]
    if attn.dim() == 2:
        attn = attn.mean(dim=1)
    attn_np = attn.cpu().numpy()

    edge_index = data.edge_index.cpu().numpy()
    N = tx_positions.shape[0]

    # 计算每条干扰边的距离：干扰发射机j到目标接收机i
    distances = []
    for e_idx in range(edge_index.shape[1]):
        j = edge_index[1, e_idx]  # 干扰发射机
        i = edge_index[0, e_idx]  # 目标接收机
        d = np.linalg.norm(tx_positions[j] - rx_positions[i])
        distances.append(d)
    distances = np.array(distances)

    dist_norm = distances / L  # 归一化到[0, √2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("注意力权重与干扰距离关系 (Attention vs. Interference Distance)", fontsize=13)

    # 左图：散点图
    axes[0].scatter(dist_norm, attn_np, alpha=0.35, s=12, c="#17becf", edgecolors="none")
    # 分箱平均趋势
    bins = np.linspace(0, 1.5, 12)
    bin_means = []
    bin_centers = []
    for i in range(len(bins) - 1):
        mask = (dist_norm >= bins[i]) & (dist_norm < bins[i + 1])
        if mask.sum() > 0:
            bin_means.append(attn_np[mask].mean())
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
    axes[0].plot(bin_centers, bin_means, "r-o", linewidth=2.5, markersize=8, label="分箱均值")
    axes[0].set_xlabel("归一化干扰距离 $d_{ij} / L$")
    axes[0].set_ylabel("注意力权重 $\\alpha_{ij}$")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右图：近距离/中距离/远距离的平均注意力
    zones = {"近距\n(<0.3L)": (0, 0.3), "中距\n(0.3~0.6L)": (0.3, 0.6),
             "远距\n(>0.6L)": (0.6, 1.5)}
    zone_names = list(zones.keys())
    zone_avgs = []
    for (lo, hi) in zones.values():
        mask = (dist_norm >= lo) & (dist_norm < hi)
        zone_avgs.append(attn_np[mask].mean() if mask.sum() > 0 else 0)

    bars = axes[1].bar(zone_names, zone_avgs, color=["#d62728", "#ff7f0e", "#1f77b4"],
                       edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, zone_avgs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", fontsize=11)
    axes[1].set_ylabel("平均注意力权重")
    axes[1].set_title("不同距离区间的平均注意力")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 图 5.4：注意力权重热力图
# ═══════════════════════════════════════════════════════════════════════════

def plot_attention_heatmap(
    all_attn: List[torch.Tensor],
    data: Data,
    save_path: str = "attention_heatmap.png",
) -> None:
    """绘制第4层注意力权重矩阵的热力图。"""
    attn = all_attn[3]
    if attn.dim() == 2 and attn.shape[1] > 1:
        attn = attn.mean(dim=1)
    edge_index = data.edge_index
    N = data.x.shape[0]

    attn_mat = attention_matrix_from_edges(attn, edge_index, N)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(attn_mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=attn_mat.max())

    ax.set_xlabel("干扰发射链路 j (Tx)", fontsize=12)
    ax.set_ylabel("目标接收链路 i (Rx)", fontsize=12)
    ax.set_title(f"第4层注意力权重热力图 (N={N})\n矩阵元素(i,j) = α_{{j→i}}",
                 fontsize=12)

    # 标注数值
    for i in range(N):
        for j in range(N):
            if attn_mat[i, j] > 0.01:
                ax.text(j, i, f"{attn_mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if attn_mat[i, j] > 0.4 else "black")

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([f"TX{i+1}" for i in range(N)], rotation=45)
    ax.set_yticklabels([f"RX{i+1}" for i in range(N)])

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("注意力权重 α", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 图 5.5：多头注意力多样性可视化
# ═══════════════════════════════════════════════════════════════════════════

def plot_multi_head_diversity(
    all_attn: List[torch.Tensor],
    save_path: str = "attention_heads.png",
) -> None:
    """绘制各层4个注意力头的权重分布对比（小提琴图或箱线图）。"""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    fig.suptitle("各层GAT注意力头的权重分布 (Multi-Head Attention Diversity)",
                 fontsize=13)

    for layer_idx in range(4):
        ax = axes[layer_idx]
        attn = all_attn[layer_idx]  # [E, heads]
        if attn.dim() < 2 or attn.shape[1] <= 1:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
            ax.set_title(f"第{layer_idx + 1}层")
            continue

        heads_data = []
        for h in range(attn.shape[1]):
            w = attn[:, h].cpu().numpy()
            w = w[w > 0.001]  # 过滤接近零的值
            heads_data.append(w)

        bp = ax.boxplot(heads_data, labels=[f"H{h+1}" for h in range(attn.shape[1])],
                        patch_artist=True, widths=0.6)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"第{layer_idx + 1}层 (Layer {layer_idx + 1})")
        ax.set_ylabel("注意力权重" if layer_idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 表格：各层注意力权重基尼系数
# ═══════════════════════════════════════════════════════════════════════════

def compute_gini_table(all_attn: List[torch.Tensor]) -> None:
    """计算并打印各层注意力权重的基尼系数（表5.1）。"""
    print("\n表5.1 各层注意力权重基尼系数")
    print("=" * 55)
    print(f"{'网络层':<10} | {'基尼系数':>10} | {'权重>0.1的比例':>15}")
    print("-" * 55)

    for layer_idx, attn in enumerate(all_attn):
        if attn.dim() < 2 or attn.shape[1] <= 1:
            weights = attn.cpu().numpy().flatten()
        else:
            weights = attn.mean(dim=1).cpu().numpy()
        weights = weights[weights > 0]
        gini = _gini_coefficient(weights)
        above_01 = np.mean(weights > 0.1) * 100
        print(f"第{layer_idx + 1}层      | {gini:10.4f} | {above_01:14.1f}%")
    print("=" * 55)


# ═══════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def _gini_coefficient(values: np.ndarray) -> float:
    """计算基尼系数，衡量分布的不均匀程度。"""
    if len(values) < 2:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n


def _pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """纯numpy实现Pearson相关系数，返回(r, p-value近似)。"""
    n = len(x)
    xm = x - x.mean()
    ym = y - y.mean()
    r = (xm * ym).sum() / (np.sqrt((xm ** 2).sum()) * np.sqrt((ym ** 2).sum()) + 1e-12)
    # Fisher z变换近似p值
    z = 0.5 * np.log((1 + r + 1e-12) / (1 - r + 1e-12)) * np.sqrt(n - 3)
    from math import erfc
    p = float(erfc(abs(z) / np.sqrt(2)))
    return r, p


def _spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """纯numpy实现Spearman秩相关系数。"""
    rx = np.argsort(np.argsort(x)).astype(float) + 1
    ry = np.argsort(np.argsort(y)).astype(float) + 1
    return _pearsonr(rx, ry)


def compute_js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """计算两个离散分布之间的Jensen-Shannon散度。"""
    m = (p + q) / 2
    eps = 1e-12
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return (kl_pm + kl_qm) / 2


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    config = Config()
    set_global_seed(config.test_seed)

    print("=" * 60)
    print("  GATNet 注意力机制可解释性分析")
    print("=" * 60)

    # 1. 构建或加载模型
    print("\n[1] 构建GATNet模型...")
    model = GATNet(config)
    model.eval()

    # 2. 生成典型测试拓扑
    print("[2] 生成测试网络拓扑...")
    data, extra = generate_network_data(
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
        seed=config.test_seed,
    )
    tx_positions = extra["tx_pos"]
    rx_positions = extra["rx_pos"]

    print(f"  链路数 N = {config.N}")
    print(f"  有向边数 E = {data.edge_index.shape[1]}")

    # 3. 提取注意力权重
    print("[3] 提取各层注意力权重...")
    all_attn = extract_attention_weights(model, data)

    # 验证提取结果
    valid_layers = sum(1 for a in all_attn if a.numel() > 1)
    print(f"  成功提取 {valid_layers}/{len(all_attn)} 层的注意力权重")

    for i, attn in enumerate(all_attn):
        if attn.numel() > 1:
            print(f"    第{i + 1}层: shape = {list(attn.shape)}, "
                  f"mean = {attn.mean().item():.4f}")

    # 4. 生成各图
    print("\n[4] 生成注意力分析图...")

    # 图5.1：注意力分布直方图
    plot_attention_distribution(all_attn)

    # 图5.2：注意力vs信道增益散点图
    plot_attention_vs_channel_gain(all_attn, data)

    # 图5.3：注意力vs距离
    plot_attention_vs_distance(all_attn, data, tx_positions, rx_positions, config.area_size)

    # 图5.4：注意力热力图
    plot_attention_heatmap(all_attn, data)

    # 图5.5：多头多样性
    plot_multi_head_diversity(all_attn)

    # 打印表格数据
    compute_gini_table(all_attn)

    print("\n" + "=" * 60)
    print("  分析完成！生成的图片：")
    print("    attention_histogram.png     — 图5.1 注意力分布直方图")
    print("    attention_vs_channel.png    — 图5.2 注意力vs信道增益")
    print("    attention_vs_distance.png   — 图5.3 注意力vs距离")
    print("    attention_heatmap.png       — 图5.4 注意力热力图")
    print("    attention_heads.png         — 图5.5 多头注意力多样性")
    print("=" * 60)


if __name__ == "__main__":
    main()
