"""
visualization.py — 实验结果可视化模块
=======================================
本模块负责生成论文所需的四张实验结果图：

    1. training_curves.png  — 训练收敛曲线（4 个子图）
       · 损失 Gap（EMA 趋势损失与最终值的相对差距）
       · 和速率 Gap
       · 平均发射功率 Gap
       · QoS 满足率 Gap
       （Gap 越小越好，最终收敛至 0 表示完全收敛）

    2. results.png          — 算法性能对比（和速率 + 运行时间）

    3. thesis_metrics.png   — 多维指标对比（SINR / BER / QoS）

    4. interference_groups.png — 三组干扰场景对比（轻/中/重）

辅助工具：
    · 移动平均平滑（MA）
    · 指数移动平均（EMA）
    · Gap 归一化（相对最终值的百分比偏差）
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from module.config import Config


# 全局绘图顺序：将 GATNet+WMMSE 作为主算法放在首位
PLOT_ALGO_ORDER = ["GATNet+WMMSE", "GATNet", "WMMSE", "GA", "Equal Power", "Random"]
GROUP_PLOT_ALGO_ORDER = ["GATNet+WMMSE", "GATNet", "WMMSE", "GA"]
PLOT_ALGO_COLORS = {
    "GATNet+WMMSE": "#1f77b4",
    "GATNet": "#17becf",
    "WMMSE": "#9467bd",
    "GA": "#d62728",
    "Equal Power": "#ff7f0e",
    "Random": "#2ca02c",
}


def _configure_matplotlib_fonts() -> None:
    """配置中英文混排字体，避免中文显示为方框。"""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    # 解决坐标轴负号显示为方框的问题
    plt.rcParams["axes.unicode_minus"] = False


def _safe_float_array(values: List[float], default: float = 0.0, target_len: int = 0) -> np.ndarray:
    """将可能含 None/NaN 的序列转换为有限浮点数组。"""
    out = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)

    arr = np.asarray(out, dtype=np.float32)
    if target_len > 0:
        if arr.size < target_len:
            arr = np.pad(arr, (0, target_len - arr.size), mode="constant", constant_values=default)
        elif arr.size > target_len:
            arr = arr[:target_len]

    if arr.size == 0:
        return np.full((target_len,), default, dtype=np.float32) if target_len > 0 else arr

    finite = np.isfinite(arr)
    if not np.any(finite):
        arr[:] = default
        return arr

    if not np.all(finite):
        idx = np.arange(arr.size)
        arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


_configure_matplotlib_fonts()


# ═══════════════════════════════════════════════════════════════════════════
# 信号平滑工具函数
# Signal Smoothing Utilities
# ═══════════════════════════════════════════════════════════════════════════

def _moving_average_edge_safe(values: List[float], window: int) -> np.ndarray:
    """
    边界安全的移动平均平滑。
    Edge-safe moving average smoothing.

    使用对称边界填充（edge padding）后做卷积，确保输出长度与输入相同，
    避免边界处的平均值失真。

    Args:
        values: 待平滑的数值序列
        window: 移动窗口大小（奇数效果更好）

    Returns:
        平滑后的数组，长度与输入相同
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size < 3:
        return arr

    win = min(window, int(arr.size))
    if win % 2 == 0:
        win -= 1  # 确保窗口为奇数
    if win <= 1:
        return arr

    pad = win // 2
    padded = np.pad(arr, (pad, pad), mode="edge")  # 边界填充
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(padded, kernel, mode="valid")


def _ema_smooth(values: List[float], alpha: float = 0.08) -> np.ndarray:
    """
    指数移动平均（EMA）平滑。
    Exponential Moving Average (EMA) smoothing.

    EMA_t = α · x_t + (1-α) · EMA_{t-1}

    alpha 越小，平滑效果越强（历史权重越大）；
    alpha 越大，跟踪当前值越敏感（适合检测快速变化）。

    Args:
        values: 待平滑的数值序列
        alpha : 平滑因子 ∈ (0, 1)，默认 0.08

    Returns:
        EMA 平滑后的数组
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr

    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, arr.size):
        ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _gap_to_final(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    计算相对最终值的归一化 Gap（用于收敛曲线可视化）。
    Compute normalized gap relative to the final value.

    Gap_t = |v_t - v_final| / (|v_final| + ε)

    Gap 为 0 表示已完全收敛至最终值，越小越好。

    Args:
        values: 数值序列（如损失历史、速率历史）
        eps   : 数值稳定常数

    Returns:
        归一化 Gap 数组，值域 ≥ 0
    """
    if values.size == 0:
        return values
    final_val = values[-1]
    return np.abs(values - final_val) / (abs(final_val) + eps)


# ═══════════════════════════════════════════════════════════════════════════
# 图表绘制函数
# Plotting Functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curve(
    loss_history: List[float],
    metrics_history: List[Dict],
    save_path: str = "training_curves.png",
) -> None:
    """
    绘制训练收敛曲线（4 子图）。
    Plot training convergence curves (4 subplots).

    子图说明：
        · 左上：损失 Gap（包含 Raw/MA/EMA/Trend 四条曲线）
        · 右上：和速率 Gap
        · 左下：平均发射功率 Gap
        · 右下：QoS 满足率 Gap

    所有 Gap 以最终收敛值为参考，越接近 0 表示训练越充分。

    Args:
        loss_history   : 每轮 EMA 趋势损失列表
        metrics_history: 每轮各指标均值列表
        save_path      : 图片保存路径
    """
    if len(loss_history) == 0:
        print("  警告: loss_history 为空，跳过 training_curves 绘制")
        return

    n_epochs = min(len(loss_history), len(metrics_history)) if metrics_history else len(loss_history)
    epochs = np.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GATNet 训练收敛曲线 (Training Convergence Curves)", fontsize=13)
    smooth_window = 41  # 移动平均窗口大小

    # ── 左上：损失 Gap ────────────────────────────────────────────────────
    trend_loss = np.asarray(loss_history[:n_epochs], dtype=np.float32)
    raw_loss = _safe_float_array(
        [m.get("raw_loss", l) for m, l in zip(metrics_history[:n_epochs], trend_loss.tolist())],
        default=float(trend_loss[-1]),
        target_len=n_epochs,
    )
    ma_loss = _moving_average_edge_safe(raw_loss.tolist(), smooth_window)
    ema_loss = _ema_smooth(raw_loss.tolist(), alpha=0.08)

    axes[0, 0].plot(epochs, _gap_to_final(raw_loss), alpha=0.2, label="Raw Loss Gap")
    axes[0, 0].plot(epochs, _gap_to_final(ma_loss), linewidth=1.8, label="MA Loss Gap")
    axes[0, 0].plot(epochs, _gap_to_final(ema_loss), linewidth=1.6, linestyle="--", label="EMA Loss Gap")
    axes[0, 0].plot(epochs, _gap_to_final(trend_loss), linewidth=2.0, label="Trend Loss Gap")
    axes[0, 0].set_title("Loss Gap to Final (↓ Better)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # ── 右上：和速率 Gap ──────────────────────────────────────────────────
    sum_rates_raw = []
    for m in metrics_history[:n_epochs]:
        val = m.get("sum_rate")
        if val is None:
            val = m.get("sumrate")
        if val is None:
            val = m.get("weighted_sum_rate")
        sum_rates_raw.append(val)

    sum_rates = _safe_float_array(sum_rates_raw, default=0.0, target_len=n_epochs)
    ma_sr = _moving_average_edge_safe(sum_rates.tolist(), smooth_window)
    ema_sr = _ema_smooth(sum_rates.tolist(), alpha=0.08)
    trend_sr = _ema_smooth(sum_rates.tolist(), alpha=0.20)

    axes[0, 1].plot(epochs, _gap_to_final(sum_rates), alpha=0.2, color="green", label="Raw Sum-Rate Gap")
    axes[0, 1].plot(epochs, _gap_to_final(ma_sr), linewidth=1.8, color="limegreen", label="MA Sum-Rate Gap")
    axes[0, 1].plot(epochs, _gap_to_final(ema_sr), linewidth=1.6, color="darkgreen", linestyle="--", label="EMA Sum-Rate Gap")
    axes[0, 1].plot(epochs, _gap_to_final(trend_sr), linewidth=2.0, color="forestgreen", label="Trend Sum-Rate Gap")
    axes[0, 1].set_title("Sum-Rate Gap to Final (↓ Better)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Normalized Gap")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)

    # ── 左下：平均功率 Gap ────────────────────────────────────────────────
    mean_powers = _safe_float_array(
        [m.get("mean_power", 0) for m in metrics_history[:n_epochs]],
        default=0.0,
        target_len=n_epochs,
    )
    axes[1, 0].plot(epochs, _gap_to_final(_ema_smooth(mean_powers)), color="orange", linewidth=1.6)
    axes[1, 0].set_title("Mean-Power Gap to Final (↓ Better)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(alpha=0.3)

    # ── 右下：QoS Gap ─────────────────────────────────────────────────────
    qos = _safe_float_array(
        [m.get("qos_satisfaction", 0) for m in metrics_history[:n_epochs]],
        default=0.0,
        target_len=n_epochs,
    )
    axes[1, 1].plot(epochs, _gap_to_final(_ema_smooth(qos)), color="purple", linewidth=1.6)
    axes[1, 1].set_title("QoS Satisfaction Gap to Final (↓ Better)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


def plot_result_bar(
    mean_results: Dict[str, float],
    save_path: str = "results.png",
) -> None:
    """
    绘制各算法和速率与运行时间的柱状图对比。
    Plot bar chart comparing sum-rate and runtime across algorithms.

    左图：加权和速率（性能越高越好）
    右图：计算延迟（时间越短越好，体现 GATNet 的实时推理优势）

    Args:
        mean_results: evaluate_algorithms() 返回的均值结果字典
        save_path   : 图片保存路径
    """
    algo_keys = PLOT_ALGO_ORDER
    rate_vals = [mean_results.get(k, 0.0) for k in algo_keys]
    time_vals = [mean_results.get(f"{k} Time (ms)", 0.0) for k in algo_keys]

    colors = [PLOT_ALGO_COLORS[k] for k in algo_keys]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle("GATNet+WMMSE vs. 基线算法综合对比 (GATNet+WMMSE vs. Baselines)", fontsize=12)

    # ── 左图：和速率对比 ──────────────────────────────────────────────────
    bars1 = axes[0].bar(algo_keys, rate_vals, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_ylabel("Average Weighted Sum Rate (bps/Hz)")
    axes[0].set_title("Sum-Rate Comparison（频谱效率对比）")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars1, rate_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9
        )

    # ── 右图：运行时间对比 ────────────────────────────────────────────────
    bars2 = axes[1].bar(algo_keys, time_vals, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_ylabel("Average Compute Time (ms/sample)")
    axes[1].set_title("Runtime Comparison（计算延迟对比）")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].tick_params(axis="x", rotation=15)
    for bar, val in zip(bars2, time_vals):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


def plot_thesis_metrics(
    mean_results: Dict[str, float],
    save_path: str = "thesis_metrics.png",
) -> None:
    """
    绘制论文核心指标对比图（SINR / BER / QoS）。
    Plot thesis core metrics comparison (SINR / BER / QoS).

    左图：平均 SINR（dB）— 越高越好
    中图：平均 BER（对数轴）— 越低越好
    右图：QoS 满足率 — 越高越好

    Args:
        mean_results: 包含各算法指标的结果字典
        save_path   : 图片保存路径
    """
    algo_keys = PLOT_ALGO_ORDER
    colors = [PLOT_ALGO_COLORS[k] for k in algo_keys]

    sinr_vals = [mean_results.get(f"{k} avg_sinr_db", mean_results.get("avg_sinr_db", 0.0)) for k in algo_keys]
    ber_vals = [mean_results.get(f"{k} avg_ber", mean_results.get("avg_ber", 0.0)) for k in algo_keys]
    qos_vals = [mean_results.get(f"{k} success_ratio", mean_results.get("success_ratio", 0.0)) for k in algo_keys]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.suptitle("GATNet+WMMSE 论文指标对比 (Thesis Metrics Comparison)", fontsize=12)

    # ── 左图：SINR ────────────────────────────────────────────────────────
    bars1 = axes[0].bar(algo_keys, sinr_vals, color=colors, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Average SINR (dB)  [↑ Better]")
    axes[0].set_ylabel("dB")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for bar, val in zip(bars1, sinr_vals):
        va = "bottom" if val >= 0 else "top"
        offset = 0.3 if val >= 0 else -0.3
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + offset,
                     f"{val:.2f}", ha="center", va=va, fontsize=9)

    # ── 中图：BER（对数轴）───────────────────────────────────────────────
    bars2 = axes[1].bar(algo_keys, ber_vals, color=colors, edgecolor="white", linewidth=0.8)
    axes[1].set_title("Average BER  [↓ Better]")
    axes[1].set_ylabel("BER")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, ber_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val,
                     f"{val:.2e}", ha="center", va="bottom", fontsize=9)

    # ── 右图：QoS 满足率 ──────────────────────────────────────────────────
    bars3 = axes[2].bar(algo_keys, qos_vals, color=colors, edgecolor="white", linewidth=0.8)
    axes[2].set_title("QoS Success Ratio  [↑ Better]")
    axes[2].set_ylabel("Ratio")
    axes[2].set_ylim(0.0, 1.1)
    axes[2].tick_params(axis="x", rotation=15)
    axes[2].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars3, qos_vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 结果表格打印函数
# Result Table Printing Functions
# ═══════════════════════════════════════════════════════════════════════════

def print_result_table(mean_results: Dict[str, float], title: str = "Evaluation Results") -> None:
    """
    打印算法对比结果表格（含论文所需的全部指标）。
    Print algorithm comparison table with all thesis metrics.
    """
    print(f"\n{'='*65}\n{title}\n{'='*65}")
    print(f"{'Algorithm':<15} | {'Weighted Sum Rate':>18} | {'Time(ms)':>10}")
    print("-" * 52)

    for key in PLOT_ALGO_ORDER:
        if key in mean_results:
            t_key = f"{key} Time (ms)"
            print(f"{key:<15} | {mean_results[key]:>18.6f} | {mean_results.get(t_key, 0.0):>10.3f}")

    print("-" * 52)
    print(f"{'GATNet Mean Power':<15} | {mean_results.get('GATNet Mean Power', 0):>28.6f}")
    print(f"{'GATNet QoS':<15} | {mean_results.get('GATNet QoS Satisfaction', 0):>28.2%}")
    print("-" * 52)
    print("GATNet 论文指标详情 (Thesis Metrics for GATNet):")
    metrics_display = [
        ("sum_rate",               "和速率 Sum Rate"),
        ("avg_spectral_efficiency","平均频谱效率 Avg SE"),
        ("success_ratio",          "QoS 满足率 Success"),
        ("outage_ratio",           "中断率 Outage"),
        ("min_rate",               "最低速率 Min Rate"),
        ("p5_rate",                "5% 分位速率 P5 Rate"),
        ("avg_sinr_db",            "平均 SINR (dB)"),
        ("min_sinr_db",            "最低 SINR (dB)"),
        ("avg_ber",                "平均 BER"),
        ("max_ber",                "最高 BER"),
        ("energy_efficiency",      "能量效率 EE"),
        ("jain_fairness",          "Jain 公平性 Fairness"),
    ]
    for key, label in metrics_display:
        val = mean_results.get(key, 0.0)
        if key in ("avg_ber", "max_ber"):
            print(f"  {label:<28}: {val:.4e}")
        elif key in ("success_ratio", "outage_ratio"):
            print(f"  {label:<28}: {val:.4f} ({val:.2%})")
        else:
            print(f"  {label:<28}: {val:.6f}")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════
# 干扰场景分组对比
# Interference Group Comparison
# ═══════════════════════════════════════════════════════════════════════════

def build_interference_group_configs(base: Config) -> Dict[str, Config]:
    """
    构建三组干扰场景的配置（轻度 / 中度 / 重度外部干扰）。
    Build configurations for three interference scenarios (Light/Medium/Heavy).

    三组场景对应论文"3 组以上对比实验"的要求，
    通过改变干扰源数量、发射功率、活跃概率和突发强度来模拟不同的干扰环境。

    轻度干扰 (Light) : 少量干扰源，低功率，低活跃率 → 接近理想环境
    中度干扰 (Medium): 标准干扰设置 → 正常工作场景
    重度干扰 (Heavy) : 多干扰源，高功率，高活跃率 + 强突发 → 恶劣干扰环境

    Args:
        base: 基础配置（训练时使用的配置）

    Returns:
        group_cfgs: 字典，键为场景名称，值为对应 Config 对象
    """
    common = {
        **base.__dict__,
        "external_interference_enabled": True,
        "compare_samples": min(base.compare_samples, base.group_compare_samples),
    }

    # 轻度干扰：模拟邻区干扰较弱的场景
    # Light interference: weak interference from distant sources
    light = Config(**{
        **common,
        "ext_interferer_count": 2,      # 较少干扰源
        "ext_tx_power": 0.4,            # 低发射功率
        "ext_activity_prob": 0.35,      # 低活跃概率
        "ext_burst_prob": 0.05,         # 极低突发概率
        "ext_burst_scale": 1.5,         # 弱突发强度
        "test_seed": base.test_seed + 100,
    })

    # 中度干扰：与训练场景相同的默认干扰设置
    # Medium interference: same as training scenario
    medium = Config(**{
        **common,
        "ext_interferer_count": 3,      # 标准干扰源数量
        "ext_tx_power": 0.8,            # 标准发射功率
        "ext_activity_prob": 0.65,      # 标准活跃概率
        "ext_burst_prob": 0.15,         # 标准突发概率
        "ext_burst_scale": 3.0,         # 标准突发强度
        "test_seed": base.test_seed + 200,
    })

    # 重度干扰：模拟密集部署、强干扰场景
    # Heavy interference: dense deployment with strong interference
    heavy = Config(**{
        **common,
        "ext_interferer_count": 6,      # 多干扰源
        "ext_tx_power": 1.2,            # 高发射功率（超过网络内最大功率）
        "ext_activity_prob": 0.90,      # 高活跃概率（近乎持续干扰）
        "ext_burst_prob": 0.35,         # 高突发概率
        "ext_burst_scale": 6.0,         # 强突发（是噪声功率的 6 倍）
        "test_seed": base.test_seed + 300,
    })

    return {"Light": light, "Medium": medium, "Heavy": heavy}


def print_interference_group_table(group_results: Dict[str, Dict[str, float]]) -> None:
    """
    打印三组干扰场景的对比结果表格。
    Print comparison table for three interference scenarios.
    """
    print(f"\n{'='*86}")
    print("三组干扰场景对比 (Three Interference Scenarios: Light / Medium / Heavy)")
    print(f"{'='*86}")
    print(f"{'场景':<10} | {'算法':<14} | {'加权和速率':>14} | {'运行时间(ms)':>12}")
    print("-" * 58)

    for group_name in ["Light", "Medium", "Heavy"]:
        res = group_results[group_name]
        for algo in GROUP_PLOT_ALGO_ORDER:
            t_key = f"{algo} Time (ms)"
            print(
                f"{group_name:<10} | {algo:<14} | "
                f"{res.get(algo, 0.0):>14.6f} | {res.get(t_key, 0.0):>12.3f}"
            )
        print("-" * 58)


def plot_interference_group_comparison(
    group_results: Dict[str, Dict[str, float]],
    save_path: str = "interference_groups.png",
) -> None:
    """
    绘制三组干扰场景的和速率与运行时间对比柱状图。
    Plot bar chart comparing sum-rate and runtime across interference scenarios.

    Args:
        group_results: {场景名: 算法结果字典} 的嵌套字典
        save_path    : 图片保存路径
    """
    groups = ["Light", "Medium", "Heavy"]
    algos = GROUP_PLOT_ALGO_ORDER
    colors = [PLOT_ALGO_COLORS[k] for k in algos]

    x = np.arange(len(groups))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    fig.suptitle("三组干扰场景对比 (Three Interference Scenario Comparison)", fontsize=12)

    # ── 左图：和速率对比 ──────────────────────────────────────────────────
    for i, (algo, color) in enumerate(zip(algos, colors)):
        rates = [group_results[g].get(algo, 0.0) for g in groups]
        axes[0].bar(x + (i - 1.5) * width, rates, width=width, label=algo, color=color)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["轻度 Light", "中度 Medium", "重度 Heavy"])
    axes[0].set_ylabel("Average Weighted Sum Rate (bps/Hz)")
    axes[0].set_title("频谱效率对比（Sum-Rate）")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)

    # ── 右图：运行时间对比 ────────────────────────────────────────────────
    for i, (algo, color) in enumerate(zip(algos, colors)):
        times = [group_results[g].get(f"{algo} Time (ms)", 0.0) for g in groups]
        axes[1].bar(x + (i - 1.5) * width, times, width=width, label=algo, color=color)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["轻度 Light", "中度 Medium", "重度 Heavy"])
    axes[1].set_ylabel("Average Compute Time (ms/sample)")
    axes[1].set_title("计算延迟对比（Runtime）")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"  已保存: {save_path}")