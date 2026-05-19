"""
Geometric Observables 演示脚本。

对比谱隙因子与几何可观测量在合成数据上的表现。
论文来源: Hammond (2026) arXiv:2605.17117
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from spectral_gap.factor import compute_gap, factor_stats
from geometric.factor import compute_geometric_observables, geometric_factor_stats

np.random.seed(42)
N_STOCKS = 50
N_DAYS = 1500

print("=" * 65)
print("Geometric Observables vs Spectral Gap — 合成数据对比")
print("=" * 65)

# --- 合成数据：三阶段 ---
# 0-500: 正常市场（低相关性，~0.2）
# 500-800: 危机前（相关性上升，~0.6）
# 800-1100: 危机（高相关性~0.8 + 波动上升）
# 1100-1500: 恢复

prices_list = []
for i in range(N_DAYS):
    if i < 500:
        rho = 0.2
        vol = 0.015
    elif i < 800:
        rho = 0.4 + 0.002 * (i - 500)  # 渐变
        vol = 0.018
    elif i < 1100:
        rho = 0.8
        vol = 0.030
    else:
        rho = 0.3
        vol = 0.015

    L = np.full((N_STOCKS, N_STOCKS), rho)
    np.fill_diagonal(L, 1.0)
    chol = np.linalg.cholesky(L)
    ret = chol @ np.random.randn(N_STOCKS)
    prices_list.append(np.exp(np.cumsum(ret)))

dates = pd.date_range("2020-01-01", periods=N_DAYS, freq="B")
prices = pd.DataFrame(prices_list, index=dates)

print(f"\n合成数据: {N_STOCKS} 支股票 x {N_DAYS} 交易日")
print(f"阶段: 正常(0-500) → 渐变(500-800) → 危机(800-1100) → 恢复(1100-1500)")

# --- 计算因子 ---
WINDOW = 120

sg = compute_gap(prices, window=WINDOW)
geo = compute_geometric_observables(prices, window=WINDOW)

# --- 对比 ---
# 合并到同一时间轴
joint = sg.join(geo, how="inner")
print(f"\n共用时间点: {len(joint)}")

# 各阶段对比
crisis_idx = joint.index[(joint.index >= dates[800]) & (joint.index < dates[1100])]
normal_idx = joint.index[(joint.index < dates[500])]

print("\n--- 正常期 (0-500) vs 危机期 (800-1100) 均值对比 ---")
for factor_name in ["spectral_gap", "market_dominance", "entropy_ratio", "market_purity", "eigenvector_rate"]:
    if factor_name not in joint.columns:
        continue
    n_mean = joint.loc[normal_idx, factor_name].mean()
    c_mean = joint.loc[crisis_idx, factor_name].mean()
    delta = c_mean - n_mean
    print(f"  {factor_name:25s}: 正常 {n_mean:8.4f} | 危机 {c_mean:8.4f} | Δ {delta:8.4f}")

# --- 三通道合成信号在危机期的表现 ---
print(f"\n--- Geometric Regime Score 在危机期的表现 ---")
if "geometric_regime_score" in joint.columns:
    gs_crisis = joint.loc[crisis_idx, "geometric_regime_score"]
    gs_normal = joint.loc[normal_idx, "geometric_regime_score"]
    if len(gs_crisis) > 0 and len(gs_normal) > 0:
        d_num = gs_crisis.mean() - gs_normal.mean()
        d_den = np.sqrt((gs_crisis.var() + gs_normal.var()) / 2)
        cohens_d = d_num / d_den if d_den > 0 else 0
        print(f"  Crisis mean z-score: {gs_crisis.mean():.4f}")
        print(f"  Normal mean z-score: {gs_normal.mean():.4f}")
        print(f"  Effect size (Cohen's d): {cohens_d:.4f}")

print(f"\n--- 统计摘要 ---")
try:
    fs = factor_stats(prices)
    print(f"  谱隙因子 signal 分布: +1={fs.get('signal_freq_1',0):.1%} | -1={fs.get('signal_freq_neg1',0):.1%} | 0={fs.get('signal_freq_0',0):.1%}")
except Exception:
    print("  谱隙因子统计: 跳过（从 raw 价格调用可能失败）")
gs = geometric_factor_stats(prices)
if 'error' not in gs:
    print(f"  几何信号 regime_alarm: {gs.get('regime_alarm_pct',0):.1%}")
    print(f"  平均谱熵比: {gs.get('mean_entropy_ratio',0):.4f}")
    print(f"  平均市场纯度: {gs.get('mean_purity',0):.4f}")
else:
    print(f"  几何信号统计: {gs}")

print("\n✅ 完成")
