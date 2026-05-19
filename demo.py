"""
Spectral Gap Factor Demo — 合成数据演示
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from spectral_gap import compute_gap, factor_stats

# ── 用例 1: 强市场因子 ──
print("=" * 55)
print("用例 1: 单一市场因子 (80% 方差)")
print("=" * 55)
np.random.seed(0)
market = np.random.randn(600)
noise = np.random.randn(600, 100) * 0.3
rets = market[:, None] * 0.8 + noise
prices = pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)))
df = compute_gap(prices, window=252)
print(f"  平均谱隙:     {df['spectral_gap'].mean():.2f}")
print(f"  市场因子占比: {df['market_dominance'].mean():.2%}")
print(f"  -1 信号占比:  {factor_stats(df)['signal_freq_neg1']:.1%}")
print()

# ── 用例 2: 多因子均衡 ──
print("=" * 55)
print("用例 2: 3 个均衡因子")
print("=" * 55)
factors = np.random.randn(600, 3) * 0.5
loadings = np.random.randn(3, 100) * 0.6
noise = np.random.randn(600, 100) * 0.3
rets = factors @ loadings + noise
prices = pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)))
df = compute_gap(prices, window=252)
print(f"  平均谱隙:     {df['spectral_gap'].mean():.2f}")
print(f"  市场因子占比: {df['market_dominance'].mean():.2%}")
print(f"  +1 信号占比:  {factor_stats(df)['signal_freq_1']:.1%}")
print()

# ── 用例 3: 纯噪声 ──
print("=" * 55)
print("用例 3: 纯噪声 (无结构)")
print("=" * 55)
rets = np.random.randn(600, 100)
prices = pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)))
df = compute_gap(prices, window=252)
print(f"  平均谱隙:     {df['spectral_gap'].mean():.2f}")
print(f"  市场因子占比: {df['market_dominance'].mean():.2%}")
stats = factor_stats(df)
print(f"  +1: {stats['signal_freq_1']:.1%}  -1: {stats['signal_freq_neg1']:.1%}  中性: {stats['signal_freq_0']:.1%}")
print()

print("✅ Demo 完成")
