"""
Spectral Gap Factor — 基于随机矩阵理论的谱间信号。

核心思想：
  N 支股票收益率相关矩阵的特征值分解中，
  λ₁（市场因子）与 λ₂（最强板块因子）之间的间隙
  提供了市场结构信息。

理论依据：
  - Marčenko-Pastur (1967): 纯噪声矩阵的特征值分布
  - Tracy-Widom (1994): 最大特征值的极限分布
  - Landon–Xian (2025, arXiv:2509.14192):
    证明谱间隙收敛到 Tracy-Widom 的速度为 O(N^{-1+ε})，
    即谱隙信号在样本量充分大时具有最优统计效率。

输入:
  - prices: DataFrame, index=日期, columns=股票代码, values=收盘价
  - window: 滚动窗口天数 (default 252)
  - market_trim: 去掉多少比例的特征值作为噪声估计 (default 0.5)

输出:
  - SpectralGap: 标准化谱隙 = (λ₁ - λ₂) / σ_noise
  - GapZScore: 相对于滚动均值的 z-score
  - MarketDominance: 市场因子解释方差比例
  - Signal: -1(结构高度集中/减仓) / 0(正常) / +1(分散化/选股空间)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def _mp_noise_bound(n: int, t: int, q: Optional[float] = None) -> float:
    """Marčenko-Pastur 噪声上界: σ²(1 + √(N/T))²."""
    q = q or n / t
    return (1.0 + np.sqrt(q)) ** 2


def _tw_beta1_threshold(p: float) -> float:
    """Tracy-Widom β=1 (实对称) 近似分位数。
    
    来源: 数值拟合 Tracy-Widom F₁ 分布。
    精确值见 Bornemann (2010) 或 TW 表。
    """
    # F₁ 分布右尾近似分位数（样条插值）
    _quantiles = {0.50: -1.52, 0.75: -0.79, 0.90: -0.30,
                  0.95: 0.00, 0.99: 0.65, 0.999: 1.50}
    keys, vals = zip(*sorted(_quantiles.items()))
    return float(np.interp(p, keys, vals))


def estimate_noise_std(eigenvalues: np.ndarray, trim: float = 0.5) -> float:
    """稳健噪声标准差估计: 取特征值分布的中位数区域。"""
    if len(eigenvalues) < 2:
        return 1.0
    n = len(eigenvalues)
    lo = int(n * (1 - trim) / 2)
    hi = int(n * (1 + trim) / 2)
    hi = min(hi, n)
    lo = max(lo, 0)
    if hi <= lo:
        return float(np.median(eigenvalues))
    trimmed = eigenvalues[lo:hi]
    return float(np.median(trimmed))


def compute_gap(
    prices: pd.DataFrame,
    window: int = 252,
    trim: float = 0.5,
) -> pd.DataFrame:
    """计算滚动谱隙因子。

    Args:
        prices: OHLC 数据，取收盘价。index=日期, columns=股票代码
        window: 滚动窗口天数
        trim: 噪声估计时去掉的特征值比例

    Returns:
        DataFrame:
          - spectral_gap: (λ₁ - λ₂) / σ_noise，标准化谱隙
          - gap_zscore: 相对滚动均值的 z-score
          - market_dominance: λ₁ / sum(λ_i)，市场因子占比
          - signal: -1(集中) / 0(正常) / +1(分散)
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    n_stocks = prices.shape[1]
    results = []

    for i in range(window, len(log_returns)):
        chunk = log_returns.iloc[i - window:i]
        if chunk.empty or chunk.shape[1] < 3:
            results.append({
                "spectral_gap": np.nan, "gap_zscore": np.nan,
                "market_dominance": np.nan, "signal": 0
            })
            continue

        # 相关矩阵 → 特征值
        corr = chunk.corr().values
        evals = np.linalg.eigvalsh(corr)[::-1]  # 降序

        if len(evals) < 2:
            results.append({
                "spectral_gap": np.nan, "gap_zscore": np.nan,
                "market_dominance": np.nan, "signal": 0
            })
            continue

        lam1, lam2 = evals[0], evals[1]
        noise_std = estimate_noise_std(evals[2:], trim)
        gap = (lam1 - lam2) / max(noise_std, 1e-10)

        # 市场因子解释方差比例
        dominance = lam1 / max(evals.sum(), 1e-10)

        results.append({
            "spectral_gap": gap,
            "gap_zscore": 0.0,  # 后面统一算 z-score
            "market_dominance": dominance,
            "signal": 0,
        })

    df = pd.DataFrame(results, index=log_returns.index[window:])

    # 滚动 z-score
    if len(df) > 60:
        df["gap_zscore"] = (
            (df["spectral_gap"] - df["spectral_gap"].rolling(60, min_periods=20).mean())
            / df["spectral_gap"].rolling(60, min_periods=20).std().clip(lower=1e-6)
        )

    # 信号生成
    tw_95 = _tw_beta1_threshold(0.95)
    df["signal"] = 0
    df.loc[df["gap_zscore"] > tw_95, "signal"] = -1   # 集中度过高
    df.loc[df["gap_zscore"] < -tw_95, "signal"] = 1    # 分散化开启

    return df


def factor_stats(df: pd.DataFrame) -> dict:
    """返回因子统计摘要。"""
    sig = df["signal"].dropna()
    if len(sig) == 0:
        return {"error": "no data"}
    return {
        "signal_freq_1": float((sig == 1).mean()),   # 分散信号占比
        "signal_freq_neg1": float((sig == -1).mean()),  # 集中信号占比
        "signal_freq_0": float((sig == 0).mean()),     # 中性占比
        "mean_gap": float(df["spectral_gap"].mean()),
        "std_gap": float(df["spectral_gap"].std()),
        "mean_dominance": float(df["market_dominance"].mean()),
    }
