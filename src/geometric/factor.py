"""
Geometric Observables — 基于相关矩阵谱几何的市场结构因子。

来源: Hammond (2026) "Geometric Observables for Financial Regime Detection" (arXiv:2605.17117)

从论文中提取了三个可实现的可观测量，加上一个合成信号：

  1. Spectral Entropy (H) — 特征值分布的香农熵
  2. Market Purity (P) — Reduced State Purity 的简化版：特征值归一化平方和
  3. Eigenvector Rate (EVR) — 主特征向量夹角变化率（Berry Phase Rate 的平价替代）
  4. Geometric Regime Score (GRS) — 三通道合成信号

与谱隙因子的关系：
  - 谱隙 = (λ₁ - λ₂) / σ_noise：侧重最大两个特征值之间的间距
  - 谱熵 = -Σ w_i log(w_i)：关注整个特征值分布的形状
  - 市场纯度 = Σ w_i²：与 MarketDominance 类似但用到全部特征值
  - 三者互补：间隙告诉你前两个特征值的分离，熵和纯度告诉你分布的整体结构

用法:
  from geometric import compute_geometric_observables
  df = compute_geometric_observables(prices)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict

_COLS = [
    "spectral_entropy",        
    "entropy_ratio",           
    "market_purity",           
    "eigenvector_overlap",     
    "eigenvector_rate",        
    "eigenvector_rate_z",      
    "geometric_regime_score",  
]

def _nan_row() -> Dict[str, float]:
    return {c: np.nan for c in _COLS}


def _safe_eig(prices_sub: pd.DataFrame) -> tuple:
    """安全计算相关矩阵及其特征分解。"""
    n = prices_sub.shape[1]
    if n < 3:
        return np.array([]), np.array([]), None
    try:
        # 快速检查：如果收益率标准差全为0，相关矩阵不可分解
        if chunk.std().min() < 1e-12:
            return np.array([]), np.array([]), None
    except Exception:
        pass
    try:
        corr = prices_sub.corr().values
        if np.isnan(corr).any() or np.isinf(corr).any():
            return np.array([]), np.array([]), None
        evals, evects = np.linalg.eigh(corr)
        # eigh 返回升序
        evals = evals[::-1]
        evects = evects[:, ::-1]
        return evals, evects, corr
    except np.linalg.LinalgError:
        return np.array([]), np.array([]), None


def _spectral_entropy(evals: np.ndarray) -> tuple:
    """计算谱熵和谱熵比（归一化到 [0,1]）。"""
    total = evals.sum()
    if total <= 0:
        return 0.0, 0.0
    w = evals / total
    entropy = -np.sum(w * np.log(w + 1e-20))
    max_entropy = np.log(len(w))
    ratio = entropy / max_entropy if max_entropy > 0 else 1.0
    return entropy, ratio


def _market_purity(evals: np.ndarray) -> float:
    """计算市场纯度（对应论文 Reduced State Purity）。
    
    公式: P = Σ (λ_i / Σ λ_j)² = 特征值归一化平方和
    
    解释:
      - P → 1：市场由一个因子主导（集中/危机前兆）
      - P → 0：市场完全分散（无结构性/正常）
    """
    total = evals.sum()
    if total <= 0:
        return 0.0
    w = evals / total
    return float(np.sum(w ** 2))


def _eigenvector_overlap(
    evect_t: np.ndarray,
    evect_t1: Optional[np.ndarray],
) -> tuple:
    """计算主特征向量重叠度 = |v₁(t)·v₁(t-1)|。
    
    Returns:
        overlap: 重叠度 (0-1)，1=无变化，0=完全翻转
        rate: 变化率 = 1 - overlap，越大表示结构变化越剧烈
    """
    if evect_t1 is None:
        return 1.0, 0.0
    v1 = evect_t[:, 0]
    v0 = evect_t1[:, 0]
    if v1.shape != v0.shape:
        return 1.0, 0.0
    overlap = float(abs(np.dot(v1, v0)))
    rate = 1.0 - overlap
    return overlap, rate


def compute_geometric_observables(
    prices: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """计算滚动几何可观测量。

    Args:
        prices: OHLC 数据，取收盘价。index=日期, columns=股票代码
        window: 滚动窗口天数

    Returns:
        DataFrame:
          - spectral_entropy: 谱熵 S = -Σ w_i log(w_i)，w_i = λ_i / Σ λ_j
          - entropy_ratio: S / log(N)，归一化到 [0,1]
          - market_purity: 市场纯度 Σ w_i²（论文 Reduced State Purity 的 RMT 版）
          - eigenvector_overlap: 主特征向量重叠度 |v₁(t)·v₁(t-1)|
          - eigenvector_rate: 变化率 = 1 - overlap
          - eigenvector_rate_z: eigenvector_rate 的 z-score
          - geometric_regime_score: 三通道合成的综合信号
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) <= window:
        df = pd.DataFrame([], index=pd.Index([], dtype=log_returns.index.dtype), columns=_COLS)
        return df

    results = []
    prev_evects = None

    for i in range(window, len(log_returns)):
        chunk = log_returns.iloc[i - window:i]
        if chunk.empty or chunk.shape[1] < 3:
            results.append(_nan_row())
            prev_evects = None
            continue

        evals, evects, _ = _safe_eig(chunk)
        if len(evals) < 3:
            results.append(_nan_row())
            prev_evects = None
            continue

        # 1. 谱熵
        entropy, ratio = _spectral_entropy(evals)

        # 2. 市场纯度
        purity = _market_purity(evals)

        # 3. 特征向量变化
        overlap, ev_rate = _eigenvector_overlap(evects, prev_evects)
        prev_evects = evects

        results.append({
            "spectral_entropy": entropy,
            "entropy_ratio": ratio,
            "market_purity": purity,
            "eigenvector_overlap": overlap,
            "eigenvector_rate": ev_rate,
            "eigenvector_rate_z": 0.0,
            "geometric_regime_score": 0.0,
        })

    df = pd.DataFrame(results, index=log_returns.index[window:])

    # 特征向量变化率的 z-score
    if len(df) > 60:
        rolling_er = df["eigenvector_rate"].rolling(60, min_periods=20)
        df["eigenvector_rate_z"] = (
            (df["eigenvector_rate"] - rolling_er.mean())
            / rolling_er.std().clip(lower=1e-6)
        )

    # 合成综合信号（等权标准分聚合）
    # 注意：熵在危机期下降（高相关→特征值集中），
    #       但特征向量变化率在危机上升。
    # 所以合成信号需要标记方向：
    #   正分 = 结构变化（危机或转变）
    #   负分 = 结构稳定（正常）
    if len(df) > 60:
        z_entropy = -(
            (df["entropy_ratio"] - df["entropy_ratio"].expanding().mean())
            / df["entropy_ratio"].expanding().std().clip(lower=1e-6)
        )  # 取负号：熵降=危机
        z_purity = (
            (df["market_purity"] - df["market_purity"].expanding().mean())
            / df["market_purity"].expanding().std().clip(lower=1e-6)
        )
        df["geometric_regime_score"] = (z_entropy + z_purity + df["eigenvector_rate_z"]) / 3.0

    return df


def geometric_factor_stats(
    prices: pd.DataFrame,
    window: int = 252,
) -> Dict[str, float]:
    """计算因子的统计摘要。"""
    df = compute_geometric_observables(prices, window)
    if df.empty or "entropy_ratio" not in df.columns:
        return {"error": "no data"}

    gs = df["geometric_regime_score"].dropna()
    er = df["entropy_ratio"].dropna()
    mp = df["market_purity"].dropna()
    evz = df["eigenvector_rate_z"].dropna()

    return {
        "n_obs": len(df),
        "mean_entropy_ratio": float(er.mean()),
        "std_entropy_ratio": float(er.std()),
        "mean_purity": float(mp.mean()),
        "std_purity": float(mp.std()),
        "mean_ev_rate_z": float(evz.mean()),
        "regime_alarm_pct": float((gs > 1.5).mean()) if len(gs) > 0 else 0.0,
        "calm_pct": float((gs < -0.5).mean()) if len(gs) > 0 else 0.0,
    }
