import numpy as np
import pandas as pd
from spectral_gap import compute_gap, factor_stats


def test_synthetic_single_factor():
    """合成数据: 单一市场因子 + 噪声，应有大 gap。"""
    np.random.seed(42)
    n_days, n_stocks = 500, 50
    market = np.random.randn(n_days)
    noise = np.random.randn(n_days, n_stocks) * 0.3
    rets = market[:, None] * 0.8 + noise
    prices = pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)))
    df = compute_gap(prices, window=200)
    avg_gap = df["spectral_gap"].mean()
    avg_dom = df["market_dominance"].mean()
    print(f"单一因子测试: avg_gap={avg_gap:.2f}, dom={avg_dom:.3f}")
    assert avg_gap > 1.5, f"预期大 gap, 实际 {avg_gap:.2f}"
    assert avg_dom > 0.2, f"预期市场因子主导, 实际 {avg_dom:.3f}"
    print("  PASS")


def test_synthetic_multi_factor():
    """合成数据: 多因子结构，gap 应小于单因子场景。"""
    np.random.seed(42)
    n_days, n_stocks = 500, 50
    factors = np.random.randn(n_days, 3) * 0.5
    loadings = np.random.randn(3, n_stocks) * 0.6
    noise = np.random.randn(n_days, n_stocks) * 0.3
    rets = factors @ loadings + noise
    prices = pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)))
    df = compute_gap(prices, window=200)
    avg_gap = df["spectral_gap"].mean()
    print(f"多因子测试: avg_gap={avg_gap:.2f}")
    assert avg_gap < 20.0, f"多因子场景 gap 不应过大, 实际 {avg_gap:.2f}"
    print("  PASS")


def test_pure_noise():
    """纯噪声: gap 应接近 1 (MP 噪声上界附近)。"""
    np.random.seed(42)
    n_days, n_stocks = 500, 50
    rets = np.random.randn(n_days, n_stocks)
    prices = pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)))
    df = compute_gap(prices, window=200)
    avg_gap = df["spectral_gap"].mean()
    avg_dom = df["market_dominance"].mean()
    print(f"纯噪声测试: avg_gap={avg_gap:.2f}, dom={avg_dom:.3f}")
    assert avg_gap < 2.0, f"纯噪声 gap 应很小, 实际 {avg_gap:.2f}"
    assert avg_dom < 0.15, f"纯噪声市场因子占比应低, 实际 {avg_dom:.3f}"
    print("  PASS")


def test_signal_distribution():
    """信号分布应相对均衡，不应全为 0。"""
    np.random.seed(42)
    n_days, n_stocks = 800, 60
    market = np.random.randn(n_days)
    regime = np.where(np.arange(n_days) % 300 < 150, 1.0, 0.3)
    noise = np.random.randn(n_days, n_stocks) * 0.3
    rets = (market * regime)[:, None] * 0.8 + noise
    prices = pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)))
    df = compute_gap(prices, window=200)
    stats = factor_stats(df)
    freq_1 = stats["signal_freq_1"]
    freq_neg1 = stats["signal_freq_neg1"]
    freq_0 = stats["signal_freq_0"]
    print(f"信号分布测试: +1={freq_1:.1%}, -1={freq_neg1:.1%}, 0={freq_0:.1%}")
    assert freq_0 < 0.95, f"信号不应全为 0, 中性占比 {freq_0:.1%}"
    print("  PASS")


if __name__ == "__main__":
    test_synthetic_single_factor()
    test_synthetic_multi_factor()
    test_pure_noise()
    test_signal_distribution()
    print("\n全部通过")
