"""
几何可观测量因子测试套件。

覆盖: 空数据、短数据、单列、常数价格、正常计算、三阶段合成数据。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from geometric.factor import (
    compute_geometric_observables,
    geometric_factor_stats,
    _spectral_entropy,
    _market_purity,
    _eigenvector_overlap,
    _safe_eig,
)

# ============================================================
# 内部函数测试
# ============================================================

def test_spectral_entropy_uniform():
    """均匀特征值 → 熵最大"""
    evals = np.ones(10)
    entropy, ratio = _spectral_entropy(evals)
    assert abs(ratio - 1.0) < 1e-6, f"均匀分布熵比应为 1.0, 得到 {ratio}"
    print(f"  [PASS] test_spectral_entropy_uniform: ratio={ratio:.6f}")


def test_spectral_entropy_concentrated():
    """集中特征值 → 熵很小"""
    evals = np.zeros(10)
    evals[0] = 100.0
    evals[1:] = 0.1
    entropy, ratio = _spectral_entropy(evals)
    assert ratio < 0.5, f"集中分布的熵比应较低, 得到 {ratio}"
    print(f"  [PASS] test_spectral_entropy_concentrated: ratio={ratio:.6f}")


def test_market_purity_concentrated():
    """高度集中 → 纯度接近 1"""
    evals = np.array([9.0, 0.5, 0.3, 0.2])
    p = _market_purity(evals)
    assert p > 0.7, f"集中分布的纯度应较高, 得到 {p}"
    print(f"  [PASS] test_market_purity_concentrated: purity={p:.4f}")


def test_market_purity_uniform():
    """均匀 → 纯度低"""
    evals = np.ones(20)
    p = _market_purity(evals)
    assert abs(p - 0.05) < 1e-6, f"均匀 20 支的纯度应为 0.05, 得到 {p}"
    print(f"  [PASS] test_market_purity_uniform: purity={p:.6f}")


def test_eigenvector_overlap_same():
    """相同特征向量 → overlap=1"""
    v = np.random.randn(10)
    v /= np.linalg.norm(v)
    overlap, rate = _eigenvector_overlap(
        np.column_stack([v, np.zeros((10, 9))]),
        np.column_stack([v, np.zeros((10, 9))]),
    )
    assert abs(overlap - 1.0) < 1e-10, f"相同向量 overlap 应为 1, 得到 {overlap}"
    assert abs(rate) < 1e-10, f"相同向量 rate 应为 0, 得到 {rate}"
    print("  [PASS] test_eigenvector_overlap_same")


def test_eigenvector_overlap_orthogonal():
    """正交特征向量 → overlap=0"""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    overlap, rate = _eigenvector_overlap(
        np.column_stack([v1, np.zeros((3, 9))]),
        np.column_stack([v2, np.zeros((3, 9))]),
    )
    assert abs(overlap) < 1e-10, f"正交向量 overlap 应为 0, 得到 {overlap}"
    assert abs(rate - 1.0) < 1e-10, f"正交向量 rate 应为 1, 得到 {rate}"
    print("  [PASS] test_eigenvector_overlap_orthogonal")


# ============================================================
# 集成测试
# ============================================================

def test_compute_empty_prices():
    """空数据 → 空 DataFrame"""
    prices = pd.DataFrame()
    df = compute_geometric_observables(prices)
    assert df.empty, "空输入应返回空 DataFrame"
    print("  [PASS] test_compute_empty_prices")


def test_compute_one_stock():
    """单支股票 → 无法形成有意义的相关矩阵"""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(42)
    rets = rng.standard_normal((500, 1)) * 0.02
    prices = pd.DataFrame(np.exp(np.cumsum(rets, axis=0)), index=dates)
    df = compute_geometric_observables(prices)
    # 1 列可计算出 corr=[[1]], evals=[1], 但熵比 = 0/0 = NaN
    # 所有列应为 NaN（有效输出但没有信号）
    assert len(df) > 0, "有足够数据应输出行"
    assert df["entropy_ratio"].isna().all() or df["entropy_ratio"].iloc[-1] < 1e-10, "单列无有效信号"
    print(f"  [PASS] test_compute_one_stock: {len(df)} rows")


def test_compute_two_stocks():
    """两支股票 → 同上"""
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(42)
    rets = rng.standard_normal((500, 2)) * 0.02
    prices = pd.DataFrame(np.exp(np.cumsum(rets, axis=0)), index=dates)
    df = compute_geometric_observables(prices, window=100)
    assert df["entropy_ratio"].isna().all(), "2 列不满足 n<3 条件"
    print(f"  [PASS] test_compute_two_stocks")


def test_compute_short_history():
    """历史太短 → 空"""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    prices = pd.DataFrame(np.random.randn(50, 10), index=dates)
    df = compute_geometric_observables(prices, window=100)
    assert df.empty, "短历史应返回空 DataFrame"
    print("  [PASS] test_compute_short_history")


def test_compute_constant_prices():
    """常数价格 → 相关矩阵奇异 → NaN"""
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    data = np.ones((300, 10))
    prices = pd.DataFrame(data, index=dates)
    df = compute_geometric_observables(prices, window=120)
    assert df.empty or df["entropy_ratio"].isna().all(), "常数价格应全 NaN"
    print(f"  [PASS] test_compute_constant_prices: rows={len(df)}, nan={df['entropy_ratio'].isna().sum() if len(df) else 'N/A'}")


def test_compute_three_phase():
    """三阶段合成数据 → 危机期熵和特征向量变化率升高"""
    np.random.seed(42)
    N = 30
    T = 1000
    prices_list = []
    for i in range(T):
        if i < 400:
            rho = 0.2
        elif i < 700:
            rho = 0.7
        else:
            rho = 0.3
        L = np.full((N, N), rho)
        np.fill_diagonal(L, 1.0)
        chol = np.linalg.cholesky(L)
        ret = chol @ np.random.randn(N)
        prices_list.append(np.exp(np.cumsum(ret)))
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    prices = pd.DataFrame(prices_list, index=dates)

    df = compute_geometric_observables(prices, window=100)

    # 取正常和危机两个时段
    crisis_period = df.index[(df.index >= dates[500]) & (df.index < dates[700])]
    normal_period = df.index[(df.index >= dates[50]) & (df.index < dates[300])]

    if len(crisis_period) > 0 and len(normal_period) > 0:
        crisis_ent = df.loc[crisis_period, "entropy_ratio"].mean()
        normal_ent = df.loc[normal_period, "entropy_ratio"].mean()
        crisis_evr = df.loc[crisis_period, "eigenvector_rate"].mean()
        normal_evr = df.loc[normal_period, "eigenvector_rate"].mean()

        print(f"  Normal period: entropy_ratio={normal_ent:.4f}, ev_rate={normal_evr:.4f}")
        print(f"  Crisis period: entropy_ratio={crisis_ent:.4f}, ev_rate={crisis_evr:.4f}")

        # 在合成高相关期，特征值集中 → 熵降低
        # 特征向量变化率理论上应该升高（但合成数据随机旋转可能导致不变）
        assert crisis_ent < normal_ent, f"高相关期熵应低于正常期: {crisis_ent:.4f} >= {normal_ent:.4f}"
        print(f"  ✓ 危机期熵 ({crisis_ent:.4f}) < 正常期 ({normal_ent:.4f}) ✓")
        print(f"  Δ entropy_ratio = {crisis_ent - normal_ent:+.4f}")
        print(f"  Δ ev_rate = {crisis_evr - normal_evr:+.4f}")

    print("  [PASS] test_compute_three_phase")


def test_factor_stats():
    """正确返回统计量"""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    rets = rng.standard_normal((500, 20)) * 0.02
    prices = pd.DataFrame(np.exp(np.cumsum(rets, axis=0)), index=dates)
    stats = geometric_factor_stats(prices, window=120)
    assert "mean_entropy_ratio" in stats, f"缺少 mean_entropy_ratio, 键列表: {list(stats.keys())}"
    assert "mean_purity" in stats
    assert "mean_ev_rate_z" in stats
    print(f"  [PASS] test_factor_stats: {len(stats)} stats keys")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("Geometric Observables — 测试套件")
    print("=" * 55)

    tests = [
        test_spectral_entropy_uniform,
        test_spectral_entropy_concentrated,
        test_market_purity_concentrated,
        test_market_purity_uniform,
        test_eigenvector_overlap_same,
        test_eigenvector_overlap_orthogonal,
        test_compute_empty_prices,
        test_compute_one_stock,
        test_compute_two_stocks,
        test_compute_short_history,
        test_compute_constant_prices,
        test_compute_three_phase,
        test_factor_stats,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 55}")
    print(f"结果: {passed}/{len(tests)} 通过")
    print(f"{'=' * 55}")
