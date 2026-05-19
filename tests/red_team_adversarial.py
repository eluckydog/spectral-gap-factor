"""红队对抗性测试"""
import numpy as np, pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from spectral_gap import compute_gap

def test_empty():
    """负例1: 空 DataFrame"""
    try:
        df = compute_gap(pd.DataFrame(), window=100)
        return f"OK, rows={len(df)}"
    except Exception as e:
        return f"CRASH: {e}"

def test_single_col():
    """负例2: 单列"""
    try:
        p = pd.DataFrame({'A': 100 * np.exp(np.random.randn(300).cumsum())})
        df = compute_gap(p, window=200)
        return f"OK, gap all NaN={df['spectral_gap'].isna().all()}"
    except Exception as e:
        return f"CRASH: {e}"

def test_constant():
    """负例3: 常数价格（零方差）"""
    try:
        p = pd.DataFrame(np.ones((300, 10)), columns=list('ABCDEFGHIJ'))
        df = compute_gap(p, window=200)
        bad = df['spectral_gap'].isna().sum()
        return f"OK, NaN gap count={bad}/{len(df)}"
    except Exception as e:
        return f"CRASH: {e}"

def test_short():
    """负例4: 数据短于窗口"""
    try:
        p = pd.DataFrame(100 * np.exp(np.random.randn(100, 10).cumsum(axis=0)))
        df = compute_gap(p, window=200)
        return f"OK, len={len(df)}"
    except Exception as e:
        return f"CRASH: {e}"

def test_large():
    """负例5: 100支股票"""
    try:
        p = pd.DataFrame(100 * np.exp(np.random.randn(500, 100).cumsum(axis=0)))
        df = compute_gap(p, window=252)
        nan_count = df['spectral_gap'].isna().sum()
        return f"OK, gap NaN={nan_count}/{len(df)}"
    except Exception as e:
        return f"CRASH: {e}"

if __name__ == "__main__":
    for name, fn in [("空DataFrame", test_empty),
                     ("单列", test_single_col),
                     ("常数价格", test_constant),
                     ("短数据", test_short),
                     ("100支股票", test_large)]:
        print(f"  {name}: {fn()}")
