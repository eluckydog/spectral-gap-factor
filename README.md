# Spectral Gap Factor · 谱隙因子

> Eigenvalue gap as a market structure signal — from RMT theory to factor code.

基于随机矩阵理论（RMT）的市场结构因子。

## ⚠️ 手感练习声明

本项目仅为**挖掘量化因子的手感练习**，未经实测验证。

- 合成数据测试通过，不代表实盘有效
- 未接入实际行情做回测
- 信号生成规则（Tracy-Widom 阈值）为启发式，未优化
- 不构成投资建议，使用后果自负

当你看到这个因子时，请理解：我们还在练习

## 核心思想

N 支股票的收益率相关矩阵作特征值分解：

```
λ₁ (最大)  → 市场因子
λ₂         → 最强板块/行业因子
λ₃⋯λk     → 次级因子
λk+1⋯λN   → 噪声
```

**谱隙** (λ₁ − λ₂) 衡量市场因子相对于最强板块因子的主导程度。

- 谱隙显著偏大 → 市场因子过度主导，分散化失效，系统性风险升高
- 谱隙收缩 → 板块/个股因子浮出水面，选股策略有空间

## 数学基础

| 理论 | 来源 | 作用 |
|------|------|------|
| Marčenko–Pastur (1967) | 随机矩阵谱理论 | 噪声特征值上界：σ²(1+√(N/T))² |
| Tracy–Widom (1994) | 随机矩阵最大特征值分布 | 谱隙显著性阈值 |
| Landon–Xian (2025) | arXiv:2509.14192 | 证明谱隙收敛到 TW 分布为 O(N^{-1+ε}) |

## 安装

```bash
pip install -r requirements.txt
```

或直接：

```bash
pip install numpy pandas scipy
```

## 使用

### 计算谱隙因子

```python
import pandas as pd
from spectral_gap import compute_gap

# 输入: index=日期, columns=股票代码
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

# 滚动窗口计算
df = compute_gap(prices, window=252)

print(df.tail())
#             spectral_gap  gap_zscore  market_dominance  signal
# 2025-01-15          3.21        1.45             0.28      -1
# 2025-01-16          2.87        0.92             0.25       0
# 2025-01-17          1.54       -0.88             0.18       1
```

### 信号含义

- `signal = -1`: 谱隙过大 → 减仓/对冲
- `signal =  0`: 正常状态
- `signal = +1`: 谱隙过小 → 选股窗口

### 统计摘要

```python
from spectral_gap import factor_stats
stats = factor_stats(df)
# {'signal_freq_1': 0.08, 'signal_freq_neg1': 0.12, 'mean_gap': 2.3, ...}
```

## 测试

```bash
cd tests
python test_factor.py
```

## 星核量化集成

在 `Strategy_Aggregator` 中作为正交信号源：

```yaml
spectral_gap:
  weight: 0.1          # 因与现有动量因子正交，权重较低
  threshold: 0.95       # Tracy-Widom 95% 分位
  conflict_mode: veto   # 与趋势信号冲突时, 谱隙信号有权否决
```

## 诚实声明

- 谱隙因子的数学框架来自经典 RMT 文献，非原创
- 论文 arXiv:2509.14192 提供了谱隙收敛率的**最优性证明**，未发明新因子
- 本仓库的价值在于：干净的工程实现 + 可复现的统计检验 + 星核量化无缝集成
- 如用此因子获利，请同时感谢 Potters、Bouchaud 和 Landon–Xian

## License

MIT
