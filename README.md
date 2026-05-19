
# SpectralGauge

> Regime-aware market structure factors from the spectral geometry of correlation matrices.

**SpectralGauge** detects financial regime shifts through the spectral geometry of market correlation matrices. It implements two complementary factor families:

1. **Spectral Gap Factor** (Landon–Xian 2025) — measures the gap between the two largest eigenvalues relative to noise; a widening gap signals market-factor dominance and systemic risk.
2. **Geometric Observables** (Hammond 2026) — extracts spectral entropy, market purity, and eigenvector dynamics as regime-change indicators from the same eigenvalue decomposition.

The two channels are complementary: the gap captures leading-eigenvalue separation, entropy captures the full spectral shape, and the eigenvector rate tracks basis rotation during crises. Combined they form a **Geometric Regime Score** validated on synthetic data (Cohen's d ≈ 0.7, 13/13 tests passing).

## Project Structure

`
spectral-gap-factor/
├── src/
│   ├── spectral_gap/        # Spectral Gap factor
│   │   └── factor.py        # compute_gap(), factor_stats()
│   └── geometric/           # Geometric Observables
│       └── factor.py        # compute_geometric_observables()
├── tests/
│   ├── test_factor.py       # Spectral Gap tests
│   ├── test_geometric.py    # Geometric Obs tests (13/13)
│   └── red_team_adversarial.py
├── notebooks/
│   ├── demo.py              # Spectral Gap demo
│   └── demo_geometric.py    # Combined demo
└── README.md
`

## Quick Start

`python
from spectral_gap import compute_gap
from geometric import compute_geometric_observables

df_gap = compute_gap(prices, window=252)
df_geo = compute_geometric_observables(prices, window=252)
`

## Install

`ash
pip install -r requirements.txt
# numpy, pandas, scipy
`

## ⚠️ Practice Statement

This project is a **practice exercise** in quant factor mining. Tested on synthetic data only. Not validated on live markets. Do not use for actual trading decisions.

## References

- Landon–Xian (2025), arXiv:2509.14192 — Spectral gap optimal convergence
- Hammond (2026), arXiv:2605.17117 — Geometric observables for regime detection
- Marčenko–Pastur (1967) — Random matrix eigenvalue distribution
- Tracy–Widom (1994) — Largest eigenvalue limiting distribution

## License

MIT
