
# Phase Gate — Pick the Asset (EUR/USD vs QQQ)

We run **both assets in Phase 1–2** with the same pipeline. At the end of Phase 2, we lock one in for the rest of the project.

## Decision Rule (end of Phase 2)
1) **Higher Sharpe** (per-regime model) on out-of-sample backtest.
2) Tie-breakers (in order):
   - Lower **Max Drawdown**.
   - More stable **regime assignments** (lower day-to-day switch rate).
   - **Simplicity wins**: if performance is close, pick the simpler model (tree > LSTM).

## What to Compare
- Predictive: Accuracy, F1, ROC-AUC (from 03/04 notebooks).
- Financial: Sharpe, Max DD, final equity, turnover (from 05 notebook).
- Stability: silhouette score, regime switch rate (from 02 notebook).

## Notebook Flow
1. `01_data_prep.ipynb`   → Aligned prices.
2. `02_features_regimes.ipynb` → Features, **EDA**, regimes (+ silhouette).
3. `03_models_baseline.ipynb`  → Baseline RF/XGB (rolling CV).
4. `04_models_per_regime.ipynb`→ Per-regime RF/XGB (fallback if regime thin).
5. `05_backtest_eval.ipynb`    → Equity curves, Sharpe, MDD, leaderboard.
6. *(Optional)* `06_models_lstm.ipynb` → LSTM baseline for comparison.

## CLI Cheatsheet
```bash
# EUR/USD
python run.py --asset eurusd

# QQQ
python run.py --asset qqq
```

## Quick Interpretation Tips
- If per-regime **Sharpe** is higher **and** Max DD is meaningfully better (≥ 15% improvement), regimes are paying off.
- If regime silhouette < 0.25 or switch rate is high, consider fewer regimes (k=2) or different feature subset.
- If LSTM is required to “win,” check overfitting risk—prefer trees unless LSTM gives a clear stability edge.
