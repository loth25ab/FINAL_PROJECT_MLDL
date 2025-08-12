# ML-Regimes (FX)

**What:** Predict EUR/USD moves with models that adapt to **market regimes** (global baseline vs per-regime vs LSTM).  
**Why:** Stability > bragging rights. Regime-aware models should cut whipsaws and drawdowns.  
**Where:** **Built to run on Google Colab** to leverage free compute + Drive.

## Run order (notebooks)
1) `01_data_prep.ipynb` → aligned price panel  
2) `02_features_regimes.ipynb` → features + **regime labels**  
3) `03_models_baseline.ipynb` → global baseline  
4) `04_models_per_regime.ipynb` → per-regime models  
5) `05_models_LSTM.ipynb` → sequence model  
6) `06_backtest_eval.ipynb` → backtest
