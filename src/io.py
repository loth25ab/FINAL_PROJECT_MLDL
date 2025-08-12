import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_yf(ticker, start, end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]  # just the OHLCV field name

    df = df.rename_axis("Date").reset_index()
    return df

def load_asset_map(cfg_asset):
    assets = {"target": cfg_asset["target"]["yf"]}
    for k,v in cfg_asset["secondary"].items():
        assets[k] = v["yf"]
    if "bench" in cfg_asset:
        assets["bench"] = cfg_asset["bench"]["yf"]
    return assets

def download_all(cfg_asset, start, end):
    out = {}
    for name, ticker in load_asset_map(cfg_asset).items():
        out[name] = fetch_yf(ticker, start, end)
    return out

def align_and_ffill(dfs: dict, price_col: str = "Close", target_key: str = "target"):
    """
    dfs: {"target": df_target, "dxy": df_dxy, ...}
    Each df should have Date + OHLCV (some may miss Volume).
    """
    merged = None
    for name, df in dfs.items():
        sub = df.copy()

        # 1) Flatten MultiIndex columns if present (yfinance multi-ticker style)
        if isinstance(sub.columns, pd.MultiIndex):
            sub.columns = ["_".join([str(x) for x in col if x is not None]) for col in sub.columns]

        # 2) Normalize expected column names and keep what exists
        #    (some series won't have Volume, etc.)
        possible = ["Date", "Open", "High", "Low", price_col, "Volume"]
        keep = [c for c in possible if c in sub.columns]
        if "Date" not in keep:
            raise KeyError(f"'Date' column missing for series '{name}'")
        sub = sub[keep].copy()

        # 3) Prefix non-Date cols with the series key
        sub.rename(columns={c: f"{name}_{c}" for c in sub.columns if c != "Date"}, inplace=True)

        # 4) Merge outer on Date
        merged = sub if merged is None else merged.merge(sub, on="Date", how="outer")

    # 5) Sort + forward fill
    merged = merged.sort_values("Date").reset_index(drop=True).ffill()

    # 6) Require target close present (guard non-string col names)
    targ_close = [
        c for c in merged.columns
        if isinstance(c, str) and c.startswith(f"{target_key}_") and c.endswith(price_col)
    ]
    if not targ_close:
        raise KeyError(
            f"No target close columns found. "
            f"Have columns like: {list(merged.columns)[:10]} ..."
        )

    return merged.dropna(subset=targ_close)
