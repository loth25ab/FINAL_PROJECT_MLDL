import pandas as pd
import numpy as np

def _pct_ret(s): return s.pct_change()

def add_price_features(df, prefix):
    c = f"{prefix}_Close"; h=f"{prefix}_High"; l=f"{prefix}_Low"
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_ret1"] = df[c].pct_change()
    for w in [5,10,20,60]:
        out[f"{prefix}_ret{w}"] = df[c].pct_change(w)
        out[f"{prefix}_vol{w}"] = df[c].pct_change().rolling(w).std() * np.sqrt(252)
        out[f"{prefix}_ma{w}"]  = df[c].rolling(w).mean()/df[c]-1
    # RSI (14)
    delta = df[c].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (down.replace(0, np.nan))
    out[f"{prefix}_rsi14"] = 100 - (100/(1+rs))
    # MACD (12,26,9)
    ema12 = df[c].ewm(span=12, adjust=False).mean()
    ema26 = df[c].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out[f"{prefix}_macd"] = macd
    out[f"{prefix}_macdsig"] = signal
    # ATR (14)
    tr = np.maximum(df[h]-df[l], np.maximum((df[h]-df[c].shift()).abs(), (df[l]-df[c].shift()).abs()))
    out[f"{prefix}_atr14"] = tr.rolling(14).mean() / df[c]
    # Bollinger (20,2)
    m = df[c].rolling(20).mean(); s = df[c].rolling(20).std()
    out[f"{prefix}_bb_pos"] = (df[c]-m)/(2*s.replace(0, np.nan))
    return out

def cross_asset_features(df, asset_names):
    feats = []
    for name in asset_names:
        feats.append(add_price_features(df, name))
    X = pd.concat(feats, axis=1)
    # rolling correlations with target
    t_ret = df["target_Close"].pct_change()
    for name in asset_names:
        if name=="target": continue
        X[f"corr_{name}_20"] = t_ret.rolling(20).corr(df[f"{name}_Close"].pct_change())
        X[f"corr_{name}_60"] = t_ret.rolling(60).corr(df[f"{name}_Close"].pct_change())
    return X

def make_supervised(df, lookahead=1, target_kind="sign"):
    y_ret = df["target_Close"].pct_change(lookahead).shift(-lookahead)
    if target_kind == "sign":
        y = (y_ret > 0).astype(int)*2 - 1   # +1 / -1
    else:
        y = y_ret
    return y
