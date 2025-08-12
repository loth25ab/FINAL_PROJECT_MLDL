import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Tuple, Union, Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# HMM support
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    _HMM_OK = True
except Exception:
    _HMM_OK = False


@dataclass
class RegimeMeta:
    method: str
    k: int
    score_name: str
    score_value: float


def _ensure_2d(X: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Union[pd.Index, None]]:
    """Return numeric 2D numpy array and original index (if DataFrame)."""
    if isinstance(X, pd.DataFrame):
        Xn = X.select_dtypes(include=[np.number]).copy()
        idx = Xn.index
        return Xn.values, idx
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X, None


def _min_run_smooth(labels: np.ndarray, min_len: int = 5) -> np.ndarray:
    """Collapse regime runs shorter than min_len into neighboring runs."""
    if min_len <= 1:
        return labels
    y = labels.copy()
    n = len(y)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and y[j + 1] == y[i]:
            j += 1
        run_len = j - i + 1
        if run_len < min_len:
            left = y[i - 1] if i > 0 else None
            right = y[j + 1] if j + 1 < n else None
            if left is None and right is None:
                pass
            elif left is None:
                y[i:j + 1] = right
            elif right is None:
                y[i:j + 1] = left
            else:
                # choose neighbor with longer contiguous span
                li = i - 1
                while li - 1 >= 0 and y[li - 1] == left:
                    li -= 1
                left_len = i - li
                rj = j + 1
                while rj + 1 < n and y[rj + 1] == right:
                    rj += 1
                right_len = rj - (j + 1) + 1
                y[i:j + 1] = left if left_len >= right_len else right
        i = j + 1
    return y


def _bic_gaussian_hmm(model: "GaussianHMM", X: np.ndarray) -> float:
    """BIC = -2*loglik + p*log(n), approximate param count for diag covars."""
    loglik = model.score(X)
    n, d = X.shape
    k = model.n_components
    p = (k - 1) + (k * (k - 1)) + (k * d) + (k * d)  # start + trans + means + diag covars
    return -2.0 * loglik + p * np.log(max(n, 1))


def _fit_best_hmm(X: np.ndarray, k_range=(2, 3, 4, 5), cov_type="diag", n_init=5, random_state=42):
    best_model = None
    best_bic = np.inf
    for k in k_range:
        for seed in range(n_init):
            model = GaussianHMM(
                n_components=k,
                covariance_type=cov_type,
                random_state=(random_state + seed),
                n_iter=200,
                tol=1e-3
            )
            try:
                model.fit(X)
                bic = _bic_gaussian_hmm(model, X)
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
            except Exception:
                continue
    if best_model is None:
        raise RuntimeError("HMM fitting failed for all initializations.")
    labels = best_model.predict(X)
    meta = RegimeMeta(method="hmm", k=best_model.n_components, score_name="bic", score_value=float(best_bic))
    return labels, meta


def detect_regimes(
    X: Union[pd.DataFrame, np.ndarray],
    methods: Union[str, Iterable[str]] = "kmeans|gmm|hmm",
    k_range: Iterable[int] = (2, 3, 4),
    min_regime_len: int = 5
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Detect market regimes using KMeans, GMM, and/or HMM.
    Chooses the best candidate by normalized score (silhouette for kmeans/gmm, BIC for hmm).

    Parameters
    ----------
    X : DataFrame or ndarray
        Feature matrix. If DataFrame, non-numeric columns are ignored; index is preserved.
    methods : str or iterable
        Any subset of {'kmeans','gmm','hmm'}. If str, use '|' separated string.
    k_range : iterable of int
        Candidate k values for KMeans/GMM. HMM internally tests (2,3,4,5).
    min_regime_len : int
        Minimum run length for post-hoc smoothing (to reduce regime flip noise).

    Returns
    -------
    labels : pd.Series
        Regime labels aligned to input index (if any).
    meta : dict
        {"method": ..., "k": ..., "score_name": ..., "score_value": ...}
    """
    Xn, idx = _ensure_2d(X)

    # Standardize (idempotent if already scaled)
    Z = StandardScaler().fit_transform(Xn)

    # normalize methods parameter
    if isinstance(methods, str):
        use = set(m.strip().lower() for m in methods.split("|") if m.strip())
    else:
        use = set(m.lower() for m in methods)

    candidates: list[Tuple[np.ndarray, RegimeMeta]] = []

    if "kmeans" in use:
        best = (-np.inf, None, None)
        for k in k_range:
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            lab = km.fit_predict(Z)
            score = silhouette_score(Z, lab)
            if score > best[0]:
                best = (score, lab, k)
        if best[1] is not None:
            lab = best[1]
            if min_regime_len and min_regime_len > 1:
                lab = _min_run_smooth(lab, min_regime_len)
            candidates.append((lab, RegimeMeta("kmeans", int(best[2]), "silhouette", float(best[0]))))

    if "gmm" in use:
        best = (-np.inf, None, None)
        for k in k_range:
            gm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=3)
            lab = gm.fit_predict(Z)
            score = silhouette_score(Z, lab)
            if score > best[0]:
                best = (score, lab, k)
        if best[1] is not None:
            lab = best[1]
            if min_regime_len and min_regime_len > 1:
                lab = _min_run_smooth(lab, min_regime_len)
            candidates.append((lab, RegimeMeta("gmm", int(best[2]), "silhouette", float(best[0]))))

    if "hmm" in use and _HMM_OK:
        lab, meta = _fit_best_hmm(Z, k_range=(2, 3, 4, 5))
        if min_regime_len and min_regime_len > 1:
            lab = _min_run_smooth(lab, min_regime_len)
        candidates.append((lab, meta))

    if not candidates:
        raise RuntimeError("No regime method produced a candidate (is hmmlearn installed?).")

    # Choose best candidate: maximize silhouette; minimize BIC (we invert it).
    def _score(m: RegimeMeta) -> float:
        return m.score_value if m.score_name == "silhouette" else -m.score_value

    labels, meta = max(candidates, key=lambda t: _score(t[1]))
    ser = pd.Series(labels, index=(idx if idx is not None else pd.RangeIndex(len(labels))))

    return ser, {
        "method": meta.method,
        "k": int(meta.k),
        "score_name": meta.score_name,
        "score_value": float(meta.score_value),
    }
