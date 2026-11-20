"""trace_core.py -- Trial-Anchored Contrast Learning (TRACE): pair construction, features, learners.

Notation   i = experiment label, j = treatment row, s = study
   y_ij = ln(T_ij / C_i)                          observed log response ratio
   y_it - y_ia = ln(T_it / T_ia)                  within-experiment contrast for a pair with a common control.
                                                  Under an additive response model, a common experiment intercept also cancels.
   g(x_t, x_a, z_i, y_a)  -> E[y_it - y_ia]       contrast learner, trained ONLY on pairs inside one unit
   prediction with anchor a:   yhat_t|a = y_a + g(.)
   k anchors:                  yhat_t   = sum_a w_a * yhat_t|a
"""
import numpy as np, pandas as pd
import lightgbm as lgb
from common import TREAT_LEVEL, MANAGE


def lgbm(**k):
    p = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, verbose=-1, random_state=0, n_jobs=4)
    p.update(k)
    return lgb.LGBMRegressor(**p)


def ordered_pairs(idx, unit, max_per_unit=None, rng=None):
    """All ordered (anchor, target) pairs of rows `idx` that share the same `unit` label."""
    idx = np.asarray(idx)
    A, T = [], []
    s = pd.Series(unit[idx], index=idx)
    for _, ix in s.groupby(s).groups.items():
        ix = np.asarray(ix)
        if len(ix) < 2:
            continue
        a, t = np.meshgrid(ix, ix)
        m = a != t
        a, t = a[m], t[m]
        if max_per_unit and len(a) > max_per_unit:
            sel = rng.choice(len(a), max_per_unit, replace=False)
            a, t = a[sel], t[sel]
        A.append(a); T.append(t)
    if not A:
        return np.array([], int), np.array([], int)
    return np.concatenate(A), np.concatenate(T)


def random_pairs(idx, n_pairs, rng):
    """Pairs drawn across ANY two training rows (PADRE-style pairwise-difference regression control)."""
    a = rng.choice(idx, n_pairs); t = rng.choice(idx, n_pairs)
    m = a != t
    return a[m], t[m]


def treat_cols(X, tier="exp"):
    cols = [c for c in X.columns if c in TREAT_LEVEL or c.startswith("FeedDetail_")]
    if tier == "study":
        cols += [c for c in X.columns if (c in ("logDuration", "SoilDepth") or c.startswith("CropClass_")) and c not in cols]
    return cols


def pair_features(X, y, A, T, tl, use_ya=True, use_site=True, use_delta=True, va=None, same_exp=None):
    Xt = X.iloc[T].reset_index(drop=True)
    Xa = X.iloc[A][tl].reset_index(drop=True)
    parts = []
    if use_site:
        parts.append(Xt)
    else:
        parts.append(Xt[tl])
    parts.append(Xa.add_prefix("a_"))
    if use_delta:
        parts.append(pd.DataFrame(Xt[tl].values - Xa.values, columns=["d_" + c for c in tl]))
    F = pd.concat(parts, axis=1)
    if use_ya:
        F["y_a"] = y[A]
    if va is not None:
        F["log_v_a"] = np.log(np.clip(va[A], 1e-6, None))
    if same_exp is not None:
        F["same_exp"] = same_exp.astype(float)
    return F


def target_weights(T):
    """each target row gets total weight 1 across its anchors"""
    n = pd.Series(T).map(pd.Series(T).value_counts()).values
    return 1.0 / n
