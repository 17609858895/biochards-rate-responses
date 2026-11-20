"""04_trace_main.py -- Trial-Anchored Contrast Learning (TRACE) vs same-information baselines.

Protocol (all on held-out STUDIES, 5-fold study-grouped CV x REPS):
  zero-shot : covariates only                               f(x_t)
  1-shot    : one measured treatment (anchor a) of the same experiment is revealed, the others are predicted
  k-shot    : k anchors; 'all' = every other row of the experiment
Same-information comparators
  persist   : yhat_t = y_a
  shift     : f(x_t) + (y_a - f(x_a))                       random-intercept offset, lambda = 1
  eb_shift  : f(x_t) + lambda*(y_a - f(x_a))                BLUP-type shrinkage, lambda estimated on training pairs (MERF-like)
  padre     : contrast learner trained on pairs of ANY two training rows (pairwise-difference regression)
TRACE       : y_a + g(x_t, x_a, delta, z, y_a), g trained only on within-experiment ordered pairs;
              g = mean of LightGBM, XGBoost, CatBoost, ExtraTrees fitted in pair space
Ablations   : -y_a, -site moderators, -delta features, single learner
Outputs     : results/trace_pairs_<endpoint>.csv, tables/T04_trace_summary.csv, tables/T04_trace_kshot.csv,
              tables/T04_trace_studytier.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from trace_core import *
from joblib import Parallel, delayed
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import xgboost as xgb

ENDPOINTS = ["CropYield", "SOC", "pH", "TN", "N2O", "GWP"]
REPS = 3


def contrast_models():
    return {
        "lgbm": lambda: lgbm(n_estimators=300),
        "xgb": lambda: xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=10,
                                        subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, n_jobs=4, random_state=0),
        "cat": lambda: CatBoostRegressor(iterations=400, learning_rate=0.04, depth=5, l2_leaf_reg=5.0, verbose=0,
                                         thread_count=4, random_seed=0),
        "et": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                    ExtraTreesRegressor(300, min_samples_leaf=5, max_features=0.4, n_jobs=4, random_state=0)),
    }


def fit_g(name, F, dy, w):
    m = contrast_models()[name]()
    if name == "et":
        m.fit(F, dy, extratreesregressor__sample_weight=w)
    else:
        m.fit(F, dy, sample_weight=w)
    return m


def one_fold(e, rep, k, tr, te, d, X, y, vT):
    exp, study = d.ExpKey.values, d.StudyID.values
    tl = treat_cols(X)
    rng = np.random.RandomState(SEED + 13 * rep + k)
    # ---- prior (zero-shot) model and its out-of-fold predictions on the training studies
    pf = np.full(len(y), np.nan)
    pf[te] = lgbm().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    for itr, ite in study_folds(study[tr], 5, seed=SEED + 5):
        pf[tr[ite]] = lgbm().fit(X.iloc[tr[itr]], y[tr[itr]]).predict(X.iloc[tr[ite]])
    # ---- experiment-tier pairs
    Atr, Ttr = ordered_pairs(tr, exp)
    Ate, Tte = ordered_pairs(te, exp)
    wtr = target_weights(Ttr)
    dy = y[Ttr] - y[Atr]
    ra, rt = y[Atr] - pf[Atr], y[Ttr] - pf[Ttr]
    lam = float(np.sum(wtr * ra * rt) / np.sum(wtr * ra * ra))
    R = dict(t=Tte, a=Ate, y=y[Tte], ya=y[Ate], zero=pf[Tte], persist=y[Ate],
             shift=pf[Tte] + (y[Ate] - pf[Ate]), eb_shift=pf[Tte] + lam * (y[Ate] - pf[Ate]))
    Ftr, Fte = pair_features(X, y, Atr, Ttr, tl), pair_features(X, y, Ate, Tte, tl)
    G = {n: fit_g(n, Ftr, dy, wtr) for n in contrast_models()}
    for n, m in G.items():
        R[f"trace_{n}"] = y[Ate] + m.predict(Fte)
    R["TRACE"] = np.mean([R[f"trace_{n}"] for n in G], axis=0)
    # residual scale of the contrast learner (training fit; used only for anchor weights)
    s2 = float(np.average((dy - np.mean([m.predict(Ftr) for m in G.values()], axis=0)) ** 2, weights=wtr))
    R["w_prec"] = 1.0 / (s2 + np.nan_to_num(vT[Ate], nan=np.nanmedian(vT)))
    # ---- ablations (LightGBM only, to keep them comparable with trace_lgbm)
    for tag, kw in [("abl_no_ya", dict(use_ya=False)), ("abl_no_site", dict(use_site=False)), ("abl_no_delta", dict(use_delta=False))]:
        R[tag] = y[Ate] + fit_g("lgbm", pair_features(X, y, Atr, Ttr, tl, **kw), dy, wtr).predict(pair_features(X, y, Ate, Tte, tl, **kw))
    Ar, Tr = random_pairs(tr, 3 * len(Atr), rng)
    R["padre"] = y[Ate] + fit_g("lgbm", pair_features(X, y, Ar, Tr, tl), y[Tr] - y[Ar], None).predict(Fte)
    R = pd.DataFrame(R).assign(rep=rep, fold=k, study=study[Tte], exp=exp[Tte], lam=lam)
    # ---- study-tier: anchor comes from ANOTHER experiment of the same study (independent control)
    As, Ts = ordered_pairs(tr, study, max_per_unit=600, rng=rng)
    Ae, Te_ = ordered_pairs(te, study)
    keep = exp[Ae] != exp[Te_]
    Ae, Te_ = Ae[keep], Te_[keep]
    S = None
    if len(Ae):
        tls = treat_cols(X, "study")
        ws = target_weights(Ts)
        Fs = pair_features(X, y, As, Ts, tls, same_exp=(exp[As] == exp[Ts]))
        Fe = pair_features(X, y, Ae, Te_, tls, same_exp=np.zeros(len(Ae), bool))
        ras, rts = y[As] - pf[As], y[Ts] - pf[Ts]
        cross = exp[As] != exp[Ts]
        lam_s = float(np.sum((ws * ras * rts)[cross]) / np.sum((ws * ras * ras)[cross])) if cross.any() else lam
        ps = np.mean([fit_g(n, Fs, y[Ts] - y[As], ws).predict(Fe) for n in ["lgbm", "xgb", "et"]], axis=0)
        S = pd.DataFrame(dict(t=Te_, a=Ae, y=y[Te_], zero=pf[Te_], persist=y[Ae], shift=pf[Te_] + y[Ae] - pf[Ae],
                              eb_shift=pf[Te_] + lam_s * (y[Ae] - pf[Ae]), TRACE=y[Ae] + ps)).assign(rep=rep, fold=k, study=study[Te_])
    return e, R, S


METHODS = ["zero", "persist", "shift", "eb_shift", "padre", "abl_no_ya", "abl_no_site", "abl_no_delta",
           "trace_lgbm", "trace_xgb", "trace_cat", "trace_et", "TRACE"]


def wr2(y, p, w):
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)


def boot_ci(R, cols, w, B=1000, seed=0):
    """cluster bootstrap over studies; returns {col: (lo, hi)} for weighted R2 and for skill vs persistence"""
    rng = np.random.RandomState(seed)
    st = R.study.values
    u = np.unique(st)
    ix = {s: np.where(st == s)[0] for s in u}
    y = R.y.values
    P = {c: R[c].values for c in cols}
    out = {c: [] for c in cols}
    for _ in range(B):
        sel = np.concatenate([ix[s] for s in rng.choice(u, len(u))])
        for c in cols:
            out[c].append(wr2(y[sel], P[c][sel], w[sel]))
    return {c: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for c, v in out.items()}


def main():
    df = load()
    jobs = []
    for e in ENDPOINTS:
        d = df[df[f"y_{e}"].notna()].reset_index(drop=True)
        X, y = design(d), d[f"y_{e}"].values
        vT = (d[f"{e}_T_SD"] ** 2 / (d[f"{e}_T_N"] * d[f"{e}_T"] ** 2)).values
        for rep in range(REPS):
            for k, (tr, te) in enumerate(study_folds(d.StudyID.values, 5, SEED + rep)):
                jobs.append(delayed(one_fold)(e, rep, k, tr, te, d, X, y, vT))
    res = Parallel(n_jobs=7, verbose=5)(jobs)
    summ, kshot, tier = [], [], []
    for e in ENDPOINTS:
        R = pd.concat([r for ee, r, s in res if ee == e], ignore_index=True)
        R.to_csv(RES / f"trace_pairs_{e}.csv", index=False)
        w = (1 / R.groupby(["rep", "t"]).y.transform("size")).values
        ci = boot_ci(R, METHODS, w)
        sse_p = np.sum(w * (R.y - R.persist) ** 2)
        for m in METHODS:
            summ.append(dict(endpoint=e, setting="1-shot", method=m, n_targets=R.t.nunique(), n_studies=R.study.nunique(),
                             R2=wr2(R.y.values, R[m].values, w), lo=ci[m][0], hi=ci[m][1],
                             RMSE=float(np.sqrt(np.average((R.y - R[m]) ** 2, weights=w))),
                             contrast_skill=1 - np.sum(w * (R.y - R[m]) ** 2) / sse_p,
                             sign_acc_contrast=float(np.average(np.sign(R.y - R.ya) == np.sign(R[m] - R.ya), weights=w)) if m != "persist" else np.nan))
        # ---- k-shot: mean over k randomly chosen anchors (k=1,2,3) and over all anchors
        rng = np.random.RandomState(1)
        for kk in [1, 2, 3, "all"]:
            rows = []
            for (rep, t), gdf in R.groupby(["rep", "t"]):
                if kk != "all" and len(gdf) < kk:
                    continue
                gsel = gdf if kk == "all" else gdf.iloc[rng.choice(len(gdf), kk, replace=False)]
                r = dict(y=gdf.y.iloc[0], rep=rep, n_anchor=len(gsel))
                for m in ["zero", "persist", "shift", "eb_shift", "padre", "TRACE"]:
                    r[m] = gsel[m].mean()
                r["TRACE_pw"] = np.average(gsel.TRACE, weights=gsel.w_prec)
                rows.append(r)
            L = pd.DataFrame(rows)
            for m in ["zero", "persist", "shift", "eb_shift", "padre", "TRACE", "TRACE_pw"]:
                kshot.append(dict(endpoint=e, k=kk, method=m, n_targets=len(L) // REPS, R2=r2_score(L.y, L[m]),
                                  RMSE=float(np.sqrt(np.mean((L.y - L[m]) ** 2)))))
        S = [s for ee, r, s in res if ee == e and s is not None]
        if S:
            S = pd.concat(S, ignore_index=True)
            S.to_csv(RES / f"trace_studytier_{e}.csv", index=False)
            ws = (1 / S.groupby(["rep", "t"]).y.transform("size")).values
            for m in ["zero", "persist", "shift", "eb_shift", "TRACE"]:
                tier.append(dict(endpoint=e, method=m, n_targets=S.t.nunique(), n_studies=S.study.nunique(),
                                 R2=wr2(S.y.values, S[m].values, ws), RMSE=float(np.sqrt(np.average((S.y - S[m]) ** 2, weights=ws)))))
    summ, kshot, tier = pd.DataFrame(summ), pd.DataFrame(kshot), pd.DataFrame(tier)
    summ.to_csv(TAB / "T04_trace_summary.csv", index=False)
    kshot.to_csv(TAB / "T04_trace_kshot.csv", index=False)
    tier.to_csv(TAB / "T04_trace_studytier.csv", index=False)
    pd.set_option("display.width", 250)
    print("== 1-shot R2 (held-out studies)\n", summ.pivot(index="method", columns="endpoint", values="R2").loc[METHODS, ENDPOINTS].round(3).to_string())
    print("== 1-shot contrast skill (1 - SSE/SSE_persist)\n", summ.pivot(index="method", columns="endpoint", values="contrast_skill").loc[METHODS, ENDPOINTS].round(3).to_string())
    print("== TRACE 95% CI\n", summ[summ.method.isin(["eb_shift", "TRACE"])][["endpoint", "method", "n_targets", "n_studies", "R2", "lo", "hi", "RMSE", "sign_acc_contrast"]].round(3).to_string())
    print("== k-shot R2\n", kshot.pivot_table(index=["endpoint", "k"], columns="method", values="R2").round(3).to_string())
    print("== study-tier (anchor from another experiment of the same study)\n", tier.pivot(index="method", columns="endpoint", values="R2").round(3).to_string())
    print(tier.groupby("endpoint")[["n_targets", "n_studies"]].first().to_string())


if __name__ == "__main__":
    main()
