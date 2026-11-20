"""05_causal_dose.py -- Within-experiment and pooled associations with application rate.

Estimand: elasticity theta = d lnRR / d ln(dose)   (effect of multiplying the rate by e)
Estimators
  pooled_ols   : lnRR ~ ln(dose)                                   between+within mixed, confounded
  pooled_dml   : partially linear DML, LightGBM nuisances on all other covariates, study-grouped cross-fitting
  pooled_pdp   : slope of the partial-dependence curve of a pooled LightGBM (what SHAP/PDP users read)
  between      : experiment means only
  within_fe    : experiment-centred slope among observed rate ladders
  within_fe_dml: legacy table identifier for linear residual adjustment of centred covariates; see script 18 for cross-fitted DML
Also: dose-bin response inside experiments, subgroup (CATE) forest table, R-learner CATE with held-out calibration.
Outputs: tables/T05_dose_elasticity.csv, T05_dose_bins.csv, T05_cate_subgroups.csv, T05_cate_rlearner.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from trace_core import lgbm

ENDPOINTS = ["CropYield", "SOC", "pH", "TN", "N2O", "GWP", "GHGI", "CH4"]


def cl_slope(x, y, cl):
    """OLS slope through the origin of y on x (both centred/residualised) with cluster-robust SE."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    th = np.sum(x * y) / np.sum(x * x)
    e = y - th * x
    s = pd.Series(x * e).groupby(np.asarray(cl)).sum()
    se = np.sqrt(np.sum(s.values ** 2)) / np.sum(x * x)
    return th, se, int(len(np.unique(cl)))


def demean(v, g):
    v = pd.Series(np.asarray(v, float))
    return (v - v.groupby(np.asarray(g)).transform("mean")).values


def crossfit_resid(X, t, groups, seed=SEED):
    r = np.zeros(len(t))
    for tr, te in study_folds(groups, 5, seed):
        r[te] = t[te] - lgbm(n_estimators=300).fit(X.iloc[tr], t[tr]).predict(X.iloc[te])
    return r


def elasticities(d, e):
    y, ld, st, ex = d[f"y_{e}"].values, d.logDose.values, d.StudyID.values, d.ExpKey.values
    out = []
    th, se, n = cl_slope(ld - ld.mean(), y - y.mean(), st); out.append(("pooled_ols", th, se, n, len(d)))
    W = design(d).drop(columns=["logDose"])
    ry, rd = crossfit_resid(W, y, st), crossfit_resid(W, ld, st)
    th, se, n = cl_slope(rd, ry, st); out.append(("pooled_dml", th, se, n, len(d)))
    # PDP slope of a pooled model
    X = design(d); m = lgbm().fit(X, y)
    grid = np.quantile(ld, np.linspace(0.05, 0.95, 15)); pd_ = []
    for gval in grid:
        Xg = X.copy(); Xg["logDose"] = gval; pd_.append(m.predict(Xg).mean())
    out.append(("pooled_pdp", np.polyfit(grid, pd_, 1)[0], np.nan, d.StudyID.nunique(), len(d)))
    # between-experiment
    g = d.groupby("ExpKey").agg(y=(f"y_{e}", "mean"), ld=("logDose", "mean"), st=("StudyID", "first"))
    th, se, n = cl_slope(g.ld - g.ld.mean(), g.y - g.y.mean(), g.st); out.append(("between", th, se, n, len(g)))
    # within-experiment
    yw, dw = demean(y, ex), demean(ld, ex)
    lad = np.abs(dw) > 1e-9
    th, se, n = cl_slope(dw[lad], yw[lad], st[lad]); out.append(("within_fe", th, se, n, int(lad.sum())))
    Wt = d[["co_N", "co_P", "co_K", "co_M", "PyrolysisTemperature_biochar", "pH_biochar", "C_biochar", "N_biochar"]].astype(float)
    Wt = Wt.fillna(Wt.median()).apply(lambda c: demean(c.values, ex))
    Wt = Wt.loc[:, Wt.abs().sum() > 1e-9].values
    exp_ok = pd.Series(lad).groupby(ex).transform("any").values
    def part(v):
        b = np.linalg.lstsq(Wt[exp_ok], v[exp_ok], rcond=None)[0]
        return v[exp_ok] - Wt[exp_ok] @ b
    th, se, n = cl_slope(part(dw), part(yw), st[exp_ok]); out.append(("within_fe_dml", th, se, n, int(exp_ok.sum())))
    return [dict(endpoint=e, estimator=k, theta=a, se=b, lo=a - 1.96 * b, hi=a + 1.96 * b, n_studies=c, n_rows=n_) for k, a, b, c, n_ in out]


BINS = [0, 5, 10, 20, 40, 1e9]
LAB = ["<=5", "5-10", "10-20", "20-40", ">40"]


def dose_bins(d, e):
    """experiment-FE regression on dose-bin dummies (reference 5-10 t/ha); cluster-robust by study."""
    y, ex, st = d[f"y_{e}"].values, d.ExpKey.values, d.StudyID.values
    b = pd.cut(d.BiocharAddition, BINS, labels=LAB).astype(str).values
    D = np.column_stack([demean((b == l).astype(float), ex) for l in LAB if l != "5-10"])
    yw = demean(y, ex)
    XtX = np.linalg.pinv(D.T @ D); beta = XtX @ D.T @ yw; res = yw - D @ beta
    meat = np.zeros((D.shape[1],) * 2)
    for s in np.unique(st):
        m = st == s; v = D[m].T @ res[m]; meat += np.outer(v, v)
    se = np.sqrt(np.diag(XtX @ meat @ XtX))
    labs = [l for l in LAB if l != "5-10"]
    return [dict(endpoint=e, dose_bin=l, effect_vs_5_10=bb, se=s_, n_rows=int((b == l).sum())) for l, bb, s_ in zip(labs, beta, se)]


SUBGROUPS = {
    "Soil pH": lambda d: np.where(d.pH_initial.isna(), None, np.where(d.pH_initial < 6.5, "acidic <6.5", "neutral/alkaline >=6.5")),
    "Initial SOC": lambda d: np.where(d.SOC_initial.isna(), None, np.where(d.SOC_initial < 10, "low <10 g/kg", "high >=10 g/kg")),
    "Texture": lambda d: np.where(d.TextureRank.isna(), None, np.where(d.TextureRank <= 2, "coarse", np.where(d.TextureRank >= 5, "fine", "medium"))),
    "Climate": lambda d: d.ClimateZone.where(d.ClimateZone.isin(["Tropic", "Subtropic", "Temperate"]), None).values,
    "Crop": lambda d: d.CropClass.where(d.CropClass.isin(["maize", "rice", "wheat"]), "other").values,
    "Feedstock": lambda d: d.FeedClass.where(d.FeedClass.isin(["cereal_straw", "wood", "husk_shell"]), None).values,
    "Pyrolysis T": lambda d: np.where(d.PyrolysisTemperature_biochar.isna(), None, np.where(d.PyrolysisTemperature_biochar < 500, "<500 C", ">=500 C")),
    "Duration": lambda d: np.where(d.TrialDuration.isna(), None, np.where(d.TrialDuration <= 365, "<=1 yr", ">1 yr")),
    "N co-applied": lambda d: np.where(d.co_N == 1, "with N", "no extra N"),
}


def subgroup_cate(d, e):
    y, ld, st, ex = d[f"y_{e}"].values, d.logDose.values, d.StudyID.values, d.ExpKey.values
    yw, dw = demean(y, ex), demean(ld, ex)
    rows = []
    for name, fn in SUBGROUPS.items():
        lab = pd.Series(fn(d)).values
        for l in pd.unique(lab[pd.notna(lab)]):
            m = (lab == l) & (np.abs(dw) > 1e-9)
            if m.sum() < 30 or len(np.unique(st[m])) < 8:
                continue
            th, se, n = cl_slope(dw[m], yw[m], st[m])
            rows.append(dict(endpoint=e, moderator=name, level=l, theta=th, se=se, lo=th - 1.96 * se, hi=th + 1.96 * se, n_rows=int(m.sum()), n_studies=n))
    return rows


def rlearner(d, e):
    """R-learner on within-experiment variation: tau(z) = argmin sum (yw - tau(z)*dw)^2, cross-fitted by study.
    Calibration: on held-out studies, estimate the within slope inside terciles of predicted tau."""
    y, ld, st, ex = d[f"y_{e}"].values, d.logDose.values, d.StudyID.values, d.ExpKey.values
    yw, dw = demean(y, ex), demean(ld, ex)
    m = np.abs(dw) > 1e-9
    Z = design(d).drop(columns=["logDose"])[m].reset_index(drop=True)
    yw, dw, st = yw[m], dw[m], st[m]
    tau = np.zeros(m.sum())
    for tr, te in study_folds(st, 5, SEED):
        g = lgbm(n_estimators=150, num_leaves=7, min_child_samples=40, learning_rate=0.03)
        g.fit(Z.iloc[tr], yw[tr] / dw[tr], sample_weight=dw[tr] ** 2)
        tau[te] = g.predict(Z.iloc[te])
    q = pd.qcut(tau, 3, labels=["low", "mid", "high"]).astype(str)
    rows = []
    for l in ["low", "mid", "high"]:
        k = q == l
        th, se, n = cl_slope(dw[k], yw[k], st[k])
        rows.append(dict(endpoint=e, tercile=l, mean_pred_tau=tau[k].mean(), obs_theta=th, se=se, n_rows=int(k.sum()), n_studies=n))
    th_c, se_c, _ = cl_slope((tau - tau.mean()) * dw, yw - np.sum(dw * yw) / np.sum(dw * dw) * dw, st)   # BLP-type heterogeneity slope
    for r in rows:
        r.update(blp_slope=th_c, blp_se=se_c)
    return rows


def main():
    df = load()
    E, B, C, R = [], [], [], []
    for e in ENDPOINTS:
        d = df[df[f"y_{e}"].notna()].reset_index(drop=True)
        E += elasticities(d, e)
        B += dose_bins(d, e)
        if e in ("CropYield", "SOC", "pH", "TN", "N2O"):
            C += subgroup_cate(d, e)
            R += rlearner(d, e)
    E, B, C, R = map(pd.DataFrame, (E, B, C, R))
    E.to_csv(TAB / "T05_dose_elasticity.csv", index=False); B.to_csv(TAB / "T05_dose_bins.csv", index=False)
    C.to_csv(TAB / "T05_cate_subgroups.csv", index=False); R.to_csv(TAB / "T05_cate_rlearner.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    E["txt"] = E.apply(lambda r: f"{r.theta:+.3f}" + ("" if np.isnan(r.se) else f"±{r.se:.3f}"), axis=1)
    print(E.pivot(index="estimator", columns="endpoint", values="txt").loc[["pooled_ols", "pooled_pdp", "pooled_dml", "between", "within_fe", "within_fe_dml"], ENDPOINTS].to_string())
    print(E[E.estimator == "within_fe"][["endpoint", "n_studies", "n_rows"]].to_string())
    B["txt"] = B.apply(lambda r: f"{r.effect_vs_5_10:+.3f}±{r.se:.3f}", axis=1)
    print(B.pivot(index="dose_bin", columns="endpoint", values="txt").loc[["<=5", "10-20", "20-40", ">40"], ENDPOINTS].to_string())
    print(C[C.endpoint == "CropYield"].round(3).to_string())
    print(R.round(3).to_string())


if __name__ == "__main__":
    main()
