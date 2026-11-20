"""18_panel_dml.py -- panel double machine learning for the rate elasticity (cross-check of the within-trial estimate).

Partially linear panel model   y_ij = theta * ln(rate_ij) + l(w_ij) + alpha_i + u_ij,   i = experiment.
Two transformations in the spirit of DML for static panels with fixed effects (Clarke & Polselli 2025); this is an
independent implementation, not a replication of their code:
  wg_dml  : within-group transformation of y, ln(rate) and the treatment-level covariates w (co-applied N/P/K/manure,
            biochar properties); nuisance functions E[y~|w~], E[d~|w~] by LightGBM, cross-fitted over studies
  cre_dml : correlated random effects (Mundlak): untransformed rows, nuisance functions on (x, experiment means of the
            treatment-level covariates including mean ln(rate)); cross-fitted over studies
Reference estimators from script 05: within_fe (no covariates) and pooled_dml (no experiment structure).
Standard errors are cluster-robust by study.  Output: tables/T18_panel_dml.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from trace_core import lgbm

ENDS = ["CropYield", "SOC", "pH", "TN", "N2O", "GWP", "GHGI"]
WCOLS = ["co_N", "co_P", "co_K", "co_M", "PyrolysisTemperature_biochar", "pH_biochar", "C_biochar", "N_biochar", "CNRatio_biochar", "Ash_biochar"]


def slope(rd, ry, cl):
    th = np.sum(rd * ry) / np.sum(rd * rd); e = ry - th * rd
    se = np.sqrt(np.sum(pd.Series(rd * e).groupby(np.asarray(cl)).sum().values ** 2)) / np.sum(rd * rd)
    return th, se


def small():
    return lgbm(n_estimators=200, num_leaves=7, min_child_samples=20, learning_rate=0.03)


def crossfit(X, t, st):
    r = np.zeros(len(t))
    for tr, te in study_folds(st, 5, SEED):
        r[te] = t[te] - small().fit(X.iloc[tr], t[tr]).predict(X.iloc[te])
    return r


def main():
    df = load(); rows = []
    old = pd.read_csv(TAB / "T05_dose_elasticity.csv").set_index(["endpoint", "estimator"])
    for e in ENDS:
        d = df[df[f"y_{e}"].notna()].reset_index(drop=True); y, ld, st, ex = d[f"y_{e}"].values, d.logDose.values, d.StudyID.values, d.ExpKey.values
        dm = lambda v: v - pd.Series(v).groupby(ex).transform("mean").values
        W = d[WCOLS].astype(float); W = W.fillna(W.groupby(ex).transform("mean")).fillna(W.median())
        # ---- within-group DML on experiments that vary the rate
        yw, dw = dm(y), dm(ld); Ww = W.apply(lambda c: dm(c.values)); lad = pd.Series(np.abs(dw) > 1e-9).groupby(ex).transform("any").values
        Wl = Ww[lad].reset_index(drop=True); Wl = Wl.loc[:, Wl.abs().sum() > 1e-9]
        ry, rd = crossfit(Wl, yw[lad], st[lad]), crossfit(Wl, dw[lad], st[lad])
        th, se = slope(rd, ry, st[lad]); rows.append(dict(endpoint=e, estimator="wg_dml", theta=th, se=se, n_rows=int(lad.sum()), n_studies=len(np.unique(st[lad]))))
        # ---- correlated random effects (Mundlak) DML on all rows
        X = design(d).drop(columns=["logDose"]); G = W.groupby(ex).transform("mean").add_prefix("m_"); G["m_logDose"] = pd.Series(ld).groupby(ex).transform("mean").values
        Xc = pd.concat([X, G], axis=1)
        ry, rd = crossfit(Xc, y, st), crossfit(Xc, ld, st)
        th, se = slope(rd, ry, st); rows.append(dict(endpoint=e, estimator="cre_dml", theta=th, se=se, n_rows=len(d), n_studies=len(np.unique(st))))
        for k in ["within_fe", "pooled_dml", "pooled_ols"]:
            r = old.loc[(e, k)]; rows.append(dict(endpoint=e, estimator=k, theta=r.theta, se=r.se, n_rows=int(r.n_rows), n_studies=int(r.n_studies)))
    T = pd.DataFrame(rows); T["lo"], T["hi"] = T.theta - 1.96 * T.se, T.theta + 1.96 * T.se
    T.to_csv(TAB / "T18_panel_dml.csv", index=False)
    T["txt"] = T.apply(lambda r: f"{r.theta:+.3f}±{r.se:.3f}", axis=1)
    pd.set_option("display.width", 220)
    print(T.pivot(index="estimator", columns="endpoint", values="txt").loc[["pooled_ols", "pooled_dml", "cre_dml", "wg_dml", "within_fe"], ENDS].to_string())


if __name__ == "__main__":
    main()
