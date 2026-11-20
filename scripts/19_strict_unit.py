"""19_strict_unit.py -- Sensitivity to matching the observed control mean within experiment.

Default unit   : (study, experiment label).  In 2.6-13.4 % of experiments the control mean of the endpoint is not unique,
                 so the control does not cancel exactly from within-experiment contrasts.
Strict unit    : (study, experiment label, control mean of the endpoint): pairs are formed only among rows that share
                 the same control value.
A  elasticities with experiment fixed effects versus strict-unit fixed effects
B  anchored prediction (fixed configuration of script 04): models trained and evaluated on strict-unit pairs, compared
   on the SAME pairs with the default-unit models of script 04; default-unit accuracy on the excluded pairs is also given
Outputs: tables/T19_strict_unit_elasticity.csv, tables/T19_strict_unit_prediction.csv
"""
import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from joblib import Parallel, delayed
M4 = importlib.import_module("04_trace_main")

E6 = M4.ENDPOINTS
ENDS = E6 + ["GHGI"]


def strict_key(d, e):
    return (d.ExpKey.astype(str) + "|" + d[f"{e}_CK"].round(8).astype(str)).factorize()[0]


def elasticities(df):
    rows = []
    for e in ENDS:
        d = df[df[f"y_{e}"].notna()].reset_index(drop=True); y, ld, st = d[f"y_{e}"].values, d.logDose.values, d.StudyID.values
        for lab, g in [("experiment", d.ExpKey.values), ("strict (same control)", strict_key(d, e))]:
            dm = lambda v: v - pd.Series(v).groupby(g).transform("mean").values
            yw, dw = dm(y), dm(ld); k = np.abs(dw) > 1e-9
            th = np.sum(dw[k] * yw[k]) / np.sum(dw[k] ** 2); r = yw[k] - th * dw[k]
            se = np.sqrt(np.sum(pd.Series(dw[k] * r).groupby(st[k]).sum().values ** 2)) / np.sum(dw[k] ** 2)
            rows.append(dict(endpoint=e, unit=lab, theta=th, se=se, n_rows=int(k.sum()), n_studies=len(np.unique(st[k])), n_units=len(np.unique(g[k]))))
    T = pd.DataFrame(rows); T.to_csv(TAB / "T19_strict_unit_elasticity.csv", index=False)
    T["txt"] = T.apply(lambda r: f"{r.theta:+.3f}±{r.se:.3f} (n={r.n_rows})", axis=1)
    print(T.pivot(index="unit", columns="endpoint", values="txt")[ENDS].to_string())


def one(e, rep, k, tr, te, d, X, y, vT):
    _, R, _ = M4.one_fold(e, rep, k, tr, te, d, X, y, vT)
    return e, R[["t", "a", "y", "ya", "persist", "eb_shift", "trace_lgbm", "TRACE", "rep", "fold", "study"]]


def prediction(df):
    jobs = []
    for e in E6:
        d = df[df[f"y_{e}"].notna()].reset_index(drop=True); X, y = design(d), d[f"y_{e}"].values
        vT = (d[f"{e}_T_SD"] ** 2 / (d[f"{e}_T_N"] * d[f"{e}_T"] ** 2)).values
        d2 = d.copy(); d2["ExpKey"] = strict_key(d, e)
        for rep in range(M4.REPS):
            for k, (tr, te) in enumerate(study_folds(d.StudyID.values, 5, SEED + rep)):
                jobs.append(delayed(one)(e, rep, k, tr, te, d2, X, y, vT))
    res = Parallel(n_jobs=7, verbose=1)(jobs)
    rows = []
    sk = lambda q, m: 1 - np.sum(q.w * (q.y - q[m]) ** 2) / np.sum(q.w * (q.y - q.persist) ** 2)
    for e in E6:
        S = pd.concat([r for ee, r in res if ee == e], ignore_index=True); S.to_csv(RES / f"strict_pairs_{e}.csv", index=False)
        Dft = pd.read_csv(RES / f"trace_pairs_{e}.csv")
        Z = Dft.merge(S[["rep", "fold", "t", "a", "eb_shift", "trace_lgbm", "TRACE"]], on=["rep", "fold", "t", "a"], how="left", suffixes=("", "_strict"))
        Z["w"] = 1 / Z.groupby(["rep", "t"]).y.transform("size"); common = Z.TRACE_strict.notna()
        for lab, q in [("pairs sharing one control", Z[common]), ("pairs with different controls", Z[~common])]:
            if len(q) < 30:
                continue
            r = dict(endpoint=e, pairs=lab, n_pairs=len(q) // M4.REPS, n_studies=q.study.nunique(), share_of_pairs=float(len(q) / len(Z)),
                     skill_TRACE_default_unit=sk(q, "TRACE"), skill_BLUP_default_unit=sk(q, "eb_shift"), R2_TRACE_default_unit=M4.wr2(q.y.values, q.TRACE.values, q.w.values))
            if lab.startswith("pairs sharing"):
                r.update(skill_TRACE_strict_unit=sk(q, "TRACE_strict"), skill_BLUP_strict_unit=sk(q, "eb_shift_strict"), R2_TRACE_strict_unit=M4.wr2(q.y.values, q.TRACE_strict.values, q.w.values))
            rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(TAB / "T19_strict_unit_prediction.csv", index=False)
    pd.set_option("display.width", 260); print(T.round(3).to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    df = load()
    elasticities(df)
    if "predict" in sys.argv[1:]:
        prediction(df)
