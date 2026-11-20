"""13_downstream_nested.py -- empirical residual intervals and pair strata recomputed from the NESTED-TUNED pair
predictions of script 10 (no model is refitted here), so that the uncertainty/boundary figure uses the same
predictions as the main performance figure.
Outputs: tables/T13_conformal_nested.csv, tables/T13_pair_strata_nested.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

ENDPOINTS = ["CropYield", "SOC", "pH", "TN", "N2O", "GWP"]
BL = ["eb_shift", "merf_blup", "gpb_blup", "gpb_blup_mom"]


def conformal(alphas=(0.1, 0.2)):
    rows = []
    for e in ENDPOINTS:
        R = pd.read_csv(RES / f"nested_pairs_{e}.csv")
        R["w"] = 1 / R.groupby(["rep", "t"]).y.transform("size")
        for alpha in alphas:
            acc = {m: [] for m in ["zero", "persist", "merf_blup", "gpb_blup", "TRACE"]}
            for (rep, k), te in R.groupby(["rep", "fold"]):
                cal = R[(R.rep == rep) & (R.fold != k)]
                for m in acc:
                    if m == "zero":
                        c, t_ = cal.drop_duplicates("t"), te.drop_duplicates("t")
                        q = np.quantile(np.abs(c.y - c[m]), 1 - alpha)
                        acc[m].append((float((np.abs(t_.y - t_[m]) <= q).mean()), 2 * q, len(t_)))
                    else:
                        s = np.abs(cal.y - cal[m]).values; o = np.argsort(s); cw = np.cumsum(cal.w.values[o]) / cal.w.sum()
                        q = s[o][min(np.searchsorted(cw, 1 - alpha), len(s) - 1)]
                        acc[m].append((float(np.average(np.abs(te.y - te[m]) <= q, weights=te.w)), 2 * q, te.w.sum()))
            for m, v in acc.items():
                v = np.array(v)
                rows.append(dict(endpoint=e, nominal=1 - alpha, method=m, coverage=np.average(v[:, 0], weights=v[:, 2]),
                                 coverage_min_fold=v[:, 0].min(), coverage_max_fold=v[:, 0].max(), width_lnRR=np.average(v[:, 1], weights=v[:, 2])))
    T = pd.DataFrame(rows)
    T["width_rel_zero"] = T.width_lnRR / T.groupby(["endpoint", "nominal"]).width_lnRR.transform(lambda s: s.iloc[0])
    T.to_csv(TAB / "T13_conformal_nested.csv", index=False)
    P = T[T.nominal == 0.9]
    print(P.pivot(index="method", columns="endpoint", values="coverage")[ENDPOINTS].round(3).to_string())
    print(P.pivot(index="method", columns="endpoint", values="width_rel_zero")[ENDPOINTS].round(3).to_string())


def strata(df):
    rows = []
    for e in ENDPOINTS:
        d = df[df[f"y_{e}"].notna()].reset_index(drop=True)
        R = pd.read_csv(RES / f"nested_pairs_{e}.csv")
        a, t = d.iloc[R.a.values].reset_index(drop=True), d.iloc[R.t.values].reset_index(drop=True)
        same_co = a.Treatment.values == t.Treatment.values
        same_bc = (a.Type_biochar.astype(str).values == t.Type_biochar.astype(str).values) & (a.PyrolysisTemperature_biochar.astype(str).values == t.PyrolysisTemperature_biochar.astype(str).values)
        dld = t.logDose.values - a.logDose.values
        R["pair_type"] = np.where(~same_co, "co-treatment differs", np.where(~same_bc, "biochar differs", np.where(np.abs(dld) > 1e-9, "rate only", "identical descriptors")))
        R["direction"] = np.where(dld > 1e-9, "target rate higher", np.where(dld < -1e-9, "target rate lower", "same rate"))
        R["best_blup"] = R[BL].values[:, int(np.argmin([np.mean((R.y - R[b]) ** 2) for b in BL]))]
        for col in ["pair_type", "direction"]:
            for lev, g in R.groupby(col):
                sp = np.sum((g.y - g.persist) ** 2)
                rows.append(dict(endpoint=e, stratum=col, level=lev, n_pairs=len(g) // R.rep.nunique(), n_studies=g.study.nunique(),
                                 skill_TRACE=1 - np.sum((g.y - g.TRACE) ** 2) / sp, skill_best_blup=1 - np.sum((g.y - g.best_blup) ** 2) / sp))
    T = pd.DataFrame(rows); T.to_csv(TAB / "T13_pair_strata_nested.csv", index=False)
    print(T.pivot_table(index=["stratum", "level"], columns="endpoint", values="skill_TRACE")[ENDPOINTS].round(3).to_string())
    print(T.pivot_table(index=["stratum", "level"], columns="endpoint", values="n_studies")[ENDPOINTS].to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    conformal(); strata(load())
