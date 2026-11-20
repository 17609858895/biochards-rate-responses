"""17_equalised_baselines.py -- mixed-effects comparators given the SAME learner budget as TRACE.

Why: in script 10 TRACE averages four tuned learners, whereas each BLUP comparator rests on one learner. Here the
covariate model f of the BLUP is built from the same four families (LightGBM, XGBoost, CatBoost, extra trees), each
tuned with the same 10 configurations in the same inner study-grouped folds, and then averaged:
    blup1 / blup4          f(x_t) + lam [y_a - f(x_a)]         one learner / mean of four
    mundlak1 / mundlak4    same, with a correlated-random-effects (Mundlak) design: every treatment-level covariate
                           enters as its experiment mean plus the within-experiment deviation, so that f can learn
                           within-experiment slopes separately from between-experiment gradients
Covariates of all treatments of a held-out experiment are known by design (only responses are unknown), so the
experiment means use no response information. lam is the moment estimate on inner out-of-fold residuals.
Same outer folds, repetitions and test pairs as script 10; outputs are merged with its pair predictions.
Outputs: results/equalised_pairs_<endpoint>.csv, tables/T17_*.csv
"""
import sys, importlib, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from trace_core import *
from joblib import Parallel, delayed
M = importlib.import_module("10_nested_tuning")

E6 = M.ENDPOINTS
KINDS = ["lgbm", "xgb", "et", "cat"]


def mundlak_design(d, X):
    cols = [c for c in X.columns if c in TREAT_LEVEL or c.startswith("FeedDetail_")]
    G = X[cols].groupby(d.ExpKey.values).transform("mean")
    return pd.concat([X, G.add_prefix("m_"), (X[cols] - G).add_prefix("w_")], axis=1)


def tune_block(tag, Xd, tr, te, inner, y, Atr, Ttr, wtr, pf, Ate, Tte):
    N = len(y); oofs, tes, sel = {}, {}, {}
    for kind in KINDS:
        best = (np.inf,)
        for ci, c in enumerate(M.configs(kind)):
            oof = np.full(N, np.nan)
            for itr, ite in inner:
                oof[ite] = M.fit(kind, c, Xd.iloc[itr], y[itr]).predict(Xd.iloc[ite])
            ra, rt = y[Atr] - oof[Atr], y[Ttr] - oof[Ttr]; p = np.zeros(len(Atr))
            for j in range(3):
                m = pf == j
                p[m] = oof[Ttr[m]] + M.mom_lambda(ra[~m], rt[~m], wtr[~m]) * ra[m]
            err = M.wmean((y[Ttr] - p) ** 2, wtr)
            if err < best[0]:
                best = (err, ci, c, oof.copy())
        sel[f"{tag}_{kind}"] = best[1]; oofs[kind] = best[3]
        f = np.full(N, np.nan); f[te] = M.fit(kind, best[2], Xd.iloc[tr], y[tr]).predict(Xd.iloc[te]); tes[kind] = f
    out = {}
    for name, ks in [(f"{tag}1", ["lgbm"]), (f"{tag}4", KINDS)]:
        oof, f = np.mean([oofs[k] for k in ks], axis=0), np.mean([tes[k] for k in ks], axis=0)
        lam = M.mom_lambda(y[Atr] - oof[Atr], y[Ttr] - oof[Ttr], wtr)
        out[name] = f[Tte] + lam * (y[Ate] - f[Ate]); out[name + "_zero"] = f[Tte]; sel[name + "_lam"] = lam
    return out, sel


def one(e, rep, k, tr, te, d, X, Xm, y):
    t0 = time.time(); exp, study, N = d.ExpKey.values, d.StudyID.values, len(y)
    inner = [(tr[a], tr[b]) for a, b in study_folds(study[tr], 3, seed=SEED + 101 + rep)]
    fold_of_row = np.full(N, -1)
    for j, (_, ite) in enumerate(inner):
        fold_of_row[ite] = j
    Atr, Ttr = ordered_pairs(tr, exp); wtr = target_weights(Ttr); Ate, Tte = ordered_pairs(te, exp); pf = fold_of_row[Ttr]
    R = dict(t=Tte, a=Ate, rep=rep, fold=k); meta = dict(endpoint=e, rep=rep, fold=k)
    for tag, Xd in [("blup", X), ("mundlak", Xm)]:
        o, s = tune_block(tag, Xd, tr, te, inner, y, Atr, Ttr, wtr, pf, Ate, Tte); R.update(o); meta.update(s)
    meta["secs"] = round(time.time() - t0, 1); print(f"done {e} rep{rep} fold{k} {meta['secs']}s", flush=True)
    return e, pd.DataFrame(R), meta


def skill(q, m):
    return 1 - np.sum(q.w * (q.y - q[m]) ** 2) / np.sum(q.w * (q.y - q.persist) ** 2)


def summarise(df, B=800):
    S, D = [], []
    new = ["blup1", "blup4", "mundlak1", "mundlak4"]
    for e in E6:
        d = df[df[f"y_{e}"].notna()].reset_index(drop=True)
        R = pd.read_csv(RES / f"nested_pairs_{e}.csv").merge(pd.read_csv(RES / f"equalised_pairs_{e}.csv"), on=["rep", "fold", "t", "a"], validate="one_to_one")
        R["w"] = 1 / R.groupby(["rep", "t"]).y.transform("size")
        R["same_co"] = d.Treatment.values[R.a.values] == d.Treatment.values[R.t.values]
        for lab, Q in [("all pairs", R), ("same co-treatment", R[R.same_co])]:
            Q = Q.reset_index(drop=True); st = Q.study.values; u = np.unique(st); ix = {s: np.where(st == s)[0] for s in u}
            rng = np.random.RandomState(0); samples = [np.concatenate([ix[s] for s in rng.choice(u, len(u))]) for _ in range(B)]
            yv, wv = Q.y.values, Q.w.values
            for m in ["eb_shift", "merf_blup"] + new + ["trace_lgbm", "TRACE"]:
                pv = Q[m].values; br = np.array([M.wr2(yv[s_], pv[s_], wv[s_]) for s_ in samples])
                S.append(dict(endpoint=e, subset=lab, method=m, R2=M.wr2(yv, pv, wv), R2_lo=np.percentile(br, 2.5), R2_hi=np.percentile(br, 97.5), contrast_skill=skill(Q, m)))
            for top, base in [("TRACE", "blup4"), ("TRACE", "mundlak4"), ("trace_lgbm", "blup1"), ("trace_lgbm", "mundlak1"), ("mundlak4", "blup4"), ("blup4", "blup1")]:
                v = np.array([skill(Q.iloc[s_], top) - skill(Q.iloc[s_], base) for s_ in samples])
                D.append(dict(endpoint=e, subset=lab, top=top, base=base, d_skill=skill(Q, top) - skill(Q, base), lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5),
                              p_le0=float((v <= 0).mean())))
    S, D = pd.DataFrame(S), pd.DataFrame(D)
    S.to_csv(TAB / "T17_equalised_summary.csv", index=False); D.to_csv(TAB / "T17_equalised_differences.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
    order = ["eb_shift", "merf_blup", "blup1", "blup4", "mundlak1", "mundlak4", "trace_lgbm", "TRACE"]
    for lab in ["all pairs", "same co-treatment"]:
        print(f"-- R2, {lab}\n", S[S.subset == lab].pivot(index="method", columns="endpoint", values="R2").loc[order, E6].round(3).to_string())
        print(f"-- contrast skill, {lab}\n", S[S.subset == lab].pivot(index="method", columns="endpoint", values="contrast_skill").loc[order, E6].round(3).to_string())
    D["txt"] = D.apply(lambda r: f"{r.d_skill:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]", axis=1)
    print(D.assign(c=D.top + " - " + D.base + " | " + D.subset).pivot(index="c", columns="endpoint", values="txt")[E6].to_string())


def main():
    df = load(); jobs = []
    for e in E6:
        d = df[df[f"y_{e}"].notna()].reset_index(drop=True); X, y = design(d), d[f"y_{e}"].values; Xm = mundlak_design(d, X)
        for rep in range(M.REPS):
            for k, (tr, te) in enumerate(study_folds(d.StudyID.values, 5, SEED + rep)):
                jobs.append(delayed(one)(e, rep, k, tr, te, d, X, Xm, y))
    res = Parallel(n_jobs=10, verbose=1)(jobs)
    for e in E6:
        pd.concat([r for ee, r, m in res if ee == e], ignore_index=True).to_csv(RES / f"equalised_pairs_{e}.csv", index=False)
    pd.DataFrame([m for _, _, m in res]).to_csv(TAB / "T17_selected_configs.csv", index=False)
    summarise(df)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarise(load())
    else:
        main()
