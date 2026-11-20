"""16_referee_checks.py -- Treatment selection, comparative skill and sensitivity checks.

A control   : is the control mean unique inside an experiment (does it cancel exactly)?
B skill     : within-trial contrast skill (1 - SSE/SSE_persistence) on ALL pairs and on pairs with IDENTICAL co-treatment
              and fits a six-predictor linear contrast baseline with an intercept  y_t - y_a ~ y_a + d ln rate + d co-N/P/K/M
              and reports the single-learner TRACE next to the 4-learner mean; paired study bootstrap vs MERF and vs linear
C ranking   : best-treatment hit rate against the rule 'apply the highest rate', all options and same-co-treatment options
D curvature : quadratic term of ln rate inside experiments (formal test of saturation / acceleration)
E hausman   : within-trial minus pooled elasticity with a study-cluster bootstrap
G holm      : Holm adjustment of the 24 TRACE-vs-mixed-effects comparisons of script 10
F prospect  : retrospective publication-year split, train on studies published <= 2018, test on >= 2019 (fixed configuration of script 04)
Outputs: tables/T16_*.csv
"""
import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

E6 = ["CropYield", "SOC", "pH", "TN", "N2O", "GWP"]
EC = ["CropYield", "SOC", "pH", "TN", "N2O", "GWP", "GHGI"]
GOOD = {"CropYield": 1, "SOC": 1, "pH": 1, "TN": 1, "N2O": -1, "GWP": -1, "GHGI": -1}
BL = ["eb_shift", "merf_blup", "gpb_blup", "gpb_blup_mom"]


def sub(df, e):
    return df[df[f"y_{e}"].notna()].reset_index(drop=True)


def boot(st, fn, B=1000, seed=0):
    rng = np.random.RandomState(seed); u = np.unique(st); ix = {s: np.where(st == s)[0] for s in u}
    v = np.array([fn(np.concatenate([ix[s] for s in rng.choice(u, len(u))])) for _ in range(B)], float)
    return np.percentile(v, 2.5, axis=0), np.percentile(v, 97.5, axis=0), (v <= 0).mean(axis=0)


def A_control(df):
    rows = []
    for e in E6:
        d = sub(df, e); g = d.groupby("ExpKey")[f"{e}_CK"].agg(["nunique", "size"]); g = g[g["size"] >= 2]
        R = pd.read_csv(RES / f"nested_pairs_{e}.csv"); R = R[R.rep == 0]
        ck = d[f"{e}_CK"].values
        rows.append(dict(endpoint=e, n_experiments_2plus=len(g), share_exp_control_not_unique=float((g["nunique"] > 1).mean()),
                         share_pairs_different_control=float(np.mean(~np.isclose(ck[R.a.values], ck[R.t.values])))))
    T = pd.DataFrame(rows); T.to_csv(TAB / "T16_control_uniqueness.csv", index=False); print(T.round(3).to_string())


def enrich(df, e):
    d = sub(df, e); R = pd.read_csv(RES / f"nested_pairs_{e}.csv")
    a, t = d.iloc[R.a.values].reset_index(drop=True), d.iloc[R.t.values].reset_index(drop=True)
    R["same_co"] = (a.Treatment.values == t.Treatment.values)
    R["dld"] = t.logDose.values - a.logDose.values
    for c in ["co_N", "co_P", "co_K", "co_M"]:
        R["d_" + c] = t[c].values - a[c].values
    R["rate_t"] = t.BiocharAddition.values
    R["w"] = 1 / R.groupby(["rep", "t"]).y.transform("size")
    # linear contrast baseline, fitted on the pairs of the other outer folds of the same repetition
    R["linear"] = np.nan
    cols = ["ya", "dld", "d_co_N", "d_co_P", "d_co_K", "d_co_M"]
    for (rep, k), te in R.groupby(["rep", "fold"]):
        tr = R[(R.rep == rep) & (R.fold != k)]
        Xtr = np.column_stack([np.ones(len(tr)), tr[cols].values]) * np.sqrt(tr.w.values)[:, None]
        beta = np.linalg.lstsq(Xtr, (tr.y - tr.ya).values * np.sqrt(tr.w.values), rcond=None)[0]
        R.loc[te.index, "linear"] = te.ya.values + np.column_stack([np.ones(len(te)), te[cols].values]) @ beta
    return R


def skill(R, m, ix=None):
    q = R if ix is None else R.iloc[ix]
    return 1 - np.sum(q.w * (q.y - q[m]) ** 2) / np.sum(q.w * (q.y - q.persist) ** 2)


def BC(df):
    S, D, K = [], [], []
    meth = ["linear", "eb_shift", "merf_blup", "gpb_blup", "gpb_blup_mom", "trace_lgbm", "TRACE"]
    for e in E6:
        R = enrich(df, e)
        for lab, Q in [("all pairs", R), ("same co-treatment", R[R.same_co]), ("co-treatment differs", R[~R.same_co])]:
            Q = Q.reset_index(drop=True); st = Q.study.values
            for m in meth:
                S.append(dict(endpoint=e, subset=lab, method=m, n_pairs=len(Q) // 3, n_studies=Q.study.nunique(), contrast_skill=skill(Q, m)))
            for top, base in [("TRACE", "merf_blup"), ("trace_lgbm", "merf_blup"), ("TRACE", "linear"), ("trace_lgbm", "linear"), ("merf_blup", "linear")]:
                lo, hi, p = boot(st, lambda ix: skill(Q, top, ix) - skill(Q, base, ix), B=600)
                D.append(dict(endpoint=e, subset=lab, top=top, base=base, d_skill=skill(Q, top) - skill(Q, base), lo=lo, hi=hi, p_le0=p))
        # ---- ranking / best remaining treatment
        s = GOOD[e]
        for lab, Q in [("all options", R), ("same co-treatment options", R[R.same_co])]:
            recs = []
            for (rep, a), g in Q.groupby(["rep", "a"]):
                if len(g) < 2 or g.y.nunique() < 2:
                    continue
                yb = (s * g.y).values; best = yb.max(); r = dict(study=g.study.iloc[0], chance=np.mean(yb == best))
                top_rate = g.rate_t.values == g.rate_t.values.max(); r["rule_highest_rate"] = float(np.mean(yb[top_rate] == best))
                for m in ["TRACE", "trace_lgbm", "merf_blup", "linear", "zero"]:
                    r[m] = float(yb[int(np.argmax(s * g[m].values))] == best)
                recs.append(r)
            G = pd.DataFrame(recs)
            if len(G) < 30:
                continue
            st = G.study.values
            for m in ["chance", "rule_highest_rate", "zero", "linear", "merf_blup", "trace_lgbm", "TRACE"]:
                lo, hi, _ = boot(st, lambda ix: G[m].values[ix].mean(), B=600)
                K.append(dict(endpoint=e, options=lab, method=m, n_decisions=len(G) // 3, n_studies=G.study.nunique(), hit_rate=G[m].mean(), lo=lo, hi=hi))
            lo, hi, p = boot(st, lambda ix: G.TRACE.values[ix].mean() - G.rule_highest_rate.values[ix].mean(), B=600)
            K.append(dict(endpoint=e, options=lab, method="TRACE_minus_rule", hit_rate=G.TRACE.mean() - G.rule_highest_rate.mean(), lo=lo, hi=hi))
    S, D, K = pd.DataFrame(S), pd.DataFrame(D), pd.DataFrame(K)
    S.to_csv(TAB / "T16_skill_subsets.csv", index=False); D.to_csv(TAB / "T16_skill_differences.csv", index=False); K.to_csv(TAB / "T16_best_treatment.csv", index=False)
    for lab in ["all pairs", "same co-treatment"]:
        print(f"-- contrast skill, {lab}\n", S[S.subset == lab].pivot(index="method", columns="endpoint", values="contrast_skill").loc[meth, E6].round(3).to_string())
    D["txt"] = D.apply(lambda r: f"{r.d_skill:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]", axis=1)
    print(D[D.subset != "co-treatment differs"].assign(c=D.top + " - " + D.base + " | " + D.subset).pivot(index="c", columns="endpoint", values="txt")[E6].to_string())
    for lab in ["all options", "same co-treatment options"]:
        print(f"-- best-treatment hit rate, {lab}\n", K[K.options == lab].pivot(index="method", columns="endpoint", values="hit_rate").reindex(columns=E6).round(3).to_string())
    print(K[K.method == "TRACE_minus_rule"][["endpoint", "options", "hit_rate", "lo", "hi"]].round(3).to_string())


def DE(df):
    rows, H = [], []
    for e in EC:
        d = sub(df, e); y, ld, st, ex = d[f"y_{e}"].values, d.logDose.values, d.StudyID.values, d.ExpKey.values
        dm = lambda v: v - pd.Series(v).groupby(ex).transform("mean").values
        yw, x1, x2 = dm(y), dm(ld), dm(ld ** 2); lad = np.abs(x1) > 1e-9
        Xm = np.column_stack([x1, x2])[lad]; XtX = np.linalg.pinv(Xm.T @ Xm); b = XtX @ Xm.T @ yw[lad]; r = yw[lad] - Xm @ b
        meat = np.zeros((2, 2))
        for s_ in np.unique(st[lad]):
            v = Xm[st[lad] == s_].T @ r[st[lad] == s_]; meat += np.outer(v, v)
        se = np.sqrt(np.diag(XtX @ meat @ XtX))
        rows.append(dict(endpoint=e, linear=b[0], quad=b[1], quad_se=se[1], quad_z=b[1] / se[1], n_rows=int(lad.sum()), n_studies=len(np.unique(st[lad]))))
        # within minus pooled (all rows) and minus pooled on the ladder sample
        def diff(ix):
            yy, ll, ee = y[ix], ld[ix], pd.factorize(pd.Series(ex[ix]).astype(str) + "_" + pd.Series(np.cumsum(np.r_[1, np.diff(ix) != 1])).astype(str))[0]
            a_, b_ = yy - pd.Series(yy).groupby(ee).transform("mean").values, ll - pd.Series(ll).groupby(ee).transform("mean").values
            w_ = np.sum(a_ * b_) / np.sum(b_ * b_); p_ = np.sum((yy - yy.mean()) * (ll - ll.mean())) / np.sum((ll - ll.mean()) ** 2)
            return w_ - p_
        full = diff(np.arange(len(y))); lo, hi, _ = boot(st, diff, B=800)
        H.append(dict(endpoint=e, within_minus_pooled=full, lo=lo, hi=hi, excludes_zero=bool(lo > 0 or hi < 0)))
    T = pd.DataFrame(rows); T.to_csv(TAB / "T16_curvature.csv", index=False); print(T.round(4).to_string())
    H = pd.DataFrame(H); H.to_csv(TAB / "T16_within_vs_pooled.csv", index=False); print(H.round(4).to_string())


def G_holm():
    D = pd.read_csv(TAB / "T10_nested_paired_bootstrap.csv"); D = D[(D.top == "TRACE") & D.base.isin(BL)].copy()
    D["p"] = np.clip(2 * np.minimum(D.p_le0, 1 - D.p_le0), 1 / 1000, 1)
    o = D.p.sort_values().index; m = len(D); adj = {}; run = 0
    for i, ix in enumerate(o):
        run = max(run, (m - i) * D.p[ix]); adj[ix] = min(run, 1.0)
    D["p_holm"] = pd.Series(adj)
    D.to_csv(TAB / "T16_holm.csv", index=False)
    print(D.pivot(index="base", columns="endpoint", values="p_holm")[E6].round(3).to_string())


def F_prospective(df):
    M4 = importlib.import_module("04_trace_main"); rows = []
    for e in E6:
        d = sub(df, e); X, y = design(d), d[f"y_{e}"].values
        vT = (d[f"{e}_T_SD"] ** 2 / (d[f"{e}_T_N"] * d[f"{e}_T"] ** 2)).values
        yr = d.groupby("StudyID").Year_Publication.transform("max").values
        tr, te = np.where(yr <= 2018)[0], np.where(yr >= 2019)[0]
        _, R, _ = M4.one_fold(e, 0, 0, tr, te, d, X, y, vT)
        w = (1 / R.groupby("t").y.transform("size")).values; sp = np.sum(w * (R.y - R.persist) ** 2)
        for m in ["zero", "persist", "shift", "eb_shift", "trace_lgbm", "TRACE"]:
            rows.append(dict(endpoint=e, method=m, n_train_studies=d.StudyID[tr].nunique(), n_test_studies=R.study.nunique(), n_targets=R.t.nunique(),
                             R2=M4.wr2(R.y.values, R[m].values, w), contrast_skill=1 - np.sum(w * (R.y - R[m]) ** 2) / sp))
        print("prospective done", e, flush=True)
    T = pd.DataFrame(rows); T.to_csv(TAB / "T16_prospective.csv", index=False)
    print(T.pivot(index="method", columns="endpoint", values="R2")[E6].round(3).to_string())
    print(T.pivot(index="method", columns="endpoint", values="contrast_skill")[E6].round(3).to_string())
    print(T.groupby("endpoint")[["n_train_studies", "n_test_studies", "n_targets"]].first().loc[E6].to_string())


if __name__ == "__main__":
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 200)
    df = load()
    todo = sys.argv[1:] or ["A", "BC", "DE", "G", "F"]
    for k, fn in [("A", lambda: A_control(df)), ("BC", lambda: BC(df)), ("DE", lambda: DE(df)), ("G", G_holm), ("F", lambda: F_prospective(df))]:
        if k in todo:
            print(f"\n===== {k}", flush=True); fn()
