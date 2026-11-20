"""10_nested_tuning.py -- nested (study-grouped) hyper-parameter tuning with EQUAL budgets + fitted mixed-effects baselines.

Outer loop : 5-fold study-grouped CV x REPS (same folds as 04).
Inner loop : 3-fold study-grouped CV inside every outer training set; N_CFG random configurations per tuned learner
             (configuration 0 is always the fixed setting used in 04, so tuning can only match or improve it in-sample).
Selection  : each method is selected on ITS OWN inner-fold objective
             zero-shot f            -> row MSE
             BLUP baselines         -> 1-shot weighted pair MSE of the BLUP prediction
             TRACE members / PADRE  -> weighted MSE of the within-experiment contrast
Methods (1 anchor of the same experiment revealed)
  persist        y_a
  eb_shift       f(x_t) + lam (y_a - f(x_a)); f = tuned LightGBM, lam by moments on inner out-of-fold residuals
  merf_blup      MERF-type EM with out-of-fold residuals: f refitted on y - b_hat, variance components by moments
  gpb_blup       GPBoost (tree boosting + experiment random intercept); lam from the fitted variance components
  gpb_blup_mom   GPBoost fixed effect, lam re-estimated by moments on inner out-of-fold residuals
  padre          tuned LightGBM on pairs of any two training rows
  trace_<m>      tuned contrast learner m in {lgbm, xgb, et, cat};  TRACE = mean of the four (pre-specified)
  TRACE_stack    NNLS weights of the four members learned on inner out-of-fold contrasts
  HYBRID_stack   NNLS over the four members + the contrasts implied by eb_shift and gpb_blup_mom
Outputs: results/nested_pairs_<endpoint>.csv, tables/T10_*.csv
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from trace_core import *
from joblib import Parallel, delayed
from scipy.optimize import nnls
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import gpboost as gpb

ENDPOINTS = ["CropYield", "SOC", "pH", "TN", "N2O", "GWP"]
REPS, N_CFG, NT = 3, 10, 3

SPACE = {
    "lgbm": (dict(n_estimators=[150, 300, 600], learning_rate=[0.02, 0.03, 0.05], num_leaves=[7, 15, 31], min_child_samples=[10, 20, 40],
                  colsample_bytree=[0.4, 0.6, 0.8], reg_lambda=[1.0, 5.0, 20.0]),
             dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, colsample_bytree=0.6, reg_lambda=5.0)),
    "xgb": (dict(n_estimators=[150, 300, 600], learning_rate=[0.02, 0.03, 0.05], max_depth=[3, 4, 6], min_child_weight=[5, 10, 20],
                 colsample_bytree=[0.4, 0.6, 0.8], reg_lambda=[1.0, 5.0, 20.0]),
            dict(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=10, colsample_bytree=0.6, reg_lambda=5.0)),
    "et": (dict(min_samples_leaf=[2, 5, 10, 20], max_features=[0.2, 0.4, 0.6, 0.8], max_depth=[None, 8, 14]),
           dict(min_samples_leaf=5, max_features=0.4, max_depth=None)),
    "cat": (dict(iterations=[200, 400], learning_rate=[0.04, 0.08], depth=[4, 5, 6], l2_leaf_reg=[3.0, 5.0, 10.0]),
            dict(iterations=400, learning_rate=0.04, depth=5, l2_leaf_reg=5.0)),
    "gpb": (dict(num_boost_round=[10, 25, 50, 100, 200], learning_rate=[0.01, 0.02, 0.05], num_leaves=[4, 7, 15], min_data_in_leaf=[20, 40, 80],
                 lambda_l2=[1.0, 5.0, 20.0], feature_fraction=[0.4, 0.6, 0.8]),
            dict(num_boost_round=25, learning_rate=0.05, num_leaves=7, min_data_in_leaf=40, lambda_l2=5.0, feature_fraction=0.6)),
}


def configs(kind, n=N_CFG, seed=7):
    space, default = SPACE[kind]
    rng = np.random.RandomState(seed + sum(map(ord, kind)))
    out = [default]
    while len(out) < n:
        c = {k: v[rng.randint(len(v))] for k, v in space.items()}
        if c not in out:
            out.append(c)
    return out


def make(kind, c):
    if kind == "lgbm":
        return lgbm(n_jobs=NT, **c)
    if kind == "xgb":
        return xgb.XGBRegressor(subsample=0.8, n_jobs=NT, random_state=0, **c)
    if kind == "et":
        return make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(300, n_jobs=NT, random_state=0, **c))
    if kind == "cat":
        return CatBoostRegressor(verbose=0, thread_count=NT, random_seed=0, **c)
    raise ValueError(kind)


def fit(kind, c, F, t, w=None):
    m = make(kind, c)
    if kind == "et":
        m.fit(F, t, extratreesregressor__sample_weight=w)
    else:
        m.fit(F, t, sample_weight=w)
    return m


def gpb_fit_predict(c, Xtr, ytr, gtr, Xte, gte):
    # the response is standardised on the training rows: GPBoost's boosting step scales with 1/error variance,
    # so an lnRR-scale response (variance ~0.05) makes the fixed effect diverge; lambda is scale-free.
    c = dict(c); rounds = c.pop("num_boost_round")
    mu, sd = float(np.mean(ytr)), float(np.std(ytr))
    gm = gpb.GPModel(group_data=gtr, likelihood="gaussian", num_parallel_threads=NT)
    bst = gpb.train(params=dict(verbose=-1, num_threads=NT, **c), train_set=gpb.Dataset(Xtr, (ytr - mu) / sd), gp_model=gm, num_boost_round=rounds)
    cp = gm.get_cov_pars().values.ravel()
    F = mu + sd * np.asarray(bst.predict(data=Xte, group_data_pred=gte, pred_latent=True)["fixed_effect"])
    return F, float(cp[1] / (cp[1] + cp[0] + 1e-12)), cp


def wmean(v, w):
    return float(np.sum(w * v) / np.sum(w))


def mom_lambda(ra, rt, w):
    return float(np.clip(np.sum(w * ra * rt) / np.sum(w * ra * ra), 0.0, 0.999))


def tune_fold(e, rep, k, tr, te, d, X, y):
    t0 = time.time()
    exp, study, N = d.ExpKey.values, d.StudyID.values, len(y)
    tl = treat_cols(X)
    rng = np.random.RandomState(SEED + 31 * rep + k)
    inner = [(tr[a], tr[b]) for a, b in study_folds(study[tr], 3, seed=SEED + 101 + rep)]
    fold_of_row = np.full(N, -1)
    for j, (_, ite) in enumerate(inner):
        fold_of_row[ite] = j
    Atr, Ttr = ordered_pairs(tr, exp); wtr = target_weights(Ttr)
    Ate, Tte = ordered_pairs(te, exp)
    pf = fold_of_row[Ttr]                                        # inner fold of every training pair
    dy = y[Ttr] - y[Atr]
    sel, R = {}, dict(t=Tte, a=Ate, y=y[Tte], ya=y[Ate], persist=y[Ate])

    # ---------- zero-shot f and moment BLUP (tuned LightGBM on rows)
    best = dict(zero=(np.inf,), eb=(np.inf,))
    for ci, c in enumerate(configs("lgbm")):
        oof = np.full(N, np.nan)
        for itr, ite in inner:
            oof[ite] = fit("lgbm", c, X.iloc[itr], y[itr]).predict(X.iloc[ite])
        ra, rt = y[Atr] - oof[Atr], y[Ttr] - oof[Ttr]
        ebp = np.zeros(len(Atr))
        for j in range(3):
            m = pf == j
            ebp[m] = oof[Ttr[m]] + mom_lambda(ra[~m], rt[~m], wtr[~m]) * ra[m]
        mz, me = float(np.mean((y[tr] - oof[tr]) ** 2)), wmean((y[Ttr] - ebp) ** 2, wtr)
        if mz < best["zero"][0]:
            best["zero"] = (mz, ci, c)
        if me < best["eb"][0]:
            best["eb"] = (me, ci, c, oof.copy(), ebp.copy())
    sel["f_zero"], sel["f_blup"] = best["zero"][1], best["eb"][1]
    R["zero"] = fit("lgbm", best["zero"][2], X.iloc[tr], y[tr]).predict(X.iloc[te])[np.searchsorted(te, Tte)]
    c_b, oof_b, eb_oof = best["eb"][2], best["eb"][3], best["eb"][4]
    fb = np.full(N, np.nan); fb[te] = fit("lgbm", c_b, X.iloc[tr], y[tr]).predict(X.iloc[te])
    lam = mom_lambda(y[Atr] - oof_b[Atr], y[Ttr] - oof_b[Ttr], wtr)
    R.update(eb_Ft=fb[Tte], eb_Fa=fb[Ate], eb_lam=lam, shift=fb[Tte] + y[Ate] - fb[Ate], eb_shift=fb[Tte] + lam * (y[Ate] - fb[Ate]))

    # ---------- MERF-type EM with out-of-fold residuals (f config = the one selected for the BLUP)
    b = np.zeros(N)
    for _ in range(4):
        oof = np.full(N, np.nan)
        for itr, ite in inner:
            oof[ite] = fit("lgbm", c_b, X.iloc[itr], (y - b)[itr]).predict(X.iloc[ite])
        r = y - oof
        sb = max(wmean(r[Atr] * r[Ttr], wtr), 1e-8)
        st_ = float(np.mean(r[np.unique(Ttr)] ** 2)); se_ = max(st_ - sb, 1e-8)
        g = pd.DataFrame(dict(r=r[tr], e=exp[tr])).groupby("e").r.agg(["mean", "size"])
        bh = (g["size"] * sb / (g["size"] * sb + se_) * g["mean"])
        b[tr] = bh.reindex(exp[tr]).values
    fm = np.full(N, np.nan); fm[te] = fit("lgbm", c_b, X.iloc[tr], (y - b)[tr]).predict(X.iloc[te])
    lam_m = sb / (sb + se_)
    R.update(merf_Ft=fm[Tte], merf_Fa=fm[Ate], merf_lam=lam_m, merf_blup=fm[Tte] + lam_m * (y[Ate] - fm[Ate]))

    # ---------- GPBoost (experiment random intercept)
    best = dict(pure=(np.inf,), mom=(np.inf,))
    for ci, c in enumerate(configs("gpb")):
        oof, lj = np.full(N, np.nan), np.zeros(3)
        try:
            for j, (itr, ite) in enumerate(inner):
                oof[ite], lj[j], _ = gpb_fit_predict(c, X.iloc[itr], y[itr], exp[itr], X.iloc[ite], exp[ite])
        except Exception as ex:
            print("GPB fail", e, ci, repr(ex)[:80], flush=True); continue
        ra, rt = y[Atr] - oof[Atr], y[Ttr] - oof[Ttr]
        pp, pm = np.zeros(len(Atr)), np.zeros(len(Atr))
        for j in range(3):
            m = pf == j
            pp[m] = oof[Ttr[m]] + lj[j] * ra[m]
            pm[m] = oof[Ttr[m]] + mom_lambda(ra[~m], rt[~m], wtr[~m]) * ra[m]
        e1, e2 = wmean((y[Ttr] - pp) ** 2, wtr), wmean((y[Ttr] - pm) ** 2, wtr)
        if e1 < best["pure"][0]:
            best["pure"] = (e1, ci, c)
        if e2 < best["mom"][0]:
            best["mom"] = (e2, ci, c, oof.copy(), pm.copy())
    sel["gpb_pure"], sel["gpb_mom"] = best["pure"][1], best["mom"][1]
    Fp, lam_p, cp = gpb_fit_predict(best["pure"][2], X.iloc[tr], y[tr], exp[tr], X.iloc[te], exp[te])
    fp = np.full(N, np.nan); fp[te] = Fp
    R.update(gpb_zero=fp[Tte], gpb_Ft=fp[Tte], gpb_Fa=fp[Ate], gpb_lam=lam_p, gpb_blup=fp[Tte] + lam_p * (y[Ate] - fp[Ate]))
    c_g, oof_g, gpbm_oof = best["mom"][2], best["mom"][3], best["mom"][4]
    Fq, _, _ = gpb_fit_predict(c_g, X.iloc[tr], y[tr], exp[tr], X.iloc[te], exp[te])
    fq = np.full(N, np.nan); fq[te] = Fq
    lam_q = mom_lambda(y[Atr] - oof_g[Atr], y[Ttr] - oof_g[Ttr], wtr)
    R.update(gpbm_Ft=fq[Tte], gpbm_Fa=fq[Ate], gpbm_lam=lam_q, gpb_blup_mom=fq[Tte] + lam_q * (y[Ate] - fq[Ate]))

    # ---------- TRACE members (tuned in pair space)
    Ftr, Fte = pair_features(X, y, Atr, Ttr, tl), pair_features(X, y, Ate, Tte, tl)
    OOF, TE = {}, {}
    for kind in ["lgbm", "xgb", "et", "cat"]:
        bst_ = (np.inf,)
        for ci, c in enumerate(configs(kind)):
            o = np.zeros(len(Atr))
            for j in range(3):
                m = pf == j
                o[m] = fit(kind, c, Ftr[~m], dy[~m], wtr[~m]).predict(Ftr[m])
            err = wmean((dy - o) ** 2, wtr)
            if err < bst_[0]:
                bst_ = (err, ci, c, o)
        sel[f"trace_{kind}"] = bst_[1]
        OOF[kind] = bst_[3]
        TE[kind] = fit(kind, bst_[2], Ftr, dy, wtr).predict(Fte)
        R[f"trace_{kind}"] = y[Ate] + TE[kind]
        if kind == "lgbm":
            c_l = bst_[2]
    R["TRACE"] = y[Ate] + np.mean([TE[m] for m in TE], axis=0)
    sw = np.sqrt(wtr)
    Z = np.column_stack([OOF[m] for m in ["lgbm", "xgb", "et", "cat"]])
    w4, _ = nnls(Z * sw[:, None], dy * sw)
    R["TRACE_stack"] = y[Ate] + np.column_stack([TE[m] for m in ["lgbm", "xgb", "et", "cat"]]) @ w4
    Z6 = np.column_stack([Z, eb_oof - y[Atr], gpbm_oof - y[Atr]])
    w6, _ = nnls(Z6 * sw[:, None], dy * sw)
    R["HYBRID_stack"] = y[Ate] + np.column_stack([TE["lgbm"], TE["xgb"], TE["et"], TE["cat"], R["eb_shift"] - y[Ate], R["gpb_blup_mom"] - y[Ate]]) @ w6
    s2 = wmean((dy - Z.mean(1)) ** 2, wtr)

    # ---------- PADRE (tuned LightGBM, pairs of any two rows) and ablations (config of trace_lgbm)
    bst_ = (np.inf,)
    for ci, c in enumerate(configs("lgbm")):
        o = np.zeros(len(Atr))
        for j, (itr, _) in enumerate(inner):
            m = pf == j
            Ar, Tr = random_pairs(itr, int((~m).sum()), np.random.RandomState(ci * 10 + j))
            o[m] = fit("lgbm", c, pair_features(X, y, Ar, Tr, tl), y[Tr] - y[Ar]).predict(Ftr[m])
        err = wmean((dy - o) ** 2, wtr)
        if err < bst_[0]:
            bst_ = (err, ci, c)
    sel["padre"] = bst_[1]
    Ar, Tr = random_pairs(tr, 3 * len(Atr), rng)
    R["padre"] = y[Ate] + fit("lgbm", bst_[2], pair_features(X, y, Ar, Tr, tl), y[Tr] - y[Ar]).predict(Fte)
    for tag, kw in [("abl_no_ya", dict(use_ya=False)), ("abl_no_site", dict(use_site=False)), ("abl_no_delta", dict(use_delta=False))]:
        R[tag] = y[Ate] + fit("lgbm", c_l, pair_features(X, y, Atr, Ttr, tl, **kw), dy, wtr).predict(pair_features(X, y, Ate, Tte, tl, **kw))
    R = pd.DataFrame(R).assign(rep=rep, fold=k, study=study[Tte], exp=exp[Tte], s2=s2)
    meta = dict(endpoint=e, rep=rep, fold=k, secs=round(time.time() - t0, 1), **{f"cfg_{a}": b_ for a, b_ in sel.items()},
                w_lgbm=w4[0], w_xgb=w4[1], w_et=w4[2], w_cat=w4[3], h_lgbm=w6[0], h_xgb=w6[1], h_et=w6[2], h_cat=w6[3], h_eb=w6[4], h_gpb=w6[5],
                lam_eb=lam, lam_merf=lam_m, lam_gpb=lam_p, lam_gpbm=lam_q, gpb_var_err=cp[0], gpb_var_group=cp[1])
    print(f"done {e} rep{rep} fold{k} {meta['secs']}s", flush=True)
    return e, R, meta


METHODS = ["zero", "gpb_zero", "persist", "shift", "eb_shift", "merf_blup", "gpb_blup", "gpb_blup_mom", "padre", "abl_no_ya", "abl_no_site",
           "abl_no_delta", "trace_lgbm", "trace_xgb", "trace_et", "trace_cat", "TRACE", "TRACE_stack", "HYBRID_stack"]


def wr2(y, p, w):
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)


def summarise(B=1000):
    summ, dif, ks = [], [], []
    for e in ENDPOINTS:
        R = pd.read_csv(RES / f"nested_pairs_{e}.csv")
        w = (1 / R.groupby(["rep", "t"]).y.transform("size")).values
        y = R.y.values; st = R.study.values; u = np.unique(st); ix = {s: np.where(st == s)[0] for s in u}
        rng = np.random.RandomState(0)
        boot = {m: [] for m in METHODS}
        for _ in range(B):
            s_ = np.concatenate([ix[s] for s in rng.choice(u, len(u))])
            for m in METHODS:
                boot[m].append(wr2(y[s_], R[m].values[s_], w[s_]))
        boot = {m: np.array(v) for m, v in boot.items()}
        sse_p = np.sum(w * (y - R.persist) ** 2)
        for m in METHODS:
            summ.append(dict(endpoint=e, method=m, n_targets=R.t.nunique(), n_studies=len(u), R2=wr2(y, R[m].values, w),
                             lo=np.percentile(boot[m], 2.5), hi=np.percentile(boot[m], 97.5),
                             RMSE=float(np.sqrt(np.average((y - R[m]) ** 2, weights=w))), contrast_skill=1 - np.sum(w * (y - R[m]) ** 2) / sse_p))
        for top in ["TRACE", "TRACE_stack", "HYBRID_stack"]:
            for base in ["persist", "eb_shift", "merf_blup", "gpb_blup", "gpb_blup_mom", "padre", "abl_no_ya", "abl_no_site", "abl_no_delta", "TRACE"]:
                if base == top:
                    continue
                v = boot[top] - boot[base]
                dif.append(dict(endpoint=e, top=top, base=base, dR2=wr2(y, R[top].values, w) - wr2(y, R[base].values, w),
                                lo=np.percentile(v, 2.5), hi=np.percentile(v, 97.5), p_le0=float((v <= 0).mean())))
        # k-shot on common targets (>=3 anchors): proper k-anchor BLUP for mixed-effects baselines
        R["n_anchor"] = R.groupby(["rep", "t"]).y.transform("size")
        C = R[R.n_anchor >= 3].copy()
        C["w_prec"] = 1.0
        for kk in [1, 2, 3]:
            acc = []
            for draw in range(5):
                r2_ = np.random.RandomState(100 + draw)
                C["u"] = r2_.rand(len(C))
                S = C[C.groupby(["rep", "t"]).u.rank(method="first") <= kk]
                gb = S.groupby(["rep", "t"])
                L = gb.agg(y=("y", "first"), persist=("persist", "mean"), TRACE=("TRACE", "mean"), TRACE_stack=("TRACE_stack", "mean"), HYBRID_stack=("HYBRID_stack", "mean"))
                for nm in ["eb", "merf", "gpb", "gpbm"]:
                    lam1 = gb[f"{nm}_lam"].first().clip(0, 0.999); rho = lam1 / (1 - lam1)
                    resid = (S.ya - S[f"{nm}_Fa"]).groupby([S.rep, S.t]).mean()
                    L[nm] = gb[f"{nm}_Ft"].first() + kk * rho / (kk * rho + 1) * resid
                acc.append({m: r2_score(L.y, L[m]) for m in L.columns if m != "y"})
            ks.append(dict(endpoint=e, k=kk, n_targets=C.groupby(["rep", "t"]).ngroups // R.rep.nunique(), **pd.DataFrame(acc).mean().to_dict()))
    summ, dif, ks = pd.DataFrame(summ), pd.DataFrame(dif), pd.DataFrame(ks)
    summ.to_csv(TAB / "T10_nested_summary.csv", index=False); dif.to_csv(TAB / "T10_nested_paired_bootstrap.csv", index=False)
    ks.rename(columns=dict(eb="eb_blup_k", merf="merf_blup_k", gpb="gpb_blup_k", gpbm="gpb_blup_mom_k")).to_csv(TAB / "T10_nested_kshot_common.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print("== nested-tuned 1-shot R2 on held-out studies\n", summ.pivot(index="method", columns="endpoint", values="R2").loc[METHODS, ENDPOINTS].round(3).to_string())
    print("== 95% CI\n", summ[summ.method.isin(["TRACE", "TRACE_stack", "HYBRID_stack"])][["endpoint", "method", "R2", "lo", "hi", "RMSE"]].round(3).to_string())
    print("== paired bootstrap (top - base)\n", dif[dif.base.isin(["eb_shift", "merf_blup", "gpb_blup", "gpb_blup_mom", "TRACE"])].round(3).to_string())
    print("== k-shot, common targets, proper k-anchor BLUP for baselines\n", ks.round(3).to_string())


def main():
    df = load()
    jobs = []
    for e in ENDPOINTS:
        d = df[df[f"y_{e}"].notna()].reset_index(drop=True)
        X, y = design(d), d[f"y_{e}"].values
        for rep in range(REPS):
            for k, (tr, te) in enumerate(study_folds(d.StudyID.values, 5, SEED + rep)):
                jobs.append(delayed(tune_fold)(e, rep, k, tr, te, d, X, y))
    res = Parallel(n_jobs=10, verbose=5)(jobs)
    for e in ENDPOINTS:
        pd.concat([r for ee, r, m in res if ee == e], ignore_index=True).to_csv(RES / f"nested_pairs_{e}.csv", index=False)
    pd.DataFrame([m for _, _, m in res]).to_csv(TAB / "T10_selected_configs.csv", index=False)
    pd.DataFrame([dict(learner=k_, cfg=i, **c) for k_ in SPACE for i, c in enumerate(configs(k_))]).to_csv(TAB / "T10_config_grid.csv", index=False)
    summarise()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarise()
    else:
        main()
