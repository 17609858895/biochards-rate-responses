"""11_zero_shot_nested.py -- does hyper-parameter tuning rescue covariate-only (zero-shot) prediction?

Ten learner families, 10 configurations each (configuration 0 = the fixed setting of 03), selected in a 3-fold inner
CV that follows the outer scheme (random rows inside random rows; studies inside studies), plus an NNLS stack of the
tuned learners built on inner out-of-fold predictions.  Outer: 5-fold, both schemes.
Output: tables/T11_zero_shot_nested.csv, tables/T11_selected_configs.csv
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor

ENDPOINTS = ["CropYield", "SOC", "pH", "TN", "N2O", "GWP"]
N_CFG, NT = 10, 3


def lin(est):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), est)


def tree(est):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), est)


SPACE = {
    "Ridge": (dict(alpha=[0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 300.0, 1000.0, 3000.0]), dict(alpha=30.0), lambda c: lin(Ridge(**c))),
    "ElasticNet": (dict(alpha=[0.003, 0.03, 0.1, 0.3], l1_ratio=[0.1, 0.5, 0.9]), dict(alpha=0.01, l1_ratio=0.3), lambda c: lin(ElasticNet(max_iter=5000, **c))),
    "SVR": (dict(C=[0.1, 0.3, 1.0, 3.0], epsilon=[0.02, 0.05, 0.1], gamma=["scale", 0.003, 0.01]), dict(C=0.5, epsilon=0.05, gamma="scale"), lambda c: lin(SVR(**c))),
    "KNN": (dict(n_neighbors=[5, 10, 25, 40, 60], weights=["uniform", "distance"]), dict(n_neighbors=15, weights="distance"), lambda c: lin(KNeighborsRegressor(**c))),
    "RandomForest": (dict(min_samples_leaf=[2, 5, 10, 20], max_features=[0.2, 0.4, 0.6, 0.8], max_depth=[None, 8, 14]), dict(min_samples_leaf=5, max_features=0.4, max_depth=None),
                     lambda c: tree(RandomForestRegressor(400, n_jobs=NT, random_state=0, **c))),
    "ExtraTrees": (dict(min_samples_leaf=[2, 5, 10, 20], max_features=[0.2, 0.4, 0.6, 0.8], max_depth=[None, 8, 14]), dict(min_samples_leaf=5, max_features=0.4, max_depth=None),
                   lambda c: tree(ExtraTreesRegressor(400, n_jobs=NT, random_state=0, **c))),
    "HistGB": (dict(max_iter=[150, 300, 600], learning_rate=[0.02, 0.03, 0.05], max_leaf_nodes=[7, 15, 31], min_samples_leaf=[10, 20, 40], l2_regularization=[1.0, 5.0, 20.0]),
               dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=5.0), lambda c: HistGradientBoostingRegressor(random_state=0, **c)),
    "XGBoost": (dict(n_estimators=[150, 300, 600], learning_rate=[0.02, 0.03, 0.05], max_depth=[3, 4, 6], min_child_weight=[5, 10, 20], colsample_bytree=[0.4, 0.6, 0.8], reg_lambda=[1.0, 5.0, 20.0]),
                dict(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=10, colsample_bytree=0.6, reg_lambda=5.0),
                lambda c: xgb.XGBRegressor(subsample=0.8, n_jobs=NT, random_state=0, **c)),
    "LightGBM": (dict(n_estimators=[150, 300, 600], learning_rate=[0.02, 0.03, 0.05], num_leaves=[7, 15, 31], min_child_samples=[10, 20, 40], colsample_bytree=[0.4, 0.6, 0.8], reg_lambda=[1.0, 5.0, 20.0]),
                 dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, colsample_bytree=0.6, reg_lambda=5.0),
                 lambda c: lgb.LGBMRegressor(subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=NT, random_state=0, **c)),
    "CatBoost": (dict(iterations=[200, 400], learning_rate=[0.04, 0.08], depth=[4, 5, 6], l2_leaf_reg=[3.0, 5.0, 10.0]), dict(iterations=400, learning_rate=0.04, depth=5, l2_leaf_reg=5.0),
                 lambda c: CatBoostRegressor(verbose=0, thread_count=NT, random_seed=0, **c)),
}


def configs(name, n=N_CFG, seed=11):
    space, default, _ = SPACE[name]
    size = int(np.prod([len(v) for v in space.values()]))
    rng = np.random.RandomState(seed + sum(map(ord, name)))
    out = [default]
    while len(out) < min(n, size + 1):
        c = {k: v[rng.randint(len(v))] for k, v in space.items()}
        if c not in out:
            out.append(c)
    return out


def one_fold(e, scheme, k, tr, te, X, y, groups):
    t0 = time.time()
    inner = study_folds(groups[tr], 3, seed=SEED + 9) if scheme == "study" else row_folds(len(tr), 3, seed=SEED + 9)
    Xtr, ytr, Xte = X.iloc[tr], y[tr], X.iloc[te]
    rows, OOF, TE, sel = [], {}, {}, dict(endpoint=e, scheme=scheme, fold=k)
    for name, (_, _, mk) in SPACE.items():
        best = (np.inf,)
        for ci, c in enumerate(configs(name)):
            o = np.zeros(len(tr))
            for itr, ite in inner:
                o[ite] = mk(c).fit(Xtr.iloc[itr], ytr[itr]).predict(Xtr.iloc[ite])
            err = float(np.mean((ytr - o) ** 2))
            if ci == 0:
                o0 = o.copy()
            if err < best[0]:
                best = (err, ci, c, o)
        sel[name] = best[1]
        OOF[name] = best[3]
        TE[name] = mk(best[2]).fit(Xtr, ytr).predict(Xte)
        TE[name + "_default"] = mk(configs(name)[0]).fit(Xtr, ytr).predict(Xte)
    mu = ytr.mean()
    names = list(SPACE)
    wts, _ = nnls(np.column_stack([OOF[n] for n in names]) - mu, ytr - mu)
    TE["Stack_tuned"] = mu + (np.column_stack([TE[n] for n in names]) - mu) @ wts
    TE["MeanBaseline"] = np.full(len(te), mu)
    for n, p in TE.items():
        rows.append(dict(endpoint=e, scheme=scheme, fold=k, model=n, n_test=len(te), sse=float(((y[te] - p) ** 2).sum())))
    print(f"done {e} {scheme} fold{k} {time.time() - t0:.0f}s", flush=True)
    return rows, sel


def main():
    df = load()
    jobs, sst = [], {}
    for e in ENDPOINTS:
        d = df[df[f"y_{e}"].notna()].reset_index(drop=True)
        X, y, g = design(d), d[f"y_{e}"].values, d.StudyID.values
        sst[e] = ((y - y.mean()) ** 2).sum()
        for scheme, folds in [("random", row_folds(len(d), 5, SEED)), ("study", study_folds(g, 5, SEED))]:
            for k, (tr, te) in enumerate(folds):
                jobs.append(delayed(one_fold)(e, scheme, k, tr, te, X, y, g))
    res = Parallel(n_jobs=10, verbose=5)(jobs)
    long = pd.DataFrame([r for rr, _ in res for r in rr])
    pd.DataFrame([s for _, s in res]).to_csv(TAB / "T11_selected_configs.csv", index=False)
    S = long.groupby(["endpoint", "scheme", "model"]).sse.sum().reset_index()
    S["R2"] = 1 - S.sse / S.endpoint.map(sst)
    S.to_csv(TAB / "T11_zero_shot_nested.csv", index=False)
    pd.set_option("display.width", 250)
    piv = S.pivot_table(index="model", columns=["endpoint", "scheme"], values="R2")[[(e, s) for e in ENDPOINTS for s in ["random", "study"]]].round(3)
    piv.columns = [f"{e[:5]}_{s[:4]}" for e, s in piv.columns]
    print(piv.to_string())


if __name__ == "__main__":
    main()
