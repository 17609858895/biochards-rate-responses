"""03_benchmark_models.py -- 20 individual learners + stacking + voting, random-row CV vs study-grouped CV.

Question: how much of the apparent accuracy of covariate-only ("zero-shot") models survives when whole
studies are held out?  Target = lnRR of each endpoint.  All preprocessing is fitted inside training folds.
Outputs: tables/T03_benchmark_long.csv, tables/T03_benchmark_summary.csv, results/bench_oof_<endpoint>.csv
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from scipy.optimize import nnls
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor

ENDPOINTS = ["CropYield", "SOC", "pH", "TN", "N2O", "GWP"]
NJ = 2


def prep(scale=True):
    steps = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        steps.append(StandardScaler())
    return steps


def P(est, scale=True):
    return make_pipeline(*prep(scale), est)


MODELS = {
    # unpenalised OLS was removed: one-hot + missing-indicator columns are exactly collinear and the
    # min-norm solution explodes on held-out rows (R2 < -1e20); Ridge/Lasso/ElasticNet cover the linear family.
    "Ridge": lambda: P(Ridge(alpha=30.0)),
    "Lasso": lambda: P(Lasso(alpha=0.005, max_iter=5000)),
    "ElasticNet": lambda: P(ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=5000)),
    "BayesRidge": lambda: P(BayesianRidge()),
    "Huber": lambda: P(HuberRegressor(alpha=10.0, max_iter=500)),
    "PLS": lambda: P(PLSRegression(n_components=4)),
    "KNN": lambda: P(KNeighborsRegressor(n_neighbors=15, weights="distance")),
    "SVR": lambda: P(SVR(C=0.5, epsilon=0.05, gamma="scale")),
    "GPR": lambda: P(GaussianProcessRegressor(ConstantKernel(1.0) * RBF(5.0) + WhiteKernel(0.5), normalize_y=True,
                                              random_state=0)),
    "DecisionTree": lambda: P(DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=0), scale=False),
    "RandomForest": lambda: P(RandomForestRegressor(400, min_samples_leaf=5, max_features=0.4, n_jobs=NJ, random_state=0), scale=False),
    "ExtraTrees": lambda: P(ExtraTreesRegressor(400, min_samples_leaf=5, max_features=0.4, n_jobs=NJ, random_state=0), scale=False),
    "Bagging": lambda: P(BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=5), n_estimators=200, max_samples=0.7, n_jobs=NJ, random_state=0), scale=False),
    "AdaBoost": lambda: P(AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=200, learning_rate=0.05, random_state=0), scale=False),
    "GBDT": lambda: P(GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, min_samples_leaf=10, random_state=0), scale=False),
    "HistGB": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=5.0, random_state=0),
    "XGBoost": lambda: xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=10, subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, n_jobs=NJ, random_state=0),
    "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, verbose=-1, n_jobs=NJ, random_state=0),
    "CatBoost": lambda: CatBoostRegressor(iterations=400, learning_rate=0.04, depth=5, l2_leaf_reg=5.0, verbose=0, thread_count=NJ, random_seed=0),
    "MLP": lambda: P(MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1.0, learning_rate_init=0.003, max_iter=600, early_stopping=True, random_state=0)),
}
STACK_BASE = ["Ridge", "SVR", "KNN", "RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost", "MLP"]


def fit_predict(name, Xtr, ytr, Xte):
    m = MODELS[name]()
    m.fit(Xtr, ytr)
    return np.asarray(m.predict(Xte)).ravel()


def one_fold(e, scheme, rep, k, tr, te, X, y, groups):
    Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y[tr]
    out, preds = [], {}
    for name in MODELS:
        t0 = time.time()
        try:
            p = fit_predict(name, Xtr, ytr, Xte)
        except Exception as ex:  # keep the run alive, record the failure
            p = np.full(len(te), ytr.mean())
            print("FAIL", e, scheme, name, repr(ex)[:80], flush=True)
        preds[name] = p
    # --- stacking: inner folds follow the outer scheme (grouped inside grouped)
    inner = study_folds(groups[tr], 5, seed=SEED + 77) if scheme == "study" else row_folds(len(tr), 5, seed=SEED + 77)
    Z = np.zeros((len(tr), len(STACK_BASE)))
    for itr, ite in inner:
        for j, name in enumerate(STACK_BASE):
            Z[ite, j] = fit_predict(name, Xtr.iloc[itr], ytr[itr], Xtr.iloc[ite])
    Zte = np.column_stack([preds[n] for n in STACK_BASE])
    mu = ytr.mean()
    wts, _ = nnls(Z - mu, ytr - mu)                       # non-negative least squares meta-learner
    preds["Stack_NNLS"] = mu + (Zte - mu) @ wts
    preds["Vote_GBM3"] = np.mean([preds[n] for n in ["XGBoost", "LightGBM", "CatBoost"]], axis=0)
    preds["MeanBaseline"] = np.full(len(te), mu)
    rows = []
    for name, p in preds.items():
        rows.append(dict(endpoint=e, scheme=scheme, rep=rep, fold=k, model=name, n_test=len(te),
                         sse=float(((y[te] - p) ** 2).sum()), sae=float(np.abs(y[te] - p).sum())))
    oof = pd.DataFrame({"row": te, "rep": rep, "scheme": scheme, **{n: preds[n] for n in ["RandomForest", "LightGBM", "CatBoost", "Stack_NNLS"]}})
    return rows, oof, dict(endpoint=e, scheme=scheme, rep=rep, fold=k, **dict(zip(STACK_BASE, wts)))


def main(reps=2):
    df = load()
    jobs, store = [], {}
    for e in ENDPOINTS:
        d = df[df[f"y_{e}"].notna()].reset_index(drop=True)
        X, y, g = design(d), d[f"y_{e}"].values, d.StudyID.values
        store[e] = (d, y)
        for rep in range(reps):
            for scheme, folds in [("random", row_folds(len(d), 5, SEED + rep)), ("study", study_folds(g, 5, SEED + rep))]:
                for k, (tr, te) in enumerate(folds):
                    jobs.append(delayed(one_fold)(e, scheme, rep, k, tr, te, X, y, g))
    print(len(jobs), "fold jobs", flush=True)
    res = Parallel(n_jobs=14, verbose=5)(jobs)
    long = pd.DataFrame([r for rr, _, _ in res for r in rr])
    long.to_csv(TAB / "T03_benchmark_long.csv", index=False)
    pd.DataFrame([w for _, _, w in res]).to_csv(TAB / "T03_stack_weights.csv", index=False)
    oof = pd.concat([o.assign(endpoint=e_) for (rr, o, w), e_ in zip(res, [w["endpoint"] for _, _, w in res])])
    oof.to_csv(RES / "bench_oof.csv", index=False)
    # pooled R2 per (endpoint, scheme, rep, model) then mean/sd across reps
    rows = []
    for (e, scheme, rep, model), s in long.groupby(["endpoint", "scheme", "rep", "model"]):
        d, y = store[e]
        sst = ((y - y.mean()) ** 2).sum()
        rows.append(dict(endpoint=e, scheme=scheme, rep=rep, model=model, R2=1 - s.sse.sum() / sst,
                         RMSE=np.sqrt(s.sse.sum() / s.n_test.sum()), MAE=s.sae.sum() / s.n_test.sum()))
    S = pd.DataFrame(rows).groupby(["endpoint", "scheme", "model"]).agg(R2=("R2", "mean"), R2_sd=("R2", "std"), RMSE=("RMSE", "mean"), MAE=("MAE", "mean")).reset_index()
    S.to_csv(TAB / "T03_benchmark_summary.csv", index=False)
    piv = S.pivot_table(index="model", columns=["endpoint", "scheme"], values="R2").round(3)
    print(piv.loc[piv[("CropYield", "study")].sort_values(ascending=False).index].to_string())


if __name__ == "__main__":
    main()
