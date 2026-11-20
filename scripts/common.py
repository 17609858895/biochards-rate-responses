"""common.py -- shared loaders, feature sets, splitters and metrics for the BiocharDS study."""
from pathlib import Path
import warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_clean" / "biochar_clean.csv"
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
TAB = ROOT / "tables"; TAB.mkdir(exist_ok=True)
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
SEED = 20261112

CLIMATE = ["Latitude", "Longitude", "Temperature", "Precipitation", "WetnessIndex"]
SOIL = ["SOC_initial", "TN_initial", "pH_initial", "BD_initial", "CEC_initial", "OlsenP_initial",
        "Sand_initial", "Silt_initial", "Clay_initial", "TextureRank"]
BIOCHAR = ["PyrolysisTemperature_biochar", "Ash_biochar", "C_biochar", "N_biochar", "P_biochar",
           "CNRatio_biochar", "pH_biochar", "CEC_biochar", "ParticleSize_biochar"]
MANAGE = ["logDose", "logDuration", "SoilDepth", "co_N", "co_P", "co_K", "co_M"]
CAT = ["ClimateZone", "FeedDetail", "CropClass"]
NUM = CLIMATE + SOIL + BIOCHAR + MANAGE
GROUPS = {"Climate": CLIMATE + ["ClimateZone"], "Soil": SOIL, "Biochar": BIOCHAR + ["FeedDetail"],
          "Management": MANAGE + ["CropClass"]}
# variables that vary inside an experiment (treatment-level) vs. those fixed per site
TREAT_LEVEL = ["logDose", "co_N", "co_P", "co_K", "co_M", "PyrolysisTemperature_biochar", "Ash_biochar",
               "C_biochar", "N_biochar", "P_biochar", "CNRatio_biochar", "pH_biochar", "CEC_biochar",
               "ParticleSize_biochar", "FeedDetail"]


def load():
    df = pd.read_csv(DATA)
    for c in CAT:
        df[c] = df[c].astype("object")
    return df


def design(df, extra_num=(), cats=CAT, num=NUM):
    """Return numeric design matrix with one-hot categories; NaN kept (trees) -- impute inside pipelines."""
    X = df[list(num) + list(extra_num)].astype(float).copy()
    for c in cats:
        d = pd.get_dummies(df[c].fillna("missing"), prefix=c, dtype=float)
        X = pd.concat([X, d], axis=1)
    X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt") for c in X.columns]
    return X


def study_folds(groups, n_splits=5, seed=SEED):
    """Shuffled GroupKFold: groups are randomly permuted before assignment (repeatable)."""
    g = pd.Series(groups).values
    u = np.unique(g)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(u))
    fold_of = {gid: i % n_splits for i, gid in zip(range(len(u)), u[perm])}
    f = np.array([fold_of[x] for x in g])
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(n_splits)]


def row_folds(n, n_splits=5, seed=SEED):
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n)))


def metrics(y, p, w=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = dict(R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=float(mean_absolute_error(y, p)),
               r=float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else np.nan,
               sign_acc=float(np.mean((y > 0) == (p > 0))))
    if w is not None:
        out["wR2"] = r2_score(y, p, sample_weight=w)
    return out
