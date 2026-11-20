"""01_build_dataset.py -- clean BiocharDS V1.0 into one analysis table.

Raw files are never modified. Output: data_clean/biochar_clean.csv + tables/T01_*.csv
Rules
  * lnRR recomputed from CK and T means (ln(T/CK)); rows with non-positive means dropped for that endpoint.
  * sampling variance v = SD_T^2/(N_T*T^2) + SD_CK^2/(N_CK*CK^2)   (Hedges et al. 1999)
  * categorical levels harmonised (whitespace/case, feedstock and crop classes)
  * no target value is imputed; predictor imputation happens inside model pipelines only.
"""
import re, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "BiocharDS_V1.0.xlsx"
OUT = ROOT / "data_clean"; OUT.mkdir(exist_ok=True)
TAB = ROOT / "tables"; TAB.mkdir(exist_ok=True)

ENDPOINTS = ["CropYield", "SOC", "TN", "pH", "CNRatio", "DOC", "NH4_N", "NO3_N", "OlsenP", "CEC",
             "K", "Ca", "Mg", "Na", "SMBC", "SMBN", "CO2", "CH4", "N2O", "GWP", "GHGI"]


def num(s):
    """coerce to float; '500-700' -> 600 ; '156. 13' -> 156.13"""
    def f(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        t = str(x).strip().replace(" ", "")
        m = re.fullmatch(r"(\d+\.?\d*)-(\d+\.?\d*)", t)
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2
        try:
            return float(t)
        except ValueError:
            return np.nan
    return s.map(f)


FEED = [
    ("manure_sludge", r"manure|litter|sludge|feedlot|pig|biowaste|domestic waste|mushroom|compost"),
    ("husk_shell", r"husk|hull|shell|chaff|cob|bran|pomace|bagasse|kernel"),
    ("wood", r"wood|tree|branch|bark|bamboo|pine|jarrah|charcoal|sawdust|prunings|spruce|twigs|trunks|vine|acacia|eucalyptus|gliricidia|elaeagnus"),
    ("cereal_straw", r"wheat straw|maize straw|rice straw|barley straw|wheat and rice straw|maize silage|sorghum|^straw$|crop straw"),
    ("other_straw", r"straw|stock|stem|rapeseed|reed|miscanthus|eupatorium|green waste|vegetable waste"),
]


def feed_class(x):
    if pd.isna(x):
        return np.nan
    t = str(x).strip().lower()
    if t.startswith("mix"):
        return "mixed"
    for k, pat in FEED:
        if re.search(pat, t):
            return k
    return "other"


def feed_detail(x):
    if pd.isna(x):
        return np.nan
    t = str(x).strip().lower()
    for k in ["wheat straw", "maize straw", "rice straw", "rice husk", "wood"]:
        if t == k:
            return k
    return feed_class(x)


CROP = {
    "maize": "maize", "rice": "rice", "wheat": "wheat", "soybean-wheat": "wheat",
    "soybean": "legume_oil", "peanut": "legume_oil", "rapeseed": "legume_oil", "mung bean": "legume_oil",
    "sesame": "legume_oil", "sunflower": "legume_oil", "phaseolus calcaltus": "legume_oil",
    "potato": "root_tuber", "sweet potato": "root_tuber", "cassava": "root_tuber", "radish": "root_tuber", "turnip": "root_tuber",
    "sorghum": "other_cereal", "barley": "other_cereal", "millet": "other_cereal", "pearl millet": "other_cereal",
    "tobacco": "cash", "tobaco": "cash", "cotton": "cash", "tea plants": "cash", "grape": "cash", "plantain": "cash",
}
FORAGE = r"grass|festuca|pasture|lucerne|phleum|stylosanthes|miscanthus|sudangrass"


def crop_class(x):
    if pd.isna(x):
        return np.nan
    t = str(x).replace("\xa0", " ").strip().lower()
    if t in CROP:
        return CROP[t]
    if re.search(FORAGE, t):
        return "forage"
    return "vegetable_other"


def main():
    df = pd.read_excel(RAW, sheet_name="Sheet1")
    df.columns = [c.strip().replace(" _", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.insert(0, "row_id", np.arange(len(df)))
    df = df.rename(columns={"Experiment ID": "ExpID"})
    # ---- numeric coercion
    for c in ["NH4_N_initial", "OlsenP_initial", "PyrolysisTemperature_biochar", "OlsenP_RR"]:
        df[c] = num(df[c])
    # ---- categories
    for c in ["Country", "ClimateZone", "Texture_initial", "Treatment"]:
        df[c] = df[c].astype("string").str.strip()
    df["Country"] = df["Country"].replace({"Greek": "Greece"})
    df["Texture_initial"] = df["Texture_initial"].str.lower()
    df["FeedClass"] = df["Type_biochar"].map(feed_class)
    df["FeedDetail"] = df["Type_biochar"].map(feed_detail)
    df["CropClass"] = df["CropType"].map(crop_class)
    # fertiliser co-treatment flags from the treatment code (difference vs. control)
    tr = df["Treatment"].fillna("B")
    df["co_N"] = tr.str.contains("N").astype(int)
    df["co_P"] = tr.str.contains("P").astype(int)
    df["co_K"] = tr.str.contains("K").astype(int)
    df["co_M"] = tr.str.contains("M").astype(int)
    df["co_any"] = (tr != "B").astype(int)
    tex_rank = {"sand": 0, "loamy sand": 1, "sandy loam": 2, "loam": 3, "silty loam": 3, "sandy clay loam": 4,
                "clay loam": 5, "silty clay loam": 5, "sandy clay": 6, "loamy clay": 6, "silty clay": 7, "clay": 8}
    df["TextureRank"] = df["Texture_initial"].map(tex_rank).astype(float)
    df["absLat"] = df["Latitude"].abs()
    df["logDose"] = np.log(df["BiocharAddition"])
    df["logDuration"] = np.log(df["TrialDuration"])
    # ---- effect sizes
    audit = []
    for e in ENDPOINTS:
        ck, t = df[f"{e}_CK"], df[f"{e}_T"]
        sdc, sdt = df[f"{e}_CK_SD"], df[f"{e}_T_SD"]
        nc, nt = df[f"{e}_CK_N"], df[f"{e}_T_N"]
        ok = (ck > 0) & (t > 0)
        y = pd.Series(np.nan, index=df.index)
        y[ok] = np.log(t[ok] / ck[ok])
        v = sdt ** 2 / (nt * t ** 2) + sdc ** 2 / (nc * ck ** 2)
        v = v.where(ok & (v > 0))
        df[f"y_{e}"] = y
        df[f"v_{e}"] = v
        cvt = sdt / t.abs()
        audit.append(dict(endpoint=e, n_pairs=int((ck.notna() & t.notna()).sum()), n_lnRR=int(y.notna().sum()),
                          n_nonpositive=int(((ck <= 0) | (t <= 0)).sum()), n_var=int(v.notna().sum()),
                          n_studies=int(df.loc[y.notna(), "StudyID"].nunique()),
                          n_exps=int(df.loc[y.notna(), "ExpID"].nunique()),
                          mean_lnRR=y.mean(), sd_lnRR=y.std(), median_v=v.median(), mean_v=v.mean(),
                          share_cv_t_0p05=float(((cvt - 0.05).abs() < 2e-3)[y.notna()].mean()),
                          share_cv_t_0p10=float(((cvt - 0.10).abs() < 2e-3)[y.notna()].mean()),
                          max_abs_diff_vs_db=float((y - df[f"{e}_LNRR"]).abs().max())))
    aud = pd.DataFrame(audit)
    aud.to_csv(TAB / "T01_endpoint_audit.csv", index=False)
    # ---- shared-control id (same experiment + same control yield value)
    key = df["StudyID"].astype(str) + "|" + df["ExpID"].astype(str) + "|" + df["CropYield_CK"].round(6).astype(str)
    df["CtrlID"] = key.factorize()[0]
    df["ExpKey"] = (df["StudyID"].astype(str) + "|" + df["ExpID"].astype(str)).factorize()[0]
    df.to_csv(OUT / "biochar_clean.csv", index=False, encoding="utf-8-sig")
    meta = dict(n_rows=len(df), n_studies=int(df.StudyID.nunique()), n_exps=int(df.ExpKey.nunique()),
                n_ctrl_groups=int(df.CtrlID.nunique()), endpoints=ENDPOINTS)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    print(aud.round(4).to_string())
    print(df["FeedClass"].value_counts(dropna=False).to_dict())
    print(df["FeedDetail"].value_counts(dropna=False).to_dict())
    print(df["CropClass"].value_counts(dropna=False).to_dict())
    print(df["ClimateZone"].value_counts(dropna=False).to_dict())
    print("other feed:", sorted(df.loc[df.FeedClass == "other", "Type_biochar"].str.lower().unique()))


if __name__ == "__main__":
    main()
