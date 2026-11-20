"""02_structure_diagnostics.py -- where does the variance of the biochar effect sit?

For each endpoint: (i) nested sum-of-squares partition of lnRR (between studies / between experiments within
study / within experiment), (ii) share attributable to sampling error (mean v / var), (iii) 'replicate ceiling':
variance among rows of one experiment that have IDENTICAL recorded treatment descriptors (cannot be explained by
any model using these descriptors), (iv) random-effects pooled effect with study-cluster robust SE.
Output: tables/T02_variance_partition.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

ENDPOINTS = ["CropYield", "SOC", "pH", "TN", "N2O", "GWP", "GHGI", "CH4", "CO2"]
TREAT_DESC = ["BiocharAddition", "Treatment", "Type_biochar", "PyrolysisTemperature_biochar", "pH_biochar", "C_biochar", "N_biochar", "TrialDuration", "CropType"]


def main():
    df = load()
    rows = []
    for e in ENDPOINTS:
        d = df[df[f"y_{e}"].notna()].copy()
        y = d[f"y_{e}"]
        v = d[f"v_{e}"].clip(upper=d[f"v_{e}"].quantile(0.99))
        tot = ((y - y.mean()) ** 2).sum()
        sm = d.groupby("StudyID")[f"y_{e}"].transform("mean")
        em = d.groupby("ExpKey")[f"y_{e}"].transform("mean")
        ss_study, ss_exp, ss_within = ((sm - y.mean()) ** 2).sum(), ((em - sm) ** 2).sum(), ((y - em) ** 2).sum()
        # replicate ceiling: identical descriptors inside one experiment
        key = d.ExpKey.astype(str) + "|" + d[TREAT_DESC].astype(str).agg("|".join, axis=1)
        gm = d.groupby(key)[f"y_{e}"].transform("mean")
        n_dup = int((d.groupby(key)[f"y_{e}"].transform("size") > 1).sum())
        ss_rep = ((y - gm) ** 2).sum()
        # random-effects mean (DL tau2 on study means, robust SE by study)
        w = 1 / (v.fillna(v.median()) + max(y.var() - v.mean(), 1e-6))
        mu = np.sum(w * y) / np.sum(w)
        s = pd.Series(w * (y - mu)).groupby(d.StudyID.values).sum()
        se = np.sqrt(np.sum(s ** 2)) / np.sum(w)
        rows.append(dict(endpoint=e, n=len(d), n_studies=d.StudyID.nunique(), n_exps=d.ExpKey.nunique(), var=y.var(),
                         share_between_study=ss_study / tot, share_between_exp=ss_exp / tot, share_within_exp=ss_within / tot,
                         share_sampling=v.mean() / y.var(), share_identical_descriptor_replicates=ss_rep / tot, n_rows_in_replicate_sets=n_dup,
                         pooled_lnRR=mu, pooled_se=se, pooled_pct=100 * (np.exp(mu) - 1),
                         pct_lo=100 * (np.exp(mu - 1.96 * se) - 1), pct_hi=100 * (np.exp(mu + 1.96 * se) - 1)))
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "T02_variance_partition.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string())


if __name__ == "__main__":
    main()
