"""Joint endpoint responses to higher rates in matched within-experiment pairs."""
from common import RES, TAB, load, np, pd

GOOD = {"CropYield": 1, "SOC": 1, "N2O": -1}


def boot_studies(st, fn, B=500, seed=0):
    rng = np.random.RandomState(seed); u = np.unique(st); ix = {s: np.where(st == s)[0] for s in u}; out = []
    for _ in range(B):
        out.append(fn(np.concatenate([ix[s] for s in rng.choice(u, len(u))])))
    return np.percentile(np.array(out, float), [2.5, 97.5], axis=0)


def C_cobenefit(df):
    rows = []
    for e1, e2 in [("CropYield", "N2O"), ("CropYield", "SOC"), ("SOC", "N2O")]:
        d = df[df[f"y_{e1}"].notna() & df[f"y_{e2}"].notna()].reset_index(drop=True)
        key = d.ExpKey.astype(str) + "|" + d.Treatment.astype(str) + "|" + d.Type_biochar.astype(str) + "|" + d.PyrolysisTemperature_biochar.astype(str)
        recs = []
        for k, g in d.groupby(key):
            g = g.groupby("BiocharAddition", as_index=False).agg(y1=(f"y_{e1}", "mean"), y2=(f"y_{e2}", "mean"), study=("StudyID", "first")).sort_values("BiocharAddition")
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    recs.append(dict(study=g.study.iloc[0], ladder=k, rate_lo=g.BiocharAddition.iloc[i], rate_hi=g.BiocharAddition.iloc[j],
                                     d1=g.y1.iloc[j] - g.y1.iloc[i], d2=g.y2.iloc[j] - g.y2.iloc[i]))
        Pp = pd.DataFrame(recs); Pp.to_csv(RES / f"x14_cobenefit_{e1}_{e2}.csv", index=False)
        s1, s2 = GOOD[e1], GOOD[e2]; b1, b2 = (s1 * Pp.d1 > 0).values, (s2 * Pp.d2 > 0).values; st = Pp.study.values
        for lab, m in [("both improve", b1 & b2), (f"{e1} only", b1 & ~b2), (f"{e2} only", ~b1 & b2), ("neither", ~b1 & ~b2)]:
            ci = boot_studies(st, lambda ix_: m[ix_].mean())
            rows.append(dict(pair=f"{e1}-{e2}", outcome=lab, share=m.mean(), lo=ci[0], hi=ci[1], n_contrasts=len(Pp), n_studies=Pp.study.nunique(),
                             corr=float(np.corrcoef(Pp.d1, Pp.d2)[0, 1])))
    T = pd.DataFrame(rows); T.to_csv(TAB / "T14_cobenefit.csv", index=False); print(T.round(3).to_string())


if __name__ == "__main__":
    C_cobenefit(load())
