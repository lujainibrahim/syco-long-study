"""Analysis of Study 3 on anticipated responses from close others after a sycophantic vs neutral AI conversation."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT_PATH = "expectations_results.csv"
N_BOOT = 5000

# likert mappings
AGREE_7 = {
    "Strongly disagree": 1, "Disagree": 2, "Somewhat disagree": 3,
    "Neither agree nor disagree": 4, "Somewhat agree": 5, "Agree": 6,
    "Strongly agree": 7,
}
CERTAINTY_7 = {
    "Very uncertain": 1, "Somewhat uncertain": 2, "Uncertain": 3,
    "Neither certain nor uncertain": 4, "Somewhat certain": 5,
    "Certain": 6, "Very certain": 7,
}
HELPFUL_7 = {
    "Not at all helpful": 1, "Slightly helpful": 2, "Somewhat helpful": 3,
    "Moderately helpful": 4, "Quite helpful": 5, "Very helpful": 6,
    "Extremely helpful": 7,
}
EASE_7 = {
    "Not at all easy": 1, "Slightly easy": 2, "Somewhat easy": 3,
    "Moderately easy": 4, "Quite easy": 5, "Very easy": 6,
    "Extremely easy": 7,
}

FK_AI_COLS = ["feeling known_1", "feeling known_2", "feeling known_3", "feeling known_4"]
FK_HUMAN_COLS = [
    "feeling-known-human_1", "feeling-known-human_2",
    "feeling-known-human_3", "feeling-known-human_4",
]
PRIMARY = [("effort_new_1", "Effort (H1)"), ("resolved", "Resolved (H2)")]
EXPLORATORY_OUTCOMES = [("good_new_1", "Feel Good (E5)"), ("feelings", "Share Feelings (E4)")]
AI_EXP = [
    ("helpfulness", "Helpfulness"),
    ("certainty",   "Certainty"),
    ("ease",        "Ease"),
    ("fk_ai",       "FK (AI)"),
]

def sig(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "†" if p < .1 else ""

def likert(s, mapping=AGREE_7):
    if pd.api.types.is_numeric_dtype(s):
        return s
    mapped = s.map(mapping)
    if mapped.notna().sum() == 0:
        return pd.to_numeric(s, errors="coerce")
    return mapped

def cohen_d(model, coef, outcome_series):
    return model.params[coef] / outcome_series.std(ddof=1)

def compute_d(grp, var):
    s = grp.loc[grp["cond"] == "sycophantic", var].dropna()
    n = grp.loc[grp["cond"] == "neutral", var].dropna()
    if len(s) < 2 or len(n) < 2:
        return (np.nan,) * 8
    n1, n2 = len(s), len(n)
    pooled = np.sqrt(((n1 - 1) * s.std() ** 2 + (n2 - 1) * n.std() ** 2) / (n1 + n2 - 2))
    d = (s.mean() - n.mean()) / pooled if pooled > 0 else 0.0
    _, p = stats.ttest_ind(s, n)
    se_d = np.sqrt(n1 + n2) / np.sqrt(n1 * n2) * np.sqrt(1 + d ** 2 * n1 * n2 / (2 * (n1 + n2)))
    return d, p, s.mean(), n.mean(), d - 1.96 * se_d, d + 1.96 * se_d, n1, n2


print("\n[load analytic sample]")
df = pd.read_csv("data/study_03_data.csv")
print(f"  Rows in analytic sample CSV: {len(df)}")
df["cond"] = df["model_condition"].str.strip().map(
    {"syco": "sycophantic", "neutral": "neutral"}
)
df["syco"] = (df["cond"] == "sycophantic").astype(int)
df["ios_n"] = pd.to_numeric(df["ios"], errors="coerce")
for c in ("effort_new_1", "good_new_1"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["resolved"] = likert(df["resolved"], AGREE_7)
df["feelings"] = likert(df["feelings"], AGREE_7)
df["helpfulness"] = likert(df["helpfulness"], HELPFUL_7)
df["certainty"] = likert(df["certainty"], CERTAINTY_7)
df["ease"] = likert(df["ease"], EASE_7)
for c in FK_AI_COLS:
    df[c + "_n"] = likert(df[c], AGREE_7)
df["fk_ai"] = df[[c + "_n" for c in FK_AI_COLS]].mean(axis=1)
if all(c in df.columns for c in FK_HUMAN_COLS):
    for c in FK_HUMAN_COLS:
        df[c + "_n"] = likert(df[c], AGREE_7)
    df["fk_human"] = df[[c + "_n" for c in FK_HUMAN_COLS]].mean(axis=1)

dat = df.copy()
n_other = (dat["relation-type"] == "Other").sum()
n_no_rel = dat["relation-type"].isna().sum()
dat = dat[dat["relation-type"].notna() & (dat["relation-type"] != "Other")].copy()
n_syco = (dat["cond"] == "sycophantic").sum()
n_neut = (dat["cond"] == "neutral").sum()
print(f"  Excluded relation-type 'Other'/missing: {n_other + n_no_rel}")
print(f"  Final analytic N: {len(dat)}  (syco={n_syco}, neutral={n_neut})")
print("\n  Relationship types:")
print(dat["relation-type"].value_counts().to_string())
print(f"\n  IOS (closeness) descriptive: M={dat['ios_n'].mean():.2f}, "
      f"SD={dat['ios_n'].std():.2f}, median={dat['ios_n'].median():.0f}")

results_rows = []

# Primary hypotheses — preregistered IOS-adjusted models
print("\n[primary hypotheses (preregistered): outcome ~ condition + ios]")
print("  Reporting unadjusted AND IOS-adjusted models for transparency,")
print("  per prereg §3.1 (Primary Analyses).\n")
for var, label in PRIMARY:
    sub = dat[[var, "syco", "ios_n"]].dropna()

    # Unadjusted
    Xu = sm.add_constant(sub[["syco"]])
    mu = sm.OLS(sub[var], Xu).fit()
    bu, pu = mu.params["syco"], mu.pvalues["syco"]
    ciu = mu.conf_int().loc["syco"]
    du = partial_d(mu, "syco")

    # IOS-adjusted
    Xa = sm.add_constant(sub[["syco", "ios_n"]])
    ma = sm.OLS(sub[var], Xa).fit()
    ba, pa = ma.params["syco"], ma.pvalues["syco"]
    cia = ma.conf_int().loc["syco"]
    rmse = np.sqrt(ma.mse_resid)
    da = ba / rmse
    syco_m = sub.loc[sub["syco"] == 1, var].mean()
    neut_m = sub.loc[sub["syco"] == 0, var].mean()
    n1, n2 = int((sub["syco"] == 1).sum()), int((sub["syco"] == 0).sum())

    print(f"  {label}")
    print(f"    Unadjusted    b={bu:+.3f} [{ciu[0]:+.3f}, {ciu[1]:+.3f}], p={pu:.3f}{sig(pu)}, d={du:+.3f}")
    print(f"    + IOS         b={ba:+.3f} [{cia[0]:+.3f}, {cia[1]:+.3f}], p={pa:.3f}{sig(pa)}, d={da:+.3f}")
    print(f"    M_syco={syco_m:.2f} (n={n1})   M_neutral={neut_m:.2f} (n={n2})\n")

    results_rows.append({
        "analysis": "Primary (+ IOS)", "subgroup": "Full sample",
        "outcome": label, "cohens_d": da,
        "ci_lower": cia[0] / rmse, "ci_upper": cia[1] / rmse,
        "p_value": pa, "mean_syco": syco_m, "mean_neutral": neut_m,
        "n_syco": n1, "n_neutral": n2,
        "b": ba, "b_ci_lower": cia[0], "b_ci_upper": cia[1], "n_total": len(sub),
    })

# E1 — Moderation by closeness (Condition × IOS interaction)
print("\n[e1 — moderation by closeness (condition × ios)]")
for var, label in PRIMARY:
    s = dat[[var, "syco", "ios_n"]].dropna()
    s["syco_x_ios"] = s["syco"] * s["ios_n"]
    X = sm.add_constant(s[["syco", "ios_n", "syco_x_ios"]])
    m = sm.OLS(s[var], X).fit()
    b, p = m.params["syco_x_ios"], m.pvalues["syco_x_ios"]
    print(f"  {label:14s}  Condition × IOS  b={b:+.3f}  p={p:.3f}{sig(p)}")

# E2 — Mediation: Condition AI experience primary outcome
print("\n[e2 — mediation by ai conversation experience]")
print("  Bootstrap CIs for indirect effects (5000 resamples).")
print("  Mediators: helpfulness, certainty, ease, feeling-known by AI.")
for var, ovlabel in PRIMARY:
    print(f"\n  {ovlabel}")
    for mvar, mlabel in AI_EXP:
        sub = dat[[var, mvar, "syco"]].dropna()
        if len(sub) < 30:
            continue
        Xa = sm.add_constant(sub[["syco"]])
        ma = sm.OLS(sub[mvar], Xa).fit()
        a = ma.params["syco"]
        Xb = sm.add_constant(sub[["syco", mvar]])
        mb = sm.OLS(sub[var], Xb).fit()
        b = mb.params[mvar]
        c_path = sm.OLS(sub[var], Xa).fit().params["syco"]
        c_prime = mb.params["syco"]
        indirect = a * b
        boot = np.empty(N_BOOT)
        idx_all = np.arange(len(sub))
        for i in range(N_BOOT):
            idx = np.random.choice(idx_all, size=len(sub), replace=True)
            bsub = sub.iloc[idx]
            try:
                ba = sm.OLS(bsub[mvar], sm.add_constant(bsub[["syco"]])).fit().params["syco"]
                bb = sm.OLS(bsub[var], sm.add_constant(bsub[["syco", mvar]])).fit().params[mvar]
                boot[i] = ba * bb
            except Exception:
                boot[i] = np.nan
        boot = boot[~np.isnan(boot)]
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        star = "*" if (ci_lo > 0 or ci_hi < 0) else ""
        print(f"    via {mlabel:12s}  a={a:+.3f}  b={b:+.3f}  c={c_path:+.3f}  c'={c_prime:+.3f}  "
              f"indirect={indirect:+.3f} 95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]{star}")

# E3 — AI conversation experience by condition
print("\n[e3 — ai conversation experience by condition]")
for var, label in AI_EXP:
    d, p, sm_, nm_, ci_lo, ci_hi, n1, n2 = compute_d(dat, var)
    print(f"  {label:14s}  M_syco={sm_:.2f} (n={n1})  M_neut={nm_:.2f} (n={n2})  "
          f"d={d:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]  p={p:.3f}{sig(p)}")

# E4 — Willingness to share feelings   /   E5 — Feel Good
print("\n[e4 / e5 — other exploratory outcomes (regressed on condition + ios)]")
for var, label in EXPLORATORY_OUTCOMES:
    sub = dat[[var, "syco", "ios_n"]].dropna()
    X = sm.add_constant(sub[["syco", "ios_n"]])
    m = sm.OLS(sub[var], X).fit()
    b, p = m.params["syco"], m.pvalues["syco"]
    ci = m.conf_int().loc["syco"]
    rmse = np.sqrt(m.mse_resid)
    d = b / rmse
    sm_ = sub.loc[sub["syco"] == 1, var].mean()
    nm_ = sub.loc[sub["syco"] == 0, var].mean()
    n1, n2 = int((sub["syco"] == 1).sum()), int((sub["syco"] == 0).sum())
    print(f"  {label:22s}  b={b:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p:.3f}{sig(p)}  d={d:+.3f}  "
          f"M_syco={sm_:.2f}  M_neut={nm_:.2f}")
    results_rows.append({
        "analysis": "Exploratory (+ IOS)", "subgroup": "Full sample",
        "outcome": label, "cohens_d": d,
        "ci_lower": ci[0] / rmse, "ci_upper": ci[1] / rmse,
        "p_value": p, "mean_syco": sm_, "mean_neutral": nm_,
        "n_syco": n1, "n_neutral": n2,
        "b": b, "b_ci_lower": ci[0], "b_ci_upper": ci[1], "n_total": len(sub),
    })

# E7 — Descriptive patterns by relationship type
print("\n[e7 — descriptive patterns by relationship type (+ ios)]")
def subgroup_ios(grp, label):
    ns = int((grp["syco"] == 1).sum())
    nn = int((grp["syco"] == 0).sum())
    if ns < 5 or nn < 5:
        print(f"\n  {label} (n={len(grp)}, syco={ns}, neut={nn}): skipped — too few obs")
        return
    print(f"\n  {label} (n={len(grp)}, syco={ns}, neut={nn}):")
    for var, ovl in PRIMARY + [("good_new_1", "Feel Good"), ("fk_ai", "FK (AI)")]:
        s = grp[[var, "syco", "ios_n"]].dropna()
        if len(s) < 10:
            continue
        X = sm.add_constant(s[["syco", "ios_n"]])
        m = sm.OLS(s[var], X).fit()
        b, p = m.params["syco"], m.pvalues["syco"]
        ci = m.conf_int().loc["syco"]
        rmse = np.sqrt(m.mse_resid)
        d = b / rmse
        sm_ = s.loc[s["syco"] == 1, var].mean()
        nm_ = s.loc[s["syco"] == 0, var].mean()
        n1, n2 = int((s["syco"] == 1).sum()), int((s["syco"] == 0).sum())
        print(f"    {ovl:12s}  d={d:+.3f}  p={p:.3f}{sig(p)}  b={b:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
        results_rows.append({
            "analysis": "Subgroup: relation-type", "subgroup": label,
            "outcome": ovl, "cohens_d": d,
            "ci_lower": ci[0] / rmse, "ci_upper": ci[1] / rmse,
            "p_value": p, "mean_syco": sm_, "mean_neutral": nm_,
            "n_syco": n1, "n_neutral": n2,
            "b": b, "b_ci_lower": ci[0], "b_ci_upper": ci[1], "n_total": len(s),
        })

for rel in dat["relation-type"].value_counts().index:
    subgroup_ios(dat[dat["relation-type"] == rel], rel)
subgroup_ios(dat[dat["relation-type"].isin(["Friend", "Close friend"])], "All Friends")
subgroup_ios(dat[dat["relation-type"].isin(["Family member"])], "Family only")
subgroup_ios(dat[dat["relation-type"] != "Romantic partner"], "Excl. Romantic")

# Floor/ceiling and balance
print("\n[descriptive: floor/ceiling on primary + exploratory outcomes]")
for var, label in PRIMARY + EXPLORATORY_OUTCOMES:
    vals = dat[var].dropna()
    floor = (vals <= 2).sum() / len(vals) * 100
    ceil = (vals >= 6).sum() / len(vals) * 100
    print(f"  {label:22s}  floor(1-2)={floor:4.0f}%   ceiling(6-7)={ceil:4.0f}%   "
          f"M={vals.mean():.2f}  SD={vals.std():.2f}")

print("\n[descriptive: condition × relationship balance]")
ct = pd.crosstab(dat["relation-type"], dat["cond"], margins=True)
print(ct.to_string())
ct0 = pd.crosstab(dat["relation-type"], dat["cond"])
if ct0.shape[0] >= 2 and ct0.shape[1] >= 2:
    chi2, p_chi, *_ = stats.chi2_contingency(ct0)
    print(f"\n  Chi-square: chi2={chi2:.2f}, p={p_chi:.3f} "
          f"({'balanced' if p_chi > .05 else 'IMBALANCED'})")
