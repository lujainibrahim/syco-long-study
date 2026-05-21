"""Analysis of Study 2 on perceptions of support after a sycophantic vs neutral AI conversation."""

import io
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import pingouin as pg

warnings.filterwarnings("ignore")

AGREE_MAP = {
    "Strongly disagree": 1, "Disagree": 2, "Somewhat disagree": 3,
    "Neither agree nor disagree": 4, "Somewhat agree": 5, "Agree": 6,
    "Strongly agree": 7,
}

EMOTIONAL_COLS = ["Q194_1", "Q194_2", "Q194_3"]
ESTEEM_COLS = ["Q192_1", "Q192_2", "Q192_4"]
INFO_COLS = ["Q193_1", "Q193_2", "Q193_3"]
CERTAINTY_COLS = ["Q158_1", "Q158_2", "Q158_3"]
ATTN_COL = "Q192_3"  

EMOTIONAL_LABELS = [
    "Understood what I'm going through",
    "Showed genuine care about how I'm feeling",
    "Made me feel heard",
]
ESTEEM_LABELS = [
    "Reminded me what I'm doing well",
    "Agreed with how I see the situation",
    "Helped me feel less guilty or at fault",
]
INFO_LABELS = [
    "Helped me see from a different angle",
    "Gave me practical advice",
    "Pointed out things I might not have considered",
]
CERTAINTY_LABELS = [
    "More sure about what to do",
    "Clearer about my next steps",
    "More confident in my decisions",
]

def sig(p):
    if p < .001:
        return "***"
    if p < .01:
        return "**"
    if p < .05:
        return "*"
    if p < .10:
        return "."
    return ""

print("\n[load & filter]")
df = pd.read_csv("data/study_02_data.csv")
print(f"  Total responses (excl. header rows): {len(df)}")
df = df[df["Q194_1"].notna()].copy()
print(f"  Rows with Q194 populated: {len(df)}")
df = df[df["Finished"].astype(str).str.strip() == "True"]
df = df[df["consent"].astype(str).str.strip() == "Yes"]
print(f"  Finished + consented + non-preview: {len(df)}")
df["cond"] = df["model_condition"].str.strip().map(
    {"syco": "sycophantic", "neutral": "neutral"}
)
n_no_cond = df["cond"].isna().sum()
df = df[df["cond"].notna()].copy()
df["syco"] = (df["cond"] == "sycophantic").astype(int)
print(f"  Dropped {n_no_cond} with no condition mapping")
# Convert all likert to numeric
for c in EMOTIONAL_COLS + ESTEEM_COLS + INFO_COLS + CERTAINTY_COLS + [ATTN_COL]:
    df[c] = df[c].map(AGREE_MAP)

# Attention check: Q192_3 == "Agree" 
attn_fail = df[ATTN_COL] != 6
n_fail = attn_fail.sum()
n_fail_syco = attn_fail[df["syco"] == 1].sum()
n_fail_neut = attn_fail[df["syco"] == 0].sum()
df = df[~attn_fail].copy()
print(f"\n  Attention check failures: {n_fail} (syco: {n_fail_syco}, neutral: {n_fail_neut})")
n_syco = (df["syco"] == 1).sum()
n_neut = (df["syco"] == 0).sum()
print(f"\n[analytic sample: n = {len(df)}  (sycophantic = {n_syco}, neutral = {n_neut})]")
print("\n[composites & internal consistency]")
df["emotional"] = df[EMOTIONAL_COLS].mean(axis=1)
df["esteem"] = df[ESTEEM_COLS].mean(axis=1)
df["informational"] = df[INFO_COLS].mean(axis=1)
df["certainty"] = df[CERTAINTY_COLS].mean(axis=1)

composites = [
    ("Emotional Support", "emotional", EMOTIONAL_COLS),
    ("Esteem Support", "esteem", ESTEEM_COLS),
    ("Informational Support", "informational", INFO_COLS),
    ("Certainty", "certainty", CERTAINTY_COLS),
]

for label, col, items in composites:
    alpha = pg.cronbach_alpha(df[items])[0]
    for cond_label, cond_val in [("sycophantic", 1), ("neutral", 0)]:
        sub = df.loc[df["syco"] == cond_val, col]
        print(f"  {label:25s} [{cond_label:12s}]: M = {sub.mean():.2f}, "
              f"SD = {sub.std():.2f}, n = {len(sub)}")
    print(f"  {label:25s}  Cronbach's α = {alpha:.3f}\n")

results_rows = []

def regress_on_condition(df, dv_col, dv_label, hypothesis_label):
    sub = df[[dv_col, "syco"]].dropna()
    X = sm.add_constant(sub[["syco"]])
    model = sm.OLS(sub[dv_col], X).fit()

    b = model.params["syco"]
    se = model.bse["syco"]
    ci = model.conf_int().loc["syco"]
    p = model.pvalues["syco"]

    s_vals = sub.loc[sub["syco"] == 1, dv_col]
    n_vals = sub.loc[sub["syco"] == 0, dv_col]
    n1, n2 = len(s_vals), len(n_vals)
    pooled_sd = np.sqrt(((n1 - 1) * s_vals.std() ** 2 + (n2 - 1) * n_vals.std() ** 2)
                        / (n1 + n2 - 2))
    d = b / pooled_sd if pooled_sd > 0 else 0
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    d_ci_lo = d - 1.96 * se_d
    d_ci_hi = d + 1.96 * se_d

    print(f"\n  {hypothesis_label}: {dv_label}")
    print(f"    Sycophantic: M = {s_vals.mean():.3f}, SD = {s_vals.std():.3f}, n = {n1}")
    print(f"    Neutral:     M = {n_vals.mean():.3f}, SD = {n_vals.std():.3f}, n = {n2}")
    print(f"    b = {b:+.3f}, SE = {se:.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}], "
          f"p = {p:.4f}{sig(p)}")
    print(f"    Cohen's d = {d:+.3f}, 95% CI [{d_ci_lo:+.3f}, {d_ci_hi:+.3f}]")

    results_rows.append({
        "hypothesis": hypothesis_label, "outcome": dv_label,
        "mean_syco": round(s_vals.mean(), 3), "sd_syco": round(s_vals.std(), 3), "n_syco": n1,
        "mean_neutral": round(n_vals.mean(), 3), "sd_neutral": round(n_vals.std(), 3), "n_neutral": n2,
        "b": round(b, 4), "se_b": round(se, 4),
        "ci_lo_b": round(ci[0], 4), "ci_hi_b": round(ci[1], 4), "p_value": round(p, 6),
        "cohens_d": round(d, 3), "se_d": round(se_d, 3),
        "ci_lo_d": round(d_ci_lo, 3), "ci_hi_d": round(d_ci_hi, 3),
    })
    return d, p

print("\n[primary hypotheses (h1, h2)]")
d_emo, p_emo = regress_on_condition(df, "emotional", "Emotional Support (composite)", "H1")
d_est, p_est = regress_on_condition(df, "esteem",    "Esteem Support (composite)",    "H2")

print("\n[exploratory: support dimensions not in h1/h2 (e1, e2)]")
d_info, p_info = regress_on_condition(df, "informational", "Informational Support (composite)", "E1")
d_cert, p_cert = regress_on_condition(df, "certainty",     "Certainty (composite)",             "E2")

print("\n[e3: effect-size comparison across support dimensions]")
print(f"\n  {'Support type':25s}  {'d':>7s}  {'p':>8s}")
print(f"  {'-' * 45}")
for label, d_val, p_val in [
    ("Emotional Support", d_emo, p_emo),
    ("Esteem Support", d_est, p_est),
    ("Informational Support", d_info, p_info),
    ("Certainty", d_cert, p_cert),
]:
    print(f"  {label:25s}  {d_val:+.3f}  {p_val:8.4f}{sig(p_val)}")

print("\n[e4: correlation between emotional and esteem support]")
sub_corr = df[["emotional", "esteem"]].dropna()
r_emo_est, p_emo_est = stats.pearsonr(sub_corr["emotional"], sub_corr["esteem"])
print(f"\n  Pearson r(N = {len(sub_corr)}) = {r_emo_est:+.3f}, p = {p_emo_est:.4f}{sig(p_emo_est)}")
print("\n[item-level results]")
item_defs = (
    [(c, l, "Emotional Support") for c, l in zip(EMOTIONAL_COLS, EMOTIONAL_LABELS)] +
    [(c, l, "Esteem Support") for c, l in zip(ESTEEM_COLS, ESTEEM_LABELS)] +
    [(c, l, "Informational Support") for c, l in zip(INFO_COLS, INFO_LABELS)] +
    [(c, l, "Certainty") for c, l in zip(CERTAINTY_COLS, CERTAINTY_LABELS)]
)

print(f"\n  {'Item':45s}  {'M_S':>5s}  {'M_N':>5s}  {'d':>6s}  {'p':>8s}")
print(f"  {'-' * 80}")
for col, label, stype in item_defs:
    s = df.loc[df["syco"] == 1, col].dropna()
    n = df.loc[df["syco"] == 0, col].dropna()
    n1, n2 = len(s), len(n)
    diff = s.mean() - n.mean()
    pooled = np.sqrt(((n1 - 1) * s.std() ** 2 + (n2 - 1) * n.std() ** 2) / (n1 + n2 - 2))
    d = diff / pooled if pooled > 0 else 0
    _, p = stats.ttest_ind(s, n, equal_var=False)
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    se_diff = np.sqrt(s.std() ** 2 / n1 + n.std() ** 2 / n2)
    print(f"  {label:45s}  {s.mean():5.2f}  {n.mean():5.2f}  {d:+.3f}  {p:8.4f}{sig(p)}")
    results_rows.append({
        "hypothesis": "Item-level", "outcome": f"{stype}: {label}",
        "mean_syco": round(s.mean(), 3), "sd_syco": round(s.std(), 3), "n_syco": n1,
        "mean_neutral": round(n.mean(), 3), "sd_neutral": round(n.std(), 3), "n_neutral": n2,
        "b": round(diff, 4), "se_b": round(se_diff, 4),
        "ci_lo_b": round(diff - 1.96 * se_diff, 4),
        "ci_hi_b": round(diff + 1.96 * se_diff, 4),
        "p_value": round(p, 6),
        "cohens_d": round(d, 3), "se_d": round(se_d, 3),
        "ci_lo_d": round(d - 1.96 * se_d, 3),
        "ci_hi_d": round(d + 1.96 * se_d, 3),
    })
out = pd.DataFrame(results_rows)
plot_rows = []
for _, r in out.iterrows():
    outcome = r["outcome"]
    hyp = r["hypothesis"]
    if hyp != "Item-level":
        item_type = "composite"
        support_type = outcome.replace(" (composite)", "")
        item_label = support_type
    else:
        item_type = "item"
        parts = outcome.split(": ", 1)
        support_type = parts[0]
        item_label = parts[1] if len(parts) > 1 else parts[0]
    plot_rows.append({
        "item_label": item_label, "support_type": support_type, "type": item_type,
        "hypothesis": hyp,
        "mean_syco": r["mean_syco"], "sd_syco": r["sd_syco"], "n_syco": int(r["n_syco"]),
        "mean_neutral": r["mean_neutral"], "sd_neutral": r["sd_neutral"], "n_neutral": int(r["n_neutral"]),
        "diff": round(r["mean_syco"] - r["mean_neutral"], 3),
        "b": r["b"], "se_b": r["se_b"], "ci_lo_b": r["ci_lo_b"], "ci_hi_b": r["ci_hi_b"],
        "p_value": r["p_value"],
        "cohens_d": r["cohens_d"], "se_d": r["se_d"],
        "ci_lo_d": r["ci_lo_d"], "ci_hi_d": r["ci_hi_d"],
    })
