"""Analysis of Study 4 mediation models for preference for reaching out to ai vs humans and social satisfaction."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

np.random.seed(42)
N_BOOT_CI   = 5000
N_BOOT_PVAL = 10000

LIKERT_COMFORT = {
    "Extremely": 7, "Very much": 6, "Quite a bit": 5,
    "Moderately": 4, "Somewhat": 3, "Slightly": 2, "Not at all": 1,
}
LIKERT_AGREE = {
    "Strongly disagree": 1, "Disagree": 2, "Somewhat disagree": 3,
    "Neither agree nor disagree": 4, "Somewhat agree": 5,
    "Agree": 6, "Strongly agree": 7,
}
AFFECT_MAP = {
    "Much worse than before": 1, "Somewhat worse than before": 2,
    "Slightly worse than before": 3, "About the same": 4,
    "Slightly better than before": 5, "Somewhat better than before": 6,
    "Much better than before": 7,
}
HELPFULNESS_MAP = {
    "Not at all helpful": 1, "Slightly helpful": 2, "Somewhat helpful": 3,
    "Moderately helpful": 4, "Quite helpful": 5, "Very helpful": 6,
    "Extremely helpful": 7,
}
CERTAINTY_MAP = {
    "Very uncertain": 1, "Somewhat uncertain": 2, "Uncertain": 3,
    "Neither certain nor uncertain": 4, "Certain": 5,
    "Somewhat certain": 6, "Very certain": 7,
}
SOCIAL_SAT_MAP = {
    "Very dissatisfied": 1, "Dissatisfied": 2, "Somewhat dissatisfied": 3,
    "Neither satisfied nor dissatisfied": 4, "Somewhat satisfied": 5,
    "Satisfied": 6, "Very satisfied": 7,
}

print("Loading data...", flush=True)
df = pd.read_csv("../data/main_sessions.csv")
df = df[df["Finished"].astype(str).str.strip().str.lower() == "true"].copy()
df = df.drop_duplicates(subset=["participant_id", "SESSION_NUM"], keep="first")

pid_condition = (
    df[["participant_id", "model_condition"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

weekly_df = df[df["SESSION_NUM"].isin(["four", "eight", "twelve"])].copy()

comfort_raw = weekly_df[["comfort_1", "comfort_2", "comfort_3"]].apply(
    lambda x: x.str.strip() if x.dtype == "object" else x
)
comfort_raw = comfort_raw.replace(LIKERT_COMFORT).apply(pd.to_numeric, errors="coerce")
weekly_df["comfort_humans"] = comfort_raw[["comfort_1", "comfort_2"]].mean(axis=1)
weekly_df["comfort_AI"] = comfort_raw["comfort_3"]
weekly_df["comfort_AI_minus_humans"] = weekly_df["comfort_AI"] - weekly_df["comfort_humans"]

for col in ["closeness-ai_1", "closeness-ai_2", "closeness-ai_3", "closeness-ai_4"]:
    weekly_df[col + "_num"] = weekly_df[col].map(LIKERT_AGREE)
weekly_df["feeling_known_ai"] = weekly_df[
    ["closeness-ai_1_num", "closeness-ai_2_num",
     "closeness-ai_3_num", "closeness-ai_4_num"]
].mean(axis=1)

for col in ["closeness-humans_1", "closeness-humans_2",
            "closeness-humans_3", "closeness-humans_4"]:
    weekly_df[col + "_num"] = weekly_df[col].map(LIKERT_AGREE)
weekly_df["feeling_known_humans"] = weekly_df[
    ["closeness-humans_1_num", "closeness-humans_2_num",
     "closeness-humans_3_num", "closeness-humans_4_num"]
].mean(axis=1)
weekly_df["feeling_known_gap"] = weekly_df["feeling_known_ai"] - weekly_df["feeling_known_humans"]

weekly_df["affect_num"]      = pd.to_numeric(weekly_df["affect"].replace(AFFECT_MAP), errors="coerce")
weekly_df["helpfulness_num"] = pd.to_numeric(weekly_df["helpfulness"].replace(HELPFULNESS_MAP), errors="coerce")
weekly_df["certainty_num"]   = pd.to_numeric(weekly_df["certainty"].replace(CERTAINTY_MAP), errors="coerce")
weekly_df["social_sat"]      = weekly_df["social-sat"].map(SOCIAL_SAT_MAP)

measures = [
    "comfort_AI_minus_humans", "feeling_known_ai", "feeling_known_humans",
    "feeling_known_gap", "affect_num", "helpfulness_num",
    "certainty_num", "social_sat",
]
person = weekly_df.groupby("participant_id")[measures].mean().reset_index()
person = person.merge(pid_condition, on="participant_id", how="left")
person = person[person["model_condition"].isin(["balanced", "syco", "challenging"])].copy()

def baron_kenny_with_ci(data, treatment_var, mediator_var, outcome_var,
                        contrast_label, mediator_label, n_boot=N_BOOT_CI):
    data = data.dropna(subset=[mediator_var, outcome_var, treatment_var]).copy()
    n = len(data)
    data["mediator_c"] = data[mediator_var] - data[mediator_var].mean()

    model_c   = smf.ols(f"{outcome_var} ~ {treatment_var}", data=data).fit()
    model_a   = smf.ols(f"{mediator_var} ~ {treatment_var}", data=data).fit()
    model_med = smf.ols(f"{outcome_var} ~ {treatment_var} + mediator_c", data=data).fit()

    c        = model_c.params[treatment_var]
    a        = model_a.params[treatment_var]
    b        = model_med.params["mediator_c"]
    c_prime  = model_med.params[treatment_var]
    indirect = a * b

    boot = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        bd = data.iloc[idx].copy()
        if bd[treatment_var].nunique() < 2:
            continue
        bd["mediator_c"] = bd[mediator_var] - bd[mediator_var].mean()
        try:
            a_b = smf.ols(f"{mediator_var} ~ {treatment_var}", data=bd).fit().params[treatment_var]
            b_b = smf.ols(f"{outcome_var} ~ {treatment_var} + mediator_c", data=bd).fit().params["mediator_c"]
            boot.append(a_b * b_b)
        except Exception:
            pass

    boot = np.array(boot)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    sig = "YES" if (ci_lo > 0 or ci_hi < 0) else "no"

    print(f"\n  {mediator_label}  (N={n}, contrast: {contrast_label})", flush=True)
    print(f"    c (total): {c:+.3f}   a: {a:+.3f}   b: {b:+.3f}   c': {c_prime:+.3f}", flush=True)
    print(f"    indirect = {indirect:+.4f}   95% boot CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  sig={sig}", flush=True)
    if abs(c) > 1e-3:
        print(f"    proportion mediated = {indirect / c:.1%}", flush=True)

def bootstrap_pvalue(data, treatment_var, mediator_var, outcome_var,
                     mediator_label, n_boot=N_BOOT_PVAL):
    data = data.dropna(subset=[mediator_var, outcome_var, treatment_var]).copy()
    n = len(data)
    boot = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        bd = data.iloc[idx]
        if bd[treatment_var].nunique() < 2:
            continue
        try:
            a = smf.ols(f"{mediator_var} ~ {treatment_var}", data=bd).fit().params[treatment_var]
            bd2 = bd.copy()
            bd2["med_c"] = bd2[mediator_var] - bd2[mediator_var].mean()
            b = smf.ols(f"{outcome_var} ~ {treatment_var} + med_c", data=bd2).fit().params["med_c"]
            boot.append(a * b)
        except Exception:
            pass

    boot = np.array(boot)
    point = boot.mean()
    ci95 = np.percentile(boot, [2.5, 97.5])
    ci99 = np.percentile(boot, [0.5, 99.5])
    p = 2 * np.mean(boot <= 0) if point > 0 else 2 * np.mean(boot >= 0)
    p = max(p, 1 / n_boot)
    stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."

    print(f"  {mediator_label}:  indirect={point:+.4f}  "
          f"95% CI [{ci95[0]:+.4f}, {ci95[1]:+.4f}]  "
          f"99% CI [{ci99[0]:+.4f}, {ci99[1]:+.4f}]  "
          f"bootstrap p={p:.4f}  {stars}", flush=True)

print("\n" + "=" * 70, flush=True)
print("1. comfort (ai - humans):  syco vs balanced  [5k boot cis]", flush=True)
print("=" * 70, flush=True)

df_bal = person[person["model_condition"].isin(["syco", "balanced"])].copy()
df_bal["syco"] = (df_bal["model_condition"] == "syco").astype(int)
print(f"\nN = {len(df_bal.dropna(subset=['comfort_AI_minus_humans']))}", flush=True)
print("Means by condition:", flush=True)
print(df_bal.groupby("model_condition")[
    ["comfort_AI_minus_humans", "feeling_known_ai",
     "affect_num", "helpfulness_num", "certainty_num"]
].mean().round(3), flush=True)

comfort_mediators = [
    ("feeling_known_ai", "Feeling Known by AI"),
    ("affect_num",       "Affect"),
    ("helpfulness_num",  "Helpfulness"),
    ("certainty_num",    "Certainty"),
]

for col, lab in comfort_mediators:
    sub = df_bal.dropna(subset=[col])
    baron_kenny_with_ci(sub, "syco", col, "comfort_AI_minus_humans",
                        "syco vs balanced", lab)

print("\n" + "=" * 70, flush=True)
print("2. comfort:  10k bootstrap for indirect-effect p-values", flush=True)
print("=" * 70, flush=True)

person_pval = person.dropna(subset=["comfort_AI_minus_humans"]).copy()
person_pval = person_pval[person_pval["model_condition"].isin(["syco", "balanced"])]
person_pval["treat"] = (person_pval["model_condition"] == "syco").astype(int)
print(f"\nN = {len(person_pval)}  (syco={int(person_pval['treat'].sum())}, "
      f"balanced={int((1 - person_pval['treat']).sum())})  "
      f"bootstrap iters = {N_BOOT_PVAL}\n", flush=True)

for col, lab in comfort_mediators:
    bootstrap_pvalue(person_pval, "treat", col, "comfort_AI_minus_humans", lab)

print("\n" + "=" * 70, flush=True)
print("3. social satisfaction  [5k boot cis]", flush=True)
print("=" * 70, flush=True)

person_ss = person.dropna(subset=["social_sat"]).copy()
print(f"\nN (any LLM condition) = {len(person_ss)}", flush=True)
social_mediators = [
    ("feeling_known_humans", "Feeling Known by Humans"),
    ("feeling_known_ai",     "Feeling Known by AI"),
    ("feeling_known_gap",    "Feeling Known Gap (AI - Humans)"),
    ("affect_num",           "Affect"),
    ("helpfulness_num",      "Helpfulness"),
    ("certainty_num",        "Certainty"),
]

for ref in ["balanced", "challenging"]:
    print(f"\n--- syco vs {ref} ---", flush=True)
    sub = person_ss[person_ss["model_condition"].isin(["syco", ref])].copy()
    sub["syco"] = (sub["model_condition"] == "syco").astype(int)
    for col, lab in social_mediators:
        baron_kenny_with_ci(sub.dropna(subset=[col]), "syco", col, "social_sat",
                            f"syco vs {ref}", lab)
