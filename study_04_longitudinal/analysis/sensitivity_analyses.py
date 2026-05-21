"""Analysis of Study 4 sensitivity tests for the headline H1 outcome (AI vs human preference)."""

import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

main     = pd.read_csv("../data/main_sessions.csv")
pre      = pd.read_csv("../data/pre_treatment.csv")
pid_cond = pd.read_csv("../data/pid_condition.csv")

COMFORT_7 = {
    "not at all": 1, "slightly": 2, "somewhat": 3,
    "moderately": 4, "quite a bit": 5, "very much": 6, "extremely": 7,
}

AGE_MID = {
    "18-24": 21, "25-35": 30, "35-44": 40, "45-54": 50,
    "55-64": 60, "65-74": 70, "75+": 80,
}
AI_USE_NUM = {
    "Never": 1, "Rarely": 2, "Monthly": 3, "Weekly": 4, "Daily": 5,
}

session_num_map = {
    "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,
    "seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,
}
weekly_sessions = ["four", "eight", "twelve"]
time_map_weekly = {"four": 1, "eight": 2, "twelve": 3}

def to_numeric(series, m):
    return series.astype(str).str.strip().str.lower().map(m).astype(float)

def add_comfort(frame, cols=("comfort_1","comfort_2","comfort_3")):
    c = pd.DataFrame({col: to_numeric(frame[col], COMFORT_7) for col in cols})
    f = frame.copy()
    f["comfort_humans"] = c[[cols[0], cols[1]]].mean(axis=1)
    f["comfort_AI"]     = c[cols[2]]
    f["comfort_AI_minus_humans"] = f["comfort_AI"] - f["comfort_humans"]
    return f

# Build weekly outcome dataframe
weekly = main[main["SESSION_NUM"].isin(weekly_sessions)].copy()
weekly = add_comfort(weekly)
weekly["time_num"]  = weekly["SESSION_NUM"].map(time_map_weekly)
# main_sessions already has model_condition
if "model_condition" not in weekly.columns:
    weekly = weekly.merge(pid_cond, on="participant_id", how="left")
weekly = weekly.dropna(subset=["comfort_AI_minus_humans", "time_num", "model_condition"])

# Build strict-completer sample (>=1 obs at all 3 weekly sessions)
sessions_per_pid = weekly.groupby("participant_id")["time_num"].nunique()
strict_pids = set(sessions_per_pid[sessions_per_pid == 3].index)

# Add baseline covariates from pre-treatment
pre = add_comfort(pre)
pre["age_mid"]    = pre["age"].map(AGE_MID)
pre["female"]     = (pre["gender"].str.lower() == "woman").astype(int)
pre["ai_use_num"] = pre["ai-use"].map(AI_USE_NUM)
pre_cov = pre[["participant_id", "comfort_AI_minus_humans",
               "age_mid", "female", "ai_use_num"]].rename(
    columns={"comfort_AI_minus_humans": "comfort_pre"}
)
pre_cov = pre_cov.drop_duplicates(subset="participant_id")

def fit(df, formula, label):
    df["participant_id"] = df["participant_id"].astype(str)
    m = smf.mixedlm(formula, data=df, groups=df["participant_id"]).fit(reml=True)
    coef_name = next(k for k in m.fe_params.index if "T.syco" in k)
    b = m.fe_params[coef_name]; se = m.bse[coef_name]; p = m.pvalues[coef_name]
    sd = np.sqrt(m.scale)
    return {
        "label": label,
        "n_obs": int(m.nobs),
        "n_pids": df["participant_id"].nunique(),
        "b": b, "se": se, "p": p,
        "ci_lo": b - 1.96*se, "ci_hi": b + 1.96*se,
        "cohens_d": b / sd,
        "d_lo": (b - 1.96*se) / sd, "d_hi": (b + 1.96*se) / sd,
    }

def run_contrast(ref, target, contrast_name):
    print(f"\n{'='*70}\n  {contrast_name}\n{'='*70}")

    # 1. Pre-registered (current main-text spec)
    sub = weekly[weekly.model_condition.isin([ref, target])].copy()
    sub["time_c"] = sub["time_num"] - sub["time_num"].mean()
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, target])
    sub = sub.merge(pre_cov[["participant_id", "comfort_pre"]], on="participant_id", how="left")
    pre_reg = fit(sub.dropna(subset=["comfort_pre"]),
                  "comfort_AI_minus_humans ~ time_c * model_condition + comfort_pre",
                  f"{contrast_name} | Pre-registered (mixed, FIML)")
    yield pre_reg

    # 2. Complete cases (all 12 sessions / all 3 weekly observations)
    cc = sub[sub.participant_id.isin(strict_pids) & sub.comfort_pre.notna()].copy()
    cc_res = fit(cc,
                 "comfort_AI_minus_humans ~ time_c * model_condition + comfort_pre",
                 f"{contrast_name} | Complete cases (all 3 weekly obs)")
    yield cc_res

    # 3. No exclusions: include rows where comfort_pre is missing too
    ne = sub.copy()
    ne["comfort_pre"] = ne["comfort_pre"].fillna(ne["comfort_pre"].mean())
    ne_res = fit(ne,
                 "comfort_AI_minus_humans ~ time_c * model_condition + comfort_pre",
                 f"{contrast_name} | No exclusions (mean-impute baseline)")
    yield ne_res

    # 4. With baseline demographics
    cov = sub.merge(
        pre_cov[["participant_id", "age_mid", "female", "ai_use_num"]],
        on="participant_id", how="left"
    ).dropna(subset=["age_mid", "female", "ai_use_num", "comfort_pre"])
    cov_res = fit(cov,
                  "comfort_AI_minus_humans ~ time_c * model_condition + comfort_pre + "
                  "age_mid + female + ai_use_num",
                  f"{contrast_name} | + age, gender, AI-use covariates")
    yield cov_res

rows = []
for ref, target, name in [
    ("balanced",    "syco", "Sycophantic vs. Neutral (primary H1)"),
    ("challenging", "syco", "Sycophantic vs. Challenging"),
]:
    for r in run_contrast(ref, target, name):
        rows.append(r)
        print(f"  {r['label']}: d = {r['cohens_d']:+.3f} "
              f"[{r['d_lo']:+.3f}, {r['d_hi']:+.3f}], "
              f"p = {r['p']:.4f}, n_obs = {r['n_obs']}, n_pids = {r['n_pids']}")

df_out = pd.DataFrame(rows)
