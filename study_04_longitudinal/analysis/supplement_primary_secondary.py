"""Analysis of Study 4 preregistered supplementary hypotheses (H5, H6, H7) plus the syco vs no-LLM contrast for H1 and the additional social-outcome models."""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

main     = pd.read_csv("../data/main_sessions.csv")
pre      = pd.read_csv("../data/pre_treatment.csv")
pid_cond = pd.read_csv("../data/pid_condition.csv")
nollm    = pd.read_csv("../data/no_llm_control.csv")

# Likert mappings
AGREE_7 = {
    "strongly disagree": 1, "disagree": 2, "somewhat disagree": 3,
    "neither agree nor disagree": 4, "somewhat agree": 5,
    "agree": 6, "strongly agree": 7,
}

AGREE_5 = {
    "strongly disagree": 1, "disagree": 2,
    "neither agree nor disagree": 3, "neither disagree nor agree": 3,
    "agree": 4, "strongly agree": 5,
}

AVERAGE_MAP = {
    "much worse than average": 1, "somewhat worse than average": 2,
    "slightly worse than average": 3, "average": 4,
    "slightly better than average": 5, "somewhat better than average": 6,
    "much better than average": 7,
}

OTHER_PEOPLE_MAP = {
    "much worse than them": 1, "somewhat worse than them": 2,
    "slightly worse than them": 3, "about the same": 4,
    "slightly better than them": 5, "somewhat better than them": 6,
    "much better than them": 7,
}

COMFORT_7 = {
    "not at all": 1, "slightly": 2, "somewhat": 3,
    "moderately": 4, "quite a bit": 5, "very much": 6, "extremely": 7,
}

LIKELIHOOD_MAP = {
    "very unlikely": 1, "unlikely": 2, "somewhat unlikely": 3,
    "neither likely nor unlikely": 4, "somewhat likely": 5,
    "likely": 6, "very likely": 7,
}

SOCIAL_SAT_MAP = {
    "very dissatisfied": 1, "dissatisfied": 2, "somewhat dissatisfied": 3,
    "neither satisfied nor dissatisfied": 4, "somewhat satisfied": 5,
    "satisfied": 6, "very satisfied": 7,
}

SOCIAL_TIME_MAP = {
    "0 days": 0, "1-2 days": 1.5, "3-4 days": 3.5,
    "5-6 days": 5.5, "every day (7 days)": 7,
}

# Helpers
def to_numeric(series, likert_map):
    return series.astype(str).str.strip().str.lower().map(likert_map).astype(float)

def composite(df, cols, likert_map):
    nums = pd.DataFrame({c: to_numeric(df[c], likert_map) for c in cols})
    return nums.mean(axis=1)

def holm_bonferroni(p_values):
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    for rank, (idx, p) in enumerate(indexed):
        adjusted[idx] = min(p * (n - rank), 1.0)
    for i in range(1, n):
        idx_cur  = indexed[i][0]
        idx_prev = indexed[i - 1][0]
        adjusted[idx_cur] = max(adjusted[idx_cur], adjusted[idx_prev])
    return adjusted

results = []

def run_ols(data, formula, condition_coef, analysis_label, outcome_label, n_label=None):
    """Fit OLS, extract condition effect, Cohen's d, CI, p."""
    data = data.dropna(subset=[condition_coef.split("[")[0].split(".")[-1] if "." in condition_coef else "model_condition"]).reset_index(drop=True)
    try:
        model = smf.ols(formula, data=data).fit()
    except Exception as e:
        print(f"  ⚠ OLS failed for {analysis_label} | {outcome_label}: {e}")
        return None

    if condition_coef not in model.params.index:
        for k in model.params.index:
            if "syco" in k.lower() or "T.syco" in k:
                condition_coef = k
                break

    b     = model.params[condition_coef]
    se    = model.bse[condition_coef]
    p     = model.pvalues[condition_coef]
    ci_lo = b - 1.96 * se
    ci_hi = b + 1.96 * se

    outcome_col = formula.split("~")[0].strip()
    _groups = [g[outcome_col].dropna() for _, g in data.groupby("model_condition")
               if g[outcome_col].dropna().size >= 2]
    if len(_groups) >= 2:
        _num = sum((g.size - 1) * g.var(ddof=1) for g in _groups)
        _den = sum(g.size for g in _groups) - len(_groups)
        pooled_sd = float(np.sqrt(_num / _den))
    else:
        pooled_sd = float(data[outcome_col].std(ddof=1))
    d     = b / pooled_sd
    d_lo  = ci_lo / pooled_sd
    d_hi  = ci_hi / pooled_sd
    n_obs = int(model.nobs)

    row = {
        "hypothesis": analysis_label,
        "outcome": outcome_label,
        "cohens_d": d, "ci_lower": d_lo, "ci_upper": d_hi,
        "p_value": p,
        "b": b, "b_ci_lower": ci_lo, "b_ci_upper": ci_hi,
        "n_obs": n_obs,
        "model_type": "OLS",
    }
    results.append(row)
    print(f"  {analysis_label} | {outcome_label}: d={d:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}], p={p:.4f}, n={n_obs}")
    return model

def run_mixed(data, formula, condition_coef, analysis_label, outcome_label):
    """Fit MixedLM (random intercept for participant), extract results."""
    data = data.dropna().reset_index(drop=True)
    data["participant_id"] = data["participant_id"].astype(str)
    try:
        model = smf.mixedlm(formula, data=data, groups=data["participant_id"],
                            re_formula="1").fit(reml=True)
    except Exception as e:
        print(f"  ⚠ MixedLM failed for {analysis_label} | {outcome_label}: {e}")
        return None

    if condition_coef not in model.fe_params.index:
        for k in model.fe_params.index:
            if "syco" in k.lower() or "T.syco" in k:
                condition_coef = k
                break

    b     = model.fe_params[condition_coef]
    se    = model.bse[condition_coef]
    p     = model.pvalues[condition_coef]
    ci_lo = b - 1.96 * se
    ci_hi = b + 1.96 * se

    outcome_col = formula.split("~")[0].strip()
    between_pid_sd = data.groupby("participant_id")[outcome_col].mean().std(ddof=1)
    d     = b / between_pid_sd
    d_lo  = ci_lo / between_pid_sd
    d_hi  = ci_hi / between_pid_sd
    n_obs = int(model.nobs)

    row = {
        "hypothesis": analysis_label,
        "outcome": outcome_label,
        "cohens_d": d, "ci_lower": d_lo, "ci_upper": d_hi,
        "p_value": p,
        "b": b, "b_ci_lower": ci_lo, "b_ci_upper": ci_hi,
        "n_obs": n_obs,
        "model_type": "MixedLM",
    }
    results.append(row)
    print(f"  {analysis_label} | {outcome_label}: d={d:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}], p={p:.4f}, n={n_obs}")
    return model

# Identify no-LLM participants
nollm_pids = set(nollm["participant_id"].dropna().unique())

session_num_map = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
if "session_numeric" not in main.columns:
    main["session_numeric"] = main["SESSION_NUM"].map(session_num_map)

weekly_sessions = ["four", "eight", "twelve"]
time_map_weekly = {"four": 1, "eight": 2, "twelve": 3}

# Build session-12 post-treatment for LLM conditions
s12 = main[main["SESSION_NUM"] == "twelve"].copy()
s12 = s12.merge(pid_cond, on="participant_id", how="left", suffixes=("", "_cond"))
if "model_condition_cond" in s12.columns:
    s12["model_condition"] = s12["model_condition_cond"].fillna(s12["model_condition"])

# Build no-LLM post-treatment
nollm["model_condition"] = "nollm"

# Assign condition in pre for no-LLM participants
pre["model_condition"] = pre.get("model_condition_y", pre.get("model_condition", np.nan))
for pid in nollm_pids:
    mask = pre["participant_id"] == pid
    if mask.any():
        pre.loc[mask, "model_condition"] = "nollm"

#  H5: INFLUENCE (donation compliance, post-only)
print("\n" + "=" * 70)
print("  h5: influence (donation compliance)")
print("=" * 70)

s12["donated"] = s12["Q2"].astype(str).str.strip().str.lower() == "yes"
s12["donation_amount"] = pd.to_numeric(s12["Q3_1"], errors="coerce").fillna(0)
s12["charity_selected"] = s12["Q4"].astype(str).str.strip().str.lower()
s12["ai_recommended"]   = s12["aiRecommendedCharity"].astype(str).str.strip().str.lower()
s12["aligned"] = s12["charity_selected"] == s12["ai_recommended"]
s12["compliance"] = 0.0
s12.loc[s12["donated"] & s12["aligned"], "compliance"] = s12.loc[s12["donated"] & s12["aligned"], "donation_amount"]

for ref, target, label in [
    ("balanced",    "syco", "H5: syco vs balanced (primary)"),
    ("challenging", "syco", "H5: syco vs challenging"),
]:
    sub = s12[s12["model_condition"].isin([ref, target])].copy()
    sub = sub.dropna(subset=["compliance"])
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, target])
    run_ols(sub, "compliance ~ model_condition", "model_condition[T.syco]", label, "Donation Compliance")

#  H6: AFFECTIVE WELL-BEING (pre-post ANCOVA)
print("\n" + "=" * 70)
print("  h6: affective well-being")
print("=" * 70)

wb_cols = [f"well-being_{i}" for i in range(1, 10)]
WB_REVERSE = {"well-being_7", "well-being_8", "well-being_9"}

def wb_composite(df, cols, likert_map, reverse_cols=WB_REVERSE):
    parts = {}
    for c in cols:
        s = to_numeric(df[c], likert_map)
        if c in reverse_cols:
            s = 6 - s
        parts[c] = s
    return pd.DataFrame(parts).mean(axis=1)

pre["WB_pre"] = wb_composite(pre, wb_cols, AGREE_5)

s12["WB_post"] = wb_composite(s12, wb_cols, AGREE_5)
nollm["WB_post"] = wb_composite(nollm, wb_cols, AGREE_5)

wb_post = pd.concat([
    s12[["participant_id", "WB_post", "model_condition"]],
    nollm[["participant_id", "WB_post", "model_condition"]],
], ignore_index=True)
wb = wb_post.merge(pre[["participant_id", "WB_pre"]], on="participant_id", how="left")
wb = wb.dropna(subset=["WB_post", "WB_pre", "model_condition"])

for ref, target, label in [
    ("challenging", "syco", "H6: syco vs challenging (primary)"),
    ("balanced",    "syco", "H6: syco vs balanced"),
    ("nollm",       "syco", "H6: syco vs no-LLM"),
]:
    sub = wb[wb["model_condition"].isin([ref, target])].copy()
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, target])
    run_ols(sub, "WB_post ~ WB_pre + model_condition", "model_condition[T.syco]", label, "Affective Well-being")

#  H7: SELF-FOCUSED ORIENTATION (pre-post ANCOVA)
print("\n" + "=" * 70)
print("  h7: self-focused orientation")
print("=" * 70)

# Pre uses 'relationship#1_*', post (session 12) uses 'relationships#1_*'
rel_pre_cols  = [f"relationship#1_{i}" for i in range(1, 9)]
rel_post_cols = [f"relationships#1_{i}" for i in range(1, 9)]
reverse_items = [1, 3, 5, 7]  # 0-indexed: items 2,4,6,8 are reverse-coded

def compute_relationship_score(df, cols, reverse_idx):
    nums = pd.DataFrame({c: to_numeric(df[c], AGREE_7) for c in cols})
    for i in reverse_idx:
        nums.iloc[:, i] = 8 - nums.iloc[:, i]
    return nums.mean(axis=1)

pre["SFO_pre"] = compute_relationship_score(pre, rel_pre_cols, reverse_items)
s12["SFO_post"] = compute_relationship_score(s12, rel_post_cols, reverse_items)
nollm["SFO_post"] = compute_relationship_score(nollm, rel_post_cols, reverse_items)

sfo_post = pd.concat([
    s12[["participant_id", "SFO_post", "model_condition"]],
    nollm[["participant_id", "SFO_post", "model_condition"]],
], ignore_index=True)
sfo = sfo_post.merge(pre[["participant_id", "SFO_pre"]], on="participant_id", how="left")
sfo = sfo.dropna(subset=["SFO_post", "SFO_pre", "model_condition"])

for ref, target, label in [
    ("challenging", "syco", "H7: syco vs challenging (primary)"),
    ("balanced",    "syco", "H7: syco vs balanced"),
    ("nollm",       "syco", "H7: syco vs no-LLM"),
]:
    sub = sfo[sfo["model_condition"].isin([ref, target])].copy()
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, target])
    run_ols(sub, "SFO_post ~ SFO_pre + model_condition", "model_condition[T.syco]", label, "Self-Focused Orientation")

#  H1 ADDITIONAL: syco vs no-LLM (ANCOVA on comfort)
print("\n" + "=" * 70)
print("  h1 additional: preference for ai - syco vs no-llm")
print("=" * 70)

def add_comfort(frame, cols=("comfort_1", "comfort_2", "comfort_3")):
    c = pd.DataFrame({col: to_numeric(frame[col], COMFORT_7) for col in cols})
    frame = frame.copy()
    frame["comfort_humans"] = c[[cols[0], cols[1]]].mean(axis=1)
    frame["comfort_AI"]     = c[cols[2]]
    frame["comfort_AI_minus_humans"] = frame["comfort_AI"] - frame["comfort_humans"]
    return frame

pre_comfort = add_comfort(pre)
s12_comfort = add_comfort(s12)
nollm_comfort = add_comfort(nollm)

# Use session 12 for syco, no_llm_control for no-LLM
comfort_syco  = s12_comfort[s12_comfort["model_condition"] == "syco"][["participant_id", "comfort_AI_minus_humans"]].copy()
comfort_syco.columns = ["participant_id", "comfort_post"]
comfort_syco["model_condition"] = "syco"

comfort_nollm = nollm_comfort[["participant_id", "comfort_AI_minus_humans"]].copy()
comfort_nollm.columns = ["participant_id", "comfort_post"]
comfort_nollm["model_condition"] = "nollm"

comfort_h1 = pd.concat([comfort_syco, comfort_nollm], ignore_index=True)
comfort_h1 = comfort_h1.merge(
    pre_comfort[["participant_id", "comfort_AI_minus_humans"]].rename(columns={"comfort_AI_minus_humans": "comfort_pre"}),
    on="participant_id", how="left"
)
comfort_h1 = comfort_h1.dropna(subset=["comfort_post", "comfort_pre", "model_condition"])
comfort_h1["model_condition"] = pd.Categorical(comfort_h1["model_condition"], categories=["nollm", "syco"])
run_ols(comfort_h1, "comfort_post ~ comfort_pre + model_condition", "model_condition[T.syco]",
        "H1: syco vs no-LLM", "Preference AI minus Humans (ANCOVA)")

#  ADDITIONAL: Social satisfaction & time vs challenging
print("\n" + "=" * 70)
print("  additional: social outcomes – syco vs challenging")
print("=" * 70)

soc_weekly = main[main["SESSION_NUM"].isin(weekly_sessions)].copy()
soc_weekly["social_time_num"] = to_numeric(soc_weekly["social-time"], SOCIAL_TIME_MAP)
soc_weekly["social_sat_num"]  = to_numeric(soc_weekly["social-sat"], SOCIAL_SAT_MAP)

social_data = soc_weekly[["participant_id", "SESSION_NUM", "session_numeric",
                           "social_time_num", "social_sat_num"]].copy()
social_data = social_data.merge(pid_cond, on="participant_id", how="left")
social_data["time_num"] = social_data["SESSION_NUM"].map(time_map_weekly)

for var, vlabel in [("social_time_num", "Social Time"), ("social_sat_num", "Social Satisfaction")]:
    sub = social_data[social_data["model_condition"].isin(["challenging", "syco"])].copy()
    sub = sub.dropna(subset=[var, "time_num", "model_condition", "participant_id"])
    sub["time_c"] = sub["time_num"] - sub["time_num"].mean()
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=["challenging", "syco"])
    run_mixed(sub, f"{var} ~ time_c * model_condition", "model_condition[T.syco]",
              "Additional: syco vs challenging", vlabel)

#  EXPORT
results_df = pd.DataFrame(results)
col_order = ["hypothesis", "outcome", "cohens_d", "ci_lower", "ci_upper",
             "p_value", "b", "b_ci_lower", "b_ci_upper", "n_obs", "model_type"]
for c in col_order:
    if c not in results_df.columns:
        results_df[c] = np.nan
results_df = results_df[col_order]

# Print Holm-Bonferroni corrected p-values per hypothesis
print("\n" + "=" * 70)
print("  holm-bonferroni corrected p-values (within each hypothesis)")
print("=" * 70)

for hyp_prefix in ["H1:", "H5:", "H6:", "H7:"]:
    hyp_rows = results_df[results_df["hypothesis"].str.startswith(hyp_prefix)]
    if len(hyp_rows) == 0:
        continue
    unique_outcomes = hyp_rows["outcome"].unique()
    for outcome in unique_outcomes:
        sub = hyp_rows[hyp_rows["outcome"] == outcome]
        if len(sub) < 2:
            continue
        raw_ps = sub["p_value"].tolist()
        adj_ps = holm_bonferroni(raw_ps)
        print(f"\n{hyp_prefix} {outcome}")
        for i, (_, row) in enumerate(sub.iterrows()):
            print(f"  {row['hypothesis']:45s}  raw p={raw_ps[i]:.4f}  adj p={adj_ps[i]:.4f}")
