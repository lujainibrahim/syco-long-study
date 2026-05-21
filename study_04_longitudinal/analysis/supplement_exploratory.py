"""Analysis of Study 4 exploratory supplement (E1, E3, E4–E8)."""

import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

main     = pd.read_csv("../data/main_sessions.csv")
pre      = pd.read_csv("../data/pre_treatment.csv")
pid_cond = pd.read_csv("../data/pid_condition.csv")
nollm    = pd.read_csv("../data/no_llm_control.csv")

AGREE_7 = {
    "strongly disagree": 1, "disagree": 2, "somewhat disagree": 3,
    "neither agree nor disagree": 4, "somewhat agree": 5,
    "agree": 6, "strongly agree": 7,
}
AGREE_5 = {
    "strongly disagree": 1, "disagree a little": 2,
    "neither agree nor disagree": 3, "agree a little": 4,
    "strongly agree": 5,
}
LIKERT_7_COMFORT = {
    "extremely": 7, "very much": 6, "quite a bit": 5,
    "moderately": 4, "somewhat": 3, "slightly": 2, "not at all": 1,
}
CERTAINTY_MAP = {
    "very uncertain": 1, "somewhat uncertain": 2, "uncertain": 3,
    "neither certain nor uncertain": 4, "certain": 5,
    "somewhat certain": 6, "very certain": 7,
}
AFFECT_MAP = {
    "much worse than before": 1, "somewhat worse than before": 2,
    "slightly worse than before": 3, "about the same": 4,
    "slightly better than before": 5, "somewhat better than before": 6,
    "much better than before": 7,
}
HELPFULNESS_MAP = {
    "not at all helpful": 1, "slightly helpful": 2, "somewhat helpful": 3,
    "moderately helpful": 4, "quite helpful": 5, "very helpful": 6,
    "extremely helpful": 7,
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
OVERCLAIM_MAP = {
    "never heard of it": 0, "slightly familiar": 1,
    "somewhat familiar": 2, "familiar": 3, "very familiar": 4,
}
LSNS_MAP = {"0": 0, "1": 1, "2": 2, "3 or 4": 3, "5-8": 4, "9+": 5}

SESSION_NUM_MAP = {
    "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
WEEKLY_SESSIONS = ["four", "eight", "twelve"]
TIME_MAP_WEEKLY = {"four": 1, "eight": 2, "twelve": 3}

def to_numeric(series, mapping):
    return series.astype(str).str.strip().str.lower().map(mapping).astype(float)

def formula_vars(formula):
    tokens = re.split(r"[~+*:\s]+", formula)
    return [t.strip() for t in tokens if t.strip() and t.strip() != "1"]

def run_ols(data, formula, coef, analysis, outcome, results):
    data = data.dropna(subset=[c for c in formula_vars(formula) if c in data.columns]).reset_index(drop=True)
    try:
        model = smf.ols(formula, data=data).fit()
    except Exception as e:
        print(f"  OLS failed for {analysis} | {outcome}: {e}", flush=True)
        return None

    if coef not in model.params.index:
        for k in model.params.index:
            if "syco" in k.lower():
                coef = k
                break

    b, se, p = model.params[coef], model.bse[coef], model.pvalues[coef]
    ci_lo, ci_hi = b - 1.96 * se, b + 1.96 * se
    sd = np.sqrt(model.mse_resid)
    d, d_lo, d_hi = b / sd, ci_lo / sd, ci_hi / sd
    n_obs = int(model.nobs)

    results.append({
        "analysis": analysis, "outcome": outcome,
        "cohens_d": d, "ci_lower": d_lo, "ci_upper": d_hi,
        "p_value": p, "b": b, "b_ci_lower": ci_lo, "b_ci_upper": ci_hi,
        "n_obs": n_obs, "model_type": "OLS",
    })
    print(f"  {analysis} | {outcome}: d={d:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}], "
          f"p={p:.4f}, n={n_obs}", flush=True)
    return model

def run_mixed(data, formula, coef_of_interest, analysis, outcome, results):
    cols_keep = [c for c in formula_vars(formula) if c in data.columns] + ["participant_id"]
    data = data.dropna(subset=cols_keep).reset_index(drop=True)
    data["participant_id"] = data["participant_id"].astype(str)

    try:
        model = smf.mixedlm(formula, data=data, groups=data["participant_id"],
                            re_formula="1").fit(reml=True)
    except Exception as e:
        print(f"  MixedLM failed for {analysis} | {outcome}: {e}", flush=True)
        return None

    found = None
    for k in model.fe_params.index:
        if coef_of_interest in k:
            found = k
            break
    if found is None:
        for k in model.fe_params.index:
            if "syco" in k.lower():
                found = k
                break
    if found is None:
        return model

    b, se, p = model.fe_params[found], model.bse[found], model.pvalues[found]
    ci_lo, ci_hi = b - 1.96 * se, b + 1.96 * se
    scale = np.sqrt(model.scale)
    d, d_lo, d_hi = b / scale, ci_lo / scale, ci_hi / scale

    results.append({
        "analysis": analysis, "outcome": outcome,
        "cohens_d": d, "ci_lower": d_lo, "ci_upper": d_hi,
        "p_value": p, "b": b, "b_ci_lower": ci_lo, "b_ci_upper": ci_hi,
        "n_obs": int(model.nobs), "model_type": "MixedLM",
        "coefficient": found,
    })
    print(f"  {analysis} | {outcome}: b={b:+.4f}, d={d:+.3f} "
          f"[{d_lo:+.3f}, {d_hi:+.3f}], p={p:.4f}, n={int(model.nobs)}", flush=True)
    return model

results = []

if "session_numeric" not in main.columns:
    main["session_numeric"] = main["SESSION_NUM"].map(SESSION_NUM_MAP)

lsns_cols = [f"lsns-{i}" for i in range(1, 7)]
for col in lsns_cols:
    pre[col + "_num"] = to_numeric(pre[col], LSNS_MAP)
pre["lsns_total"] = pre[[c + "_num" for c in lsns_cols]].sum(axis=1)

pre["big5_2_num"] = to_numeric(pre["big5_2"], AGREE_5)
pre["big5_7_num"] = to_numeric(pre["big5_7"], AGREE_5)
pre["big5_7_rev"] = 6 - pre["big5_7_num"]
pre["agreeableness"] = pre[["big5_2_num", "big5_7_rev"]].mean(axis=1)

cred_cols = ["objectivity_1", "objectivity_2", "objectivity_3"]
for col in cred_cols:
    pre[col + "_num"] = to_numeric(pre[col], AGREE_7)
pre["ai_credibility"] = pre[[c + "_num" for c in cred_cols]].mean(axis=1)

for mod in ["lsns_total", "agreeableness", "ai_credibility"]:
    pre[mod + "_c"] = pre[mod] - pre[mod].mean()

moderator_df = pre[[
    "participant_id",
    "lsns_total", "lsns_total_c",
    "agreeableness", "agreeableness_c",
    "ai_credibility", "ai_credibility_c",
]].copy().drop_duplicates(subset=["participant_id"])

weekly_df = main[main["SESSION_NUM"].isin(WEEKLY_SESSIONS)].copy()
for col in ["comfort_1", "comfort_2", "comfort_3"]:
    weekly_df[col + "_num"] = to_numeric(weekly_df[col], LIKERT_7_COMFORT)
weekly_df["comfort_humans"] = weekly_df[["comfort_1_num", "comfort_2_num"]].mean(axis=1)
weekly_df["comfort_AI"] = weekly_df["comfort_3_num"]
weekly_df["comfort_AI_minus_humans"] = weekly_df["comfort_AI"] - weekly_df["comfort_humans"]
weekly_df["time_num"] = weekly_df["SESSION_NUM"].map(TIME_MAP_WEEKLY)

fk_human_cols = ["closeness-humans_1", "closeness-humans_2",
                 "closeness-humans_3", "closeness-humans_4"]
for col in fk_human_cols:
    weekly_df[col + "_num"] = to_numeric(weekly_df[col], AGREE_7)
weekly_df["fk_humans"] = weekly_df[[c + "_num" for c in fk_human_cols]].mean(axis=1)

comfort_data = weekly_df[["participant_id", "SESSION_NUM", "time_num",
                          "comfort_AI_minus_humans"]].copy()
comfort_data = comfort_data.merge(pid_cond, on="participant_id", how="left")
comfort_data = comfort_data.merge(moderator_df, on="participant_id", how="left")
comfort_data = comfort_data[comfort_data["model_condition"].isin(
    ["balanced", "syco", "challenging"])].copy()
comfort_data["time_c"] = comfort_data["time_num"] - comfort_data["time_num"].mean()

fk_data = weekly_df[["participant_id", "SESSION_NUM", "time_num", "fk_humans"]].copy()
fk_data = fk_data.merge(pid_cond, on="participant_id", how="left")
fk_data = fk_data.merge(moderator_df, on="participant_id", how="left")
fk_data = fk_data[fk_data["model_condition"].isin(["balanced", "syco", "challenging"])].copy()
fk_data["time_c"] = fk_data["time_num"] - fk_data["time_num"].mean()

print("Moderator descriptives:")
for mod in ["lsns_total", "agreeableness", "ai_credibility"]:
    v = moderator_df[mod].dropna()
    print(f"  {mod}: M={v.mean():.2f}, SD={v.std():.2f}, n={len(v)}", flush=True)

print("\n" + "=" * 70)
print("  e1a: social network strength (lsns-6) moderator")
print("=" * 70, flush=True)
for ref, suf in [("balanced", "syco vs balanced"), ("challenging", "syco vs challenging")]:
    sub = comfort_data[comfort_data["model_condition"].isin([ref, "syco"])].copy()
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, "syco"])
    print(f"\n--- H1 (AI Preference): LSNS x {suf} ---", flush=True)
    run_mixed(sub,
              "comfort_AI_minus_humans ~ time_c + model_condition * lsns_total_c",
              "model_condition[T.syco]:lsns_total_c",
              f"E1a: LSNS x condition ({suf})",
              "AI Preference (comfort AI - humans)",
              results)

    sub_fk = fk_data[fk_data["model_condition"].isin([ref, "syco"])].copy()
    sub_fk["model_condition"] = pd.Categorical(sub_fk["model_condition"], categories=[ref, "syco"])
    print(f"\n--- H2 (FK Humans): LSNS x {suf} ---", flush=True)
    run_mixed(sub_fk,
              "fk_humans ~ time_c + model_condition * lsns_total_c",
              "model_condition[T.syco]:lsns_total_c",
              f"E1a: LSNS x condition ({suf})",
              "Feeling Known by Humans",
              results)

print("\n" + "=" * 70)
print("  e1b: agreeableness (bfi) moderator")
print("=" * 70, flush=True)
for ref, suf in [("balanced", "syco vs balanced"), ("challenging", "syco vs challenging")]:
    sub = comfort_data[comfort_data["model_condition"].isin([ref, "syco"])].copy()
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, "syco"])
    print(f"\n--- H1 (AI Preference): Agreeableness x {suf} ---", flush=True)
    run_mixed(sub,
              "comfort_AI_minus_humans ~ time_c + model_condition * agreeableness_c",
              "model_condition[T.syco]:agreeableness_c",
              f"E1b: Agreeableness x condition ({suf})",
              "AI Preference (comfort AI - humans)",
              results)

print("\n" + "=" * 70)
print("  e1c: ai credibility moderator")
print("=" * 70, flush=True)
for ref, suf in [("balanced", "syco vs balanced"), ("challenging", "syco vs challenging")]:
    sub = comfort_data[comfort_data["model_condition"].isin([ref, "syco"])].copy()
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, "syco"])
    print(f"\n--- H1 (AI Preference): AI Credibility x {suf} ---", flush=True)
    run_mixed(sub,
              "comfort_AI_minus_humans ~ time_c + model_condition * ai_credibility_c",
              "model_condition[T.syco]:ai_credibility_c",
              f"E1c: AI Credibility x condition ({suf})",
              "AI Preference (comfort AI - humans)",
              results)

    sub_fk = fk_data[fk_data["model_condition"].isin([ref, "syco"])].copy()
    sub_fk["model_condition"] = pd.Categorical(sub_fk["model_condition"], categories=[ref, "syco"])
    print(f"\n--- H2 (FK Humans): AI Credibility x {suf} ---", flush=True)
    run_mixed(sub_fk,
              "fk_humans ~ time_c + model_condition * ai_credibility_c",
              "model_condition[T.syco]:ai_credibility_c",
              f"E1c: AI Credibility x condition ({suf})",
              "Feeling Known by Humans",
              results)

main["affect_num"]      = to_numeric(main["affect"], AFFECT_MAP)
main["helpfulness_num"] = to_numeric(main["helpfulness"], HELPFULNESS_MAP)
main["certainty_num"]   = to_numeric(main["certainty"], CERTAINTY_MAP)
main["intentions_num"]  = to_numeric(main["intentions"], LIKELIHOOD_MAP)

fk_ai_cols = ["closeness-ai_1", "closeness-ai_2", "closeness-ai_3", "closeness-ai_4"]
for col in fk_ai_cols:
    main[col + "_num"] = to_numeric(main[col], AGREE_7)
main["fk_ai"] = main[[c + "_num" for c in fk_ai_cols]].mean(axis=1)

df_llm = main[main["model_condition"].isin(["balanced", "syco", "challenging"])].copy()
df_llm["time_c"] = df_llm["session_numeric"] - df_llm["session_numeric"].mean()

session_measures = [
    ("affect_num",      "Affective Shift"),
    ("helpfulness_num", "Helpfulness"),
    ("certainty_num",   "Certainty"),
    ("fk_ai",           "Feeling Known by AI"),
    ("intentions_num",  "Intention to Act"),
]

print("\n" + "=" * 70)
print("  e3: session-level trajectories (condition x time)")
print("=" * 70, flush=True)

for ref, ref_label in [("balanced", "syco vs balanced"), ("challenging", "syco vs challenging")]:
    print(f"\n  Reference = {ref}", flush=True)

    for var, vlabel in session_measures:
        sub = df_llm[df_llm["model_condition"].isin([ref, "syco"])].copy()
        sub = sub.dropna(subset=[var, "time_c", "model_condition", "participant_id"]).reset_index(drop=True)
        sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, "syco"])
        sub["participant_id"] = sub["participant_id"].astype(str)

        try:
            model = smf.mixedlm(f"{var} ~ time_c * model_condition", data=sub,
                                groups=sub["participant_id"], re_formula="1").fit(reml=True)
        except Exception as e:
            print(f"  failed for {vlabel}: {e}", flush=True)
            continue

        scale = np.sqrt(model.scale)
        for coef_key, role in [
            ("model_condition[T.syco]",         "main effect"),
            ("time_c:model_condition[T.syco]",  "condition x time"),
            ("time_c",                          "time main effect"),
        ]:
            if coef_key not in model.fe_params.index:
                continue
            b  = model.fe_params[coef_key]
            se = model.bse[coef_key]
            p  = model.pvalues[coef_key]
            d  = b / scale
            results.append({
                "analysis": f"E3: {ref_label} ({role})",
                "outcome": vlabel, "coefficient": coef_key,
                "cohens_d": d,
                "ci_lower": (b - 1.96 * se) / scale,
                "ci_upper": (b + 1.96 * se) / scale,
                "p_value": p, "b": b,
                "b_ci_lower": b - 1.96 * se, "b_ci_upper": b + 1.96 * se,
                "n_obs": int(model.nobs), "model_type": "MixedLM",
            })
            if role != "time main effect":
                print(f"    {vlabel} [{ref_label}] {role}: d={d:+.3f}, p={p:.4f}", flush=True)

print("\nSession-level descriptive means:", flush=True)
for var, vlabel in session_measures:
    means = df_llm.groupby(["model_condition", "session_numeric"])[var].mean().unstack()
    print(f"\n{vlabel}:")
    print(means.round(3).to_string())

print("\n" + "=" * 70)
print("  e4: cognitive reflection test")
print("=" * 70, flush=True)

def score_crt_1(v):
    s = str(v).strip().lower()
    return 1 if s in ("mary", "mary.") else 0

def score_crt_2(v):
    s = str(v).strip().lower()
    pats = ["no stair", "there are no stair", "there were no stair",
            "none", "zero", "0", "no stairs", "there aren't",
            "one story", "one-story", "doesn't have stairs",
            "don't have stairs"]
    return 1 if any(p in s for p in pats) else 0

def score_crt_3(v):
    s = str(v).strip().lower()
    return 1 if s in ("2", "two", "2.") else 0

def score_crt_4(v):
    s = str(v).strip().lower()
    pats = ["no smoke", "none", "nowhere", "doesn't blow",
            "electric", "there is no smoke", "no direction",
            "it doesn't", "trick"]
    return 1 if any(p in s for p in pats) else 0

def score_crt_5(v):
    s = str(v).strip().lower()
    if s in ("b", "(b)", "a", "(a)"):
        return 0
    if "yolk of the egg is white" in s or "yolk of an egg" in s:
        return 0
    pats = ["neither", "yellow", "not white",
            "trick", "can't be white", "none of"]
    return 1 if any(p in s for p in pats) else 0

def compute_crt(df):
    df = df.copy()
    df["crt1"] = df["crt-1"].apply(score_crt_1)
    df["crt2"] = df["crt-2"].apply(score_crt_2)
    df["crt3"] = df["crt-3"].apply(score_crt_3)
    df["crt4"] = df["crt-4"].apply(score_crt_4)
    df["crt5"] = df["crt-5"].apply(score_crt_5)
    df["crt_total"] = df[[f"crt{i}" for i in range(1, 6)]].sum(axis=1)
    return df

s12 = main[main["SESSION_NUM"] == "twelve"].copy()
s12 = s12.merge(pid_cond, on="participant_id", how="left", suffixes=("", "_cond"))
if "model_condition_cond" in s12.columns:
    s12["model_condition"] = s12["model_condition_cond"].fillna(s12["model_condition"])

nollm["model_condition"] = "nollm"

s12_crt = compute_crt(s12[s12["model_condition"].isin(["balanced", "syco", "challenging"])])
nollm_crt = compute_crt(nollm)
crt_cols = ["participant_id", "crt_total", "model_condition"] + [f"crt{i}" for i in range(1, 6)]
crt_all = pd.concat([s12_crt[crt_cols], nollm_crt[crt_cols]], ignore_index=True)

print("\nCRT descriptives by condition:")
print(crt_all.groupby("model_condition")["crt_total"].agg(["mean", "std", "count"]))
print("\nItem-level accuracy:")
for i in range(1, 6):
    print(f"  CRT-{i}: {crt_all[f'crt{i}'].mean():.2%} correct overall")

for ref, target, label in [
    ("balanced",    "syco", "E4: syco vs balanced"),
    ("challenging", "syco", "E4: syco vs challenging"),
    ("nollm",       "syco", "E4: syco vs no-LLM"),
]:
    sub = crt_all[crt_all["model_condition"].isin([ref, target])].copy()
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, target])
    run_ols(sub, "crt_total ~ model_condition", "model_condition[T.syco]",
            label, "CRT Total", results)

print("\n" + "=" * 70)
print("  e5: social time & satisfaction (condition x time)")
print("=" * 70, flush=True)

soc_weekly = main[main["SESSION_NUM"].isin(WEEKLY_SESSIONS)].copy()
soc_weekly["social_time_num"] = to_numeric(soc_weekly["social-time"], SOCIAL_TIME_MAP)
soc_weekly["social_sat_num"]  = to_numeric(soc_weekly["social-sat"],  SOCIAL_SAT_MAP)

social_data = soc_weekly[["participant_id", "SESSION_NUM", "session_numeric",
                          "social_time_num", "social_sat_num"]].copy()
social_data = social_data.merge(pid_cond, on="participant_id", how="left")
social_data["time_num"] = social_data["SESSION_NUM"].map(TIME_MAP_WEEKLY)

sub = social_data[social_data["model_condition"].isin(["balanced", "syco", "challenging"])].copy()
sub = sub.dropna(subset=["social_time_num", "social_sat_num", "time_num",
                         "model_condition", "participant_id"])
sub["time_c"] = sub["time_num"] - sub["time_num"].mean()
sub["model_condition"] = pd.Categorical(sub["model_condition"],
                                        categories=["balanced", "syco", "challenging"])

print("\nCondition x Time interactions (3-condition model, balanced=ref):")
for var, vlabel in [("social_time_num", "Social Time"),
                    ("social_sat_num",  "Social Satisfaction")]:
    print(f"\n--- {vlabel} ---")
    data_sub = sub.dropna(subset=[var]).copy()
    try:
        model = smf.mixedlm(f"{var} ~ time_c * model_condition", data=data_sub,
                            groups=data_sub["participant_id"], re_formula="1").fit(reml=True)
        print(model.summary().tables[1])
        for coef in model.fe_params.index:
            if "time_c:model_condition" in coef:
                b, se, p = model.fe_params[coef], model.bse[coef], model.pvalues[coef]
                scale = np.sqrt(model.scale)
                d = b / scale
                results.append({
                    "analysis": "E5: condition x time interaction",
                    "outcome": f"{vlabel} ({coef})",
                    "cohens_d": d,
                    "ci_lower": (b - 1.96 * se) / scale,
                    "ci_upper": (b + 1.96 * se) / scale,
                    "p_value": p, "b": b,
                    "b_ci_lower": b - 1.96 * se, "b_ci_upper": b + 1.96 * se,
                    "n_obs": int(model.nobs), "model_type": "MixedLM",
                })
                print(f"  {coef}: d={d:+.3f}, p={p:.4f}")
    except Exception as e:
        print(f"  failed: {e}", flush=True)

print("\n" + "=" * 70)
print("  e6: intention to act")
print("=" * 70, flush=True)

df_llm_intent = df_llm.dropna(subset=["intentions_num"]).copy()
person_intent = (df_llm_intent.groupby(["participant_id", "model_condition"])["intentions_num"]
                 .mean().reset_index()
                 .rename(columns={"intentions_num": "mean_intention"}))

print("\nMean intention by condition:")
print(person_intent.groupby("model_condition")["mean_intention"].agg(["mean", "std", "count"]))

for ref, target, label in [
    ("balanced",    "syco", "E6: syco vs balanced"),
    ("challenging", "syco", "E6: syco vs challenging"),
]:
    sub = person_intent[person_intent["model_condition"].isin([ref, target])].copy()
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, target])
    run_ols(sub, "mean_intention ~ model_condition", "model_condition[T.syco]",
            label, "Mean Intention to Act", results)

print("\nIntention trajectory (condition x time, balanced=ref):")
df_traj = df_llm_intent.copy()
df_traj["model_condition"] = pd.Categorical(df_traj["model_condition"],
                                            categories=["balanced", "syco", "challenging"])
try:
    model = smf.mixedlm("intentions_num ~ time_c * model_condition", data=df_traj,
                        groups=df_traj["participant_id"], re_formula="1").fit(reml=True)
    print(model.summary().tables[1])
    for coef in model.fe_params.index:
        if "time_c:model_condition" in coef:
            b, se, p = model.fe_params[coef], model.bse[coef], model.pvalues[coef]
            d = b / np.sqrt(model.scale)
            results.append({
                "analysis": "E6: intention trajectory",
                "outcome": f"Intention ({coef})",
                "cohens_d": d,
                "ci_lower": (b - 1.96 * se) / np.sqrt(model.scale),
                "ci_upper": (b + 1.96 * se) / np.sqrt(model.scale),
                "p_value": p, "b": b,
                "b_ci_lower": b - 1.96 * se, "b_ci_upper": b + 1.96 * se,
                "n_obs": int(model.nobs), "model_type": "MixedLM",
            })
            print(f"  {coef}: d={d:+.3f}, p={p:.4f}")
except Exception as e:
    print(f"  failed: {e}", flush=True)

print("\n--- E6 (continued): AI-influenced decisions (weekly check-ins)")
BIN_COLS = ["decisions-binary_1", "decisions-binary_2",
            "decisions-binary_3", "decisions-binary_4"]
weekly_act = main[main["session_numeric"].isin([4, 8, 12])].copy()
weekly_act["n_topics_acted"] = sum(weekly_act[c].notna().astype(int) for c in BIN_COLS)
weekly_act["acted_any"] = (weekly_act["n_topics_acted"] > 0).astype(int)
w = weekly_act.dropna(subset=["model_condition", "participant_id"]).copy()

pp = (w.groupby(["participant_id", "model_condition"])
        .agg(n_topics_mean=("n_topics_acted", "mean"),
             n_topics_total=("n_topics_acted", "sum"),
             acted_any_rate=("acted_any", "mean"),
             n_check_ins=("n_topics_acted", "size"))
        .reset_index())

print(f"\nN participants with >=1 weekly check-in (LLM conditions): {len(pp)}")
print("Mean topics acted on per check-in, by condition:")
print(pp.groupby("model_condition")["n_topics_mean"].agg(["count", "mean", "std"]).round(3).to_string())
print("\n'Acted on >=1 topic' rate by condition:")
print(pp.groupby("model_condition")["acted_any_rate"].agg(["count", "mean", "std"]).round(3).to_string())

def welch_d(x, y):
    nx, ny = len(x), len(y)
    mx, my = float(np.mean(x)), float(np.mean(y))
    vx, vy = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    sp = np.sqrt((vx + vy) / 2.0)
    d  = (mx - my) / sp if sp > 0 else float("nan")
    t, p = stats.ttest_ind(x, y, equal_var=False)
    se = np.sqrt(vx / nx + vy / ny)
    ci = (mx - my - 1.96 * se, mx - my + 1.96 * se)
    return mx, my, mx - my, ci, t, p, d

print("\nPairwise contrasts (per-participant n_topics_mean):")
for a, b in [("syco", "balanced"), ("syco", "challenging"), ("challenging", "balanced")]:
    xa = pp.loc[pp.model_condition == a, "n_topics_mean"]
    xb = pp.loc[pp.model_condition == b, "n_topics_mean"]
    mx, my, diff, ci, t, p, d = welch_d(xa, xb)
    print(f"  {a:12s} vs {b:12s}: M_a={mx:.3f}, M_b={my:.3f}, "
          f"diff={diff:+.3f} 95%CI[{ci[0]:+.3f},{ci[1]:+.3f}], "
          f"t={t:.2f}, p={p:.3f}, d={d:.3f}")

w["session_c"] = w["session_numeric"] - 4
w["sycoF"] = (w["model_condition"] == "syco").astype(int)
w["chalF"] = (w["model_condition"] == "challenging").astype(int)
print("\nOLS trajectory test (HC1):")
mod_traj = smf.ols(
    "n_topics_acted ~ sycoF + chalF + session_c + sycoF:session_c + chalF:session_c",
    data=w,
).fit(cov_type="HC1")
print(mod_traj.summary().tables[1])

def block_for(s):
    if s in (1, 2, 3, 4):  return 4
    if s in (5, 6, 7, 8):  return 8
    if s in (9, 10, 11, 12): return 12
    return None

main["week_block"] = main["session_numeric"].apply(block_for)
intent_per_block = (main.dropna(subset=["intentions_num", "week_block"])
                       .groupby(["participant_id", "week_block"])["intentions_num"]
                       .mean().reset_index()
                       .rename(columns={"intentions_num": "mean_intent"}))
w2 = w.merge(intent_per_block, left_on=["participant_id", "session_numeric"],
                                right_on=["participant_id", "week_block"])
print(f"\nN weekly obs with intention data: {len(w2)}")
print(f"Pearson r(mean_intent, n_topics_acted) = "
      f"{w2['mean_intent'].corr(w2['n_topics_acted']):.3f}")
mod_pred = smf.ols("n_topics_acted ~ mean_intent + sycoF + chalF", data=w2).fit(cov_type="HC1")
print("\nOLS n_topics_acted ~ mean_intent + condition (HC1):")
print(mod_pred.summary().tables[1])

print("\n" + "=" * 70)
print("  e7: ai preferences")
print("=" * 70, flush=True)

s12["likelihood_num"] = to_numeric(s12["likelihood"], LIKELIHOOD_MAP)
s12["reachout_binary"] = (
    s12["reach-out"].astype(str).str.strip().str.lower().str.contains("notified")
).astype(float)

print("\nLikelihood of reuse by condition:")
print(s12.groupby("model_condition")["likelihood_num"].agg(["mean", "std", "count"]))
print("\nOpt-in rate by condition:")
print(s12.groupby("model_condition")["reachout_binary"].agg(["mean", "count"]))

for ref, target, label in [
    ("balanced",    "syco", "E7: syco vs balanced"),
    ("challenging", "syco", "E7: syco vs challenging"),
]:
    sub = s12[s12["model_condition"].isin([ref, target])].copy()
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, target])
    run_ols(sub.dropna(subset=["likelihood_num"]),
            "likelihood_num ~ model_condition", "model_condition[T.syco]",
            label, "Likelihood of Reuse", results)
    run_ols(sub.dropna(subset=["reachout_binary"]),
            "reachout_binary ~ model_condition", "model_condition[T.syco]",
            label, "Behavioral Opt-in", results)

print("\n" + "=" * 70)
print("  e8: overclaiming")
print("=" * 70, flush=True)

foil_items = [2, 4, 7, 8, 10, 11]
foil_cols  = [f"overclaim_{i}" for i in foil_items]

def compute_overclaiming(df):
    df = df.copy()
    for col in foil_cols:
        df[col + "_num"] = to_numeric(df[col], OVERCLAIM_MAP)
    df["overclaiming"] = df[[c + "_num" for c in foil_cols]].mean(axis=1)
    return df

s12_oc = compute_overclaiming(s12[s12["model_condition"].isin(["balanced", "syco", "challenging"])])
nollm_oc = compute_overclaiming(nollm)
oc_cols = ["participant_id", "overclaiming", "model_condition"]
oc_all = pd.concat([s12_oc[oc_cols], nollm_oc[oc_cols]], ignore_index=True)

print("\nOverclaiming (mean foil familiarity) by condition:")
print(oc_all.groupby("model_condition")["overclaiming"].agg(["mean", "std", "count"]))

for ref, target, label in [
    ("balanced",    "syco", "E8: syco vs balanced"),
    ("challenging", "syco", "E8: syco vs challenging"),
    ("nollm",       "syco", "E8: syco vs no-LLM"),
]:
    sub = oc_all[oc_all["model_condition"].isin([ref, target])].copy()
    sub["model_condition"] = pd.Categorical(sub["model_condition"], categories=[ref, target])
    run_ols(sub, "overclaiming ~ model_condition", "model_condition[T.syco]",
            label, "Overclaiming", results)
