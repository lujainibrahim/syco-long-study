"""Analysis of Study 4 longitudinal attrition."""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

ALL_SESSIONS = ["one", "two", "three", "four", "five", "six",
                "seven", "eight", "nine", "ten", "eleven", "twelve"]
CONDITIONS = ["balanced", "syco", "challenging"]

# Likert mappings

AGREE_7 = {"Strongly disagree": 1, "Disagree": 2, "Somewhat disagree": 3,
           "Neither agree nor disagree": 4, "Somewhat agree": 5, "Agree": 6,
           "Strongly agree": 7}

AGREE_5 = {"Strongly disagree": 1, "Disagree a little": 2,
           "Neither agree nor disagree": 3, "Agree a little": 4,
           "Strongly agree": 5}

AGREE_5B = {"Strongly disagree": 1, "Disagree": 2,
            "Neither agree nor disagree": 3, "Agree": 4, "Strongly agree": 5}

COMFORT_MAP = {"Not at all": 1, "Slightly": 2, "Somewhat": 3,
               "Moderately": 4, "Quite a bit": 5, "Very much": 6,
               "Extremely": 7}

LSNS_MAP = {"0": 0, "1": 1, "2": 2, "3 or 4": 3.5, "5-8": 6.5, "9+": 9}

AI_USE_MAP = {"Never": 1, "Rarely": 2, "Monthly": 3, "Weekly": 4, "Daily": 5}

AVERAGE_MAP = {"Much worse than average": 1, "Somewhat worse than average": 2,
               "About average": 3, "Somewhat better than average": 4,
               "Much better than average": 5}

OTHER_MAP = {"Much worse than them": 1, "Somewhat worse than them": 2,
             "About the same": 3, "Somewhat better than them": 4,
             "Much better than them": 5}

# Scale definitions

SCALES = {
    "AI use frequency": {"cols": ["ai-use"], "map": AI_USE_MAP},
    "LSNS-1": {"cols": ["lsns-1"], "map": LSNS_MAP},
    "LSNS-2": {"cols": ["lsns-2"], "map": LSNS_MAP},
    "LSNS-3": {"cols": ["lsns-3"], "map": LSNS_MAP},
    "LSNS-4": {"cols": ["lsns-4"], "map": LSNS_MAP},
    "LSNS-5": {"cols": ["lsns-5"], "map": LSNS_MAP},
    "LSNS-6": {"cols": ["lsns-6"], "map": LSNS_MAP},
    "Big5-1": {"cols": ["big5_1"], "map": AGREE_5},
    "Big5-2": {"cols": ["big5_2"], "map": AGREE_5},
    "Big5-3": {"cols": ["big5_3"], "map": AGREE_5},
    "Big5-4": {"cols": ["big5_4"], "map": AGREE_5},
    "Big5-5": {"cols": ["big5_5"], "map": AGREE_5},
    "Big5-6": {"cols": ["big5_6"], "map": AGREE_5},
    "Big5-7": {"cols": ["big5_7"], "map": AGREE_5},
    "Big5-8": {"cols": ["big5_8"], "map": AGREE_5},
    "Big5-9": {"cols": ["big5_9"], "map": AGREE_5},
    "Big5-10": {"cols": ["big5_10"], "map": AGREE_5},
    "Comfort-1 (AI personal)": {"cols": ["comfort_1"], "map": COMFORT_MAP},
    "Comfort-2 (AI emotional)": {"cols": ["comfort_2"], "map": COMFORT_MAP},
    "Comfort-3 (humans)": {"cols": ["comfort_3"], "map": COMFORT_MAP},
    "Well-being-1": {"cols": ["well-being_1"], "map": AGREE_5B},
    "Well-being-2": {"cols": ["well-being_2"], "map": AGREE_5B},
    "Well-being-3": {"cols": ["well-being_3"], "map": AGREE_5B},
    "Well-being-4": {"cols": ["well-being_4"], "map": AGREE_5B},
    "Well-being-5": {"cols": ["well-being_5"], "map": AGREE_5B},
    "Well-being-6": {"cols": ["well-being_6"], "map": AGREE_5B},
    "Well-being-7": {"cols": ["well-being_7"], "map": AGREE_5B},
    "Well-being-8": {"cols": ["well-being_8"], "map": AGREE_5B},
    "Well-being-9": {"cols": ["well-being_9"], "map": AGREE_5B},
    "Objectivity-1": {"cols": ["objectivity_1"], "map": AGREE_7},
    "Objectivity-2": {"cols": ["objectivity_2"], "map": AGREE_7},
    "Objectivity-3": {"cols": ["objectivity_3"], "map": AGREE_7},
    "FK humans-1": {"cols": ["closeness-humans_1"], "map": AGREE_7},
    "FK humans-2": {"cols": ["closeness-humans_2"], "map": AGREE_7},
    "FK humans-3": {"cols": ["closeness-humans_3"], "map": AGREE_7},
    "FK humans-4": {"cols": ["closeness-humans_4"], "map": AGREE_7},
    "SIH-1": {"cols": ["SIH_1"], "map": AGREE_7},
    "SIH-2": {"cols": ["SIH_2"], "map": AGREE_7},
    "SIH-3": {"cols": ["SIH_3"], "map": AGREE_7},
    "Self-avg-1": {"cols": ["average_1"], "map": AVERAGE_MAP},
    "Self-avg-2": {"cols": ["average_2"], "map": AVERAGE_MAP},
    "Self-avg-3": {"cols": ["average_3"], "map": AVERAGE_MAP},
    "Self-avg-4": {"cols": ["average_4"], "map": AVERAGE_MAP},
    "Self-avg-5": {"cols": ["average_5"], "map": AVERAGE_MAP},
    "Other-ppl-1": {"cols": ["other-people_1"], "map": OTHER_MAP},
    "Other-ppl-2": {"cols": ["other-people_2"], "map": OTHER_MAP},
    "Other-ppl-3": {"cols": ["other-people_3"], "map": OTHER_MAP},
    "Other-ppl-4": {"cols": ["other-people_4"], "map": OTHER_MAP},
    "Other-ppl-5": {"cols": ["other-people_5"], "map": OTHER_MAP},
    "Relationship-1": {"cols": ["relationship#1_1"], "map": AGREE_7},
    "Relationship-2": {"cols": ["relationship#1_2"], "map": AGREE_7},
    "Relationship-3": {"cols": ["relationship#1_3"], "map": AGREE_7},
    "Relationship-4": {"cols": ["relationship#1_4"], "map": AGREE_7},
    "Relationship-5": {"cols": ["relationship#1_5"], "map": AGREE_7},
    "Relationship-6": {"cols": ["relationship#1_6"], "map": AGREE_7},
    "Relationship-7": {"cols": ["relationship#1_7"], "map": AGREE_7},
    "Relationship-8": {"cols": ["relationship#1_8"], "map": AGREE_7},
}

CATEGORICAL_VARS = {
    "Gender": "gender",
    "Education": "education",
    "Politics": "politics",
    "Age bracket": "age",
}


def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * a.std()**2 + (nb - 1) * b.std()**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan

def proportion_ci(k, n, z=1.96):
    p = k / n if n > 0 else 0
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
    return p, max(0, p - z * se), min(1, p + z * se)

# Load data

main = pd.read_csv("../data/main_sessions.csv",
                    usecols=["participant_id", "SESSION_NUM", "model_condition", "session_numeric"])
pre = pd.read_csv("../data/pre_treatment.csv")
pid_cond = pd.read_csv("../data/pid_condition.csv")

main = main[main["model_condition"].isin(CONDITIONS)]
pid_cond = pid_cond[pid_cond["model_condition"].isin(CONDITIONS)]

# Per-participant session sets
pid_sessions = main.groupby("participant_id")["SESSION_NUM"].apply(set)
pid_n_sessions = pid_sessions.apply(len)
pid_cond_map = pid_cond.set_index("participant_id")["model_condition"]

all_pids = set(pid_cond_map.index)

# Three completer definitions
strict_complete = set(pid_sessions[pid_sessions.apply(lambda s: set(ALL_SESSIONS).issubset(s))].index)
session12_complete = set(pid_sessions[pid_sessions.apply(lambda s: "twelve" in s)].index)
high_adherence = set(pid_sessions[pid_n_sessions >= 9].index)

defs = {
    "Strict (all 12)": strict_complete,
    "Session-12 (final session)": session12_complete,
    "High-adherence (>=9)": high_adherence,
}

# Prepare pre-treatment with numeric columns
cond_col = "model_condition"
pre_analysis = pre[pre[cond_col].isin(CONDITIONS)].copy()
pre_analysis = pre_analysis.drop_duplicates(subset="participant_id", keep="first")

for name, info in SCALES.items():
    col = info["cols"][0]
    if col in pre_analysis.columns:
        pre_analysis[f"_{name}"] = pre_analysis[col].map(info["map"])

# Results

print("=" * 80)
print("longitudinal attrition analysis")
print("=" * 80)

# Section 1: Descriptive overview

print("\n" + "=" * 80)
print("section 1: descriptive overview")
print("=" * 80)

n_total = len(all_pids)
print(f"\nTotal enrolled (LLM conditions): {n_total}")
print(f"Conditions: {pid_cond_map.value_counts().to_dict()}")
print(f"\nCompleter counts:")
for label, pids in defs.items():
    dropouts = all_pids - pids
    rate = len(dropouts) / n_total * 100
    print(f"  {label}: {len(pids)} completers, {len(dropouts)} dropouts ({rate:.1f}% attrition)")

print(f"\n{'Session':<12}", end="")
for c in CONDITIONS:
    print(f"  {c:>12}", end="")
print(f"  {'Total':>8}")
print("-" * 60)

for sess in ALL_SESSIONS:
    n_by_cond = {}
    for c in CONDITIONS:
        c_pids = set(pid_cond_map[pid_cond_map == c].index)
        n_by_cond[c] = sum(1 for p in c_pids if sess in pid_sessions.get(p, set()))
    total = sum(n_by_cond.values())
    print(f"{sess:<12}", end="")
    for c in CONDITIONS:
        print(f"  {n_by_cond[c]:>12}", end="")
    print(f"  {total:>8}")

print(f"\nSession completion distribution:")
dist = pid_n_sessions.value_counts().sort_index()
for n_sess, count in dist.items():
    print(f"  {n_sess:2d} sessions: {count:4d} participants ({count/n_total*100:.1f}%)")

# Section 2: Differential attrition

print("\n" + "=" * 80)
print("section 2: differential attrition (by condition)")
print("=" * 80)

for label, completers in defs.items():
    print(f"\n--- {label} ---")
    dropouts = all_pids - completers

    table = {}
    for c in CONDITIONS:
        c_pids = set(pid_cond_map[pid_cond_map == c].index)
        n_c = len(c_pids)
        n_drop = len(c_pids & dropouts)
        n_comp = len(c_pids & completers)
        p, lo, hi = proportion_ci(n_drop, n_c)
        table[c] = {"n": n_c, "dropout": n_drop, "completer": n_comp,
                     "rate": p, "ci_lo": lo, "ci_hi": hi}

    print(f"\n  {'Condition':<14} {'N':>5} {'Dropout':>8} {'Completer':>10} {'Rate':>7} {'95% CI':>16}")
    print(f"  {'-'*62}")
    for c in CONDITIONS:
        t = table[c]
        print(f"  {c:<14} {t['n']:>5} {t['dropout']:>8} {t['completer']:>10}"
            f" {t['rate']:>6.1%} [{t['ci_lo']:.3f}, {t['ci_hi']:.3f}]")

    # Chi-square: dropout ~ condition
    obs = np.array([[table[c]["completer"], table[c]["dropout"]] for c in CONDITIONS])
    chi2_val, p_val, dof, _ = stats.chi2_contingency(obs)
    print(f"\n  Chi-square (3x2): chi2 = {chi2_val:.3f}, df = {dof}, p = {p_val:.4f}")

    # Pairwise comparisons
    pairs = [("balanced", "syco"), ("balanced", "challenging"), ("syco", "challenging")]
    for c1, c2 in pairs:
        obs2 = np.array([[table[c1]["completer"], table[c1]["dropout"]],
                         [table[c2]["completer"], table[c2]["dropout"]]])
        chi2_pw, p_pw, _, _ = stats.chi2_contingency(obs2, correction=True)
        print(f"  {c1} vs {c2}: chi2 = {chi2_pw:.3f}, p = {p_pw:.3f}")

# Section 3: Selective attrition

print("\n" + "=" * 80)
print("section 3: selective attrition (baseline characteristics)")
print("=" * 80)

for label, completers in defs.items():
    print(f"\n{'='*70}")
    print(f"  Definition: {label}")
    print(f"{'='*70}")

    pre_analysis["_completer"] = pre_analysis["participant_id"].isin(completers).astype(int)
    comp = pre_analysis[pre_analysis["_completer"] == 1]
    drop = pre_analysis[pre_analysis["_completer"] == 0]
    print(f"  Completers with pre-treatment data: {len(comp)}")
    print(f"  Dropouts with pre-treatment data:   {len(drop)}")

    # Continuous variables
    print(f"\n  {'Variable':<28} {'Comp M(SD)':>14} {'Drop M(SD)':>14} {'d':>7} {'t':>7} {'p':>8}")
    print(f"  {'-'*80}")

    sig_vars = []
    for name in SCALES:
        col = f"_{name}"
        if col not in pre_analysis.columns:
            continue
        c_vals = comp[col].dropna()
        d_vals = drop[col].dropna()
        if len(c_vals) < 5 or len(d_vals) < 5:
            continue
        d_val = cohen_d(c_vals, d_vals)
        t_val, p_val = stats.ttest_ind(c_vals, d_vals, equal_var=False)
        marker = " *" if p_val < 0.05 else ""
        if p_val < 0.05:
            sig_vars.append((name, d_val, p_val))
        print(f"  {name:<28} {c_vals.mean():>6.2f}({c_vals.std():>4.2f})"
            f" {d_vals.mean():>6.2f}({d_vals.std():>4.2f})"
            f" {d_val:>+6.2f} {t_val:>+6.2f}  {p_val:>6.4f}{marker}")

    # Categorical variables
    print(f"\n  Categorical variables:")
    print(f"  {'Variable':<20} {'chi2':>8} {'p':>8} {'note':>30}")
    print(f"  {'-'*68}")

    for vname, col in CATEGORICAL_VARS.items():
        if col not in pre_analysis.columns:
            continue
        ct = pd.crosstab(pre_analysis[col].fillna("Missing"), pre_analysis["_completer"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2_val, p_val, dof, _ = stats.chi2_contingency(ct)
        marker = " *" if p_val < 0.05 else ""
        top_comp = comp[col].mode().iloc[0] if len(comp[col].dropna()) > 0 else "?"
        top_drop = drop[col].mode().iloc[0] if len(drop[col].dropna()) > 0 else "?"
        print(f"  {vname:<20} {chi2_val:>8.2f} {p_val:>8.4f}"
            f"  mode: comp={top_comp}, drop={top_drop}{marker}")

    if sig_vars:
        print(f"\n  Significant (p < .05) continuous variables:")
        for name, d_val, p_val in sorted(sig_vars, key=lambda x: x[2]):
            print(f"    {name}: d = {d_val:+.3f}, p = {p_val:.4f}")
    else:
        print(f"\n  No significant continuous variables at p < .05")

# Section 4: Condition x attrition interaction

print("\n" + "=" * 80)
print("section 4: condition x attrition interaction")
print("=" * 80)
print("  Testing: dropout ~ condition * baseline_var (logistic regression)")
print("  Only testing variables from composites / key scales\n")
KEY_COMPOSITES = {
    "Comfort AI (avg 1-2)": ["_Comfort-1 (AI personal)", "_Comfort-2 (AI emotional)"],
    "Comfort humans (3)": ["_Comfort-3 (humans)"],
    "Well-being (avg)": [f"_Well-being-{i}" for i in range(1, 10)],
    "FK humans (avg)": [f"_FK humans-{i}" for i in range(1, 5)],
    "SIH (avg)": [f"_SIH-{i}" for i in range(1, 4)],
    "Relationship (avg)": [f"_Relationship-{i}" for i in range(1, 9)],
    "Big5 (avg)": [f"_Big5-{i}" for i in range(1, 11)],
    "LSNS (avg)": [f"_LSNS-{i}" for i in range(1, 7)],
    "AI use frequency": ["_AI use frequency"],
}

for label, completers in defs.items():
    print(f"\n--- {label} ---")
    pre_analysis["_completer"] = pre_analysis["participant_id"].isin(completers).astype(int)
    pre_analysis["_dropout"] = 1 - pre_analysis["_completer"]

    # Condition dummies (balanced as reference)
    pre_analysis["_syco"] = (pre_analysis[cond_col] == "syco").astype(int)
    pre_analysis["_challenging"] = (pre_analysis[cond_col] == "challenging").astype(int)

    print(f"\n  {'Baseline var':<25} {'Interaction p':>14} {'Note':>35}")
    print(f"  {'-'*76}")

    for comp_name, comp_cols in KEY_COMPOSITES.items():
        available = [c for c in comp_cols if c in pre_analysis.columns]
        if not available:
            continue
        pre_analysis["_bvar"] = pre_analysis[available].mean(axis=1)
        subset = pre_analysis[["_dropout", "_syco", "_challenging", "_bvar"]].dropna()
        if len(subset) < 20:
            continue

        subset["_syco_x_bvar"] = subset["_syco"] * subset["_bvar"]
        subset["_chal_x_bvar"] = subset["_challenging"] * subset["_bvar"]

        try:
            X = sm.add_constant(subset[["_syco", "_challenging", "_bvar",
                                        "_syco_x_bvar", "_chal_x_bvar"]])
            y = subset["_dropout"]
            model = sm.Logit(y, X).fit(disp=0)

            # Joint test of interaction terms
            r_matrix = np.zeros((2, len(model.params)))
            idx_sx = list(model.params.index).index("_syco_x_bvar")
            idx_cx = list(model.params.index).index("_chal_x_bvar")
            r_matrix[0, idx_sx] = 1
            r_matrix[1, idx_cx] = 1
            wald = model.wald_test(r_matrix)
            p_interact = float(wald.pvalue)

            note = ""
            if p_interact < 0.05:
                b_sx = model.params["_syco_x_bvar"]
                b_cx = model.params["_chal_x_bvar"]
                note = f"syco*bvar={b_sx:+.3f}, chal*bvar={b_cx:+.3f} *"
            print(f"  {comp_name:<25} {p_interact:>14.4f} {note:>35}")
        except Exception as e:
            print(f"  {comp_name:<25} {'FAILED':>14} {str(e)[:35]:>35}")

# Section 5: No-LLM control arm attrition

print("\n" + "=" * 80)
print("section 5: no-llm control arm attrition")
print("=" * 80)

no_llm_assignments = pd.read_csv("../data/assignments_no_llm.csv")
no_llm_control = pd.read_csv("../data/no_llm_control.csv")

nollm_assigned_pids = set(no_llm_assignments["prolific_id"].unique())
nollm_completed_pids = set(no_llm_control["participant_id"].unique())
nollm_dropped_pids = nollm_assigned_pids - nollm_completed_pids

n_assigned = len(nollm_assigned_pids)
n_completed = len(nollm_completed_pids)
n_dropped = len(nollm_dropped_pids)
nollm_rate, nollm_lo, nollm_hi = proportion_ci(n_dropped, n_assigned)

print(f"\n  Assigned (completed pre-treatment): {n_assigned}")
print(f"  Completed no-LLM outcome survey:   {n_completed}")
print(f"  Dropped out:                        {n_dropped}")
print(f"  Attrition rate: {nollm_rate:.1%}  95% CI [{nollm_lo:.3f}, {nollm_hi:.3f}]")
# Compare with LLM conditions
print(f"\n  Comparison with LLM conditions:")
for label, completers in defs.items():
    llm_dropouts = all_pids - completers
    llm_rate = len(llm_dropouts) / n_total
    print(f"    {label}: LLM = {llm_rate:.1%}, No-LLM = {nollm_rate:.1%}")

# Chi-square: no-LLM vs overall LLM (using Session-12 definition)
llm_n = n_total
llm_comp = len(session12_complete)
llm_drop = llm_n - llm_comp
obs_nollm_vs_llm = np.array([[n_completed, n_dropped],
                              [llm_comp, llm_drop]])
chi2_nv, p_nv, _, _ = stats.chi2_contingency(obs_nollm_vs_llm, correction=True)
print(f"\n  No-LLM vs LLM (Session-12 definition) chi2 = {chi2_nv:.3f}, p = {p_nv:.4f}")
# Selective attrition for no-LLM
print(f"\n  --- Selective attrition (no-LLM arm) ---")
nollm_pre = pre[pre["participant_id"].isin(nollm_assigned_pids)].copy()
nollm_pre = nollm_pre.drop_duplicates(subset="participant_id", keep="first")

for name, info in SCALES.items():
    col = info["cols"][0]
    if col in nollm_pre.columns:
        nollm_pre[f"_{name}"] = nollm_pre[col].map(info["map"])

nollm_pre["_completer"] = nollm_pre["participant_id"].isin(nollm_completed_pids).astype(int)
nollm_comp = nollm_pre[nollm_pre["_completer"] == 1]
nollm_drop = nollm_pre[nollm_pre["_completer"] == 0]
print(f"  Completers with pre-treatment data: {len(nollm_comp)}")
print(f"  Dropouts with pre-treatment data:   {len(nollm_drop)}")
print(f"\n  {'Variable':<28} {'Comp M(SD)':>14} {'Drop M(SD)':>14} {'d':>7} {'t':>7} {'p':>8}")
print(f"  {'-'*80}")
nollm_sig = []
for name in SCALES:
    col = f"_{name}"
    if col not in nollm_pre.columns:
        continue
    c_vals = nollm_comp[col].dropna()
    d_vals = nollm_drop[col].dropna()
    if len(c_vals) < 5 or len(d_vals) < 5:
        continue
    d_val = cohen_d(c_vals, d_vals)
    t_val, p_val = stats.ttest_ind(c_vals, d_vals, equal_var=False)
    marker = " *" if p_val < 0.05 else ""
    if p_val < 0.05:
        nollm_sig.append((name, d_val, p_val))
    print(f"  {name:<28} {c_vals.mean():>6.2f}({c_vals.std():>4.2f})"
        f" {d_vals.mean():>6.2f}({d_vals.std():>4.2f})"
        f" {d_val:>+6.2f} {t_val:>+6.2f}  {p_val:>6.4f}{marker}")

print(f"\n  Categorical variables:")
print(f"  {'Variable':<20} {'chi2':>8} {'p':>8} {'note':>30}")
print(f"  {'-'*68}")
for vname, col in CATEGORICAL_VARS.items():
    if col not in nollm_pre.columns:
        continue
    ct = pd.crosstab(nollm_pre[col].fillna("Missing"), nollm_pre["_completer"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        continue
    chi2_val, p_val, dof, _ = stats.chi2_contingency(ct)
    marker = " *" if p_val < 0.05 else ""
    top_comp = nollm_comp[col].mode().iloc[0] if len(nollm_comp[col].dropna()) > 0 else "?"
    top_drop = nollm_drop[col].mode().iloc[0] if len(nollm_drop[col].dropna()) > 0 else "?"
    print(f"  {vname:<20} {chi2_val:>8.2f} {p_val:>8.4f}"
        f"  mode: comp={top_comp}, drop={top_drop}{marker}")

if nollm_sig:
    print(f"\n  Significant (p < .05) continuous variables:")
    for name, d_val, p_val in sorted(nollm_sig, key=lambda x: x[2]):
        print(f"    {name}: d = {d_val:+.3f}, p = {p_val:.4f}")
else:
    print(f"\n  No significant continuous variables at p < .05")

print(f"\n{'='*80}")
print(f"{'='*80}")