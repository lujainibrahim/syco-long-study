"""Analysis of the Study 4 manipulation check."""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings("ignore")

CONDITIONS = ["balanced", "syco", "challenging"]
COND_LABELS = {"balanced": "Balanced", "syco": "Sycophantic", "challenging": "Challenging"}

LIKERT_7 = {
    "Strongly disagree": 1, "Disagree": 2, "Somewhat disagree": 3,
    "Neither agree nor disagree": 4, "Somewhat agree": 5, "Agree": 6,
    "Strongly agree": 7,
}

ITEMS = {
    "mani-check_1": "The AI chatbot questioned my assumptions",
    "mani-check_2": "The AI chatbot told me I was right",
    "mani-check_3": "The AI chatbot went along with how I was thinking about the situation",
    "mani-check_4": "The AI chatbot pointed out flaws in my reasoning",
    "mani-check_5": "The AI chatbot said my thinking made sense",
    "mani-check_6": "The AI chatbot took my view at face value",
    "mani-check_7": "The AI chatbot challenged my thinking",
    "mani-check_8": "The AI chatbot agreed with me",
    "mani-check_9": "The AI chatbot didn't question the way I saw things",
}

SUBSCALES = {
    "Challenge": ["mani-check_1", "mani-check_4", "mani-check_7"],
    "Explicit agreement": ["mani-check_2", "mani-check_5", "mani-check_8"],
    "Premise acceptance": ["mani-check_3", "mani-check_6", "mani-check_9"],
}

def cohen_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * a.std()**2 + (nb - 1) * b.std()**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


df = pd.read_csv("../data/main_sessions.csv")
session12 = df[(df["SESSION_NUM"] == "twelve") & (df["model_condition"].isin(CONDITIONS))].copy()

for col in ITEMS:
    session12[col + "_num"] = session12[col].map(LIKERT_7)

for subscale, items in SUBSCALES.items():
    session12[subscale] = session12[[i + "_num" for i in items]].mean(axis=1)

analysis = session12.dropna(subset=list(SUBSCALES.keys()))

print("=" * 80)
print("manipulation check analysis")
print("=" * 80)

print(f"\nParticipants with complete manipulation check data (session 12): {len(analysis)}")
for c in CONDITIONS:
    print(f"  {COND_LABELS[c]}: {len(analysis[analysis['model_condition'] == c])}")

# Item-level results

print("\n" + "=" * 80)
print("section 1: item-level results by condition")
print("=" * 80)

print(f"\n  {'Item':<55} {'Bal M(SD)':>12} {'Syc M(SD)':>12} {'Cha M(SD)':>12}")
print(f"  {'-'*92}")
for col, wording in ITEMS.items():
    num_col = col + "_num"
    parts = []
    for c in CONDITIONS:
        vals = analysis.loc[analysis["model_condition"] == c, num_col].dropna()
        parts.append(f"{vals.mean():>5.2f}({vals.std():.2f})")
    print(f"  {wording:<55} {parts[0]:>12} {parts[1]:>12} {parts[2]:>12}")

# Subscale-level analysis

print("\n" + "=" * 80)
print("section 2: subscale analysis")
print("=" * 80)

for subscale in SUBSCALES:
    print(f"\n--- {subscale} ---")

    # Cronbach's alpha
    items_num = [i + "_num" for i in SUBSCALES[subscale]]
    item_data = analysis[items_num].dropna()
    k = len(items_num)
    item_vars = item_data.var(axis=0)
    total_var = item_data.sum(axis=1).var()
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    print(f"  Cronbach's alpha: {alpha:.3f}")

    # Descriptives by condition
    print(f"\n  {'Condition':<14} {'N':>5} {'Mean':>7} {'SD':>7} {'95% CI':>18}")
    print(f"  {'-'*54}")
    for c in CONDITIONS:
        vals = analysis.loc[analysis["model_condition"] == c, subscale].dropna()
        se = vals.std() / np.sqrt(len(vals))
        lo, hi = vals.mean() - 1.96 * se, vals.mean() + 1.96 * se
        print(f"  {COND_LABELS[c]:<14} {len(vals):>5} {vals.mean():>7.3f} {vals.std():>7.3f}"
            f"  [{lo:.3f}, {hi:.3f}]")

    # One-way ANOVA
    groups = [analysis.loc[analysis["model_condition"] == c, subscale].dropna() for c in CONDITIONS]
    f_val, p_anova = stats.f_oneway(*groups)
    # Eta-squared
    grand_mean = analysis[subscale].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
    ss_total = sum(((g - grand_mean)**2).sum() for g in groups)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    print(f"\n  One-way ANOVA: F(2, {len(analysis) - 3}) = {f_val:.2f}, p = {p_anova:.6f}, η² = {eta_sq:.4f}")

    # Pairwise comparisons (Welch t-tests, Holm-Bonferroni corrected)
    print(f"\n  Pairwise comparisons (Welch t-tests, Holm-Bonferroni corrected):")
    print(f"  {'Comparison':<28} {'Diff':>7} {'d':>7} {'p(raw)':>9} {'p(Holm)':>9}")
    print(f"  {'-'*62}")

    pairs = [("balanced", "syco"), ("balanced", "challenging"), ("syco", "challenging")]
    raw_ps = []
    pair_stats = []
    for c1, c2 in pairs:
        g1 = analysis.loc[analysis["model_condition"] == c1, subscale].dropna()
        g2 = analysis.loc[analysis["model_condition"] == c2, subscale].dropna()
        d = cohen_d(g1, g2)
        diff = g1.mean() - g2.mean()
        t_val, p_raw = stats.ttest_ind(g1, g2, equal_var=False)
        raw_ps.append(p_raw)
        pair_stats.append((c1, c2, diff, d, p_raw))

    _, p_holm, _, _ = multipletests(raw_ps, method="holm")

    for (c1, c2, diff, d, p_raw), p_adj in zip(pair_stats, p_holm):
        sig = " ***" if p_adj < 0.001 else " **" if p_adj < 0.01 else " *" if p_adj < 0.05 else ""
        print(f"  {COND_LABELS[c1]} vs {COND_LABELS[c2]:<14} {diff:>+7.3f} {d:>+7.3f} {p_raw:>9.6f} {p_adj:>9.6f}{sig}")

# Ordinal pattern check

print("\n" + "=" * 80)
print("section 3: expected ordinal patterns")
print("=" * 80)

print("\n  Expected pattern for successful manipulation:")
print("    Challenge:          Challenging > Balanced > Sycophantic")
print("    Explicit agreement: Sycophantic > Balanced > Challenging")
print("    Premise acceptance: Sycophantic > Balanced > Challenging")
print("\n  Observed means:")
print(f"  {'Subscale':<22} {'Balanced':>10} {'Sycophantic':>12} {'Challenging':>12} {'Pattern':>10}")
print(f"  {'-'*68}")
for subscale in SUBSCALES:
    means = {}
    for c in CONDITIONS:
        means[c] = analysis.loc[analysis["model_condition"] == c, subscale].mean()

    if subscale == "Challenge":
        expected_order = means["challenging"] > means["balanced"] > means["syco"]
    else:
        expected_order = means["syco"] > means["balanced"] > means["challenging"]

    pattern = "Correct" if expected_order else "✗ Partial"
    print(f"  {subscale:<22} {means['balanced']:>10.3f} {means['syco']:>12.3f}"
        f" {means['challenging']:>12.3f} {pattern:>10}")

# Regression (condition predicting subscales)

print("\n" + "=" * 80)
print("section 4: ols regression (balanced as reference)")
print("=" * 80)

analysis["syco_dummy"] = (analysis["model_condition"] == "syco").astype(int)
analysis["chal_dummy"] = (analysis["model_condition"] == "challenging").astype(int)

for subscale in SUBSCALES:
    print(f"\n--- {subscale} ---")
    y = analysis[subscale]
    X = sm.add_constant(analysis[["syco_dummy", "chal_dummy"]])
    model = sm.OLS(y, X).fit()

    print(f"  R² = {model.rsquared:.4f}, F = {model.fvalue:.2f}, p = {model.f_pvalue:.6f}")
    print(f"  {'Predictor':<16} {'B':>8} {'SE':>8} {'t':>8} {'p':>10} {'95% CI':>20}")
    print(f"  {'-'*72}")
    for var in ["const", "syco_dummy", "chal_dummy"]:
        label = {"const": "Intercept", "syco_dummy": "Sycophantic", "chal_dummy": "Challenging"}[var]
        b = model.params[var]
        se = model.bse[var]
        t = model.tvalues[var]
        p = model.pvalues[var]
        ci = model.conf_int().loc[var]
        sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
        print(f"  {label:<16} {b:>8.3f} {se:>8.3f} {t:>8.2f} {p:>10.4f}"
            f"  [{ci[0]:.3f}, {ci[1]:.3f}]{sig}")


print(f"\n{'='*80}")
print(f"{'='*80}")