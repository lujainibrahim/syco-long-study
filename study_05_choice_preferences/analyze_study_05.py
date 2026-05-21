"""Analysis of Study 5 on user preferences across sycophantic, neutral, and challenging AI models."""

import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

STYLES = ["sycophantic", "neutral", "challenging"]

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

def holm_bonferroni(pvals, alpha=0.05):
    """Return Holm-Bonferroni adjusted p-values, preserving input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        adj_p = (m - i) * pvals[idx]
        running = max(running, adj_p)
        adj[idx] = min(running, 1.0)
    return adj

print("\n[load analytic file]")
df = pd.read_csv("data/study_05_data.csv")
print(f"  Rows in choices_full.csv: {len(df)}")
print(f"  Finished == True: {(df['Finished'] == True).sum()}")

df = df[df["Finished"] == True].copy()
df = df[df["consent"].astype(str).str.strip().str.lower().isin(["yes", "agree"])].copy()
print(f"  Finished + consented: {len(df)}")

n_h1 = df["first_choice_style"].notna().sum()
n_rank = df["third_choice_style"].notna().sum()
print(f"\n  H1 sample (first_choice_style observed): N = {n_h1}")
print(f"  E1 sample (full ranking derivable):       N = {n_rank}")
print(f"\n[analytic sample: n = {len(df)} (preregistered target n = 500)]")
print("\n[h1: first-choice preference]")
choice_counts = df["first_choice_style"].value_counts()
n = int(choice_counts.sum())

print(f"\n  First-choice distribution (N = {n}):")
for s in STYLES:
    c = int(choice_counts.get(s, 0))
    print(f"    {s:13s}: {c:3d}/{n}  ({c / n:.1%})")

n_syco = int(choice_counts.get("sycophantic", 0))
binom = stats.binomtest(n_syco, n, 1 / 3, alternative="greater")
ci = binom.proportion_ci(confidence_level=0.95)
print(f"\n  Binomial test (one-sided, sycophantic > 1/3):")
print(f"    Observed proportion = {n_syco / n:.3f}, 95% CI [{ci.low:.3f}, {ci.high:.3f}]")
print(f"    p = {binom.pvalue:.6f}{sig(binom.pvalue)}")
observed = [int(choice_counts.get(s, 0)) for s in STYLES]
expected = [n / 3] * 3
chi2, p_chi = stats.chisquare(observed, expected)
w = np.sqrt(chi2 / n)
print(f"\n  Chi-square goodness-of-fit (omnibus, vs. uniform):")
print(f"    chi2(2) = {chi2:.3f}, p = {p_chi:.6f}{sig(p_chi)}, Cohen's w = {w:.3f}")
print("\n[e1: ranking analysis]")
rank_df = df.dropna(subset=["first_choice_style", "second_choice_style",
                            "third_choice_style"]).copy()

for s in STYLES:
    rank_df[f"rank_{s}"] = rank_df.apply(
        lambda r, s=s: 1 if r["first_choice_style"] == s
        else 2 if r["second_choice_style"] == s else 3, axis=1
    )

print(f"\n  Per-style rank summary (N with full ranking = {len(rank_df)}):")
print(f"    {'style':13s}  {'mean':>5s}  {'median':>6s}")
for s in STYLES:
    col = f"rank_{s}"
    print(f"    {s:13s}  {rank_df[col].mean():5.2f}  {rank_df[col].median():6.1f}")

friedman_stat, friedman_p = stats.friedmanchisquare(
    rank_df["rank_sycophantic"], rank_df["rank_neutral"], rank_df["rank_challenging"]
)
kendall_w = friedman_stat / (len(rank_df) * (3 - 1))
print(f"\n  Friedman test (omnibus):")
print(f"    chi2(2) = {friedman_stat:.3f}, p = {friedman_p:.6f}{sig(friedman_p)},  "
      f"Kendall's W = {kendall_w:.3f}")

if friedman_p < 0.05:
    print(f"\n  Pairwise Wilcoxon signed-rank tests (Holm-Bonferroni):")
    pairs = [("sycophantic", "neutral"),
             ("sycophantic", "challenging"),
             ("neutral",     "challenging")]
    raw_p = []
    raw_stats = []
    rank_biserial = []
    for a, b in pairs:
        stat, p = stats.wilcoxon(rank_df[f"rank_{a}"], rank_df[f"rank_{b}"])
        diffs = rank_df[f"rank_{a}"] - rank_df[f"rank_{b}"]
        diffs_nz = diffs[diffs != 0]
        n_pos = (diffs_nz > 0).sum()
        n_neg = (diffs_nz < 0).sum()
        r_rb = (n_pos - n_neg) / len(diffs_nz) if len(diffs_nz) > 0 else 0
        raw_p.append(p)
        raw_stats.append(stat)
        rank_biserial.append(r_rb)
    adj_p = holm_bonferroni(raw_p)
    for (a, b), W, p, p_adj, r_rb in zip(pairs, raw_stats, raw_p, adj_p, rank_biserial):
        print(f"    {a:13s} vs {b:13s}: W = {W:7.1f}, p = {p:.6f}, "
              f"p_holm = {p_adj:.4f}{sig(p_adj)}, r_rb = {r_rb:+.3f}")

# Full distribution of the six possible orderings
print(f"\n  Full ranking distribution (six possible orderings):")
ordering_lookup = {"sycophantic": "S", "neutral": "N", "challenging": "C"}
rank_df["ordering"] = (
    rank_df["first_choice_style"].map(ordering_lookup) + " > "
    + rank_df["second_choice_style"].map(ordering_lookup) + " > "
    + rank_df["third_choice_style"].map(ordering_lookup)
)
ord_counts = rank_df["ordering"].value_counts()
for pattern in ["S > N > C", "S > C > N", "N > S > C", "N > C > S",
                "C > S > N", "C > N > S"]:
    c = int(ord_counts.get(pattern, 0))
    print(f"    {pattern:11s}  {c:3d}  ({c / len(rank_df):.1%})")


print("\n[e2: sampling-order check]")
def _label_to_style(row, label):
    try:
        mapping = json.loads(row["label_style_map"])
        return mapping.get(label)
    except Exception:
        return None

def _first_sampled_style(row):
    if pd.isna(row.get("sampling_order")):
        return None
    first_label = str(row["sampling_order"]).split(",")[0].strip()
    return _label_to_style(row, first_label)

def _chose_first_sampled(row):
    if pd.isna(row.get("sampling_order")) or pd.isna(row.get("choice")):
        return None
    first_label = str(row["sampling_order"]).split(",")[0].strip()
    chosen_label = str(row["choice"]).replace("Model ", "").strip()
    return chosen_label == first_label

df["first_sampled_style"] = df.apply(_first_sampled_style, axis=1)
df["chose_first_sampled"] = df.apply(_chose_first_sampled, axis=1)

valid_first = df["chose_first_sampled"].dropna()
n_first_total = len(valid_first)
n_first = int(valid_first.sum())
print(f"\n  Chose the first sampled model: {n_first}/{n_first_total} ({n_first / n_first_total:.1%})")
print(f"  Chance baseline: 33.3%")
binom_first = stats.binomtest(n_first, n_first_total, 1 / 3)
print(f"  Two-sided binomial test (vs. 1/3): p = {binom_first.pvalue:.6f}{sig(binom_first.pvalue)}")
print(f"\n  Distribution of first-sampled style (before any conversation):")
fs = df["first_sampled_style"].value_counts(dropna=False)
fs_total = fs.sum()
for s in STYLES:
    c = int(fs.get(s, 0))
    print(f"    {s:13s}: {c:3d}  ({c / fs_total:.1%})")

print("\n[e3: topic sensitivity]")
if "selected_topic_name" in df.columns:
    topic_choice = df.dropna(subset=["selected_topic_name", "first_choice_style"]).copy()
    # Restrict to topics with >= 20 responses to keep cells well-populated.
    topic_counts = topic_choice["selected_topic_name"].value_counts()
    keep_topics = topic_counts[topic_counts >= 20].index
    sub = topic_choice[topic_choice["selected_topic_name"].isin(keep_topics)].copy()
    ct = pd.crosstab(sub["selected_topic_name"], sub["first_choice_style"])
    ct = ct.reindex(columns=STYLES, fill_value=0)
    print(f"\n  Topics retained (>= 20 responses each): {len(keep_topics)}, N covered = {len(sub)}")
    print(ct.to_string())
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2_t, p_t, dof_t, _ = stats.chi2_contingency(ct)
        print(f"\n  Chi-square test of independence: "
              f"chi2({dof_t}) = {chi2_t:.3f}, p = {p_t:.4f}{sig(p_t)}")

print("\n[e4: choice reasons]")
REASON_COLS = {
    "reason_4": "It felt like it understood me best",
    "reason_5": "It gave the most useful advice",
    "reason_6": "It was the most objective",
    "reason_7": "It was the easiest to talk to",
    "reason_8": "Other",
}

reason_rows = []
print(f"\n  Endorsement frequencies (each option allows multiple selections):")
print(f"    {'Reason':45s}  {'overall':>8s}  "
      f"{'syco':>5s}  {'neut':>5s}  {'chall':>5s}")
for col, label in REASON_COLS.items():
    overall = df[col].notna().sum()
    by_style = {}
    for s in STYLES:
        by_style[s] = df.loc[df["first_choice_style"] == s, col].notna().sum()
    print(f"    {label:45s}  {overall:>8d}  "
          f"{by_style['sycophantic']:>5d}  {by_style['neutral']:>5d}  "
          f"{by_style['challenging']:>5d}")
    reason_rows.append({
        "reason": label,
        "overall_endorsed": int(overall),
        "endorsed_sycophantic": int(by_style["sycophantic"]),
        "endorsed_neutral": int(by_style["neutral"]),
        "endorsed_challenging": int(by_style["challenging"]),
    })

# Chi-square: for each reason, is endorsement (yes/no) independent of chosen style?
print(f"\n  Omnibus chi-square tests (3 first-choice groups x endorsed/not):")
for col, label in REASON_COLS.items():
    if label == "Other":
        continue
    df["__endorsed"] = df[col].notna().astype(int)
    ct = pd.crosstab(df["first_choice_style"], df["__endorsed"])
    if ct.shape[0] >= 2 and ct.shape[1] == 2:
        chi2_r, p_r, dof_r, _ = stats.chi2_contingency(ct)
        print(f"    {label:45s}: chi2({dof_r}) = {chi2_r:6.2f}, p = {p_r:.4f}{sig(p_r)}")
df.drop(columns=["__endorsed"], inplace=True, errors="ignore")

# Pairwise 2x2 chi-square tests
print(f"\n  Pairwise 2x2 chi-square tests (each first-choice pair x endorsed/not):")
pairwise_pairs = [("sycophantic", "neutral"),
                  ("sycophantic", "challenging"),
                  ("neutral",     "challenging")]
pairwise_rows = []
for col, label in REASON_COLS.items():
    if label == "Other":
        continue
    df["__e"] = df[col].notna().astype(int)
    rates = df.groupby("first_choice_style")["__e"].agg(["sum", "count"])
    print(f"\n    {label}")
    for s in STYLES:
        n_y = int(rates.loc[s, "sum"])
        n_t = int(rates.loc[s, "count"])
        print(f"      {s:13s}: {n_y}/{n_t} ({n_y / n_t:.1%})")
    for a, b in pairwise_pairs:
        sub = df[df["first_choice_style"].isin([a, b])]
        ct2 = pd.crosstab(sub["first_choice_style"], sub["__e"])
        if ct2.shape == (2, 2):
            chi2_p, p_p, dof_p, _ = stats.chi2_contingency(ct2)
            print(f"      {a:13s} vs {b:13s}: chi2({dof_p}) = {chi2_p:6.2f}, "
                  f"p = {p_p:.4f}{sig(p_p)}")
            pairwise_rows.append({
                "reason": label,
                "group_a": a, "group_b": b,
                "rate_a": round(rates.loc[a, "sum"] / rates.loc[a, "count"], 4),
                "rate_b": round(rates.loc[b, "sum"] / rates.loc[b, "count"], 4),
                "chi2_dof1": round(chi2_p, 4),
                "p_value": round(p_p, 6),
            })
df.drop(columns=["__e"], inplace=True, errors="ignore")
pairwise_df = pd.DataFrame(pairwise_rows)
summary_rows = [{
    "analysis": "H1 first-choice (sycophantic vs. uniform)",
    "n": n,
    "stat_name": "binomial p (one-sided)",
    "stat_value": binom.pvalue,
    "extra": f"prop = {n_syco / n:.3f}, 95% CI [{ci.low:.3f}, {ci.high:.3f}]",
}, {
    "analysis": "H1 chi-square goodness-of-fit",
    "n": n,
    "stat_name": "chi2(2)",
    "stat_value": chi2,
    "extra": f"p = {p_chi:.6f}, Cohen's w = {w:.3f}",
}, {
    "analysis": "E1 Friedman ranking",
    "n": len(rank_df),
    "stat_name": "chi2(2)",
    "stat_value": friedman_stat,
    "extra": f"p = {friedman_p:.6f}, Kendall's W = {kendall_w:.3f}",
}, {
    "analysis": "First-choice count: sycophantic",
    "n": n,
    "stat_name": "count (proportion)",
    "stat_value": n_syco,
    "extra": f"{n_syco / n:.3%}",
}, {
    "analysis": "First-choice count: neutral",
    "n": n,
    "stat_name": "count (proportion)",
    "stat_value": int(choice_counts.get("neutral", 0)),
    "extra": f"{choice_counts.get('neutral', 0) / n:.3%}",
}, {
    "analysis": "First-choice count: challenging",
    "n": n,
    "stat_name": "count (proportion)",
    "stat_value": int(choice_counts.get("challenging", 0)),
    "extra": f"{choice_counts.get('challenging', 0) / n:.3%}",
}]
ord_table = (
    rank_df["ordering"].value_counts()
    .rename_axis("ordering").reset_index(name="n")
)
ord_table["proportion"] = (ord_table["n"] / len(rank_df)).round(4)
