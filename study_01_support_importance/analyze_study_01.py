"""Analysis of Study 1 on expected support from a human vs an AI listener."""

import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)

df = pd.read_csv('data/study_01_data.csv')
print(f"Total responses (excl. header rows): {len(df)}")
df = df[df['Finished'].astype(str).str.strip() == 'True']
df = df[df['consent'].astype(str).str.strip() == 'Yes']
print(f"Finished + consented + non-preview: {len(df)}")
print(f"Condition split: {df['condition'].value_counts(dropna=False).to_dict()}")
scale_map = {
    'Not at all important': 1,
    'Slightly important': 2,
    'Somewhat important': 3, 'Somewhat impotrant': 3,
    'Moderately important': 4,
    'Quite important': 5, 'Quite imporant': 5,
    'Very important': 6,
    'Extremely important': 7,
}

# Human condition items
h_emo = ['Q137_1', 'Q137_2', 'Q137_3']
h_inf = ['Q138_1', 'Q138_2', 'Q138_4']
h_att = 'Q138_3'                       # attention check ("Very important")
h_cer = ['Q139_1', 'Q139_2', 'Q139_3']
h_est = ['Q140_1', 'Q140_2', 'Q140_3']

# AI condition items
a_emo = ['Q151_1', 'Q151_2', 'Q151_3']
a_inf = ['Q152_1', 'Q152_2', 'Q152_4']
a_att = 'Q152_3'                       # attention check ("Very important")
a_cer = ['Q153_1', 'Q153_2', 'Q153_3']
a_est = ['Q154_1', 'Q154_2', 'Q154_3']

all_items = (h_emo + h_inf + [h_att] + h_cer + h_est +
             a_emo + a_inf + [a_att] + a_cer + a_est)
for col in all_items:
    if col in df.columns:
        df[col] = df[col].map(scale_map)

human_mask = df['condition'] == 'human'
ai_mask = df['condition'] == 'ai'

att_pass = pd.Series(False, index=df.index)
att_pass.loc[human_mask] = df.loc[human_mask, h_att] == 6
att_pass.loc[ai_mask] = df.loc[ai_mask, a_att] == 6

n_fail_h = (~att_pass.loc[human_mask]).sum()
n_fail_a = (~att_pass.loc[ai_mask]).sum()
print(f"\nAttention check failures: {(~att_pass).sum()} total  "
      f"(Human: {n_fail_h}, AI: {n_fail_a})")

df = df[att_pass].copy()
n_human = (df['condition'] == 'human').sum()
n_ai = (df['condition'] == 'ai').sum()
print(f"\n{'='*70}")
print(f"final analytic sample: n = {len(df)}  (human: {n_human}, ai: {n_ai})")
print(f"{'='*70}")
for i in range(3):
    df[f'emo_{i+1}'] = np.where(df['condition'] == 'human', df[h_emo[i]], df[a_emo[i]])
    df[f'inf_{i+1}'] = np.where(df['condition'] == 'human', df[h_inf[i]], df[a_inf[i]])
    df[f'cer_{i+1}'] = np.where(df['condition'] == 'human', df[h_cer[i]], df[a_cer[i]])
    df[f'est_{i+1}'] = np.where(df['condition'] == 'human', df[h_est[i]], df[a_est[i]])

for col in [f'{t}_{i}' for t in ['emo', 'inf', 'cer', 'est'] for i in [1, 2, 3]]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['emotional']     = df[['emo_1', 'emo_2', 'emo_3']].mean(axis=1)
df['informational'] = df[['inf_1', 'inf_2', 'inf_3']].mean(axis=1)
df['certainty']     = df[['cer_1', 'cer_2', 'cer_3']].mean(axis=1)
df['esteem']        = df[['est_1', 'est_2', 'est_3']].mean(axis=1)

support_types = ['emotional', 'esteem', 'informational', 'certainty']
item_cols = {
    'emotional':     ['emo_1', 'emo_2', 'emo_3'],
    'esteem':        ['est_1', 'est_2', 'est_3'],
    'informational': ['inf_1', 'inf_2', 'inf_3'],
    'certainty':     ['cer_1', 'cer_2', 'cer_3'],
}

print("\n" + "=" * 70)
print("descriptive statistics & internal consistency")
print("=" * 70)

for st in support_types:
    alpha_val = pg.cronbach_alpha(df[item_cols[st]])[0]
    print(f"\n{st.upper()} (Cronbach's α = {alpha_val:.3f})")
    for cond in ['human', 'ai']:
        sub = df[df['condition'] == cond][st]
        print(f"  {cond:6s}: M = {sub.mean():.2f}, SD = {sub.std():.2f}, n = {len(sub)}")

print("\n--- Composite means by condition ---")
summary = df.groupby('condition')[support_types].agg(['mean', 'std']).round(2)
print(summary.to_string())

print("\n" + "=" * 70)
print("primary analysis: mixed anova")
print("Source (between: human vs AI) x Support Type (within: 4 levels)")
print("=" * 70)

df['pid'] = range(len(df))
long = df.melt(
    id_vars=['pid', 'condition'],
    value_vars=support_types,
    var_name='support_type',
    value_name='importance',
)

aov = pg.mixed_anova(
    data=long, dv='importance', between='condition',
    within='support_type', subject='pid',
)
print(aov.round(4).to_string(index=False))

spher = pg.sphericity(data=long, dv='importance', within='support_type', subject='pid')
print(f"\nMauchly's sphericity: W = {spher[1]:.4f}, p = {spher[2]:.4f}")
if spher[2] < 0.05:
    print("  -> Sphericity violated; use GG-corrected p-values above.")

print("\n" + "=" * 70)
print("simple effects: human vs ai per support type")
print("=" * 70)

gaps = {}
for st in support_types:
    h = df[df['condition'] == 'human'][st]
    a = df[df['condition'] == 'ai'][st]
    t_stat, p_val = stats.ttest_ind(h, a)
    d = pg.compute_effsize(h, a, eftype='cohen')
    diff = h.mean() - a.mean()
    gaps[st] = {'diff': diff, 'd': d, 't': t_stat, 'p': p_val}
    print(f"\n{st.upper()}")
    print(f"  Human: M = {h.mean():.2f}, SD = {h.std():.2f}")
    print(f"  AI:    M = {a.mean():.2f}, SD = {a.std():.2f}")
    print(f"  Diff (H-AI) = {diff:+.2f},  t({len(h)+len(a)-2}) = {t_stat:.3f},  "
          f"p = {p_val:.4f},  Cohen's d = {d:.3f}")


print("\n" + "=" * 70)
print("planned interaction contrasts (s2)")
print("Within-person: support_type_A - informational, then between conditions.")
print("Preregistered: emotional - informational, esteem - informational.")
print("Exploratory:    certainty - informational.")
print("=" * 70)

contrasts = [
    ('emotional', 'informational', 'Emotional gap vs Informational gap  [preregistered]'),
    ('esteem',    'informational', 'Esteem gap vs Informational gap     [preregistered]'),
    ('certainty', 'informational', 'Certainty gap vs Informational gap  [exploratory]'),
]
for st_a, st_b, label in contrasts:
    df[f'{st_a}_minus_{st_b}'] = df[st_a] - df[st_b]
    h_c = df[df['condition'] == 'human'][f'{st_a}_minus_{st_b}']
    a_c = df[df['condition'] == 'ai'][f'{st_a}_minus_{st_b}']
    t, p = stats.ttest_ind(h_c, a_c)
    d = pg.compute_effsize(h_c, a_c, eftype='cohen')
    print(f"\n{label}")
    print(f"  Human ({st_a}-{st_b}): M = {h_c.mean():.2f}")
    print(f"  AI    ({st_a}-{st_b}): M = {a_c.mean():.2f}")
    print(f"  Interaction contrast:  t = {t:.3f}, p = {p:.4f}, d = {d:.3f}")

print("\n" + "=" * 70)
print("exploratory: item-level human-ai gaps")
print("=" * 70)

item_labels = {
    'emo_1': "Understand what I'm going through",
    'emo_2': 'Show genuine care',
    'emo_3': 'Make me feel heard',
    'est_1': 'Remind me what I am doing well',
    'est_2': 'Agree with how I see the situation',
    'est_3': 'Feel less guilty or at fault',
    'inf_1': 'See from a different angle',
    'inf_2': 'Practical advice',
    'inf_3': 'Point out things not considering',
    'cer_1': 'More sure about what to do',
    'cer_2': 'Clearer about next steps',
    'cer_3': 'More confident in decision',
}
support_lookup = {'emo': 'emotional', 'est': 'esteem',
                  'inf': 'informational', 'cer': 'certainty'}

print(f"\n{'Item':<38s} {'M_H':>5s} {'M_AI':>5s} {'Diff':>6s} {'d':>7s} {'p':>8s}")
print("-" * 75)

item_rows = []
for item, label in item_labels.items():
    h = df[df['condition'] == 'human'][item].astype(float)
    a = df[df['condition'] == 'ai'][item].astype(float)
    d = pg.compute_effsize(h, a, eftype='cohen')
    t, p = stats.ttest_ind(h, a)
    print(f"{label:<38s} {h.mean():5.2f} {a.mean():5.2f} "
          f"{h.mean()-a.mean():+6.2f} {d:+7.3f} {p:8.4f}")
    item_rows.append({
        'support_type': support_lookup[item.split('_')[0]],
        'item_code': item,
        'item_label': label,
        'n_human': len(h),
        'mean_human': round(h.mean(), 3),
        'sd_human': round(h.std(), 3),
        'n_ai': len(a),
        'mean_ai': round(a.mean(), 3),
        'sd_ai': round(a.std(), 3),
        'mean_diff_h_minus_ai': round(h.mean() - a.mean(), 3),
        'cohens_d': round(d, 3),
        't_stat': round(t, 3),
        'p_value': round(p, 4),
    })

print("\n" + "=" * 70)
print("exploratory: ai usage moderation (ai arm only)")
print("=" * 70)

ai_df = df[df['condition'] == 'ai'].copy()
usage_map = {'Daily': 5, 'Weekly': 4, 'Monthly': 3, 'Rarely': 2, 'Never': 1}
ai_df['ai_usage_num'] = ai_df['Q149'].map(usage_map)
print(f"\nAI usage frequency distribution (n = {len(ai_df)}):")
print(ai_df['Q149'].value_counts().sort_index().to_string())

valid = ai_df.dropna(subset=['ai_usage_num'])
if len(valid) >= 5:
    print(f"\nCorrelation with support expectations:")
    for st in support_types:
        r, p = stats.pearsonr(valid['ai_usage_num'], valid[st])
        print(f"  {st:<15s}: r = {r:+.3f}, p = {p:.4f}")
else:
    print(f"\n  Too few valid cases ({len(valid)}) for correlation analysis.")

print("\n" + "=" * 70)
print("exploratory: relationship type moderation (human arm only)")
print("=" * 70)

human_df = df[df['condition'] == 'human'].copy()
print(f"\nRelationship type distribution (n = {len(human_df)}):")
print(human_df['relation-type'].value_counts().to_string())

for st in support_types:
    groups = {name: grp[st].values for name, grp in human_df.groupby('relation-type') if len(grp) >= 2}
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups.values())
        print(f"\n  {st:<15s}: F({len(groups)-1},{len(human_df)-len(groups)}) "
              f"= {f_stat:.3f}, p = {p_val:.4f}")
        for name, vals in groups.items():
            print(f"    {name:<20s}: M = {vals.mean():.2f}, SD = {vals.std():.2f}, n = {len(vals)}")
    else:
        print(f"\n  {st:<15s}: Too few relationship types with n >= 2 for ANOVA.")

print("\n" + "=" * 70)
print("exploratory: within-source differentiation")
print("Do people differentiate among support types within each condition?")
print("=" * 70)

for cond in ['human', 'ai']:
    sub = df[df['condition'] == cond].copy()
    sub['pid_local'] = range(len(sub))
    sub_long = sub.melt(
        id_vars=['pid_local'],
        value_vars=support_types,
        var_name='support_type',
        value_name='importance',
    )
    rm_aov = pg.rm_anova(
        data=sub_long, dv='importance',
        within='support_type', subject='pid_local',
    )
    print(f"\n{cond.upper()} condition - Repeated-measures ANOVA:")
    print(rm_aov.round(4).to_string(index=False))

    pw = pg.pairwise_tests(
        data=sub_long, dv='importance',
        within='support_type', subject='pid_local',
        padjust='none',
    )
    print(f"\n  Pairwise comparisons (uncorrected):")
    cols_show = ['Contrast', 'A', 'B', 'T', 'dof', 'p-unc', 'cohen']
    cols_avail = [c for c in cols_show if c in pw.columns]
    print(pw[cols_avail].round(4).to_string(index=False))

print("\n" + "=" * 70)
print("sample descriptives")
print("=" * 70)

invest_map = {'None at all': 1, 'A little': 2, 'A moderate amount': 3,
              'A lot': 4, 'A great deal': 5}
df['investment_unified'] = df['investment'].map(invest_map)
df.loc[df['condition'] == 'ai', 'investment_unified'] = (
    df.loc[df['condition'] == 'ai', 'Q142'].map(invest_map)
)

print(f"\nInvestment (situation on mind):")
for cond in ['human', 'ai']:
    vals = df[df['condition'] == cond]['investment_unified'].dropna()
    if len(vals) > 0:
        print(f"  {cond:6s}: M = {vals.mean():.2f}, SD = {vals.std():.2f}, n = {len(vals)}")

human_df = df[df['condition'] == 'human'].copy()
human_df['ios_num'] = pd.to_numeric(human_df['ios'], errors='coerce')
ios = human_df['ios_num'].dropna()
if len(ios) > 0:
    print(f"\nIOS closeness (human only): M = {ios.mean():.2f}, SD = {ios.std():.2f}, n = {len(ios)}")

print("\n" + "=" * 70)
print("analysis complete")
print("=" * 70)
