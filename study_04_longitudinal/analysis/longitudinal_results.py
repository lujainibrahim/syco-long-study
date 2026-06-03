"""Analysis of Study 4 hypothesis tests reported in the main body of the paper."""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../data/main_sessions.csv")

# basic cleaning
df = df[df['Finished'].astype(str).str.strip().str.lower() == 'true'].copy()
df = df.drop_duplicates(subset=['participant_id', 'SESSION_NUM'], keep='first')

pid_condition = (
    df[['participant_id', 'model_condition']]
    .drop_duplicates()
    .reset_index(drop=True)
)

# session numeric map (all 12 sessions)
session_num_map = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
    'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12
}
df['session_numeric'] = df['SESSION_NUM'].map(session_num_map)

# pre-treatment survey
pre_session = pd.read_csv("../data/pre_treatment.csv")
pre_session['SESSION_NUM'] = 'pre'
pre_session['session_numeric'] = 0

# weekly time map for 4-session models (pre, four, eight, twelve)
time_map_weekly = {'pre': 0, 'four': 1, 'eight': 2, 'twelve': 3}

results_rows = []

def run_mixed_model(data, formula, groups, condition_coef, analysis_label, outcome_label):
    data = data.reset_index(drop=True)
    data['participant_id'] = data['participant_id'].astype(str)

    model = smf.mixedlm(formula, data=data, groups=data['participant_id'],
                        re_formula='1').fit(reml=True)
    print(model.summary())

    b = model.fe_params[condition_coef]
    se = model.bse[condition_coef]
    p = model.pvalues[condition_coef]
    ci_lo = b - 1.96 * se
    ci_hi = b + 1.96 * se
    outcome_col = formula.split('~')[0].strip()
    between_pid_sd = data.groupby('participant_id')[outcome_col].mean().std(ddof=1)
    d = b / between_pid_sd
    d_lo = ci_lo / between_pid_sd
    d_hi = ci_hi / between_pid_sd
    n_obs = int(model.nobs)

    row = {
        'analysis': analysis_label,
        'outcome': outcome_label,
        'cohens_d': d,
        'ci_lower': d_lo,
        'ci_upper': d_hi,
        'p_value': p,
        'b': b,
        'b_ci_lower': ci_lo,
        'b_ci_upper': ci_hi,
        'n_obs': n_obs,
    }
    results_rows.append(row)
    print(f"  {analysis_label} | {outcome_label}: d={d:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}], p={p:.4f}\n")
    return model

# Likert mappings
LIKERT_7 = {
    'Extremely': 7, 'Very much': 6, 'Quite a bit': 5,
    'Moderately': 4, 'Somewhat': 3, 'Slightly': 2, 'Not at all': 1
}

LIKERT_AGREE = {
    'Strongly disagree': 1, 'Disagree': 2, 'Somewhat disagree': 3,
    'Neither agree nor disagree': 4, 'Somewhat agree': 5,
    'Agree': 6, 'Strongly agree': 7
}

CERTAINTY_MAP = {
    'Very uncertain': 1, 'Somewhat uncertain': 2, 'Uncertain': 3,
    'Neither certain nor uncertain': 4, 'Certain': 5,
    'Somewhat certain': 6, 'Very certain': 7
}

AFFECT_MAP = {
    'Much worse than before': 1, 'Somewhat worse than before': 2,
    'Slightly worse than before': 3, 'About the same': 4,
    'Slightly better than before': 5, 'Somewhat better than before': 6,
    'Much better than before': 7
}

HELPFULNESS_MAP = {
    'Not at all helpful': 1, 'Slightly helpful': 2, 'Somewhat helpful': 3,
    'Moderately helpful': 4, 'Quite helpful': 5, 'Very helpful': 6,
    'Extremely helpful': 7
}

SOCIAL_TIME_MAP = {
    '0 days': 0, '1-2 days': 1.5, '3-4 days': 3.5,
    '5-6 days': 5.5, 'every day (7 days)': 7
}

SOCIAL_SAT_MAP = {
    'Very dissatisfied': 1, 'Dissatisfied': 2, 'Somewhat dissatisfied': 3,
    'Neither satisfied nor dissatisfied': 4, 'Somewhat satisfied': 5,
    'Satisfied': 6, 'Very satisfied': 7
}

print("=" * 65)
print("  1. comfort ai vs humans")
print("=" * 65)

weekly_sessions = ['four', 'eight', 'twelve']
weekly_df = df[df['SESSION_NUM'].isin(weekly_sessions)][
    ['comfort_1', 'comfort_2', 'comfort_3', 'participant_id', 'SESSION_NUM', 'session_numeric']
].copy()

def add_comfort(frame):
    c = frame[['comfort_1', 'comfort_2', 'comfort_3']].apply(
        lambda x: x.str.strip() if x.dtype == 'object' else x)
    c = c.replace(LIKERT_7).apply(pd.to_numeric, errors='coerce')
    frame['comfort_humans'] = c[['comfort_1', 'comfort_2']].mean(axis=1)
    frame['comfort_AI'] = c['comfort_3']
    frame['comfort_AI_minus_humans'] = frame['comfort_AI'] - frame['comfort_humans']
    return frame

weekly_df = add_comfort(weekly_df)
pre_comfort = add_comfort(
    pre_session[['comfort_1', 'comfort_2', 'comfort_3', 'participant_id', 'SESSION_NUM', 'session_numeric']].copy()
)

comfort_data = pd.concat([
    weekly_df[['participant_id', 'SESSION_NUM', 'comfort_AI_minus_humans', 'session_numeric',
               'comfort_humans', 'comfort_AI']],
    pre_comfort[['participant_id', 'SESSION_NUM', 'comfort_AI_minus_humans', 'session_numeric',
                 'comfort_humans', 'comfort_AI']]
], ignore_index=True)
comfort_data = comfort_data.merge(pid_condition, how='left', on='participant_id')
comfort_data = comfort_data.dropna(subset=['participant_id'])

# syco vs balanced
comfort_bal = comfort_data[comfort_data['model_condition'].isin(['balanced', 'syco'])].copy()
comfort_bal['time_num'] = comfort_bal['SESSION_NUM'].map(time_map_weekly)
comfort_bal = comfort_bal[comfort_bal['time_num'] > 0].copy()
comfort_bal = comfort_bal.dropna(
    subset=['comfort_AI_minus_humans', 'time_num', 'model_condition', 'participant_id'])
comfort_bal['time_c'] = comfort_bal['time_num'] - comfort_bal['time_num'].mean()
comfort_bal['model_condition'] = pd.Categorical(
    comfort_bal['model_condition'], categories=['balanced', 'syco'])

print("\n--- Syco vs Balanced ---")
run_mixed_model(
    comfort_bal,
    'comfort_AI_minus_humans ~ time_c * model_condition',
    'participant_id',
    'model_condition[T.syco]',
    'Comfort AI vs Humans (syco vs balanced)',
    'comfort_AI_minus_humans'
)

# syco vs challenging
comfort_chal = comfort_data[comfort_data['model_condition'].isin(['challenging', 'syco'])].copy()
comfort_chal['time_num'] = comfort_chal['SESSION_NUM'].map(time_map_weekly)
comfort_chal = comfort_chal[comfort_chal['time_num'] > 0].copy()
comfort_chal = comfort_chal.dropna(
    subset=['comfort_AI_minus_humans', 'time_num', 'model_condition', 'participant_id'])
comfort_chal['time_c'] = comfort_chal['time_num'] - comfort_chal['time_num'].mean()
comfort_chal['model_condition'] = pd.Categorical(
    comfort_chal['model_condition'], categories=['challenging', 'syco'])

print("\n--- Syco vs Challenging ---")
run_mixed_model(
    comfort_chal,
    'comfort_AI_minus_humans ~ time_c * model_condition',
    'participant_id',
    'model_condition[T.syco]',
    'Comfort AI vs Humans (syco vs challenging)',
    'comfort_AI_minus_humans'
)

print("=" * 65)
print("  2. session-level measures (affect, helpfulness, certainty, fk ai)")
print("=" * 65)

# feeling known AI
fk_cols = ['closeness-ai_1', 'closeness-ai_2', 'closeness-ai_3', 'closeness-ai_4']
for col in fk_cols:
    df[col + '_num'] = df[col].replace(LIKERT_AGREE).astype(float)
df['feeling_known_ai'] = df[[c + '_num' for c in fk_cols]].mean(axis=1)

df['certainty_num'] = df['certainty'].replace(CERTAINTY_MAP).astype(float)
df['affect_num'] = df['affect'].replace(AFFECT_MAP).astype(float)
df['helpfulness_num'] = df['helpfulness'].replace(HELPFULNESS_MAP).astype(float)

# session numeric for all 12
df['session_num_numeric'] = df['SESSION_NUM'].map(session_num_map)

df_llm = df[df['model_condition'].isin(['balanced', 'syco', 'challenging'])].copy()
df_llm['model_condition'] = pd.Categorical(
    df_llm['model_condition'], categories=['balanced', 'syco', 'challenging'])
df_llm['time_c'] = df_llm['session_num_numeric'] - df_llm['session_num_numeric'].mean()

session_measures = [
    ('affect_num', 'Affect'),
    ('helpfulness_num', 'Helpfulness'),
    ('certainty_num', 'Certainty'),
    ('feeling_known_ai', 'Feeling Known (AI)'),
]

for var, label in session_measures:
    sub = df_llm.dropna(subset=[var]).copy()
    print(f"\n--- {label} ---")
    run_mixed_model(
        sub,
        f'{var} ~ time_c * model_condition',
        'participant_id',
        'model_condition[T.syco]',
        'Session-level (syco vs balanced)',
        label
    )

# syco vs challenging: re-level with challenging as reference
df_llm_chal = df_llm.copy()
df_llm_chal['model_condition'] = pd.Categorical(
    df_llm_chal['model_condition'], categories=['challenging', 'syco', 'balanced'])

print("\n" + "-" * 65)
print("  session-level: syco vs challenging")
print("-" * 65)

for var, label in session_measures:
    sub = df_llm_chal.dropna(subset=[var]).copy()
    print(f"\n--- {label} ---")
    run_mixed_model(
        sub,
        f'{var} ~ time_c * model_condition',
        'participant_id',
        'model_condition[T.syco]',
        'Session-level (syco vs challenging)',
        label
    )

# 3 & 4. SOCIAL TIME & SOCIAL SATISFACTION
print("=" * 65)
print("  3 & 4. social time / social satisfaction")
print("=" * 65)

soc_weekly = df[df['SESSION_NUM'].isin(weekly_sessions)].copy()
soc_weekly['social_time_num'] = soc_weekly['social-time'].map(SOCIAL_TIME_MAP)
soc_weekly['social_sat_num'] = soc_weekly['social-sat'].map(SOCIAL_SAT_MAP)

social_data = soc_weekly[['participant_id', 'SESSION_NUM', 'session_numeric',
                           'social_time_num', 'social_sat_num']].copy()
social_data = social_data.merge(pid_condition, on='participant_id', how='left')
social_data['time_num'] = social_data['SESSION_NUM'].map({'four': 1, 'eight': 2, 'twelve': 3})

for var, label in [('social_time_num', 'Social Time'), ('social_sat_num', 'Social Satisfaction')]:
    sub = social_data[social_data['model_condition'].isin(['balanced', 'syco'])].copy()
    sub = sub.dropna(subset=[var])
    sub['time_c'] = sub['time_num'] - sub['time_num'].mean()
    sub['model_condition'] = pd.Categorical(sub['model_condition'], categories=['balanced', 'syco'])

    print(f"\n--- {label} (syco vs balanced) ---")
    run_mixed_model(
        sub,
        f'{var} ~ time_c * model_condition',
        'participant_id',
        'model_condition[T.syco]',
        'Social (syco vs balanced)',
        label
    )

print("=" * 65)
print("  5. feeling known ai - humans")
print("=" * 65)

fk_human_cols = ['closeness-humans_1', 'closeness-humans_2',
                 'closeness-humans_3', 'closeness-humans_4']

# build from weekly sessions + pre
weekly_fk = df[df['SESSION_NUM'].isin(weekly_sessions)].copy()
for col in fk_human_cols:
    weekly_fk[col + '_num'] = weekly_fk[col].replace(LIKERT_AGREE).astype(float)
weekly_fk['feeling_known_humans'] = weekly_fk[[c + '_num' for c in fk_human_cols]].mean(axis=1)

# pre session FK humans
for col in fk_human_cols:
    if col in pre_session.columns:
        pre_session[col + '_num'] = pre_session[col].replace(LIKERT_AGREE).astype(float)
if all(c + '_num' in pre_session.columns for c in fk_human_cols):
    pre_session['feeling_known_humans'] = pre_session[[c + '_num' for c in fk_human_cols]].mean(axis=1)


fk_ai_cols_pre = ['closeness-ai_1', 'closeness-ai_2', 'closeness-ai_3', 'closeness-ai_4']
for col in fk_ai_cols_pre:
    if col in pre_session.columns:
        pre_session[col + '_num'] = pre_session[col].replace(LIKERT_AGREE).astype(float)
if all(c + '_num' in pre_session.columns for c in fk_ai_cols_pre):
    pre_session['feeling_known_ai'] = pre_session[[c + '_num' for c in fk_ai_cols_pre]].mean(axis=1)

cols_needed = ['participant_id', 'SESSION_NUM', 'session_numeric',
               'feeling_known_ai', 'feeling_known_humans']

known_data_parts = []
for part in [weekly_fk, pre_session]:
    if all(c in part.columns for c in cols_needed):
        known_data_parts.append(part[cols_needed].copy())

if known_data_parts:
    known_data = pd.concat(known_data_parts, ignore_index=True)
    known_data = known_data.merge(pid_condition, how='left', on='participant_id')
    known_data['AI_minus_humans'] = known_data['feeling_known_ai'] - known_data['feeling_known_humans']

    # syco vs balanced
    kn_bal = known_data[known_data['model_condition'].isin(['balanced', 'syco'])].copy()
    kn_bal['time_num'] = kn_bal['SESSION_NUM'].map(time_map_weekly)
    kn_bal = kn_bal[kn_bal['time_num'] > 0].copy()
    kn_bal = kn_bal.dropna(subset=['AI_minus_humans', 'time_num', 'model_condition', 'participant_id'])
    kn_bal['time_c'] = kn_bal['time_num'] - kn_bal['time_num'].mean()
    kn_bal['model_condition'] = pd.Categorical(
        kn_bal['model_condition'], categories=['balanced', 'syco'])

    print("\n--- Syco vs Balanced ---")
    run_mixed_model(
        kn_bal,
        'AI_minus_humans ~ time_c * model_condition',
        'participant_id',
        'model_condition[T.syco]',
        'FK AI minus Humans (syco vs balanced)',
        'AI_minus_humans'
    )

    # syco vs challenging
    kn_chal = known_data[known_data['model_condition'].isin(['challenging', 'syco'])].copy()
    kn_chal['time_num'] = kn_chal['SESSION_NUM'].map(time_map_weekly)
    kn_chal = kn_chal[kn_chal['time_num'] > 0].copy()
    kn_chal = kn_chal.dropna(subset=['AI_minus_humans', 'time_num', 'model_condition', 'participant_id'])
    kn_chal['time_c'] = kn_chal['time_num'] - kn_chal['time_num'].mean()
    kn_chal['model_condition'] = pd.Categorical(
        kn_chal['model_condition'], categories=['challenging', 'syco'])

    print("\n--- Syco vs Challenging ---")
    run_mixed_model(
        kn_chal,
        'AI_minus_humans ~ time_c * model_condition',
        'participant_id',
        'model_condition[T.syco]',
        'FK AI minus Humans (syco vs challenging)',
        'AI_minus_humans'
    )
else:
    print("  ⚠ Could not build feeling-known data – columns missing from pre-session.")

print("=" * 65)
print("  6. feeling known by humans")
print("=" * 65)

fkh = weekly_fk.dropna(subset=['feeling_known_humans']).copy()
fkh['time_num'] = fkh['SESSION_NUM'].map(time_map_weekly)
fkh = fkh[fkh['time_num'] > 0].copy()

for ref in ['balanced', 'challenging']:
    sub = fkh[fkh['model_condition'].isin([ref, 'syco'])].copy()
    sub['time_c'] = sub['time_num'] - sub['time_num'].mean()
    sub['model_condition'] = pd.Categorical(sub['model_condition'], categories=[ref, 'syco'])
    sub = sub.dropna(subset=['feeling_known_humans', 'time_c', 'model_condition', 'participant_id'])

    print(f"\n--- Syco vs {ref.capitalize()} ---")
    run_mixed_model(
        sub,
        'feeling_known_humans ~ time_c * model_condition',
        'participant_id',
        'model_condition[T.syco]',
        f'FK Humans (syco vs {ref})',
        'feeling_known_humans',
    )

nollm = pd.read_csv("../data/no_llm_control.csv")

# Likert / scale maps used by the IH and SE
AGREE_7_LOWER = {
    "strongly disagree": 1, "disagree": 2, "somewhat disagree": 3,
    "neither agree nor disagree": 4, "somewhat agree": 5,
    "agree": 6, "strongly agree": 7,
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

def _to_num(series, mapping):
    return series.astype(str).str.strip().str.lower().map(mapping).astype(float)

def _composite(df_, cols, mapping):
    return pd.DataFrame({c: _to_num(df_[c], mapping) for c in cols}).mean(axis=1)

def run_ols_ancova(data, formula, condition_coef, analysis_label, outcome_label):
    data = data.reset_index(drop=True)
    model = smf.ols(formula, data=data).fit()
    if condition_coef not in model.params.index:
        for k in model.params.index:
            if "syco" in k.lower():
                condition_coef = k
                break
    b = model.params[condition_coef]
    se = model.bse[condition_coef]
    p = model.pvalues[condition_coef]
    ci_lo, ci_hi = b - 1.96 * se, b + 1.96 * se
    outcome_col = formula.split('~')[0].strip()
    _groups = [g[outcome_col].dropna() for _, g in data.groupby("model_condition")
               if g[outcome_col].dropna().size >= 2]
    if len(_groups) >= 2:
        _num = sum((g.size - 1) * g.var(ddof=1) for g in _groups)
        _den = sum(g.size for g in _groups) - len(_groups)
        pooled_sd = float(np.sqrt(_num / _den))
    else:
        pooled_sd = float(data[outcome_col].std(ddof=1))
    d = b / pooled_sd
    d_lo, d_hi = ci_lo / pooled_sd, ci_hi / pooled_sd
    n_obs = int(model.nobs)
    results_rows.append({
        'analysis': analysis_label, 'outcome': outcome_label,
        'cohens_d': d, 'ci_lower': d_lo, 'ci_upper': d_hi,
        'p_value': p, 'b': b, 'b_ci_lower': ci_lo, 'b_ci_upper': ci_hi,
        'n_obs': n_obs,
    })
    print(f"  {analysis_label} | {outcome_label}: "
          f"d={d:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}], p={p:.4f}, n={n_obs}")
    return model

# Session-12 post-treatment for LLM conditions
s12 = df[df['SESSION_NUM'] == 'twelve'].copy()

# Assign no-LLM rows with condition = "nollm"
nollm['model_condition'] = 'nollm'

# Add condition to pre_session for no-LLM participants
nollm_pids = set(nollm['participant_id'].dropna().unique())
if 'model_condition' not in pre_session.columns:
    pre_session['model_condition'] = np.nan
for pid in nollm_pids:
    mask = pre_session['participant_id'] == pid
    if mask.any():
        pre_session.loc[mask, 'model_condition'] = 'nollm'

# H2 (syco vs no-LLM): pre-post ANCOVA on feeling known by humans.
fk_human_cols = ['closeness-humans_1', 'closeness-humans_2',
                 'closeness-humans_3', 'closeness-humans_4']
pre_session['FK_humans_pre'] = _composite(pre_session, fk_human_cols, AGREE_7_LOWER)
s12['FK_humans_post']        = _composite(s12,         fk_human_cols, AGREE_7_LOWER)
nollm['FK_humans_post']      = _composite(nollm,       fk_human_cols, AGREE_7_LOWER)

fk_syco  = s12[s12['model_condition'] == 'syco'][['participant_id', 'FK_humans_post']].copy()
fk_syco['model_condition'] = 'syco'
fk_nollm = nollm[['participant_id', 'FK_humans_post']].copy()
fk_nollm['model_condition'] = 'nollm'

fk_h2 = pd.concat([fk_syco, fk_nollm], ignore_index=True)
fk_h2 = fk_h2.merge(pre_session[['participant_id', 'FK_humans_pre']], on='participant_id', how='left')
fk_h2 = fk_h2.dropna(subset=['FK_humans_post', 'FK_humans_pre', 'model_condition'])
fk_h2['model_condition'] = pd.Categorical(fk_h2['model_condition'], categories=['nollm', 'syco'])
run_ols_ancova(fk_h2, 'FK_humans_post ~ FK_humans_pre + model_condition',
               'model_condition[T.syco]',
               'H2: syco vs no-LLM', 'Feeling Known by Humans (ANCOVA)')

print("=" * 65)
print("  7. intellectual humility (h3, pre-post ancova)")
print("=" * 65)

ih_pre_cols  = ['SIH_1', 'SIH_2', 'SIH_3']
ih_post_cols = ['SIHS_1', 'SIHS_2', 'SIHS_3']

pre_session['IH_pre'] = _composite(pre_session, ih_pre_cols, AGREE_7_LOWER)
s12['IH_post']        = _composite(s12,         ih_post_cols, AGREE_7_LOWER)
nollm['IH_post']      = _composite(nollm,       ih_post_cols, AGREE_7_LOWER)

ih_post = pd.concat([
    s12[['participant_id', 'IH_post', 'model_condition']],
    nollm[['participant_id', 'IH_post', 'model_condition']],
], ignore_index=True)
ih = ih_post.merge(pre_session[['participant_id', 'IH_pre']], on='participant_id', how='left')
ih = ih.dropna(subset=['IH_post', 'IH_pre', 'model_condition'])

for ref, target, label in [
    ('challenging', 'syco', 'H3: syco vs challenging (primary)'),
    ('balanced',    'syco', 'H3: syco vs balanced'),
    ('nollm',       'syco', 'H3: syco vs no-LLM'),
]:
    sub = ih[ih['model_condition'].isin([ref, target])].copy()
    sub['model_condition'] = pd.Categorical(sub['model_condition'], categories=[ref, target])
    run_ols_ancova(sub, 'IH_post ~ IH_pre + model_condition', 'model_condition[T.syco]',
                   label, 'Intellectual Humility')

print("=" * 65)
print("  8. self-enhancement (h4, pre-post ancova)")
print("=" * 65)

avg_cols = [f'average_{i}'      for i in range(1, 6)]
oth_cols = [f'other-people_{i}' for i in range(1, 6)]

pre_session['SE_avg_pre'] = _composite(pre_session, avg_cols, AVERAGE_MAP)
pre_session['SE_oth_pre'] = _composite(pre_session, oth_cols, OTHER_PEOPLE_MAP)
s12['SE_avg_post']        = _composite(s12,         avg_cols, AVERAGE_MAP)
s12['SE_oth_post']        = _composite(s12,         oth_cols, OTHER_PEOPLE_MAP)
nollm['SE_avg_post']      = _composite(nollm,       avg_cols, AVERAGE_MAP)
nollm['SE_oth_post']      = _composite(nollm,       oth_cols, OTHER_PEOPLE_MAP)

se_post = pd.concat([
    s12[['participant_id', 'SE_avg_post', 'SE_oth_post', 'model_condition']],
    nollm[['participant_id', 'SE_avg_post', 'SE_oth_post', 'model_condition']],
], ignore_index=True)
se = se_post.merge(pre_session[['participant_id', 'SE_avg_pre', 'SE_oth_pre']],
                   on='participant_id', how='left')
se = se.dropna(subset=['SE_avg_post', 'SE_oth_post', 'SE_avg_pre', 'SE_oth_pre', 'model_condition'])

for ref, target, label in [
    ('challenging', 'syco', 'H4: syco vs challenging (primary)'),
    ('balanced',    'syco', 'H4: syco vs balanced'),
    ('nollm',       'syco', 'H4: syco vs no-LLM'),
]:
    sub = se[se['model_condition'].isin([ref, target])].copy()
    sub['model_condition'] = pd.Categorical(sub['model_condition'], categories=[ref, target])
    run_ols_ancova(sub, 'SE_avg_post ~ SE_avg_pre + model_condition', 'model_condition[T.syco]',
                   label, 'Self-Enhancement vs Average')
    run_ols_ancova(sub, 'SE_oth_post ~ SE_oth_pre + model_condition', 'model_condition[T.syco]',
                   label, 'Self-Enhancement vs Others')
results_df = pd.DataFrame(results_rows)
col_order = ['analysis', 'outcome', 'cohens_d', 'ci_lower', 'ci_upper',
             'p_value', 'b', 'b_ci_lower', 'b_ci_upper', 'n_obs']
for c in col_order:
    if c not in results_df.columns:
        results_df[c] = np.nan
results_df = results_df[col_order]

