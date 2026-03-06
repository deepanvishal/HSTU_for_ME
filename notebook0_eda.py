# ============================================================
# NOTEBOOK 0: EDA — EVIDENCE OF PREDICTABLE PATTERNS
# Purpose: Justify to stakeholders that next-visit prediction
#          has signal worth modeling
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from collections import Counter, defaultdict
from scipy.stats import entropy
from google.cloud import bigquery

os.makedirs('./eda', exist_ok=True)

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
PLOT_TOP_N = 20
PROJECT    = 'anbc-hcb-dev'
DATASET    = 'provider_ds_netconf_data_hcb_dev'

client = bigquery.Client(project=PROJECT)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("Loading visit sequences...")
df_seq = client.query(f"""
    SELECT
        member_id
        ,visit_seq_num
        ,visit_date
        ,delta_t_bucket
        ,specialty_codes
        ,dx_list
    FROM `{PROJECT}.{DATASET}.A870800_claims_gen_rec_visit_sequence`
    ORDER BY member_id, visit_seq_num
""").to_dataframe(create_bqstorage_client=True)

print(f"Total visit rows: {len(df_seq):,}")
print(f"Unique members:   {df_seq['member_id'].nunique():,}")

# ============================================================
# STEP 2: BASIC SEQUENCE STATS
# ============================================================
print("\n--- SEQUENCE STATS ---")

seq_lengths = df_seq.groupby('member_id')['visit_seq_num'].count()

print(f"Avg visits per member:    {seq_lengths.mean():.1f}")
print(f"Median visits per member: {seq_lengths.median():.1f}")
print(f"Min visits:               {seq_lengths.min()}")
print(f"Max visits:               {seq_lengths.max()}")
print(f"Members with 1 visit:     {(seq_lengths == 1).sum():,} ({(seq_lengths == 1).mean()*100:.1f}%)")
print(f"Members with 2+ visits:   {(seq_lengths >= 2).sum():,} ({(seq_lengths >= 2).mean()*100:.1f}%)")
print(f"Members with 5+ visits:   {(seq_lengths >= 5).sum():,} ({(seq_lengths >= 5).mean()*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

seq_lengths.clip(upper=30).hist(bins=30, ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of Visits per Member (capped at 30)')
axes[0].set_xlabel('Number of Visits')
axes[0].set_ylabel('Number of Members')

seq_lengths[seq_lengths >= 2].clip(upper=30).hist(bins=30, ax=axes[1], color='seagreen', edgecolor='white')
axes[1].set_title('Members with 2+ Visits (predictable population)')
axes[1].set_xlabel('Number of Visits')
axes[1].set_ylabel('Number of Members')

plt.tight_layout()
plt.savefig('./eda/01_sequence_length_distribution.png', dpi=150)
plt.close()
print("Saved: 01_sequence_length_distribution.png")

# ============================================================
# STEP 3: DELTA_T DISTRIBUTION — IS TIMING PREDICTABLE?
# ============================================================
print("\n--- DELTA_T ANALYSIS ---")

# bucket boundaries (approximate days)
bucket_labels = {
    0: '0-1d', 1: '2-3d', 2: '4-7d', 3: '8-14d',
    4: '15-30d', 5: '31-60d', 6: '61-90d', 7: '91-180d',
    8: '181-365d', 9: '366d+'
}

df_seq_with_delta = df_seq[df_seq['delta_t_bucket'] > 0].copy()
dt_counts = df_seq_with_delta['delta_t_bucket'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(
    [bucket_labels.get(i, str(i)) for i in dt_counts.index]
    ,dt_counts.values
    ,color='steelblue', edgecolor='white'
)
ax.set_title('Distribution of Time Between Consecutive Visits (Delta T)')
ax.set_xlabel('Days Between Visits')
ax.set_ylabel('Visit Count')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('./eda/02_delta_t_distribution.png', dpi=150)
plt.close()
print("Saved: 02_delta_t_distribution.png")

# concentration score — what % of visits fall in top 3 buckets?
top3_pct = dt_counts.nlargest(3).sum() / dt_counts.sum() * 100
print(f"Top 3 delta_t buckets capture: {top3_pct:.1f}% of all visit transitions")
print("→ If high, visit timing is concentrated — easier to predict")

# ============================================================
# STEP 4: DX CODE FREQUENCY — IS DATA DOMINATED BY FEW CODES?
# ============================================================
print("\n--- DX CODE FREQUENCY ---")

df_dx_exploded = df_seq.explode('dx_list').dropna(subset=['dx_list'])
df_dx_exploded = df_dx_exploded[df_dx_exploded['dx_list'] != '']

dx_counts  = df_dx_exploded['dx_list'].value_counts()
total_dx   = len(dx_counts)
top20_pct  = dx_counts.head(20).sum()  / dx_counts.sum() * 100
top100_pct = dx_counts.head(100).sum() / dx_counts.sum() * 100

print(f"Unique dx codes:            {total_dx:,}")
print(f"Top 20 codes cover:         {top20_pct:.1f}% of all dx occurrences")
print(f"Top 100 codes cover:        {top100_pct:.1f}% of all dx occurrences")
print("→ High % = concentrated distribution = easier to predict common pathways")

fig, ax = plt.subplots(figsize=(14, 6))
top20 = dx_counts.head(PLOT_TOP_N)
ax.barh(top20.index[::-1], top20.values[::-1], color='steelblue', edgecolor='white')
ax.set_title(f'Top {PLOT_TOP_N} Most Frequent Diagnosis Codes')
ax.set_xlabel('Occurrence Count')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig('./eda/03_top_dx_codes.png', dpi=150)
plt.close()
print("Saved: 03_top_dx_codes.png")

# ============================================================
# STEP 5: SEQUENCE REPEATABILITY — DO PATTERNS REPEAT?
# ============================================================
print("\n--- SEQUENCE REPEATABILITY (dx transitions) ---")

# build consecutive visit pairs per member
df_sorted = df_seq.sort_values(['member_id', 'visit_seq_num'])
df_sorted['dx_primary'] = df_sorted['dx_list'].apply(
    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
)

# shift to get next visit dx
df_sorted['dx_next'] = df_sorted.groupby('member_id')['dx_primary'].shift(-1)
df_pairs = df_sorted.dropna(subset=['dx_primary', 'dx_next'])
df_pairs = df_pairs[df_pairs['dx_primary'] != df_pairs['dx_next']]  # exclude self-loops

# top dx → dx transitions
transition_counts = df_pairs.groupby(['dx_primary', 'dx_next']).size().reset_index(name='count')
transition_counts = transition_counts.sort_values('count', ascending=False)

top_transitions = transition_counts.head(PLOT_TOP_N).copy()
top_transitions['transition'] = top_transitions['dx_primary'] + ' → ' + top_transitions['dx_next']

print(f"Total unique dx→dx transitions: {len(transition_counts):,}")
print(f"\nTop {PLOT_TOP_N} most common transitions:")
print(top_transitions[['transition', 'count']].to_string(index=False))

fig, ax = plt.subplots(figsize=(14, 7))
ax.barh(
    top_transitions['transition'][::-1]
    ,top_transitions['count'][::-1]
    ,color='steelblue', edgecolor='white'
)
ax.set_title(f'Top {PLOT_TOP_N} Most Frequent Diagnosis Transitions (dx_A → dx_B)')
ax.set_xlabel('Number of Members with This Transition')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
plt.savefig('./eda/04_top_dx_transitions.png', dpi=150)
plt.close()
print("Saved: 04_top_dx_transitions.png")

# transition coverage — what % of all transitions do top N cover?
total_transitions     = transition_counts['count'].sum()
top20_trans_pct       = top_transitions['count'].sum() / total_transitions * 100
print(f"\nTop {PLOT_TOP_N} transitions cover: {top20_trans_pct:.1f}% of all transitions")
print("→ High % = strong repeating patterns = model has signal to learn")

# ============================================================
# STEP 6: TRANSITION PROBABILITY — HOW CONCENTRATED IS NEXT VISIT?
# ============================================================
print("\n--- TRANSITION PROBABILITY CONCENTRATION ---")

# for each source dx code, what is the probability of the top next code?
transition_probs = transition_counts.copy()
source_totals    = transition_probs.groupby('dx_primary')['count'].sum().rename('total')
transition_probs = transition_probs.join(source_totals, on='dx_primary')
transition_probs['prob'] = transition_probs['count'] / transition_probs['total']

# top transition probability per source dx
top_prob_per_source = transition_probs.sort_values('prob', ascending=False).groupby('dx_primary').first()
top_prob_per_source = top_prob_per_source[top_prob_per_source['total'] >= 100]  # filter low volume

print(f"\nTop 20 most predictable source dx codes (highest P(next | current)):")
print(
    top_prob_per_source[['dx_next', 'prob', 'total']]
    .sort_values('prob', ascending=False)
    .head(20)
    .to_string()
)

avg_top_prob = top_prob_per_source['prob'].mean()
print(f"\nAverage top-1 transition probability: {avg_top_prob:.3f}")
print("→ If > 0.3, given current dx there is a strong dominant next dx")

# ============================================================
# STEP 7: VISIT ENTROPY — HOW RANDOM IS THE NEXT VISIT?
# ============================================================
print("\n--- VISIT ENTROPY ANALYSIS ---")

# compute entropy of next-visit distribution per source dx
def compute_entropy(group):
    probs = group['prob'].values
    return entropy(probs, base=2)

# filter to dx codes with 50+ observations
valid_sources = source_totals[source_totals >= 50].index
entropy_df    = (
    transition_probs[transition_probs['dx_primary'].isin(valid_sources)]
    .groupby('dx_primary')
    .apply(compute_entropy)
    .reset_index(name='entropy')
)
entropy_df = entropy_df.join(source_totals, on='dx_primary')
entropy_df = entropy_df.sort_values('entropy')

low_entropy  = (entropy_df['entropy'] < 1.0).sum()
high_entropy = (entropy_df['entropy'] > 3.0).sum()
total_codes  = len(entropy_df)

print(f"Dx codes with low entropy  (<1.0 bits, highly predictable): {low_entropy:,} ({low_entropy/total_codes*100:.1f}%)")
print(f"Dx codes with high entropy (>3.0 bits, near random):        {high_entropy:,} ({high_entropy/total_codes*100:.1f}%)")
print(f"Median entropy: {entropy_df['entropy'].median():.2f} bits")
print("→ Lower median entropy = stronger predictable patterns in data")

fig, ax = plt.subplots(figsize=(12, 5))
entropy_df['entropy'].hist(bins=50, ax=ax, color='steelblue', edgecolor='white')
ax.axvline(entropy_df['entropy'].median(), color='red', linestyle='--', label=f"Median: {entropy_df['entropy'].median():.2f} bits")
ax.axvline(1.0, color='green', linestyle=':', label='Low entropy threshold (1.0)')
ax.set_title('Distribution of Next-Visit Entropy per Diagnosis Code')
ax.set_xlabel('Entropy (bits) — lower = more predictable')
ax.set_ylabel('Number of Diagnosis Codes')
ax.legend()
plt.tight_layout()
plt.savefig('./eda/05_visit_entropy_distribution.png', dpi=150)
plt.close()
print("Saved: 05_visit_entropy_distribution.png")

# most predictable dx codes
print(f"\nTop 15 most predictable dx codes (lowest entropy, min 100 observations):")
print(
    entropy_df[entropy_df['total'] >= 100]
    .head(15)[['dx_primary', 'entropy', 'total']]
    .to_string(index=False)
)

# ============================================================
# STEP 8: REPEAT VISIT RATE — DO MEMBERS RETURN?
# ============================================================
print("\n--- REPEAT VISIT RATE ---")

# for each member, how many visits are to the same dx code as a previous visit?
df_sorted['dx_primary_str'] = df_sorted['dx_primary'].astype(str)
member_dx_sets = df_sorted.groupby('member_id')['dx_primary_str'].apply(list)

repeat_rates = []
for member_id, dx_list in member_dx_sets.items():
    if len(dx_list) < 2:
        continue
    seen     = set()
    repeats  = 0
    for dx in dx_list:
        if dx in seen:
            repeats += 1
        seen.add(dx)
    repeat_rates.append(repeats / len(dx_list))

repeat_arr      = np.array(repeat_rates)
pct_with_repeat = (repeat_arr > 0).mean() * 100
avg_repeat_rate = repeat_arr.mean() * 100

print(f"Members with at least one repeat dx code: {pct_with_repeat:.1f}%")
print(f"Average repeat visit rate per member:     {avg_repeat_rate:.1f}%")
print("→ High repeat rate = chronic condition patterns = strong predictable signal")

# ============================================================
# STEP 9: TEMPORAL CONSISTENCY — IS DELTA_T CONSISTENT PER DX?
# ============================================================
print("\n--- TEMPORAL CONSISTENCY PER DX CODE ---")

df_delta = df_pairs.copy()
df_delta['delta_t'] = df_delta['delta_t_bucket']

# coefficient of variation of delta_t per source dx (lower = more consistent timing)
dt_stats = (
    df_delta[df_delta['dx_primary'].isin(dx_counts.head(50).index)]
    .groupby('dx_primary')['delta_t']
    .agg(['mean', 'std', 'count'])
    .assign(cv=lambda x: x['std'] / x['mean'])
    .sort_values('cv')
)

print(f"\nTop 15 dx codes with most consistent visit timing (lowest CV):")
print(dt_stats.head(15).to_string())
print("\n→ Low CV = visit timing is consistent = temporal model adds value")

# ============================================================
# STEP 10: SUMMARY TABLE FOR STAKEHOLDERS
# ============================================================
print("\n--- SUMMARY FOR STAKEHOLDERS ---")

summary = {
    'Total Members'                              : f"{df_seq['member_id'].nunique():,}",
    'Members with 2+ Visits'                     : f"{(seq_lengths >= 2).sum():,} ({(seq_lengths >= 2).mean()*100:.1f}%)",
    'Avg Visits per Member'                      : f"{seq_lengths.mean():.1f}",
    'Unique Dx Codes'                            : f"{total_dx:,}",
    'Top 20 Dx Codes Coverage'                   : f"{top20_pct:.1f}%",
    'Top 20 Transitions Coverage'                : f"{top20_trans_pct:.1f}%",
    'Avg Top-1 Transition Probability'           : f"{avg_top_prob:.3f}",
    'Dx Codes with Low Entropy (<1 bit)'         : f"{low_entropy:,} ({low_entropy/total_codes*100:.1f}%)",
    'Median Next-Visit Entropy'                  : f"{entropy_df['entropy'].median():.2f} bits",
    'Members with Repeat Dx Visits'              : f"{pct_with_repeat:.1f}%",
}

df_summary = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
print(df_summary.to_string(index=False))
df_summary.to_csv('./eda/summary_stats.csv', index=False)

# ============================================================
# STEP 11: SAVE TRANSITION TABLE
# ============================================================
transition_counts.head(200).to_csv('./eda/top_transitions.csv', index=False)
top_prob_per_source.sort_values('prob', ascending=False).head(100).to_csv('./eda/top_predictable_dx_codes.csv')
entropy_df.sort_values('entropy').to_csv('./eda/dx_entropy.csv', index=False)

print("\nFiles saved to ./eda/:")
for f in sorted(os.listdir('./eda')):
    size = f"{os.path.getsize(f'./eda/{f}') / 1e3:.1f} KB"
    print(f"  {f}  {size}")

print("\nNotebook 0 complete")
