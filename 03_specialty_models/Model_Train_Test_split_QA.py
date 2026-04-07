# ============================================================
# NB_06 — Train Test Split Validation
# Purpose : Validate the quality of the train/test split
#           before running any model. Ensures distributions
#           are consistent and the split is sound.
# Train   : A870800_gen_rec_model_train (trigger < 2024)
# Test    : A870800_gen_rec_model_test  (trigger >= 2024)
# ============================================================
from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, Markdown

client = bigquery.Client(project="anbc-hcb-dev")
DATASET = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
TRAIN   = f"`{DATASET}.A870800_gen_rec_model_train`"
TEST    = f"`{DATASET}.A870800_gen_rec_model_test`"

def fmt_count(x):
    return f"{int(x):,}"

def fmt_pct(x):
    return f"{x:.1f}%"

display(Markdown("""
---
# NB 06 — Train Test Split Validation

## Purpose

Before running any model, validate that the train and test splits
are consistent and representative.

A poor split could cause the model to train on data that does not
reflect the test distribution — leading to artificially inflated
or deflated metrics.

**Split rule:** `trigger_date < 2024-01-01` → Train, `trigger_date >= 2024-01-01` → Test

**Checks in this notebook:**
1. Volume — how many triggers, members, rows in each split
2. Diagnosis coverage — are the same DX codes in both splits
3. Specialty coverage — are the same label specialties in both splits
4. Cohort distribution — is the member segment split balanced
5. Time bucket distribution — T0_30, T30_60, T60_180 coverage
6. Label distribution — does label_specialty distribution look similar
7. Sequence length — are pre-trigger sequences of similar length
8. Trigger year distribution — confirms the split boundary

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — VOLUME
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 1 — Volume

Basic row counts and unique member and trigger counts per split.
"""))

volume = client.query(f"""
SELECT
    'Train' AS split
    ,COUNT(*) AS total_rows
    ,COUNT(DISTINCT member_id) AS unique_members
    ,COUNT(DISTINCT CONCAT(member_id, '_',
        CAST(trigger_date AS STRING), '_', trigger_dx)) AS unique_triggers
    ,COUNT(DISTINCT trigger_dx) AS unique_dx_codes
    ,COUNT(DISTINCT label_specialty) AS unique_specialties
FROM {TRAIN}
UNION ALL
SELECT
    'Test' AS split
    ,COUNT(*) AS total_rows
    ,COUNT(DISTINCT member_id) AS unique_members
    ,COUNT(DISTINCT CONCAT(member_id, '_',
        CAST(trigger_date AS STRING), '_', trigger_dx)) AS unique_triggers
    ,COUNT(DISTINCT trigger_dx) AS unique_dx_codes
    ,COUNT(DISTINCT label_specialty) AS unique_specialties
FROM {TEST}
""").to_dataframe()

total_rows = volume.set_index("split")["total_rows"]
volume["% of Total Rows"] = volume.apply(
    lambda r: fmt_pct(r["total_rows"] / total_rows.sum() * 100), axis=1)

display(volume.rename(columns={
    "split": "Split",
    "total_rows": "Total Rows",
    "unique_members": "Unique Members",
    "unique_triggers": "Unique Triggers",
    "unique_dx_codes": "Unique DX Codes",
    "unique_specialties": "Unique Specialties"
}).reset_index(drop=True))

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
colors = ["#4C9BE8", "#F4845F"]
metrics = ["total_rows", "unique_members", "unique_triggers"]
titles  = ["Total Rows", "Unique Members", "Unique Triggers"]

for i, (metric, title) in enumerate(zip(metrics, titles)):
    axes[i].bar(volume["split"], volume[metric], color=colors, alpha=0.85, width=0.5)
    for j, (val, sp) in enumerate(zip(volume[metric], volume["split"])):
        axes[i].text(j, val * 1.01, fmt_count(val),
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[i].set_title(title, fontsize=11, fontweight="bold")
    axes[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
    axes[i].grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Train vs Test — Volume Comparison", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("tt_volume.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRIGGER YEAR DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 2 — Trigger Year Distribution

Confirms the split boundary is correct.
All 2022 and 2023 triggers should be in train.
All 2024 and 2025 triggers should be in test.
Any overlap indicates a split error.
"""))

year_dist = client.query(f"""
SELECT 'Train' AS split, EXTRACT(YEAR FROM trigger_date) AS trigger_year
    ,COUNT(*) AS row_count
    ,COUNT(DISTINCT CONCAT(member_id, '_',
        CAST(trigger_date AS STRING), '_', trigger_dx)) AS unique_triggers
FROM {TRAIN}
GROUP BY trigger_year
UNION ALL
SELECT 'Test' AS split, EXTRACT(YEAR FROM trigger_date) AS trigger_year
    ,COUNT(*) AS row_count
    ,COUNT(DISTINCT CONCAT(member_id, '_',
        CAST(trigger_date AS STRING), '_', trigger_dx)) AS unique_triggers
FROM {TEST}
GROUP BY trigger_year
ORDER BY split, trigger_year
""").to_dataframe()

display(year_dist.rename(columns={
    "split": "Split", "trigger_year": "Year",
    "row_count": "Row Count", "unique_triggers": "Unique Triggers"
}).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(14, 6))
train_yr = year_dist[year_dist["split"] == "Train"]
test_yr  = year_dist[year_dist["split"] == "Test"]
x = sorted(year_dist["trigger_year"].unique())
width = 0.35
xi = np.arange(len(x))
ax.bar(xi - width/2,
       [train_yr[train_yr["trigger_year"] == y]["unique_triggers"].sum() for y in x],
       width, label="Train", color="#4C9BE8", alpha=0.85)
ax.bar(xi + width/2,
       [test_yr[test_yr["trigger_year"] == y]["unique_triggers"].sum() for y in x],
       width, label="Test", color="#F4845F", alpha=0.85)
ax.set_xticks(xi)
ax.set_xticklabels([str(y) for y in x])
ax.set_ylabel("Unique Triggers", fontsize=10)
ax.set_title("Trigger Year Distribution — Train vs Test",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
ax.axvline(1.5, color="red", linestyle="--", alpha=0.7, label="Split boundary (2024)")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("tt_year_dist.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DIAGNOSIS COVERAGE
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 3 — Diagnosis Coverage

Are the same diagnosis codes represented in both train and test?

DX codes only in test = model has never seen this diagnosis during training.
These triggers cannot be predicted by Markov — they affect coverage.
Sequential models with learned representations handle these better.
"""))

dx_coverage = client.query(f"""
WITH train_dx AS (
    SELECT DISTINCT trigger_dx FROM {TRAIN}
),
test_dx AS (
    SELECT DISTINCT trigger_dx FROM {TEST}
)
SELECT
    COUNT(DISTINCT t.trigger_dx)                         AS train_only_dx
    ,COUNT(DISTINCT te.trigger_dx)                       AS test_only_dx
    ,COUNT(DISTINCT CASE WHEN t.trigger_dx IS NOT NULL
        AND te.trigger_dx IS NOT NULL
        THEN t.trigger_dx END)                           AS shared_dx
FROM train_dx t
FULL OUTER JOIN test_dx te ON t.trigger_dx = te.trigger_dx
""").to_dataframe()

d = dx_coverage.iloc[0]
total_dx = d["train_only_dx"] + d["test_only_dx"] + d["shared_dx"]

display(Markdown(f"""
| Metric | Count | % |
|---|---|---|
| DX codes in train only | {fmt_count(d['train_only_dx'])} | {fmt_pct(d['train_only_dx']/total_dx*100)} |
| DX codes in test only | {fmt_count(d['test_only_dx'])} | {fmt_pct(d['test_only_dx']/total_dx*100)} |
| DX codes in both (shared) | {fmt_count(d['shared_dx'])} | {fmt_pct(d['shared_dx']/total_dx*100)} |
"""))

fig, ax = plt.subplots(figsize=(8, 6))
labels = ["Train Only", "Test Only", "Shared"]
sizes  = [d["train_only_dx"], d["test_only_dx"], d["shared_dx"]]
colors = ["#4C9BE8", "#F4845F", "#5DBE7E"]
ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors,
       startangle=90, textprops={"fontsize": 10})
ax.set_title("Diagnosis Code Coverage — Train vs Test",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("tt_dx_coverage.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SPECIALTY COVERAGE
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 4 — Specialty Coverage

Are the same label specialties represented in both splits?
Specialties only in test means the model never saw those labels during training.
"""))

spec_coverage = client.query(f"""
WITH train_spec AS (
    SELECT DISTINCT label_specialty FROM {TRAIN}
),
test_spec AS (
    SELECT DISTINCT label_specialty FROM {TEST}
)
SELECT
    COUNT(DISTINCT t.label_specialty)                    AS train_only_spec
    ,COUNT(DISTINCT te.label_specialty)                  AS test_only_spec
    ,COUNT(DISTINCT CASE WHEN t.label_specialty IS NOT NULL
        AND te.label_specialty IS NOT NULL
        THEN t.label_specialty END)                      AS shared_spec
FROM train_spec t
FULL OUTER JOIN test_spec te ON t.label_specialty = te.label_specialty
""").to_dataframe()

s = spec_coverage.iloc[0]
total_spec = s["train_only_spec"] + s["test_only_spec"] + s["shared_spec"]

display(Markdown(f"""
| Metric | Count | % |
|---|---|---|
| Specialties in train only | {fmt_count(s['train_only_spec'])} | {fmt_pct(s['train_only_spec']/total_spec*100)} |
| Specialties in test only | {fmt_count(s['test_only_spec'])} | {fmt_pct(s['test_only_spec']/total_spec*100)} |
| Specialties in both (shared) | {fmt_count(s['shared_spec'])} | {fmt_pct(s['shared_spec']/total_spec*100)} |
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — COHORT DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 5 — Cohort Distribution

Is the member segment split balanced between train and test?
A large imbalance suggests one cohort is over-represented in one split
which could bias model training or evaluation.
"""))

cohort_dist = client.query(f"""
SELECT 'Train' AS split, member_segment
    ,COUNT(DISTINCT CONCAT(member_id, '_',
        CAST(trigger_date AS STRING), '_', trigger_dx)) AS unique_triggers
FROM {TRAIN}
GROUP BY member_segment
UNION ALL
SELECT 'Test' AS split, member_segment
    ,COUNT(DISTINCT CONCAT(member_id, '_',
        CAST(trigger_date AS STRING), '_', trigger_dx)) AS unique_triggers
FROM {TEST}
GROUP BY member_segment
ORDER BY split, member_segment
""").to_dataframe()

# Compute % within split
cohort_dist["pct"] = cohort_dist.groupby("split")["unique_triggers"].transform(
    lambda x: x / x.sum() * 100
).round(1)

display(cohort_dist.rename(columns={
    "split": "Split", "member_segment": "Cohort",
    "unique_triggers": "Unique Triggers", "pct": "% Within Split"
}).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(14, 7))
cohorts = cohort_dist["member_segment"].unique()
x = np.arange(len(cohorts))
width = 0.35

train_vals = [cohort_dist[(cohort_dist["split"] == "Train") &
              (cohort_dist["member_segment"] == c)]["pct"].sum() for c in cohorts]
test_vals  = [cohort_dist[(cohort_dist["split"] == "Test") &
              (cohort_dist["member_segment"] == c)]["pct"].sum() for c in cohorts]

ax.bar(x - width/2, train_vals, width, label="Train", color="#4C9BE8", alpha=0.85)
ax.bar(x + width/2, test_vals,  width, label="Test",  color="#F4845F", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(cohorts)
ax.set_ylabel("% Within Split", fontsize=10)
ax.set_title("Cohort Distribution — Train vs Test",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("tt_cohort_dist.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TIME BUCKET DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 6 — Time Bucket Distribution

How are T0_30, T30_60, T60_180 labels distributed across train and test?
A large imbalance means the model trains on a different window mix than it evaluates on.
"""))

bucket_dist = client.query(f"""
SELECT 'Train' AS split, time_bucket
    ,COUNT(*) AS row_count
    ,COUNT(DISTINCT CONCAT(member_id, '_',
        CAST(trigger_date AS STRING), '_', trigger_dx)) AS unique_triggers
FROM {TRAIN}
GROUP BY time_bucket
UNION ALL
SELECT 'Test' AS split, time_bucket
    ,COUNT(*) AS row_count
    ,COUNT(DISTINCT CONCAT(member_id, '_',
        CAST(trigger_date AS STRING), '_', trigger_dx)) AS unique_triggers
FROM {TEST}
GROUP BY time_bucket
ORDER BY split, time_bucket
""").to_dataframe()

bucket_dist["pct"] = bucket_dist.groupby("split")["row_count"].transform(
    lambda x: x / x.sum() * 100
).round(1)

display(bucket_dist.rename(columns={
    "split": "Split", "time_bucket": "Time Bucket",
    "row_count": "Row Count", "unique_triggers": "Unique Triggers",
    "pct": "% Within Split"
}).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(12, 6))
buckets = ["T0_30", "T30_60", "T60_180"]
x = np.arange(len(buckets))
width = 0.35

train_pct = [bucket_dist[(bucket_dist["split"] == "Train") &
             (bucket_dist["time_bucket"] == b)]["pct"].sum() for b in buckets]
test_pct  = [bucket_dist[(bucket_dist["split"] == "Test") &
             (bucket_dist["time_bucket"] == b)]["pct"].sum() for b in buckets]

ax.bar(x - width/2, train_pct, width, label="Train", color="#4C9BE8", alpha=0.85)
ax.bar(x + width/2, test_pct,  width, label="Test",  color="#F4845F", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(buckets)
ax.set_ylabel("% Within Split", fontsize=10)
ax.set_title("Time Bucket Distribution — Train vs Test",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("tt_bucket_dist.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — LABEL SPECIALTY DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 7 — Label Specialty Distribution

Are the top specialties distributed similarly in train and test?
Large gaps indicate the model may not see enough examples of certain
specialties during training to predict them reliably in test.
"""))

spec_dist = client.query(f"""
WITH train_spec AS (
    SELECT label_specialty
        ,COUNT(*) AS train_count
    FROM {TRAIN}
    WHERE label_specialty IS NOT NULL
    GROUP BY label_specialty
),
test_spec AS (
    SELECT label_specialty
        ,COUNT(*) AS test_count
    FROM {TEST}
    WHERE label_specialty IS NOT NULL
    GROUP BY label_specialty
)
SELECT
    COALESCE(t.label_specialty, te.label_specialty) AS label_specialty
    ,COALESCE(t.train_count, 0) AS train_count
    ,COALESCE(te.test_count, 0) AS test_count
FROM train_spec t
FULL OUTER JOIN test_spec te ON t.label_specialty = te.label_specialty
ORDER BY train_count DESC
LIMIT 20
""").to_dataframe()

spec_dist["train_pct"] = (spec_dist["train_count"] /
                           spec_dist["train_count"].sum() * 100).round(2)
spec_dist["test_pct"]  = (spec_dist["test_count"] /
                           spec_dist["test_count"].sum() * 100).round(2)
spec_dist["pct_diff"]  = (spec_dist["train_pct"] - spec_dist["test_pct"]).round(2)

display(spec_dist[[
    "label_specialty", "train_count", "train_pct",
    "test_count", "test_pct", "pct_diff"
]].rename(columns={
    "label_specialty": "Specialty",
    "train_count": "Train Count",
    "train_pct": "Train %",
    "test_count": "Test Count",
    "test_pct": "Test %",
    "pct_diff": "Train - Test % Diff"
}).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(16, 8))
x = np.arange(len(spec_dist))
width = 0.35
ax.bar(x - width/2, spec_dist["train_pct"], width,
       label="Train", color="#4C9BE8", alpha=0.85)
ax.bar(x + width/2, spec_dist["test_pct"],  width,
       label="Test",  color="#F4845F", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(spec_dist["label_specialty"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("% of Labels", fontsize=10)
ax.set_title("Top 20 Label Specialties — Train vs Test Distribution",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("tt_label_dist.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SEQUENCE LENGTH DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 8 — Sequence Length Distribution

How long are the pre-trigger visit sequences in train vs test?
If test sequences are significantly shorter than train sequences,
the model may not generalize well — it trained on rich history
but predicts from sparse history.
"""))

seq_len = client.query(f"""
SELECT 'Train' AS split
    ,ARRAY_LENGTH(visit_sequence) AS seq_length
    ,COUNT(*) AS row_count
FROM {TRAIN}
GROUP BY seq_length
UNION ALL
SELECT 'Test' AS split
    ,ARRAY_LENGTH(visit_sequence) AS seq_length
    ,COUNT(*) AS row_count
FROM {TEST}
GROUP BY seq_length
ORDER BY split, seq_length
""").to_dataframe()

# Compute percentiles per split
for split in ["Train", "Test"]:
    sub = seq_len[seq_len["split"] == split]
    expanded = np.repeat(sub["seq_length"].values, sub["row_count"].values)
    pcts = np.percentile(expanded, [10, 25, 50, 75, 90, 95])
    display(Markdown(f"""
**{split} sequence length percentiles:**
p10={pcts[0]:.0f}, p25={pcts[1]:.0f}, p50={pcts[2]:.0f},
p75={pcts[3]:.0f}, p90={pcts[4]:.0f}, p95={pcts[5]:.0f}
"""))

fig, ax = plt.subplots(figsize=(14, 6))
cap = 50
for split, color in [("Train", "#4C9BE8"), ("Test", "#F4845F")]:
    sub = seq_len[(seq_len["split"] == split) & (seq_len["seq_length"] <= cap)]
    ax.plot(sub["seq_length"], sub["row_count"],
            color=color, linewidth=2, label=split, alpha=0.85)
ax.set_xlabel("Sequence Length (visits before trigger)", fontsize=10)
ax.set_ylabel("Row Count", fontsize=10)
ax.set_title("Sequence Length Distribution — Train vs Test\n(capped at 50)",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("tt_seq_length.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Summary

Review the outputs above before proceeding to model training.

**Green flags — split is sound:**
- Train is larger than test in volume
- No trigger years overlap between splits
- Shared DX codes are high — model has seen most test diagnoses
- Cohort distribution is similar across splits
- Time bucket mix is similar across splits
- Top specialties have similar % in train and test
- Sequence lengths are similar across splits

**Red flags — investigate before training:**
- Test has many DX codes not seen in train — high coverage gaps expected
- Cohort imbalance — one cohort over-represented in test
- Label specialty distribution differs significantly — model may not predict
  rare specialties reliably
- Test sequences much shorter than train — model may not generalize

---
"""))
