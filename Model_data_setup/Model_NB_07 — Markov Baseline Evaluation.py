# ============================================================
# NB_07 — Markov Baseline Evaluation
# Purpose : Evaluate Markov Order 1 transition probabilities
#           as a baseline predictor for next specialty
# Train   : A870800_gen_rec_markov_train (pre-2024 triggers)
# Test    : A870800_gen_rec_model_test (2024+ triggers)
# Metrics : Hit@K, Precision@K, Recall@K, NDCG@K
#           for K = 1, 3, 5
#           per window T0_30, T30_60, T60_180
# Grain   : One evaluation per member + trigger + time_bucket
#           True label set = all specialties visited in window
# ============================================================
from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, Markdown

client = bigquery.Client(project="anbc-hcb-dev")
DATASET = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
K_VALUES = [1, 3, 5]
WINDOWS = ["T0_30", "T30_60", "T60_180"]

display(Markdown("""
---
# NB 07 — Markov Baseline Evaluation

## What This Notebook Does

Evaluates the Markov Order 1 transition model as a prediction baseline.

The Markov model computes `P(next_specialty | trigger_dx)` from pre-2024
training data. For each test trigger it produces a ranked list of predicted
specialties. That ranked list is evaluated against the set of specialties
the member actually visited within each time window.

**This is the performance floor.** Every subsequent model — SASRec,
BERT4Rec, HSTU — must beat these numbers to justify its complexity.

**Evaluation grain:** One evaluation per `member + trigger + time_bucket`.
True label set = all specialties visited within that window.
Multiple correct labels are possible — recall and precision reflect
how many the model recovered.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("## Section 1 — Load Train Probabilities and Test Data"))

# Markov transition probabilities — pre-2024 only
markov_train = client.query(f"""
    SELECT
        trigger_dx
        ,next_specialty
        ,member_segment
        ,transition_count
        ,unique_members
    FROM `{DATASET}.A870800_gen_rec_markov_train`
    WHERE next_specialty IS NOT NULL
""").to_dataframe()

markov_train["transition_count"] = markov_train["transition_count"].astype(float)

# Compute transition probability per trigger_dx + member_segment
dx_totals = (
    markov_train.groupby(["trigger_dx", "member_segment"])["transition_count"]
    .sum().reset_index().rename(columns={"transition_count": "dx_total"})
)
markov_train = markov_train.merge(dx_totals, on=["trigger_dx", "member_segment"])
markov_train["probability"] = (
    markov_train["transition_count"] / markov_train["dx_total"]
).round(6)

display(Markdown(f"""
**Markov train — unique trigger DX codes:** {markov_train['trigger_dx'].nunique():,}
**Markov train — unique specialties:** {markov_train['next_specialty'].nunique():,}
**Markov train — total transition pairs:** {len(markov_train):,}
"""))

# Test data — 2024+ triggers with specialty labels
test_raw = client.query(f"""
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,trigger_dx_clean
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,label_specialty
        ,time_bucket
    FROM `{DATASET}.A870800_gen_rec_model_test`
    WHERE label_specialty IS NOT NULL
""").to_dataframe()

display(Markdown(f"""
**Test set — total rows:** {len(test_raw):,}
**Test set — unique members:** {test_raw['member_id'].nunique():,}
**Test set — unique triggers:** {test_raw.groupby(['member_id','trigger_date','trigger_dx']).ngroups:,}
"""))

# Show test distribution by window
window_dist = test_raw.groupby("time_bucket")["member_id"].count().reset_index()
window_dist.columns = ["Time Bucket", "Row Count"]
display(window_dist)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BUILD TRUE LABEL SETS
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 2 — Build True Label Sets

Group test data to get the full set of specialties visited per
`member + trigger + time_bucket`.

This is the ground truth each prediction is evaluated against.
A trigger with 3 specialty visits within T0_30 has a true label set of size 3.
"""))

true_labels = (
    test_raw.groupby(
        ["member_id", "trigger_date", "trigger_dx",
         "trigger_dx_clean", "member_segment",
         "is_t30_qualified", "is_t60_qualified", "is_t180_qualified",
         "time_bucket"],
        as_index=False
    )["label_specialty"]
    .apply(set)
    .reset_index()
    .rename(columns={"label_specialty": "true_label_set"})
)

true_labels["true_label_count"] = true_labels["true_label_set"].apply(len)

display(Markdown(f"""
**Evaluation rows (member + trigger + time_bucket):** {len(true_labels):,}
"""))

# Distribution of true label set sizes
label_dist = (
    true_labels.groupby("true_label_count")["member_id"]
    .count().reset_index()
    .rename(columns={"member_id": "count", "true_label_count": "True Label Set Size"})
)
display(Markdown("#### Distribution of True Label Set Sizes"))
display(label_dist.head(10).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(label_dist["True Label Set Size"].astype(str),
       label_dist["count"], color="#4C9BE8", alpha=0.85)
ax.set_xlabel("Number of Specialties Visited per Trigger per Window", fontsize=10)
ax.set_ylabel("Number of Evaluation Rows", fontsize=10)
ax.set_title("Distribution of True Label Set Sizes",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("true_label_dist.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GENERATE MARKOV PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 3 — Generate Markov Predictions

For each test trigger, look up transition probabilities from the train set.
Rank specialties by probability — top K predictions used for evaluation.

If a trigger DX has no training data — no prediction is possible.
These are tracked as coverage gaps.
"""))

# Top K predictions per trigger_dx + member_segment
max_k = max(K_VALUES)
markov_topk = (
    markov_train.sort_values(
        ["trigger_dx", "member_segment", "probability"], ascending=[True, True, False]
    )
    .groupby(["trigger_dx", "member_segment"])
    .head(max_k)
    .groupby(["trigger_dx", "member_segment"])["next_specialty"]
    .apply(list)
    .reset_index()
    .rename(columns={"next_specialty": "predicted_specialties"})
)

# Join predictions to true labels
eval_df = true_labels.merge(
    markov_topk,
    on=["trigger_dx", "member_segment"],
    how="left"
)

# Coverage — how many triggers have Markov predictions
covered = eval_df["predicted_specialties"].notna().sum()
total = len(eval_df)

display(Markdown(f"""
**Triggers with Markov predictions:** {covered:,} ({covered/total*100:.1f}%)
**Triggers with no Markov prediction (unseen DX):** {total-covered:,} ({(total-covered)/total*100:.1f}%)

Triggers with no prediction are excluded from metric computation.
Coverage rate is reported alongside all metrics.
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — EVALUATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def hit_at_k(true_set, predicted, k):
    return int(len(true_set.intersection(set(predicted[:k]))) > 0)

def precision_at_k(true_set, predicted, k):
    hits = len(true_set.intersection(set(predicted[:k])))
    return hits / k

def recall_at_k(true_set, predicted, k):
    hits = len(true_set.intersection(set(predicted[:k])))
    return hits / len(true_set) if len(true_set) > 0 else 0

def ndcg_at_k(true_set, predicted, k):
    dcg = 0.0
    for rank, item in enumerate(predicted[:k], start=1):
        if item in true_set:
            dcg += 1.0 / np.log2(rank + 1)
    # ideal DCG — all true labels ranked at top
    idcg = sum(1.0 / np.log2(rank + 1)
               for rank in range(1, min(len(true_set), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate(df, k):
    rows = df[df["predicted_specialties"].notna()].copy()
    rows["hit"]       = rows.apply(lambda r: hit_at_k(r["true_label_set"], r["predicted_specialties"], k), axis=1)
    rows["precision"] = rows.apply(lambda r: precision_at_k(r["true_label_set"], r["predicted_specialties"], k), axis=1)
    rows["recall"]    = rows.apply(lambda r: recall_at_k(r["true_label_set"], r["predicted_specialties"], k), axis=1)
    rows["ndcg"]      = rows.apply(lambda r: ndcg_at_k(r["true_label_set"], r["predicted_specialties"], k), axis=1)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — COMPUTE METRICS
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 5 — Evaluation Results

Metrics computed per time window and per K value.
Coverage = % of test triggers with at least one Markov prediction.
"""))

# Window qualification filter
window_qual_map = {
    "T0_30":   "is_t30_qualified",
    "T30_60":  "is_t60_qualified",
    "T60_180": "is_t180_qualified"
}

results = []

for window in WINDOWS:
    qual_col = window_qual_map[window]
    window_df = eval_df[
        (eval_df["time_bucket"] == window) &
        (eval_df[qual_col] == True)
    ].copy()

    total_window = len(window_df)
    covered_window = window_df["predicted_specialties"].notna().sum()

    for k in K_VALUES:
        scored = evaluate(window_df, k)
        results.append({
            "Time Window": window,
            "K": k,
            "Coverage": f"{covered_window/total_window*100:.1f}%" if total_window > 0 else "N/A",
            "Hit@K":       round(scored["hit"].mean(), 4),
            "Precision@K": round(scored["precision"].mean(), 4),
            "Recall@K":    round(scored["recall"].mean(), 4),
            "NDCG@K":      round(scored["ndcg"].mean(), 4),
            "N Evaluated": covered_window,
            "N Total":     total_window
        })

results_df = pd.DataFrame(results)

display(Markdown("### Markov Baseline — Full Results"))
display(results_df.reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VISUALIZE METRICS
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 6 — Metric Visualization

Each chart shows one metric across K values and time windows.
These are the baseline numbers all subsequent models are compared against.
"""))

metrics = ["Hit@K", "Precision@K", "Recall@K", "NDCG@K"]
colors  = {"T0_30": "#5DBE7E", "T30_60": "#F7C948", "T60_180": "#F4845F"}
markers = {"T0_30": "o", "T30_60": "s", "T60_180": "^"}

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    for window in WINDOWS:
        sub = results_df[results_df["Time Window"] == window].sort_values("K")
        ax.plot(sub["K"], sub[metric],
                color=colors[window], marker=markers[window],
                linewidth=2, markersize=8, label=window)
        for _, row in sub.iterrows():
            ax.annotate(f"{row[metric]:.3f}",
                        (row["K"], row[metric]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color=colors[window])
    ax.set_title(f"Markov Baseline — {metric}", fontsize=12, fontweight="bold")
    ax.set_xlabel("K", fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)

fig.suptitle("Markov Baseline — Evaluation Metrics by Window and K",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("markov_baseline_metrics.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — METRICS BY MEMBER SEGMENT
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 7 — Metrics by Member Segment

Break down performance by cohort — Adult Female, Adult Male, Senior, Children.
Helps identify whether the model performs consistently across demographics.
"""))

COHORTS = ["Adult_Female", "Adult_Male", "Senior", "Children"]
cohort_results = []

for window in WINDOWS:
    qual_col = window_qual_map[window]
    window_df = eval_df[
        (eval_df["time_bucket"] == window) &
        (eval_df[qual_col] == True)
    ].copy()

    for cohort in COHORTS:
        cohort_df = window_df[window_df["member_segment"] == cohort]
        if len(cohort_df) == 0:
            continue
        for k in K_VALUES:
            scored = evaluate(cohort_df, k)
            cohort_results.append({
                "Time Window": window,
                "Cohort": cohort,
                "K": k,
                "Hit@K":       round(scored["hit"].mean(), 4),
                "Precision@K": round(scored["precision"].mean(), 4),
                "Recall@K":    round(scored["recall"].mean(), 4),
                "NDCG@K":      round(scored["ndcg"].mean(), 4),
                "N Evaluated": scored["predicted_specialties"].notna().sum()
            })

cohort_df_results = pd.DataFrame(cohort_results)

display(Markdown("### Metrics by Cohort — K=3"))
display(cohort_df_results[cohort_df_results["K"] == 3].reset_index(drop=True))

fig, axes = plt.subplots(1, 2, figsize=(22, 8))
cohort_colors = {
    "Adult_Female": "#4C9BE8",
    "Adult_Male": "#F4845F",
    "Senior": "#5DBE7E",
    "Children": "#F7C948"
}

for idx, metric in enumerate(["Hit@K", "NDCG@K"]):
    ax = axes[idx]
    sub = cohort_df_results[cohort_df_results["K"] == 3]
    for cohort in COHORTS:
        c = sub[sub["Cohort"] == cohort].sort_values("Time Window")
        if c.empty:
            continue
        ax.plot(c["Time Window"], c[metric],
                color=cohort_colors[cohort], marker="o",
                linewidth=2, markersize=8, label=cohort)
    ax.set_title(f"{metric} at K=3 by Cohort", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Window", fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)

fig.suptitle("Markov Baseline — Performance by Member Segment",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("markov_baseline_by_cohort.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — COVERAGE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 8 — Coverage Analysis

Coverage = % of test triggers where the Markov model has a prediction.
Triggers with unseen DX codes in the test set cannot be predicted.
Low coverage is a known limitation of the Markov baseline.
Sequential models with learned representations handle unseen patterns better.
"""))

coverage_by_window = (
    eval_df.groupby("time_bucket")
    .apply(lambda g: pd.Series({
        "total": len(g),
        "covered": g["predicted_specialties"].notna().sum(),
        "coverage_pct": g["predicted_specialties"].notna().mean() * 100
    }))
    .reset_index()
)

display(coverage_by_window.rename(columns={
    "time_bucket": "Time Window",
    "total": "Total Triggers",
    "covered": "Covered",
    "coverage_pct": "Coverage %"
}).reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 9 — Summary

Markov baseline performance numbers.
All subsequent models evaluated on the same test set using the same metrics.
"""))

summary_k3 = results_df[results_df["K"] == 3][[
    "Time Window", "Coverage", "Hit@K", "Precision@K", "Recall@K", "NDCG@K"
]].reset_index(drop=True)

display(Markdown("### Markov Baseline — K=3 Summary"))
display(summary_k3)

display(Markdown("""
---
**These numbers are the baseline.**
Any model that does not beat these metrics at K=3 across all three windows
is not worth the added complexity.

Next — SASRec evaluation on the same test set with the same metrics.
---
"""))
