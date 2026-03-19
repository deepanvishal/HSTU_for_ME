# ============================================================
# NB_07 — Markov Baseline Visualization
# Purpose : Visualize Markov baseline evaluation metrics
#           All computation done in BigQuery
#           Python for visualization only
# Source  : A870800_gen_rec_markov_metrics
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
WINDOWS  = ["T0_30", "T30_60", "T60_180"]
COHORTS  = ["Adult_Female", "Adult_Male", "Senior", "Children"]
METRICS  = ["hit_at_k", "precision_at_k", "recall_at_k", "ndcg_at_k"]
METRIC_LABELS = {
    "hit_at_k":       "Hit@K",
    "precision_at_k": "Precision@K",
    "recall_at_k":    "Recall@K",
    "ndcg_at_k":      "NDCG@K"
}
WINDOW_COLORS  = {"T0_30": "#5DBE7E", "T30_60": "#F7C948", "T60_180": "#F4845F"}
WINDOW_MARKERS = {"T0_30": "o",       "T30_60": "s",       "T60_180": "^"}
COHORT_COLORS  = {
    "Adult_Female": "#4C9BE8",
    "Adult_Male":   "#F4845F",
    "Senior":       "#5DBE7E",
    "Children":     "#F7C948",
    "ALL":          "#333333"
}

display(Markdown("""
---
# NB 07 — Markov Baseline
## Evaluation Results

The Markov baseline computes `P(next_specialty | trigger_dx)` from
pre-2024 training transitions and applies it to 2024+ test triggers.

**Evaluation grain:** One evaluation per `member + trigger + time_bucket`.
True label set = all specialties the member visited within that window.

**Metrics:**
- **Hit@K** — 1 if any top K prediction is in the true label set
- **Precision@K** — fraction of top K predictions that are correct
- **Recall@K** — fraction of true labels found in top K
- **NDCG@K** — position-weighted score — correct labels ranked higher score better

All subsequent models are evaluated on the same test set using these same metrics.
These numbers are the performance floor.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df = client.query(f"""
    SELECT *
    FROM `{DATASET}.A870800_gen_rec_markov_metrics`
    ORDER BY time_bucket, k, member_segment
""").to_dataframe()

for col in METRICS + ["n_evaluated"]:
    df[col] = df[col].astype(float)

overall = df[df["member_segment"] == "ALL"].copy()
by_segment = df[df["member_segment"] != "ALL"].copy()

display(Markdown("### Markov Baseline — Full Results Table"))
display(df.rename(columns={
    "time_bucket":    "Time Window",
    "k":              "K",
    "member_segment": "Segment",
    "n_evaluated":    "N Evaluated",
    "hit_at_k":       "Hit@K",
    "precision_at_k": "Precision@K",
    "recall_at_k":    "Recall@K",
    "ndcg_at_k":      "NDCG@K"
}).reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — OVERALL METRICS BY WINDOW AND K
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 1 — Overall Metrics by Window and K

Each line represents one time window.
X axis is K — the number of predictions shown to the member.
Y axis is the metric value.

A steeper rise from K=1 to K=5 means the model ranks correct
answers lower — it finds them but not at the top.
A flat line means the model either always or never predicts correctly
regardless of how many predictions are shown.
"""))

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for i, metric in enumerate(METRICS):
    ax = axes[i]
    for window in WINDOWS:
        sub = overall[overall["time_bucket"] == window].sort_values("k")
        ax.plot(sub["k"], sub[metric],
                color=WINDOW_COLORS[window],
                marker=WINDOW_MARKERS[window],
                linewidth=2, markersize=8, label=window)
        for _, row in sub.iterrows():
            ax.annotate(f"{row[metric]:.3f}",
                        (row["k"], row[metric]),
                        textcoords="offset points",
                        xytext=(6, 4), fontsize=8,
                        color=WINDOW_COLORS[window])
    ax.set_title(f"Markov Baseline — {METRIC_LABELS[metric]}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("K", fontsize=10)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
    ax.set_xticks(K_VALUES)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)

fig.suptitle("Markov Baseline — All Metrics by Window and K",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("markov_all_metrics.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — K=3 SUMMARY — SIDE BY SIDE BY WINDOW
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 2 — K=3 Summary

K=3 is the primary evaluation point — the model shows the top 3 specialties.
This section shows all four metrics at K=3 across the three time windows.
"""))

k3 = overall[overall["k"] == 3].sort_values("time_bucket")

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(WINDOWS))
width = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]
colors  = ["#4C9BE8", "#5DBE7E", "#F4845F", "#F7C948"]

for j, (metric, color, offset) in enumerate(zip(METRICS, colors, offsets)):
    vals = [k3[k3["time_bucket"] == w][metric].values[0]
            if len(k3[k3["time_bucket"] == w]) > 0 else 0
            for w in WINDOWS]
    bars = ax.bar(x + offset * width, vals, width,
                  label=METRIC_LABELS[metric], color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(WINDOWS)
ax.set_ylabel("Metric Value", fontsize=10)
ax.set_title("Markov Baseline — All Metrics at K=3 by Time Window",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("markov_k3_summary.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — METRICS BY COHORT
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 3 — Metrics by Member Segment

Does the Markov model perform consistently across cohorts?

Large gaps between cohorts indicate the model has stronger routing
signal for some populations — likely Seniors and Adult Female who
have more consistent specialist referral patterns.
"""))

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
axes = axes.flatten()

for i, metric in enumerate(METRICS):
    ax = axes[i]
    for cohort in COHORTS:
        sub = by_segment[
            (by_segment["member_segment"] == cohort) &
            (by_segment["k"] == 3)
        ].sort_values("time_bucket")
        if sub.empty:
            continue
        ax.plot(sub["time_bucket"], sub[metric],
                color=COHORT_COLORS[cohort],
                marker="o", linewidth=2, markersize=8, label=cohort)
        for _, row in sub.iterrows():
            ax.annotate(f"{row[metric]:.3f}",
                        (row["time_bucket"], row[metric]),
                        textcoords="offset points",
                        xytext=(5, 4), fontsize=7,
                        color=COHORT_COLORS[cohort])

    # Overall line for reference
    sub_all = overall[overall["k"] == 3].sort_values("time_bucket")
    ax.plot(sub_all["time_bucket"], sub_all[metric],
            color=COHORT_COLORS["ALL"], marker="D",
            linewidth=2, markersize=8, linestyle="--", label="ALL")

    ax.set_title(f"{METRIC_LABELS[metric]} at K=3 by Cohort",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Time Window", fontsize=10)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)

fig.suptitle("Markov Baseline — Performance by Member Segment (K=3)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("markov_by_cohort.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NDCG HEATMAP — WINDOW VS K
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 4 — NDCG Heatmap

NDCG across all windows and K values.
Darker green = higher NDCG = model ranks correct specialties higher.
This is the single most important metric for ranking quality.
"""))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for idx, (segment, title) in enumerate([("ALL", "All Members"), (None, "By Cohort")]):
    if segment == "ALL":
        pivot = overall.pivot_table(
            index="time_bucket", columns="k",
            values="ndcg_at_k", aggfunc="mean"
        ).reindex(WINDOWS)

        import seaborn as sns
        sns.heatmap(pivot, ax=axes[0], cmap="YlGn",
                    annot=True, fmt=".3f", annot_kws={"size": 11},
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "NDCG@K"})
        axes[0].set_title("NDCG — All Members", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("K", fontsize=10)
        axes[0].set_ylabel("Time Window", fontsize=10)
    else:
        # NDCG at K=3 per cohort per window
        pivot = by_segment[by_segment["k"] == 3].pivot_table(
            index="time_bucket", columns="member_segment",
            values="ndcg_at_k", aggfunc="mean"
        ).reindex(WINDOWS)

        sns.heatmap(pivot, ax=axes[1], cmap="YlGn",
                    annot=True, fmt=".3f", annot_kws={"size": 10},
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "NDCG@K=3"})
        axes[1].set_title("NDCG at K=3 — By Cohort", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Member Segment", fontsize=10)
        axes[1].set_ylabel("Time Window", fontsize=10)
        plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

fig.suptitle("Markov Baseline — NDCG Heatmap",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("markov_ndcg_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — BASELINE SUMMARY CARD
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 5 — Baseline Summary

These are the numbers every subsequent model must beat.
K=3 is the primary evaluation point.
T0_30 is the most clinically urgent window.
"""))

summary_rows = []
for window in WINDOWS:
    row = {"Time Window": window}
    for metric in METRICS:
        val = overall[
            (overall["time_bucket"] == window) &
            (overall["k"] == 3)
        ][metric].values
        row[METRIC_LABELS[metric]] = f"{val[0]:.4f}" if len(val) > 0 else "N/A"
    n = overall[
        (overall["time_bucket"] == window) &
        (overall["k"] == 3)
    ]["n_evaluated"].values
    row["N Evaluated"] = f"{int(n[0]):,}" if len(n) > 0 else "N/A"
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
display(summary_df.reset_index(drop=True))

display(Markdown("""
---
**Next:** SASRec model trained on 1% sample.
Evaluated on same test set with same metrics.
Any improvement over these numbers justifies the model complexity.
---
"""))
