# ============================================================
# NB_PMA_02 — SASRec_T30_insights.py
# Purpose : SASRec-only, T0_30-only deep dive
#           Focus on WHERE predictions are good and WHY
#           Goal: actionable insights, not model comparison
# Sources : A870800_gen_rec_pma_provider_summary_5pct
#           A870800_gen_rec_pma_dx_summary_5pct
#           A870800_gen_rec_pma_transition_bucket_5pct
#           A870800_gen_rec_provider_eval_5pct
# Sections:
#   A — SASRec T30 Snapshot
#   B — Highest Volume Outbound Providers + Performance
#   C — Highest Volume Inbound Providers + Performance
#   D — DX with Highest Predictive Power
#   E — Providers with Highest Inbound Accuracy
#   F — Performance over Transition Volumes
#   G — Performance over Member Segments
#   H — Performance over Specialties
# ============================================================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from google.cloud import bigquery
from IPython.display import display, Markdown

DS     = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
client = bigquery.Client(project="anbc-hcb-dev")

MODEL   = "SASRec"
WINDOW  = "T0_30"
COLOR   = "#4C72B0"    # SASRec blue throughout
ACCENT  = "#DD8452"    # highlight color

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    10,
})

def fmt_pct(v):
    return f"{v*100:.1f}%" if pd.notna(v) else "—"

display(Markdown(f"""
# SASRec — T0-30 Day Predictions: Where Do We Win?
**Model:** {MODEL} | **Window:** Next 30 days | **Sample:** 5pct

> Focus: Identify where predictions are reliable enough to act on.
> A prediction is actionable when Hit@3 ≥ 0.20 and volume ≥ 100 triggers.
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section A — SASRec T30 Overall Snapshot"))

snap = client.query(f"""
    SELECT
        COUNT(*)                                         AS n_triggers
        ,SUM(tp)                                         AS total_tp
        ,SUM(fp)                                         AS total_fp
        ,SUM(fn)                                         AS total_fn
        ,ROUND(AVG(hit_at_1), 4)                         AS hit_at_1
        ,ROUND(AVG(hit_at_3), 4)                         AS hit_at_3
        ,ROUND(AVG(hit_at_5), 4)                         AS hit_at_5
        ,ROUND(AVG(ndcg_at_3), 4)                        AS ndcg_at_3
        ,ROUND(AVG(precision_at_3), 4)                   AS precision_at_3
        ,ROUND(AVG(recall_at_3), 4)                      AS recall_at_3
        ,ROUND(SUM(tp) / NULLIF(SUM(tp) + SUM(fp), 0), 4) AS overall_precision
        ,ROUND(SUM(tp) / NULLIF(SUM(tp) + SUM(fn), 0), 4) AS overall_recall
    FROM `{DS}.A870800_gen_rec_provider_eval_5pct`
    WHERE model = '{MODEL}'
      AND time_bucket = '{WINDOW}'
""").to_dataframe()

s = snap.iloc[0]
display(Markdown(f"""
| Metric | Value |
|--------|-------|
| Total triggers | {s['n_triggers']:,.0f} |
| Hit@1 | {fmt_pct(s['hit_at_1'])} |
| Hit@3 | {fmt_pct(s['hit_at_3'])} |
| Hit@5 | {fmt_pct(s['hit_at_5'])} |
| NDCG@3 | {s['ndcg_at_3']:.4f} |
| Precision@3 | {fmt_pct(s['precision_at_3'])} |
| Recall@3 | {fmt_pct(s['recall_at_3'])} |
| Overall Precision (TP/TP+FP) | {fmt_pct(s['overall_precision'])} |
| Overall Recall (TP/TP+FN) | {fmt_pct(s['overall_recall'])} |
| Total TP | {s['total_tp']:,.0f} |
| Total FP | {s['total_fp']:,.0f} |
| Total FN | {s['total_fn']:,.0f} |
"""))

print(f"Section A done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — HIGHEST VOLUME OUTBOUND PROVIDERS + PERFORMANCE
# Where are members coming FROM most often — and how well do we predict?
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section B — Highest Volume Outbound Providers"))
display(Markdown("*For members leaving these providers — volume vs prediction accuracy*"))

outbound_vol = client.query(f"""
    SELECT
        srv_prvdr_id
        ,COALESCE(provider_name, CAST(srv_prvdr_id AS STRING)) AS provider_name
        ,COALESCE(specialty_desc, 'Unknown')                   AS specialty_desc
        ,n_triggers
        ,total_outbound_transitions
        ,total_tp, total_fp, total_fn
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_3
        ,overall_precision
    FROM `{DS}.A870800_gen_rec_pma_provider_summary_5pct`
    WHERE model            = '{MODEL}'
      AND time_bucket      = '{WINDOW}'
      AND provider_direction = 'Outbound'
      AND n_triggers       > 0
    ORDER BY n_triggers DESC
    LIMIT 20
""").to_dataframe()

outbound_vol["label"] = outbound_vol.apply(
    lambda r: f"{str(r['provider_name'])[:30]}" if pd.notna(r["provider_name"])
              else str(r["srv_prvdr_id"]), axis=1
)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Top 20 Outbound Providers — Volume vs Accuracy (T0_30)",
             fontsize=13, fontweight="bold")

# Left: bar chart sorted by volume
ax = axes[0]
top20 = outbound_vol.sort_values("n_triggers", ascending=True)
bars  = ax.barh(top20["label"], top20["n_triggers"], color=COLOR, alpha=0.8)
ax.set_xlabel("Number of Triggers")
ax.set_title("Volume (n triggers)")
# Annotate hit@3 on bars
for bar, (_, row) in zip(bars, top20.iterrows()):
    ax.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height()/2,
            f"Hit@3: {fmt_pct(row['hit_at_3'])}",
            va="center", ha="left", fontsize=7, color="white", fontweight="bold")

# Right: scatter — volume vs hit@3, sized by total_transitions
ax2 = axes[1]
sc = ax2.scatter(outbound_vol["n_triggers"], outbound_vol["hit_at_3"],
                 s=outbound_vol["total_outbound_transitions"].clip(upper=5000) / 20 + 30,
                 c=outbound_vol["hit_at_3"], cmap="RdYlGn",
                 vmin=0, vmax=0.5, alpha=0.8, edgecolors="gray", linewidth=0.5)
plt.colorbar(sc, ax=ax2, label="Hit@3")
ax2.axhline(0.20, color="gray", linestyle="--", linewidth=1, label="Actionable threshold (0.20)")
ax2.set_xlabel("Number of Triggers (volume)")
ax2.set_ylabel("Hit@3")
ax2.set_title("Volume vs Accuracy\n(bubble size = outbound transitions)")
ax2.legend(fontsize=8)

# Annotate top 5 by volume
for _, row in outbound_vol.head(5).iterrows():
    ax2.annotate(str(row["label"])[:20],
                 (row["n_triggers"], row["hit_at_3"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)

plt.tight_layout()
plt.show()

display(Markdown("### Top 20 Outbound Providers — Full Table"))
display(outbound_vol[[
    "provider_name", "specialty_desc", "n_triggers",
    "total_outbound_transitions", "hit_at_1", "hit_at_3", "hit_at_5",
    "ndcg_at_3", "overall_precision"
]].rename(columns={
    "provider_name": "Provider", "specialty_desc": "Specialty",
    "n_triggers": "N Triggers", "total_outbound_transitions": "Training Transitions",
    "hit_at_1": "Hit@1", "hit_at_3": "Hit@3", "hit_at_5": "Hit@5",
    "ndcg_at_3": "NDCG@3", "overall_precision": "Precision"
}).reset_index(drop=True))

print(f"Section B done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — HIGHEST VOLUME INBOUND PROVIDERS + PERFORMANCE
# Which providers are most often predicted as next destination?
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section C — Highest Volume Inbound Providers"))
display(Markdown("*Providers most frequently predicted as next destination — and how accurate*"))

inbound_vol = client.query(f"""
    SELECT
        srv_prvdr_id
        ,COALESCE(provider_name, CAST(srv_prvdr_id AS STRING)) AS provider_name
        ,COALESCE(specialty_desc, 'Unknown')                   AS specialty_desc
        ,n_triggers                                            AS n_times_predicted
        ,total_inbound_transitions
        ,total_tp, total_fp
        ,overall_precision
    FROM `{DS}.A870800_gen_rec_pma_provider_summary_5pct`
    WHERE model            = '{MODEL}'
      AND time_bucket      = '{WINDOW}'
      AND provider_direction = 'Inbound'
      AND n_triggers       > 0
    ORDER BY n_triggers DESC
    LIMIT 20
""").to_dataframe()

inbound_vol["label"] = inbound_vol.apply(
    lambda r: f"{str(r['provider_name'])[:30]}" if pd.notna(r["provider_name"])
              else str(r["srv_prvdr_id"]), axis=1
)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Top 20 Inbound Providers — Volume vs Accuracy (T0_30)",
             fontsize=13, fontweight="bold")

ax = axes[0]
top20i = inbound_vol.sort_values("n_times_predicted", ascending=True)
bars = ax.barh(top20i["label"], top20i["n_times_predicted"], color=ACCENT, alpha=0.8)
ax.set_xlabel("Times Predicted")
ax.set_title("Prediction Volume")
for bar, (_, row) in zip(bars, top20i.iterrows()):
    ax.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height()/2,
            f"Prec: {fmt_pct(row['overall_precision'])}",
            va="center", ha="left", fontsize=7, color="white", fontweight="bold")

ax2 = axes[1]
sc = ax2.scatter(inbound_vol["n_times_predicted"], inbound_vol["overall_precision"],
                 s=inbound_vol["total_inbound_transitions"].clip(upper=5000) / 20 + 30,
                 c=inbound_vol["overall_precision"], cmap="RdYlGn",
                 vmin=0, vmax=0.5, alpha=0.8, edgecolors="gray", linewidth=0.5)
plt.colorbar(sc, ax=ax2, label="Precision")
ax2.axhline(0.20, color="gray", linestyle="--", linewidth=1, label="Actionable threshold (0.20)")
ax2.set_xlabel("Times Predicted (volume)")
ax2.set_ylabel("Precision (TP / Times Predicted)")
ax2.set_title("Prediction Volume vs Precision\n(bubble size = inbound transitions)")
ax2.legend(fontsize=8)

for _, row in inbound_vol.head(5).iterrows():
    ax2.annotate(str(row["label"])[:20],
                 (row["n_times_predicted"], row["overall_precision"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)

plt.tight_layout()
plt.show()

display(Markdown("### Top 20 Inbound Providers — Full Table"))
display(inbound_vol[[
    "provider_name", "specialty_desc", "n_times_predicted",
    "total_inbound_transitions", "total_tp", "total_fp", "overall_precision"
]].rename(columns={
    "provider_name": "Provider", "specialty_desc": "Specialty",
    "n_times_predicted": "Times Predicted",
    "total_inbound_transitions": "Training Transitions",
    "total_tp": "TP", "total_fp": "FP", "overall_precision": "Precision"
}).reset_index(drop=True))

print(f"Section C done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — DX WITH HIGHEST PREDICTIVE POWER
# When this diagnosis fires — we predict the next provider well
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section D — Diagnosis Codes with Highest Predictive Power"))
display(Markdown("*Minimum 100 triggers — DX codes where SASRec is most reliable*"))

dx_power = client.query(f"""
    SELECT
        trigger_dx
        ,COALESCE(trigger_dx_desc, trigger_dx)           AS trigger_dx_desc
        ,n_triggers
        ,total_tp, total_fp, total_fn
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_3, precision_at_3, recall_at_3
        ,total_volume
        ,accuracy_rank
    FROM `{DS}.A870800_gen_rec_pma_dx_summary_5pct`
    WHERE model         = '{MODEL}'
      AND time_bucket   = '{WINDOW}'
      AND n_triggers   >= 100
    ORDER BY hit_at_3 DESC
    LIMIT 20
""").to_dataframe()

dx_power["label"] = dx_power.apply(
    lambda r: f"{str(r['trigger_dx_desc'])[:45]}\n({r['trigger_dx']})"
              if r["trigger_dx_desc"] != r["trigger_dx"] else r["trigger_dx"],
    axis=1
)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("Top 20 DX by Predictive Power — SASRec T0_30 (min 100 triggers)",
             fontsize=13, fontweight="bold")

ax = axes[0]
top20dx = dx_power.sort_values("hit_at_3", ascending=True)
colors  = [COLOR if v >= 0.20 else "#cccccc" for v in top20dx["hit_at_3"]]
ax.barh(top20dx["label"], top20dx["hit_at_3"], color=colors, alpha=0.85)
ax.axvline(0.20, color="red", linestyle="--", linewidth=1, label="Actionable threshold (0.20)")
ax.set_xlabel("Hit@3")
ax.set_title("Hit@3 by Diagnosis Code")
ax.legend(fontsize=8)

ax2 = axes[1]
sc = ax2.scatter(dx_power["n_triggers"], dx_power["hit_at_3"],
                 s=60, c=dx_power["hit_at_3"], cmap="RdYlGn",
                 vmin=0, vmax=0.5, alpha=0.85, edgecolors="gray", linewidth=0.5)
plt.colorbar(sc, ax=ax2, label="Hit@3")
ax2.axhline(0.20, color="gray", linestyle="--", linewidth=1, label="Actionable threshold")
ax2.set_xlabel("Number of Triggers (volume)")
ax2.set_ylabel("Hit@3")
ax2.set_title("Volume vs Accuracy by DX")
ax2.legend(fontsize=8)
for _, row in dx_power.head(5).iterrows():
    ax2.annotate(row["trigger_dx"],
                 (row["n_triggers"], row["hit_at_3"]),
                 textcoords="offset points", xytext=(5, 5), fontsize=7)

plt.tight_layout()
plt.show()

actionable = dx_power[dx_power["hit_at_3"] >= 0.20]
display(Markdown(f"""
**{len(actionable)} of top-20 DX codes have Hit@3 ≥ 20% — actionable for care management**
"""))
display(dx_power[[
    "trigger_dx", "trigger_dx_desc", "n_triggers",
    "hit_at_1", "hit_at_3", "hit_at_5", "ndcg_at_3", "precision_at_3"
]].rename(columns={
    "trigger_dx": "DX", "trigger_dx_desc": "Description",
    "n_triggers": "N Triggers", "hit_at_1": "Hit@1",
    "hit_at_3": "Hit@3", "hit_at_5": "Hit@5",
    "ndcg_at_3": "NDCG@3", "precision_at_3": "Precision@3"
}).reset_index(drop=True))

print(f"Section D done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — PROVIDERS WITH HIGHEST INBOUND ACCURACY
# These providers — when predicted — are almost always correct
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section E — Providers with Highest Inbound Accuracy"))
display(Markdown("*When SASRec predicts these providers — it is almost always right*"))

inbound_acc = client.query(f"""
    SELECT
        srv_prvdr_id
        ,COALESCE(provider_name, CAST(srv_prvdr_id AS STRING)) AS provider_name
        ,COALESCE(specialty_desc, 'Unknown')                   AS specialty_desc
        ,n_triggers                                            AS n_times_predicted
        ,total_inbound_transitions
        ,total_tp, total_fp
        ,overall_precision
    FROM `{DS}.A870800_gen_rec_pma_provider_summary_5pct`
    WHERE model            = '{MODEL}'
      AND time_bucket      = '{WINDOW}'
      AND provider_direction = 'Inbound'
      AND n_triggers       >= 50
    ORDER BY overall_precision DESC
    LIMIT 20
""").to_dataframe()

inbound_acc["label"] = inbound_acc.apply(
    lambda r: f"{str(r['provider_name'])[:30]}" if pd.notna(r["provider_name"])
              else str(r["srv_prvdr_id"]), axis=1
)

fig, ax = plt.subplots(figsize=(12, 8))
fig.suptitle("Top 20 Inbound Providers by Accuracy — SASRec T0_30 (min 50 predictions)",
             fontsize=13, fontweight="bold")

top20a = inbound_acc.sort_values("overall_precision", ascending=True)
colors  = [COLOR if v >= 0.30 else ACCENT for v in top20a["overall_precision"]]
bars    = ax.barh(top20a["label"], top20a["overall_precision"], color=colors, alpha=0.85)
ax.axvline(0.30, color="red", linestyle="--", linewidth=1, label="High accuracy threshold (0.30)")
ax.set_xlabel("Precision (TP / Times Predicted)")
ax.set_title("Inbound Provider Precision")
ax.legend(fontsize=8)

for bar, (_, row) in zip(bars, top20a.iterrows()):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"n={row['n_times_predicted']:,}",
            va="center", fontsize=7, color="gray")

plt.tight_layout()
plt.show()

high_acc = inbound_acc[inbound_acc["overall_precision"] >= 0.30]
display(Markdown(f"""
**{len(high_acc)} providers have precision ≥ 30% — when predicted, nearly certain visit**
"""))
display(inbound_acc[[
    "provider_name", "specialty_desc", "n_times_predicted",
    "total_inbound_transitions", "total_tp", "total_fp", "overall_precision"
]].rename(columns={
    "provider_name": "Provider", "specialty_desc": "Specialty",
    "n_times_predicted": "Times Predicted",
    "total_inbound_transitions": "Training Transitions",
    "total_tp": "TP", "total_fp": "FP", "overall_precision": "Precision"
}).reset_index(drop=True))

print(f"Section E done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — PERFORMANCE OVER TRANSITION VOLUMES
# Does SASRec capture high-transition pathways that Markov dominates?
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section F — SASRec Performance by Transition Evidence"))
display(Markdown("*Low/Medium/High buckets — where does SASRec win vs where does evidence help?*"))

bucket_df = client.query(f"""
    SELECT
        transition_bucket
        ,n_triggers
        ,avg_transition_count
        ,threshold_p33
        ,threshold_p67
        ,total_tp, total_fp, total_fn
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_3, ndcg_at_5
        ,overall_precision, overall_recall
    FROM `{DS}.A870800_gen_rec_pma_transition_bucket_5pct`
    WHERE model       = '{MODEL}'
      AND time_bucket = '{WINDOW}'
    ORDER BY CASE transition_bucket
        WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END
""").to_dataframe()

p33 = bucket_df["threshold_p33"].iloc[0] if len(bucket_df) > 0 else 0
p67 = bucket_df["threshold_p67"].iloc[0] if len(bucket_df) > 0 else 0

display(Markdown(f"""
**Bucket thresholds:**
- Low: ≤ {p33:,.0f} transitions | Medium: {p33:,.0f}–{p67:,.0f} | High: > {p67:,.0f}
"""))

bcolors = {"Low": "#d9534f", "Medium": "#f0ad4e", "High": "#5cb85c"}
buckets = ["Low", "Medium", "High"]
bucket_df = bucket_df.set_index("transition_bucket").reindex(buckets).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("SASRec Performance by Transition Evidence Bucket — T0_30",
             fontsize=13, fontweight="bold")

for ax, metric, ylabel in zip(axes,
    ["hit_at_3",        "ndcg_at_3",       "overall_precision"],
    ["Hit@3",           "NDCG@3",           "Overall Precision"]):
    colors = [bcolors.get(b, "#888888") for b in bucket_df["transition_bucket"]]
    bars   = ax.bar(bucket_df["transition_bucket"],
                    bucket_df[metric].fillna(0),
                    color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.set_ylim(0, max(bucket_df[metric].fillna(0).max() * 1.3, 0.05))
    for bar, (_, row) in zip(bars, bucket_df.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.003,
                f"{row[metric]:.3f}\nn={row['n_triggers']:,.0f}",
                ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.show()

display(bucket_df[[
    "transition_bucket", "n_triggers", "avg_transition_count",
    "hit_at_1", "hit_at_3", "hit_at_5",
    "ndcg_at_3", "overall_precision", "overall_recall",
    "total_tp", "total_fp", "total_fn"
]].rename(columns={
    "transition_bucket": "Bucket",
    "n_triggers": "N Triggers",
    "avg_transition_count": "Avg Transitions",
    "hit_at_1": "Hit@1", "hit_at_3": "Hit@3", "hit_at_5": "Hit@5",
    "ndcg_at_3": "NDCG@3",
    "overall_precision": "Precision", "overall_recall": "Recall",
    "total_tp": "TP", "total_fp": "FP", "total_fn": "FN"
}).reset_index(drop=True))

print(f"Section F done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION G — PERFORMANCE OVER MEMBER SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section G — SASRec Performance by Member Segment"))

seg_df = client.query(f"""
    SELECT
        member_segment
        ,COUNT(*)                                        AS n_triggers
        ,SUM(tp)                                         AS total_tp
        ,SUM(fp)                                         AS total_fp
        ,SUM(fn)                                         AS total_fn
        ,ROUND(AVG(hit_at_1), 4)                         AS hit_at_1
        ,ROUND(AVG(hit_at_3), 4)                         AS hit_at_3
        ,ROUND(AVG(hit_at_5), 4)                         AS hit_at_5
        ,ROUND(AVG(ndcg_at_3), 4)                        AS ndcg_at_3
        ,ROUND(AVG(precision_at_3), 4)                   AS precision_at_3
        ,ROUND(SUM(tp) / NULLIF(SUM(tp) + SUM(fp), 0), 4) AS overall_precision
    FROM `{DS}.A870800_gen_rec_provider_eval_5pct`
    WHERE model       = '{MODEL}'
      AND time_bucket = '{WINDOW}'
      AND member_segment IS NOT NULL
    GROUP BY member_segment
    ORDER BY n_triggers DESC
""").to_dataframe()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("SASRec Performance by Member Segment — T0_30",
             fontsize=13, fontweight="bold")

seg_sorted = seg_df.sort_values("hit_at_3", ascending=True)

ax = axes[0]
ax.barh(seg_sorted["member_segment"], seg_sorted["hit_at_3"],
        color=COLOR, alpha=0.85)
ax.set_xlabel("Hit@3")
ax.set_title("Hit@3 by Segment")
for i, (_, row) in enumerate(seg_sorted.iterrows()):
    ax.text(row["hit_at_3"] + 0.002, i,
            f"n={row['n_triggers']:,}", va="center", fontsize=8)

ax2 = axes[1]
ax2.scatter(seg_df["n_triggers"], seg_df["hit_at_3"],
            s=100, color=COLOR, alpha=0.85, edgecolors="gray")
for _, row in seg_df.iterrows():
    ax2.annotate(str(row["member_segment"])[:20],
                 (row["n_triggers"], row["hit_at_3"]),
                 textcoords="offset points", xytext=(5, 3), fontsize=8)
ax2.axhline(0.20, color="gray", linestyle="--", linewidth=1)
ax2.set_xlabel("Number of Triggers")
ax2.set_ylabel("Hit@3")
ax2.set_title("Volume vs Accuracy by Segment")

plt.tight_layout()
plt.show()

display(seg_df[[
    "member_segment", "n_triggers", "hit_at_1", "hit_at_3", "hit_at_5",
    "ndcg_at_3", "precision_at_3", "overall_precision",
    "total_tp", "total_fp", "total_fn"
]].rename(columns={
    "member_segment": "Segment", "n_triggers": "N Triggers",
    "hit_at_1": "Hit@1", "hit_at_3": "Hit@3", "hit_at_5": "Hit@5",
    "ndcg_at_3": "NDCG@3", "precision_at_3": "Precision@3",
    "overall_precision": "Overall Precision",
    "total_tp": "TP", "total_fp": "FP", "total_fn": "FN"
}).reset_index(drop=True))

print(f"Section G done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION H — PERFORMANCE OVER SPECIALTIES
# H1: Highest volume specialty
# H2: Specialty with best predictions
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section H — SASRec Performance by Specialty"))
display(Markdown("*From the outbound providers — aggregate by specialty*"))

spec_df = client.query(f"""
    SELECT
        COALESCE(specialty_desc, 'Unknown')              AS specialty_desc
        ,SUM(n_triggers)                                 AS total_triggers
        ,ROUND(SUM(total_tp * 1.0) /
               NULLIF(SUM(total_tp + total_fp), 0), 4)   AS overall_precision
        ,ROUND(AVG(hit_at_3), 4)                         AS avg_hit_at_3
        ,ROUND(AVG(hit_at_5), 4)                         AS avg_hit_at_5
        ,ROUND(AVG(ndcg_at_3), 4)                        AS avg_ndcg_at_3
        ,SUM(total_tp)                                   AS total_tp
        ,SUM(total_fp)                                   AS total_fp
        ,SUM(total_fn)                                   AS total_fn
        ,COUNT(DISTINCT srv_prvdr_id)                    AS n_providers
    FROM `{DS}.A870800_gen_rec_pma_provider_summary_5pct`
    WHERE model            = '{MODEL}'
      AND time_bucket      = '{WINDOW}'
      AND provider_direction = 'Outbound'
      AND specialty_desc  != 'Unknown'
    GROUP BY specialty_desc
    HAVING SUM(n_triggers) >= 50
    ORDER BY total_triggers DESC
""").to_dataframe()

display(Markdown(f"**{len(spec_df)} specialties with ≥ 50 triggers**"))

# H1 — Top 15 by volume
top_vol  = spec_df.sort_values("total_triggers", ascending=False).head(15)
# H2 — Top 15 by avg_hit_at_3
top_acc  = spec_df.sort_values("avg_hit_at_3",   ascending=False).head(15)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("SASRec by Specialty — T0_30 (min 50 triggers, outbound)",
             fontsize=13, fontweight="bold")

# H1: Highest volume
ax = axes[0]
tv = top_vol.sort_values("total_triggers", ascending=True)
bars = ax.barh(tv["specialty_desc"], tv["total_triggers"],
               color=COLOR, alpha=0.85)
ax.set_xlabel("Total Triggers")
ax.set_title("H1: Highest Volume Specialties")
for bar, (_, row) in zip(bars, tv.iterrows()):
    ax.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height()/2,
            f"Hit@3: {fmt_pct(row['avg_hit_at_3'])}",
            va="center", ha="left", fontsize=7, color="white", fontweight="bold")

# H2: Best predictions
ax2 = axes[1]
ta  = top_acc.sort_values("avg_hit_at_3", ascending=True)
colors = [COLOR if v >= 0.20 else "#cccccc" for v in ta["avg_hit_at_3"]]
bars2 = ax2.barh(ta["specialty_desc"], ta["avg_hit_at_3"],
                 color=colors, alpha=0.85)
ax2.axvline(0.20, color="red", linestyle="--", linewidth=1)
ax2.set_xlabel("Avg Hit@3")
ax2.set_title("H2: Most Predictable Specialties")
for bar, (_, row) in zip(bars2, ta.iterrows()):
    ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"n={row['total_triggers']:,}",
             va="center", fontsize=7, color="gray")

plt.tight_layout()
plt.show()

display(Markdown("### H1: Top 15 Specialties by Volume"))
display(top_vol[[
    "specialty_desc", "n_providers", "total_triggers",
    "avg_hit_at_3", "avg_hit_at_5", "avg_ndcg_at_3", "overall_precision",
    "total_tp", "total_fp", "total_fn"
]].rename(columns={
    "specialty_desc": "Specialty", "n_providers": "N Providers",
    "total_triggers": "N Triggers", "avg_hit_at_3": "Avg Hit@3",
    "avg_hit_at_5": "Avg Hit@5", "avg_ndcg_at_3": "Avg NDCG@3",
    "overall_precision": "Precision",
    "total_tp": "TP", "total_fp": "FP", "total_fn": "FN"
}).reset_index(drop=True))

display(Markdown("### H2: Top 15 Specialties by Prediction Accuracy"))
display(top_acc[[
    "specialty_desc", "n_providers", "total_triggers",
    "avg_hit_at_3", "avg_hit_at_5", "avg_ndcg_at_3", "overall_precision"
]].rename(columns={
    "specialty_desc": "Specialty", "n_providers": "N Providers",
    "total_triggers": "N Triggers", "avg_hit_at_3": "Avg Hit@3",
    "avg_hit_at_5": "Avg Hit@5", "avg_ndcg_at_3": "Avg NDCG@3",
    "overall_precision": "Precision"
}).reset_index(drop=True))

print(f"Section H done — {time.time()-t0:.1f}s")
print("\nNB_PMA_02 complete")
