# ============================================================
# NB_PMA_01 — post_model_analysis.py
# Purpose : Post-model analysis visualization
#           BQ pulls pre-aggregated tables, Python only visualizes
# Sources : A870800_gen_rec_pma_transition_bucket_5pct  SQL_PMA_01
#           A870800_gen_rec_pma_dx_summary_5pct          SQL_PMA_02
#           A870800_gen_rec_pma_provider_dx_summary_5pct SQL_PMA_03
#           A870800_gen_rec_pma_provider_summary_5pct    SQL_PMA_04
# Sections:
#   A — Transition Bucket Performance (Low/Med/High evidence)
#   B — Top 10 DX by Volume
#   C — Top 10 DX by Accuracy
#   D — Top 10 Outbound Providers
#   E — Top 10 Inbound Providers
#   F — Provider + DX Transition Table (Mandatory 5a)
#   G — Member Predictability
# ============================================================

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from google.cloud import bigquery
from IPython.display import display, Markdown

DS     = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
client = bigquery.Client(project="anbc-hcb-dev")

MODELS   = ["SASRec", "BERT4Rec", "HSTU", "Markov"]
MCOLORS  = {"SASRec": "#4C72B0", "BERT4Rec": "#DD8452",
             "HSTU": "#8172B2",   "Markov": "#55A868"}
WINDOWS  = ["T0_30", "T30_60", "T60_180"]
WLABELS  = {"T0_30": "0-30 Days", "T30_60": "30-60 Days", "T60_180": "60-180 Days"}
BUCKETS  = ["Low", "Medium", "High"]
BCOLORS  = {"Low": "#d9534f", "Medium": "#f0ad4e", "High": "#5cb85c"}

plt.rcParams.update({
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size":       10,
})

display(Markdown("""
# Provider Recommendation Model — Post-Model Analysis
**Sample:** 5pct | **Models:** SASRec, BERT4Rec, HSTU, Markov
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — TRANSITION EVIDENCE BUCKET PERFORMANCE
# Goal: Show Markov dominates High evidence, DL models dominate Low/Medium
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section A — Performance by Transition Evidence Bucket"))

bucket_df = client.query(f"""
    SELECT
        transition_bucket, model, time_bucket,
        n_triggers, avg_transition_count,
        threshold_p33, threshold_p67,
        hit_at_1, hit_at_3, hit_at_5,
        ndcg_at_3, ndcg_at_5,
        overall_precision, overall_recall
    FROM `{DS}.A870800_gen_rec_pma_transition_bucket_5pct`
    ORDER BY time_bucket, transition_bucket, model
""").to_dataframe()

print(f"Loaded {len(bucket_df):,} rows")
p33 = bucket_df["threshold_p33"].iloc[0]
p67 = bucket_df["threshold_p67"].iloc[0]
display(Markdown(f"""
**Bucket thresholds (by trigger_dx transition count in training):**
- Low:    ≤ {p33:,.0f} transitions
- Medium: {p33:,.0f} – {p67:,.0f} transitions
- High:   > {p67:,.0f} transitions
"""))

# One chart per window — grouped bar: bucket on x, model as hue, hit@3 on y
for window in WINDOWS:
    sub = bucket_df[bucket_df["time_bucket"] == window].copy()
    if sub.empty:
        continue

    sub["transition_bucket"] = pd.Categorical(sub["transition_bucket"],
                                               categories=BUCKETS, ordered=True)
    sub = sub.sort_values("transition_bucket")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Transition Evidence Bucket Performance — {WLABELS[window]}",
                 fontsize=13, fontweight="bold")

    for ax, metric, ylabel in zip(axes,
                                   ["hit_at_3",       "ndcg_at_3"],
                                   ["Hit@3",           "NDCG@3"]):
        pivot = sub.pivot(index="transition_bucket", columns="model",
                          values=metric).reindex(BUCKETS)
        x     = np.arange(len(BUCKETS))
        width = 0.18
        for j, model in enumerate([m for m in MODELS if m in pivot.columns]):
            ax.bar(x + j * width, pivot[model].fillna(0),
                   width, label=model, color=MCOLORS[model], alpha=0.85)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(BUCKETS)
        ax.set_xlabel("Transition Evidence Bucket")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.set_ylim(0, min(1.0, (pivot.max().max() or 0.1) * 1.3))

    plt.tight_layout()
    plt.show()

    # Summary table
    pivot_hit  = sub.pivot(index="transition_bucket", columns="model", values="hit_at_3").reindex(BUCKETS).round(4)
    pivot_ndcg = sub.pivot(index="transition_bucket", columns="model", values="ndcg_at_3").reindex(BUCKETS).round(4)
    display(Markdown(f"**Hit@3 by bucket — {WLABELS[window]}**"))
    display(pivot_hit)
    display(Markdown(f"**NDCG@3 by bucket — {WLABELS[window]}**"))
    display(pivot_ndcg)

print(f"Section A done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — TOP 10 DX BY VOLUME
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section B — Top 10 Trigger DX by Volume"))

dx_df = client.query(f"""
    SELECT
        trigger_dx, trigger_dx_desc, model, time_bucket,
        n_triggers, total_tp, total_fp, total_fn,
        hit_at_1, hit_at_3, hit_at_5,
        ndcg_at_3, precision_at_3, recall_at_3,
        volume_rank, accuracy_rank, total_volume,
        avg_hit_at_3_all_models
    FROM `{DS}.A870800_gen_rec_pma_dx_summary_5pct`
    ORDER BY total_volume DESC, model, time_bucket
""").to_dataframe()

print(f"Loaded {len(dx_df):,} rows")

for window in WINDOWS:
    sub = dx_df[(dx_df["time_bucket"] == window) &
                (dx_df["volume_rank"] <= 10)].copy()
    if sub.empty:
        continue

    # Use best label: dx_desc if available else raw code
    sub["dx_label"] = sub.apply(
        lambda r: f"{r['trigger_dx_desc'][:40]}\n({r['trigger_dx']})"
                  if r["trigger_dx_desc"] != r["trigger_dx"] else r["trigger_dx"],
        axis=1
    )

    # Order DX by volume descending
    dx_order = (sub.groupby("dx_label")["total_volume"]
                .first().sort_values(ascending=True).index.tolist())

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f"Top 10 Trigger DX by Volume — Hit@3 — {WLABELS[window]}",
                 fontsize=13, fontweight="bold")

    height = 0.18
    y      = np.arange(len(dx_order))
    models_in = [m for m in MODELS if m in sub["model"].unique()]

    for j, model in enumerate(models_in):
        vals = [sub[(sub["dx_label"] == dx) & (sub["model"] == model)]["hit_at_3"]
                .values[0] if len(sub[(sub["dx_label"] == dx) & (sub["model"] == model)]) > 0
                else 0
                for dx in dx_order]
        ax.barh(y + j * height, vals, height,
                label=model, color=MCOLORS[model], alpha=0.85)

    ax.set_yticks(y + height * 1.5)
    ax.set_yticklabels(dx_order, fontsize=8)
    ax.set_xlabel("Hit@3")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.0)
    plt.tight_layout()
    plt.show()

    # Summary table — best model per DX
    tbl = (sub[sub["model"].isin(models_in)]
           .sort_values(["volume_rank", "model"])
           .pivot(index="trigger_dx", columns="model", values="hit_at_3")
           .round(4))
    tbl.insert(0, "dx_desc", sub.drop_duplicates("trigger_dx")
               .set_index("trigger_dx")["trigger_dx_desc"])
    tbl.insert(1, "volume",  sub.drop_duplicates("trigger_dx")
               .set_index("trigger_dx")["total_volume"])
    display(tbl)

print(f"Section B done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — TOP 10 DX BY ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section C — Top 10 Trigger DX by Prediction Accuracy"))
display(Markdown("*Minimum 100 triggers — DX codes where predictions are most reliable*"))

for window in WINDOWS:
    sub = dx_df[(dx_df["time_bucket"] == window) &
                (dx_df["accuracy_rank"].notna()) &
                (dx_df["accuracy_rank"] <= 10)].copy()
    if sub.empty:
        continue

    sub["dx_label"] = sub.apply(
        lambda r: f"{r['trigger_dx_desc'][:40]}\n({r['trigger_dx']})"
                  if r["trigger_dx_desc"] != r["trigger_dx"] else r["trigger_dx"],
        axis=1
    )
    dx_order = (sub.groupby("dx_label")["avg_hit_at_3_all_models"]
                .first().sort_values(ascending=True).index.tolist())

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f"Top 10 Trigger DX by Accuracy — Hit@3 — {WLABELS[window]}",
                 fontsize=13, fontweight="bold")

    height = 0.18
    y      = np.arange(len(dx_order))
    models_in = [m for m in MODELS if m in sub["model"].unique()]

    for j, model in enumerate(models_in):
        vals = [sub[(sub["dx_label"] == dx) & (sub["model"] == model)]["hit_at_3"]
                .values[0] if len(sub[(sub["dx_label"] == dx) &
                                       (sub["model"] == model)]) > 0 else 0
                for dx in dx_order]
        ax.barh(y + j * height, vals, height,
                label=model, color=MCOLORS[model], alpha=0.85)

    ax.set_yticks(y + height * 1.5)
    ax.set_yticklabels(dx_order, fontsize=8)
    ax.set_xlabel("Hit@3")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.0)
    plt.tight_layout()
    plt.show()

    tbl = (sub[sub["model"].isin(models_in)]
           .sort_values(["accuracy_rank", "model"])
           .pivot(index="trigger_dx", columns="model", values="hit_at_3")
           .round(4))
    tbl.insert(0, "dx_desc", sub.drop_duplicates("trigger_dx")
               .set_index("trigger_dx")["trigger_dx_desc"])
    tbl.insert(1, "avg_hit@3", sub.drop_duplicates("trigger_dx")
               .set_index("trigger_dx")["avg_hit_at_3_all_models"].round(4))
    display(tbl)

print(f"Section C done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — TOP 10 OUTBOUND PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section D — Top 10 Outbound Providers by Prediction Accuracy"))
display(Markdown("*For members leaving these providers — we predict their next provider reliably*"))

prov_df = client.query(f"""
    SELECT
        provider_direction, srv_prvdr_id, provider_name,
        primary_specialty, specialty_desc,
        model, time_bucket,
        n_triggers, total_outbound_transitions, total_inbound_transitions,
        is_top80, total_tp, total_fp, total_fn,
        hit_at_1, hit_at_3, hit_at_5,
        ndcg_at_3, overall_precision, overall_recall,
        accuracy_rank
    FROM `{DS}.A870800_gen_rec_pma_provider_summary_5pct`
    ORDER BY provider_direction, time_bucket, model, accuracy_rank
""").to_dataframe()

print(f"Loaded {len(prov_df):,} rows")

outbound_df = prov_df[prov_df["provider_direction"] == "Outbound"].copy()

for window in WINDOWS:
    sub = outbound_df[(outbound_df["time_bucket"] == window) &
                      (outbound_df["accuracy_rank"] <= 10)].copy()
    if sub.empty:
        continue

    sub["prov_label"] = sub.apply(
        lambda r: f"{str(r['provider_name'])[:35]}\n({r['srv_prvdr_id']})"
                  if pd.notna(r["provider_name"]) else str(r["srv_prvdr_id"]),
        axis=1
    )
    prov_order = (sub.groupby("prov_label")["hit_at_3"]
                  .max().sort_values(ascending=True).index.tolist())

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f"Top 10 Outbound Providers — Hit@3 — {WLABELS[window]}",
                 fontsize=13, fontweight="bold")

    height = 0.18
    y      = np.arange(len(prov_order))
    models_in = [m for m in MODELS if m in sub["model"].unique()]

    for j, model in enumerate(models_in):
        vals = [sub[(sub["prov_label"] == p) & (sub["model"] == model)]["hit_at_3"]
                .values[0] if len(sub[(sub["prov_label"] == p) &
                                       (sub["model"] == model)]) > 0 else 0
                for p in prov_order]
        ax.barh(y + j * height, vals, height,
                label=model, color=MCOLORS[model], alpha=0.85)

    ax.set_yticks(y + height * 1.5)
    ax.set_yticklabels(prov_order, fontsize=8)
    ax.set_xlabel("Hit@3")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    tbl = (sub[sub["model"].isin(models_in)]
           .sort_values(["accuracy_rank", "model"])
           .drop_duplicates(["srv_prvdr_id", "model"])
           .pivot(index="srv_prvdr_id", columns="model", values="hit_at_3")
           .round(4))
    tbl.insert(0, "provider_name", sub.drop_duplicates("srv_prvdr_id")
               .set_index("srv_prvdr_id")["provider_name"])
    tbl.insert(1, "specialty", sub.drop_duplicates("srv_prvdr_id")
               .set_index("srv_prvdr_id")["specialty_desc"])
    tbl.insert(2, "n_triggers", sub.drop_duplicates("srv_prvdr_id")
               .set_index("srv_prvdr_id")["n_triggers"])
    display(tbl)

print(f"Section D done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — TOP 10 INBOUND PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section E — Top 10 Inbound Providers by Prediction Accuracy"))
display(Markdown("*These providers are most often correctly predicted as next destination*"))

inbound_df = prov_df[prov_df["provider_direction"] == "Inbound"].copy()

for window in WINDOWS:
    sub = inbound_df[(inbound_df["time_bucket"] == window) &
                     (inbound_df["accuracy_rank"] <= 10)].copy()
    if sub.empty:
        continue

    sub["prov_label"] = sub.apply(
        lambda r: f"{str(r['provider_name'])[:35]}\n({r['srv_prvdr_id']})"
                  if pd.notna(r["provider_name"]) else str(r["srv_prvdr_id"]),
        axis=1
    )
    prov_order = (sub.groupby("prov_label")["overall_precision"]
                  .max().sort_values(ascending=True).index.tolist())

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f"Top 10 Inbound Providers — Precision — {WLABELS[window]}",
                 fontsize=13, fontweight="bold")

    height = 0.18
    y      = np.arange(len(prov_order))
    models_in = [m for m in MODELS if m in sub["model"].unique()]

    for j, model in enumerate(models_in):
        vals = [sub[(sub["prov_label"] == p) & (sub["model"] == model)]["overall_precision"]
                .values[0] if len(sub[(sub["prov_label"] == p) &
                                       (sub["model"] == model)]) > 0 else 0
                for p in prov_order]
        ax.barh(y + j * height, vals, height,
                label=model, color=MCOLORS[model], alpha=0.85)

    ax.set_yticks(y + height * 1.5)
    ax.set_yticklabels(prov_order, fontsize=8)
    ax.set_xlabel("Precision (TP / Times Predicted)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    tbl = (sub[sub["model"].isin(models_in)]
           .sort_values(["accuracy_rank", "model"])
           .drop_duplicates(["srv_prvdr_id", "model"])
           .pivot(index="srv_prvdr_id", columns="model", values="overall_precision")
           .round(4))
    tbl.insert(0, "provider_name", sub.drop_duplicates("srv_prvdr_id")
               .set_index("srv_prvdr_id")["provider_name"])
    tbl.insert(1, "specialty", sub.drop_duplicates("srv_prvdr_id")
               .set_index("srv_prvdr_id")["specialty_desc"])
    tbl.insert(2, "n_predicted", sub.drop_duplicates("srv_prvdr_id")
               .set_index("srv_prvdr_id")["n_triggers"])
    display(tbl)

print(f"Section E done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — PROVIDER + DX TRANSITION TABLE (Mandatory 5a)
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section F — Provider + DX Prediction Summary (Top pairs by volume)"))
display(Markdown("*For each from_provider + trigger_dx → predicted to_provider: accuracy and evidence*"))

# Pull top 200 rows by n_times_predicted for T0_30 across all models
pdx_df = client.query(f"""
    SELECT
        from_provider, from_provider_name, from_specialty_desc,
        trigger_dx, trigger_dx_desc,
        to_provider, to_provider_name, to_specialty_desc,
        model, time_bucket,
        n_times_predicted, n_tp, n_fp,
        precision_this_pair,
        avg_pred_score, training_transition_count
    FROM `{DS}.A870800_gen_rec_pma_provider_dx_summary_5pct`
    WHERE time_bucket = 'T0_30'
    ORDER BY n_times_predicted DESC
    LIMIT 500
""").to_dataframe()

print(f"Loaded {len(pdx_df):,} rows (top 500 by volume, T0_30)")

# Display per model
for model in MODELS:
    sub = pdx_df[pdx_df["model"] == model].head(20)
    if sub.empty:
        continue
    display(Markdown(f"### {model} — Top 20 (from_provider, DX, to_provider) pairs"))
    display(sub[[
        "from_provider_name", "from_specialty_desc",
        "trigger_dx", "trigger_dx_desc",
        "to_provider_name", "to_specialty_desc",
        "n_times_predicted", "n_tp", "n_fp",
        "precision_this_pair", "avg_pred_score",
        "training_transition_count"
    ]].rename(columns={
        "from_provider_name":   "From Provider",
        "from_specialty_desc":  "From Specialty",
        "trigger_dx":           "DX",
        "trigger_dx_desc":      "DX Description",
        "to_provider_name":     "To Provider",
        "to_specialty_desc":    "To Specialty",
        "n_times_predicted":    "N Predicted",
        "n_tp":                 "TP",
        "n_fp":                 "FP",
        "precision_this_pair":  "Precision",
        "avg_pred_score":       "Avg Score",
        "training_transition_count": "Training Evidence",
    }).reset_index(drop=True))

print(f"Section F done — {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION G — MEMBER PREDICTABILITY
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section G — Member Predictability"))
display(Markdown("*What % of members had at least one correct T30 prediction?*"))

member_df = client.query(f"""
    SELECT
        model
        ,COUNT(DISTINCT member_id)                       AS total_members
        ,COUNTIF(hit_at_1 = 1.0)                         AS n_hit_at_1
        ,COUNTIF(hit_at_3 = 1.0)                         AS n_hit_at_3
        ,COUNTIF(hit_at_5 = 1.0)                         AS n_hit_at_5
        ,COUNT(*)                                        AS n_triggers
    FROM `{DS}.A870800_gen_rec_provider_eval_5pct`
    WHERE time_bucket = 'T0_30'
    GROUP BY model
    ORDER BY model
""").to_dataframe()

member_df["pct_hit@1"] = (member_df["n_hit_at_1"] / member_df["n_triggers"] * 100).round(1)
member_df["pct_hit@3"] = (member_df["n_hit_at_3"] / member_df["n_triggers"] * 100).round(1)
member_df["pct_hit@5"] = (member_df["n_hit_at_5"] / member_df["n_triggers"] * 100).round(1)

display(Markdown("### % of T30 triggers with at least one correct prediction"))
display(member_df[["model","n_triggers","pct_hit@1","pct_hit@3","pct_hit@5"]])

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(member_df))
width = 0.25
for j, (col, label) in enumerate(zip(["pct_hit@1","pct_hit@3","pct_hit@5"],
                                       ["Hit@1","Hit@3","Hit@5"])):
    ax.bar(x + j*width, member_df[col], width, label=label, alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(member_df["model"])
ax.set_ylabel("% of triggers with a correct prediction")
ax.set_title("Member Predictability — T0_30 Window")
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()
plt.show()

# Reliability statement
best_model = member_df.loc[member_df["pct_hit@3"].idxmax(), "model"]
best_pct   = member_df["pct_hit@3"].max()
display(Markdown(f"""
### Reliability Statement
**Best model:** {best_model} — **{best_pct:.1f}%** of T30 triggers have at least one
correct provider prediction in the top 3.

This means for **{best_pct:.1f}% of clinical trigger events**, the model reliably
identifies the next provider — enabling actionable care management interventions.
"""))

print(f"Section G done — {time.time()-t0:.1f}s")
print("NB_PMA_01 complete")
