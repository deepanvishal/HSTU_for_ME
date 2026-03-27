# ============================================================
# Provider Refinement Visuals
# 4 charts for R1-R4 slides
# Metric: Hit@5 throughout
# Requires: client, DS
# Sources: provider_eval_5pct, pma_transition_bucket_5pct,
#          pma_provider_summary_5pct, pma_dx_summary_5pct
# ============================================================
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from google.cloud import bigquery
from IPython.display import display, Markdown

DS = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
client = bigquery.Client(project="anbc-hcb-dev")

PROV_BEST  = "SASRec"
MODELS     = ["SASRec", "HSTU", "BERT4Rec", "Markov"]
MCOLORS    = {"SASRec": "#3B82F6", "HSTU": "#8172B2", "BERT4Rec": "#059669", "Markov": "#9CA3AF"}
WINDOWS    = ["T0_30", "T30_60", "T60_180"]
WLABELS    = {"T0_30": "0–30 Days", "T30_60": "30–60 Days", "T60_180": "60–180 Days"}
GREEN      = "#059669"
RED        = "#DC2626"
GREY       = "#9CA3AF"
YELLOW     = "#F59E0B"
BLUE       = "#3B82F6"
PURPLE     = "#8172B2"
TEXT_DARK  = "#1F2937"
TEXT_MED   = "#6B7280"
GRID_CLR   = "#E5E7EB"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelcolor": TEXT_DARK,
    "axes.edgecolor": GRID_CLR,
    "xtick.color": TEXT_MED,
    "ytick.color": TEXT_MED,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

OUT = "./presentation_visuals/"
os.makedirs(OUT, exist_ok=True)
t_total = time.time()


# ══════════════════════════════════════════════════════════════
# DATA LOADS
# ══════════════════════════════════════════════════════════════
display(Markdown("## Loading provider data..."))

prov_overall = client.query(f"""
    SELECT model, time_bucket, COUNT(*) AS n_triggers, ROUND(AVG(hit_at_5), 4) AS hit_at_5
    FROM `{DS}.A870800_gen_rec_provider_eval_5pct`
    WHERE (tp + fn) > 0
    GROUP BY model, time_bucket
    ORDER BY model, time_bucket
""").to_dataframe()

prov_seg = client.query(f"""
    SELECT member_segment, COUNT(*) AS n_triggers, ROUND(AVG(hit_at_5), 4) AS hit_at_5
    FROM `{DS}.A870800_gen_rec_provider_eval_5pct`
    WHERE time_bucket = 'T0_30' AND model = '{PROV_BEST}' AND (tp + fn) > 0
    GROUP BY member_segment ORDER BY hit_at_5 DESC
""").to_dataframe()
prov_seg["member_segment"] = prov_seg["member_segment"].replace({
    "Adult_Female": "Adult Female", "Adult_Male": "Adult Male"})

prov_buckets = client.query(f"""
    SELECT transition_bucket, model, hit_at_5, n_triggers
    FROM `{DS}.A870800_gen_rec_pma_transition_bucket_5pct`
    WHERE time_bucket = 'T0_30'
    ORDER BY model, transition_bucket
""").to_dataframe()

prov_outbound = client.query(f"""
    SELECT provider_name, specialty_desc, n_triggers, hit_at_5
    FROM `{DS}.A870800_gen_rec_pma_provider_summary_5pct`
    WHERE provider_direction = 'Outbound' AND time_bucket = 'T0_30'
      AND model = '{PROV_BEST}'
      AND specialty_desc NOT LIKE '%Lab%' AND specialty_desc NOT LIKE '%lab%'
      AND n_triggers >= 20
    ORDER BY hit_at_5 DESC LIMIT 10
""").to_dataframe()

prov_inbound = client.query(f"""
    SELECT provider_name, specialty_desc, n_triggers, overall_precision
    FROM `{DS}.A870800_gen_rec_pma_provider_summary_5pct`
    WHERE provider_direction = 'Inbound' AND time_bucket = 'T0_30'
      AND model = '{PROV_BEST}'
      AND specialty_desc NOT LIKE '%Lab%' AND specialty_desc NOT LIKE '%lab%'
      AND n_triggers >= 20
    ORDER BY overall_precision DESC LIMIT 10
""").to_dataframe()

prov_dx = client.query(f"""
    SELECT trigger_dx, dx_desc, n_triggers, hit_at_5
    FROM `{DS}.A870800_gen_rec_pma_dx_summary_5pct`
    WHERE time_bucket = 'T0_30' AND model = '{PROV_BEST}' AND n_triggers >= 100
    ORDER BY hit_at_5 DESC LIMIT 10
""").to_dataframe()

print(f"Provider data loaded — {time.time()-t_total:.1f}s")


# ══════════════════════════════════════════════════════════════
# PROV-VIS-02: MODEL COMPARISON — 4 MODELS × 3 WINDOWS (R2)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## PROV-VIS-02: Provider Model Comparison"))

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(WINDOWS))
w = 0.18

for i, model in enumerate(MODELS):
    sub = prov_overall[prov_overall["model"] == model].set_index("time_bucket").reindex(WINDOWS)
    vals = sub["hit_at_5"].fillna(0).values * 100
    bars = ax.bar(x + i*w - 1.5*w, vals, w, label=model,
                  color=MCOLORS[model], edgecolor="white", alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([WLABELS[w_] for w_ in WINDOWS])
ax.set_ylabel("Hit@5 (%)")
ax.set_ylim(0, 80)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
ax.grid(axis="y", color=GRID_CLR, linewidth=0.5)
ax.set_title("Provider-Level Prediction Accuracy\n4 Models × 3 Windows at Hit@5 (5% sample)")
plt.tight_layout()
plt.savefig(f"{OUT}prov_vis_02_model_comparison.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# PROV-VIS-03a: OUTBOUND TOP 10 PROVIDERS (R3 left)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## PROV-VIS-03a: Top 10 Outbound Providers (excl Lab)"))

n = len(prov_outbound)
labels_out = [f"{row['provider_name'][:25]}\n{row['specialty_desc'][:20]}" for _, row in prov_outbound.iterrows()]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(n), prov_outbound["hit_at_5"].values * 100,
        color=BLUE, edgecolor="white", height=0.6)
ax.set_yticks(range(n))
ax.set_yticklabels(labels_out, fontsize=8)
ax.invert_yaxis()
for i, (v, trg) in enumerate(zip(prov_outbound["hit_at_5"].values, prov_outbound["n_triggers"].values)):
    ax.text(v*100 + 0.3, i, f"n={int(trg):,}", va="center", fontsize=7, color=TEXT_MED)
ax.set_xlabel("Hit@5 (%)")
ax.set_xlim(0, 110)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title(f"Outbound: When a member leaves this provider,\nhow often can we predict who they see next? ({PROV_BEST}, T30)")
plt.tight_layout()
plt.savefig(f"{OUT}prov_vis_03a_outbound_top10.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# PROV-VIS-03b: INBOUND TOP 10 PROVIDERS (R3 right)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## PROV-VIS-03b: Top 10 Inbound Providers (excl Lab)"))

n = len(prov_inbound)
labels_in = [f"{row['provider_name'][:25]}\n{row['specialty_desc'][:20]}" for _, row in prov_inbound.iterrows()]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(n), prov_inbound["overall_precision"].values * 100,
        color=GREEN, edgecolor="white", height=0.6)
ax.set_yticks(range(n))
ax.set_yticklabels(labels_in, fontsize=8)
ax.invert_yaxis()
for i, (v, trg) in enumerate(zip(prov_inbound["overall_precision"].values, prov_inbound["n_triggers"].values)):
    ax.text(v*100 + 0.3, i, f"n={int(trg):,}", va="center", fontsize=7, color=TEXT_MED)
ax.set_xlabel("Precision (%)")
ax.set_xlim(0, 110)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title(f"Inbound: When we predict this provider as destination,\nhow often are we correct? ({PROV_BEST}, T30)")
plt.tight_layout()
plt.savefig(f"{OUT}prov_vis_03b_inbound_top10.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# PROV-VIS-04: EVIDENCE BUCKETS × 4 MODELS (R4)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## PROV-VIS-04: Evidence Bucket Performance"))

bucket_order = ["Low", "Medium", "High"]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(bucket_order))
w = 0.18

for i, model in enumerate(MODELS):
    sub = prov_buckets[prov_buckets["model"] == model].copy()
    sub["bucket_order"] = sub["transition_bucket"].map({b: j for j, b in enumerate(bucket_order)})
    sub = sub.sort_values("bucket_order")
    vals = sub["hit_at_5"].values * 100
    bars = ax.bar(x + i*w - 1.5*w, vals, w, label=model,
                  color=MCOLORS[model], edgecolor="white", alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(bucket_order)
ax.set_xlabel("Transition Evidence Level")
ax.set_ylabel("Hit@5 (%)")
ax.set_ylim(0, 55)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
ax.grid(axis="y", color=GRID_CLR, linewidth=0.5)
ax.set_title("Provider Prediction by Training Evidence Level\nSequence models maintain accuracy where Markov collapses (T30)")
plt.tight_layout()
plt.savefig(f"{OUT}prov_vis_04_evidence_buckets.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# PROV-VIS-EXTRA: SEGMENT HEATMAP
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## PROV-VIS-EXTRA: Segment Performance"))

import seaborn as sns
prov_seg_all = client.query(f"""
    SELECT model, member_segment, ROUND(AVG(hit_at_5), 4) AS hit_at_5
    FROM `{DS}.A870800_gen_rec_provider_eval_5pct`
    WHERE time_bucket = 'T0_30' AND (tp + fn) > 0
    GROUP BY model, member_segment
""").to_dataframe()
prov_seg_all["member_segment"] = prov_seg_all["member_segment"].replace({
    "Adult_Female": "Adult Female", "Adult_Male": "Adult Male"})
prov_seg_all = prov_seg_all.rename(columns={"member_segment": "Member Segment"})

seg_pivot = prov_seg_all[prov_seg_all["Member Segment"] != "Unknown"].pivot(
    index="model", columns="Member Segment", values="hit_at_5"
).reindex(MODELS) * 100

fig, ax = plt.subplots(figsize=(8, 3.5))
sns.heatmap(seg_pivot, annot=True, fmt=".1f", cmap="YlGn",
            linewidths=0.5, ax=ax, vmin=0, vmax=60,
            cbar_kws={"label": "Hit@5 %"})
ax.set_title("Provider Hit@5 by Model × Member Segment at T30")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}prov_vis_extra_segment_heatmap.png")
plt.show()


# ══════════════════════════════════════════════════════════════
display(Markdown(f"""
---
## Provider Visuals Generated

| File | Slide | Direction |
|---|---|---|
| prov_vis_02_model_comparison.png | R2 | Overall |
| prov_vis_03a_outbound_top10.png | R3 left | Outbound |
| prov_vis_03b_inbound_top10.png | R3 right | Inbound |
| prov_vis_04_evidence_buckets.png | R4 | Overall |
| prov_vis_extra_segment_heatmap.png | Appendix | Overall |

Total time: {time.time()-t_total:.1f}s
"""))
print("Provider visuals done.")
