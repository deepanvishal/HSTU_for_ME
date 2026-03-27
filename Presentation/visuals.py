# ============================================================
# Presentation Visuals — All Charts (v2)
# All feedback applied:
#   - VIS-01: legend bottom center
#   - VIS-02: four layout options for specialty ranking
#   - VIS-04c: provider count top/bottom 10 with count labels
#   - VIS-07: transition volume as labels
#   - VIS-09: score-based confidence (max prediction score)
#   - VIS-11: no quadrant labels, x-cap at max+5, top N labels only
#   - All % axes show % symbol
#   - No redundant info
# Metric: Hit@5 throughout
# ============================================================
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from google.cloud import bigquery
from IPython.display import display, Markdown

DS = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
client = bigquery.Client(project="anbc-hcb-dev")

BEST_MODEL = "BERT4Rec"
MODELS     = ["BERT4Rec", "SASRec", "Markov"]
MCOLORS    = {"BERT4Rec": "#059669", "SASRec": "#3B82F6", "Markov": "#9CA3AF"}
WINDOWS    = ["T0_30", "T30_60", "T60_180"]
WLABELS    = {"T0_30": "0–30 Days", "T30_60": "30–60 Days", "T60_180": "60–180 Days"}
GREEN      = "#059669"
RED        = "#DC2626"
GREY       = "#9CA3AF"
YELLOW     = "#F59E0B"
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
display(Markdown("## Loading data..."))

perf_full = client.query(f"""
    SELECT model, time_bucket, member_segment, n_triggers, hit_at_5, ndcg_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_full`
    WHERE member_segment = 'ALL'
    ORDER BY model, time_bucket
""").to_dataframe()

perf_seg = client.query(f"""
    SELECT model, time_bucket, member_segment, n_triggers, hit_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_full`
    WHERE member_segment != 'ALL' AND time_bucket = 'T0_30'
    ORDER BY model, member_segment
""").to_dataframe()

perf_spec = client.query(f"""
    SELECT ending_specialty, total_appearances, predicted_at_5
        ,ROUND(SAFE_DIVIDE(predicted_at_5, total_appearances), 4) AS hit_rate_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_by_ending_specialty`
    WHERE time_bucket = 'T0_30' AND model = '{BEST_MODEL}' AND total_appearances >= 20
    ORDER BY hit_rate_at_5 DESC
""").to_dataframe()

spec_names = client.query(f"""
    SELECT DISTINCT next_specialty AS code, next_specialty_desc AS name
    FROM `{DS}.A870800_gen_rec_markov_train` WHERE next_specialty IS NOT NULL
""").to_dataframe()

perf_spec = perf_spec.merge(spec_names, left_on="ending_specialty", right_on="code", how="left")
perf_spec["display"] = perf_spec["name"].fillna(perf_spec["ending_specialty"])

perf_dx = client.query(f"""
    WITH all_scores AS (
        SELECT trigger_dx, hit_at_5
        FROM `{DS}.A870800_gen_rec_trigger_scores`
        WHERE time_bucket = 'T0_30' AND model = '{BEST_MODEL}'
          AND true_labels IS NOT NULL AND true_labels != ''
          AND top5_predictions IS NOT NULL AND top5_predictions != ''
    )
    SELECT trigger_dx, COUNT(*) AS trigger_volume, ROUND(AVG(hit_at_5), 4) AS hit_at_5
    FROM all_scores GROUP BY 1 HAVING COUNT(*) >= 20 ORDER BY hit_at_5 DESC
""").to_dataframe()

ccsr_map = client.query(f"""
    SELECT DISTINCT trigger_dx, trigger_ccsr_desc
    FROM `{DS}.A870800_gen_rec_markov_train` WHERE trigger_ccsr_desc IS NOT NULL
""").to_dataframe()
perf_dx = perf_dx.merge(ccsr_map, on="trigger_dx", how="left")
perf_dx["display"] = perf_dx["trigger_ccsr_desc"].fillna(perf_dx["trigger_dx"])

seq_buckets = client.query(f"""
    WITH seq_lengths AS (
        SELECT member_id, trigger_date, trigger_dx, MAX(recency_rank) AS seq_length
        FROM `{DS}.A870800_gen_rec_test_sequences_5pct` GROUP BY 1, 2, 3
    ),
    joined AS (
        SELECT s.hit_at_5,
            CASE WHEN sl.seq_length < 5 THEN '<5'
                 WHEN sl.seq_length < 10 THEN '5-9'
                 WHEN sl.seq_length < 15 THEN '10-14'
                 ELSE '15-20' END AS seq_bucket
        FROM `{DS}.A870800_gen_rec_trigger_scores` s
        JOIN seq_lengths sl
            ON CAST(s.member_id AS STRING) = CAST(sl.member_id AS STRING)
            AND CAST(s.trigger_date AS DATE) = sl.trigger_date
            AND s.trigger_dx = sl.trigger_dx
        WHERE s.time_bucket = 'T0_30' AND s.model = '{BEST_MODEL}'
          AND s.true_labels IS NOT NULL AND s.true_labels != ''
          AND s.top5_predictions IS NOT NULL AND s.top5_predictions != ''
    )
    SELECT seq_bucket, COUNT(*) AS n, ROUND(AVG(hit_at_5), 4) AS hit_at_5
    FROM joined GROUP BY 1
""").to_dataframe()
seq_buckets["order"] = seq_buckets["seq_bucket"].map({"<5":0,"5-9":1,"10-14":2,"15-20":3})
seq_buckets = seq_buckets.sort_values("order")

# Score-based confidence (Option A)
conf_scores = client.query(f"""
    WITH parsed AS (
        SELECT
            CAST(SPLIT(top5_scores, '|')[OFFSET(0)] AS FLOAT64) AS top1_score
        FROM `{DS}.A870800_gen_rec_trigger_scores`
        WHERE time_bucket = 'T0_30' AND model = '{BEST_MODEL}'
          AND true_labels IS NOT NULL AND true_labels != ''
          AND top5_predictions IS NOT NULL AND top5_predictions != ''
          AND top5_scores IS NOT NULL AND top5_scores != ''
    )
    SELECT
        CASE
            WHEN top1_score >= 0.5 THEN 'High (score >= 0.5)'
            WHEN top1_score >= 0.2 THEN 'Medium (0.2 - 0.5)'
            ELSE 'Low (score < 0.2)'
        END AS confidence_bucket
        ,COUNT(*) AS n_triggers
    FROM parsed
    GROUP BY 1
""").to_dataframe()

prov_counts = client.query(f"""
    SELECT specialty_ctg_cd AS code, COUNT(DISTINCT srv_prvdr_id) AS provider_count
    FROM `{DS}.A870800_claims_gen_rec_2022_2025_sfl`
    WHERE specialty_ctg_cd IS NOT NULL AND TRIM(specialty_ctg_cd) != ''
    GROUP BY 1 ORDER BY 2 DESC
""").to_dataframe()
prov_counts = prov_counts.merge(spec_names, on="code", how="left")
prov_counts["display"] = prov_counts["name"].fillna(prov_counts["code"])

spend_spec = client.query(f"""
    SELECT grouping_code AS code, grouping_desc AS name
        ,SUM(total_allowed_amt) AS total_spend, SUM(visit_count) AS visits
    FROM `{DS}.A870800_gen_rec_f_spend_summary`
    WHERE summary_type = 'specialty' AND visit_type IN ('downstream', 'v2')
      AND grouping_code IS NOT NULL AND grouping_code != ''
    GROUP BY 1, 2 ORDER BY 3 DESC
""").to_dataframe()

print(f"All data loaded — {time.time()-t_total:.1f}s")


# ══════════════════════════════════════════════════════════════
# VIS-01: MODEL COMPARISON — GROUPED BAR (W2-1, W3-3)
# Legend: bottom center, outside chart
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-01: Model Comparison"))

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(WINDOWS))
w = 0.25

for i, model in enumerate(MODELS):
    sub = perf_full[perf_full["model"] == model].set_index("time_bucket").reindex(WINDOWS)
    vals = sub["hit_at_5"].fillna(0).values * 100
    bars = ax.bar(x + i*w - w, vals, w, label=model,
                  color=MCOLORS[model], edgecolor="white", alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([WLABELS[w_] for w_ in WINDOWS])
ax.set_ylabel("Hit@5 (%)")
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
ax.grid(axis="y", color=GRID_CLR, linewidth=0.5)
ax.set_title("Prediction Accuracy Across Time Windows")
plt.tight_layout()
plt.savefig(f"{OUT}vis_01_model_comparison.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-02: SPECIALTY RANKING — INBOUND (W2-2, W3-4)
# Four options for layout
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-02: Inbound Specialty Ranking — 4 Options"))

avg_h5 = perf_spec["hit_rate_at_5"].mean()

# OPTION A: Top 15 + Bottom 15
display(Markdown("### Option A: Top 15 + Bottom 15"))
top15 = perf_spec.head(15)
bot15 = perf_spec.tail(15).iloc[::-1]
combo_a = pd.concat([top15, bot15])
n_a = len(combo_a)
colors_a = [GREEN]*15 + [RED]*15

fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(range(n_a), combo_a["hit_rate_at_5"].values * 100,
        color=colors_a, edgecolor="white", height=0.7)
ax.set_yticks(range(n_a))
ax.set_yticklabels(combo_a["display"].values, fontsize=8)
ax.invert_yaxis()
for i, (v, vol) in enumerate(zip(combo_a["hit_rate_at_5"].values, combo_a["total_appearances"].values)):
    ax.text(v*100 + 0.5, i, f"n={int(vol):,}", va="center", fontsize=7, color=TEXT_MED)
ax.axhline(14.5, color=TEXT_MED, linestyle="--", linewidth=0.8)
ax.set_xlabel("Hit@5 (%)")
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Inbound: When members visit this specialty, how often did we predict it?\nTop 15 + Bottom 15")
plt.tight_layout()
plt.savefig(f"{OUT}vis_02a_specialty_top15_bot15.png")
plt.show()

# OPTION B: Two separate charts
display(Markdown("### Option B: Top 20 (separate)"))
top20 = perf_spec.head(20)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(top20)), top20["hit_rate_at_5"].values * 100,
        color=GREEN, edgecolor="white", height=0.7)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["display"].values, fontsize=9)
ax.invert_yaxis()
for i, (v, vol) in enumerate(zip(top20["hit_rate_at_5"].values, top20["total_appearances"].values)):
    ax.text(v*100 + 0.5, i, f"n={int(vol):,}", va="center", fontsize=7, color=TEXT_MED)
ax.set_xlabel("Hit@5 (%)")
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Inbound: Top 20 Most Predictable Specialties")
plt.tight_layout()
plt.savefig(f"{OUT}vis_02b_specialty_top20.png")
plt.show()

display(Markdown("### Option B: Bottom 20 (separate)"))
bot20 = perf_spec.tail(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(bot20)), bot20["hit_rate_at_5"].values * 100,
        color=RED, edgecolor="white", height=0.7)
ax.set_yticks(range(len(bot20)))
ax.set_yticklabels(bot20["display"].values, fontsize=9)
ax.invert_yaxis()
for i, (v, vol) in enumerate(zip(bot20["hit_rate_at_5"].values, bot20["total_appearances"].values)):
    ax.text(max(v*100 + 0.5, 1), i, f"n={int(vol):,}", va="center", fontsize=7, color=TEXT_MED)
ax.set_xlabel("Hit@5 (%)")
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Inbound: Bottom 20 Least Predictable Specialties")
plt.tight_layout()
plt.savefig(f"{OUT}vis_02b_specialty_bot20.png")
plt.show()

# OPTION C: Top 10 + Bottom 10 compact
display(Markdown("### Option C: Top 10 + Bottom 10"))
top10 = perf_spec.head(10)
bot10 = perf_spec.tail(10).iloc[::-1]
combo_c = pd.concat([top10, bot10])
n_c = len(combo_c)
colors_c = [GREEN]*10 + [RED]*10

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.barh(range(n_c), combo_c["hit_rate_at_5"].values * 100,
        color=colors_c, edgecolor="white", height=0.7)
ax.set_yticks(range(n_c))
ax.set_yticklabels(combo_c["display"].values, fontsize=9)
ax.invert_yaxis()
for i, (v, vol) in enumerate(zip(combo_c["hit_rate_at_5"].values, combo_c["total_appearances"].values)):
    ax.text(max(v*100 + 0.5, 1), i, f"n={int(vol):,}", va="center", fontsize=7, color=TEXT_MED)
ax.axhline(9.5, color=TEXT_MED, linestyle="--", linewidth=0.8)
ax.set_xlabel("Hit@5 (%)")
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Inbound: Top 10 + Bottom 10 Specialties")
plt.tight_layout()
plt.savefig(f"{OUT}vis_02c_specialty_top10_bot10.png")
plt.show()

# OPTION D: Tier grouping
display(Markdown("### Option D: Tier Grouping"))
perf_spec["tier"] = pd.cut(perf_spec["hit_rate_at_5"],
                            bins=[-0.01, 0.30, 0.60, 1.01],
                            labels=["Below 30%", "30–60%", "Above 60%"])
tier_counts = perf_spec.groupby("tier", observed=True).agg(
    count=("ending_specialty", "count"),
    avg_hit=("hit_rate_at_5", "mean"),
    total_vol=("total_appearances", "sum")
).reset_index()
tier_colors = [RED, YELLOW, GREEN]

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh(tier_counts["tier"], tier_counts["count"], color=tier_colors, edgecolor="white", height=0.5)
for i, row in tier_counts.iterrows():
    ax.text(row["count"] + 0.5, i,
            f"{int(row['count'])} specialties  |  avg Hit@5: {row['avg_hit']*100:.1f}%",
            va="center", fontsize=9)
ax.set_xlabel("Number of Specialties")
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Inbound: Specialties Grouped by Prediction Accuracy Tier")
plt.tight_layout()
plt.savefig(f"{OUT}vis_02d_specialty_tiers.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-04a: SEQUENCE DEPTH (W2-4 left)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-04a: Sequence Depth"))

fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.bar(seq_buckets["seq_bucket"], seq_buckets["hit_at_5"] * 100,
       color=GREEN, edgecolor="white", width=0.6)
for _, row in seq_buckets.iterrows():
    ax.text(row["seq_bucket"], row["hit_at_5"]*100 + 1,
            f"{row['hit_at_5']*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Hit@5 (%)")
ax.set_xlabel("Prior Visit Count")
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="y", color=GRID_CLR, linewidth=0.5)
ax.set_title("Sequence Depth", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}vis_04a_sequence_depth.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-04b: TIME WINDOW (W2-4 center)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-04b: Time Window"))

best_w = perf_full[perf_full["model"] == BEST_MODEL].set_index("time_bucket").reindex(WINDOWS)
wlabels_short = ["T30", "T60", "T180"]
wcolors = [GREEN, "#34D399", "#6EE7B7"]

fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.bar(wlabels_short, best_w["hit_at_5"].values * 100, color=wcolors, edgecolor="white", width=0.6)
for i, v in enumerate(best_w["hit_at_5"].values * 100):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Hit@5 (%)")
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="y", color=GRID_CLR, linewidth=0.5)
ax.set_title("Time Window", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}vis_04b_time_window.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-04c: PROVIDER COUNT — TOP 10 + BOTTOM 10 (W2-4 right)
# Count as labels. Shows why provider-level prediction is harder.
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-04c: Provider Count per Specialty"))

top10_p = prov_counts.head(10)
bot10_p = prov_counts.tail(10).iloc[::-1]
combo_p = pd.concat([top10_p, bot10_p])
p_colors = [RED]*10 + [GREEN]*10

fig, ax = plt.subplots(figsize=(5, 5))
ax.barh(range(len(combo_p)), combo_p["provider_count"].values,
        color=p_colors, edgecolor="white", height=0.6)
ax.set_yticks(range(len(combo_p)))
ax.set_yticklabels(combo_p["display"].values, fontsize=7)
ax.invert_yaxis()
for i, v in enumerate(combo_p["provider_count"].values):
    ax.text(v + 200, i, f"{int(v):,}", va="center", fontsize=7, color=TEXT_MED)
ax.axhline(9.5, color=TEXT_MED, linestyle="--", linewidth=0.8)
ax.set_xlabel("Unique Providers")
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Provider Count per Specialty\nTop 10 vs Bottom 10", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}vis_04c_provider_density.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-05: DOLLAR FUNNEL (W3-1)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-05: Dollar Funnel"))

funnel = [
    ("All Claims", 46.3, "311M"),
    ("Trigger-Day Claims", 17.4, "103M"),
    ("Trigger Dx Claims", 0.5, "3.6M"),
    ("Next Visit (V2)", 11.8, "78M"),
    ("T180 Downstream", 27.0, "180M"),
]

fig, ax = plt.subplots(figsize=(10, 4.5))
widths = [d[1] for d in funnel]
bar_colors = [GREY, GREY, GREY, GREEN, GREEN]

bars = ax.barh(range(len(funnel)), widths, color=bar_colors, edgecolor="white", height=0.55)
ax.set_yticks(range(len(funnel)))
ax.set_yticklabels([d[0] for d in funnel], fontsize=10)
ax.invert_yaxis()
for bar, (_, spend, claims) in zip(bars, funnel):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"${spend}B  |  {claims} claims", va="center", fontsize=9)
ax.set_xlabel("Spend ($B)")
ax.set_xlim(0, 60)
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Dollar Funnel — Total Claims to Prediction Scope")
plt.tight_layout()
plt.savefig(f"{OUT}vis_05_dollar_funnel.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-07: CONDITION PATHWAY — OUTBOUND (W3-5)
# Transition volume as labels
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-07: Outbound Condition Pathways"))

for subset, label, clr, fname in [
    (perf_dx.head(15), "Top 15 — Most Predictable", GREEN, "vis_07a_conditions_top15"),
    (perf_dx.tail(15).iloc[::-1], "Bottom 15 — Least Predictable", RED, "vis_07b_conditions_bot15"),
]:
    n = len(subset)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.35)))
    ax.barh(range(n), subset["hit_at_5"].values * 100, color=clr, edgecolor="white", height=0.6)
    ax.set_yticks(range(n))
    names = [d[:50] + "..." if len(d) > 50 else d for d in subset["display"].values]
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    for i, (v, vol) in enumerate(zip(subset["hit_at_5"].values, subset["trigger_volume"].values)):
        ax.text(max(v*100 + 0.5, 1), i, f"n={int(vol):,}", va="center", fontsize=7, color=TEXT_MED)
    ax.set_xlabel("Hit@5 (%)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
    ax.set_title(f"Outbound: Given this diagnosis, how often can we predict the next specialty?\n{label}")
    plt.tight_layout()
    plt.savefig(f"{OUT}{fname}.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-08: PREDICTION DRIVERS (W3-6)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-08: Prediction Drivers"))

seq_range = (seq_buckets["hit_at_5"].max() - seq_buckets["hit_at_5"].min()) * 100
seg_data = perf_seg[(perf_seg["model"] == BEST_MODEL) & (perf_seg["member_segment"] != "Unknown")]
seg_range = (seg_data["hit_at_5"].max() - seg_data["hit_at_5"].min()) * 100

drivers = pd.DataFrame([
    {"Driver": "Visit History Depth", "Impact": seq_range,
     "Detail": f"{seq_buckets['hit_at_5'].min()*100:.1f}% (<5 visits) → {seq_buckets['hit_at_5'].max()*100:.1f}% (15-20 visits)"},
    {"Driver": "Member Cohort", "Impact": seg_range,
     "Detail": f"{seg_data['hit_at_5'].min()*100:.1f}% (Children) → {seg_data['hit_at_5'].max()*100:.1f}% (Senior)"},
]).sort_values("Impact", ascending=True)

fig, ax = plt.subplots(figsize=(8, 3))
bars = ax.barh(drivers["Driver"], drivers["Impact"], color=GREEN, edgecolor="white", height=0.4)
for i, (bar, detail) in enumerate(zip(bars, drivers["Detail"])):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{bar.get_width():.1f}% range  ({detail})",
            va="center", fontsize=8, color=TEXT_MED)
ax.set_xlabel("Hit@5 Range Across Levels (%)")
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Sources of Prediction Variance\nWider range = stronger driver of accuracy differences")
plt.tight_layout()
plt.savefig(f"{OUT}vis_08_drivers.png")
plt.show()

display(Markdown("""
**Rationale:** We compute Hit@5 at each level of a potential driver (e.g., <5 visits, 5-9, 10-14, 15-20).
The range (max - min) across levels measures how much that factor shifts accuracy.
A wider range means the factor has more influence on whether the prediction succeeds.
This is a model-agnostic importance measure — no feature weights needed.
"""))


# ══════════════════════════════════════════════════════════════
# VIS-09: CONFIDENCE DONUT — SCORE-BASED (W3-7)
# High: top-1 prediction score >= 0.5
# Medium: 0.2 - 0.5
# Low: < 0.2
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-09: Prediction Confidence (Score-Based)"))

total_conf = float(conf_scores["n_triggers"].sum())
bucket_order_conf = ["High (score >= 0.5)", "Medium (0.2 - 0.5)", "Low (score < 0.2)"]
conf_ordered = conf_scores.set_index("confidence_bucket").reindex(bucket_order_conf).fillna(0)
sizes = conf_ordered["n_triggers"].values
pcts = sizes / total_conf * 100
conf_colors = [GREEN, YELLOW, RED]

fig, ax = plt.subplots(figsize=(5.5, 5.5))
wedges, texts, autotexts = ax.pie(
    pcts, colors=conf_colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2)
)
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight("bold")

high_pct = pcts[0]
ax.text(0, 0, f"{high_pct:.1f}%\nhigh\nconfidence",
        ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT_DARK)
ax.legend(
    [f"High (score ≥ 0.5): {int(sizes[0]):,}",
     f"Medium (0.2–0.5): {int(sizes[1]):,}",
     f"Low (< 0.2): {int(sizes[2]):,}"],
    loc="lower center", bbox_to_anchor=(0.5, -0.1), fontsize=9
)
ax.set_title("Prediction Confidence at T30\nBased on top-1 prediction score", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}vis_09_confidence_donut.png")
plt.show()

display(Markdown("""
**Logic:** Confidence is the model's top-1 prediction probability score — available *before* seeing the outcome.
- High (≥ 0.5): model assigns >50% probability to its top prediction
- Medium (0.2–0.5): moderate certainty
- Low (< 0.2): model is uncertain — prediction unreliable
"""))


# ══════════════════════════════════════════════════════════════
# VIS-11: COST vs ACCURACY QUADRANT — INBOUND (A7)
# No quadrant labels. X-cap at max+5B. Top N labels only.
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-11: Cost vs Accuracy Quadrant (Inbound)"))

quad = spend_spec.merge(
    perf_spec[["ending_specialty", "hit_rate_at_5", "total_appearances", "display"]],
    left_on="code", right_on="ending_specialty", how="inner"
)
quad["spend_b"] = quad["total_spend"].astype(float) / 1e9

max_spend = quad["spend_b"].max()
x_cap = max_spend + 5
avg_hit = quad["hit_rate_at_5"].mean()
med_spend = quad["spend_b"].median()

fig, ax = plt.subplots(figsize=(10, 7))
sizes_q = np.clip(quad["total_appearances"] / quad["total_appearances"].max() * 400, 30, 400)

for idx, row in quad.iterrows():
    clr = GREEN if row["hit_rate_at_5"] >= avg_hit else RED
    ax.scatter(row["spend_b"], row["hit_rate_at_5"] * 100,
               s=sizes_q.loc[idx] if idx in sizes_q.index else 60,
               c=clr, alpha=0.7, edgecolors="white", linewidth=0.5)

# Top 10 by spend only — avoids label clutter
top_n_labels = quad.nlargest(10, "spend_b")
for _, row in top_n_labels.iterrows():
    ax.annotate(row["display"][:25],
                (row["spend_b"], row["hit_rate_at_5"] * 100),
                fontsize=7, xytext=(5, 5), textcoords="offset points", color=TEXT_MED)

ax.axvline(med_spend, color=GRID_CLR, linestyle="--", linewidth=1)
ax.axhline(avg_hit * 100, color=GRID_CLR, linestyle="--", linewidth=1)

ax.set_xlabel("Total Downstream Spend ($B)")
ax.set_ylabel("Inbound Hit@5 at T30 (%)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xlim(0, x_cap)
ax.set_ylim(0, 100)
ax.grid(color=GRID_CLR, linewidth=0.5, alpha=0.5)
ax.set_title("Inbound: Downstream Spend vs Prediction Accuracy by Specialty\n"
             "Point size = trigger volume. Dashed lines = median spend / average accuracy.")
plt.tight_layout()
plt.savefig(f"{OUT}vis_11_quadrant.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-EXTRA: CONSISTENCY HEATMAP (A8)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-EXTRA: Consistency Heatmap"))

import seaborn as sns
seg_pivot = perf_seg[perf_seg["member_segment"] != "Unknown"].pivot(
    index="model", columns="member_segment", values="hit_at_5"
).reindex(MODELS) * 100

fig, ax = plt.subplots(figsize=(8, 3.5))
sns.heatmap(seg_pivot, annot=True, fmt=".1f", cmap="YlGn",
            linewidths=0.5, ax=ax, vmin=60, vmax=100,
            cbar_kws={"label": "Hit@5 %"})
ax.set_title("Hit@5 by Model × Segment at T30")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}vis_extra_heatmap.png")
plt.show()


# ══════════════════════════════════════════════════════════════
display(Markdown(f"""
---
## All Visuals Generated

| File | Slide | Direction |
|---|---|---|
| vis_01_model_comparison.png | W2-1, W3-3 | Overall |
| vis_02a_specialty_top15_bot15.png | W2-2, W3-4 | Inbound — Option A |
| vis_02b_specialty_top20.png | W2-2, W3-4 | Inbound — Option B (top) |
| vis_02b_specialty_bot20.png | W2-2, W3-4 | Inbound — Option B (bottom) |
| vis_02c_specialty_top10_bot10.png | W2-2, W3-4 | Inbound — Option C |
| vis_02d_specialty_tiers.png | W2-2, W3-4 | Inbound — Option D |
| vis_04a_sequence_depth.png | W2-4 left | Overall |
| vis_04b_time_window.png | W2-4 center | Overall |
| vis_04c_provider_density.png | W2-4 right | Overall |
| vis_05_dollar_funnel.png | W3-1 | N/A |
| vis_07a_conditions_top15.png | W3-5 | Outbound |
| vis_07b_conditions_bot15.png | W3-5 | Outbound |
| vis_08_drivers.png | W3-6 | Overall |
| vis_09_confidence_donut.png | W3-7 | Score-based |
| vis_11_quadrant.png | A7 | Inbound |
| vis_extra_heatmap.png | A8 | Overall |

Total time: {time.time()-t_total:.1f}s
"""))
print("All visuals done.")
