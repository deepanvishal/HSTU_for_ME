# ============================================================
# Presentation Visuals — All Charts
# Source: analysis_perf_full (null-filtered, apples-to-apples 5pct)
#         analysis_perf_by_ending_specialty (inbound to specialty)
#         analysis_perf_by_diag (outbound from diagnosis)
#         trigger_scores + markov_trigger_scores_5pct (raw)
#         f_spend_summary (dollar context)
#         markov_train (specialty/CCSR name mappings)
#         claims_gen_rec_2022_2025_sfl (provider counts)
#         test_sequences_5pct (sequence lengths)
# Metric: Hit@5 throughout
# Requires: client, DS from Block 1
# ============================================================
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np
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
import os; os.makedirs(OUT, exist_ok=True)

t_total = time.time()


# ══════════════════════════════════════════════════════════════
# DATA LOADS (all from BQ, null-filtered where needed)
# ══════════════════════════════════════════════════════════════
display(Markdown("## Loading data..."))

# 1. Overall performance (null-filtered)
perf_full = client.query(f"""
    SELECT model, time_bucket, member_segment
        ,n_triggers, hit_at_5, ndcg_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_full`
    WHERE member_segment = 'ALL'
    ORDER BY model, time_bucket
""").to_dataframe()

# 2. By segment (null-filtered)
perf_seg = client.query(f"""
    SELECT model, time_bucket, member_segment
        ,n_triggers, hit_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_full`
    WHERE member_segment != 'ALL'
      AND time_bucket = 'T0_30'
    ORDER BY model, member_segment
""").to_dataframe()

# 3. By ending specialty — INBOUND (already null-filtered in SQL)
perf_spec = client.query(f"""
    SELECT ending_specialty, total_appearances, predicted_at_5
        ,ROUND(SAFE_DIVIDE(predicted_at_5, total_appearances), 4) AS hit_rate_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_by_ending_specialty`
    WHERE time_bucket = 'T0_30'
      AND model = '{BEST_MODEL}'
      AND total_appearances >= 20
    ORDER BY hit_rate_at_5 DESC
""").to_dataframe()

# 4. Specialty name mapping
spec_names = client.query(f"""
    SELECT DISTINCT next_specialty AS code, next_specialty_desc AS name
    FROM `{DS}.A870800_gen_rec_markov_train`
    WHERE next_specialty IS NOT NULL
""").to_dataframe()

perf_spec = perf_spec.merge(spec_names, left_on="ending_specialty", right_on="code", how="left")
perf_spec["display"] = perf_spec["name"].fillna(perf_spec["ending_specialty"])

# 5. By diagnosis — OUTBOUND (need null filter)
perf_dx = client.query(f"""
    WITH all_scores AS (
        SELECT trigger_dx, hit_at_5
        FROM `{DS}.A870800_gen_rec_trigger_scores`
        WHERE time_bucket = 'T0_30' AND model = '{BEST_MODEL}'
          AND true_labels IS NOT NULL AND true_labels != ''
          AND top5_predictions IS NOT NULL AND top5_predictions != ''
        UNION ALL
        SELECT trigger_dx, hit_at_5
        FROM `{DS}.A870800_gen_rec_markov_trigger_scores_5pct`
        WHERE time_bucket = 'T0_30' AND model = 'Markov'
          AND true_labels IS NOT NULL AND true_labels != ''
          AND top5_predictions IS NOT NULL AND top5_predictions != ''
    )
    SELECT trigger_dx
        ,COUNT(*) AS trigger_volume
        ,ROUND(AVG(hit_at_5), 4) AS hit_at_5
    FROM all_scores
    GROUP BY trigger_dx
    HAVING COUNT(*) >= 20
    ORDER BY hit_at_5 DESC
""").to_dataframe()

# CCSR mapping for diagnosis names
ccsr_map = client.query(f"""
    SELECT DISTINCT trigger_dx, trigger_ccsr_desc
    FROM `{DS}.A870800_gen_rec_markov_train`
    WHERE trigger_ccsr_desc IS NOT NULL
""").to_dataframe()
perf_dx = perf_dx.merge(ccsr_map, on="trigger_dx", how="left")
perf_dx["display"] = perf_dx["trigger_ccsr_desc"].fillna(perf_dx["trigger_dx"])

# 6. Sequence length buckets (null-filtered, Hit@5)
seq_buckets = client.query(f"""
    WITH seq_lengths AS (
        SELECT member_id, trigger_date, trigger_dx
            ,MAX(recency_rank) AS seq_length
        FROM `{DS}.A870800_gen_rec_test_sequences_5pct`
        GROUP BY 1, 2, 3
    ),
    joined AS (
        SELECT s.hit_at_5
            ,CASE
                WHEN sl.seq_length < 5  THEN '<5'
                WHEN sl.seq_length < 10 THEN '5-9'
                WHEN sl.seq_length < 15 THEN '10-14'
                ELSE '15-20'
            END AS seq_bucket
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
bucket_order = {"<5": 0, "5-9": 1, "10-14": 2, "15-20": 3}
seq_buckets["order"] = seq_buckets["seq_bucket"].map(bucket_order)
seq_buckets = seq_buckets.sort_values("order")

# 7. Confidence distribution (Hit@5, null-filtered)
conf = client.query(f"""
    SELECT hit_at_5, COUNT(*) AS n
    FROM `{DS}.A870800_gen_rec_trigger_scores`
    WHERE time_bucket = 'T0_30' AND model = '{BEST_MODEL}'
      AND true_labels IS NOT NULL AND true_labels != ''
      AND top5_predictions IS NOT NULL AND top5_predictions != ''
    GROUP BY 1
""").to_dataframe()

# 8. Provider count per specialty
prov_counts = client.query(f"""
    SELECT specialty_ctg_cd AS code
        ,COUNT(DISTINCT srv_prvdr_id) AS provider_count
    FROM `{DS}.A870800_claims_gen_rec_2022_2025_sfl`
    WHERE specialty_ctg_cd IS NOT NULL AND TRIM(specialty_ctg_cd) != ''
    GROUP BY 1 ORDER BY 2 DESC
""").to_dataframe()
prov_counts = prov_counts.merge(spec_names, on="code", how="left")
prov_counts["display"] = prov_counts["name"].fillna(prov_counts["code"])

# 9. Spend per specialty (for quadrant)
spend_spec = client.query(f"""
    SELECT grouping_code AS code, grouping_desc AS name
        ,SUM(total_allowed_amt) AS total_spend
        ,SUM(visit_count) AS visits
    FROM `{DS}.A870800_gen_rec_f_spend_summary`
    WHERE summary_type = 'specialty'
      AND visit_type IN ('downstream', 'v2')
      AND grouping_code IS NOT NULL AND grouping_code != ''
    GROUP BY 1, 2 ORDER BY 3 DESC
""").to_dataframe()

print(f"All data loaded — {time.time()-t_total:.1f}s")


# ══════════════════════════════════════════════════════════════
# VIS-01: MODEL COMPARISON — GROUPED BAR (W2-1, W3-3)
# Overall Hit@5 across T30/T60/T180 — baseline vs best
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-01: Model Comparison — Hit@5 Across Windows"))

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(WINDOWS))
w = 0.25

for i, model in enumerate(MODELS):
    sub = perf_full[perf_full["model"] == model].set_index("time_bucket").reindex(WINDOWS)
    vals = sub["hit_at_5"].fillna(0).values * 100
    bars = ax.bar(x + i * w - w, vals, w, label=model,
                  color=MCOLORS[model], edgecolor="white", alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=TEXT_DARK)

ax.set_xticks(x)
ax.set_xticklabels([WLABELS[w_] for w_ in WINDOWS], fontsize=11)
ax.set_ylabel("Prediction Accuracy (Hit@5)", fontsize=11)
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(fontsize=10, loc="upper left")
ax.grid(axis="y", color=GRID_CLR, linewidth=0.5)
ax.set_title("Prediction Accuracy Across Time Windows\nAll Models at K=5 (5% sample, null-filtered)")
plt.tight_layout()
plt.savefig(f"{OUT}vis_01_model_comparison.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-02: SPECIALTY RANKING — HORIZONTAL BAR (W2-2, W3-4)
# INBOUND: When members end up at this specialty, how often
# did we predict it in top 5?
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-02: Specialty Ranking — Inbound Hit@5 at T30"))

df = perf_spec.copy()
n_spec = len(df)
avg_h5 = df["hit_rate_at_5"].mean()

colors = []
for i, row in df.iterrows():
    if df.index.get_loc(i) < 10:
        colors.append(GREEN)
    elif df.index.get_loc(i) >= n_spec - 10:
        colors.append(RED)
    else:
        colors.append(GREY)

fig, ax = plt.subplots(figsize=(10, max(8, n_spec * 0.28)))
y = np.arange(n_spec)
ax.barh(y, df["hit_rate_at_5"].values * 100, color=colors, edgecolor="white", height=0.7)
ax.set_yticks(y)
ax.set_yticklabels(df["display"].values, fontsize=8)
ax.invert_yaxis()
ax.axvline(avg_h5 * 100, color=TEXT_MED, linestyle="--", linewidth=1)
ax.text(avg_h5 * 100 + 1, n_spec - 1, f"Avg: {avg_h5*100:.1f}%",
        fontsize=9, color=TEXT_MED, style="italic")
ax.set_xlabel("Prediction Accuracy (Hit@5)", fontsize=11)
ax.set_xlim(0, 100)
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title(f"Inbound Specialty Prediction Accuracy at T30 ({BEST_MODEL})\n"
             "When members visit this specialty, how often did we predict it?",
             fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}vis_02_specialty_ranking.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-04a: SEQUENCE DEPTH — MINI BAR (W2-4 left)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-04a: Sequence Depth Signal"))

fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.bar(seq_buckets["seq_bucket"], seq_buckets["hit_at_5"] * 100,
       color=GREEN, edgecolor="white", width=0.6)
for i, row in seq_buckets.iterrows():
    ax.text(row["seq_bucket"], row["hit_at_5"] * 100 + 1,
            f"{row['hit_at_5']*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Hit@5 at T30 (%)")
ax.set_xlabel("Prior Visit Count")
ax.set_ylim(0, 100)
ax.grid(axis="y", color=GRID_CLR, linewidth=0.5)
ax.set_title("Sequence Depth", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}vis_04a_sequence_depth.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-04b: TIME WINDOW — MINI BAR (W2-4 center)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-04b: Time Window Signal"))

best_windows = perf_full[perf_full["model"] == BEST_MODEL].set_index("time_bucket").reindex(WINDOWS)
wlabels_short = ["T30", "T60", "T180"]
wcolors = [GREEN, "#34D399", "#6EE7B7"]

fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax.bar(wlabels_short, best_windows["hit_at_5"].values * 100,
       color=wcolors, edgecolor="white", width=0.6)
for i, v in enumerate(best_windows["hit_at_5"].values * 100):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_ylabel("Hit@5 (%)")
ax.set_ylim(0, 105)
ax.grid(axis="y", color=GRID_CLR, linewidth=0.5)
ax.set_title("Time Window", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}vis_04b_time_window.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-04c: PROVIDER DENSITY — MINI BAR (W2-4 right)
# Top 5 and bottom 5 specialties by provider count
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-04c: Provider Density"))

top5 = prov_counts.head(5)
bot5 = prov_counts.tail(5).iloc[::-1]
prov_plot = pd.concat([top5, bot5])
prov_colors = [RED]*5 + [GREEN]*5

fig, ax = plt.subplots(figsize=(3.5, 4))
ax.barh(range(len(prov_plot)), prov_plot["provider_count"].values,
        color=prov_colors, edgecolor="white", height=0.6)
ax.set_yticks(range(len(prov_plot)))
ax.set_yticklabels(prov_plot["display"].values, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Unique Providers")
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Provider Density", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}vis_04c_provider_density.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-05: DOLLAR FUNNEL (W3-1)
# Hardcoded from Block 6a results
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-05: Dollar Funnel"))

funnel_data = [
    ("All Claims", 46.3, "311M"),
    ("Trigger-Day Claims", 17.4, "103M"),
    ("Trigger Dx Claims", 0.5, "3.6M"),
    ("Next Visit (V2) Claims", 11.8, "78M"),
    ("T180 Downstream (approx)", 27.0, "180M"),
]

fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(funnel_data))
widths = [d[1] for d in funnel_data]
max_w = max(widths)
bar_colors = [GREY, GREY, GREY, GREEN, GREEN]

bars = ax.barh(y_pos, widths, color=bar_colors, edgecolor="white", height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels([d[0] for d in funnel_data], fontsize=10)
ax.invert_yaxis()
for bar, (label, spend, claims) in zip(bars, funnel_data):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"${spend}B  |  {claims} claims",
            va="center", fontsize=9, color=TEXT_DARK)
ax.set_xlabel("Spend ($B)", fontsize=11)
ax.set_xlim(0, max_w * 1.8)
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Dollar Funnel — From Total Claims to Prediction Scope", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}vis_05_dollar_funnel.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-07: CONDITION PATHWAY CONSISTENCY — HORIZONTAL BAR (W3-5)
# OUTBOUND: Given this diagnosis, how often can we predict
# the next specialty?
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-07: Condition Pathway Consistency — Outbound Hit@5 at T30"))

dx_top15 = perf_dx.head(15)
dx_bot15 = perf_dx.tail(15).iloc[::-1]

for subset, label, fname in [
    (dx_top15, "Top 15 — Most Predictable (Outbound)", "vis_07a_conditions_top15"),
    (dx_bot15, "Bottom 15 — Least Predictable (Outbound)", "vis_07b_conditions_bot15"),
]:
    n = len(subset)
    clr = GREEN if "Top" in label else RED
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.35)))
    ax.barh(range(n), subset["hit_at_5"].values * 100,
            color=clr, edgecolor="white", height=0.6)
    ax.set_yticks(range(n))
    display_names = [d[:55] + "..." if len(d) > 55 else d for d in subset["display"].values]
    ax.set_yticklabels(display_names, fontsize=8)
    ax.invert_yaxis()
    for i, v in enumerate(subset["hit_at_5"].values * 100):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Prediction Accuracy (Hit@5 at T30)", fontsize=10)
    ax.set_xlim(0, 110)
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
    ax.set_title(f"Outbound Diagnosis Pathway Consistency — {label}\n"
                 f"Given this diagnosis, how often can we predict the next specialty? ({BEST_MODEL})",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUT}{fname}.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-08: DRIVER IMPORTANCE — HORIZONTAL BAR (W3-6)
# Sequence depth range + segment range as proxy
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-08: Prediction Drivers"))

seq_range = seq_buckets["hit_at_5"].max() - seq_buckets["hit_at_5"].min()
seg_data = perf_seg[(perf_seg["model"] == BEST_MODEL) &
                    (perf_seg["member_segment"] != "Unknown")]
seg_range = seg_data["hit_at_5"].max() - seg_data["hit_at_5"].min()

drivers = pd.DataFrame([
    {"Driver": "Visit History Depth\n(prior visit count)", "Impact": seq_range * 100},
    {"Driver": "Member Cohort\n(age/gender segment)", "Impact": seg_range * 100},
]).sort_values("Impact", ascending=True)

fig, ax = plt.subplots(figsize=(7, 3))
ax.barh(drivers["Driver"], drivers["Impact"], color=GREEN, edgecolor="white", height=0.5)
for i, v in enumerate(drivers["Impact"]):
    ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("Hit@5 Range Across Levels (%)", fontsize=10)
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Sources of Prediction Variance\n"
             "Wider range = stronger driver of accuracy differences", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}vis_08_drivers.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-09: DONUT — CONFIDENCE DISTRIBUTION (W3-7)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-09: Confidence Distribution"))

total = float(conf["n"].sum())
high = float(conf[conf["hit_at_5"] == 1]["n"].sum())
low = total - high
high_pct = high / total * 100
low_pct = low / total * 100

fig, ax = plt.subplots(figsize=(5, 5))
sizes = [high_pct, low_pct]
colors_donut = [GREEN, RED]
wedges, texts, autotexts = ax.pie(
    sizes, colors=colors_donut, autopct="%1.1f%%",
    startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2)
)
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight("bold")
ax.text(0, 0, f"{high_pct:.1f}%\nhigh\nconfidence",
        ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT_DARK)
ax.legend(["High confidence (Hit@5 = 1)", "Low confidence (Hit@5 = 0)"],
          loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=9)
ax.set_title("Prediction Confidence Distribution at T30", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUT}vis_09_confidence_donut.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-11: COST vs ACCURACY QUADRANT (Appendix A7)
# INBOUND: specialty accuracy vs downstream spend
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-11: Cost vs Accuracy Quadrant (Inbound)"))

quad = spend_spec.merge(
    perf_spec[["ending_specialty", "hit_rate_at_5", "total_appearances", "display"]],
    left_on="code", right_on="ending_specialty", how="inner"
)
quad["spend_b"] = quad["total_spend"].astype(float) / 1e9

med_spend = quad["spend_b"].median()
avg_hit = quad["hit_rate_at_5"].mean()

fig, ax = plt.subplots(figsize=(10, 7))
sizes = np.clip(quad["total_appearances"] / quad["total_appearances"].max() * 400, 30, 400)

for _, row in quad.iterrows():
    clr = GREEN if row["hit_rate_at_5"] >= avg_hit and row["spend_b"] >= med_spend else \
          RED if row["hit_rate_at_5"] < avg_hit and row["spend_b"] >= med_spend else \
          GREY
    ax.scatter(row["spend_b"], row["hit_rate_at_5"] * 100,
               s=sizes[_] if _ in sizes.index else 60,
               c=clr, alpha=0.7, edgecolors="white", linewidth=0.5)
    if row["spend_b"] > med_spend * 1.5 or row["hit_rate_at_5"] > 0.7 or row["hit_rate_at_5"] < 0.05:
        ax.annotate(row["display"][:25], (row["spend_b"], row["hit_rate_at_5"] * 100),
                    fontsize=7, xytext=(4, 4), textcoords="offset points", color=TEXT_MED)

ax.axvline(med_spend, color=GRID_CLR, linestyle="--", linewidth=1)
ax.axhline(avg_hit * 100, color=GRID_CLR, linestyle="--", linewidth=1)

ax.text(quad["spend_b"].max() * 0.95, avg_hit * 100 + 3, "Deploy Now",
        ha="right", fontsize=9, color=GREEN, style="italic")
ax.text(quad["spend_b"].max() * 0.95, avg_hit * 100 - 5, "Priority to Improve",
        ha="right", fontsize=9, color=RED, style="italic")
ax.text(med_spend * 0.3, avg_hit * 100 + 3, "Low Cost, High Accuracy",
        ha="center", fontsize=9, color=TEXT_MED, style="italic")
ax.text(med_spend * 0.3, avg_hit * 100 - 5, "Deprioritize",
        ha="center", fontsize=9, color=TEXT_MED, style="italic")

ax.set_xlabel("Total Downstream Spend ($B)", fontsize=11)
ax.set_ylabel("Inbound Prediction Accuracy (Hit@5 at T30) %", fontsize=11)
ax.set_title("Inbound: Cost vs Accuracy by Ending Specialty\n"
             "Point size = trigger volume. Quadrant = deployment priority.", fontsize=12)
ax.grid(color=GRID_CLR, linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUT}vis_11_quadrant.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# VIS-EXTRA: CONSISTENCY HEATMAP (Appendix A8)
# ══════════════════════════════════════════════════════════════
display(Markdown("---\n## VIS-EXTRA: Consistency by Segment"))

seg_pivot = perf_seg[perf_seg["member_segment"] != "Unknown"].pivot(
    index="model", columns="member_segment", values="hit_at_5"
).reindex(MODELS) * 100

fig, ax = plt.subplots(figsize=(8, 3.5))
import seaborn as sns
sns.heatmap(seg_pivot, annot=True, fmt=".1f", cmap="YlGn",
            linewidths=0.5, ax=ax, vmin=60, vmax=100,
            cbar_kws={"label": "Hit@5 %"})
ax.set_title("Hit@5 by Model × Member Segment at T30\n"
             "Lower variance = more consistent model", fontsize=11)
ax.set_ylabel("")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{OUT}vis_extra_consistency_heatmap.png")
plt.show()


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
display(Markdown(f"""
---
## Visuals Generated

| File | Slide | Direction |
|---|---|---|
| vis_01_model_comparison.png | W2-1, W3-3 | Overall |
| vis_02_specialty_ranking.png | W2-2, W3-4 | Inbound to specialty |
| vis_04a_sequence_depth.png | W2-4 left | Overall |
| vis_04b_time_window.png | W2-4 center | Overall |
| vis_04c_provider_density.png | W2-4 right | Overall |
| vis_05_dollar_funnel.png | W3-1 | N/A |
| vis_07a_conditions_top15.png | W3-5 | Outbound from diagnosis |
| vis_07b_conditions_bot15.png | W3-5 | Outbound from diagnosis |
| vis_08_drivers.png | W3-6 | Overall |
| vis_09_confidence_donut.png | W3-7 | Overall |
| vis_11_quadrant.png | A7 | Inbound to specialty |
| vis_extra_consistency_heatmap.png | A8 | Overall |

All saved to `{OUT}`

Total time: {time.time()-t_total:.1f}s
"""))

print("All visuals done.")
