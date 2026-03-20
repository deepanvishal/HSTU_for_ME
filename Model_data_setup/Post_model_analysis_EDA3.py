# ============================================================
# NB_Analysis_03_ending_specialty.py
# Purpose : Model performance broken down by ending specialty
# Source  : A870800_gen_rec_analysis_perf_by_ending_specialty
# ============================================================
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from google.cloud import bigquery
from IPython.display import display, Markdown

DS     = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
client = bigquery.Client(project="anbc-hcb-dev")

MODELS  = ["SASRec", "BERT4Rec", "Markov"]
MCOLORS = {"SASRec": "#4C72B0", "BERT4Rec": "#DD8452", "Markov": "#55A868"}

display(Markdown("""
# Model Performance Analysis — By Ending Specialty
**Caveats:**
- A specialty "appearance" = it exists in the true label set for a trigger
- Predicted@K = the specialty appeared in the model's top-K predictions for that trigger
- One trigger can contribute to multiple specialties (multi-label)
- Specialties with fewer than 20 appearances excluded from accuracy charts
- Probability cutoff analysis deferred
"""))


# ════════════════════════════════════════════════════════════
# SECTION A — Load and Audit
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section A — Load and Audit"))

df = client.query(f"""
    SELECT model, time_bucket, ending_specialty
        ,total_appearances
        ,predicted_at_1, predicted_at_3, predicted_at_5
        ,hit_rate_at_5
        ,avg_ndcg_at_3
    FROM `{DS}.A870800_gen_rec_analysis_perf_by_ending_specialty`
    ORDER BY model, time_bucket, total_appearances DESC
""").to_dataframe()

# Compute rates
df["hit_rate_at_1"] = df["predicted_at_1"] / df["total_appearances"].clip(lower=1)
df["hit_rate_at_3"] = df["predicted_at_3"] / df["total_appearances"].clip(lower=1)

print(f"Loaded {len(df):,} rows")
print(f"Unique ending specialties : {df['ending_specialty'].nunique():,}")
print(f"Section A done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION B — True Label Density
# Distribution of how many specialties are in each trigger's true label set
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section B — True Label Density"))

display(Markdown("""
**Note:** Label density is inferred from the ending specialty table.
Total appearances per trigger = number of specialties in its true label set.
A trigger with 3 true labels contributes 3 rows here.
"""))

# For T0_30 SASRec (all models same denominator) — count appearances per trigger
density = client.query(f"""
    SELECT
        time_bucket
        ,APPROX_QUANTILES(total_appearances, 100)[OFFSET(50)] AS median_appearances
        ,AVG(total_appearances)                                AS avg_appearances
        ,MAX(total_appearances)                                AS max_appearances
    FROM (
        SELECT time_bucket, SUM(total_appearances) AS total_appearances
        FROM `{DS}.A870800_gen_rec_analysis_perf_by_ending_specialty`
        WHERE model = 'SASRec'
        GROUP BY time_bucket, ending_specialty
    )
    GROUP BY time_bucket
""").to_dataframe()
display(density)
print(f"Section B done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION C — Top 15 Ending Specialties by Volume
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section C — Top 15 Ending Specialties by Volume (T0 to 30 Days)"))

c_base = df[(df["time_bucket"] == "T0_30") & (df["model"] == MODELS[0])].copy()
top15_sp = c_base.nlargest(15, "total_appearances")["ending_specialty"].tolist()

c = df[(df["time_bucket"] == "T0_30") & (df["ending_specialty"].isin(top15_sp))].copy()

fig, axes = plt.subplots(2, 1, figsize=(16, 14))

# Chart 1 — appearance volume
ax = axes[0]
x = np.arange(len(top15_sp))
w = 0.25
for i, model in enumerate([m for m in MODELS if m in c["model"].unique()]):
    sub = c[c["model"] == model].set_index("ending_specialty").reindex(top15_sp)
    ax.bar(x + i * w, sub["total_appearances"].fillna(0), w,
           label=model, color=MCOLORS.get(model, "#999"), edgecolor="white")
ax.set_xticks(x + w)
ax.set_xticklabels(top15_sp, rotation=40, ha="right", fontsize=9)
ax.set_title("Total Appearances — Top 15 Ending Specialties, T0 to 30 Days",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Appearances in True Label Set")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Chart 2 — predicted counts at K=3
ax = axes[1]
for i, model in enumerate([m for m in MODELS if m in c["model"].unique()]):
    sub = c[c["model"] == model].set_index("ending_specialty").reindex(top15_sp)
    ax.bar(x + i * w, sub["predicted_at_3"].fillna(0), w,
           label=model, color=MCOLORS.get(model, "#999"), edgecolor="white")
ax.set_xticks(x + w)
ax.set_xticklabels(top15_sp, rotation=40, ha="right", fontsize=9)
ax.set_title("Correctly Predicted (Top 3) — Top 15 Ending Specialties, T0 to 30 Days",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Count Correctly Predicted at K=3")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("analysis_03_top15_specialty_volume.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Section C done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION D — Top 15 by Prediction Accuracy (Hit Rate at K=3)
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section D — Top 15 Specialties by Prediction Accuracy (T0 to 30 Days, min 20 appearances)"))

d = df[(df["time_bucket"] == "T0_30") & (df["total_appearances"] >= 20)].copy()

fig, axes = plt.subplots(1, len([m for m in MODELS if m in d["model"].unique()]),
                          figsize=(18, 7))
if not hasattr(axes, "__iter__"):
    axes = [axes]

for ax, model in zip(axes, [m for m in MODELS if m in d["model"].unique()]):
    sub = d[d["model"] == model].nlargest(15, "hit_rate_at_3")
    bars = ax.barh(sub["ending_specialty"][::-1], sub["hit_rate_at_3"][::-1],
                   color=MCOLORS.get(model, "#999"), edgecolor="white")
    for bar, n in zip(bars, sub["total_appearances"][::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={n:,}", va="center", fontsize=8)
    ax.set_title(f"{model}\nTop 15 — Hit Rate at K=3",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Hit Rate at K=3")
    ax.set_xlim(0, 1.2)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.suptitle("Top 15 Specialties by Prediction Accuracy — T0 to 30 Days, min 20 appearances",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("analysis_03_top15_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Section D done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION E — Per-Specialty Heatmap: All Models vs Top 15 Specialties
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section E — Model Comparison Heatmap: Hit Rate at K=3 by Specialty (T0 to 30 Days)"))

e = df[(df["time_bucket"] == "T0_30") & (df["total_appearances"] >= 20)].copy()

# Top 15 by total appearances (using SASRec as reference for ranking)
e_base = e[e["model"] == MODELS[0]].nlargest(15, "total_appearances")["ending_specialty"].tolist()
e_plot = e[e["ending_specialty"].isin(e_base)].copy()

pivot = e_plot.pivot_table(
    index="ending_specialty", columns="model", values="hit_rate_at_3"
).reindex(e_base)
pivot = pivot[[m for m in MODELS if m in pivot.columns]]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=11)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        ax.text(j, i, f"{val:.2f}" if not np.isnan(val) else "—",
                ha="center", va="center", fontsize=10,
                color="black" if val < 0.6 else "white")
plt.colorbar(im, ax=ax, label="Hit Rate at K=3")
ax.set_title("Hit Rate at K=3 — Top 15 Specialties by Volume\nT0 to 30 Days, min 20 appearances",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("analysis_03_heatmap_model_vs_specialty.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Section E done — {time.time()-t0:.1f}s")
print("NB_Analysis_03 complete")
