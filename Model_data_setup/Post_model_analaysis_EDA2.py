# ============================================================
# NB_Analysis_02_diagnosis.py
# Purpose : Model performance broken down by diagnosis code
# Source  : A870800_gen_rec_analysis_perf_by_diag
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
# Model Performance Analysis — By Diagnosis Code
**Caveats:**
- Diagnosis codes with fewer than 20 triggers excluded from Top 15 accuracy charts
  (small N inflates accuracy — not reliable signal)
- All windows available in data; charts default to T0 to 30 Days
- Markov predictions are window-agnostic
"""))


# ════════════════════════════════════════════════════════════
# SECTION A — Load and Audit
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section A — Load and Audit"))

df = client.query(f"""
    SELECT model, time_bucket, trigger_dx, trigger_volume
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_1, ndcg_at_3, ndcg_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_by_diag`
    ORDER BY model, time_bucket, trigger_volume DESC
""").to_dataframe()

print(f"Loaded {len(df):,} rows")
print(f"Unique diagnosis codes : {df['trigger_dx'].nunique():,}")
print(f"Models present         : {sorted(df['model'].unique())}")
print(f"Time buckets           : {sorted(df['time_bucket'].unique())}")
print(f"Section A done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION B — Top 15 Diagnosis Codes by Trigger Volume
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section B — Top 15 Diagnosis Codes by Trigger Volume (T0 to 30 Days)"))

# Volume is same across models for same dx+window — use one model to rank
b_base = df[(df["time_bucket"] == "T0_30") & (df["model"] == MODELS[0])].copy()
top15_dx = b_base.nlargest(15, "trigger_volume")["trigger_dx"].tolist()

b = df[(df["time_bucket"] == "T0_30") & (df["trigger_dx"].isin(top15_dx))].copy()

fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(top15_dx))
w = 0.25
for i, model in enumerate([m for m in MODELS if m in b["model"].unique()]):
    sub = b[b["model"] == model].set_index("trigger_dx").reindex(top15_dx)
    bars = ax.bar(x + i * w, sub["trigger_volume"].fillna(0), w,
                  label=model, color=MCOLORS.get(model, "#999"), edgecolor="white")

ax.set_xticks(x + w)
ax.set_xticklabels(top15_dx, rotation=40, ha="right", fontsize=9)
ax.set_title("Top 15 Diagnosis Codes by Trigger Volume — T0 to 30 Days",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Trigger Count")
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("analysis_02_top15_volume.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Section B done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION C — NDCG@3 for Top 15 Diagnosis Codes by Volume
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section C — NDCG@3 for Top 15 Diagnosis Codes (T0 to 30 Days)"))

c = df[(df["time_bucket"] == "T0_30") & (df["trigger_dx"].isin(top15_dx))].copy()

fig, ax = plt.subplots(figsize=(16, 7))
for i, model in enumerate([m for m in MODELS if m in c["model"].unique()]):
    sub = c[c["model"] == model].set_index("trigger_dx").reindex(top15_dx)
    ax.plot(range(len(top15_dx)), sub["ndcg_at_3"].fillna(0),
            marker="o", linewidth=2, markersize=6,
            label=model, color=MCOLORS.get(model, "#999"))

ax.set_xticks(range(len(top15_dx)))
ax.set_xticklabels(top15_dx, rotation=40, ha="right", fontsize=9)
ax.set_title("NDCG@3 by Diagnosis Code — Top 15 by Volume, T0 to 30 Days",
             fontsize=11, fontweight="bold")
ax.set_ylabel("NDCG@3")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("analysis_02_ndcg_top15.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Section C done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION D — Best and Worst Predicted Diagnosis Codes
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section D — Best and Worst Predicted Diagnosis Codes (SASRec, T0 to 30 Days, min 20 triggers)"))

# SASRec only for best/worst — filter low volume
d = df[(df["time_bucket"] == "T0_30") & (df["model"] == "SASRec") &
       (df["trigger_volume"] >= 20)].copy()

best  = d.nlargest(15, "hit_at_3")[["trigger_dx", "trigger_volume", "hit_at_3", "ndcg_at_3"]]
worst = d.nsmallest(15, "hit_at_3")[["trigger_dx", "trigger_volume", "hit_at_3", "ndcg_at_3"]]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for ax, data, title, color in zip(
    axes,
    [best, worst],
    ["Top 15 — Best Predicted Diagnosis Codes",
     "Top 15 — Hardest Diagnosis Codes to Predict"],
    ["#55A868", "#C44E52"]
):
    bars = ax.barh(data["trigger_dx"][::-1], data["hit_at_3"][::-1],
                   color=color, edgecolor="white")
    for bar, vol in zip(bars, data["trigger_volume"][::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"n={vol:,}", va="center", fontsize=8)
    ax.set_title(f"SASRec — {title}\nT0 to 30 Days, min 20 triggers",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Hit@3")
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("analysis_02_best_worst_dx.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Section D done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION E — SASRec Lift over Markov by Diagnosis Code
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section E — SASRec Lift over Markov by Diagnosis Code (T0 to 30 Days, NDCG@3)"))

e_sasrec = df[(df["time_bucket"] == "T0_30") & (df["model"] == "SASRec")].set_index("trigger_dx")
e_markov = df[(df["time_bucket"] == "T0_30") & (df["model"] == "Markov")].set_index("trigger_dx")

lift = e_sasrec[["trigger_volume", "ndcg_at_3"]].join(
    e_markov[["ndcg_at_3"]], rsuffix="_markov"
).dropna()
lift["lift"] = lift["ndcg_at_3"] - lift["ndcg_at_3_markov"]
lift = lift[lift["trigger_volume"] >= 20]

top15_lift   = lift.nlargest(15, "lift")
bot15_lift   = lift.nsmallest(15, "lift")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, data, title in zip(
    axes,
    [top15_lift, bot15_lift],
    ["Top 15 — SASRec Gains Most over Markov",
     "Top 15 — Markov Outperforms SASRec"]
):
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in data["lift"][::-1]]
    ax.barh(data.index[::-1], data["lift"][::-1], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{title}\nNDCG@3 Lift — T0 to 30 Days, min 20 triggers",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("SASRec NDCG@3 minus Markov NDCG@3")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("analysis_02_lift_over_markov.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Section E done — {time.time()-t0:.1f}s")
print("NB_Analysis_02 complete")
