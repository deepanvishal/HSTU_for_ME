# ============================================================
# NB_Analysis_01_overall.py
# Purpose : Overall model performance comparison
# Source  : A870800_gen_rec_analysis_perf_overall
# Models  : SASRec, BERT4Rec, Markov
# ============================================================
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from google.cloud import bigquery
from IPython.display import display, Markdown

DS     = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
client = bigquery.Client(project="anbc-hcb-dev")

WINDOWS = ["T0_30", "T30_60", "T60_180"]
MODELS  = ["SASRec", "BERT4Rec", "Markov"]
MCOLORS = {"SASRec": "#4C72B0", "BERT4Rec": "#DD8452", "Markov": "#55A868"}

display(Markdown("""
# Model Performance Analysis — Overall
**Caveats:**
- Models predict top 5 specialties per trigger per time window
- True label set may contain more than 5 specialties
- Hit@K is binary per trigger — 1 if any top-K prediction matches true label set
- Probability cutoff analysis deferred
- Markov predictions are window-agnostic; same top-5 applied to all windows
"""))


# ════════════════════════════════════════════════════════════
# SECTION A — Data Audit
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section A — Data Audit"))

df = client.query(f"""
    SELECT model, time_bucket, member_segment, n_triggers
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_1, ndcg_at_3, ndcg_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_overall`
    ORDER BY model, time_bucket, member_segment
""").to_dataframe()

print(f"Loaded {len(df):,} rows")

# Row count audit
audit = df[df["member_segment"] == "ALL"].groupby(["model", "time_bucket"])["n_triggers"].sum().unstack("time_bucket")
display(Markdown("### Trigger counts by model and time window (member segment = ALL)"))
display(audit)

# Flag missing models
for m in MODELS:
    if m not in df["model"].unique():
        print(f"WARNING: {m} not found in table — check scoring ran successfully")

print(f"Section A done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION B — Overall Metrics Table + Heatmap
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section B — Overall Metrics (K=3, T0 to 30 Days)"))

b = df[(df["member_segment"] == "ALL") & (df["time_bucket"] == "T0_30")][
    ["model", "hit_at_3", "ndcg_at_3", "hit_at_1", "hit_at_5", "ndcg_at_1", "ndcg_at_5", "n_triggers"]
].rename(columns={
    "model":      "Model",
    "n_triggers": "Triggers",
    "hit_at_1":   "Hit@1", "hit_at_3": "Hit@3", "hit_at_5": "Hit@5",
    "ndcg_at_1":  "NDCG@1","ndcg_at_3": "NDCG@3","ndcg_at_5": "NDCG@5",
}).set_index("Model")

display(Markdown("**T0 to 30 Days — All Members**"))
display(b)

# Heatmap — model vs metric at K=3
fig, ax = plt.subplots(figsize=(10, 4))
heat_data = df[(df["member_segment"] == "ALL") & (df["time_bucket"] == "T0_30")][
    ["model", "hit_at_3", "ndcg_at_3"]
].set_index("model").rename(columns={"hit_at_3": "Hit@3", "ndcg_at_3": "NDCG@3"})
heat_data = heat_data.reindex([m for m in MODELS if m in heat_data.index])
sns.heatmap(heat_data, ax=ax, annot=True, fmt=".3f", cmap="YlGn",
            linewidths=0.5, annot_kws={"size": 13})
ax.set_title("Hit@3 and NDCG@3 — T0 to 30 Days, All Members",
             fontsize=12, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("analysis_01_overall_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Section B done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION C — Hit@3 and NDCG@3 by Time Window
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section C — Performance by Time Window"))

c = df[df["member_segment"] == "ALL"].copy()
c["time_bucket"] = pd.Categorical(c["time_bucket"], categories=WINDOWS, ordered=True)
c = c.sort_values(["time_bucket", "model"])

WLABELS = {"T0_30": "0-30 Days", "T30_60": "30-60 Days", "T60_180": "60-180 Days"}
c["Window"] = c["time_bucket"].map(WLABELS)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, metric, label in zip(axes,
                              ["hit_at_3", "ndcg_at_3"],
                              ["Hit@3", "NDCG@3"]):
    pivot = c.pivot(index="Window", columns="model", values=metric)
    pivot = pivot[[m for m in MODELS if m in pivot.columns]]
    x = np.arange(len(pivot))
    w = 0.25
    for i, model in enumerate(pivot.columns):
        bars = ax.bar(x + i * w, pivot[model], w,
                      label=model, color=MCOLORS.get(model, "#999"),
                      edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(pivot.index, fontsize=10)
    ax.set_title(f"{label} by Time Window — All Members",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel(label)
    ax.set_ylim(0, min(1.05, pivot.values.max() * 1.3 + 0.05))
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("analysis_01_by_window.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Section C done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION D — Performance by Member Segment
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section D — Performance by Member Segment (T0 to 30 Days, Hit@3)"))

d = df[(df["time_bucket"] == "T0_30") & (df["member_segment"] != "ALL")].copy()

if d.empty:
    print("No segment-level data — member_segment breakdown not available")
else:
    segments = sorted(d["member_segment"].unique())
    fig, ax = plt.subplots(figsize=(max(12, len(segments) * 3), 6))
    x = np.arange(len(segments))
    w = 0.25
    for i, model in enumerate([m for m in MODELS if m in d["model"].unique()]):
        vals = [d[(d["member_segment"] == s) & (d["model"] == model)]["hit_at_3"].values
                for s in segments]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        bars = ax.bar(x + i * w, vals, w, label=model,
                      color=MCOLORS.get(model, "#999"), edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + w)
    ax.set_xticklabels(segments, fontsize=10, rotation=15, ha="right")
    ax.set_title("Hit@3 by Member Segment — T0 to 30 Days",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Hit@3")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("analysis_01_by_segment.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Section D done — {time.time()-t0:.1f}s")
print("NB_Analysis_01 complete")
