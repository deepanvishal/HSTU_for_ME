# ============================================================
# NB_Analysis_05_window_comparison.py
# Purpose : Compare all 3 models across time windows at K=5
# Source  : A870800_gen_rec_analysis_perf_overall
# ============================================================
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.cloud import bigquery
from IPython.display import display, Markdown

DS     = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
client = bigquery.Client(project="anbc-hcb-dev")

WINDOWS  = ["T0_30", "T30_60", "T60_180"]
WLABELS  = {"T0_30": "0-30 Days", "T30_60": "30-60 Days", "T60_180": "60-180 Days"}
MODELS   = ["SASRec", "BERT4Rec", "Markov"]
MCOLORS  = {"SASRec": "#4C72B0", "BERT4Rec": "#DD8452", "Markov": "#55A868"}
MMARKERS = {"SASRec": "o", "BERT4Rec": "s", "Markov": "^"}

display(Markdown("""
# Time Window Comparison — All Models at K=5
**Note:** Markov predictions are window-agnostic.
Performance degradation at T30-60 and T60-180 reflects absence of temporal modeling.
"""))

df = client.query(f"""
    SELECT model, time_bucket, n_triggers
        ,hit_at_5, ndcg_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_overall`
    WHERE member_segment = 'ALL'
    ORDER BY model, time_bucket
""").to_dataframe()

df["time_bucket"] = pd.Categorical(df["time_bucket"], WINDOWS, ordered=True)
df = df.sort_values(["model", "time_bucket"])
models_here = [m for m in MODELS if m in df["model"].unique()]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ── Chart 1: Hit@5 line plot across windows ───────────────
ax = axes[0]
for model in models_here:
    sub = df[df["model"] == model].set_index("time_bucket").reindex(WINDOWS)
    ax.plot([WLABELS[w] for w in WINDOWS], sub["hit_at_5"].fillna(0),
            marker=MMARKERS.get(model, "o"), linewidth=2.5, markersize=9,
            label=model, color=MCOLORS.get(model, "#999"))
    for w in WINDOWS:
        val = sub.loc[w, "hit_at_5"] if w in sub.index else np.nan
        n   = sub.loc[w, "n_triggers"] if w in sub.index else np.nan
        if pd.notna(val):
            ax.annotate(f"{val:.3f}\nn={int(n):,}",
                        xy=(WLABELS[w], val),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8,
                        color=MCOLORS.get(model, "#999"))
ax.set_title("Hit@5 Across Time Windows", fontsize=12, fontweight="bold")
ax.set_ylabel("Hit@5")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.4)

# ── Chart 2: NDCG@5 line plot across windows ─────────────
ax = axes[1]
for model in models_here:
    sub = df[df["model"] == model].set_index("time_bucket").reindex(WINDOWS)
    ax.plot([WLABELS[w] for w in WINDOWS], sub["ndcg_at_5"].fillna(0),
            marker=MMARKERS.get(model, "o"), linewidth=2.5, markersize=9,
            label=model, color=MCOLORS.get(model, "#999"))
    for w in WINDOWS:
        val = sub.loc[w, "ndcg_at_5"] if w in sub.index else np.nan
        if pd.notna(val):
            ax.annotate(f"{val:.3f}",
                        xy=(WLABELS[w], val),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8,
                        color=MCOLORS.get(model, "#999"))
ax.set_title("NDCG@5 Across Time Windows", fontsize=12, fontweight="bold")
ax.set_ylabel("NDCG@5")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(linestyle="--", alpha=0.4)

# ── Chart 3: Heatmap — model × window, both metrics ──────
ax = axes[2]
heat_data = df.pivot_table(index="model", columns="time_bucket",
                            values="hit_at_5", aggfunc="first")
heat_data = heat_data.reindex(models_here).reindex(columns=WINDOWS)
heat_data.columns = [WLABELS[w] for w in WINDOWS]

# Annotation: Hit@5 / NDCG@5
ndcg_data = df.pivot_table(index="model", columns="time_bucket",
                             values="ndcg_at_5", aggfunc="first")
ndcg_data = ndcg_data.reindex(models_here).reindex(columns=WINDOWS)
n_rows, n_cols = heat_data.shape
annot = np.full((n_rows, n_cols), "", dtype=object)
for i, model in enumerate(models_here):
    for j, w in enumerate(WINDOWS):
        h = heat_data.iloc[i, j]
        n = ndcg_data.loc[model, w] if model in ndcg_data.index else np.nan
        if pd.notna(h) and pd.notna(n):
            annot[i, j] = f"Hit={h:.3f}\nNDCG={n:.3f}"

sns.heatmap(heat_data.astype(float).fillna(0), ax=ax,
            annot=annot, fmt="", cmap="YlGn",
            linewidths=0.8, annot_kws={"size": 9},
            vmin=0, vmax=1, cbar_kws={"label": "Hit@5"})
ax.set_title("Hit@5 + NDCG@5 Summary\n(cell = Hit@5, annotation = both)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

fig.suptitle("Model Comparison — All Time Windows at K=5",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("analysis_05_window_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("NB_Analysis_05 complete")
