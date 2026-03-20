# ============================================================
# NB_Analysis_05_window_comparison.py
# Purpose : Compare all 3 models across time windows at K=5
#           Hit@5, NDCG@5, Precision@5, Recall@5
# Source  : A870800_gen_rec_analysis_perf_full
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

METRICS = [
    ("hit_at_5",       "Hit@5"),
    ("ndcg_at_5",      "NDCG@5"),
    ("precision_at_5", "Precision@5"),
    ("recall_at_5",    "Recall@5"),
]

display(Markdown("""
# Time Window Comparison — All Models at K=5
Hit@5, NDCG@5, Precision@5, Recall@5 across T0-30, T30-60, T60-180.

**Note:** Markov predictions are window-agnostic — same top-5 applied across all windows.
Performance degradation at T30-60 and T60-180 reflects absence of temporal modeling.
"""))

t0 = time.time()
df = client.query(f"""
    SELECT model, time_bucket, n_triggers
        ,hit_at_5, ndcg_at_5, precision_at_5, recall_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_full`
    WHERE member_segment = 'ALL'
    ORDER BY model, time_bucket
""").to_dataframe()

df["time_bucket"] = pd.Categorical(df["time_bucket"], WINDOWS, ordered=True)
df = df.sort_values(["model", "time_bucket"])
models_here  = [m for m in MODELS if m in df["model"].unique()]
wlabels_list = [WLABELS[w] for w in WINDOWS]

print(f"Loaded {len(df):,} rows — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# CHART 1 — 2x2 Line plots: all 4 metrics across windows
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for ax, (col, label) in zip(axes, METRICS):
    for model in models_here:
        sub  = df[df["model"] == model].set_index("time_bucket").reindex(WINDOWS)
        vals = sub[col].fillna(0).values
        ax.plot(wlabels_list, vals,
                marker=MMARKERS.get(model, "o"), linewidth=2.5, markersize=9,
                label=model, color=MCOLORS.get(model, "#999"))
        for w, v in zip(WINDOWS, vals):
            ax.annotate(f"{v:.3f}",
                        xy=(WLABELS[w], v),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8.5,
                        color=MCOLORS.get(model, "#999"))

    ax.set_title(f"{label} Across Time Windows", fontsize=12, fontweight="bold")
    ax.set_ylabel(label)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)

# Trigger counts as shared footnote
n_ref = (df[df["model"] == models_here[0]]
         .set_index("time_bucket").reindex(WINDOWS)["n_triggers"].fillna(0))
footnote = "  |  ".join([f"{WLABELS[w]}: n={int(n):,}"
                          for w, n in zip(WINDOWS, n_ref)])
fig.text(0.5, -0.01, f"Trigger counts — {footnote}",
         ha="center", fontsize=9, color="#555555")
fig.suptitle("Model Comparison — All Metrics at K=5",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("analysis_05_line_plots.png", dpi=150, bbox_inches="tight")
plt.show()


# ════════════════════════════════════════════════════════════
# CHART 2 — 2x2 Heatmaps: model × window, one per metric
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.flatten()

n_vals = (df[df["model"] == models_here[0]]
          .set_index("time_bucket").reindex(WINDOWS)["n_triggers"].fillna(0))

for ax, (col, label) in zip(axes, METRICS):
    pivot = (df.pivot_table(index="model", columns="time_bucket",
                             values=col, aggfunc="first")
             .reindex(models_here).reindex(columns=WINDOWS)
             .astype(float).fillna(0))
    pivot.columns = wlabels_list

    annot = np.full(pivot.shape, "", dtype=object)
    for i, model in enumerate(models_here):
        for j in range(len(WINDOWS)):
            v = pivot.iloc[i, j]
            n = n_vals.iloc[j]
            annot[i, j] = f"{v:.3f}\nn={int(n):,}"

    sns.heatmap(pivot, ax=ax, annot=annot, fmt="", cmap="YlGn",
                linewidths=0.8, annot_kws={"size": 9},
                vmin=0, vmax=1, cbar_kws={"label": label})
    ax.set_title(f"{label}", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)

fig.suptitle("Performance Heatmaps — All Metrics at K=5",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("analysis_05_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()


# ════════════════════════════════════════════════════════════
# CHART 3 — Grouped bar: metric × model per window
# One chart per window — all 4 metrics side by side per model
# ════════════════════════════════════════════════════════════
for window in WINDOWS:
    sub = df[df["time_bucket"] == window]
    if sub.empty:
        continue

    metric_cols   = [c for c, _ in METRICS]
    metric_labels = [l for _, l in METRICS]
    n_metrics     = len(metric_cols)
    n_models      = len(models_here)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_metrics)
    w = 0.7 / n_models

    for i, model in enumerate(models_here):
        row  = sub[sub["model"] == model]
        vals = [row[c].values[0] if len(row) > 0 else 0 for c in metric_cols]
        bars = ax.bar(x + i * w - (n_models - 1) * w / 2, vals, w,
                      label=model, color=MCOLORS.get(model, "#999"),
                      edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    n_here = sub[sub["model"] == models_here[0]]["n_triggers"].values
    n_here = int(n_here[0]) if len(n_here) > 0 else 0
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_title(f"All Metrics at K=5 — {WLABELS[window]}  (n={n_here:,} triggers)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"analysis_05_bars_{window}.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# Summary tables
# ════════════════════════════════════════════════════════════
display(Markdown("---\n### Summary Tables — All Metrics at K=5"))
for window in WINDOWS:
    sub = (df[df["time_bucket"] == window]
           [["model", "n_triggers",
             "hit_at_5", "ndcg_at_5", "precision_at_5", "recall_at_5"]]
           .rename(columns={
               "model": "Model", "n_triggers": "Triggers",
               "hit_at_5": "Hit@5", "ndcg_at_5": "NDCG@5",
               "precision_at_5": "Precision@5", "recall_at_5": "Recall@5",
           })
           .set_index("Model").reindex(models_here))
    display(Markdown(f"**{WLABELS[window]}**"))
    display(sub)

print("NB_Analysis_05 complete")
