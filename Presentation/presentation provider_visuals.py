# ============================================================
# PROV-VIS-06: Provider Transition Tables + Segment Heatmap
# Requires: t2, t3, seg_pivot from Block 8c
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os

GREEN    = "#059669"
RED      = "#DC2626"
BLUE     = "#3B82F6"
GREY     = "#9CA3AF"
PURPLE   = "#8172B2"
TEXT_DARK = "#1F2937"
TEXT_MED  = "#6B7280"
GRID_CLR  = "#E5E7EB"
MCOLORS  = {"SASRec": BLUE, "HSTU": PURPLE, "BERT4Rec": GREEN, "Markov": GREY}
MODELS   = ["SASRec", "HSTU", "BERT4Rec", "Markov"]
OUT = "./presentation_visuals/"
os.makedirs(OUT, exist_ok=True)


# ── VIS-06a: Top Transitions by Volume (excl Lab) ────────────
display(Markdown("---\n## PROV-VIS-06a: Top Transitions by Volume (excl Lab)"))

df = t2.head(10).copy()
n = len(df)
labels = [f"{r['from_name'][:18]} → {r['to_name'][:18]}" for _, r in df.iterrows()]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(range(n), df["times_predicted"].values, color=BLUE, edgecolor="white", height=0.6)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
for i, row in enumerate(df.itertuples()):
    ax.text(row.times_predicted + 5, i,
            f"correct: {int(row.times_correct)} ({row.accuracy*100:.0f}%)  |  train: {int(row.train_transitions):,}",
            va="center", fontsize=7, color=TEXT_MED)
ax.set_xlabel("Times Predicted (rank-1)")
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Top 10 Provider Transitions by Prediction Volume (excl Lab)\n"
             f"From → To  |  SASRec T30  |  Accuracy & training evidence shown")
plt.tight_layout()
plt.savefig(f"{OUT}prov_vis_06a_transitions_volume.png")
plt.show()


# ── VIS-06b: Highest Accuracy Transitions (excl Lab) ─────────
display(Markdown("---\n## PROV-VIS-06b: Highest Accuracy Transitions (excl Lab)"))

df = t3.head(10).copy()
n = len(df)
labels = [f"{r['from_name'][:18]} → {r['to_name'][:18]}" for _, r in df.iterrows()]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(range(n), df["accuracy"].values * 100, color=GREEN, edgecolor="white", height=0.6)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
for i, row in enumerate(df.itertuples()):
    ax.text(row.accuracy*100 + 0.5, i,
            f"n={int(row.times_predicted)}  |  train: {int(row.train_transitions):,}",
            va="center", fontsize=7, color=TEXT_MED)
ax.set_xlabel("Accuracy (%)")
ax.set_xlim(0, 110)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
ax.set_title("Top 10 Most Accurate Provider Transitions (excl Lab, min 20 predictions)\n"
             f"From → To  |  SASRec T30  |  Prediction count & training evidence shown")
plt.tight_layout()
plt.savefig(f"{OUT}prov_vis_06b_transitions_accuracy.png")
plt.show()


# ── VIS-06c: Segment Heatmap — All Models ─────────────────────
display(Markdown("---\n## PROV-VIS-06c: All Models × Segment"))

fig, ax = plt.subplots(figsize=(8, 3.5))
plot_data = seg_pivot.reindex(MODELS)
plot_data = plot_data[[c for c in plot_data.columns if c != "Unknown"]]
sns.heatmap(plot_data, annot=True, fmt=".1f", cmap="YlGn",
            linewidths=0.5, ax=ax, vmin=0, vmax=55,
            cbar_kws={"label": "Hit@5 %"})
ax.set_title("Provider Hit@5 by Model × Member Segment at T30")
ax.set_ylabel("")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{OUT}prov_vis_06c_segment_heatmap.png")
plt.show()


display(Markdown(f"""
---
## Provider Transition Visuals Generated

| File | Content | Direction |
|---|---|---|
| prov_vis_06a_transitions_volume.png | Top 10 transitions by volume (excl Lab) | From → To |
| prov_vis_06b_transitions_accuracy.png | Top 10 transitions by accuracy (excl Lab) | From → To |
| prov_vis_06c_segment_heatmap.png | All models × segment | Overall |
"""))
print("Provider transition visuals done.")
