# ============================================================
# NB_Analysis_02_diagnosis.py
# Purpose : Model performance by diagnosis code
# Source  : A870800_gen_rec_analysis_perf_by_diag
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
MIN_VOL  = 20

display(Markdown("""
# Model Performance Analysis — By Diagnosis Code
**Caveats:**
- Diagnosis codes with fewer than 20 triggers excluded from accuracy charts
- trigger_volume = number of test triggers with this diagnosis code
- All 3 time windows shown; Markov predictions are window-agnostic
- Probability cutoff analysis deferred
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
print(f"Models                 : {sorted(df['model'].unique())}")

# Audit — total transitions per window per model
audit = (df.groupby(["model", "time_bucket"])["trigger_volume"]
         .sum().unstack("time_bucket")
         .reindex(MODELS).reindex(columns=WINDOWS))
display(Markdown("### Total transitions (triggers) by model and time window"))
display(audit)
print(f"Section A done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION B — Top 15 Diagnosis Codes by Trigger Volume (All Windows)
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section B — Top 15 Diagnosis Codes by Trigger Volume"))

# Rank by T0_30 volume, one model reference
ref = df[(df["time_bucket"] == "T0_30") & (df["model"] == MODELS[0])]
top15_dx = ref.nlargest(15, "trigger_volume")["trigger_dx"].tolist()

for window in WINDOWS:
    sub = df[(df["time_bucket"] == window) & (df["trigger_dx"].isin(top15_dx))]
    models_here = [m for m in MODELS if m in sub["model"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Chart 1 — volume bars with Hit@3 label
    ax = axes[0]
    x = np.arange(len(top15_dx))
    w = 0.25
    for i, model in enumerate(models_here):
        s = sub[sub["model"] == model].set_index("trigger_dx").reindex(top15_dx)
        bars = ax.bar(x + i * w, s["trigger_volume"].fillna(0), w,
                      label=model, color=MCOLORS.get(model, "#999"),
                      edgecolor="white")
        for bar, hit in zip(bars, s["hit_at_3"].fillna(0)):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"H3={hit:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color=MCOLORS.get(model, "#999"))
    ax.set_xticks(x + w)
    ax.set_xticklabels(top15_dx, rotation=40, ha="right", fontsize=8)
    ax.set_title(f"Trigger Volume — {WLABELS[window]}\n(bar label = Hit@3 per model)",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Trigger Count")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Chart 2 — NDCG@3 with trigger volume annotation
    ax = axes[1]
    for model in models_here:
        s = sub[sub["model"] == model].set_index("trigger_dx").reindex(top15_dx)
        ax.plot(range(len(top15_dx)), s["ndcg_at_3"].fillna(0),
                marker="o", linewidth=2, markersize=6,
                label=model, color=MCOLORS.get(model, "#999"))

    # Annotate volume once (same across models)
    ref_sub = sub[sub["model"] == models_here[0]].set_index("trigger_dx").reindex(top15_dx)
    for j, (dx, vol) in enumerate(zip(top15_dx, ref_sub["trigger_volume"].fillna(0))):
        ax.text(j, -0.07, f"n={int(vol):,}", ha="center", va="top",
                fontsize=6.5, color="#666666", rotation=45)

    ax.set_xticks(range(len(top15_dx)))
    ax.set_xticklabels(top15_dx, rotation=40, ha="right", fontsize=8)
    ax.set_title(f"NDCG@3 by Diagnosis Code — {WLABELS[window]}\n(n = trigger volume below axis)",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("NDCG@3")
    ax.set_ylim(-0.15, 1.05)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"analysis_02_top15_{window}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Section B done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION C — Best and Worst Predicted Diagnosis Codes
# All windows, SASRec, min 20 triggers
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section C — Best and Worst Predicted Diagnosis Codes (SASRec, min 20 triggers)"))

for window in WINDOWS:
    sub = df[(df["time_bucket"] == window) & (df["model"] == "SASRec") &
             (df["trigger_volume"] >= MIN_VOL)].copy()
    if sub.empty:
        print(f"No data for {window} with min {MIN_VOL} triggers")
        continue

    best  = sub.nlargest(15, "hit_at_3")
    worst = sub.nsmallest(15, "hit_at_3")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, data, title, color in zip(
        axes,
        [best, worst],
        ["Top 15 Best Predicted", "Top 15 Hardest to Predict"],
        ["#55A868", "#C44E52"]
    ):
        rev = data.iloc[::-1]
        bars = ax.barh(rev["trigger_dx"], rev["hit_at_3"], color=color, edgecolor="white")
        for bar, row in zip(bars, rev.itertuples()):
            # Volume + NDCG as second label
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"n={int(row.trigger_volume):,}  NDCG@3={row.ndcg_at_3:.3f}",
                    va="center", fontsize=8)
        ax.set_title(f"SASRec — {title}\n{WLABELS[window]}, min {MIN_VOL} triggers",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Hit@3")
        ax.set_xlim(0, 1.3)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"analysis_02_best_worst_{window}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Section C done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION D — SASRec Lift over Markov (All Windows)
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section D — SASRec Lift over Markov by Diagnosis Code"))

for window in WINDOWS:
    s_sas = df[(df["time_bucket"] == window) & (df["model"] == "SASRec")].set_index("trigger_dx")
    s_mar = df[(df["time_bucket"] == window) & (df["model"] == "Markov")].set_index("trigger_dx")

    lift = s_sas[["trigger_volume", "ndcg_at_3"]].join(
        s_mar[["ndcg_at_3"]], rsuffix="_markov"
    ).dropna()
    lift["lift"] = lift["ndcg_at_3"] - lift["ndcg_at_3_markov"]
    lift = lift[lift["trigger_volume"] >= MIN_VOL]

    top_gain = lift.nlargest(15, "lift")
    top_loss = lift.nsmallest(15, "lift")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    for ax, data, title in zip(
        axes,
        [top_gain, top_loss],
        ["SASRec Gains Most over Markov", "Markov Outperforms SASRec"]
    ):
        colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in data["lift"][::-1]]
        bars = ax.barh(data.index[::-1], data["lift"][::-1],
                       color=colors, edgecolor="white")
        for bar, row in zip(bars, data.iloc[::-1].itertuples()):
            ax.text(bar.get_width() + 0.002 * np.sign(bar.get_width() + 1e-9),
                    bar.get_y() + bar.get_height() / 2,
                    f"n={int(row.trigger_volume):,}  SAS={row.ndcg_at_3:.3f}  MKV={row.ndcg_at_3_markov:.3f}",
                    va="center", fontsize=7.5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{title}\nNDCG@3 lift — {WLABELS[window]}, min {MIN_VOL} triggers",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("SASRec NDCG@3 minus Markov NDCG@3")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"analysis_02_lift_{window}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Section D done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION E — Heatmap: Diagnosis Code × Model (T0_30, Hit@3)
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section E — Heatmap: Top 15 Diagnosis Codes × Model"))

for window in WINDOWS:
    sub = df[(df["time_bucket"] == window) & (df["trigger_dx"].isin(top15_dx))].copy()
    for k in [3, 5]:
        pivot = sub.pivot_table(index="trigger_dx", columns="model",
                                values=f"hit_at_{k}", aggfunc="first")
        pivot = pivot.reindex(top15_dx)
        pivot = pivot[[m for m in MODELS if m in pivot.columns]]

        # Add volume column as annotation text
        vol_ref = (sub[sub["model"] == MODELS[0]]
                   .set_index("trigger_dx")["trigger_volume"]
                   .reindex(top15_dx))

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlGn",
                    linewidths=0.5, annot_kws={"size": 10},
                    vmin=0, vmax=1, cbar_kws={"label": f"Hit@{k}"})

        # Overlay trigger volume on y-axis labels
        ylabels = [f"{dx}  (n={int(vol_ref.get(dx, 0)):,})"
                   for dx in top15_dx]
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_title(f"Hit@{k} Heatmap — Top 15 Diagnosis Codes by Volume\n{WLABELS[window]}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(f"analysis_02_heatmap_k{k}_{window}.png", dpi=150, bbox_inches="tight")
        plt.show()

print(f"Section E done — {time.time()-t0:.1f}s")
print("NB_Analysis_02 complete")
