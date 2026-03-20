# ============================================================
# NB_Analysis_03_ending_specialty.py
# Purpose : Model performance by ending specialty
# Source  : A870800_gen_rec_analysis_perf_by_ending_specialty
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
MIN_APP  = 20

display(Markdown("""
# Model Performance Analysis — By Ending Specialty
**Caveats:**
- total_appearances = how many triggers had this specialty in their true label set
- predicted_at_K = how many times the model correctly included it in top-K predictions
- One trigger contributes to multiple specialties (multi-label true label sets)
- Specialties with fewer than 20 appearances excluded from accuracy charts
- Markov predictions are window-agnostic; same top-5 across all time windows
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
        ,hit_rate_at_5, avg_ndcg_at_3
    FROM `{DS}.A870800_gen_rec_analysis_perf_by_ending_specialty`
    ORDER BY model, time_bucket, total_appearances DESC
""").to_dataframe()

df["hit_rate_at_1"] = (df["predicted_at_1"] / df["total_appearances"].clip(lower=1)).round(4)
df["hit_rate_at_3"] = (df["predicted_at_3"] / df["total_appearances"].clip(lower=1)).round(4)

print(f"Loaded {len(df):,} rows")
print(f"Unique ending specialties : {df['ending_specialty'].nunique():,}")
print(f"Models                    : {sorted(df['model'].unique())}")

# Audit — total appearances per window per model
audit = (df.groupby(["model", "time_bucket"])["total_appearances"]
         .sum().unstack("time_bucket")
         .reindex(MODELS).reindex(columns=WINDOWS))
display(Markdown("### Total specialty appearances in true label sets by model and window"))
display(audit)
print(f"Section A done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION B — Top 15 Ending Specialties by Volume (All Windows)
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section B — Top 15 Ending Specialties by Volume"))

ref = df[(df["time_bucket"] == "T0_30") & (df["model"] == MODELS[0])]
top15_sp = ref.nlargest(15, "total_appearances")["ending_specialty"].tolist()

for window in WINDOWS:
    sub = df[(df["time_bucket"] == window) & (df["ending_specialty"].isin(top15_sp))]
    models_here = [m for m in MODELS if m in sub["model"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Chart 1 — appearances with predicted_at_3 label
    ax = axes[0]
    x = np.arange(len(top15_sp))
    w = 0.25
    for i, model in enumerate(models_here):
        s = sub[sub["model"] == model].set_index("ending_specialty").reindex(top15_sp)
        bars = ax.bar(x + i * w, s["total_appearances"].fillna(0), w,
                      label=model, color=MCOLORS.get(model, "#999"),
                      edgecolor="white")
        for bar, p3, hr3 in zip(bars,
                                 s["predicted_at_3"].fillna(0),
                                 s["hit_rate_at_3"].fillna(0)):
            h = bar.get_height()
            if h > 0:
                # Count of correct predictions inside bar
                ax.text(bar.get_x() + bar.get_width() / 2, h / 2,
                        f"{int(p3):,}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold", rotation=90)
                # Hit rate above bar
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                        f"{hr3:.2f}", ha="center", va="bottom", fontsize=6.5,
                        color=MCOLORS.get(model, "#999"))
    ax.set_xticks(x + w)
    ax.set_xticklabels(top15_sp, rotation=40, ha="right", fontsize=8)
    ax.set_title(f"Total Appearances — {WLABELS[window]}\n"
                 f"(inside bar = predicted correctly at K=3, above bar = Hit Rate@3)",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Appearances in True Label Set")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Chart 2 — hit rate at K=1, 3, 5 for each specialty (SASRec only, cleaner)
    ax = axes[1]
    for k, ls, marker in zip([1, 3, 5], ["-", "--", ":"], ["o", "s", "^"]):
        s = sub[sub["model"] == "SASRec"].set_index("ending_specialty").reindex(top15_sp)
        ax.plot(range(len(top15_sp)), s[f"hit_rate_at_{k}"].fillna(0),
                marker=marker, linewidth=2, markersize=5, linestyle=ls,
                label=f"Hit@{k}", color="#4C72B0", alpha=0.5 + k * 0.1)

    # Annotate total appearances
    ref_sub = sub[sub["model"] == MODELS[0]].set_index("ending_specialty").reindex(top15_sp)
    for j, n in enumerate(ref_sub["total_appearances"].fillna(0)):
        ax.text(j, -0.09, f"n={int(n):,}", ha="center", va="top",
                fontsize=6.5, color="#666666", rotation=45)

    ax.set_xticks(range(len(top15_sp)))
    ax.set_xticklabels(top15_sp, rotation=40, ha="right", fontsize=8)
    ax.set_title(f"SASRec — Hit Rate at K=1/3/5 — {WLABELS[window]}\n"
                 f"(n = total appearances below axis)",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Hit Rate")
    ax.set_ylim(-0.15, 1.05)
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"analysis_03_top15_volume_{window}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Section B done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION C — Top 15 by Prediction Accuracy (All Windows)
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section C — Top 15 Specialties by Prediction Accuracy (min 20 appearances)"))

for window in WINDOWS:
    filt = df[(df["time_bucket"] == window) & (df["total_appearances"] >= MIN_APP)]
    models_here = [m for m in MODELS if m in filt["model"].unique()]

    fig, axes = plt.subplots(1, len(models_here), figsize=(7 * len(models_here), 7))
    if len(models_here) == 1:
        axes = [axes]

    for ax, model in zip(axes, models_here):
        sub = filt[filt["model"] == model].nlargest(15, "hit_rate_at_3")
        rev = sub.iloc[::-1]
        bars = ax.barh(rev["ending_specialty"], rev["hit_rate_at_3"],
                       color=MCOLORS.get(model, "#999"), edgecolor="white")
        for bar, row in zip(bars, rev.itertuples()):
            ax.text(bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"n={int(row.total_appearances):,}  "
                    f"@1={row.hit_rate_at_1:.2f}  "
                    f"@5={row.hit_rate_at_5:.2f}  "
                    f"NDCG={row.avg_ndcg_at_3:.3f}",
                    va="center", fontsize=7.5)
        ax.set_title(f"{model}\nTop 15 by Hit@3 — {WLABELS[window]}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Hit Rate at K=3")
        ax.set_xlim(0, 1.5)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.suptitle(f"Top 15 Specialties by Prediction Accuracy — {WLABELS[window]}, min {MIN_APP} appearances",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"analysis_03_accuracy_{window}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Section C done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION D — Heatmap: Top 15 Specialties × Model, K=3 and K=5
# All Windows
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section D — Heatmap: Model vs Specialty (K=3 and K=5)"))

for window in WINDOWS:
    sub = df[(df["time_bucket"] == window) &
             (df["ending_specialty"].isin(top15_sp))].copy()

    for k, rate_col in [(3, "hit_rate_at_3"), (5, "hit_rate_at_5")]:
        pivot = sub.pivot_table(index="ending_specialty", columns="model",
                                values=rate_col, aggfunc="first")
        pivot = pivot.reindex(top15_sp)
        pivot = pivot[[m for m in MODELS if m in pivot.columns]]

        # Y-axis label = specialty + appearances
        vol_ref = (sub[sub["model"] == MODELS[0]]
                   .set_index("ending_specialty")["total_appearances"]
                   .reindex(top15_sp))

        # Annotation: hit rate + raw count — numpy string array avoids dtype object error
        pred_col  = f"predicted_at_{k}"
        n_rows    = len(top15_sp)
        n_cols    = len(pivot.columns)
        annot_arr = np.full((n_rows, n_cols), "—", dtype=object)
        for j, model in enumerate(pivot.columns):
            s = sub[sub["model"] == model].set_index("ending_specialty").reindex(top15_sp)
            for i, sp in enumerate(top15_sp):
                rate = s[rate_col].iloc[i] if sp in s.index else np.nan
                pred = s[pred_col].iloc[i] if sp in s.index else np.nan
                if pd.notna(rate) and pd.notna(pred):
                    annot_arr[i, j] = f"{float(rate):.2f}\n({int(pred):,})"

        # fillna(0) — NaN in pivot breaks matplotlib imshow with dtype object error
        pivot_float = pivot.astype(float).fillna(0)

        # Height scales with row count — prevents y label overlap
        fig_h = max(8, len(top15_sp) * 0.55)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        sns.heatmap(pivot_float, ax=ax,
                    annot=annot_arr, fmt="", cmap="YlGn",
                    linewidths=0.5, annot_kws={"size": 9},
                    vmin=0, vmax=1, cbar_kws={"label": f"Hit Rate @{k}"})

        ylabels = [f"{sp}  (n={int(vol_ref.get(sp, 0)):,})" for sp in top15_sp]
        # rotation=0 keeps labels horizontal; ha="right" aligns to cell edge
        ax.set_yticklabels(ylabels, fontsize=9, rotation=0, ha="right", va="center")
        ax.set_title(f"Hit Rate @{k} — Top 15 Specialties by Volume\n"
                     f"{WLABELS[window]}  |  cell = rate (correctly predicted count)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        # Extra left margin for long specialty names
        plt.subplots_adjust(left=0.35)
        plt.savefig(f"analysis_03_heatmap_k{k}_{window}.png", dpi=150, bbox_inches="tight")
        plt.show()

print(f"Section D done — {time.time()-t0:.1f}s")
print("NB_Analysis_03 complete")
