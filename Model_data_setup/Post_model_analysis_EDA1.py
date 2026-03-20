# ============================================================
# NB_Analysis_01_overall.py
# Purpose : Overall model performance comparison
# Source  : A870800_gen_rec_analysis_perf_overall
# ============================================================
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
K_VALUES = [1, 3, 5]

display(Markdown("""
# Model Performance Analysis — Overall
**Caveats:**
- Models predict top 5 specialties per trigger per time window
- True label set may contain more than 5 specialties
- Hit@K is binary per trigger — 1 if any top-K prediction matches true label set
- Markov predictions are window-agnostic; same top-5 applied across all time windows
- Probability cutoff analysis deferred
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

audit = (df[df["member_segment"] == "ALL"]
         .groupby(["model", "time_bucket"])["n_triggers"]
         .first()
         .unstack("time_bucket")
         .reindex(MODELS)
         .reindex(columns=WINDOWS))
display(Markdown("### Trigger counts by model and time window (ALL segments)"))
display(audit)

for m in MODELS:
    if m not in df["model"].unique():
        print(f"WARNING: {m} not found — check scoring ran successfully")

print(f"Section A done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION B — Summary Table: All Models, All Windows, K=1/3/5
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section B — Summary Table: All Models, All Windows, All K"))

base = df[df["member_segment"] == "ALL"].copy()
base["Window"] = base["time_bucket"].map(WLABELS)

for window in WINDOWS:
    sub = base[base["time_bucket"] == window][
        ["model", "n_triggers",
         "hit_at_1", "hit_at_3", "hit_at_5",
         "ndcg_at_1", "ndcg_at_3", "ndcg_at_5"]
    ].rename(columns={
        "model": "Model", "n_triggers": "Triggers",
        "hit_at_1": "Hit@1", "hit_at_3": "Hit@3", "hit_at_5": "Hit@5",
        "ndcg_at_1": "NDCG@1", "ndcg_at_3": "NDCG@3", "ndcg_at_5": "NDCG@5",
    }).set_index("Model")
    display(Markdown(f"**{WLABELS[window]}**"))
    display(sub)

print(f"Section B done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION C — Mini Dashboard: All Models × All Windows × K=3 and K=5
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section C — Mini Dashboard: All Models, All Windows"))

base_all = df[df["member_segment"] == "ALL"].copy()
base_all["time_bucket"] = pd.Categorical(base_all["time_bucket"], WINDOWS, ordered=True)
base_all = base_all.sort_values("time_bucket")

metrics_dashboard = [
    ("hit_at_3",  "ndcg_at_3",  "K=3"),
    ("hit_at_5",  "ndcg_at_5",  "K=5"),
]

for hit_col, ndcg_col, k_label in metrics_dashboard:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, metric, mlabel in zip(axes,
                                   [hit_col, ndcg_col],
                                   [f"Hit@{k_label[-1]}", f"NDCG@{k_label[-1]}"]):
        x = np.arange(len(WINDOWS))
        w = 0.25
        models_here = [m for m in MODELS if m in base_all["model"].unique()]
        for i, model in enumerate(models_here):
            sub = (base_all[base_all["model"] == model]
                   .set_index("time_bucket")
                   .reindex(WINDOWS))
            vals   = sub[metric].fillna(0).values
            ntrig  = sub["n_triggers"].fillna(0).values
            bars   = ax.bar(x + i * w, vals, w,
                            label=model, color=MCOLORS.get(model, "#999"),
                            edgecolor="white")
            for bar, v, n in zip(bars, vals, ntrig):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold")
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                        f"n={int(n):,}", ha="center", va="center",
                        fontsize=7, color="white", rotation=90)

        ax.set_xticks(x + w)
        ax.set_xticklabels([WLABELS[w_] for w_ in WINDOWS], fontsize=10)
        ax.set_title(f"{mlabel} by Time Window — {k_label}, All Members",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel(mlabel)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"Model Comparison Dashboard — {k_label}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"analysis_01_dashboard_{k_label.replace('=','')}.png",
                dpi=150, bbox_inches="tight")
    plt.show()

print(f"Section C done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION D — Heatmap Table: Model × Window × Metric
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section D — Heatmap Table: Model × Window × K"))

for k in K_VALUES:
    hit_col  = f"hit_at_{k}"
    ndcg_col = f"ndcg_at_{k}"

    pivot_hit  = (base_all[base_all["member_segment"] == "ALL"]
                  .pivot_table(index="model", columns="time_bucket",
                               values=hit_col, aggfunc="first")
                  .reindex(MODELS)
                  .reindex(columns=WINDOWS))
    pivot_ndcg = (base_all[base_all["member_segment"] == "ALL"]
                  .pivot_table(index="model", columns="time_bucket",
                               values=ndcg_col, aggfunc="first")
                  .reindex(MODELS)
                  .reindex(columns=WINDOWS))

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    for ax, pivot, label in zip(axes,
                                  [pivot_hit, pivot_ndcg],
                                  [f"Hit@{k}", f"NDCG@{k}"]):
        pivot.columns = [WLABELS[c] for c in pivot.columns]
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlGn",
                    linewidths=0.5, annot_kws={"size": 12},
                    vmin=0, vmax=1,
                    cbar_kws={"label": label})
        ax.set_title(f"{label} — All Members", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(f"Performance Heatmap — K={k}", fontsize=12,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"analysis_01_heatmap_k{k}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Section D done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION E — Performance by Member Segment (All Windows)
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section E — Performance by Member Segment"))

seg = df[df["member_segment"] != "ALL"].copy()

if seg.empty:
    print("No segment-level data available")
else:
    segments = sorted(seg["member_segment"].unique())
    models_here = [m for m in MODELS if m in seg["model"].unique()]

    for window in WINDOWS:
        sub = seg[seg["time_bucket"] == window]
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        for ax, metric, label in zip(axes,
                                      ["hit_at_3", "ndcg_at_3"],
                                      ["Hit@3", "NDCG@3"]):
            x = np.arange(len(segments))
            w = 0.25
            for i, model in enumerate(models_here):
                vals = [sub[(sub["member_segment"] == s) &
                            (sub["model"] == model)][metric].values
                        for s in segments]
                ntrig = [sub[(sub["member_segment"] == s) &
                             (sub["model"] == model)]["n_triggers"].values
                         for s in segments]
                vals  = [v[0] if len(v) > 0 else 0 for v in vals]
                ntrig = [n[0] if len(n) > 0 else 0 for n in ntrig]
                bars  = ax.bar(x + i * w, vals, w,
                               label=model, color=MCOLORS.get(model, "#999"),
                               edgecolor="white")
                for bar, v, n in zip(bars, vals, ntrig):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.008,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
                    if n > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() / 2,
                                f"n={int(n):,}", ha="center", va="center",
                                fontsize=7, color="white", rotation=90)
            ax.set_xticks(x + w)
            ax.set_xticklabels(segments, fontsize=10, rotation=15, ha="right")
            ax.set_title(f"{label} by Member Segment — {WLABELS[window]}",
                         fontsize=11, fontweight="bold")
            ax.set_ylabel(label)
            ax.set_ylim(0, 1.1)
            ax.legend(fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"analysis_01_segment_{window}.png", dpi=150, bbox_inches="tight")
        plt.show()

print(f"Section E done — {time.time()-t0:.1f}s")
print("NB_Analysis_01 complete")
