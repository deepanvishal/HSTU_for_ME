# ============================================================
# NB_Analysis_02_diagnosis.py
# Purpose : Model performance by diagnosis code
#           Focus on codes covering 80% of test transition volume
# Source  : A870800_gen_rec_analysis_perf_by_diag
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

WINDOWS  = ["T0_30", "T30_60", "T60_180"]
WLABELS  = {"T0_30": "0-30 Days", "T30_60": "30-60 Days", "T60_180": "60-180 Days"}
MODELS   = ["SASRec", "BERT4Rec", "Markov"]
MCOLORS  = {"SASRec": "#4C72B0", "BERT4Rec": "#DD8452", "Markov": "#55A868"}
MIN_VOL  = 20
PARETO   = 0.80

display(Markdown("""
# Model Performance Analysis — By Diagnosis Code
**Methodology:**
- Analysis scoped to diagnosis codes covering 80% of test transition volume
- Low-volume codes excluded from all charts — model wins/losses there are noise
- Hit@3 is the primary metric: did the model predict the right specialty in top 3?
- All 3 time windows shown; Markov predictions are window-agnostic
"""))


# ════════════════════════════════════════════════════════════
# SECTION A — Pareto Cutoff: Codes Covering 80% of Volume
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section A — Pareto: Diagnosis Codes Covering 80% of Test Volume"))

df = client.query(f"""
    SELECT model, time_bucket, trigger_dx, trigger_volume
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_1, ndcg_at_3, ndcg_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_by_diag`
    ORDER BY model, time_bucket, trigger_volume DESC
""").to_dataframe()

# Use T0_30 + one model as reference for volume ranking — same triggers
ref = (df[(df["time_bucket"] == "T0_30") & (df["model"] == MODELS[0])]
       .sort_values("trigger_volume", ascending=False)
       .copy())

ref["cum_vol"]   = ref["trigger_volume"].cumsum()
ref["cum_share"] = ref["cum_vol"] / ref["trigger_volume"].sum()
pareto_dx        = ref[ref["cum_share"] <= PARETO]["trigger_dx"].tolist()

# Edge case: always include the code that crosses 80%
if len(pareto_dx) < len(ref):
    pareto_dx.append(ref.iloc[len(pareto_dx)]["trigger_dx"])

total_dx    = ref["trigger_dx"].nunique()
pareto_vol  = ref[ref["trigger_dx"].isin(pareto_dx)]["trigger_volume"].sum()
total_vol   = ref["trigger_volume"].sum()

display(Markdown(f"""
| | Count | Share of Volume |
|---|---|---|
| Total diagnosis codes | {total_dx:,} | 100% |
| Codes in 80% set | {len(pareto_dx):,} | {pareto_vol/total_vol*100:.1f}% |
| Codes excluded | {total_dx - len(pareto_dx):,} | {(total_vol-pareto_vol)/total_vol*100:.1f}% |
"""))

# Pareto curve
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(len(ref)), ref["cum_share"] * 100,
        color="#4C72B0", linewidth=2)
ax.axhline(80, color="#C44E52", linewidth=1, linestyle="--")
ax.axvline(len(pareto_dx), color="#C44E52", linewidth=1, linestyle="--")
ax.fill_between(range(len(pareto_dx)),
                ref["cum_share"].iloc[:len(pareto_dx)] * 100,
                alpha=0.15, color="#4C72B0")
ax.text(len(pareto_dx) + 1, 40,
        f"{len(pareto_dx)} codes\n= {pareto_vol/total_vol*100:.0f}% of volume",
        fontsize=9, color="#C44E52")
ax.set_xlabel("Diagnosis Codes (ranked by volume)")
ax.set_ylabel("Cumulative % of Transitions")
ax.set_title("Pareto Curve — Transition Volume by Diagnosis Code",
             fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("analysis_02_pareto.png", dpi=150, bbox_inches="tight")
plt.show()

# Filter working set
df_p = df[df["trigger_dx"].isin(pareto_dx)].copy()
# Preserve volume sort order
vol_order = ref[ref["trigger_dx"].isin(pareto_dx)]["trigger_dx"].tolist()
df_p["dx_order"] = df_p["trigger_dx"].map({dx: i for i, dx in enumerate(vol_order)})
df_p = df_p.sort_values("dx_order")

print(f"Working set: {len(pareto_dx)} codes, {pareto_vol:,} transitions")
print(f"Section A done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION B — Hit@3 per Model on 80% Codes (All Windows)
# Bar height = Hit@3, x-axis ordered by volume descending
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section B — Hit@3 on High-Volume Codes (80% Pareto Set)"))

display(Markdown("""
Bars show **Hit@3** (did the model predict the right specialty in top 3?).
X-axis ordered by trigger volume — leftmost codes have the most transitions.
Volume annotated on x-axis so scale of each code is visible.
"""))

for window in WINDOWS:
    sub = df_p[df_p["time_bucket"] == window]
    models_here = [m for m in MODELS if m in sub["model"].unique()]

    # Limit display to top 30 if pareto set is large — still 80% coverage
    dx_display = vol_order[:30] if len(vol_order) > 30 else vol_order
    sub = sub[sub["trigger_dx"].isin(dx_display)]

    fig, ax = plt.subplots(figsize=(max(16, len(dx_display) * 0.6), 6))
    x = np.arange(len(dx_display))
    w = 0.8 / len(models_here)

    for i, model in enumerate(models_here):
        s = (sub[sub["model"] == model]
             .set_index("trigger_dx")
             .reindex(dx_display))
        bars = ax.bar(x + i * w - 0.4 + w / 2,
                      s["hit_at_3"].fillna(0), w,
                      label=model, color=MCOLORS.get(model, "#999"),
                      edgecolor="white", alpha=0.85)

    # Volume on x-axis as secondary label — not cluttering bar tops
    vol_labels = [f"{dx}\nn={int(ref[ref['trigger_dx']==dx]['trigger_volume'].values[0]):,}"
                  if dx in ref["trigger_dx"].values else dx
                  for dx in dx_display]
    ax.set_xticks(x)
    ax.set_xticklabels(vol_labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_title(f"Hit@3 by Diagnosis Code — {WLABELS[window]}\n"
                 f"Ordered by volume (left = highest). 80% pareto set shown.",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Hit@3")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Reference line — mean Hit@3 across all codes for each model
    for model in models_here:
        mean_h3 = sub[sub["model"] == model]["hit_at_3"].mean()
        ax.axhline(mean_h3, color=MCOLORS.get(model, "#999"),
                   linewidth=1, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"analysis_02_hit3_{window}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Section B done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION C — Scatter: Volume vs Accuracy
# Shows whether data density correlates with prediction quality
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section C — Volume vs Accuracy: Does More Data Help?"))

display(Markdown("""
Each point = one diagnosis code. Color = in 80% pareto set (blue) or not (grey).
Dotted lines = median split. Upper-right = high volume + high accuracy.
"""))

for window in WINDOWS:
    full_sub = df[df["time_bucket"] == window].copy()
    full_sub["in_pareto"] = full_sub["trigger_dx"].isin(pareto_dx)

    models_here = [m for m in MODELS if m in full_sub["model"].unique()]
    fig, axes = plt.subplots(1, len(models_here),
                              figsize=(7 * len(models_here), 5),
                              sharey=True)
    if len(models_here) == 1:
        axes = [axes]

    for ax, model in zip(axes, models_here):
        s = full_sub[full_sub["model"] == model]
        # Grey for out-of-pareto, colored for in-pareto
        ax.scatter(s[~s["in_pareto"]]["trigger_volume"],
                   s[~s["in_pareto"]]["hit_at_3"],
                   c="#CCCCCC", s=30, alpha=0.5, label="Outside 80% set")
        ax.scatter(s[s["in_pareto"]]["trigger_volume"],
                   s[s["in_pareto"]]["hit_at_3"],
                   c=MCOLORS.get(model, "#999"), s=60, alpha=0.8,
                   label="80% pareto set", edgecolors="white", linewidth=0.5)

        # Median lines
        med_vol = s["trigger_volume"].median()
        med_h3  = s["hit_at_3"].median()
        ax.axvline(med_vol, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.axhline(med_h3,  color="black", linewidth=0.8, linestyle="--", alpha=0.4)

        # Label top 5 highest-volume codes
        top5 = s.nlargest(5, "trigger_volume")
        for _, row in top5.iterrows():
            ax.annotate(row["trigger_dx"],
                        (row["trigger_volume"], row["hit_at_3"]),
                        textcoords="offset points", xytext=(4, 3),
                        fontsize=7, color="#333333")

        ax.set_title(f"{model}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Trigger Volume")
        ax.set_ylabel("Hit@3" if model == models_here[0] else "")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(linestyle="--", alpha=0.3)

    fig.suptitle(f"Trigger Volume vs Hit@3 — {WLABELS[window]}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"analysis_02_scatter_{window}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Section C done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION D — NDCG@3 Heatmap: 80% Codes × Model
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section D — NDCG@3 Heatmap: High-Volume Codes × Model"))

display(Markdown("""
Cells show NDCG@3. Y-axis ordered by volume (top = highest volume).
Immediately shows who wins where it matters.
"""))

for window in WINDOWS:
    sub = df_p[df_p["time_bucket"] == window]
    dx_display = vol_order[:25] if len(vol_order) > 25 else vol_order

    pivot = (sub[sub["trigger_dx"].isin(dx_display)]
             .pivot_table(index="trigger_dx", columns="model",
                          values="ndcg_at_3", aggfunc="first")
             .reindex(dx_display)
             .astype(float).fillna(0))
    pivot = pivot[[m for m in MODELS if m in pivot.columns]]

    # Y-axis: code + volume
    vol_map = ref.set_index("trigger_dx")["trigger_volume"]
    ylabels = [f"{dx}  (n={int(vol_map.get(dx, 0)):,})" for dx in dx_display]

    fig, ax = plt.subplots(figsize=(10, max(6, len(dx_display) * 0.42)))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlGn",
                linewidths=0.5, annot_kws={"size": 9},
                vmin=0, vmax=1, cbar_kws={"label": "NDCG@3"})
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title(f"NDCG@3 — 80% Pareto Codes × Model\n{WLABELS[window]}  |  ordered by volume (top = highest)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"analysis_02_heatmap_{window}.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"Section D done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION E — Biggest Wins and Losses Within 80% Set
# Where does SASRec beat Markov on codes that actually matter
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section E — Model Lift Within 80% Pareto Set (All Windows)"))

display(Markdown("""
SASRec NDCG@3 minus Markov NDCG@3 — restricted to 80% pareto set.
A lift of +0.05 on a 1,000-transition code matters far more than +0.20 on a 10-transition code.
"""))

for window in WINDOWS:
    s_sas = (df_p[(df_p["time_bucket"] == window) & (df_p["model"] == "SASRec")]
             .set_index("trigger_dx"))
    s_mar = (df_p[(df_p["time_bucket"] == window) & (df_p["model"] == "Markov")]
             .set_index("trigger_dx"))

    if s_sas.empty or s_mar.empty:
        print(f"Skipping {window} — missing model data")
        continue

    lift = s_sas[["trigger_volume", "ndcg_at_3"]].join(
        s_mar[["ndcg_at_3"]], rsuffix="_markov"
    ).dropna()
    lift["lift"]           = lift["ndcg_at_3"] - lift["ndcg_at_3_markov"]
    lift["volume_x_lift"]  = lift["trigger_volume"] * lift["lift"].abs()
    lift = lift.sort_values("lift", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, data, title in zip(
        axes,
        [lift.head(15), lift.tail(15).iloc[::-1]],
        ["SASRec Gains Most (vs Markov)", "Markov Outperforms SASRec"]
    ):
        colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in data["lift"]]
        bars = ax.barh(data.index, data["lift"], color=colors, edgecolor="white")
        for bar, row in zip(bars, data.itertuples()):
            label = (f"n={int(row.trigger_volume):,}"
                     f"  SAS={row.ndcg_at_3:.3f}"
                     f"  MKV={row.ndcg_at_3_markov:.3f}")
            xpos = bar.get_width() + 0.003 if bar.get_width() >= 0 else bar.get_width() - 0.003
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=7.5,
                    ha="left" if bar.get_width() >= 0 else "right")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{title}\nNDCG@3 lift — {WLABELS[window]}, 80% pareto set",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("SASRec NDCG@3 minus Markov NDCG@3")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"analysis_02_lift_{window}.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Summary stat
    sas_wins = (lift["lift"] > 0).sum()
    mar_wins = (lift["lift"] < 0).sum()
    ties     = (lift["lift"] == 0).sum()
    wtd_lift = (lift["lift"] * lift["trigger_volume"]).sum() / lift["trigger_volume"].sum()
    display(Markdown(
        f"**{WLABELS[window]}** — SASRec wins: {sas_wins} codes | "
        f"Markov wins: {mar_wins} codes | Ties: {ties} | "
        f"Volume-weighted lift: {wtd_lift:+.4f}"
    ))

print(f"Section E done — {time.time()-t0:.1f}s")
print("NB_Analysis_02 complete")
