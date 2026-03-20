# ============================================================
# NB_Analysis_04_recommendations.py
# Purpose : Data-driven recommendation for fine-tuning target
#           Identify high-cost / high-volume specialties where
#           model accuracy is low — highest ROI to improve
# Sources : A870800_gen_rec_f_spend_summary
#           A870800_gen_rec_analysis_perf_by_ending_specialty
# ============================================================
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from google.cloud import bigquery
from IPython.display import display, Markdown

DS     = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
client = bigquery.Client(project="anbc-hcb-dev")

MODELS  = ["SASRec", "BERT4Rec", "Markov"]
MCOLORS = {"SASRec": "#4C72B0", "BERT4Rec": "#DD8452", "Markov": "#55A868"}

display(Markdown("""
# Recommendation Analysis — Fine-Tuning Target Identification

**Framework:**
High-cost specialties where the model currently predicts poorly
are the highest ROI candidates for fine-tuning or focused modeling.

**Quadrant logic:**
- High cost + High accuracy → Deploy now, monitor
- High cost + Low accuracy → Fine-tuning target (priority)
- Low cost + High accuracy → Nice to have
- Low cost + Low accuracy → Deprioritize

**Recommendation options presented:**
1. Fine-tune existing model on high-cost specialty subset
2. Binary classifier — predict Yes/No for one target specialty
3. Provider-level prediction for one specialty (higher complexity, higher value)
"""))


# ════════════════════════════════════════════════════════════
# SECTION A — Load Spend Data
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section A — Load Spend by Specialty"))

spend_df = client.query(f"""
    SELECT
        specialty_ctg_cd                                 AS ending_specialty
        ,SUM(allowed_amt)                                AS total_allowed_amt
        ,COUNT(DISTINCT member_id)                       AS member_count
        ,COUNT(*)                                        AS visit_count
        ,ROUND(SUM(allowed_amt) / NULLIF(COUNT(*), 0), 2) AS avg_cost_per_visit
    FROM `{DS}.A870800_gen_rec_f_spend_summary`
    WHERE specialty_ctg_cd IS NOT NULL
    GROUP BY specialty_ctg_cd
    ORDER BY total_allowed_amt DESC
""").to_dataframe()

print(f"Loaded {len(spend_df):,} specialties from spend summary")
print(f"Total allowed amount : ${spend_df['total_allowed_amt'].sum():,.0f}")
display(spend_df.head(10))
print(f"Section A done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION B — Load Accuracy Data
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section B — Load Prediction Accuracy by Specialty"))

acc_df = client.query(f"""
    SELECT
        model
        ,ending_specialty
        ,total_appearances
        ,hit_rate_at_3
        ,avg_ndcg_at_3
    FROM `{DS}.A870800_gen_rec_analysis_perf_by_ending_specialty`
    WHERE time_bucket = 'T0_30'
      AND total_appearances >= 20
    ORDER BY model, total_appearances DESC
""").to_dataframe()

print(f"Loaded {len(acc_df):,} rows")
print(f"Section B done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION C — Join Spend + Accuracy, Build Priority Matrix
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section C — Priority Matrix: Cost vs Prediction Accuracy"))

merged = acc_df.merge(spend_df, on="ending_specialty", how="inner")
merged["log_spend"] = np.log1p(merged["total_allowed_amt"])

# Quadrant thresholds — median splits
acc_median   = merged["hit_rate_at_3"].median()
spend_median = merged["total_allowed_amt"].median()

def quadrant(row):
    hi_acc   = row["hit_rate_at_3"]    >= acc_median
    hi_spend = row["total_allowed_amt"] >= spend_median
    if hi_spend and not hi_acc:
        return "Fine-Tuning Target"
    elif hi_spend and hi_acc:
        return "Deploy Now"
    elif not hi_spend and hi_acc:
        return "Low Priority"
    else:
        return "Deprioritize"

merged["quadrant"] = merged.apply(quadrant, axis=1)

QCOLORS = {
    "Fine-Tuning Target": "#C44E52",
    "Deploy Now":         "#55A868",
    "Low Priority":       "#8172B2",
    "Deprioritize":       "#CCCCCC",
}

# One scatter per model
models_present = [m for m in MODELS if m in merged["model"].unique()]
fig, axes = plt.subplots(1, len(models_present),
                          figsize=(8 * len(models_present), 7),
                          sharey=False)
if len(models_present) == 1:
    axes = [axes]

for ax, model in zip(axes, models_present):
    sub = merged[merged["model"] == model].copy()

    for q, color in QCOLORS.items():
        s = sub[sub["quadrant"] == q]
        ax.scatter(s["hit_rate_at_3"],
                   s["total_allowed_amt"] / 1e6,
                   s=s["total_appearances"].clip(upper=500) * 2,
                   c=color, alpha=0.7, edgecolors="white", linewidth=0.5,
                   label=q)

    # Quadrant lines
    ax.axvline(acc_median,   color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(spend_median / 1e6, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Label top 10 fine-tuning targets
    targets = sub[sub["quadrant"] == "Fine-Tuning Target"].nlargest(10, "total_allowed_amt")
    for _, row in targets.iterrows():
        ax.annotate(row["ending_specialty"],
                    (row["hit_rate_at_3"], row["total_allowed_amt"] / 1e6),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7, color="#C44E52", fontweight="bold")

    ax.set_title(f"{model} — Cost vs Prediction Accuracy\nBubble size = trigger volume",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Hit Rate at K=3 (T0 to 30 Days)")
    ax.set_ylabel("Total Allowed Amount ($M)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle="--", alpha=0.3)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(acc_median - 0.02, ylim[1] * 0.95, "Fine-Tuning\nTarget",
            ha="right", va="top", fontsize=8, color="#C44E52", style="italic")
    ax.text(acc_median + 0.02, ylim[1] * 0.95, "Deploy\nNow",
            ha="left", va="top", fontsize=8, color="#55A868", style="italic")

plt.tight_layout()
plt.savefig("analysis_04_cost_vs_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Section C done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION D — Top Fine-Tuning Targets Table
# ════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section D — Top Fine-Tuning Targets (SASRec, T0 to 30 Days)"))

display(Markdown("""
**Interpretation:** Specialties below median Hit@3 but above median spend.
These are the highest-ROI candidates — high network cost, model currently misses them.
"""))

targets = merged[(merged["model"] == "SASRec") &
                 (merged["quadrant"] == "Fine-Tuning Target")].copy()
targets = targets.sort_values("total_allowed_amt", ascending=False).head(15)
targets["total_allowed_amt_M"] = (targets["total_allowed_amt"] / 1e6).round(2)

display_cols = {
    "ending_specialty":    "Specialty",
    "total_allowed_amt_M": "Total Cost ($M)",
    "visit_count":         "Visit Count",
    "avg_cost_per_visit":  "Avg Cost Per Visit ($)",
    "total_appearances":   "Trigger Appearances",
    "hit_rate_at_3":       "Hit Rate at K=3",
    "avg_ndcg_at_3":       "NDCG@3",
}
out = targets[list(display_cols.keys())].rename(columns=display_cols)
out = out.reset_index(drop=True)
display(out)

# Bar chart — top 10 targets ranked by cost
fig, ax = plt.subplots(figsize=(14, 6))
top10 = targets.head(10)
bars = ax.barh(top10["ending_specialty"][::-1],
               top10["total_allowed_amt_M"][::-1],
               color="#C44E52", edgecolor="white")
for bar, acc in zip(bars, top10["hit_rate_at_3"][::-1]):
    ax.text(bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"Hit@3={acc:.2f}", va="center", fontsize=9)
ax.set_title("Top 10 Fine-Tuning Targets — High Cost, Low Prediction Accuracy\nSASRec, T0 to 30 Days",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Total Allowed Amount ($M)")
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("analysis_04_top_targets.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Section D done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION E — Recommendation Summary
# ════════════════════════════════════════════════════════════
display(Markdown("---\n## Section E — Recommendation Summary"))

# Pull top target for dynamic text
top_target = targets.iloc[0]["ending_specialty"] if len(targets) > 0 else "<specialty>"
top_cost_m = targets.iloc[0]["total_allowed_amt_M"] if len(targets) > 0 else 0
top_acc    = targets.iloc[0]["hit_rate_at_3"] if len(targets) > 0 else 0

display(Markdown(f"""
### Three Paths Forward

**Path 1 — Fine-tune existing model on high-cost specialty subset**
- Focus training data on triggers where true label includes a top-cost specialty
- Expected gain: few % improvement in Hit@3 for target specialties
- Complexity: Low — same architecture, reweighted training signal
- Justification: {top_target} has ${top_cost_m:.1f}M in allowed spend, current Hit@3 = {top_acc:.2f}

**Path 2 — Binary classifier: predict Yes/No for one target specialty**
- Single specialty, single model, fully interpretable
- Works well when specialty has high enough trigger volume (see table above)
- Complexity: Low — logistic/gradient boosted, easy to explain to network team
- Limitation: Does not generalize across specialties

**Path 3 — Provider-level prediction within one specialty**
- Predict which specific provider a member will visit next
- Highest business value (network optimization, outreach targeting)
- Complexity: High — sparse labels, provider turnover, cold-start problem
- Recommended only after Path 1 or 2 validates the specialty signal

### Recommended Sequence
1. Run this analysis with full population (not 5% sample) to confirm cost rankings
2. Select top 1-2 specialties from fine-tuning targets table
3. Implement Path 1 first — measurable lift in 1-2 weeks
4. Present Path 3 as Phase 2 roadmap item to network team
"""))

print("NB_Analysis_04 complete")
