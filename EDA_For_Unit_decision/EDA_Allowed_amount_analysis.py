from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, Markdown

client = bigquery.Client(project="anbc-hcb-dev")
DATASET = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
TABLE = "A870800_gen_rec_f_dx_to_specialty_first_encounter"
THRESHOLD = 100
COHORTS = ["Adult_Female", "Adult_Male", "Senior", "Children"]

df = client.query(f"""
    SELECT * FROM `{DATASET}.{TABLE}`
    WHERE transition_count >= {THRESHOLD}
      AND conditional_entropy IS NOT NULL
""").to_dataframe()

for col in ["transition_count", "unique_members", "conditional_probability",
            "conditional_entropy", "total_allowed_amt",
            "avg_allowed_per_transition", "avg_allowed_per_member"]:
    df[col] = df[col].astype(float)

df["member_density"] = (df["unique_members"] / df["transition_count"]).round(3)


def fmt_millions(x):
    return f"${x/1_000_000:.2f}M"

def fmt_usd(x):
    return f"${x:,.0f}"

def fmt_count(x):
    return f"{x:,.0f}"

def style_table(df_display):
    return df_display.style.format({
        col: fmt_millions for col in df_display.columns if "Total Spend" in col
    } | {
        col: fmt_usd for col in df_display.columns
        if any(k in col for k in ["Avg Spend", "Avg Cost", "Per Member", "Per Visit"])
    } | {
        col: fmt_count for col in df_display.columns
        if any(k in col for k in ["Transitions", "Members", "Count"])
    })


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — PARETO ON TOTAL COST
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
# Analysis 1 — Pareto on Total Cost

Identifies the diagnosis to specialty transitions that account for 80% of total
allowed spend across all first encounter visits.

From that high-value set, surfaces the most impactful starting diagnoses and
destination specialties by both volume and spend.

The 80% threshold is a standard Pareto cutoff — a small number of transitions
typically drive the majority of cost. Understanding which transitions these are
directly informs care navigation prioritization and network adequacy planning.

Total spend figures are shown in millions USD. Per-member and per-visit figures
are shown in USD.

---
"""))

transition_agg = (
    df.groupby(["current_dx", "current_dx_desc", "next_specialty",
                "next_specialty_desc", "current_ccsr", "current_ccsr_desc"],
               as_index=False)
    .agg(transition_count=("transition_count", "sum"),
         unique_members=("unique_members", "sum"),
         total_allowed_amt=("total_allowed_amt", "sum"),
         avg_allowed_per_member=("avg_allowed_per_member", "mean"),
         conditional_entropy=("conditional_entropy", "mean"),
         conditional_probability=("conditional_probability", "mean"),
         member_density=("member_density", "mean"))
    .sort_values("total_allowed_amt", ascending=False)
)

total_spend = transition_agg["total_allowed_amt"].sum()
transition_agg["cumulative_spend"] = transition_agg["total_allowed_amt"].cumsum()
transition_agg["cumulative_pct"] = transition_agg["cumulative_spend"] / total_spend
pareto_df = transition_agg[transition_agg["cumulative_pct"] <= 0.80].copy()

display(Markdown(f"""
**Total transitions:** {len(transition_agg):,}
**Total allowed spend:** {fmt_millions(total_spend)}
**Transitions in 80% spend:** {len(pareto_df):,} ({len(pareto_df)/len(transition_agg)*100:.1f}% of all transitions)

**{len(pareto_df)/len(transition_agg)*100:.1f}% of transition pairs drive 80% of total spend.**
The remaining {100 - len(pareto_df)/len(transition_agg)*100:.1f}% are low-cost long-tail transitions.
All analyses below focus on this high-value set.
"""))

# ── PARETO CURVE ──────────────────────────────────────────────────────────────
display(Markdown("""
#### Pareto Curve — Cumulative Spend Concentration

Shows how quickly total allowed spend concentrates in the top transitions when
ranked by cost. The steeper the curve, the more concentrated the spend.
The red line marks the 80% threshold used to define the high-value transition set.
"""))

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(1, len(transition_agg) + 1)
ax.plot(x, transition_agg["cumulative_pct"] * 100, color="#4C9BE8", linewidth=2)
ax.axhline(80, color="red", linestyle="--", alpha=0.7, label="80% spend threshold")
ax.axvline(len(pareto_df), color="orange", linestyle="--", alpha=0.7,
           label=f"{len(pareto_df):,} transitions cover 80% of spend")
ax.fill_between(x[:len(pareto_df)],
                transition_agg["cumulative_pct"].values[:len(pareto_df)] * 100,
                alpha=0.1, color="#4C9BE8")
ax.set_xlabel("Number of Transitions (ranked by spend)", fontsize=10)
ax.set_ylabel("Cumulative Spend %", fontsize=10)
ax.set_title("Pareto Curve — Cumulative Spend Concentration", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
plt.tight_layout()
plt.savefig("pareto_curve.png", dpi=150, bbox_inches="tight")
plt.show()

# ── TOP DX ────────────────────────────────────────────────────────────────────
display(Markdown("""
### Most Impactful Starting Diagnoses — Volume and Spend

Each diagnosis is ranked by total allowed spend within the 80% pareto set.
Total spend shown in millions USD.
Member density shows what fraction of transitions are from unique members —
a density close to 1.0 means most transitions represent different members
(generalizable signal). A low density means the same members are generating
repeat transitions (concentrated signal).
"""))

dx_summary = (
    pareto_df.groupby(["current_dx", "current_dx_desc", "current_ccsr_desc"],
                      as_index=False)
    .agg(total_transitions=("transition_count", "sum"),
         unique_members=("unique_members", "sum"),
         total_allowed_amt=("total_allowed_amt", "sum"),
         avg_allowed_per_member=("avg_allowed_per_member", "mean"),
         avg_entropy=("conditional_entropy", "mean"),
         avg_member_density=("member_density", "mean"))
    .sort_values("total_allowed_amt", ascending=False)
    .head(20)
)

dx_display = dx_summary[[
    "current_dx_desc", "current_ccsr_desc", "total_transitions",
    "unique_members", "total_allowed_amt", "avg_allowed_per_member",
    "avg_entropy", "avg_member_density"
]].rename(columns={
    "current_dx_desc": "Diagnosis",
    "current_ccsr_desc": "Clinical Domain",
    "total_transitions": "Total Transitions",
    "unique_members": "Unique Members",
    "total_allowed_amt": "Total Spend (USD M)",
    "avg_allowed_per_member": "Avg Spend Per Member ($)",
    "avg_entropy": "Avg Entropy",
    "avg_member_density": "Member Density"
}).reset_index(drop=True)

display(dx_display.style.format({
    "Total Spend (USD M)": lambda x: fmt_millions(x),
    "Avg Spend Per Member ($)": lambda x: fmt_usd(x),
    "Total Transitions": lambda x: fmt_count(x),
    "Unique Members": lambda x: fmt_count(x),
    "Avg Entropy": "{:.4f}",
    "Member Density": "{:.3f}"
}))

fig, axes = plt.subplots(1, 2, figsize=(24, 10))
dx_vol = dx_summary.sort_values("total_transitions", ascending=True).tail(15)
axes[0].barh(dx_vol["current_dx_desc"], dx_vol["total_transitions"],
             color="#4C9BE8", alpha=0.85)
axes[0].set_xlabel("Total Transitions", fontsize=10)
axes[0].set_title("Top 15 Diagnoses by Volume\n(within 80% spend)", fontsize=12, fontweight="bold")
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.setp(axes[0].get_yticklabels(), fontsize=8)

dx_spend = dx_summary.sort_values("total_allowed_amt", ascending=True).tail(15)
axes[1].barh(dx_spend["current_dx_desc"], dx_spend["total_allowed_amt"] / 1_000_000,
             color="#F4845F", alpha=0.85)
axes[1].set_xlabel("Total Allowed Amount (USD M)", fontsize=10)
axes[1].set_title("Top 15 Diagnoses by Spend\n(within 80% spend)", fontsize=12, fontweight="bold")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))
plt.setp(axes[1].get_yticklabels(), fontsize=8)

fig.suptitle("Most Impactful Starting Diagnoses — Pareto Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("pareto_dx.png", dpi=150, bbox_inches="tight")
plt.show()

# ── TOP SPECIALTY ─────────────────────────────────────────────────────────────
display(Markdown("""
### Most Impactful Destination Specialties — Volume and Spend

Each specialty is ranked by total allowed spend within the 80% pareto set.
Total spend shown in millions USD.
"""))

spec_summary = (
    pareto_df.groupby(["next_specialty", "next_specialty_desc"], as_index=False)
    .agg(total_transitions=("transition_count", "sum"),
         unique_members=("unique_members", "sum"),
         total_allowed_amt=("total_allowed_amt", "sum"),
         avg_allowed_per_member=("avg_allowed_per_member", "mean"),
         avg_entropy=("conditional_entropy", "mean"),
         avg_member_density=("member_density", "mean"))
    .sort_values("total_allowed_amt", ascending=False)
    .head(20)
)

spec_display = spec_summary[[
    "next_specialty_desc", "total_transitions", "unique_members",
    "total_allowed_amt", "avg_allowed_per_member", "avg_entropy", "avg_member_density"
]].rename(columns={
    "next_specialty_desc": "Specialty",
    "total_transitions": "Total Transitions",
    "unique_members": "Unique Members",
    "total_allowed_amt": "Total Spend (USD M)",
    "avg_allowed_per_member": "Avg Spend Per Member ($)",
    "avg_entropy": "Avg Entropy",
    "avg_member_density": "Member Density"
}).reset_index(drop=True)

display(spec_display.style.format({
    "Total Spend (USD M)": lambda x: fmt_millions(x),
    "Avg Spend Per Member ($)": lambda x: fmt_usd(x),
    "Total Transitions": lambda x: fmt_count(x),
    "Unique Members": lambda x: fmt_count(x),
    "Avg Entropy": "{:.4f}",
    "Member Density": "{:.3f}"
}))

fig, axes = plt.subplots(1, 2, figsize=(24, 10))
spec_vol = spec_summary.sort_values("total_transitions", ascending=True).tail(15)
axes[0].barh(spec_vol["next_specialty_desc"], spec_vol["total_transitions"],
             color="#4C9BE8", alpha=0.85)
axes[0].set_xlabel("Total Transitions", fontsize=10)
axes[0].set_title("Top 15 Specialties by Volume\n(within 80% spend)", fontsize=12, fontweight="bold")
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.setp(axes[0].get_yticklabels(), fontsize=8)

spec_spend = spec_summary.sort_values("total_allowed_amt", ascending=True).tail(15)
axes[1].barh(spec_spend["next_specialty_desc"], spec_spend["total_allowed_amt"] / 1_000_000,
             color="#F4845F", alpha=0.85)
axes[1].set_xlabel("Total Allowed Amount (USD M)", fontsize=10)
axes[1].set_title("Top 15 Specialties by Spend\n(within 80% spend)", fontsize=12, fontweight="bold")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:.1f}M"))
plt.setp(axes[1].get_yticklabels(), fontsize=8)

fig.suptitle("Most Impactful Destination Specialties — Pareto Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("pareto_specialty.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ADDITION 1 — ENTROPY vs COST QUADRANT
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Addition 1 — Predictability vs Cost Quadrant

The most actionable transitions sit in the top right quadrant —
**high conditional probability and high cost per member**.

These pathways are both predictable and expensive — the ideal target for
care navigation and proactive intervention. The model can reliably predict
where the member will go, and the cost impact of routing them correctly is high.

**Quadrant definitions:**
- **Top Right — Predictable High Cost:** High probability, high cost. Act here first.
- **Top Left — Unpredictable High Cost:** High cost but scattered routing. Harder to intervene.
- **Bottom Right — Predictable Low Cost:** Reliable pathways but lower financial impact.
- **Bottom Left — Noise:** Low probability, low cost. Low priority.

Dot size reflects transition volume. Color reflects entropy — green is more predictable.
Median probability and median cost per member used as quadrant dividers.
"""))

fig, axes = plt.subplots(2, 2, figsize=(24, 20))
axes = axes.flatten()

prob_median = df["conditional_probability"].median()
cost_median = df["avg_allowed_per_member"].median()

for i, cohort in enumerate(COHORTS):
    ax = axes[i]
    sub = df[df["member_segment"] == cohort].copy()
    if sub.empty:
        ax.set_title(f"{cohort} — No Data")
        continue
    scatter = ax.scatter(
        sub["conditional_probability"],
        sub["avg_allowed_per_member"],
        c=sub["conditional_entropy"],
        s=sub["transition_count"] / sub["transition_count"].max() * 400 + 20,
        alpha=0.6,
        cmap="RdYlGn_r"
    )
    plt.colorbar(scatter, ax=ax, label="Entropy")
    ax.axvline(prob_median, color="grey", linestyle="--", alpha=0.5)
    ax.axhline(cost_median, color="grey", linestyle="--", alpha=0.5)
    ax.text(0.98, 0.98, "Predictable\nHigh Cost", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="red", fontweight="bold")
    ax.text(0.02, 0.98, "Unpredictable\nHigh Cost", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color="orange", fontweight="bold")
    ax.text(0.98, 0.02, "Predictable\nLow Cost", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="green", fontweight="bold")
    ax.text(0.02, 0.02, "Noise", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8, color="grey", fontweight="bold")
    ax.set_xlabel("Conditional Probability", fontsize=9)
    ax.set_ylabel("Avg Allowed Per Member ($)", fontsize=9)
    ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_usd(x)))
    ax.grid(True, linestyle="--", alpha=0.3)

fig.suptitle("Predictability vs Cost Quadrant — First Encounter Transitions",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("quadrant_cost_entropy.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ADDITION 2 — MEMBER DENSITY
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Addition 2 — Member Density

Member density = unique members / total transitions.

A density of 1.0 means every transition comes from a different member —
the pathway is broad and generalizable across the population.

A low density means the same members are generating repeat transitions —
the signal may be concentrated in a small high-utilizer population.

**Why this matters for modeling:**
High density transitions produce more generalizable model predictions.
Low density transitions may overfit to a small group of high-utilizers.

Left chart shows transitions with highest density — broadest population signal.
Right chart shows transitions with lowest density — concentrated in few members.
"""))

density_top = pareto_df.sort_values("member_density", ascending=False).head(20).copy()
density_top["transition_label"] = (
    density_top["current_dx_desc"].str[:25] + " → " +
    density_top["next_specialty_desc"].str[:20]
)

density_low = pareto_df.sort_values("member_density", ascending=True).head(20).copy()
density_low["transition_label"] = (
    density_low["current_dx_desc"].str[:25] + " → " +
    density_low["next_specialty_desc"].str[:20]
)

fig, axes = plt.subplots(1, 2, figsize=(24, 10))

d_high = density_top.sort_values("member_density", ascending=True)
axes[0].barh(d_high["transition_label"], d_high["member_density"],
             color="#5DBE7E", alpha=0.85)
axes[0].set_xlabel("Member Density (unique members / transitions)", fontsize=9)
axes[0].set_title("Highest Member Density\n(most generalizable transitions)",
                  fontsize=11, fontweight="bold")
axes[0].axvline(1.0, color="red", linestyle="--", alpha=0.5, label="Density = 1.0")
axes[0].legend(fontsize=8)
plt.setp(axes[0].get_yticklabels(), fontsize=7)

d_low = density_low.sort_values("member_density", ascending=False)
axes[1].barh(d_low["transition_label"], d_low["member_density"],
             color="#F4845F", alpha=0.85)
axes[1].set_xlabel("Member Density (unique members / transitions)", fontsize=9)
axes[1].set_title("Lowest Member Density\n(concentrated in high-utilizers)",
                  fontsize=11, fontweight="bold")
plt.setp(axes[1].get_yticklabels(), fontsize=7)

fig.suptitle("Member Density — Generalizability of Transition Signal",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("member_density.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ADDITION 3 — COHORT BREAKDOWN FOR TOP 10 PARETO TRANSITIONS
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Addition 3 — Cohort Breakdown for Top 10 High-Value Transitions

For the top 10 transitions by total spend, this section shows how volume
and spend split across member cohorts — Adult Female, Adult Male, Senior, Children.

Some high-cost pathways may be entirely driven by one cohort — typically Seniors.
This has direct implications for:
- Model scope — should the model be cohort-specific?
- Care navigation targeting — which population to prioritize
- Network adequacy — which demographics need more specialist capacity

Total spend shown in millions USD. Per-member cost shown in USD.
"""))

top10_pairs = pareto_df.head(10)[["current_dx", "next_specialty"]].values.tolist()

top10_cohort = df[
    df.apply(lambda r: [r["current_dx"], r["next_specialty"]] in top10_pairs, axis=1)
].copy()

top10_cohort["transition_label"] = (
    top10_cohort["current_dx_desc"].str[:20] + " → " +
    top10_cohort["next_specialty_desc"].str[:15]
)

pivot_vol = top10_cohort.pivot_table(
    index="transition_label", columns="member_segment",
    values="transition_count", aggfunc="sum"
).fillna(0)

pivot_cost = top10_cohort.pivot_table(
    index="transition_label", columns="member_segment",
    values="avg_allowed_per_member", aggfunc="mean"
).fillna(0)

cohort_colors = {
    "Adult_Female": "#4C9BE8",
    "Adult_Male":   "#F4845F",
    "Senior":       "#5DBE7E",
    "Children":     "#F7C948"
}
vol_colors  = [cohort_colors.get(c, "#AAAAAA") for c in pivot_vol.columns]
cost_colors = [cohort_colors.get(c, "#AAAAAA") for c in pivot_cost.columns]

fig, axes = plt.subplots(1, 2, figsize=(26, 12))

pivot_vol.plot(kind="barh", stacked=True, ax=axes[0], color=vol_colors, alpha=0.85)
axes[0].set_xlabel("Total Transitions", fontsize=9)
axes[0].set_title("Transition Volume by Cohort\nTop 10 High-Value Transitions",
                  fontsize=11, fontweight="bold")
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[0].legend(title="Cohort", fontsize=8)
plt.setp(axes[0].get_yticklabels(), fontsize=8)

pivot_cost.plot(kind="barh", ax=axes[1], color=cost_colors, alpha=0.85)
axes[1].set_xlabel("Avg Allowed Per Member ($)", fontsize=9)
axes[1].set_title("Average Cost Per Member by Cohort\nTop 10 High-Value Transitions",
                  fontsize=11, fontweight="bold")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_usd(x)))
axes[1].legend(title="Cohort", fontsize=8)
plt.setp(axes[1].get_yticklabels(), fontsize=8)

fig.suptitle("Cohort Breakdown — Top 10 High-Value Transitions",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("cohort_breakdown_top10.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — COST PER MEMBER
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
# Analysis 2 — Most Expensive Specialties Per Member

Identifies specialties with the highest average allowed amount per member
regardless of transition volume.

This is different from total spend — a specialty may rank low on total spend
but very high on per-member cost if only a small number of members use it
but each visit is extremely expensive.

High per-member cost specialties are important for:
- Risk stratification — members routed here are high-cost
- Network design — ensuring sufficient in-network capacity to avoid out-of-network costs
- Care management — identifying members who may need case management intervention

Per-member and per-visit costs shown in USD. Total spend shown in millions USD.

---
"""))

spec_cost = (
    df.groupby(["next_specialty", "next_specialty_desc"], as_index=False)
    .agg(total_transitions=("transition_count", "sum"),
         unique_members=("unique_members", "sum"),
         total_allowed_amt=("total_allowed_amt", "sum"),
         avg_allowed_per_member=("avg_allowed_per_member", "mean"),
         avg_entropy=("conditional_entropy", "mean"),
         avg_member_density=("member_density", "mean"))
    .sort_values("avg_allowed_per_member", ascending=False)
    .head(15)
)

display(Markdown("### Top 15 Most Expensive Specialties Per Member"))

spec_cost_display = spec_cost[[
    "next_specialty_desc", "avg_allowed_per_member", "total_transitions",
    "unique_members", "total_allowed_amt", "avg_entropy", "avg_member_density"
]].rename(columns={
    "next_specialty_desc": "Specialty",
    "avg_allowed_per_member": "Avg Spend Per Member ($)",
    "total_transitions": "Total Transitions",
    "unique_members": "Unique Members",
    "total_allowed_amt": "Total Spend (USD M)",
    "avg_entropy": "Avg Entropy",
    "avg_member_density": "Member Density"
}).reset_index(drop=True)

display(spec_cost_display.style.format({
    "Total Spend (USD M)": lambda x: fmt_millions(x),
    "Avg Spend Per Member ($)": lambda x: fmt_usd(x),
    "Total Transitions": lambda x: fmt_count(x),
    "Unique Members": lambda x: fmt_count(x),
    "Avg Entropy": "{:.4f}",
    "Member Density": "{:.3f}"
}))

display(Markdown("""
#### Average Cost Per Member by Specialty

Specialties ranked purely by per-member cost.
Color intensity reflects cost level — darker red means higher cost.
Member count annotated on each bar for context — a high-cost specialty
with very few members may represent an outlier or niche pathway.
"""))

fig, ax = plt.subplots(figsize=(14, 8))
spec_plot = spec_cost.sort_values("avg_allowed_per_member", ascending=True)
colors = plt.cm.RdYlGn_r(
    (spec_plot["avg_allowed_per_member"] - spec_plot["avg_allowed_per_member"].min())
    / (spec_plot["avg_allowed_per_member"].max() - spec_plot["avg_allowed_per_member"].min() + 1e-9)
)
bars = ax.barh(spec_plot["next_specialty_desc"], spec_plot["avg_allowed_per_member"],
               color=colors, alpha=0.85)
for bar, (_, row) in zip(bars, spec_plot.iterrows()):
    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{fmt_count(row['unique_members'])} members",
            va="center", fontsize=7, color="grey")
ax.set_xlabel("Average Allowed Amount Per Member ($)", fontsize=10)
ax.set_title("Most Expensive Specialties Per Member", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_usd(x)))
plt.setp(ax.get_yticklabels(), fontsize=9)
plt.tight_layout()
plt.savefig("cost_per_member_specialty.png", dpi=150, bbox_inches="tight")
plt.show()

display(Markdown("""
### Diagnoses Driving Members to High-Cost Specialties

For each of the top 10 most expensive specialties, the chart below shows which
diagnosis codes are routing members there and the average cost per member
for each diagnosis pathway.

**Which conditions are creating the most expensive specialist referrals?**
Per-member costs shown in USD.
"""))

top_expensive_specs = spec_cost["next_specialty"].head(10).tolist()

dx_for_expensive = (
    df[df["next_specialty"].isin(top_expensive_specs)]
    .groupby(["next_specialty", "next_specialty_desc",
              "current_dx", "current_dx_desc"], as_index=False)
    .agg(transition_count=("transition_count", "sum"),
         unique_members=("unique_members", "sum"),
         avg_allowed_per_member=("avg_allowed_per_member", "mean"),
         conditional_entropy=("conditional_entropy", "mean"))
    .sort_values(["next_specialty_desc", "avg_allowed_per_member"], ascending=[True, False])
)

fig, axes = plt.subplots(2, 5, figsize=(32, 16))
axes = axes.flatten()

for i, spec in enumerate(top_expensive_specs):
    ax = axes[i]
    sub = (
        dx_for_expensive[dx_for_expensive["next_specialty"] == spec]
        .sort_values("avg_allowed_per_member", ascending=False)
        .head(5)
    )
    spec_desc = sub["next_specialty_desc"].iloc[0] if not sub.empty else spec
    if sub.empty:
        ax.set_title(f"{spec_desc}\nNo Data")
        continue
    colors = plt.cm.RdYlGn_r(
        (sub["avg_allowed_per_member"] - sub["avg_allowed_per_member"].min())
        / (sub["avg_allowed_per_member"].max() - sub["avg_allowed_per_member"].min() + 1e-9)
    )
    ax.barh(sub["current_dx_desc"], sub["avg_allowed_per_member"],
            color=colors, alpha=0.85)
    ax.set_title(f"{spec_desc}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Avg Spend Per Member ($)", fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_usd(x)))
    plt.setp(ax.get_yticklabels(), fontsize=7)

fig.suptitle("Top 5 Diagnoses Driving Members to High-Cost Specialties",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("dx_driving_expensive_specialties.png", dpi=150, bbox_inches="tight")
plt.show()
