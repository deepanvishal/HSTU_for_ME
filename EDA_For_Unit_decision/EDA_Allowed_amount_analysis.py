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

df = client.query(f"""
    SELECT * FROM `{DATASET}.{TABLE}`
    WHERE transition_count >= {THRESHOLD}
      AND conditional_entropy IS NOT NULL
""").to_dataframe()

for col in ["transition_count", "unique_members", "conditional_probability",
            "conditional_entropy", "total_allowed_amt",
            "avg_allowed_per_transition", "avg_allowed_per_member"]:
    df[col] = df[col].astype(float)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — PARETO ON TOTAL COST
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
# Analysis 1 — Pareto on Total Cost

Identifies the DX to specialty transitions that account for 80% of total allowed spend.
From that high-value set, surfaces the most impactful starting diagnoses and destination
specialties by both volume and spend.

---
"""))

# Step 1 — aggregate to dx + specialty level across all cohorts
transition_agg = (
    df.groupby(["current_dx", "current_dx_desc", "next_specialty",
                "next_specialty_desc", "current_ccsr", "current_ccsr_desc"],
               as_index=False)
    .agg(transition_count=("transition_count", "sum"),
         unique_members=("unique_members", "sum"),
         total_allowed_amt=("total_allowed_amt", "sum"),
         avg_allowed_per_member=("avg_allowed_per_member", "mean"),
         conditional_entropy=("conditional_entropy", "mean"))
    .sort_values("total_allowed_amt", ascending=False)
)

# Step 2 — cumulative spend and pareto cutoff at 80%
total_spend = transition_agg["total_allowed_amt"].sum()
transition_agg["cumulative_spend"] = transition_agg["total_allowed_amt"].cumsum()
transition_agg["cumulative_pct"] = transition_agg["cumulative_spend"] / total_spend
pareto_df = transition_agg[transition_agg["cumulative_pct"] <= 0.80].copy()

display(Markdown(f"""
**Total transitions:** {len(transition_agg):,}
**Total allowed spend:** ${total_spend:,.0f}
**Transitions in 80% spend:** {len(pareto_df):,} ({len(pareto_df)/len(transition_agg)*100:.1f}% of transitions)
"""))

# ── TOP DX BY VOLUME AND SPEND ────────────────────────────────────────────────
display(Markdown("### Most Impactful Starting Diagnoses — by Volume and Spend"))

dx_summary = (
    pareto_df.groupby(["current_dx", "current_dx_desc", "current_ccsr_desc"],
                      as_index=False)
    .agg(total_transitions=("transition_count", "sum"),
         unique_members=("unique_members", "sum"),
         total_allowed_amt=("total_allowed_amt", "sum"),
         avg_allowed_per_member=("avg_allowed_per_member", "mean"),
         avg_entropy=("conditional_entropy", "mean"))
    .sort_values("total_allowed_amt", ascending=False)
    .head(20)
)

display(dx_summary[[
    "current_dx_desc", "current_ccsr_desc", "total_transitions",
    "unique_members", "total_allowed_amt", "avg_allowed_per_member", "avg_entropy"
]].rename(columns={
    "current_dx_desc": "Diagnosis",
    "current_ccsr_desc": "Clinical Domain",
    "total_transitions": "Total Transitions",
    "unique_members": "Unique Members",
    "total_allowed_amt": "Total Spend ($)",
    "avg_allowed_per_member": "Avg Spend Per Member ($)",
    "avg_entropy": "Avg Entropy"
}).reset_index(drop=True))

# ── DX CHARTS ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

# by volume
dx_vol = dx_summary.sort_values("total_transitions", ascending=True).tail(15)
axes[0].barh(dx_vol["current_dx_desc"], dx_vol["total_transitions"], color="#4C9BE8", alpha=0.85)
axes[0].set_xlabel("Total Transitions", fontsize=10)
axes[0].set_title("Top 15 Diagnoses by Volume\n(within 80% spend)", fontsize=12, fontweight="bold")
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.setp(axes[0].get_yticklabels(), fontsize=8)

# by spend
dx_spend = dx_summary.sort_values("total_allowed_amt", ascending=True).tail(15)
axes[1].barh(dx_spend["current_dx_desc"], dx_spend["total_allowed_amt"], color="#F4845F", alpha=0.85)
axes[1].set_xlabel("Total Allowed Amount ($)", fontsize=10)
axes[1].set_title("Top 15 Diagnoses by Spend\n(within 80% spend)", fontsize=12, fontweight="bold")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.setp(axes[1].get_yticklabels(), fontsize=8)

fig.suptitle("Most Impactful Starting Diagnoses — Pareto Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("pareto_dx.png", dpi=150, bbox_inches="tight")
plt.show()

# ── TOP SPECIALTY BY VOLUME AND SPEND ─────────────────────────────────────────
display(Markdown("### Most Impactful Destination Specialties — by Volume and Spend"))

spec_summary = (
    pareto_df.groupby(["next_specialty", "next_specialty_desc"], as_index=False)
    .agg(total_transitions=("transition_count", "sum"),
         unique_members=("unique_members", "sum"),
         total_allowed_amt=("total_allowed_amt", "sum"),
         avg_allowed_per_member=("avg_allowed_per_member", "mean"),
         avg_entropy=("conditional_entropy", "mean"))
    .sort_values("total_allowed_amt", ascending=False)
    .head(20)
)

display(spec_summary[[
    "next_specialty_desc", "total_transitions", "unique_members",
    "total_allowed_amt", "avg_allowed_per_member", "avg_entropy"
]].rename(columns={
    "next_specialty_desc": "Specialty",
    "total_transitions": "Total Transitions",
    "unique_members": "Unique Members",
    "total_allowed_amt": "Total Spend ($)",
    "avg_allowed_per_member": "Avg Spend Per Member ($)",
    "avg_entropy": "Avg Entropy"
}).reset_index(drop=True))

# ── SPECIALTY CHARTS ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

spec_vol = spec_summary.sort_values("total_transitions", ascending=True).tail(15)
axes[0].barh(spec_vol["next_specialty_desc"], spec_vol["total_transitions"],
             color="#4C9BE8", alpha=0.85)
axes[0].set_xlabel("Total Transitions", fontsize=10)
axes[0].set_title("Top 15 Specialties by Volume\n(within 80% spend)", fontsize=12, fontweight="bold")
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.setp(axes[0].get_yticklabels(), fontsize=8)

spec_spend = spec_summary.sort_values("total_allowed_amt", ascending=True).tail(15)
axes[1].barh(spec_spend["next_specialty_desc"], spec_spend["total_allowed_amt"],
             color="#F4845F", alpha=0.85)
axes[1].set_xlabel("Total Allowed Amount ($)", fontsize=10)
axes[1].set_title("Top 15 Specialties by Spend\n(within 80% spend)", fontsize=12, fontweight="bold")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.setp(axes[1].get_yticklabels(), fontsize=8)

fig.suptitle("Most Impactful Destination Specialties — Pareto Set", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("pareto_specialty.png", dpi=150, bbox_inches="tight")
plt.show()

# ── PARETO CURVE ──────────────────────────────────────────────────────────────
display(Markdown("""
#### Pareto Curve — Cumulative Spend vs Transition Count

Shows how quickly total spend concentrates in top transitions.
The red line marks the 80% threshold.
"""))

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(1, len(transition_agg) + 1)
ax.plot(x, transition_agg["cumulative_pct"] * 100, color="#4C9BE8", linewidth=2)
ax.axhline(80, color="red", linestyle="--", alpha=0.7, label="80% threshold")
ax.axvline(len(pareto_df), color="orange", linestyle="--", alpha=0.7,
           label=f"{len(pareto_df):,} transitions")
ax.set_xlabel("Number of Transitions (ranked by spend)", fontsize=10)
ax.set_ylabel("Cumulative Spend %", fontsize=10)
ax.set_title("Pareto Curve — Cumulative Spend Concentration", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
plt.tight_layout()
plt.savefig("pareto_curve.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — COST PER MEMBER
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
# Analysis 2 — Most Expensive Specialties Per Member

Identifies the specialties with the highest average allowed amount per member
regardless of transition volume.

For each high-cost specialty, surfaces which diagnosis codes are routing members there.

---
"""))

# Step 1 — rank specialties by avg cost per member
spec_cost = (
    df.groupby(["next_specialty", "next_specialty_desc"], as_index=False)
    .agg(total_transitions=("transition_count", "sum"),
         unique_members=("unique_members", "sum"),
         total_allowed_amt=("total_allowed_amt", "sum"),
         avg_allowed_per_member=("avg_allowed_per_member", "mean"),
         avg_entropy=("conditional_entropy", "mean"))
    .sort_values("avg_allowed_per_member", ascending=False)
    .head(15)
)

display(Markdown("### Top 15 Most Expensive Specialties Per Member"))
display(spec_cost[[
    "next_specialty_desc", "avg_allowed_per_member", "total_transitions",
    "unique_members", "total_allowed_amt", "avg_entropy"
]].rename(columns={
    "next_specialty_desc": "Specialty",
    "avg_allowed_per_member": "Avg Spend Per Member ($)",
    "total_transitions": "Total Transitions",
    "unique_members": "Unique Members",
    "total_allowed_amt": "Total Spend ($)",
    "avg_entropy": "Avg Entropy"
}).reset_index(drop=True))

# ── COST PER MEMBER BAR CHART ─────────────────────────────────────────────────
display(Markdown("""
#### Average Cost Per Member by Specialty

Specialties ranked by average allowed amount per member.
Size of bar indicates cost — not volume.
"""))

fig, ax = plt.subplots(figsize=(14, 8))
spec_plot = spec_cost.sort_values("avg_allowed_per_member", ascending=True)
colors = plt.cm.RdYlGn_r(
    (spec_plot["avg_allowed_per_member"] - spec_plot["avg_allowed_per_member"].min())
    / (spec_plot["avg_allowed_per_member"].max() - spec_plot["avg_allowed_per_member"].min() + 1e-9)
)
ax.barh(spec_plot["next_specialty_desc"], spec_plot["avg_allowed_per_member"],
        color=colors, alpha=0.85)
ax.set_xlabel("Average Allowed Amount Per Member ($)", fontsize=10)
ax.set_title("Most Expensive Specialties Per Member", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
plt.setp(ax.get_yticklabels(), fontsize=9)
plt.tight_layout()
plt.savefig("cost_per_member_specialty.png", dpi=150, bbox_inches="tight")
plt.show()

# Step 2 — for top 10 expensive specialties, show contributing DX codes
top_expensive_specs = spec_cost["next_specialty"].head(10).tolist()

display(Markdown("### Diagnoses Driving Members to High-Cost Specialties"))

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

# chart — top 5 DX per expensive specialty
fig, axes = plt.subplots(2, 5, figsize=(30, 14))
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
    ax.barh(sub["current_dx_desc"], sub["avg_allowed_per_member"], color=colors, alpha=0.85)
    ax.set_title(f"{spec_desc}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Avg Spend Per Member ($)", fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.setp(ax.get_yticklabels(), fontsize=7)

fig.suptitle("Top 5 Diagnoses Driving Members to High-Cost Specialties",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("dx_driving_expensive_specialties.png", dpi=150, bbox_inches="tight")
plt.show()
