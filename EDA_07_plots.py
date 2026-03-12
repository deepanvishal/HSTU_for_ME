from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import networkx as nx
from IPython.display import display, Markdown

client = bigquery.Client(project="anbc-hcb-dev")
DATASET = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
THRESHOLD = 100
COHORTS = ["Adult_Female", "Adult_Male", "Senior", "Children"]

# ── ENTROPY SUMMARY ───────────────────────────────────────────────────────────
query_entropy = f"""
SELECT
    combination
    ,markov_order
    ,member_segment
    ,weighted_avg_entropy
    ,median_entropy
    ,total_transitions
    ,unique_current_states
FROM `{DATASET}.A870800_gen_rec_f_entropy_summary`
ORDER BY markov_order, weighted_avg_entropy ASC
"""
df_entropy = client.query(query_entropy).to_dataframe()

# ── ORDER 1 SUMMARY TABLE ─────────────────────────────────────────────────────
display(Markdown("""
---
# First Encounter Transition Analysis
## Order 1 — Single Visit Context

This analysis examines what happens **after a member's first encounter with a new diagnosis**.
For each combination of what we observe at the first visit and what we predict at the next visit,
we measure **predictive power using conditional entropy**.

A **lower entropy value means higher predictive power** — the first visit strongly
suggests where the member will go next.
A **higher entropy value means lower predictive power** — the next visit is unpredictable
from the first encounter alone.

---
"""))

order1_summary = (
    df_entropy[df_entropy["markov_order"] == 1]
    .groupby("combination", as_index=False)["weighted_avg_entropy"]
    .mean()
    .sort_values("weighted_avg_entropy")
)

def parse_combination(c):
    parts = c.split("_to_")
    mapping = {
        "dx": "Diagnosis Code (ICD-10)",
        "ccsr": "CCSR Clinical Category",
        "specialty": "Provider Specialty"
    }
    return mapping.get(parts[0], parts[0]), mapping.get(parts[1], parts[1])

order1_summary[["First Visit Unit", "Prediction Unit"]] = pd.DataFrame(
    order1_summary["combination"].apply(parse_combination).tolist(),
    index=order1_summary.index
)
order1_summary["Predictive Power (Entropy)"] = order1_summary["weighted_avg_entropy"].round(4)
order1_summary["Signal Strength"] = order1_summary["weighted_avg_entropy"].apply(
    lambda x: "Strong" if x < 1.0 else ("Moderate" if x < 2.0 else "Weak")
)

display(order1_summary[["First Visit Unit", "Prediction Unit", "Predictive Power (Entropy)", "Signal Strength"]]
        .reset_index(drop=True)
        .style.background_gradient(subset=["Predictive Power (Entropy)"], cmap="RdYlGn_r")
        .set_caption("Order 1 Transition Combinations — Ranked by Predictive Power"))


# ── LOAD ORDER 1 TABLES ───────────────────────────────────────────────────────
def load_table(table_suffix, extra_cols=""):
    return client.query(f"""
        SELECT * FROM `{DATASET}.A870800_gen_rec_f_{table_suffix}`
        WHERE transition_count >= {THRESHOLD}
          AND conditional_entropy IS NOT NULL
    """).to_dataframe()

df_dx_spec   = load_table("dx_to_specialty_order1")
df_dx_ccsr   = load_table("dx_to_ccsr_order1")
df_dx_dx     = load_table("dx_to_dx_order1")

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def top10_transitions(df, current_col, next_col, current_desc, next_desc):
    return (
        df.groupby([current_col, current_desc, next_col, next_desc, "member_segment"], as_index=False)
        .agg(transition_count=("transition_count", "sum"),
             unique_members=("unique_members", "sum"),
             conditional_probability=("conditional_probability", "mean"),
             conditional_entropy=("conditional_entropy", "mean"))
        .sort_values("transition_count", ascending=False)
    )

def plot_heatmap(df, current_col, next_col, current_desc, next_desc, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()
    for i, cohort in enumerate(COHORTS):
        ax = axes[i]
        sub = df[df["member_segment"] == cohort]
        top_current = sub.groupby(current_col)["transition_count"].sum().nlargest(15).index
        top_next = sub.groupby(next_col)["transition_count"].sum().nlargest(15).index
        sub = sub[sub[current_col].isin(top_current) & sub[next_col].isin(top_next)]
        pivot = (
            sub.groupby([current_desc, next_desc], as_index=False)["transition_count"]
            .sum()
            .pivot_table(index=current_desc, columns=next_desc, values="transition_count", aggfunc="sum")
            .fillna(0)
            .astype(float)
        )
        if pivot.empty:
            ax.set_title(f"{cohort} — No Data")
            continue
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="lightgrey",
                    annot=True, fmt=".0f", annot_kws={"size": 7},
                    cbar_kws={"label": "Transition Count"})
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
        ax.set_xlabel(next_desc.replace("_", " "), fontsize=9)
        ax.set_ylabel(current_desc.replace("_", " "), fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()

def plot_ccsr_entropy(df, current_col, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    axes = axes.flatten()
    for i, cohort in enumerate(COHORTS):
        ax = axes[i]
        sub = df[df["member_segment"] == cohort]
        domain = (
            sub.groupby(["current_ccsr", "current_ccsr_desc"], as_index=False)
            .apply(lambda g: pd.Series({
                "weighted_avg_entropy": np.average(
                    g["conditional_entropy"], weights=g["transition_count"]),
                "total_transitions": g["transition_count"].sum()
            }))
            .reset_index(drop=True)
            .nlargest(10, "total_transitions")
            .sort_values("weighted_avg_entropy")
        )
        if domain.empty:
            ax.set_title(f"{cohort} — No Data")
            continue
        colors = plt.cm.RdYlGn_r(
            (domain["weighted_avg_entropy"] - domain["weighted_avg_entropy"].min())
            / (domain["weighted_avg_entropy"].max() - domain["weighted_avg_entropy"].min() + 1e-9)
        )
        ax.barh(domain["current_ccsr_desc"], domain["weighted_avg_entropy"], color=colors)
        ax.set_xlabel("Weighted Average Entropy", fontsize=9)
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.invert_yaxis()
        plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()

def plot_scatter(df, current_col, current_desc, next_col, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    axes = axes.flatten()
    for i, cohort in enumerate(COHORTS):
        ax = axes[i]
        sub = df[df["member_segment"] == cohort].copy()
        top_current = sub.groupby(current_col)["transition_count"].sum().nlargest(15).index
        sub = sub[sub[current_col].isin(top_current)].copy()
        sub["transition_count"] = sub["transition_count"].astype(float)
        sub["conditional_probability"] = sub["conditional_probability"].astype(float)
        sub["unique_members"] = sub["unique_members"].astype(float)
        if sub.empty:
            ax.set_title(f"{cohort} — No Data")
            continue
        scatter = ax.scatter(
            sub["transition_count"],
            sub["conditional_probability"],
            c=sub["conditional_entropy"].astype(float),
            s=sub["unique_members"] / sub["unique_members"].max() * 300 + 20,
            alpha=0.6,
            cmap="RdYlGn_r"
        )
        plt.colorbar(scatter, ax=ax, label="Entropy")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50% threshold")
        ax.set_xlabel("Transition Count", fontsize=9)
        ax.set_ylabel("Conditional Probability", fontsize=9)
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()

def plot_network(df, current_col, current_desc, next_col, next_desc, cohort, title, filename):
    sub = (
        df[df["member_segment"] == cohort]
        .groupby([current_col, current_desc, next_col, next_desc], as_index=False)
        ["transition_count"].sum()
        .sort_values("transition_count", ascending=False)
        .head(20)
    )
    if sub.empty:
        print(f"No data for {cohort}")
        return

    G = nx.DiGraph()
    for _, row in sub.iterrows():
        G.add_edge(row[current_desc], row[next_desc], weight=row["transition_count"])

    current_nodes = sub[current_desc].unique().tolist()
    next_nodes    = sub[next_desc].unique().tolist()
    node_colors   = ["#4C9BE8" if n in current_nodes else "#F4845F" for n in G.nodes()]
    max_weight    = sub["transition_count"].max()
    edge_widths   = [G[u][v]["weight"] / max_weight * 12 + 2 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(22, 16))
    pos = nx.spring_layout(G, k=4.0, seed=42, iterations=100)
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="#555555",
                           arrowsize=30, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.1",
                           min_source_margin=30, min_target_margin=30, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)
    ax.set_title(f"{title} — {cohort}", fontsize=14, fontweight="bold")
    ax.axis("off")
    blue_patch  = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C9BE8",
                              markersize=12, label="First Visit State")
    orange_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#F4845F",
                               markersize=12, label="Next Visit State")
    ax.legend(handles=[blue_patch, orange_patch], loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — DIAGNOSIS CODE TO PROVIDER SPECIALTY
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Part 1 — Diagnosis Code to Provider Specialty

**What this section answers:**
Given that a member presents with a specific diagnosis for the first time,
how predictably does that diagnosis route them to a particular provider specialty at the next visit?

A strong signal here means the diagnosis itself is a reliable indicator of where the member will seek care next.
This is the most actionable combination for care navigation and provider matching.

---
"""))

top10_spec = top10_transitions(df_dx_spec, "current_dx", "next_specialty",
                                "current_dx_desc", "next_specialty_desc")

display(Markdown("### Top 10 Transitions — Diagnosis Code to Provider Specialty"))
display(top10_spec.head(10)[[
    "current_dx_desc", "next_specialty_desc", "member_segment",
    "transition_count", "unique_members", "conditional_probability", "conditional_entropy"
]].rename(columns={
    "current_dx_desc": "Diagnosis",
    "next_specialty_desc": "Next Specialty",
    "member_segment": "Cohort",
    "transition_count": "Transition Count",
    "unique_members": "Unique Members",
    "conditional_probability": "Probability",
    "conditional_entropy": "Entropy"
}).reset_index(drop=True))

display(Markdown("""
#### Heatmap — Transition Volume by Diagnosis and Specialty

Each cell shows how many times members with a given diagnosis visited a given specialty next.
Darker cells indicate higher volume transitions.
"""))
plot_heatmap(df_dx_spec, "current_dx", "next_specialty", "current_dx_desc", "next_specialty_desc",
             "Diagnosis Code to Provider Specialty — Transition Volume by Cohort",
             "heatmap_dx_to_specialty.png")

display(Markdown("""
#### Clinical Domain Entropy — Top 10 Domains by Volume

This chart shows how predictable specialty routing is for each clinical domain.
Domains on the left (green) have strong routing patterns.
Domains on the right (red) scatter across many specialties unpredictably.
"""))
plot_ccsr_entropy(df_dx_spec, "current_dx",
                  "Clinical Domain Entropy — Diagnosis to Specialty by Cohort",
                  "entropy_dx_to_specialty.png")

display(Markdown("""
#### Probability vs Volume — Transition Confidence

Each dot represents one diagnosis to specialty transition.
Dots above the red line (50%) are high confidence transitions.
Dot size reflects number of unique members driving that transition.
Color reflects entropy — green means more predictable.
"""))
plot_scatter(df_dx_spec, "current_dx", "current_dx_desc", "next_specialty",
             "Transition Confidence — Diagnosis to Specialty by Cohort",
             "scatter_dx_to_specialty.png")

display(Markdown("""
#### Transition Network — Top 20 Paths

Blue nodes are diagnosis codes at the first visit.
Orange nodes are provider specialties at the next visit.
Arrow thickness reflects transition volume — thicker means more members followed that path.
"""))
for cohort in COHORTS:
    plot_network(df_dx_spec, "current_dx", "current_dx_desc", "next_specialty", "next_specialty_desc",
                 cohort, "Diagnosis to Specialty Network",
                 f"network_dx_to_specialty_{cohort}.png")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — DIAGNOSIS CODE TO CCSR CLINICAL CATEGORY
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Part 2 — Diagnosis Code to CCSR Clinical Category

**What this section answers:**
Given a first diagnosis encounter, how predictably does that lead to a specific
clinical category of conditions at the next visit?

CCSR categories group thousands of ICD-10 codes into ~530 clinical categories.
This gives a higher-level view of clinical progression —
does a first diabetes encounter reliably predict the next visit involves cardiovascular conditions?

---
"""))

top10_ccsr = top10_transitions(df_dx_ccsr, "current_dx", "next_ccsr",
                                "current_dx_desc", "next_ccsr_desc")

display(Markdown("### Top 10 Transitions — Diagnosis Code to CCSR Category"))
display(top10_ccsr.head(10)[[
    "current_dx_desc", "next_ccsr_desc", "member_segment",
    "transition_count", "unique_members", "conditional_probability", "conditional_entropy"
]].rename(columns={
    "current_dx_desc": "Diagnosis",
    "next_ccsr_desc": "Next Clinical Category",
    "member_segment": "Cohort",
    "transition_count": "Transition Count",
    "unique_members": "Unique Members",
    "conditional_probability": "Probability",
    "conditional_entropy": "Entropy"
}).reset_index(drop=True))

display(Markdown("""
#### Heatmap — Transition Volume by Diagnosis and Clinical Category
"""))
plot_heatmap(df_dx_ccsr, "current_dx", "next_ccsr", "current_dx_desc", "next_ccsr_desc",
             "Diagnosis Code to CCSR Category — Transition Volume by Cohort",
             "heatmap_dx_to_ccsr.png")

display(Markdown("""
#### Clinical Domain Entropy — Top 10 Domains by Volume
"""))
plot_ccsr_entropy(df_dx_ccsr, "current_dx",
                  "Clinical Domain Entropy — Diagnosis to CCSR by Cohort",
                  "entropy_dx_to_ccsr.png")

display(Markdown("""
#### Probability vs Volume — Transition Confidence
"""))
plot_scatter(df_dx_ccsr, "current_dx", "current_dx_desc", "next_ccsr",
             "Transition Confidence — Diagnosis to CCSR by Cohort",
             "scatter_dx_to_ccsr.png")

display(Markdown("""
#### Transition Network — Top 20 Paths
"""))
for cohort in COHORTS:
    plot_network(df_dx_ccsr, "current_dx", "current_dx_desc", "next_ccsr", "next_ccsr_desc",
                 cohort, "Diagnosis to CCSR Network",
                 f"network_dx_to_ccsr_{cohort}.png")


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — DIAGNOSIS CODE TO DIAGNOSIS CODE
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Part 3 — Diagnosis Code to Diagnosis Code

**What this section answers:**
Given a first encounter with a diagnosis, what diagnosis is the member most likely
to present with at their next visit?

This captures clinical progression patterns — does a first hypertension diagnosis
predict diabetes at the next visit? Does a first anxiety diagnosis predict depression?

This is the most granular and most complex combination — 70,000 possible ICD-10 codes
on both sides. The 100+ transition filter ensures only reliable, repeatable patterns appear.

---
"""))

top10_dx = top10_transitions(df_dx_dx, "current_dx", "next_dx",
                              "current_dx_desc", "next_dx_desc")

display(Markdown("### Top 10 Transitions — Diagnosis Code to Diagnosis Code"))
display(top10_dx.head(10)[[
    "current_dx_desc", "next_dx_desc", "member_segment",
    "transition_count", "unique_members", "conditional_probability", "conditional_entropy"
]].rename(columns={
    "current_dx_desc": "First Diagnosis",
    "next_dx_desc": "Next Diagnosis",
    "member_segment": "Cohort",
    "transition_count": "Transition Count",
    "unique_members": "Unique Members",
    "conditional_probability": "Probability",
    "conditional_entropy": "Entropy"
}).reset_index(drop=True))

display(Markdown("""
#### Heatmap — Transition Volume by Diagnosis Pair
"""))
plot_heatmap(df_dx_dx, "current_dx", "next_dx", "current_dx_desc", "next_dx_desc",
             "Diagnosis Code to Diagnosis Code — Transition Volume by Cohort",
             "heatmap_dx_to_dx.png")

display(Markdown("""
#### Clinical Domain Entropy — Top 10 Domains by Volume
"""))
plot_ccsr_entropy(df_dx_dx, "current_dx",
                  "Clinical Domain Entropy — Diagnosis to Diagnosis by Cohort",
                  "entropy_dx_to_dx.png")

display(Markdown("""
#### Probability vs Volume — Transition Confidence
"""))
plot_scatter(df_dx_dx, "current_dx", "current_dx_desc", "next_dx",
             "Transition Confidence — Diagnosis to Diagnosis by Cohort",
             "scatter_dx_to_dx.png")

display(Markdown("""
#### Transition Network — Top 20 Paths
"""))
for cohort in COHORTS:
    plot_network(df_dx_dx, "current_dx", "current_dx_desc", "next_dx", "next_dx_desc",
                 cohort, "Diagnosis to Diagnosis Network",
                 f"network_dx_to_dx_{cohort}.png")
