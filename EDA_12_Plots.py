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
COHORTS = ["Adult_Female", "Adult_Male", "Senior", "Children"]
WINDOW  = "T60"

df = client.query(f"""
    SELECT * FROM `{DATASET}.A870800_gen_rec_f_track1_penetration`
    WHERE time_window = '{WINDOW}'
""").to_dataframe()

for col in ["penetration_rate", "binary_entropy", "members_visited",
            "total_members", "visit_count"]:
    df[col] = df[col].astype(float)


# ── TOP 10 BY BINARY ENTROPY ──────────────────────────────────────────────────
def top10_by_entropy(df):
    return (
        df.groupby(["trigger_dx", "trigger_dx_desc", "visit_specialty",
                    "visit_specialty_desc", "member_segment"], as_index=False)
        .agg(members_visited=("members_visited", "sum"),
             total_members=("total_members", "mean"),
             penetration_rate=("penetration_rate", "mean"),
             binary_entropy=("binary_entropy", "mean"))
        .query("penetration_rate >= 0.10")
        .sort_values("binary_entropy")
    )


# ── HEATMAP — BINARY ENTROPY ──────────────────────────────────────────────────
def plot_heatmap(df, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()
    for i, cohort in enumerate(COHORTS):
        ax = axes[i]
        sub = df[df["member_segment"] == cohort]
        top_dx   = sub.groupby("trigger_dx")["members_visited"].sum().nlargest(12).index
        top_spec = sub.groupby("visit_specialty")["members_visited"].sum().nlargest(12).index
        sub = sub[sub["trigger_dx"].isin(top_dx) & sub["visit_specialty"].isin(top_spec)]
        pivot = (
            sub.groupby(["trigger_dx_desc", "visit_specialty_desc"], as_index=False)
            ["binary_entropy"].mean()
            .pivot_table(index="trigger_dx_desc", columns="visit_specialty_desc",
                         values="binary_entropy", aggfunc="mean")
            .fillna(0).astype(float)
        )
        if pivot.empty:
            ax.set_title(f"{cohort} — No Data")
            continue
        sns.heatmap(pivot, ax=ax, cmap="RdYlGn_r", linewidths=0.3, linecolor="lightgrey",
                    annot=True, fmt=".2f", annot_kws={"size": 7},
                    cbar_kws={"label": "Binary Entropy"})
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Visited Specialty", fontsize=9)
        ax.set_ylabel("Trigger Diagnosis", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


# ── ENTROPY BAR CHART — TOP 5 DIAGNOSES ──────────────────────────────────────
def plot_entropy_bar(df, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()
    for i, cohort in enumerate(COHORTS):
        ax = axes[i]
        sub = df[df["member_segment"] == cohort]
        top5_dx = sub.groupby("trigger_dx")["members_visited"].sum().nlargest(5).index
        sub = sub[sub["trigger_dx"].isin(top5_dx)].copy()
        if sub.empty:
            ax.set_title(f"{cohort} — No Data")
            continue
        plot_data = (
            sub.groupby(["trigger_dx_desc", "visit_specialty_desc"], as_index=False)
            .agg(binary_entropy=("binary_entropy", "mean"),
                 penetration_rate=("penetration_rate", "mean"))
            .query("penetration_rate >= 0.10")
            .sort_values(["trigger_dx_desc", "binary_entropy"])
        )
        colors = plt.cm.RdYlGn_r(
            plot_data["binary_entropy"] / 0.693
        )
        bars = ax.barh(
            plot_data["visit_specialty_desc"] + " | " + plot_data["trigger_dx_desc"].str[:25],
            plot_data["binary_entropy"],
            color=colors,
            alpha=0.85
        )
        ax.axvline(0.693, color="red", linestyle="--", alpha=0.5, label="Max Uncertainty")
        ax.set_xlabel("Binary Entropy", fontsize=9)
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


# ── BIPARTITE NETWORK — EDGE THICKNESS BY PENETRATION, COLOR BY ENTROPY ──────
def plot_network_bipartite(df, cohort, title, filename, top_n=15):
    sub = (
        df[df["member_segment"] == cohort]
        .groupby(["trigger_dx_desc", "visit_specialty_desc"], as_index=False)
        .agg(penetration_rate=("penetration_rate", "mean"),
             binary_entropy=("binary_entropy", "mean"))
        .query("penetration_rate >= 0.10")
        .sort_values("binary_entropy")
        .head(top_n)
    )
    if sub.empty:
        print(f"No data for {cohort}")
        return

    G = nx.DiGraph()
    for _, row in sub.iterrows():
        G.add_node(row["trigger_dx_desc"],      layer=0)
        G.add_node(row["visit_specialty_desc"], layer=1)
        G.add_edge(row["trigger_dx_desc"], row["visit_specialty_desc"],
                   weight=row["penetration_rate"],
                   entropy=row["binary_entropy"])

    left_nodes  = sub["trigger_dx_desc"].unique().tolist()
    right_nodes = sub["visit_specialty_desc"].unique().tolist()

    pos = {}
    for j, n in enumerate(left_nodes):
        pos[n] = (0, j * 2.5)
    for j, n in enumerate(right_nodes):
        pos[n] = (5, j * 2.5)

    max_weight  = max([G[u][v]["weight"] for u, v in G.edges()], default=1)
    edge_widths = [G[u][v]["weight"] / max_weight * 12 + 2 for u, v in G.edges()]
    edge_colors = [plt.cm.RdYlGn_r(G[u][v]["entropy"] / 0.693) for u, v in G.edges()]
    node_colors = ["#4C9BE8" if n in left_nodes else "#F4845F" for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(24, 18))
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                           arrowsize=30, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.1",
                           min_source_margin=30, min_target_margin=30, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    blue_patch   = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C9BE8",
                               markersize=12, label="Trigger Diagnosis")
    orange_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#F4845F",
                               markersize=12, label="Visited Specialty")
    thick_line   = plt.Line2D([0], [0], color="grey", linewidth=4,
                               label="Arrow thickness = Penetration Rate")
    color_line   = plt.Line2D([0], [0], color="green", linewidth=4,
                               label="Arrow color = Entropy (green = predictable)")
    ax.legend(handles=[blue_patch, orange_patch, thick_line, color_line],
              loc="upper left", fontsize=9)
    ax.set_title(f"{title} — {cohort}", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# T30 ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown(f"""
---
## Time Window Analysis — {WINDOW}

Given a member's first encounter with a new diagnosis, this section identifies
which specialties are visited within {WINDOW} days — and how predictably.

**Penetration rate** — percentage of members with that trigger diagnosis
who visited a given specialty within {WINDOW} days.

**Binary entropy** — how uncertain we are about whether a member will visit
that specialty. Low entropy means the visit is near-certain or near-absent.
High entropy means roughly half of members visit — member-level factors
are likely driving the decision.

Only specialties with penetration rate above 10% are included.

---
"""))

top10 = top10_by_entropy(df)

display(Markdown(f"### Top 10 — Lowest Binary Entropy with Penetration Rate above 10% — {WINDOW}"))
display(top10.head(10)[[
    "trigger_dx_desc", "visit_specialty_desc", "member_segment",
    "members_visited", "total_members", "penetration_rate", "binary_entropy"
]].rename(columns={
    "trigger_dx_desc": "Trigger Diagnosis",
    "visit_specialty_desc": "Visited Specialty",
    "member_segment": "Cohort",
    "members_visited": "Members Visited",
    "total_members": "Total Members",
    "penetration_rate": "Penetration Rate",
    "binary_entropy": "Binary Entropy"
}).reset_index(drop=True))

display(Markdown(f"""
#### Heatmap — Binary Entropy by Diagnosis and Specialty

Each cell shows the binary entropy for a given diagnosis to specialty pair within {WINDOW} days.
Green cells are highly predictable — the visit almost always or almost never happens.
Red cells are uncertain — member level factors are driving the routing.
"""))
plot_heatmap(df,
             f"Binary Entropy — Diagnosis to Specialty — {WINDOW} by Cohort",
             f"heatmap_{WINDOW}.png")

display(Markdown(f"""
#### Entropy Bar Chart — Top 5 Diagnoses

Each bar shows the binary entropy for a specialty within a given trigger diagnosis.
Bars close to zero are highly predictable visits.
Bars near 0.693 (red line) are the most uncertain — worth investigating what
member level factors drive the routing decision.
"""))
plot_entropy_bar(df,
                 f"Binary Entropy by Specialty — {WINDOW} by Cohort",
                 f"entropy_bar_{WINDOW}.png")

display(Markdown(f"""
#### Transition Network — Top 15 Paths

Blue nodes are trigger diagnoses. Orange nodes are visited specialties.
Arrow thickness reflects penetration rate — thicker means more members visited.
Arrow color reflects binary entropy — green means predictable, red means uncertain.
"""))
for cohort in COHORTS:
    plot_network_bipartite(df, cohort,
                           f"Diagnosis to Specialty Network — {WINDOW}",
                           f"network_{WINDOW}_{cohort}.png")
