# ── EDA_T30.py ────────────────────────────────────────────────────────────────
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
WINDOW  = "T30"

df = client.query(f"""
    SELECT * FROM `{DATASET}.A870800_gen_rec_f_track1_penetration`
    WHERE time_window = '{WINDOW}'
""").to_dataframe()


def top10_by_penetration(df):
    return (
        df.groupby(["trigger_dx", "trigger_dx_desc", "visit_specialty",
                    "visit_specialty_desc", "member_segment"], as_index=False)
        .agg(members_visited=("members_visited", "sum"),
             visit_count=("visit_count", "sum"),
             total_members=("total_members", "mean"),
             penetration_rate=("penetration_rate", "mean"))
        .sort_values("penetration_rate", ascending=False)
    )


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
            ["penetration_rate"].mean()
            .pivot_table(index="trigger_dx_desc", columns="visit_specialty_desc",
                         values="penetration_rate", aggfunc="mean")
            .fillna(0).astype(float)
        )
        if pivot.empty:
            ax.set_title(f"{cohort} — No Data")
            continue
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="lightgrey",
                    annot=True, fmt=".2f", annot_kws={"size": 7},
                    cbar_kws={"label": "Penetration Rate"})
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Visited Specialty", fontsize=9)
        ax.set_ylabel("Trigger Diagnosis", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


def plot_care_bundle(df, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()
    for i, cohort in enumerate(COHORTS):
        ax = axes[i]
        sub = df[df["member_segment"] == cohort]
        top5_dx = sub.groupby("trigger_dx")["members_visited"].sum().nlargest(5).index
        sub = sub[sub["trigger_dx"].isin(top5_dx)]
        pivot = (
            sub.groupby(["trigger_dx_desc", "visit_specialty_desc"], as_index=False)
            ["penetration_rate"].mean()
            .pivot_table(index="trigger_dx_desc", columns="visit_specialty_desc",
                         values="penetration_rate", aggfunc="mean")
            .fillna(0).astype(float)
        )
        if pivot.empty:
            ax.set_title(f"{cohort} — No Data")
            continue
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", legend=True)
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Trigger Diagnosis", fontsize=9)
        ax.set_ylabel("Penetration Rate", fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.legend(loc="upper right", fontsize=6, ncol=2)
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


def plot_network_bipartite(df, cohort, title, filename, top_n=15):
    sub = (
        df[df["member_segment"] == cohort]
        .groupby(["trigger_dx_desc", "visit_specialty_desc"], as_index=False)
        ["penetration_rate"].mean()
        .sort_values("penetration_rate", ascending=False)
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
                   weight=row["penetration_rate"])

    left_nodes  = sub["trigger_dx_desc"].unique().tolist()
    right_nodes = sub["visit_specialty_desc"].unique().tolist()

    pos = {}
    for j, n in enumerate(left_nodes):
        pos[n] = (0, j * 2.5)
    for j, n in enumerate(right_nodes):
        pos[n] = (4, j * 2.5)

    max_weight  = max([G[u][v]["weight"] for u, v in G.edges()], default=1)
    edge_widths = [G[u][v]["weight"] / max_weight * 12 + 2 for u, v in G.edges()]
    node_colors = ["#4C9BE8" if n in left_nodes else "#F4845F" for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(22, 16))
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="#555555",
                           arrowsize=30, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.1",
                           min_source_margin=30, min_target_margin=30, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    blue_patch   = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C9BE8",
                               markersize=12, label="Trigger Diagnosis")
    orange_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#F4845F",
                               markersize=12, label="Visited Specialty")
    ax.legend(handles=[blue_patch, orange_patch], loc="upper left", fontsize=10)
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

Given a member's first encounter with a new diagnosis, this section shows which
specialties were visited within {WINDOW} days — and what percentage of members
with that diagnosis made each visit.

Penetration rate = members who visited that specialty within {WINDOW} days
divided by total members with that trigger diagnosis.

A high penetration rate means that specialty visit is a near-certain part of
the care bundle for that diagnosis.

---
"""))

top10 = top10_by_penetration(df)

display(Markdown(f"### Top 10 Transitions by Penetration Rate — {WINDOW}"))
display(top10.head(10)[[
    "trigger_dx_desc", "visit_specialty_desc", "member_segment",
    "members_visited", "total_members", "penetration_rate"
]].rename(columns={
    "trigger_dx_desc": "Trigger Diagnosis",
    "visit_specialty_desc": "Visited Specialty",
    "member_segment": "Cohort",
    "members_visited": "Members Visited",
    "total_members": "Total Members",
    "penetration_rate": "Penetration Rate"
}).reset_index(drop=True))

display(Markdown(f"""
#### Heatmap — Trigger Diagnosis vs Visited Specialty

Each cell shows the penetration rate — what proportion of members with a given
diagnosis visited a given specialty within {WINDOW} days.
Darker cells indicate higher penetration.
"""))
plot_heatmap(df,
             f"Diagnosis to Specialty Penetration Rate — {WINDOW} by Cohort",
             f"heatmap_{WINDOW}.png")

display(Markdown(f"""
#### Care Bundle Composition — Top 5 Diagnoses

Each bar represents one trigger diagnosis. The stacked segments show what
proportion of members visited each specialty within {WINDOW} days.
This reveals the full care bundle — not just the first visit.
"""))
plot_care_bundle(df,
                 f"Care Bundle Composition — {WINDOW} by Cohort",
                 f"bundle_{WINDOW}.png")

display(Markdown(f"""
#### Transition Network — Top 15 Paths by Penetration Rate

Blue nodes are trigger diagnoses. Orange nodes are visited specialties.
Arrow thickness reflects penetration rate — thicker means more members followed that path.
"""))
for cohort in COHORTS:
    plot_network_bipartite(df, cohort,
                           f"Diagnosis to Specialty Network — {WINDOW}",
                           f"network_{WINDOW}_{cohort}.png")
