# ============================================================
# EDA_any_visits.py
# Lens: Any Sequential Visits
# Source: A870800_gen_rec_f_dx_to_specialty_any
# ============================================================
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
LENS_LABEL = "Any Sequential Visits"
TABLE = "A870800_gen_rec_f_dx_to_specialty_any"

df = client.query(f"""
    SELECT * FROM `{DATASET}.{TABLE}`
    WHERE transition_count >= {THRESHOLD}
      AND conditional_entropy IS NOT NULL
""").to_dataframe()

for col in ["transition_count", "unique_members", "conditional_probability",
            "conditional_entropy", "avg_allowed_per_transition", "avg_allowed_per_member"]:
    df[col] = df[col].astype(float)


def plot_top10(df, title):
    top10 = (
        df.groupby(["current_dx", "current_dx_desc", "next_specialty",
                    "next_specialty_desc", "member_segment"], as_index=False)
        .agg(transition_count=("transition_count", "sum"),
             unique_members=("unique_members", "sum"),
             conditional_probability=("conditional_probability", "mean"),
             conditional_entropy=("conditional_entropy", "mean"),
             avg_allowed_per_transition=("avg_allowed_per_transition", "mean"),
             avg_allowed_per_member=("avg_allowed_per_member", "mean"))
        .sort_values("transition_count", ascending=False)
        .head(10)
    )
    display(Markdown(f"### Top 10 Transitions — {title}"))
    display(top10[[
        "current_dx_desc", "next_specialty_desc", "member_segment",
        "transition_count", "unique_members", "conditional_probability",
        "conditional_entropy", "avg_allowed_per_transition", "avg_allowed_per_member"
    ]].rename(columns={
        "current_dx_desc": "Diagnosis",
        "next_specialty_desc": "Next Specialty",
        "member_segment": "Cohort",
        "transition_count": "Transitions",
        "unique_members": "Unique Members",
        "conditional_probability": "Probability",
        "conditional_entropy": "Entropy",
        "avg_allowed_per_transition": "Avg Cost Per Visit",
        "avg_allowed_per_member": "Avg Cost Per Member"
    }).reset_index(drop=True))


def plot_entropy_bar(df, title, filename):
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


def plot_heatmap(df, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()
    for i, cohort in enumerate(COHORTS):
        ax = axes[i]
        sub = df[df["member_segment"] == cohort]
        top_dx = sub.groupby("current_dx")["transition_count"].sum().nlargest(12).index
        top_spec = sub.groupby("next_specialty")["transition_count"].sum().nlargest(12).index
        sub = sub[sub["current_dx"].isin(top_dx) & sub["next_specialty"].isin(top_spec)]
        pivot = (
            sub.groupby(["current_dx_desc", "next_specialty_desc"], as_index=False)
            ["transition_count"].sum()
            .pivot_table(index="current_dx_desc", columns="next_specialty_desc",
                         values="transition_count", aggfunc="sum")
            .fillna(0).astype(float)
        )
        if pivot.empty:
            ax.set_title(f"{cohort} — No Data")
            continue
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="lightgrey",
                    annot=True, fmt=".0f", annot_kws={"size": 7},
                    cbar_kws={"label": "Transition Count"})
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Next Specialty", fontsize=9)
        ax.set_ylabel("Diagnosis Code", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cost_heatmap(df, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()
    for i, cohort in enumerate(COHORTS):
        ax = axes[i]
        sub = df[df["member_segment"] == cohort]
        top_dx = sub.groupby("current_dx")["transition_count"].sum().nlargest(12).index
        top_spec = sub.groupby("next_specialty")["transition_count"].sum().nlargest(12).index
        sub = sub[sub["current_dx"].isin(top_dx) & sub["next_specialty"].isin(top_spec)]
        pivot = (
            sub.groupby(["current_dx_desc", "next_specialty_desc"], as_index=False)
            ["avg_allowed_per_member"].mean()
            .pivot_table(index="current_dx_desc", columns="next_specialty_desc",
                         values="avg_allowed_per_member", aggfunc="mean")
            .fillna(0).astype(float)
        )
        if pivot.empty:
            ax.set_title(f"{cohort} — No Data")
            continue
        sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="lightgrey",
                    annot=True, fmt=".0f", annot_kws={"size": 7},
                    cbar_kws={"label": "Avg Allowed Per Member ($)"})
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Next Specialty", fontsize=9)
        ax.set_ylabel("Diagnosis Code", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cost_entropy_scatter(df, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    axes = axes.flatten()
    for i, cohort in enumerate(COHORTS):
        ax = axes[i]
        sub = df[df["member_segment"] == cohort].copy()
        top_dx = sub.groupby("current_dx")["transition_count"].sum().nlargest(15).index
        sub = sub[sub["current_dx"].isin(top_dx)].copy()
        if sub.empty:
            ax.set_title(f"{cohort} — No Data")
            continue
        scatter = ax.scatter(
            sub["conditional_probability"],
            sub["avg_allowed_per_member"],
            c=sub["conditional_entropy"],
            s=sub["transition_count"] / sub["transition_count"].max() * 300 + 20,
            alpha=0.6,
            cmap="RdYlGn_r"
        )
        plt.colorbar(scatter, ax=ax, label="Entropy")
        ax.set_xlabel("Conditional Probability", fontsize=9)
        ax.set_ylabel("Avg Allowed Per Member ($)", fontsize=9)
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


def plot_network_bipartite(df, cohort, title, filename, top_n=15):
    sub = (
        df[df["member_segment"] == cohort]
        .groupby(["current_dx_desc", "next_specialty_desc"], as_index=False)
        .agg(transition_count=("transition_count", "sum"),
             conditional_entropy=("conditional_entropy", "mean"))
        .sort_values("transition_count", ascending=False)
        .head(top_n)
    )
    if sub.empty:
        print(f"No data for {cohort}")
        return

    G = nx.DiGraph()
    for _, row in sub.iterrows():
        G.add_node(row["current_dx_desc"],    layer=0)
        G.add_node(row["next_specialty_desc"], layer=1)
        G.add_edge(row["current_dx_desc"], row["next_specialty_desc"],
                   weight=row["transition_count"],
                   entropy=row["conditional_entropy"])

    left_nodes  = sub["current_dx_desc"].unique().tolist()
    right_nodes = sub["next_specialty_desc"].unique().tolist()

    pos = {}
    for j, n in enumerate(left_nodes):
        pos[n] = (0, j * 2.5)
    for j, n in enumerate(right_nodes):
        pos[n] = (5, j * 2.5)

    max_weight  = max([G[u][v]["weight"] for u, v in G.edges()], default=1)
    edge_widths = [G[u][v]["weight"] / max_weight * 12 + 2 for u, v in G.edges()]
    edge_colors = [plt.cm.RdYlGn_r(G[u][v]["entropy"] / 2.0) for u, v in G.edges()]
    node_colors = ["#4C9BE8" if n in left_nodes else "#F4845F" for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(24, 18))
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors,
                           arrowsize=30, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.1",
                           min_source_margin=30, min_target_margin=30, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    blue_patch   = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C9BE8",
                               markersize=12, label="Diagnosis Code")
    orange_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#F4845F",
                               markersize=12, label="Next Specialty")
    thick_line   = plt.Line2D([0], [0], color="grey", linewidth=4,
                               label="Arrow thickness = Transition Volume")
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
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown(f"""
---
# Diagnosis Code to Provider Specialty
## Lens: {LENS_LABEL}

This analysis examines transition patterns from diagnosis codes to provider specialties
across all sequential visit pairs in the data.

Each transition represents one member moving from a visit with a given diagnosis
to their next visit at a given specialty.

Entropy measures predictability — lower entropy means the diagnosis reliably
predicts the next specialty. Cost metrics show the financial value of each pathway.

---
"""))

plot_top10(df, LENS_LABEL)

display(Markdown("""
#### Clinical Domain Entropy — Top 10 Domains by Volume

Each bar shows how predictable specialty routing is for a given clinical domain.
Green bars indicate strong, predictable routing.
Red bars indicate scattered, unpredictable routing.
"""))
plot_entropy_bar(df,
                 f"Clinical Domain Entropy — {LENS_LABEL}",
                 f"entropy_bar_{TABLE}.png")

display(Markdown("""
#### Heatmap — Transition Volume by Diagnosis and Specialty

Each cell shows how many times members with a given diagnosis visited a given specialty next.
Darker cells indicate higher volume transitions.
"""))
plot_heatmap(df,
             f"Transition Volume — {LENS_LABEL}",
             f"heatmap_volume_{TABLE}.png")

display(Markdown("""
#### Heatmap — Average Cost Per Member by Diagnosis and Specialty

Each cell shows the average allowed amount per member for that diagnosis to specialty transition.
Darker cells indicate higher cost pathways.
Combined with entropy — high cost and low entropy transitions are the most actionable.
"""))
plot_cost_heatmap(df,
                  f"Average Cost Per Member — {LENS_LABEL}",
                  f"heatmap_cost_{TABLE}.png")

display(Markdown("""
#### Cost vs Probability Scatter

Each dot represents one diagnosis to specialty transition.
X axis — conditional probability — how predictable the transition is.
Y axis — average allowed amount per member — how expensive the pathway is.
Dot size reflects transition volume. Color reflects entropy.

Top right quadrant — high probability, high cost — most valuable predictable pathways.
"""))
plot_cost_entropy_scatter(df,
                          f"Cost vs Probability — {LENS_LABEL}",
                          f"scatter_cost_{TABLE}.png")

display(Markdown("""
#### Transition Network — Top 15 Paths

Blue nodes are diagnosis codes. Orange nodes are provider specialties.
Arrow thickness reflects transition volume.
Arrow color reflects entropy — green means predictable, red means uncertain.
"""))
for cohort in COHORTS:
    plot_network_bipartite(df, cohort,
                           f"Diagnosis to Specialty Network — {LENS_LABEL}",
                           f"network_{TABLE}_{cohort}.png")
