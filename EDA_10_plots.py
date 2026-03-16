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

df = client.query(f"""
    SELECT * FROM `{DATASET}.A870800_gen_rec_f_dx_to_specialty_order2`
    WHERE transition_count >= {THRESHOLD}
      AND conditional_entropy IS NOT NULL
""").to_dataframe()


def top10_transitions(df):
    return (
        df.groupby(["current_dx_v1", "current_dx_v1_desc",
                    "current_dx_v2", "current_dx_v2_desc",
                    "next_specialty", "next_specialty_desc", "member_segment"], as_index=False)
        .agg(transition_count=("transition_count", "sum"),
             unique_members=("unique_members", "sum"),
             conditional_probability=("conditional_probability", "mean"),
             conditional_entropy=("conditional_entropy", "mean"))
        .sort_values("transition_count", ascending=False)
    )


def plot_heatmap(df, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()
    for i, cohort in enumerate(COHORTS):
        ax = axes[i]
        sub = df[df["member_segment"] == cohort]
        top_v1   = sub.groupby("current_dx_v1")["transition_count"].sum().nlargest(12).index
        top_spec = sub.groupby("next_specialty")["transition_count"].sum().nlargest(12).index
        sub = sub[sub["current_dx_v1"].isin(top_v1) & sub["next_specialty"].isin(top_spec)]
        pivot = (
            sub.groupby(["current_dx_v1_desc", "next_specialty_desc"], as_index=False)
            ["transition_count"].sum()
            .pivot_table(index="current_dx_v1_desc", columns="next_specialty_desc",
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
        ax.set_ylabel("Trigger Diagnosis", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ccsr_entropy(df, title, filename):
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    axes = axes.flatten()
    for i, cohort in enumerate(COHORTS):
        ax = axes[i]
        sub = df[df["member_segment"] == cohort]
        domain = (
            sub.groupby(["current_ccsr_v1", "current_ccsr_v1_desc"], as_index=False)
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
        ax.barh(domain["current_ccsr_v1_desc"], domain["weighted_avg_entropy"], color=colors)
        ax.set_xlabel("Weighted Average Entropy", fontsize=9)
        ax.set_title(f"{cohort}", fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.invert_yaxis()
        plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


def plot_network_3layer(df, cohort, title, filename, top_n=15):
    sub = (
        df[df["member_segment"] == cohort]
        .groupby(["current_dx_v1_desc", "current_dx_v2_desc", "next_specialty_desc"],
                 as_index=False)["transition_count"].sum()
        .sort_values("transition_count", ascending=False)
        .head(top_n)
    )
    if sub.empty:
        print(f"No data for {cohort}")
        return

    G = nx.DiGraph()
    for _, row in sub.iterrows():
        v1_label   = row["current_dx_v1_desc"] + " (V1)"
        v2_label   = row["current_dx_v2_desc"] + " (V2)"
        spec_label = row["next_specialty_desc"]

        G.add_node(v1_label,   layer=0)
        G.add_node(v2_label,   layer=1)
        G.add_node(spec_label, layer=2)
        G.add_edge(v1_label,   v2_label,   weight=row["transition_count"])
        G.add_edge(v2_label,   spec_label, weight=row["transition_count"])

    layer0 = [n for n, d in G.nodes(data=True) if d.get("layer") == 0]
    layer1 = [n for n, d in G.nodes(data=True) if d.get("layer") == 1]
    layer2 = [n for n, d in G.nodes(data=True) if d.get("layer") == 2]

    pos = {}
    for j, n in enumerate(layer0):
        pos[n] = (0, j * 2.5)
    for j, n in enumerate(layer1):
        pos[n] = (4, j * 2.5)
    for j, n in enumerate(layer2):
        pos[n] = (8, j * 2.5)

    max_weight  = max([G[u][v]["weight"] for u, v in G.edges()], default=1)
    edge_widths = [G[u][v]["weight"] / max_weight * 12 + 2 for u, v in G.edges()]
    node_colors = ["#4C9BE8" if G.nodes[n].get("layer") == 0
                   else "#F4845F" if G.nodes[n].get("layer") == 1
                   else "#5DBE7E" for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(28, 18))
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color=node_colors, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="#555555",
                           arrowsize=30, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.1",
                           min_source_margin=30, min_target_margin=30, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    blue_patch   = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C9BE8",
                               markersize=12, label="Trigger Diagnosis (V1)")
    orange_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#F4845F",
                               markersize=12, label="Second Visit Diagnosis (V2)")
    green_patch  = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#5DBE7E",
                               markersize=12, label="Next Specialty")
    ax.legend(handles=[blue_patch, orange_patch, green_patch], loc="upper left", fontsize=10)
    ax.set_title(f"{title} — {cohort}", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# ORDER 2 — DIAGNOSIS CODE TO PROVIDER SPECIALTY
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Order 2 — Diagnosis Code to Provider Specialty

Given the trigger diagnosis at the first encounter and the diagnosis at the second visit,
how predictably does that pair route the member to a specific provider specialty?

Order 2 conditions the prediction on two visits — the trigger diagnosis and the
intermediate V2 diagnosis — giving a more specific and reliable signal than Order 1 alone.

---
"""))

top10 = top10_transitions(df)

display(Markdown("### Top 10 Transitions — Trigger Diagnosis and V2 Diagnosis to Provider Specialty"))
display(top10.head(10)[[
    "current_dx_v1_desc", "current_dx_v2_desc", "next_specialty_desc",
    "member_segment", "transition_count", "unique_members",
    "conditional_probability", "conditional_entropy"
]].rename(columns={
    "current_dx_v1_desc": "Trigger Diagnosis",
    "current_dx_v2_desc": "V2 Diagnosis",
    "next_specialty_desc": "Next Specialty",
    "member_segment": "Cohort",
    "transition_count": "Transition Count",
    "unique_members": "Unique Members",
    "conditional_probability": "Probability",
    "conditional_entropy": "Entropy"
}).reset_index(drop=True))

display(Markdown("""
#### Heatmap — Trigger Diagnosis vs Next Specialty

Each cell shows how many times a given trigger diagnosis led to a given specialty
after passing through V2. Darker cells indicate higher volume transitions.
"""))
plot_heatmap(df,
             "Order 2 Diagnosis to Specialty — Transition Volume by Cohort",
             "heatmap_order2_dx_to_specialty.png")

display(Markdown("""
#### Clinical Domain Entropy — Top 10 Domains by Volume

Each bar shows how predictable specialty routing is for a given clinical domain
when conditioning on both the trigger diagnosis and the V2 diagnosis.
Green bars indicate strong, predictable routing. Red bars indicate scattered routing.
"""))
plot_ccsr_entropy(df,
                  "Clinical Domain Entropy — Order 2 Diagnosis to Specialty by Cohort",
                  "entropy_order2_dx_to_specialty.png")

display(Markdown("""
#### Transition Network — Three Layer Chain

Blue nodes are trigger diagnoses at the first encounter.
Orange nodes are diagnoses observed at the second visit.
Green nodes are the predicted provider specialties.
Arrow thickness reflects transition volume.
"""))
for cohort in COHORTS:
    plot_network_3layer(df, cohort,
                        "Order 2 Diagnosis Chain to Specialty",
                        f"network_order2_dx_to_specialty_{cohort}.png")
