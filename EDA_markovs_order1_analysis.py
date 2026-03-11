from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

client = bigquery.Client(project="anbc-hcb-dev")

# ── 1. FULL TABLE with DX and Specialty descriptions ─────────────────────────
query_main = """
SELECT
    t.current_dx
    ,COALESCE(dx.icd9_dx_description, t.current_dx)  AS dx_description
    ,t.next_specialty
    ,COALESCE(sp.global_lookup_desc, t.next_specialty) AS specialty_description
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,t.conditional_probability
    ,t.conditional_entropy
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_specialty_order1` t
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx
    ON t.current_dx = dx.icd9_dx_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
    ON t.next_specialty = sp.global_lookup_cd
    AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
WHERE t.transition_count >= 100
  AND t.unique_members >= 30
  AND t.conditional_entropy IS NOT NULL
"""
df = client.query(query_main).to_dataframe()

# ── 2. TOP 20 TRANSITIONS ─────────────────────────────────────────────────────
top20 = (
    df.groupby(["current_dx", "dx_description", "next_specialty", "specialty_description"], as_index=False)
    ["transition_count"].sum()
    .sort_values("transition_count", ascending=False)
    .head(20)
)
print(top20[["dx_description", "specialty_description", "transition_count"]].to_string(index=False))

# ── 3. HEATMAP — top 20 DX × top 15 specialties ──────────────────────────────
top_dx   = top20["current_dx"].unique()
top_spec = (
    df.groupby("next_specialty")["transition_count"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .index.tolist()
)

heat_df = (
    df[df["current_dx"].isin(top_dx) & df["next_specialty"].isin(top_spec)]
    .groupby(["dx_description", "specialty_description"], as_index=False)["transition_count"]
    .sum()
    .pivot(index="dx_description", columns="specialty_description", values="transition_count")
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(18, 12))
sns.heatmap(
    heat_df,
    ax=ax,
    cmap="YlOrRd",
    linewidths=0.3,
    linecolor="lightgrey",
    fmt=".0f",
    annot=True,
    annot_kws={"size": 7},
    cbar_kws={"label": "Transition Count"}
)
ax.set_title("Top 20 DX → Top 15 Specialties (Transition Volume)", fontsize=13, pad=12)
ax.set_xlabel("Next Specialty")
ax.set_ylabel("Current DX")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("heatmap_dx_to_specialty.png", dpi=150)
plt.show()

# ── 4. CCSR DOMAIN ENTROPY ────────────────────────────────────────────────────
query_ccsr = """
SELECT
    t.current_dx
    ,COALESCE(dx.icd9_dx_description, t.current_dx)  AS dx_description
    ,c.ccsr_category
    ,c.ccsr_category_description
    ,t.member_segment
    ,t.transition_count
    ,t.conditional_entropy
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_specialty_order1` t
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx
    ON t.current_dx = dx.icd9_dx_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mw_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` c
    ON REPLACE(t.current_dx, '.', '') = c.icd_10_cm_code
WHERE t.transition_count >= 100
  AND t.unique_members >= 30
  AND t.conditional_entropy IS NOT NULL
  AND c.ccsr_category IS NOT NULL
"""
df_ccsr = client.query(query_ccsr).to_dataframe()

domain_entropy = (
    df_ccsr.groupby(["ccsr_category", "ccsr_category_description"])
    .apply(lambda g: pd.Series({
        "weighted_avg_entropy": np.average(g["conditional_entropy"], weights=g["transition_count"]),
        "total_transitions": g["transition_count"].sum()
    }))
    .reset_index()
    .sort_values("weighted_avg_entropy")
)

top_domains = (
    domain_entropy.nlargest(30, "total_transitions")
    .sort_values("weighted_avg_entropy")
)

fig, ax = plt.subplots(figsize=(14, 10))
ax.barh(
    top_domains["ccsr_category_description"],
    top_domains["weighted_avg_entropy"],
    color=plt.cm.RdYlGn_r(
        (top_domains["weighted_avg_entropy"] - top_domains["weighted_avg_entropy"].min())
        / (top_domains["weighted_avg_entropy"].max() - top_domains["weighted_avg_entropy"].min())
    )
)
ax.set_xlabel("Weighted Average Conditional Entropy", fontsize=11)
ax.set_title("DX→Specialty Signal Strength by Clinical Domain\n(lower = more predictable routing)", fontsize=12)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("entropy_by_domain.png", dpi=150)
plt.show()

print("\nTop 10 most predictable domains:")
print(domain_entropy.sort_values("weighted_avg_entropy").head(10)[
    ["ccsr_category", "ccsr_category_description", "weighted_avg_entropy", "total_transitions"]
].to_string(index=False))



# ── 5. PROBABILITY VS VOLUME SCATTER ─────────────────────────────────────────
scatter_df = df[df["current_dx"].isin(top_dx)].copy()

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=scatter_df,
    x="transition_count",
    y="conditional_probability",
    hue="dx_description",
    size="unique_members",
    sizes=(20, 400),
    alpha=0.6,
    palette="viridis"
)
plt.axhline(0.5, color='red', linestyle='--', alpha=0.5)
plt.title("Transition Confidence: Volume vs Probability", fontsize=14)
plt.xlabel("Total Transition Count", fontsize=12)
plt.ylabel("Conditional Probability", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Diagnosis", fontsize=7)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("scatter_volume_vs_probability.png", dpi=150)
plt.show()

# ── 6. NETWORKX DIRECTED GRAPH ────────────────────────────────────────────────
import networkx as nx

G = nx.DiGraph()
for _, row in top20.iterrows():
    G.add_edge(
        row["dx_description"],
        row["specialty_description"],
        weight=row["transition_count"]
    )

dx_nodes      = top20["dx_description"].unique().tolist()
spec_nodes    = top20["specialty_description"].unique().tolist()
node_colors   = ["#4C9BE8" if n in dx_nodes else "#F4845F" for n in G.nodes()]
weights       = [G[u][v]["weight"] / top20["transition_count"].max() * 8 for u, v in G.edges()]

plt.figure(figsize=(16, 11))
pos = nx.spring_layout(G, k=3.0, seed=42)
nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_colors, alpha=0.85)
nx.draw_networkx_edges(G, pos, width=weights, edge_color="grey", arrowsize=18, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold")
plt.title("Diagnosis → Specialty Flow Network (Top 20 Transitions)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig("network_dx_to_specialty.png", dpi=150)
plt.show()
