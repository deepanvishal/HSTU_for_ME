from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, Markdown

client = bigquery.Client(project="anbc-hcb-dev")
DATASET = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"

df = client.query(f"""
    SELECT * FROM `{DATASET}.A870800_gen_rec_f_markov_predictions`
""").to_dataframe()


def compute_metrics(df, k):
    results = []
    for (member_id, trigger_date, trigger_dx, v2_dx, time_window, member_segment), group in df.groupby(
        ["member_id", "trigger_date", "trigger_dx", "v2_dx", "time_window", "member_segment"]
    ):
        actual    = group["actual_specialty"].iloc[0]
        top_k     = group.sort_values("specialty_rank").head(k)["predicted_specialty"].tolist()

        hit       = 1 if actual in top_k else 0
        precision = hit / k
        recall    = hit
        rank      = top_k.index(actual) + 1 if actual in top_k else None
        dcg       = 1 / np.log2(rank + 1) if rank else 0
        idcg      = 1 / np.log2(2)
        ndcg      = dcg / idcg

        results.append({
            "member_id":      member_id,
            "trigger_dx":     trigger_dx,
            "v2_dx":          v2_dx,
            "time_window":    time_window,
            "member_segment": member_segment,
            "k":              k,
            "hit":            hit,
            "precision":      precision,
            "recall":         recall,
            "ndcg":           ndcg
        })
    return pd.DataFrame(results)


all_metrics = pd.concat([compute_metrics(df, k) for k in [1, 3, 5]], ignore_index=True)

summary = (
    all_metrics
    .groupby(["time_window", "member_segment", "k"], as_index=False)
    .agg(
        hit_rate=("hit", "mean"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        ndcg=("ndcg", "mean"),
        total_members=("member_id", "nunique")
    )
    .round(4)
    .sort_values(["time_window", "member_segment", "k"])
)

display(Markdown("""
---
## Markov Baseline — Model Metrics

Transition probabilities from pre-2024 claims used as a naive predictor.
Test set covers triggers from January 2024 onwards.

This establishes the performance floor — any sequential model must beat these numbers
to justify the added complexity.

---
"""))

for window in ["T30", "T60", "T180"]:
    display(Markdown(f"### {window}"))
    display(
        summary[summary["time_window"] == window][[
            "member_segment", "k", "hit_rate", "precision", "recall", "ndcg", "total_members"
        ]].rename(columns={
            "member_segment": "Cohort",
            "k": "K",
            "hit_rate": "Hit@K",
            "precision": "Precision@K",
            "recall": "Recall@K",
            "ndcg": "NDCG@K",
            "total_members": "Test Members"
        }).reset_index(drop=True)
    )


def plot_metrics(summary, filename):
    windows   = ["T30", "T60", "T180"]
    metrics   = ["hit_rate", "precision", "recall", "ndcg"]
    labels    = ["Hit@K", "Precision@K", "Recall@K", "NDCG@K"]
    k_values  = [1, 3, 5]
    colors    = ["#4C9BE8", "#F4845F", "#5DBE7E"]

    fig, axes = plt.subplots(len(metrics), len(windows), figsize=(24, 20))

    for col, window in enumerate(windows):
        sub = summary[summary["time_window"] == window]
        for row, (metric, label) in enumerate(zip(metrics, labels)):
            ax = axes[row][col]
            for ki, (k, color) in enumerate(zip(k_values, colors)):
                k_sub = sub[sub["k"] == k].sort_values("member_segment")
                ax.plot(k_sub["member_segment"], k_sub[metric],
                        marker="o", linewidth=2, color=color, label=f"K={k}")
            ax.set_title(f"{label} — {window}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Cohort", fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            ax.legend(fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=7)

    fig.suptitle("Markov Baseline Metrics — T30, T60, T180 by Cohort and K",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()


display(Markdown("""
#### Metrics Visualization — All Windows and Cohorts

Each line represents a different K value.
Higher values indicate better predictive performance.
Compare across time windows to see if signal strengthens over longer windows.
"""))
plot_metrics(summary, "markov_baseline_metrics.png")
