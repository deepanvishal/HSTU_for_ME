# ============================================================
# Block 9 — Specialty Deployment Simulation (Aug 2024)
# For each (trigger_dx, rank-1 predicted specialty):
#   - Times predicted
#   - Avg prediction score
#   - Times actually correct
#   - Accuracy
# Source: trigger_scores (BERT4Rec, T30, null-filtered)
# Requires: client, DS
# ============================================================
from IPython.display import display, Markdown

REF_MONTH_START = "2024-08-01"
REF_MONTH_END   = "2024-08-31"
BEST_MODEL      = "BERT4Rec"

sim = client.query(f"""
    WITH parsed AS (
        SELECT
            trigger_dx
            ,SPLIT(top5_predictions, '|')[OFFSET(0)]    AS rank1_specialty
            ,CAST(SPLIT(top5_scores, '|')[OFFSET(0)] AS FLOAT64) AS rank1_score
            ,true_labels
            ,IF(CONCAT('|', true_labels, '|')
                LIKE CONCAT('%|', SPLIT(top5_predictions, '|')[OFFSET(0)], '|%'),
                1, 0)                                    AS rank1_correct
        FROM `{DS}.A870800_gen_rec_trigger_scores`
        WHERE time_bucket = 'T0_30'
          AND model = '{BEST_MODEL}'
          AND true_labels IS NOT NULL AND true_labels != ''
          AND top5_predictions IS NOT NULL AND top5_predictions != ''
          AND CAST(trigger_date AS DATE) BETWEEN '{REF_MONTH_START}' AND '{REF_MONTH_END}'
    )
    SELECT
        trigger_dx
        ,rank1_specialty
        ,COUNT(*)                                        AS times_predicted
        ,ROUND(AVG(rank1_score), 4)                      AS avg_score
        ,SUM(rank1_correct)                              AS times_correct
        ,ROUND(SAFE_DIVIDE(SUM(rank1_correct), COUNT(*)), 4) AS accuracy
    FROM parsed
    GROUP BY 1, 2
    ORDER BY times_predicted DESC
""").to_dataframe()

# CCSR name lookup
ccsr = client.query(f"""
    SELECT DISTINCT trigger_dx, trigger_ccsr_desc
    FROM `{DS}.A870800_gen_rec_markov_train`
    WHERE trigger_ccsr_desc IS NOT NULL
""").to_dataframe()

spec_names = client.query(f"""
    SELECT DISTINCT next_specialty AS code, next_specialty_desc AS name
    FROM `{DS}.A870800_gen_rec_markov_train`
    WHERE next_specialty IS NOT NULL
""").to_dataframe()

sim = sim.merge(ccsr, on="trigger_dx", how="left")
sim = sim.merge(spec_names, left_on="rank1_specialty", right_on="code", how="left")
sim["dx_display"] = sim["trigger_ccsr_desc"].fillna(sim["trigger_dx"])
sim["spec_display"] = sim["name"].fillna(sim["rank1_specialty"])

# Summary
total_pred = sim["times_predicted"].sum()
total_correct = sim["times_correct"].sum()
distinct_pathways = len(sim)

display(Markdown(f"""
### Specialty Deployment Simulation — August 2024

| Metric | Value |
|---|---|
| Reference month | Aug 2024 |
| Model | {BEST_MODEL} |
| Total triggers scored | {total_pred:,} |
| Rank-1 correct | {total_correct:,} |
| Rank-1 accuracy | {total_correct/total_pred*100:.1f}% |
| Distinct (dx → specialty) pathways | {distinct_pathways:,} |
"""))

# Top 30 by volume
display(Markdown("### Top 30 Pathways by Volume"))
display(sim.head(30)[["dx_display", "spec_display", "times_predicted",
                       "avg_score", "times_correct", "accuracy"]])

# Top 30 by accuracy (min 20 predictions)
display(Markdown("### Top 30 Pathways by Accuracy (min 20 predictions)"))
high_acc = sim[sim["times_predicted"] >= 20].sort_values("accuracy", ascending=False)
display(high_acc.head(30)[["dx_display", "spec_display", "times_predicted",
                            "avg_score", "times_correct", "accuracy"]])

print("Block 9 done.")
