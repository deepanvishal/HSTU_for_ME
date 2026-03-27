# ============================================================
# Block 8d — Explicit Dollar Value of Provider Predictions
# No approximations. Every number traced to SFL.
#
# Layer A: Total scored triggers at T30
# Layer B: Correct predictions (hit_at_5 = 1)
# Layer C: Rank-1 predictions that are correct
# Layer D: Match rank-1 correct predictions to V2 visit in SFL → real $
# Layer E: Match ALL rank-1 predictions to V2 visit in SFL → total $
#
# Requires: client, DS
# Source: provider_eval_5pct, visits_qualified, claims SFL
# ============================================================
from IPython.display import display, Markdown

# ── Step 1-5: One query, explicit layers ──────────────────────
dollar_layers = client.query(f"""
    WITH

    -- Layer A: all scored triggers at T30
    scored AS (
        SELECT
            CAST(member_id AS STRING)                    AS member_id
            ,CAST(trigger_date AS STRING)                AS trigger_date
            ,CAST(trigger_dx AS STRING)                  AS trigger_dx
            ,hit_at_5
            ,SPLIT(top5_predictions, '|')[OFFSET(0)]     AS rank1_provider
            ,true_labels
        FROM `{DS}.A870800_gen_rec_provider_eval_5pct`
        WHERE time_bucket = 'T0_30'
          AND model = 'SASRec'
          AND (tp + fn) > 0
          AND top5_predictions IS NOT NULL
          AND top5_predictions != ''
    ),

    -- Layer C: rank-1 correct = rank1_provider appears in true_labels
    with_rank1_flag AS (
        SELECT
            *
            ,IF(CONCAT('|', true_labels, '|')
                LIKE CONCAT('%|', rank1_provider, '|%'), 1, 0) AS rank1_correct
        FROM scored
    ),

    -- Get V2 visit date per trigger from visits_qualified
    v2_dates AS (
        SELECT
            CAST(member_id AS STRING)                    AS member_id
            ,CAST(trigger_date AS STRING)                AS trigger_date
            ,CAST(trigger_dx AS STRING)                  AS trigger_dx
            ,MIN(visit_date)                             AS v2_date
        FROM `{DS}.A870800_gen_rec_visits_qualified`
        WHERE days_since_trigger > 0
          AND specialty_ctg_cd IS NOT NULL
        GROUP BY 1, 2, 3
    ),

    -- Join triggers to V2 dates
    with_v2 AS (
        SELECT
            s.*
            ,v.v2_date
        FROM with_rank1_flag s
        LEFT JOIN v2_dates v
            ON s.member_id = v.member_id
            AND s.trigger_date = v.trigger_date
            AND s.trigger_dx = v.trigger_dx
    ),

    -- Layer D: rank-1 correct + matched to SFL on member + v2_date + provider
    correct_with_spend AS (
        SELECT
            w.member_id
            ,w.trigger_date
            ,w.trigger_dx
            ,w.rank1_provider
            ,w.rank1_correct
            ,w.hit_at_5
            ,w.v2_date
            ,SUM(CAST(c.allowed_amt AS FLOAT64))         AS rank1_v2_spend
        FROM with_v2 w
        LEFT JOIN `{DS}.A870800_claims_gen_rec_2022_2025_sfl` c
            ON w.member_id = CAST(c.member_id AS STRING)
            AND w.v2_date = c.srv_start_dt
            AND w.rank1_provider = CAST(c.srv_prvdr_id AS STRING)
        WHERE w.v2_date IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5, 6, 7
    ),

    -- Aggregate layers
    summary AS (
        SELECT
            COUNT(*)                                     AS layer_a_total_triggers
            ,SUM(CAST(hit_at_5 AS INT64))                AS layer_b_hit5_correct
            ,SUM(rank1_correct)                          AS layer_c_rank1_correct
            ,SUM(IF(rank1_correct = 1 AND rank1_v2_spend IS NOT NULL, 1, 0))
                                                         AS layer_d_rank1_correct_matched
            ,SUM(IF(rank1_correct = 1, rank1_v2_spend, 0))
                                                         AS layer_d_rank1_correct_spend
            ,SUM(IF(rank1_v2_spend IS NOT NULL, 1, 0))   AS layer_e_all_rank1_matched
            ,SUM(COALESCE(rank1_v2_spend, 0))            AS layer_e_all_rank1_spend
        FROM correct_with_spend
    )

    SELECT * FROM summary
""").to_dataframe().iloc[0]

A = int(dollar_layers['layer_a_total_triggers'])
B = int(dollar_layers['layer_b_hit5_correct'])
C = int(dollar_layers['layer_c_rank1_correct'])
D_count = int(dollar_layers['layer_d_rank1_correct_matched'])
D_spend = float(dollar_layers['layer_d_rank1_correct_spend'])
E_count = int(dollar_layers['layer_e_all_rank1_matched'])
E_spend = float(dollar_layers['layer_e_all_rank1_spend'])

display(Markdown(f"""
### Provider Prediction — Dollar Value (Explicit, No Approximation)

| Layer | Description | Count | Allowed Amount |
|---|---|---|---|
| A | Total scored triggers (SASRec, T30) | {A:,} | — |
| B | Correct predictions (Hit@5 = 1) | {B:,} | — |
| C | Rank-1 prediction is correct | {C:,} | — |
| D | Rank-1 correct + matched to V2 in SFL | {D_count:,} | ${D_spend/1e9:.2f}B |
| E | ALL rank-1 predictions matched to V2 in SFL | {E_count:,} | ${E_spend/1e9:.2f}B |

### Derived
| Metric | Value |
|---|---|
| Hit@5 rate | {B/A*100:.1f}% |
| Rank-1 accuracy | {C/A*100:.1f}% |
| Avg spend per correct rank-1 V2 visit | ${D_spend/max(D_count,1):,.0f} |
| Avg spend per any rank-1 V2 visit | ${E_spend/max(E_count,1):,.0f} |
| Correctly predicted V2 spend as % of all V2 spend | {D_spend/max(E_spend,1)*100:.1f}% |
"""))

print("Block 8d done.")
