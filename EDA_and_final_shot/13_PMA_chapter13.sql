-- ============================================================
-- Ear Infection Diagnosis — Predictions from 3 sources
-- ICD-9 ear infection codes (raw format in trigger_dx):
--   382.9  / 3829  — Otitis media unspecified
--   382.00 / 38200 — Acute suppurative otitis media
--   381.00 / 38100 — Acute nonsuppurative otitis media
--   380.10 / 38010 — Infective otitis externa
-- ============================================================

-- ------------------------------------------------------------
-- Q1. MARKOV — Transition probabilities for ear infection triggers
--     probability = transition_count / total transitions from that dx
-- ------------------------------------------------------------
WITH ear_dx_transitions AS (
    SELECT
        trigger_dx
        ,trigger_dx_clean
        ,trigger_ccsr_desc                                 AS dx_description
        ,next_specialty
        ,next_specialty_desc
        ,SUM(transition_count)                             AS transition_count
        ,SUM(unique_members)                               AS unique_members
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE trigger_dx IN ('382.9','3829','382.00','38200',
                         '381.00','38100','380.10','38010')
      AND next_specialty IS NOT NULL
    GROUP BY trigger_dx, trigger_dx_clean, trigger_ccsr_desc,
             next_specialty, next_specialty_desc
),
totals AS (
    SELECT
        trigger_dx
        ,SUM(transition_count)                             AS total_transitions
    FROM ear_dx_transitions
    GROUP BY trigger_dx
)
SELECT
    e.trigger_dx
    ,e.trigger_dx_clean
    ,e.dx_description
    ,e.next_specialty
    ,e.next_specialty_desc
    ,e.transition_count
    ,e.unique_members
    ,t.total_transitions
    ,ROUND(100.0 * e.transition_count / t.total_transitions, 2) AS transition_pct
FROM ear_dx_transitions e
JOIN totals t ON e.trigger_dx = t.trigger_dx
ORDER BY e.trigger_dx, transition_pct DESC
;

-- ------------------------------------------------------------
-- Q2. BERT4Rec — Top 5 predictions for ear infection triggers
--     Parse pipe-separated top5_predictions + top5_scores
--     Aggregate avg score per predicted specialty across triggers
-- ------------------------------------------------------------
WITH ear_scores AS (
    SELECT
        trigger_dx
        ,time_bucket
        ,hit_at_5
        ,SPLIT(top5_predictions, '|')                      AS pred_arr
        ,SPLIT(top5_scores, '|')                           AS score_arr
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_scores`
    WHERE model       = 'BERT4Rec'
      AND time_bucket = 'T0_30'
      AND trigger_dx IN ('382.9','3829','382.00','38200',
                         '381.00','38100','380.10','38010')
),
exploded AS (
    SELECT
        trigger_dx
        ,time_bucket
        ,hit_at_5
        ,pred                                              AS predicted_specialty
        ,SAFE_CAST(score AS FLOAT64)                       AS score
        ,pos + 1                                           AS rank_position
    FROM ear_scores
    CROSS JOIN UNNEST(pred_arr)  AS pred  WITH OFFSET pos
    JOIN       UNNEST(score_arr) AS score WITH OFFSET spos
        ON pos = spos
)
SELECT
    trigger_dx
    ,predicted_specialty
    ,COUNT(*)                                              AS times_predicted
    ,ROUND(AVG(score), 4)                                  AS avg_score
    ,ROUND(MIN(score), 4)                                  AS min_score
    ,ROUND(MAX(score), 4)                                  AS max_score
    ,ROUND(AVG(CAST(rank_position AS FLOAT64)), 2)         AS avg_rank_position
    ,ROUND(AVG(hit_at_5), 4)                               AS avg_hit_at_5
FROM exploded
GROUP BY trigger_dx, predicted_specialty
ORDER BY trigger_dx, avg_score DESC
;

-- ------------------------------------------------------------
-- Q3. SASRec — Top 5 predictions for ear infection triggers
--     Same structure as Q2 for direct comparison
-- ------------------------------------------------------------
WITH ear_scores AS (
    SELECT
        trigger_dx
        ,time_bucket
        ,hit_at_5
        ,SPLIT(top5_predictions, '|')                      AS pred_arr
        ,SPLIT(top5_scores, '|')                           AS score_arr
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_scores`
    WHERE model       = 'SASRec'
      AND time_bucket = 'T0_30'
      AND trigger_dx IN ('382.9','3829','382.00','38200',
                         '381.00','38100','380.10','38010')
),
exploded AS (
    SELECT
        trigger_dx
        ,time_bucket
        ,hit_at_5
        ,pred                                              AS predicted_specialty
        ,SAFE_CAST(score AS FLOAT64)                       AS score
        ,pos + 1                                           AS rank_position
    FROM ear_scores
    CROSS JOIN UNNEST(pred_arr)  AS pred  WITH OFFSET pos
    JOIN       UNNEST(score_arr) AS score WITH OFFSET spos
        ON pos = spos
)
SELECT
    trigger_dx
    ,predicted_specialty
    ,COUNT(*)                                              AS times_predicted
    ,ROUND(AVG(score), 4)                                  AS avg_score
    ,ROUND(MIN(score), 4)                                  AS min_score
    ,ROUND(MAX(score), 4)                                  AS max_score
    ,ROUND(AVG(CAST(rank_position AS FLOAT64)), 2)         AS avg_rank_position
    ,ROUND(AVG(hit_at_5), 4)                               AS avg_hit_at_5
FROM exploded
GROUP BY trigger_dx, predicted_specialty
ORDER BY trigger_dx, avg_score DESC
;
