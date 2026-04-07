-- ============================================================
-- EDA_ear_infection_predictions.sql
-- Purpose : How well did each model predict next visit
--           for ear infection diagnosis triggers?
-- ICD-10 ear infection codes (raw / clean no-decimal):
--   H66.91 / H6691  — Otitis media unspecified, right ear
--   H66.92 / H6692  — Otitis media unspecified, left ear
--   H66.93 / H6693  — Otitis media unspecified, bilateral
--   H66.001/ H66001 — Acute suppurative otitis media, right ear
--   H66.002/ H66002 — Acute suppurative otitis media, left ear
--   H66.90 / H6690  — Otitis media unspecified, unspecified ear
-- ============================================================

-- ------------------------------------------------------------
-- Q1. MARKOV — Training transition probabilities
--     Shows what the frequency model learned from training data
--     Facts: how often did each specialty follow ear infection
--            in training data, and what % of the time
-- ------------------------------------------------------------
WITH ear_transitions AS (
    SELECT
        trigger_dx
        ,trigger_dx_clean
        ,trigger_ccsr_desc                                 AS dx_description
        ,next_specialty
        ,next_specialty_desc
        ,SUM(transition_count)                             AS transition_count
        ,SUM(unique_members)                               AS unique_members
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE trigger_dx IN ('H66.91','H6691','H66.92','H6692',
                         'H66.93','H6693','H66.001','H66001',
                         'H66.002','H66002','H66.90','H6690')
      AND next_specialty IS NOT NULL
    GROUP BY
        trigger_dx, trigger_dx_clean, trigger_ccsr_desc
        ,next_specialty, next_specialty_desc
),
totals AS (
    SELECT trigger_dx, SUM(transition_count) AS total_transitions
    FROM ear_transitions
    GROUP BY trigger_dx
)
SELECT
    e.trigger_dx
    ,e.trigger_dx_clean
    ,e.dx_description
    ,e.next_specialty
    ,e.next_specialty_desc
    ,e.transition_count                                    AS times_in_training
    ,e.unique_members                                      AS unique_members_in_training
    ,t.total_transitions                                   AS total_training_transitions
    ,ROUND(100.0 * e.transition_count
        / t.total_transitions, 2)                          AS transition_probability_pct
FROM ear_transitions e
JOIN totals t ON e.trigger_dx = t.trigger_dx
ORDER BY e.trigger_dx, transition_probability_pct DESC
;

-- ------------------------------------------------------------
-- Q2. BERT4Rec — Prediction accuracy for ear infection triggers
--     For each specialty the model predicted:
--       times_predicted    — how many triggers it appeared in top5
--       times_true         — how many of those it was actually correct
--       times_in_true_set  — how many triggers had it in true labels
--       precision          — times_true / times_predicted
--       recall             — times_true / times_in_true_set
-- ------------------------------------------------------------
WITH ear_triggers AS (
    -- All BERT4Rec scored ear infection triggers
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,SPLIT(top5_predictions, '|')                      AS pred_arr
        ,SPLIT(top5_scores, '|')                           AS score_arr
        ,SPLIT(true_labels, '|')                           AS true_arr
        ,hit_at_1
        ,hit_at_3
        ,hit_at_5
        ,ndcg_at_3
        ,ndcg_at_5
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_scores`
    WHERE model       = 'BERT4Rec'
      AND time_bucket = 'T0_30'
      AND trigger_dx  IN ('H66.91','H6691','H66.92','H6692',
                          'H66.93','H6693','H66.001','H66001',
                          'H66.002','H66002','H66.90','H6690')
),
-- Overall performance summary first
overall AS (
    SELECT
        trigger_dx
        ,COUNT(*)                                          AS total_triggers
        ,ROUND(AVG(hit_at_1), 4)                           AS avg_hit_at_1
        ,ROUND(AVG(hit_at_3), 4)                           AS avg_hit_at_3
        ,ROUND(AVG(hit_at_5), 4)                           AS avg_hit_at_5
        ,ROUND(AVG(ndcg_at_3), 4)                          AS avg_ndcg_at_3
        ,ROUND(AVG(ndcg_at_5), 4)                          AS avg_ndcg_at_5
        ,COUNTIF(hit_at_5 = 1)                             AS triggers_hit_at_5
        ,COUNTIF(hit_at_5 = 0)                             AS triggers_missed_at_5
    FROM ear_triggers
    GROUP BY trigger_dx
),
-- Explode predictions to get per-specialty stats
exploded_preds AS (
    SELECT
        t.trigger_dx
        ,pred                                              AS predicted_specialty
        ,SAFE_CAST(score AS FLOAT64)                       AS score
        ,pos + 1                                           AS rank_position
        -- Is this prediction in the true label set?
        ,CASE WHEN pred IN UNNEST(t.true_arr)
              THEN 1 ELSE 0 END                            AS is_correct
    FROM ear_triggers t
    CROSS JOIN UNNEST(pred_arr)  AS pred  WITH OFFSET pos
    JOIN       UNNEST(score_arr) AS score WITH OFFSET spos
        ON pos = spos
),
-- Explode true labels to get how often each specialty appeared
exploded_true AS (
    SELECT
        trigger_dx
        ,true_label                                        AS specialty
        ,COUNT(*)                                          AS times_in_true_set
    FROM ear_triggers
    CROSS JOIN UNNEST(true_arr) AS true_label
    GROUP BY trigger_dx, true_label
),
pred_summary AS (
    SELECT
        trigger_dx
        ,predicted_specialty
        ,COUNT(*)                                          AS times_predicted
        ,SUM(is_correct)                                   AS times_correct
        ,ROUND(AVG(score), 4)                              AS avg_score
        ,ROUND(AVG(rank_position), 2)                      AS avg_rank_position
    FROM exploded_preds
    GROUP BY trigger_dx, predicted_specialty
)
-- Summary: overall metrics
SELECT
    'OVERALL_PERFORMANCE'                                  AS section
    ,o.trigger_dx
    ,NULL                                                  AS predicted_specialty
    ,o.total_triggers
    ,NULL                                                  AS times_predicted
    ,NULL                                                  AS times_correct
    ,NULL                                                  AS times_in_true_set
    ,o.avg_hit_at_1
    ,o.avg_hit_at_3
    ,o.avg_hit_at_5
    ,o.avg_ndcg_at_3
    ,NULL                                                  AS precision_when_predicted
    ,NULL                                                  AS recall_of_true_labels
    ,NULL                                                  AS avg_score
    ,NULL                                                  AS avg_rank_position
FROM overall o

UNION ALL

-- Per-specialty breakdown
SELECT
    'PER_SPECIALTY'                                        AS section
    ,p.trigger_dx
    ,p.predicted_specialty
    ,o.total_triggers
    ,p.times_predicted
    ,p.times_correct
    ,COALESCE(tr.times_in_true_set, 0)                     AS times_in_true_set
    ,NULL                                                  AS avg_hit_at_1
    ,NULL                                                  AS avg_hit_at_3
    ,NULL                                                  AS avg_hit_at_5
    ,NULL                                                  AS avg_ndcg_at_3
    ,ROUND(100.0 * p.times_correct
        / NULLIF(p.times_predicted, 0), 2)                 AS precision_when_predicted
    ,ROUND(100.0 * p.times_correct
        / NULLIF(tr.times_in_true_set, 0), 2)              AS recall_of_true_labels
    ,p.avg_score
    ,p.avg_rank_position
FROM pred_summary p
JOIN overall o ON p.trigger_dx = o.trigger_dx
LEFT JOIN exploded_true tr
    ON p.trigger_dx = tr.trigger_dx
    AND p.predicted_specialty = tr.specialty
ORDER BY section DESC, trigger_dx, avg_score DESC
;

-- ------------------------------------------------------------
-- Q3. SASRec — identical structure to Q2 for direct comparison
-- ------------------------------------------------------------
WITH ear_triggers AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,SPLIT(top5_predictions, '|')                      AS pred_arr
        ,SPLIT(top5_scores, '|')                           AS score_arr
        ,SPLIT(true_labels, '|')                           AS true_arr
        ,hit_at_1
        ,hit_at_3
        ,hit_at_5
        ,ndcg_at_3
        ,ndcg_at_5
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_scores`
    WHERE model       = 'SASRec'
      AND time_bucket = 'T0_30'
      AND trigger_dx  IN ('H66.91','H6691','H66.92','H6692',
                          'H66.93','H6693','H66.001','H66001',
                          'H66.002','H66002','H66.90','H6690')
),
overall AS (
    SELECT
        trigger_dx
        ,COUNT(*)                                          AS total_triggers
        ,ROUND(AVG(hit_at_1), 4)                           AS avg_hit_at_1
        ,ROUND(AVG(hit_at_3), 4)                           AS avg_hit_at_3
        ,ROUND(AVG(hit_at_5), 4)                           AS avg_hit_at_5
        ,ROUND(AVG(ndcg_at_3), 4)                          AS avg_ndcg_at_3
        ,ROUND(AVG(ndcg_at_5), 4)                          AS avg_ndcg_at_5
        ,COUNTIF(hit_at_5 = 1)                             AS triggers_hit_at_5
        ,COUNTIF(hit_at_5 = 0)                             AS triggers_missed_at_5
    FROM ear_triggers
    GROUP BY trigger_dx
),
exploded_preds AS (
    SELECT
        t.trigger_dx
        ,pred                                              AS predicted_specialty
        ,SAFE_CAST(score AS FLOAT64)                       AS score
        ,pos + 1                                           AS rank_position
        ,CASE WHEN pred IN UNNEST(t.true_arr)
              THEN 1 ELSE 0 END                            AS is_correct
    FROM ear_triggers t
    CROSS JOIN UNNEST(pred_arr)  AS pred  WITH OFFSET pos
    JOIN       UNNEST(score_arr) AS score WITH OFFSET spos
        ON pos = spos
),
exploded_true AS (
    SELECT
        trigger_dx
        ,true_label                                        AS specialty
        ,COUNT(*)                                          AS times_in_true_set
    FROM ear_triggers
    CROSS JOIN UNNEST(true_arr) AS true_label
    GROUP BY trigger_dx, true_label
),
pred_summary AS (
    SELECT
        trigger_dx
        ,predicted_specialty
        ,COUNT(*)                                          AS times_predicted
        ,SUM(is_correct)                                   AS times_correct
        ,ROUND(AVG(score), 4)                              AS avg_score
        ,ROUND(AVG(rank_position), 2)                      AS avg_rank_position
    FROM exploded_preds
    GROUP BY trigger_dx, predicted_specialty
)
SELECT
    'OVERALL_PERFORMANCE'                                  AS section
    ,o.trigger_dx
    ,NULL                                                  AS predicted_specialty
    ,o.total_triggers
    ,NULL                                                  AS times_predicted
    ,NULL                                                  AS times_correct
    ,NULL                                                  AS times_in_true_set
    ,o.avg_hit_at_1
    ,o.avg_hit_at_3
    ,o.avg_hit_at_5
    ,o.avg_ndcg_at_3
    ,NULL                                                  AS precision_when_predicted
    ,NULL                                                  AS recall_of_true_labels
    ,NULL                                                  AS avg_score
    ,NULL                                                  AS avg_rank_position
FROM overall o

UNION ALL

SELECT
    'PER_SPECIALTY'                                        AS section
    ,p.trigger_dx
    ,p.predicted_specialty
    ,o.total_triggers
    ,p.times_predicted
    ,p.times_correct
    ,COALESCE(tr.times_in_true_set, 0)                     AS times_in_true_set
    ,NULL
    ,NULL
    ,NULL
    ,NULL
    ,ROUND(100.0 * p.times_correct
        / NULLIF(p.times_predicted, 0), 2)                 AS precision_when_predicted
    ,ROUND(100.0 * p.times_correct
        / NULLIF(tr.times_in_true_set, 0), 2)              AS recall_of_true_labels
    ,p.avg_score
    ,p.avg_rank_position
FROM pred_summary p
JOIN overall o ON p.trigger_dx = o.trigger_dx
LEFT JOIN exploded_true tr
    ON p.trigger_dx = tr.trigger_dx
    AND p.predicted_specialty = tr.specialty
ORDER BY section DESC, trigger_dx, avg_score DESC
;
