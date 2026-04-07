-- ============================================================
-- SQL_PMA_01 — TRANSITION EVIDENCE BUCKETS + MODEL PERFORMANCE
-- Purpose : Segment trigger_dx codes into Low/Medium/High
--           evidence buckets using transition counts from
--           the Markov train table (which has trigger_dx grain)
--           Show model performance per bucket per model per window
-- Goal    : Markov dominates High evidence,
--           SASRec/HSTU dominate Low/Medium evidence
-- Sources : A870800_gen_rec_provider_eval_5pct          (SQL_05)
--           A870800_gen_rec_provider_markov_train_5pct  (SQL_02)
--             grain: (trigger_dx, member_segment, from_provider, to_provider)
--             has transition_count at this grain
-- Output  : A870800_gen_rec_pma_transition_bucket_5pct
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_pma_transition_bucket_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH

-- 1. Sum transition counts per trigger_dx from markov train table
-- This gives total observed evidence for each trigger DX in training data
dx_evidence AS (
    SELECT
        CAST(trigger_dx AS STRING)                       AS trigger_dx
        ,SUM(transition_count)                           AS total_transitions
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_5pct`
    WHERE trigger_dx IS NOT NULL
    GROUP BY trigger_dx
),

-- 2. 33rd and 67th percentile thresholds
percentiles AS (
    SELECT
        APPROX_QUANTILES(total_transitions, 3)[OFFSET(1)] AS p33
        ,APPROX_QUANTILES(total_transitions, 3)[OFFSET(2)] AS p67
    FROM dx_evidence
),

-- 3. Assign Low / Medium / High to each trigger_dx
bucketed AS (
    SELECT
        d.trigger_dx
        ,d.total_transitions
        ,CASE
            WHEN d.total_transitions <= p.p33 THEN 'Low'
            WHEN d.total_transitions <= p.p67 THEN 'Medium'
            ELSE 'High'
         END                                             AS transition_bucket
        ,p.p33                                           AS threshold_p33
        ,p.p67                                           AS threshold_p67
    FROM dx_evidence d
    CROSS JOIN percentiles p
),

-- 4. Join eval table to buckets on trigger_dx
eval_bucketed AS (
    SELECT
        e.model
        ,e.time_bucket
        ,e.tp
        ,e.fp
        ,e.fn
        ,e.hit_at_1
        ,e.hit_at_3
        ,e.hit_at_5
        ,e.ndcg_at_1
        ,e.ndcg_at_3
        ,e.ndcg_at_5
        ,e.precision_at_3
        ,e.recall_at_3
        ,COALESCE(b.transition_bucket, 'Unknown')        AS transition_bucket
        ,COALESCE(b.total_transitions, 0)                AS total_transitions
        ,b.threshold_p33
        ,b.threshold_p67
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_5pct` e
    LEFT JOIN bucketed b
        ON CAST(e.trigger_dx AS STRING) = b.trigger_dx
)

-- 5. Aggregate per (transition_bucket, model, time_bucket)
SELECT
    transition_bucket
    ,model
    ,time_bucket
    ,COUNT(*)                                            AS n_triggers
    ,ROUND(AVG(total_transitions), 0)                    AS avg_transition_count
    ,ANY_VALUE(threshold_p33)                            AS threshold_p33
    ,ANY_VALUE(threshold_p67)                            AS threshold_p67
    ,SUM(tp)                                             AS total_tp
    ,SUM(fp)                                             AS total_fp
    ,SUM(fn)                                             AS total_fn
    ,ROUND(SUM(tp) / NULLIF(SUM(tp) + SUM(fp), 0), 4)   AS overall_precision
    ,ROUND(SUM(tp) / NULLIF(SUM(tp) + SUM(fn), 0), 4)   AS overall_recall
    ,ROUND(AVG(hit_at_1), 4)                             AS hit_at_1
    ,ROUND(AVG(hit_at_3), 4)                             AS hit_at_3
    ,ROUND(AVG(hit_at_5), 4)                             AS hit_at_5
    ,ROUND(AVG(ndcg_at_1), 4)                            AS ndcg_at_1
    ,ROUND(AVG(ndcg_at_3), 4)                            AS ndcg_at_3
    ,ROUND(AVG(ndcg_at_5), 4)                            AS ndcg_at_5
    ,ROUND(AVG(precision_at_3), 4)                       AS precision_at_3
    ,ROUND(AVG(recall_at_3), 4)                          AS recall_at_3
FROM eval_bucketed
WHERE transition_bucket != 'Unknown'
GROUP BY transition_bucket, model, time_bucket
ORDER BY time_bucket
    ,CASE transition_bucket WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END
    ,model
;
