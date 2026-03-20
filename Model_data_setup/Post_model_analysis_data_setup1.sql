-- ============================================================
-- Analysis_Layer1_BQ_tables.sql
-- Purpose : 3 materialized analysis tables for model comparison
-- Sources : A870800_gen_rec_trigger_scores        (SASRec, BERT4Rec)
--           A870800_gen_rec_markov_trigger_scores  (Markov)
-- CAVEATS:
--   Models predict top 5 specialties per trigger per time window.
--   True label set may contain more than 5 specialties.
--   Hit@K is binary per trigger — 1 if any top-K prediction matches.
--   Probability cutoff analysis deferred.
-- ============================================================


-- ============================================================
-- TABLE 1 — OVERALL PERFORMANCE
-- Metrics by model × time_bucket × member_segment
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_overall`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH all_scores AS (
    SELECT model, time_bucket, member_segment
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_1, ndcg_at_3, ndcg_at_5
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_scores`
    UNION ALL
    SELECT model, time_bucket, member_segment
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_1, ndcg_at_3, ndcg_at_5
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_trigger_scores`
),
by_segment AS (
    SELECT
        model, time_bucket, member_segment
        ,COUNT(*)                                        AS n_triggers
        ,ROUND(AVG(hit_at_1), 4)                         AS hit_at_1
        ,ROUND(AVG(hit_at_3), 4)                         AS hit_at_3
        ,ROUND(AVG(hit_at_5), 4)                         AS hit_at_5
        ,ROUND(AVG(ndcg_at_1), 4)                        AS ndcg_at_1
        ,ROUND(AVG(ndcg_at_3), 4)                        AS ndcg_at_3
        ,ROUND(AVG(ndcg_at_5), 4)                        AS ndcg_at_5
    FROM all_scores
    GROUP BY model, time_bucket, member_segment
),
overall AS (
    SELECT
        model, time_bucket
        ,'ALL'                                           AS member_segment
        ,COUNT(*)                                        AS n_triggers
        ,ROUND(AVG(hit_at_1), 4)                         AS hit_at_1
        ,ROUND(AVG(hit_at_3), 4)                         AS hit_at_3
        ,ROUND(AVG(hit_at_5), 4)                         AS hit_at_5
        ,ROUND(AVG(ndcg_at_1), 4)                        AS ndcg_at_1
        ,ROUND(AVG(ndcg_at_3), 4)                        AS ndcg_at_3
        ,ROUND(AVG(ndcg_at_5), 4)                        AS ndcg_at_5
    FROM all_scores
    GROUP BY model, time_bucket
)
SELECT * FROM by_segment
UNION ALL
SELECT * FROM overall
ORDER BY model, time_bucket, member_segment;


-- ============================================================
-- TABLE 2 — PERFORMANCE BY DIAGNOSIS CODE
-- trigger_volume enables Top 15 by volume chart
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_diag`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH all_scores AS (
    SELECT model, time_bucket, trigger_dx
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_1, ndcg_at_3, ndcg_at_5
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_scores`
    UNION ALL
    SELECT model, time_bucket, trigger_dx
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_1, ndcg_at_3, ndcg_at_5
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_trigger_scores`
)
SELECT
    model, time_bucket, trigger_dx
    ,COUNT(*)                                            AS trigger_volume
    ,ROUND(AVG(hit_at_1), 4)                             AS hit_at_1
    ,ROUND(AVG(hit_at_3), 4)                             AS hit_at_3
    ,ROUND(AVG(hit_at_5), 4)                             AS hit_at_5
    ,ROUND(AVG(ndcg_at_1), 4)                            AS ndcg_at_1
    ,ROUND(AVG(ndcg_at_3), 4)                            AS ndcg_at_3
    ,ROUND(AVG(ndcg_at_5), 4)                            AS ndcg_at_5
FROM all_scores
GROUP BY model, time_bucket, trigger_dx
ORDER BY model, time_bucket, trigger_volume DESC;


-- ============================================================
-- TABLE 3 — PERFORMANCE BY ENDING SPECIALTY
-- For each specialty appearing in true labels:
--   total appearances, predicted correctly at K=1/3/5, rates
-- Approach: split pipe-string predictions to array per row,
--           check by position — no fragile string matching
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_ending_specialty`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH all_scores AS (
    SELECT model, time_bucket, true_labels, top5_predictions, ndcg_at_3
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_scores`
    UNION ALL
    SELECT model, time_bucket, true_labels, top5_predictions, ndcg_at_3
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_trigger_scores`
),
-- Parse pipe strings to arrays once
parsed AS (
    SELECT
        model
        ,time_bucket
        ,SPLIT(true_labels, '|')                         AS true_arr
        ,SPLIT(top5_predictions, '|')                    AS pred_arr
        ,ndcg_at_3
    FROM all_scores
    WHERE true_labels IS NOT NULL AND true_labels != ''
),
-- Explode true labels — one row per specialty per trigger
exploded AS (
    SELECT
        model
        ,time_bucket
        ,ending_specialty
        ,pred_arr
        ,ndcg_at_3
    FROM parsed
    CROSS JOIN UNNEST(true_arr) AS ending_specialty
    WHERE ending_specialty IS NOT NULL AND ending_specialty != ''
)
SELECT
    model
    ,time_bucket
    ,ending_specialty
    ,COUNT(*)                                            AS total_appearances
    -- Predicted at K=1: specialty is pred_arr[0]
    ,SUM(CASE WHEN ARRAY_LENGTH(pred_arr) >= 1
                   AND pred_arr[OFFSET(0)] = ending_specialty
              THEN 1 ELSE 0 END)                         AS predicted_at_1
    -- Predicted at K=3: specialty in pred_arr[0..2]
    ,SUM(CASE WHEN ending_specialty IN UNNEST(
                  ARRAY(SELECT x FROM UNNEST(pred_arr) AS x WITH OFFSET pos
                        WHERE pos < 3))
              THEN 1 ELSE 0 END)                         AS predicted_at_3
    -- Predicted at K=5: specialty anywhere in pred_arr
    ,SUM(CASE WHEN ending_specialty IN UNNEST(pred_arr)
              THEN 1 ELSE 0 END)                         AS predicted_at_5
    ,ROUND(SUM(CASE WHEN ending_specialty IN UNNEST(pred_arr)
                    THEN 1 ELSE 0 END) * 1.0
           / NULLIF(COUNT(*), 0), 4)                     AS hit_rate_at_5
    ,ROUND(AVG(ndcg_at_3), 4)                            AS avg_ndcg_at_3
FROM exploded
GROUP BY model, time_bucket, ending_specialty
ORDER BY model, time_bucket, total_appearances DESC;
