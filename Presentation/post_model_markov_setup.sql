-- ============================================================
-- Analysis_Layer1_BQ_tables_5pct.sql
-- Purpose : 3 materialized analysis tables for model comparison
--           ALL models on 5% sample — apples to apples
-- Sources : A870800_gen_rec_trigger_scores              (SASRec, BERT4Rec — already 5pct)
--           A870800_gen_rec_markov_trigger_scores_5pct   (Markov — now 5pct)
-- ============================================================


-- ============================================================
-- TABLE 1 — OVERALL PERFORMANCE
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
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_trigger_scores_5pct`
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
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_trigger_scores_5pct`
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
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_ending_specialty`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH all_scores AS (
    SELECT model, time_bucket, true_labels, top5_predictions, ndcg_at_3
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_scores`
    UNION ALL
    SELECT model, time_bucket, true_labels, top5_predictions, ndcg_at_3
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_trigger_scores_5pct`
),
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
    ,SUM(CASE WHEN ARRAY_LENGTH(pred_arr) >= 1
                   AND pred_arr[OFFSET(0)] = ending_specialty
              THEN 1 ELSE 0 END)                         AS predicted_at_1
    ,SUM(CASE WHEN ending_specialty IN UNNEST(
                  ARRAY(SELECT x FROM UNNEST(pred_arr) AS x WITH OFFSET pos
                        WHERE pos < 3))
              THEN 1 ELSE 0 END)                         AS predicted_at_3
    ,SUM(CASE WHEN ending_specialty IN UNNEST(pred_arr)
              THEN 1 ELSE 0 END)                         AS predicted_at_5
    ,ROUND(SUM(CASE WHEN ending_specialty IN UNNEST(pred_arr)
                    THEN 1 ELSE 0 END) * 1.0
           / NULLIF(COUNT(*), 0), 4)                     AS hit_rate_at_5
    ,ROUND(AVG(ndcg_at_3), 4)                            AS avg_ndcg_at_3
FROM exploded
GROUP BY model, time_bucket, ending_specialty
ORDER BY model, time_bucket, total_appearances DESC;
