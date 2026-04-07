-- ============================================================
-- SQL_05 — PROVIDER MODEL EVALUATION WITH TP/FP/FN
-- Purpose : One row per (member, trigger_date, trigger_dx, time_bucket, model)
--           for all 4 models with TP, FP, FN and ranking metrics
-- Sources : A870800_gen_rec_provider_trigger_scores       (DL models — NB_07)
--           A870800_gen_rec_provider_markov_predictions_5pct  (Markov — SQL_04)
--           A870800_gen_rec_provider_model_test_agg_5pct  (true labels — SQL_02)
-- Output  : A870800_gen_rec_provider_eval_5pct
-- TP = # predicted in true labels
-- FP = # predicted NOT in true labels
-- FN = # true labels NOT in predictions
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH

-- ── 1. DL model rows — already have predictions + true labels ─────────────
dl AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,time_bucket
        ,model
        ,sample
        ,CAST(run_timestamp AS STRING)                   AS run_timestamp
        ,top5_predictions
        ,top5_scores
        ,true_labels
        -- Split to arrays for set operations
        ,SPLIT(top5_predictions, '|')                    AS pred_array
        ,SPLIT(true_labels,      '|')                    AS true_array
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_trigger_scores`
    WHERE sample = '5pct'
      AND top5_predictions IS NOT NULL
      AND true_labels IS NOT NULL
      AND top5_predictions != ''
      AND true_labels != ''
),

-- ── 2. Markov — aggregate per trigger from flat predictions table ──────────
markov_agg AS (
    SELECT
        member_id
        ,CAST(trigger_date AS STRING)                    AS trigger_date
        ,CAST(trigger_dx   AS STRING)                    AS trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        -- Predictions as pipe string and array (cast NPI to STRING)
        ,ARRAY_TO_STRING(
            ARRAY_AGG(CAST(predicted_provider AS STRING)
                ORDER BY prediction_rank LIMIT 5),
            '|'
         )                                               AS top5_predictions
        ,ARRAY_AGG(CAST(predicted_provider AS STRING)
            ORDER BY prediction_rank LIMIT 5)            AS pred_array
        -- Scores as pipe string
        ,ARRAY_TO_STRING(
            ARRAY_AGG(CAST(ROUND(probability, 4) AS STRING)
                ORDER BY prediction_rank LIMIT 5),
            '|'
         )                                               AS top5_scores
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_predictions_5pct`
    WHERE predicted_provider IS NOT NULL
    GROUP BY
        member_id, trigger_date, trigger_dx, member_segment,
        is_t30_qualified, is_t60_qualified, is_t180_qualified
),

-- ── 3. True labels from test_agg — cast each element to STRING ────────────
test_labels AS (
    SELECT
        member_id
        ,CAST(trigger_date AS STRING)                    AS trigger_date
        ,CAST(trigger_dx   AS STRING)                    AS trigger_dx
        ,member_segment
        -- Cast each element in the array to STRING
        ,(SELECT ARRAY_AGG(CAST(x AS STRING)) FROM UNNEST(lab_t30)  x WHERE x IS NOT NULL) AS true_t30
        ,(SELECT ARRAY_AGG(CAST(x AS STRING)) FROM UNNEST(lab_t60)  x WHERE x IS NOT NULL) AS true_t60
        ,(SELECT ARRAY_AGG(CAST(x AS STRING)) FROM UNNEST(lab_t180) x WHERE x IS NOT NULL) AS true_t180
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_agg_5pct`
),

-- ── 4. Markov rows — one per qualifying window ────────────────────────────
markov_rows AS (
    SELECT
        m.member_id
        ,m.trigger_date
        ,m.trigger_dx
        ,m.member_segment
        ,w.time_bucket
        ,'Markov'                                        AS model
        ,'5pct'                                          AS sample
        ,CAST(CURRENT_TIMESTAMP() AS STRING)             AS run_timestamp
        ,m.top5_predictions
        ,m.top5_scores
        ,ARRAY_TO_STRING(w.true_array, '|')              AS true_labels
        ,m.pred_array
        ,w.true_array
    FROM markov_agg m
    JOIN test_labels tl
        ON  m.member_id      = tl.member_id
        AND m.trigger_date   = tl.trigger_date
        AND m.trigger_dx     = tl.trigger_dx
        AND m.member_segment = tl.member_segment
    CROSS JOIN UNNEST([
        STRUCT('T0_30'   AS time_bucket, tl.true_t30  AS true_array, m.is_t30_qualified  AS qualified),
        STRUCT('T30_60'  AS time_bucket, tl.true_t60  AS true_array, m.is_t60_qualified  AS qualified),
        STRUCT('T60_180' AS time_bucket, tl.true_t180 AS true_array, m.is_t180_qualified AS qualified)
    ]) AS w
    WHERE w.qualified = TRUE
      AND ARRAY_LENGTH(COALESCE(w.true_array, [])) > 0
),

-- ── 5. Union all 4 models ──────────────────────────────────────────────────
all_preds AS (
    SELECT
        member_id, trigger_date, trigger_dx, member_segment,
        time_bucket, model, sample, run_timestamp,
        top5_predictions, top5_scores, true_labels,
        pred_array, true_array
    FROM dl

    UNION ALL

    SELECT
        member_id, trigger_date, trigger_dx, member_segment,
        time_bucket, model, sample, run_timestamp,
        top5_predictions, top5_scores, true_labels,
        pred_array, true_array
    FROM markov_rows
),

-- ── 6. TP / FP / FN + counts ──────────────────────────────────────────────
with_counts AS (
    SELECT
        member_id, trigger_date, trigger_dx, member_segment,
        time_bucket, model, sample, run_timestamp,
        top5_predictions, top5_scores, true_labels,
        pred_array, true_array,
        ARRAY_LENGTH(pred_array)                         AS n_predicted,
        ARRAY_LENGTH(true_array)                         AS n_true,
        -- TP
        (SELECT COUNT(*) FROM UNNEST(pred_array) p WHERE p IN UNNEST(true_array)) AS tp,
        -- FP
        (SELECT COUNT(*) FROM UNNEST(pred_array) p WHERE p NOT IN UNNEST(true_array)) AS fp,
        -- FN
        (SELECT COUNT(*) FROM UNNEST(true_array) t WHERE t NOT IN UNNEST(pred_array)) AS fn
    FROM all_preds
),

-- ── 7. Ranking metrics ─────────────────────────────────────────────────────
with_metrics AS (
    SELECT
        member_id, trigger_date, trigger_dx, member_segment,
        time_bucket, model, sample, run_timestamp,
        top5_predictions, top5_scores, true_labels,
        n_predicted, n_true, tp, fp, fn,

        -- Hit@K
        IF((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,1)) p WHERE p IN UNNEST(true_array))>0, 1.0, 0.0) AS hit_at_1,
        IF((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,3)) p WHERE p IN UNNEST(true_array))>0, 1.0, 0.0) AS hit_at_3,
        IF((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,5)) p WHERE p IN UNNEST(true_array))>0, 1.0, 0.0) AS hit_at_5,

        -- Precision@K
        ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,1)) p WHERE p IN UNNEST(true_array)) / 1.0, 4) AS precision_at_1,
        ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,3)) p WHERE p IN UNNEST(true_array)) / 3.0, 4) AS precision_at_3,
        ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,5)) p WHERE p IN UNNEST(true_array)) / 5.0, 4) AS precision_at_5,

        -- Recall@K
        ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,1)) p WHERE p IN UNNEST(true_array)) / NULLIF(n_true,0), 4) AS recall_at_1,
        ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,3)) p WHERE p IN UNNEST(true_array)) / NULLIF(n_true,0), 4) AS recall_at_3,
        ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,5)) p WHERE p IN UNNEST(true_array)) / NULLIF(n_true,0), 4) AS recall_at_5,

        -- NDCG@K
        ROUND(
            (SELECT IFNULL(SUM(1.0/(LOG(pos+2)/LOG(2))),0)
             FROM UNNEST(ARRAY_SLICE(pred_array,0,1)) p WITH OFFSET pos WHERE p IN UNNEST(true_array))
            / NULLIF((SELECT SUM(1.0/(LOG(ip+2)/LOG(2))) FROM UNNEST(GENERATE_ARRAY(0,LEAST(n_true,1)-1)) ip),0)
        ,4) AS ndcg_at_1,
        ROUND(
            (SELECT IFNULL(SUM(1.0/(LOG(pos+2)/LOG(2))),0)
             FROM UNNEST(ARRAY_SLICE(pred_array,0,3)) p WITH OFFSET pos WHERE p IN UNNEST(true_array))
            / NULLIF((SELECT SUM(1.0/(LOG(ip+2)/LOG(2))) FROM UNNEST(GENERATE_ARRAY(0,LEAST(n_true,3)-1)) ip),0)
        ,4) AS ndcg_at_3,
        ROUND(
            (SELECT IFNULL(SUM(1.0/(LOG(pos+2)/LOG(2))),0)
             FROM UNNEST(ARRAY_SLICE(pred_array,0,5)) p WITH OFFSET pos WHERE p IN UNNEST(true_array))
            / NULLIF((SELECT SUM(1.0/(LOG(ip+2)/LOG(2))) FROM UNNEST(GENERATE_ARRAY(0,LEAST(n_true,5)-1)) ip),0)
        ,4) AS ndcg_at_5

    FROM with_counts
)

SELECT * EXCEPT(pred_array, true_array)
FROM with_metrics
ORDER BY model, time_bucket, member_id, trigger_date
;


-- ══════════════════════════════════════════════════════════════════════════════
-- SUMMARY — run after table created
-- ══════════════════════════════════════════════════════════════════════════════
-- SELECT
--     model, time_bucket
--     ,COUNT(*)                      AS n_triggers
--     ,SUM(tp)                       AS total_tp
--     ,SUM(fp)                       AS total_fp
--     ,SUM(fn)                       AS total_fn
--     ,ROUND(AVG(hit_at_1),4)        AS hit_at_1
--     ,ROUND(AVG(hit_at_3),4)        AS hit_at_3
--     ,ROUND(AVG(hit_at_5),4)        AS hit_at_5
--     ,ROUND(AVG(ndcg_at_3),4)       AS ndcg_at_3
--     ,ROUND(AVG(ndcg_at_5),4)       AS ndcg_at_5
--     ,ROUND(AVG(precision_at_3),4)  AS precision_at_3
--     ,ROUND(AVG(recall_at_3),4)     AS recall_at_3
-- FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_5pct`
-- GROUP BY model, time_bucket
-- ORDER BY time_bucket, model
;
