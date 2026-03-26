-- ============================================================
-- SQL_05 — PROVIDER MODEL EVALUATION WITH TP/FP/FN
-- Purpose : Compute TP, FP, FN per trigger per window per model
--           for all 4 models (SASRec, BERT4Rec, HSTU, Markov)
--           Creates ONE new table — no updates to existing tables
-- Sources : A870800_gen_rec_provider_trigger_scores
--               (NB_07 output — SASRec, BERT4Rec, HSTU predictions)
--           A870800_gen_rec_provider_markov_predictions_5pct
--               (SQL_04 output — Markov top-5 predictions)
--           A870800_gen_rec_provider_model_test_agg_5pct
--               (true labels per trigger per window)
-- Output  : A870800_gen_rec_provider_eval_5pct
--               One row per (member, trigger_date, trigger_dx, window, model)
--               Columns: top5_predictions, true_labels,
--                        tp, fp, fn,
--                        hit_at_1, hit_at_3, hit_at_5,
--                        ndcg_at_1, ndcg_at_3, ndcg_at_5
-- Notes:
--   TP = predicted providers that ARE in true labels
--   FP = predicted providers that are NOT in true labels
--   FN = true label providers that were NOT in top-5 predictions
--   TN = not computed (31K - TP - FP - FN = meaningless at this scale)
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

-- ── STEP 1: Deep learning model predictions from NB_07 ────────────────────
-- Already has top5_predictions and true_labels as pipe-delimited strings
WITH dl_predictions AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,time_bucket
        ,model
        ,sample
        ,run_timestamp
        -- Split pipe-delimited strings back to arrays
        ,SPLIT(top5_predictions, '|')                    AS pred_array
        ,SPLIT(true_labels, '|')                         AS true_array
        ,top5_predictions
        ,true_labels
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_trigger_scores`
    WHERE sample = '5pct'
),

-- ── STEP 2: Markov predictions — aggregate top-5 per trigger per window ───
-- Markov predictions table has one row per (trigger, predicted_provider, rank)
-- Need to aggregate to one row per trigger with ranked array
markov_preds_agg AS (
    SELECT
        mp.member_id
        ,CAST(mp.trigger_date AS STRING)                 AS trigger_date
        ,mp.trigger_dx
        ,mp.member_segment
        ,mp.is_t30_qualified
        ,mp.is_t60_qualified
        ,mp.is_t180_qualified
        ,ARRAY_AGG(
            CAST(mp.predicted_provider AS STRING)
            ORDER BY mp.prediction_rank
            LIMIT 5
        )                                                AS pred_array
        ,ARRAY_TO_STRING(
            ARRAY_AGG(
                CAST(ROUND(mp.probability, 4) AS STRING)
                ORDER BY mp.prediction_rank
                LIMIT 5
            ), '|'
         )                                               AS top5_scores
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_predictions_5pct` mp
    WHERE mp.predicted_provider IS NOT NULL
    GROUP BY
        mp.member_id, mp.trigger_date, mp.trigger_dx,
        mp.member_segment, mp.is_t30_qualified,
        mp.is_t60_qualified, mp.is_t180_qualified
),

-- ── STEP 3: True labels per trigger per window from test_agg ─────────────
-- lab_t30/t60/t180 are ARRAY<STRING> from ARRAY_AGG in SQL_02
true_labels_all AS (
    SELECT
        member_id
        ,CAST(trigger_date AS STRING)                    AS trigger_date
        ,trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,lab_t30                                         AS true_t30
        ,lab_t60                                         AS true_t60
        ,lab_t180                                        AS true_t180
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_agg_5pct`
),

-- ── STEP 4: Markov rows — one per qualifying window ───────────────────────
markov_rows AS (
    SELECT
        mp.member_id
        ,mp.trigger_date
        ,mp.trigger_dx
        ,mp.member_segment
        ,'Markov'                                        AS model
        ,'5pct'                                          AS sample
        ,CURRENT_TIMESTAMP()                             AS run_timestamp
        ,w.time_bucket
        ,mp.pred_array
        ,w.true_array
        ,ARRAY_TO_STRING(mp.pred_array, '|')             AS top5_predictions
        ,mp.top5_scores                                  AS top5_scores
        ,ARRAY_TO_STRING(w.true_array, '|')              AS true_labels
    FROM markov_preds_agg mp
    JOIN true_labels_all tl
        ON  mp.member_id      = tl.member_id
        AND mp.trigger_date   = tl.trigger_date
        AND mp.trigger_dx     = tl.trigger_dx
        AND mp.member_segment = tl.member_segment
    -- Unpivot windows using CROSS JOIN with inline table
    CROSS JOIN UNNEST([
        STRUCT('T0_30'   AS time_bucket, (SELECT ARRAY_AGG(CAST(x AS STRING)) FROM UNNEST(tl.true_t30)  x) AS true_array, mp.is_t30_qualified  AS qualified),
        STRUCT('T30_60'  AS time_bucket, (SELECT ARRAY_AGG(CAST(x AS STRING)) FROM UNNEST(tl.true_t60)  x) AS true_array, mp.is_t60_qualified  AS qualified),
        STRUCT('T60_180' AS time_bucket, (SELECT ARRAY_AGG(CAST(x AS STRING)) FROM UNNEST(tl.true_t180) x) AS true_array, mp.is_t180_qualified AS qualified)
    ]) AS w
    WHERE w.qualified = TRUE
      AND ARRAY_LENGTH(w.true_array) > 0
),

-- ── STEP 5: Union DL + Markov into one unified table ─────────────────────
all_predictions AS (
    -- Deep learning models
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,time_bucket
        ,model
        ,sample
        ,CAST(run_timestamp AS STRING)                   AS run_timestamp
        ,pred_array
        ,true_array
        ,top5_predictions
        ,top5_scores
        ,true_labels
    FROM dl_predictions

    UNION ALL

    -- Markov
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,time_bucket
        ,model
        ,sample
        ,CAST(run_timestamp AS STRING)                   AS run_timestamp
        ,pred_array
        ,true_array
        ,top5_predictions
        ,top5_scores
        ,true_labels
    FROM markov_rows
),

-- ── STEP 6: Compute TP, FP, FN per row ───────────────────────────────────
with_tpfpfn AS (
    SELECT
        *
        -- TP: predicted providers that ARE in true labels
        ,(
            SELECT COUNT(*)
            FROM UNNEST(pred_array) AS pred
            WHERE pred IN UNNEST(true_array)
        )                                                AS tp

        -- FP: predicted providers NOT in true labels
        ,(
            SELECT COUNT(*)
            FROM UNNEST(pred_array) AS pred
            WHERE pred NOT IN UNNEST(true_array)
        )                                                AS fp

        -- FN: true label providers NOT in top-5 predictions
        ,(
            SELECT COUNT(*)
            FROM UNNEST(true_array) AS truth
            WHERE truth NOT IN UNNEST(pred_array)
        )                                                AS fn

        -- Total predictions made (up to 5)
        ,ARRAY_LENGTH(pred_array)                        AS n_predicted

        -- Total true labels
        ,ARRAY_LENGTH(true_array)                        AS n_true

    FROM all_predictions
),

-- ── STEP 7: Compute Hit@K and NDCG@K ─────────────────────────────────────
with_metrics AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,time_bucket
        ,model
        ,sample
        ,run_timestamp
        ,top5_predictions
        ,true_labels
        ,tp
        ,fp
        ,fn
        ,n_predicted
        ,n_true

        -- Hit@K — 1 if at least one of top-K predictions is in true labels
        ,CASE WHEN (
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array, 0, 1)) AS p WHERE p IN UNNEST(true_array)
        ) > 0 THEN 1.0 ELSE 0.0 END                    AS hit_at_1
        ,CASE WHEN (
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array, 0, 3)) AS p WHERE p IN UNNEST(true_array)
        ) > 0 THEN 1.0 ELSE 0.0 END                    AS hit_at_3
        ,CASE WHEN (
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array, 0, 5)) AS p WHERE p IN UNNEST(true_array)
        ) > 0 THEN 1.0 ELSE 0.0 END                    AS hit_at_5

        -- NDCG@K — DCG / IDCG
        -- DCG: sum of 1/log2(rank+1) for hits in top-K
        -- IDCG: ideal DCG if top min(|true|, K) are all correct
        ,ROUND((
            SELECT SUM(1.0 / (LOG(pos + 2) / LOG(2)))
            FROM UNNEST(ARRAY_SLICE(pred_array, 0, 1)) AS pred WITH OFFSET pos
            WHERE pred IN UNNEST(true_array)
        ) / NULLIF((
            SELECT SUM(1.0 / (LOG(ip + 2) / LOG(2)))
            FROM UNNEST(GENERATE_ARRAY(0, LEAST(n_true, 1) - 1)) AS ip
        ), 0), 4)                                        AS ndcg_at_1

        ,ROUND((
            SELECT SUM(1.0 / (LOG(pos + 2) / LOG(2)))
            FROM UNNEST(ARRAY_SLICE(pred_array, 0, 3)) AS pred WITH OFFSET pos
            WHERE pred IN UNNEST(true_array)
        ) / NULLIF((
            SELECT SUM(1.0 / (LOG(ip + 2) / LOG(2)))
            FROM UNNEST(GENERATE_ARRAY(0, LEAST(n_true, 3) - 1)) AS ip
        ), 0), 4)                                        AS ndcg_at_3

        ,ROUND((
            SELECT SUM(1.0 / (LOG(pos + 2) / LOG(2)))
            FROM UNNEST(ARRAY_SLICE(pred_array, 0, 5)) AS pred WITH OFFSET pos
            WHERE pred IN UNNEST(true_array)
        ) / NULLIF((
            SELECT SUM(1.0 / (LOG(ip + 2) / LOG(2)))
            FROM UNNEST(GENERATE_ARRAY(0, LEAST(n_true, 5) - 1)) AS ip
        ), 0), 4)                                        AS ndcg_at_5

        -- Precision@K = TP in top-K / K
        ,ROUND((
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array, 0, 1)) AS p WHERE p IN UNNEST(true_array)
        ) / 1.0, 4)                                     AS precision_at_1
        ,ROUND((
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array, 0, 3)) AS p WHERE p IN UNNEST(true_array)
        ) / 3.0, 4)                                     AS precision_at_3
        ,ROUND((
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array, 0, 5)) AS p WHERE p IN UNNEST(true_array)
        ) / 5.0, 4)                                     AS precision_at_5

        -- Recall@K = TP in top-K / |true|
        ,ROUND((
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array, 0, 1)) AS p WHERE p IN UNNEST(true_array)
        ) / NULLIF(n_true, 0), 4)                       AS recall_at_1
        ,ROUND((
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array, 0, 3)) AS p WHERE p IN UNNEST(true_array)
        ) / NULLIF(n_true, 0), 4)                       AS recall_at_3
        ,ROUND((
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array, 0, 5)) AS p WHERE p IN UNNEST(true_array)
        ) / NULLIF(n_true, 0), 4)                       AS recall_at_5

    FROM with_tpfpfn
)

SELECT * FROM with_metrics
ORDER BY model, time_bucket, member_id, trigger_date
;


-- ══════════════════════════════════════════════════════════════════════════════
-- SUMMARY VIEW — Aggregate metrics per model per window
-- Run this after the table is created to get the comparison table
-- ══════════════════════════════════════════════════════════════════════════════
-- SELECT
--     model
--     ,time_bucket
--     ,COUNT(*)                        AS n_triggers
--     ,SUM(tp)                         AS total_tp
--     ,SUM(fp)                         AS total_fp
--     ,SUM(fn)                         AS total_fn
--     ,ROUND(AVG(hit_at_1), 4)         AS hit_at_1
--     ,ROUND(AVG(hit_at_3), 4)         AS hit_at_3
--     ,ROUND(AVG(hit_at_5), 4)         AS hit_at_5
--     ,ROUND(AVG(ndcg_at_1), 4)        AS ndcg_at_1
--     ,ROUND(AVG(ndcg_at_3), 4)        AS ndcg_at_3
--     ,ROUND(AVG(ndcg_at_5), 4)        AS ndcg_at_5
--     ,ROUND(AVG(precision_at_3), 4)   AS precision_at_3
--     ,ROUND(AVG(recall_at_3), 4)      AS recall_at_3
-- FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_5pct`
-- GROUP BY model, time_bucket
-- ORDER BY time_bucket, model
;
