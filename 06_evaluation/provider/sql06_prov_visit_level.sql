-- ============================================================
-- SQL_06 — VISIT LEVEL ROLLUP
-- Purpose : Roll up A870800_gen_rec_provider_eval_5pct
--           from (member, trigger_date, trigger_dx) grain
--           to (member, trigger_date) grain — visit level
-- Logic:
--   Predictions: union across all DX fired on same visit date
--                keep HIGHEST score per provider (best signal)
--                re-rank by score → top-5 at visit level
--   True labels: union of all true providers across all DX
--   Trigger DX:  ARRAY_AGG of all DX codes that fired that day
--   TP/FP/FN:    recomputed on merged visit-level sets
-- Sources : A870800_gen_rec_provider_eval_5pct  (SQL_05 output)
-- Output  : A870800_gen_rec_provider_eval_visit_5pct
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_visit_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

-- ── STEP 1: Flatten predictions to (member, date, model, window, provider, score) ──
-- Split pipe-delimited top5_predictions and top5_scores back to rows
WITH pred_flat AS (
    SELECT
        member_id
        ,trigger_date
        ,member_segment
        ,time_bucket
        ,model
        ,sample
        ,run_timestamp
        ,trigger_dx                                      -- keep for array agg later
        ,SPLIT(true_labels, '|')                         AS true_array_dx  -- per-DX true labels
        ,pred_provider
        ,CAST(pred_score AS FLOAT64)                     AS pred_score
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_5pct`
    -- Zip predictions and scores together by position
    CROSS JOIN UNNEST(SPLIT(top5_predictions, '|')) AS pred_provider
        WITH OFFSET pred_pos
    CROSS JOIN UNNEST(SPLIT(top5_scores, '|'))      AS pred_score
        WITH OFFSET score_pos
    WHERE pred_pos = score_pos                           -- align by position
      AND pred_provider != ''
      AND pred_provider IS NOT NULL
),

-- ── STEP 2: Per provider, keep HIGHEST score across all DX on same visit ──
best_pred_per_provider AS (
    SELECT
        member_id
        ,trigger_date
        ,member_segment
        ,time_bucket
        ,model
        ,sample
        ,run_timestamp
        ,pred_provider
        ,MAX(pred_score)                                 AS best_score
    FROM pred_flat
    GROUP BY
        member_id, trigger_date, member_segment,
        time_bucket, model, sample, run_timestamp,
        pred_provider
),

-- ── STEP 3: Re-rank by best_score → top-5 at visit level ─────────────────
visit_top5 AS (
    SELECT
        member_id
        ,trigger_date
        ,member_segment
        ,time_bucket
        ,model
        ,sample
        ,run_timestamp
        ,pred_provider
        ,best_score
        ,ROW_NUMBER() OVER (
            PARTITION BY member_id, trigger_date, member_segment,
                         time_bucket, model, sample
            ORDER BY best_score DESC
        )                                                AS visit_rank
    FROM best_pred_per_provider
),

-- ── STEP 4: Aggregate top-5 predictions into arrays at visit level ────────
visit_preds_agg AS (
    SELECT
        member_id
        ,trigger_date
        ,member_segment
        ,time_bucket
        ,model
        ,sample
        ,run_timestamp
        ,ARRAY_AGG(pred_provider ORDER BY visit_rank)    AS pred_array
        ,ARRAY_AGG(best_score    ORDER BY visit_rank)    AS score_array
        ,ARRAY_TO_STRING(
            ARRAY_AGG(pred_provider ORDER BY visit_rank),
            '|'
         )                                               AS top5_predictions
        ,ARRAY_TO_STRING(
            ARRAY_AGG(CAST(ROUND(best_score,4) AS STRING) ORDER BY visit_rank),
            '|'
         )                                               AS top5_scores
    FROM visit_top5
    WHERE visit_rank <= 5
    GROUP BY
        member_id, trigger_date, member_segment,
        time_bucket, model, sample, run_timestamp
),

-- ── STEP 5: Aggregate true labels + DX codes at visit level ──────────────
visit_true_agg AS (
    SELECT
        member_id
        ,trigger_date
        ,member_segment
        ,time_bucket
        ,model
        ,sample
        -- All DX codes that fired on this visit date
        ,ARRAY_AGG(DISTINCT trigger_dx
            ORDER BY trigger_dx)                         AS trigger_dx_array
        -- Union of all true providers across all DX
        ,ARRAY_AGG(DISTINCT true_provider
            ORDER BY true_provider)                      AS true_array
    FROM pred_flat
    -- Unnest per-DX true label array to get individual providers
    CROSS JOIN UNNEST(true_array_dx) AS true_provider
    WHERE true_provider != ''
      AND true_provider IS NOT NULL
    GROUP BY
        member_id, trigger_date, member_segment,
        time_bucket, model, sample
),

-- ── STEP 6: Join predictions + true labels + compute TP/FP/FN ────────────
visit_combined AS (
    SELECT
        p.member_id
        ,p.trigger_date
        ,p.member_segment
        ,p.time_bucket
        ,p.model
        ,p.sample
        ,p.run_timestamp
        ,t.trigger_dx_array
        ,ARRAY_TO_STRING(t.trigger_dx_array, '|')        AS trigger_dx_str
        ,p.top5_predictions
        ,p.top5_scores
        ,ARRAY_TO_STRING(t.true_array, '|')              AS true_labels
        ,p.pred_array
        ,t.true_array
        ,ARRAY_LENGTH(p.pred_array)                      AS n_predicted
        ,ARRAY_LENGTH(t.true_array)                      AS n_true

        -- TP: predicted providers that ARE in true labels
        ,(
            SELECT COUNT(*)
            FROM UNNEST(p.pred_array) AS pred
            WHERE pred IN UNNEST(t.true_array)
        )                                                AS tp

        -- FP: predicted providers NOT in true labels
        ,(
            SELECT COUNT(*)
            FROM UNNEST(p.pred_array) AS pred
            WHERE pred NOT IN UNNEST(t.true_array)
        )                                                AS fp

        -- FN: true providers NOT in top-5 predictions
        ,(
            SELECT COUNT(*)
            FROM UNNEST(t.true_array) AS truth
            WHERE truth NOT IN UNNEST(p.pred_array)
        )                                                AS fn

    FROM visit_preds_agg p
    JOIN visit_true_agg t
        ON  p.member_id      = t.member_id
        AND p.trigger_date   = t.trigger_date
        AND p.member_segment = t.member_segment
        AND p.time_bucket    = t.time_bucket
        AND p.model          = t.model
        AND p.sample         = t.sample
),

-- ── STEP 7: Compute Hit@K, NDCG@K, Precision@K, Recall@K ────────────────
final AS (
    SELECT
        member_id
        ,trigger_date
        ,member_segment
        ,time_bucket
        ,model
        ,sample
        ,run_timestamp
        ,trigger_dx_array
        ,trigger_dx_str
        ,top5_predictions
        ,top5_scores
        ,true_labels
        ,tp
        ,fp
        ,fn
        ,n_predicted
        ,n_true

        -- Hit@K
        ,CASE WHEN (
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,1)) p WHERE p IN UNNEST(true_array)
        ) > 0 THEN 1.0 ELSE 0.0 END                    AS hit_at_1
        ,CASE WHEN (
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,3)) p WHERE p IN UNNEST(true_array)
        ) > 0 THEN 1.0 ELSE 0.0 END                    AS hit_at_3
        ,CASE WHEN (
            SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,5)) p WHERE p IN UNNEST(true_array)
        ) > 0 THEN 1.0 ELSE 0.0 END                    AS hit_at_5

        -- NDCG@K
        ,ROUND((
            SELECT SUM(1.0/(LOG(pos+2)/LOG(2)))
            FROM UNNEST(ARRAY_SLICE(pred_array,0,1)) p WITH OFFSET pos
            WHERE p IN UNNEST(true_array)
        ) / NULLIF((
            SELECT SUM(1.0/(LOG(ip+2)/LOG(2)))
            FROM UNNEST(GENERATE_ARRAY(0,LEAST(n_true,1)-1)) ip
        ),0),4)                                         AS ndcg_at_1
        ,ROUND((
            SELECT SUM(1.0/(LOG(pos+2)/LOG(2)))
            FROM UNNEST(ARRAY_SLICE(pred_array,0,3)) p WITH OFFSET pos
            WHERE p IN UNNEST(true_array)
        ) / NULLIF((
            SELECT SUM(1.0/(LOG(ip+2)/LOG(2)))
            FROM UNNEST(GENERATE_ARRAY(0,LEAST(n_true,3)-1)) ip
        ),0),4)                                         AS ndcg_at_3
        ,ROUND((
            SELECT SUM(1.0/(LOG(pos+2)/LOG(2)))
            FROM UNNEST(ARRAY_SLICE(pred_array,0,5)) p WITH OFFSET pos
            WHERE p IN UNNEST(true_array)
        ) / NULLIF((
            SELECT SUM(1.0/(LOG(ip+2)/LOG(2)))
            FROM UNNEST(GENERATE_ARRAY(0,LEAST(n_true,5)-1)) ip
        ),0),4)                                         AS ndcg_at_5

        -- Precision@K
        ,ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,1)) p WHERE p IN UNNEST(true_array))/1.0,4) AS precision_at_1
        ,ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,3)) p WHERE p IN UNNEST(true_array))/3.0,4) AS precision_at_3
        ,ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,5)) p WHERE p IN UNNEST(true_array))/5.0,4) AS precision_at_5

        -- Recall@K
        ,ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,1)) p WHERE p IN UNNEST(true_array))/NULLIF(n_true,0),4) AS recall_at_1
        ,ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,3)) p WHERE p IN UNNEST(true_array))/NULLIF(n_true,0),4) AS recall_at_3
        ,ROUND((SELECT COUNT(*) FROM UNNEST(ARRAY_SLICE(pred_array,0,5)) p WHERE p IN UNNEST(true_array))/NULLIF(n_true,0),4) AS recall_at_5

    FROM visit_combined
)

SELECT * FROM final
ORDER BY model, time_bucket, member_id, trigger_date
;


-- ══════════════════════════════════════════════════════════════════════════════
-- VISIT LEVEL SUMMARY — run after table is created
-- ══════════════════════════════════════════════════════════════════════════════
-- SELECT
--     model
--     ,time_bucket
--     ,COUNT(*)                        AS n_visits
--     ,SUM(tp)                         AS total_tp
--     ,SUM(fp)                         AS total_fp
--     ,SUM(fn)                         AS total_fn
--     ,ROUND(AVG(hit_at_1), 4)         AS hit_at_1
--     ,ROUND(AVG(hit_at_3), 4)         AS hit_at_3
--     ,ROUND(AVG(hit_at_5), 4)         AS hit_at_5
--     ,ROUND(AVG(ndcg_at_3), 4)        AS ndcg_at_3
--     ,ROUND(AVG(ndcg_at_5), 4)        AS ndcg_at_5
--     ,ROUND(AVG(precision_at_3), 4)   AS precision_at_3
--     ,ROUND(AVG(recall_at_3), 4)      AS recall_at_3
-- FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_visit_5pct`
-- GROUP BY model, time_bucket
-- ORDER BY time_bucket, model
;
