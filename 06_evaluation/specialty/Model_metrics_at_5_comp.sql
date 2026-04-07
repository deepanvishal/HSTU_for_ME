-- ============================================================
-- Analysis_Layer1_perf_all_metrics.sql
-- Purpose : Full metrics table — Hit, NDCG, Precision, Recall
--           at K=1, 3, 5 for all models × windows × segments
--           Computed from pipe-string columns in trigger_scores
-- New table — does not touch existing analysis tables
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_full`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH

-- ── Combine both score tables ─────────────────────────────
all_scores AS (
    SELECT
        model, time_bucket, member_segment
        ,true_labels
        ,top5_predictions
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_1, ndcg_at_3, ndcg_at_5
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_scores`

    UNION ALL

    SELECT
        model, time_bucket, member_segment
        ,true_labels
        ,top5_predictions
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_1, ndcg_at_3, ndcg_at_5
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_trigger_scores`
),

-- ── Parse pipe strings, compute hits_at_k and true_label_count ──
parsed AS (
    SELECT
        model
        ,time_bucket
        ,member_segment
        ,hit_at_1, hit_at_3, hit_at_5
        ,ndcg_at_1, ndcg_at_3, ndcg_at_5

        -- True label count from pipe string
        ,ARRAY_LENGTH(SPLIT(true_labels, '|'))           AS true_label_count

        -- hits_at_k = count of predictions in top-k that are in true label set
        ,(SELECT COUNT(*)
          FROM UNNEST(SPLIT(top5_predictions, '|')) AS pred WITH OFFSET pos
          WHERE pos < 1
            AND pred IN UNNEST(SPLIT(true_labels, '|')))  AS hits_at_1

        ,(SELECT COUNT(*)
          FROM UNNEST(SPLIT(top5_predictions, '|')) AS pred WITH OFFSET pos
          WHERE pos < 3
            AND pred IN UNNEST(SPLIT(true_labels, '|')))  AS hits_at_3

        ,(SELECT COUNT(*)
          FROM UNNEST(SPLIT(top5_predictions, '|')) AS pred WITH OFFSET pos
          WHERE pos < 5
            AND pred IN UNNEST(SPLIT(true_labels, '|')))  AS hits_at_5

    FROM all_scores
    WHERE true_labels  IS NOT NULL AND true_labels  != ''
      AND top5_predictions IS NOT NULL AND top5_predictions != ''
),

-- ── Per-segment aggregation ───────────────────────────────
by_segment AS (
    SELECT
        model
        ,time_bucket
        ,member_segment
        ,COUNT(*)                                        AS n_triggers

        -- Hit@K (already binary — just average)
        ,ROUND(AVG(hit_at_1), 4)                         AS hit_at_1
        ,ROUND(AVG(hit_at_3), 4)                         AS hit_at_3
        ,ROUND(AVG(hit_at_5), 4)                         AS hit_at_5

        -- NDCG@K
        ,ROUND(AVG(ndcg_at_1), 4)                        AS ndcg_at_1
        ,ROUND(AVG(ndcg_at_3), 4)                        AS ndcg_at_3
        ,ROUND(AVG(ndcg_at_5), 4)                        AS ndcg_at_5

        -- Precision@K = hits_at_k / k
        ,ROUND(AVG(hits_at_1 / 1.0), 4)                  AS precision_at_1
        ,ROUND(AVG(hits_at_3 / 3.0), 4)                  AS precision_at_3
        ,ROUND(AVG(hits_at_5 / 5.0), 4)                  AS precision_at_5

        -- Recall@K = hits_at_k / true_label_count
        ,ROUND(AVG(SAFE_DIVIDE(hits_at_1,
            NULLIF(true_label_count, 0))), 4)             AS recall_at_1
        ,ROUND(AVG(SAFE_DIVIDE(hits_at_3,
            NULLIF(true_label_count, 0))), 4)             AS recall_at_3
        ,ROUND(AVG(SAFE_DIVIDE(hits_at_5,
            NULLIF(true_label_count, 0))), 4)             AS recall_at_5

    FROM parsed
    GROUP BY model, time_bucket, member_segment
),

-- ── Overall (ALL segments) ────────────────────────────────
overall AS (
    SELECT
        model
        ,time_bucket
        ,'ALL'                                           AS member_segment
        ,COUNT(*)                                        AS n_triggers

        ,ROUND(AVG(hit_at_1), 4)                         AS hit_at_1
        ,ROUND(AVG(hit_at_3), 4)                         AS hit_at_3
        ,ROUND(AVG(hit_at_5), 4)                         AS hit_at_5

        ,ROUND(AVG(ndcg_at_1), 4)                        AS ndcg_at_1
        ,ROUND(AVG(ndcg_at_3), 4)                        AS ndcg_at_3
        ,ROUND(AVG(ndcg_at_5), 4)                        AS ndcg_at_5

        ,ROUND(AVG(hits_at_1 / 1.0), 4)                  AS precision_at_1
        ,ROUND(AVG(hits_at_3 / 3.0), 4)                  AS precision_at_3
        ,ROUND(AVG(hits_at_5 / 5.0), 4)                  AS precision_at_5

        ,ROUND(AVG(SAFE_DIVIDE(hits_at_1,
            NULLIF(true_label_count, 0))), 4)             AS recall_at_1
        ,ROUND(AVG(SAFE_DIVIDE(hits_at_3,
            NULLIF(true_label_count, 0))), 4)             AS recall_at_3
        ,ROUND(AVG(SAFE_DIVIDE(hits_at_5,
            NULLIF(true_label_count, 0))), 4)             AS recall_at_5

    FROM parsed
    GROUP BY model, time_bucket
)

SELECT * FROM by_segment
UNION ALL
SELECT * FROM overall
ORDER BY model, time_bucket, member_segment
