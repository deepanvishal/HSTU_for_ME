-- ============================================================
-- Markov_trigger_scores_5pct.sql
-- Purpose : Roll up line-level Markov predictions to trigger level
--           Score against true labels — same schema as trigger_scores
--           Written to SEPARATE table A870800_gen_rec_markov_trigger_scores_5pct
-- Sources : A870800_gen_rec_markov_predictions_5pct
--           A870800_gen_rec_markov_true_labels_5pct
-- Note    : trigger_date is DATE in both source tables — CAST to STRING on output
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_trigger_scores_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH

rolled AS (
    SELECT
        member_id
        ,CAST(trigger_date AS STRING)                    AS trigger_date
        ,trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,ARRAY_AGG(predicted_specialty
            ORDER BY prediction_rank ASC)                AS preds
        ,ARRAY_AGG(CAST(probability AS FLOAT64)
            ORDER BY prediction_rank ASC)                AS scores
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_predictions_5pct`
    WHERE predicted_specialty IS NOT NULL
    GROUP BY
        member_id, trigger_date, trigger_dx, member_segment
        ,is_t30_qualified, is_t60_qualified, is_t180_qualified
),

labels AS (
    SELECT
        member_id
        ,CAST(trigger_date AS STRING)                    AS trigger_date
        ,trigger_dx
        ,member_segment
        ,time_bucket
        ,true_label_set
        ,true_label_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_true_labels_5pct`
),

joined AS (
    SELECT
        r.member_id
        ,r.trigger_date
        ,r.trigger_dx
        ,r.member_segment
        ,l.time_bucket
        ,l.true_label_set
        ,l.true_label_count
        ,r.preds
        ,r.scores
    FROM rolled r
    INNER JOIN labels l
        ON  r.member_id      = l.member_id
        AND r.trigger_date   = l.trigger_date
        AND r.trigger_dx     = l.trigger_dx
        AND r.member_segment = l.member_segment
    WHERE
        (l.time_bucket = 'T0_30'   AND r.is_t30_qualified  = TRUE)
     OR (l.time_bucket = 'T30_60'  AND r.is_t60_qualified  = TRUE)
     OR (l.time_bucket = 'T60_180' AND r.is_t180_qualified = TRUE)
),

scored AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,time_bucket

        ,ARRAY_TO_STRING(ARRAY(
            SELECT sp FROM UNNEST(true_label_set) AS sp ORDER BY sp
        ), '|')                                          AS true_labels

        ,ARRAY_TO_STRING(ARRAY(
            SELECT s FROM UNNEST(preds) AS s WITH OFFSET pos
            WHERE pos < 5
            ORDER BY pos
        ), '|')                                          AS top5_predictions

        ,ARRAY_TO_STRING(ARRAY(
            SELECT CAST(ROUND(s, 4) AS STRING)
            FROM UNNEST(scores) AS s WITH OFFSET pos
            WHERE pos < 5
            ORDER BY pos
        ), '|')                                          AS top5_scores

        ,CASE WHEN (
            SELECT COUNT(*) FROM UNNEST(preds) AS p WITH OFFSET pos
            WHERE pos < 1 AND p IN UNNEST(true_label_set)
        ) > 0 THEN 1.0 ELSE 0.0 END                     AS hit_at_1

        ,CASE WHEN (
            SELECT COUNT(*) FROM UNNEST(preds) AS p WITH OFFSET pos
            WHERE pos < 3 AND p IN UNNEST(true_label_set)
        ) > 0 THEN 1.0 ELSE 0.0 END                     AS hit_at_3

        ,CASE WHEN (
            SELECT COUNT(*) FROM UNNEST(preds) AS p WITH OFFSET pos
            WHERE pos < 5 AND p IN UNNEST(true_label_set)
        ) > 0 THEN 1.0 ELSE 0.0 END                     AS hit_at_5

        ,ROUND((
            SELECT
                SUM(CASE WHEN p IN UNNEST(true_label_set)
                         THEN 1.0 / (LOG(pos + 2) / LOG(2))
                         ELSE 0.0 END)
                / NULLIF((
                    SELECT SUM(1.0 / (LOG(ideal + 2) / LOG(2)))
                    FROM UNNEST(GENERATE_ARRAY(0, LEAST(true_label_count, 1) - 1)) AS ideal
                ), 0)
            FROM UNNEST(preds) AS p WITH OFFSET pos
            WHERE pos < 1
        ), 4)                                            AS ndcg_at_1

        ,ROUND((
            SELECT
                SUM(CASE WHEN p IN UNNEST(true_label_set)
                         THEN 1.0 / (LOG(pos + 2) / LOG(2))
                         ELSE 0.0 END)
                / NULLIF((
                    SELECT SUM(1.0 / (LOG(ideal + 2) / LOG(2)))
                    FROM UNNEST(GENERATE_ARRAY(0, LEAST(true_label_count, 3) - 1)) AS ideal
                ), 0)
            FROM UNNEST(preds) AS p WITH OFFSET pos
            WHERE pos < 3
        ), 4)                                            AS ndcg_at_3

        ,ROUND((
            SELECT
                SUM(CASE WHEN p IN UNNEST(true_label_set)
                         THEN 1.0 / (LOG(pos + 2) / LOG(2))
                         ELSE 0.0 END)
                / NULLIF((
                    SELECT SUM(1.0 / (LOG(ideal + 2) / LOG(2)))
                    FROM UNNEST(GENERATE_ARRAY(0, LEAST(true_label_count, 5) - 1)) AS ideal
                ), 0)
            FROM UNNEST(preds) AS p WITH OFFSET pos
            WHERE pos < 5
        ), 4)                                            AS ndcg_at_5

        ,'Markov'                                        AS model
        ,'5pct'                                          AS sample
        ,FORMAT_TIMESTAMP('%Y-%m-%d_%H-%M-%S',
            CURRENT_TIMESTAMP(), 'UTC')                  AS run_timestamp

    FROM joined
)

SELECT * FROM scored
