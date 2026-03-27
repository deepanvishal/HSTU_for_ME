-- ============================================================
-- MARKOV 5PCT — PREDICTIONS
-- Purpose : Top 5 specialty predictions per test trigger
--           using 5% sample ONLY — apples to apples
-- Source  : markov_train_5pct, model_test_5pct
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_predictions_5pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_predictions_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH train_probs AS (
    SELECT
        trigger_dx
        ,member_segment
        ,next_specialty
        ,transition_count
        ,SUM(transition_count) OVER (
            PARTITION BY trigger_dx, member_segment
        )                                                AS dx_total
        ,ROUND(transition_count /
            SUM(transition_count) OVER (
                PARTITION BY trigger_dx, member_segment
            ), 6)                                        AS probability
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train_5pct`
    WHERE next_specialty IS NOT NULL
),
ranked_probs AS (
    SELECT
        trigger_dx
        ,member_segment
        ,next_specialty
        ,probability
        ,ROW_NUMBER() OVER (
            PARTITION BY trigger_dx, member_segment
            ORDER BY probability DESC
        )                                                AS rank
    FROM train_probs
),
test_triggers AS (
    SELECT DISTINCT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test_5pct`
)
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,p.next_specialty                                    AS predicted_specialty
    ,p.probability
    ,p.rank                                              AS prediction_rank
FROM test_triggers t
LEFT JOIN ranked_probs p
    ON t.trigger_dx = p.trigger_dx
    AND t.member_segment = p.member_segment
    AND p.rank <= 5;


-- ============================================================
-- MARKOV 5PCT — TRUE LABEL SETS
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_true_labels_5pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_true_labels_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    member_id
    ,trigger_date
    ,trigger_dx
    ,member_segment
    ,time_bucket
    ,ARRAY_AGG(DISTINCT label_specialty
        ORDER BY label_specialty)                        AS true_label_set
    ,COUNT(DISTINCT label_specialty)                     AS true_label_count
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test_5pct`
WHERE label_specialty IS NOT NULL
GROUP BY
    member_id, trigger_date, trigger_dx,
    member_segment, time_bucket;


-- ============================================================
-- MARKOV 5PCT — METRICS
-- Hit@K, Precision@K, Recall@K, NDCG@K
-- K = 1, 3, 5
-- Per time window, per segment + overall
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_metrics_5pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_metrics_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH predictions_agg AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,ARRAY_AGG(
            predicted_specialty
            ORDER BY prediction_rank
        )                                                AS predicted_specialties
        ,COUNT(predicted_specialty)                      AS prediction_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_predictions_5pct`
    WHERE predicted_specialty IS NOT NULL
    GROUP BY
        member_id, trigger_date, trigger_dx, member_segment
        ,is_t30_qualified, is_t60_qualified, is_t180_qualified
),
joined AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.time_bucket
        ,t.true_label_set
        ,t.true_label_count
        ,p.predicted_specialties
        ,p.prediction_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_true_labels_5pct` t
    INNER JOIN predictions_agg p
        ON t.member_id = p.member_id
        AND t.trigger_date = p.trigger_date
        AND t.trigger_dx = p.trigger_dx
        AND t.member_segment = p.member_segment
    WHERE
        (t.time_bucket = 'T0_30'   AND p.is_t30_qualified  = TRUE)
     OR (t.time_bucket = 'T30_60'  AND p.is_t60_qualified  = TRUE)
     OR (t.time_bucket = 'T60_180' AND p.is_t180_qualified = TRUE)
),
per_trigger_k AS (
    SELECT
        j.member_id
        ,j.trigger_date
        ,j.trigger_dx
        ,j.member_segment
        ,j.time_bucket
        ,j.true_label_count
        ,j.true_label_set
        ,j.predicted_specialties
        ,j.prediction_count
        ,k_val
        ,(
            SELECT COUNT(*)
            FROM UNNEST(j.predicted_specialties) AS pred WITH OFFSET pos
            WHERE pos < k_val
              AND pred IN UNNEST(j.true_label_set)
        )                                                AS hits_at_k
    FROM joined j
    CROSS JOIN UNNEST([1, 3, 5]) AS k_val
),
predictions_flat AS (
    SELECT
        p.member_id
        ,p.trigger_date
        ,p.trigger_dx
        ,p.member_segment
        ,p.time_bucket
        ,p.k_val
        ,p.true_label_set
        ,p.true_label_count
        ,p.hits_at_k
        ,pred
        ,pos
    FROM per_trigger_k p
    CROSS JOIN UNNEST(p.predicted_specialties) AS pred WITH OFFSET pos
    WHERE pos < p.k_val
),
dcg_per_trigger AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,time_bucket
        ,k_val
        ,true_label_count
        ,hits_at_k
        ,SUM(
            CASE WHEN pred IN UNNEST(true_label_set)
                 THEN 1.0 / (LOG(pos + 2) / LOG(2))
                 ELSE 0.0
            END
        )                                                AS dcg
    FROM predictions_flat
    GROUP BY
        member_id, trigger_date, trigger_dx
        ,member_segment, time_bucket, k_val
        ,true_label_count, hits_at_k
),
with_metrics AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,time_bucket
        ,k_val
        ,true_label_count
        ,hits_at_k
        ,CASE WHEN hits_at_k > 0 THEN 1.0 ELSE 0.0 END  AS hit_at_k
        ,ROUND(hits_at_k / k_val, 6)                     AS precision_at_k
        ,ROUND(hits_at_k / NULLIF(true_label_count, 0), 6) AS recall_at_k
        ,ROUND(
            dcg / NULLIF((
                SELECT SUM(1.0 / (LOG(ideal_pos + 2) / LOG(2)))
                FROM UNNEST(
                    GENERATE_ARRAY(0, LEAST(true_label_count, k_val) - 1)
                ) AS ideal_pos
            ), 0)
        , 6)                                             AS ndcg_at_k
    FROM dcg_per_trigger
),
metrics_by_segment AS (
    SELECT
        time_bucket
        ,k_val                                           AS k
        ,member_segment
        ,COUNT(*)                                        AS n_evaluated
        ,ROUND(AVG(hit_at_k), 4)                         AS hit_at_k
        ,ROUND(AVG(precision_at_k), 4)                   AS precision_at_k
        ,ROUND(AVG(recall_at_k), 4)                      AS recall_at_k
        ,ROUND(AVG(ndcg_at_k), 4)                        AS ndcg_at_k
    FROM with_metrics
    GROUP BY time_bucket, k_val, member_segment
),
metrics_overall AS (
    SELECT
        time_bucket
        ,k_val                                           AS k
        ,'ALL'                                           AS member_segment
        ,COUNT(*)                                        AS n_evaluated
        ,ROUND(AVG(hit_at_k), 4)                         AS hit_at_k
        ,ROUND(AVG(precision_at_k), 4)                   AS precision_at_k
        ,ROUND(AVG(recall_at_k), 4)                      AS recall_at_k
        ,ROUND(AVG(ndcg_at_k), 4)                        AS ndcg_at_k
    FROM with_metrics
    GROUP BY time_bucket, k_val
)
SELECT * FROM metrics_by_segment
UNION ALL
SELECT * FROM metrics_overall
ORDER BY time_bucket, k, member_segment;
