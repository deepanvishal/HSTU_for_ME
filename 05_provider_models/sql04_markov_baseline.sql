-- ============================================================
-- SQL_04 — PROVIDER MARKOV BASELINE PER SAMPLE SIZE
-- Purpose : Transition probability + binary entropy baseline
--           Top-5 provider predictions per test trigger
--           Metrics: Hit@K, Precision@K, Recall@K, NDCG@K
--           K = 1, 3, 5 | Windows: T0_30, T30_60, T60_180
-- Mirrors : Model_Markov.sql
-- Sources : A870800_gen_rec_provider_markov_train_{pct}
--           A870800_gen_rec_provider_model_test_{pct}
-- Changes vs existing:
--   Predictions at (trigger_dx, member_segment, from_provider) grain
--   Fallback to (trigger_dx, member_segment) when from_provider OOV
--   Binary entropy — NOT standard entropy
--     probabilities per (trigger_dx, from_provider) do not sum to 1
--     because multiple providers can be visited on same day
--   Single prediction set applied across all three time windows
-- ============================================================


-- ══════════════════════════════════════════════════════════════════════════════
-- 1 PCT TABLES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── PROVIDER MARKOV PREDICTIONS 1PCT ─────────────────────────────────────────
-- Top-5 providers per (trigger_dx, member_segment, from_provider)
-- Fallback: if from_provider OOV in markov_train,
--           use (trigger_dx, member_segment) aggregate predictions
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_predictions_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH train_probs AS (
    -- P(to_provider | trigger_dx, member_segment, from_provider)
    -- Probability = transition_count / sum over all to_providers for this (dx, seg, from)
    -- NOTE: probabilities do NOT sum to 1 — multiple providers per day means
    --       a member may visit several providers, not just the argmax
    SELECT
        trigger_dx
        ,member_segment
        ,from_provider
        ,to_provider
        ,transition_count
        ,SUM(transition_count) OVER (
            PARTITION BY trigger_dx, member_segment, from_provider
        )                                                AS from_total
        ,ROUND(transition_count * 1.0 /
            SUM(transition_count) OVER (
                PARTITION BY trigger_dx, member_segment, from_provider
            ), 6)                                        AS probability
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_1pct`
    WHERE to_provider IS NOT NULL
),

ranked_probs AS (
    -- Rank to_providers by probability per (trigger_dx, member_segment, from_provider)
    SELECT
        trigger_dx
        ,member_segment
        ,from_provider
        ,to_provider
        ,probability
        ,ROW_NUMBER() OVER (
            PARTITION BY trigger_dx, member_segment, from_provider
            ORDER BY probability DESC
        )                                                AS rank
    FROM train_probs
),

-- Fallback: (trigger_dx, member_segment) aggregate when from_provider OOV
fallback_probs AS (
    SELECT
        trigger_dx
        ,member_segment
        ,to_provider
        ,SUM(transition_count)                           AS total_count
        ,ROUND(SUM(transition_count) * 1.0 /
            SUM(SUM(transition_count)) OVER (
                PARTITION BY trigger_dx, member_segment
            ), 6)                                        AS probability
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_1pct`
    WHERE to_provider IS NOT NULL
    GROUP BY trigger_dx, member_segment, to_provider
),

ranked_fallback AS (
    SELECT
        trigger_dx
        ,member_segment
        ,to_provider
        ,probability
        ,ROW_NUMBER() OVER (
            PARTITION BY trigger_dx, member_segment
            ORDER BY probability DESC
        )                                                AS rank
    FROM fallback_probs
),

test_triggers AS (
    SELECT DISTINCT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,from_provider
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_1pct`
),

-- Flag which triggers have from_provider in markov_train
triggers_with_from AS (
    SELECT DISTINCT
        trigger_dx
        ,member_segment
        ,from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_1pct`
)

SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.from_provider
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,COALESCE(p.to_provider, fb.to_provider)             AS predicted_provider
    ,COALESCE(p.probability,  fb.probability)             AS probability
    ,COALESCE(p.rank,         fb.rank)                   AS prediction_rank
    ,CASE
        WHEN tf.from_provider IS NOT NULL THEN 'from_provider'
        ELSE 'fallback_dx_segment'
     END                                                 AS prediction_source
FROM test_triggers t
LEFT JOIN triggers_with_from tf
    ON  tf.trigger_dx     = t.trigger_dx
    AND tf.member_segment = t.member_segment
    AND tf.from_provider  = t.from_provider
-- Primary: use from_provider predictions when available
LEFT JOIN ranked_probs p
    ON  p.trigger_dx     = t.trigger_dx
    AND p.member_segment = t.member_segment
    AND p.from_provider  = t.from_provider
    AND tf.from_provider IS NOT NULL    -- only join when from_provider known
    AND p.rank           <= 5
-- Fallback: use dx+segment predictions when from_provider OOV
LEFT JOIN ranked_fallback fb
    ON  fb.trigger_dx     = t.trigger_dx
    AND fb.member_segment = t.member_segment
    AND tf.from_provider  IS NULL       -- only use fallback when OOV
    AND fb.rank           <= 5
WHERE COALESCE(p.to_provider, fb.to_provider) IS NOT NULL
;


-- ── PROVIDER MARKOV ENTROPY 1PCT ──────────────────────────────────────────────
-- Binary entropy per (trigger_dx, from_provider, to_provider) pair
-- Binary entropy H = -p*log2(p) - (1-p)*log2(1-p)
-- Used as uncertainty signal — NOT standard entropy
-- Standard entropy invalid here because P(to | from, dx) do not sum to 1
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_entropy_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH probs AS (
    SELECT
        trigger_dx
        ,member_segment
        ,from_provider
        ,to_provider
        ,transition_count
        ,ROUND(transition_count * 1.0 /
            SUM(transition_count) OVER (
                PARTITION BY trigger_dx, member_segment, from_provider
            ), 6)                                        AS probability
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_1pct`
    WHERE to_provider IS NOT NULL
)

SELECT
    trigger_dx
    ,member_segment
    ,from_provider
    ,to_provider
    ,transition_count
    ,probability
    -- Binary entropy per pair: H(p) = -p*log2(p) - (1-p)*log2(1-p)
    -- Clamp p to (0,1) exclusive to avoid log(0)
    ,ROUND(
        -(GREATEST(probability, 0.000001) *
          LOG(GREATEST(probability, 0.000001)) / LOG(2))
        -((1 - GREATEST(probability, 0.000001)) *
          LOG(GREATEST(1 - GREATEST(probability, 0.000001), 0.000001)) / LOG(2))
    , 6)                                                 AS binary_entropy
    -- Mean binary entropy per (trigger_dx, from_provider) — uncertainty signal
    ,ROUND(AVG(
        -(GREATEST(probability, 0.000001) *
          LOG(GREATEST(probability, 0.000001)) / LOG(2))
        -((1 - GREATEST(probability, 0.000001)) *
          LOG(GREATEST(1 - GREATEST(probability, 0.000001), 0.000001)) / LOG(2))
    ) OVER (
        PARTITION BY trigger_dx, member_segment, from_provider
    ), 6)                                                AS mean_binary_entropy
FROM probs
;


-- ── PROVIDER MARKOV TRUE LABELS 1PCT ─────────────────────────────────────────
-- All providers visited per member + trigger + time_bucket — ground truth
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_true_labels_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    member_id
    ,trigger_date
    ,trigger_dx
    ,member_segment
    ,time_bucket
    ,ARRAY_AGG(DISTINCT label_provider
        ORDER BY label_provider)                         AS true_label_set
    ,COUNT(DISTINCT label_provider)                      AS true_label_count
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_1pct`
WHERE label_provider IS NOT NULL
GROUP BY
    member_id, trigger_date, trigger_dx, member_segment, time_bucket
;


-- ── PROVIDER MARKOV METRICS 1PCT ─────────────────────────────────────────────
-- Hit@K, Precision@K, Recall@K, NDCG@K — mirrors existing Model_Markov.sql
-- Single prediction set applied across all three windows
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_metrics_1pct`
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
            predicted_provider
            ORDER BY prediction_rank
        )                                                AS predicted_providers
        ,COUNT(predicted_provider)                       AS prediction_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_predictions_1pct`
    WHERE predicted_provider IS NOT NULL
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
        ,p.predicted_providers
        ,p.prediction_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_true_labels_1pct` t
    INNER JOIN predictions_agg p
        ON  t.member_id      = p.member_id
        AND t.trigger_date   = p.trigger_date
        AND t.trigger_dx     = p.trigger_dx
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
        ,j.predicted_providers
        ,j.prediction_count
        ,k_val
        ,(
            SELECT COUNT(*)
            FROM UNNEST(j.predicted_providers) AS pred WITH OFFSET pos
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
    CROSS JOIN UNNEST(p.predicted_providers) AS pred WITH OFFSET pos
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
ORDER BY time_bucket, k, member_segment
;


-- ══════════════════════════════════════════════════════════════════════════════
-- 5 PCT TABLES
-- (identical logic — only sample suffix changes)
-- ══════════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_predictions_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH train_probs AS (
    SELECT
        trigger_dx, member_segment, from_provider, to_provider, transition_count
        ,ROUND(transition_count * 1.0 /
            SUM(transition_count) OVER (PARTITION BY trigger_dx, member_segment, from_provider)
        , 6)                                             AS probability
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_5pct`
    WHERE to_provider IS NOT NULL
),
ranked_probs AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY trigger_dx, member_segment, from_provider ORDER BY probability DESC
    ) AS rank FROM train_probs
),
fallback_probs AS (
    SELECT trigger_dx, member_segment, to_provider
        ,ROUND(SUM(transition_count) * 1.0 /
            SUM(SUM(transition_count)) OVER (PARTITION BY trigger_dx, member_segment), 6) AS probability
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_5pct`
    WHERE to_provider IS NOT NULL
    GROUP BY trigger_dx, member_segment, to_provider
),
ranked_fallback AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY trigger_dx, member_segment ORDER BY probability DESC
    ) AS rank FROM fallback_probs
),
test_triggers AS (
    SELECT DISTINCT member_id, trigger_date, trigger_dx, member_segment,
        from_provider, is_t30_qualified, is_t60_qualified, is_t180_qualified
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_5pct`
),
triggers_with_from AS (
    SELECT DISTINCT trigger_dx, member_segment, from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_5pct`
)
SELECT
    t.member_id, t.trigger_date, t.trigger_dx, t.member_segment, t.from_provider,
    t.is_t30_qualified, t.is_t60_qualified, t.is_t180_qualified,
    COALESCE(p.to_provider, fb.to_provider)              AS predicted_provider,
    COALESCE(p.probability, fb.probability)               AS probability,
    COALESCE(p.rank, fb.rank)                            AS prediction_rank,
    CASE WHEN tf.from_provider IS NOT NULL THEN 'from_provider' ELSE 'fallback_dx_segment' END
                                                         AS prediction_source
FROM test_triggers t
LEFT JOIN triggers_with_from tf
    ON tf.trigger_dx = t.trigger_dx AND tf.member_segment = t.member_segment AND tf.from_provider = t.from_provider
LEFT JOIN ranked_probs p
    ON p.trigger_dx = t.trigger_dx AND p.member_segment = t.member_segment
    AND p.from_provider = t.from_provider AND tf.from_provider IS NOT NULL AND p.rank <= 5
LEFT JOIN ranked_fallback fb
    ON fb.trigger_dx = t.trigger_dx AND fb.member_segment = t.member_segment
    AND tf.from_provider IS NULL AND fb.rank <= 5
WHERE COALESCE(p.to_provider, fb.to_provider) IS NOT NULL
;

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_entropy_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH probs AS (
    SELECT trigger_dx, member_segment, from_provider, to_provider, transition_count
        ,ROUND(transition_count * 1.0 /
            SUM(transition_count) OVER (PARTITION BY trigger_dx, member_segment, from_provider), 6) AS probability
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_5pct`
    WHERE to_provider IS NOT NULL
)
SELECT
    trigger_dx, member_segment, from_provider, to_provider, transition_count, probability
    ,ROUND(-(GREATEST(probability,0.000001)*LOG(GREATEST(probability,0.000001))/LOG(2))
           -((1-GREATEST(probability,0.000001))*LOG(GREATEST(1-GREATEST(probability,0.000001),0.000001))/LOG(2)), 6)
                                                         AS binary_entropy
    ,ROUND(AVG(-(GREATEST(probability,0.000001)*LOG(GREATEST(probability,0.000001))/LOG(2))
               -((1-GREATEST(probability,0.000001))*LOG(GREATEST(1-GREATEST(probability,0.000001),0.000001))/LOG(2)))
        OVER (PARTITION BY trigger_dx, member_segment, from_provider), 6)
                                                         AS mean_binary_entropy
FROM probs
;

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_true_labels_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT member_id, trigger_date, trigger_dx, member_segment, time_bucket
    ,ARRAY_AGG(DISTINCT label_provider ORDER BY label_provider) AS true_label_set
    ,COUNT(DISTINCT label_provider)                      AS true_label_count
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_5pct`
WHERE label_provider IS NOT NULL
GROUP BY member_id, trigger_date, trigger_dx, member_segment, time_bucket
;

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_metrics_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH predictions_agg AS (
    SELECT member_id, trigger_date, trigger_dx, member_segment,
        is_t30_qualified, is_t60_qualified, is_t180_qualified,
        ARRAY_AGG(predicted_provider ORDER BY prediction_rank) AS predicted_providers,
        COUNT(predicted_provider) AS prediction_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_predictions_5pct`
    WHERE predicted_provider IS NOT NULL
    GROUP BY member_id, trigger_date, trigger_dx, member_segment, is_t30_qualified, is_t60_qualified, is_t180_qualified
),
joined AS (
    SELECT t.member_id, t.trigger_date, t.trigger_dx, t.member_segment,
        t.time_bucket, t.true_label_set, t.true_label_count,
        p.predicted_providers, p.prediction_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_true_labels_5pct` t
    INNER JOIN predictions_agg p
        ON t.member_id=p.member_id AND t.trigger_date=p.trigger_date
        AND t.trigger_dx=p.trigger_dx AND t.member_segment=p.member_segment
    WHERE (t.time_bucket='T0_30' AND p.is_t30_qualified=TRUE)
       OR (t.time_bucket='T30_60' AND p.is_t60_qualified=TRUE)
       OR (t.time_bucket='T60_180' AND p.is_t180_qualified=TRUE)
),
per_trigger_k AS (
    SELECT j.*, k_val,
        (SELECT COUNT(*) FROM UNNEST(j.predicted_providers) AS pred WITH OFFSET pos
         WHERE pos < k_val AND pred IN UNNEST(j.true_label_set)) AS hits_at_k
    FROM joined j CROSS JOIN UNNEST([1, 3, 5]) AS k_val
),
predictions_flat AS (
    SELECT p.member_id, p.trigger_date, p.trigger_dx, p.member_segment,
        p.time_bucket, p.k_val, p.true_label_set, p.true_label_count, p.hits_at_k, pred, pos
    FROM per_trigger_k p CROSS JOIN UNNEST(p.predicted_providers) AS pred WITH OFFSET pos
    WHERE pos < p.k_val
),
dcg_per_trigger AS (
    SELECT member_id, trigger_date, trigger_dx, member_segment, time_bucket, k_val,
        true_label_count, hits_at_k,
        SUM(CASE WHEN pred IN UNNEST(true_label_set) THEN 1.0/(LOG(pos+2)/LOG(2)) ELSE 0.0 END) AS dcg
    FROM predictions_flat
    GROUP BY member_id, trigger_date, trigger_dx, member_segment, time_bucket, k_val, true_label_count, hits_at_k
),
with_metrics AS (
    SELECT *, CASE WHEN hits_at_k>0 THEN 1.0 ELSE 0.0 END AS hit_at_k,
        ROUND(hits_at_k/k_val,6) AS precision_at_k,
        ROUND(hits_at_k/NULLIF(true_label_count,0),6) AS recall_at_k,
        ROUND(dcg/NULLIF((SELECT SUM(1.0/(LOG(ip+2)/LOG(2))) FROM UNNEST(GENERATE_ARRAY(0,LEAST(true_label_count,k_val)-1)) AS ip),0),6) AS ndcg_at_k
    FROM dcg_per_trigger
)
SELECT time_bucket, k_val AS k, member_segment, COUNT(*) AS n_evaluated,
    ROUND(AVG(hit_at_k),4) AS hit_at_k, ROUND(AVG(precision_at_k),4) AS precision_at_k,
    ROUND(AVG(recall_at_k),4) AS recall_at_k, ROUND(AVG(ndcg_at_k),4) AS ndcg_at_k
FROM with_metrics GROUP BY time_bucket, k_val, member_segment
UNION ALL
SELECT time_bucket, k_val AS k, 'ALL', COUNT(*), ROUND(AVG(hit_at_k),4),
    ROUND(AVG(precision_at_k),4), ROUND(AVG(recall_at_k),4), ROUND(AVG(ndcg_at_k),4)
FROM with_metrics GROUP BY time_bucket, k_val
ORDER BY time_bucket, k, member_segment
;


-- ══════════════════════════════════════════════════════════════════════════════
-- 10 PCT TABLES
-- ══════════════════════════════════════════════════════════════════════════════

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_predictions_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH train_probs AS (
    SELECT trigger_dx, member_segment, from_provider, to_provider, transition_count
        ,ROUND(transition_count * 1.0 /
            SUM(transition_count) OVER (PARTITION BY trigger_dx, member_segment, from_provider), 6) AS probability
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_10pct`
    WHERE to_provider IS NOT NULL
),
ranked_probs AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY trigger_dx, member_segment, from_provider ORDER BY probability DESC
    ) AS rank FROM train_probs
),
fallback_probs AS (
    SELECT trigger_dx, member_segment, to_provider
        ,ROUND(SUM(transition_count)*1.0/SUM(SUM(transition_count)) OVER (PARTITION BY trigger_dx, member_segment),6) AS probability
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_10pct`
    WHERE to_provider IS NOT NULL
    GROUP BY trigger_dx, member_segment, to_provider
),
ranked_fallback AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY trigger_dx, member_segment ORDER BY probability DESC) AS rank
    FROM fallback_probs
),
test_triggers AS (
    SELECT DISTINCT member_id, trigger_date, trigger_dx, member_segment,
        from_provider, is_t30_qualified, is_t60_qualified, is_t180_qualified
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_10pct`
),
triggers_with_from AS (
    SELECT DISTINCT trigger_dx, member_segment, from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_10pct`
)
SELECT
    t.member_id, t.trigger_date, t.trigger_dx, t.member_segment, t.from_provider,
    t.is_t30_qualified, t.is_t60_qualified, t.is_t180_qualified,
    COALESCE(p.to_provider, fb.to_provider)              AS predicted_provider,
    COALESCE(p.probability, fb.probability)               AS probability,
    COALESCE(p.rank, fb.rank)                            AS prediction_rank,
    CASE WHEN tf.from_provider IS NOT NULL THEN 'from_provider' ELSE 'fallback_dx_segment' END
                                                         AS prediction_source
FROM test_triggers t
LEFT JOIN triggers_with_from tf
    ON tf.trigger_dx=t.trigger_dx AND tf.member_segment=t.member_segment AND tf.from_provider=t.from_provider
LEFT JOIN ranked_probs p
    ON p.trigger_dx=t.trigger_dx AND p.member_segment=t.member_segment
    AND p.from_provider=t.from_provider AND tf.from_provider IS NOT NULL AND p.rank<=5
LEFT JOIN ranked_fallback fb
    ON fb.trigger_dx=t.trigger_dx AND fb.member_segment=t.member_segment
    AND tf.from_provider IS NULL AND fb.rank<=5
WHERE COALESCE(p.to_provider, fb.to_provider) IS NOT NULL
;

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_entropy_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH probs AS (
    SELECT trigger_dx, member_segment, from_provider, to_provider, transition_count
        ,ROUND(transition_count*1.0/SUM(transition_count) OVER (PARTITION BY trigger_dx, member_segment, from_provider),6) AS probability
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_10pct`
    WHERE to_provider IS NOT NULL
)
SELECT trigger_dx, member_segment, from_provider, to_provider, transition_count, probability
    ,ROUND(-(GREATEST(probability,0.000001)*LOG(GREATEST(probability,0.000001))/LOG(2))
           -((1-GREATEST(probability,0.000001))*LOG(GREATEST(1-GREATEST(probability,0.000001),0.000001))/LOG(2)),6) AS binary_entropy
    ,ROUND(AVG(-(GREATEST(probability,0.000001)*LOG(GREATEST(probability,0.000001))/LOG(2))
               -((1-GREATEST(probability,0.000001))*LOG(GREATEST(1-GREATEST(probability,0.000001),0.000001))/LOG(2)))
        OVER (PARTITION BY trigger_dx, member_segment, from_provider),6) AS mean_binary_entropy
FROM probs
;

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_true_labels_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT member_id, trigger_date, trigger_dx, member_segment, time_bucket
    ,ARRAY_AGG(DISTINCT label_provider ORDER BY label_provider) AS true_label_set
    ,COUNT(DISTINCT label_provider) AS true_label_count
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_10pct`
WHERE label_provider IS NOT NULL
GROUP BY member_id, trigger_date, trigger_dx, member_segment, time_bucket
;

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_metrics_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH predictions_agg AS (
    SELECT member_id, trigger_date, trigger_dx, member_segment,
        is_t30_qualified, is_t60_qualified, is_t180_qualified,
        ARRAY_AGG(predicted_provider ORDER BY prediction_rank) AS predicted_providers,
        COUNT(predicted_provider) AS prediction_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_predictions_10pct`
    WHERE predicted_provider IS NOT NULL
    GROUP BY member_id, trigger_date, trigger_dx, member_segment, is_t30_qualified, is_t60_qualified, is_t180_qualified
),
joined AS (
    SELECT t.member_id, t.trigger_date, t.trigger_dx, t.member_segment,
        t.time_bucket, t.true_label_set, t.true_label_count,
        p.predicted_providers, p.prediction_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_true_labels_10pct` t
    INNER JOIN predictions_agg p
        ON t.member_id=p.member_id AND t.trigger_date=p.trigger_date
        AND t.trigger_dx=p.trigger_dx AND t.member_segment=p.member_segment
    WHERE (t.time_bucket='T0_30' AND p.is_t30_qualified=TRUE)
       OR (t.time_bucket='T30_60' AND p.is_t60_qualified=TRUE)
       OR (t.time_bucket='T60_180' AND p.is_t180_qualified=TRUE)
),
per_trigger_k AS (
    SELECT j.*, k_val,
        (SELECT COUNT(*) FROM UNNEST(j.predicted_providers) AS pred WITH OFFSET pos
         WHERE pos < k_val AND pred IN UNNEST(j.true_label_set)) AS hits_at_k
    FROM joined j CROSS JOIN UNNEST([1, 3, 5]) AS k_val
),
predictions_flat AS (
    SELECT p.member_id, p.trigger_date, p.trigger_dx, p.member_segment,
        p.time_bucket, p.k_val, p.true_label_set, p.true_label_count, p.hits_at_k, pred, pos
    FROM per_trigger_k p CROSS JOIN UNNEST(p.predicted_providers) AS pred WITH OFFSET pos
    WHERE pos < p.k_val
),
dcg_per_trigger AS (
    SELECT member_id, trigger_date, trigger_dx, member_segment, time_bucket, k_val,
        true_label_count, hits_at_k,
        SUM(CASE WHEN pred IN UNNEST(true_label_set) THEN 1.0/(LOG(pos+2)/LOG(2)) ELSE 0.0 END) AS dcg
    FROM predictions_flat
    GROUP BY member_id, trigger_date, trigger_dx, member_segment, time_bucket, k_val, true_label_count, hits_at_k
),
with_metrics AS (
    SELECT *, CASE WHEN hits_at_k>0 THEN 1.0 ELSE 0.0 END AS hit_at_k,
        ROUND(hits_at_k/k_val,6) AS precision_at_k,
        ROUND(hits_at_k/NULLIF(true_label_count,0),6) AS recall_at_k,
        ROUND(dcg/NULLIF((SELECT SUM(1.0/(LOG(ip+2)/LOG(2))) FROM UNNEST(GENERATE_ARRAY(0,LEAST(true_label_count,k_val)-1)) AS ip),0),6) AS ndcg_at_k
    FROM dcg_per_trigger
)
SELECT time_bucket, k_val AS k, member_segment, COUNT(*) AS n_evaluated,
    ROUND(AVG(hit_at_k),4) AS hit_at_k, ROUND(AVG(precision_at_k),4) AS precision_at_k,
    ROUND(AVG(recall_at_k),4) AS recall_at_k, ROUND(AVG(ndcg_at_k),4) AS ndcg_at_k
FROM with_metrics GROUP BY time_bucket, k_val, member_segment
UNION ALL
SELECT time_bucket, k_val AS k, 'ALL', COUNT(*), ROUND(AVG(hit_at_k),4),
    ROUND(AVG(precision_at_k),4), ROUND(AVG(recall_at_k),4), ROUND(AVG(ndcg_at_k),4)
FROM with_metrics GROUP BY time_bucket, k_val
ORDER BY time_bucket, k, member_segment
;
