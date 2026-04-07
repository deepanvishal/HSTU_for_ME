-- ============================================================
-- NB_PMA_03_diagnosis_predictor.sql
-- Purpose : Which diagnosis triggers drive highest predictions?
-- Primary metric : Hit@5, T0_30, BERT4Rec
-- Analyses:
--   Q1. Top 10 by volume
--   Q2. Top 10 by Hit@5 accuracy
--   Q3. Volume tiers (33/66 pct) — avg Hit@5 per tier
--   Q4. Top 10 by volume within each tier
-- Source  : A870800_gen_rec_analysis_perf_by_diag
--           A870800_gen_rec_markov_train (for dx descriptions)
-- ============================================================

-- ============================================================
-- Q1. Top 10 diagnosis triggers by VOLUME
--     Most common triggers — are the frequent ones predictable?
-- ============================================================
WITH dx_desc AS (
    -- Get description for each trigger_dx from markov_train
    SELECT DISTINCT
        trigger_dx
        ,trigger_dx_clean
        ,trigger_ccsr
        ,trigger_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE trigger_dx IS NOT NULL
)
SELECT
    p.trigger_dx
    ,d.trigger_dx_clean
    ,d.trigger_ccsr
    ,d.trigger_ccsr_desc                                   AS dx_description
    ,p.trigger_volume
    ,ROUND(p.hit_at_5, 4)                                  AS hit_at_5
    ,ROUND(p.ndcg_at_5, 4)                                 AS ndcg_at_5
    ,ROUND(p.hit_at_3, 4)                                  AS hit_at_3
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_diag` p
LEFT JOIN dx_desc d ON p.trigger_dx = d.trigger_dx
WHERE p.model        = 'BERT4Rec'
  AND p.time_bucket  = 'T0_30'
  AND p.trigger_volume >= 20
ORDER BY p.trigger_volume DESC
LIMIT 10
;

-- ============================================================
-- Q2. Top 10 diagnosis triggers by HIT@5 ACCURACY
--     Most predictable triggers — what are these conditions?
-- ============================================================
WITH dx_desc AS (
    SELECT DISTINCT
        trigger_dx
        ,trigger_dx_clean
        ,trigger_ccsr
        ,trigger_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE trigger_dx IS NOT NULL
)
SELECT
    p.trigger_dx
    ,d.trigger_dx_clean
    ,d.trigger_ccsr
    ,d.trigger_ccsr_desc                                   AS dx_description
    ,p.trigger_volume
    ,ROUND(p.hit_at_5, 4)                                  AS hit_at_5
    ,ROUND(p.ndcg_at_5, 4)                                 AS ndcg_at_5
    ,ROUND(p.hit_at_3, 4)                                  AS hit_at_3
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_diag` p
LEFT JOIN dx_desc d ON p.trigger_dx = d.trigger_dx
WHERE p.model        = 'BERT4Rec'
  AND p.time_bucket  = 'T0_30'
  AND p.trigger_volume >= 20
ORDER BY p.hit_at_5 DESC
LIMIT 10
;

-- ============================================================
-- Q3. Volume tiers — 33/66 pct cutoffs
--     Assign Low / Medium / High based on trigger_volume
--     Show avg Hit@5 per tier
-- ============================================================
WITH base AS (
    SELECT
        trigger_dx
        ,trigger_volume
        ,hit_at_5
        ,ndcg_at_5
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_diag`
    WHERE model        = 'BERT4Rec'
      AND time_bucket  = 'T0_30'
      AND trigger_volume >= 20
),
percentiles AS (
    SELECT
        APPROX_QUANTILES(trigger_volume, 100)[OFFSET(33)]  AS p33
        ,APPROX_QUANTILES(trigger_volume, 100)[OFFSET(66)] AS p66
    FROM base
),
tiered AS (
    SELECT
        b.trigger_dx
        ,b.trigger_volume
        ,b.hit_at_5
        ,b.ndcg_at_5
        ,CASE
            WHEN b.trigger_volume <= p.p33 THEN '1_Low'
            WHEN b.trigger_volume <= p.p66 THEN '2_Medium'
            ELSE                               '3_High'
        END                                                AS volume_tier
    FROM base b
    CROSS JOIN percentiles p
)
SELECT
    volume_tier
    ,COUNT(DISTINCT trigger_dx)                            AS dx_count
    ,SUM(trigger_volume)                                   AS total_volume
    ,MIN(trigger_volume)                                   AS min_volume
    ,MAX(trigger_volume)                                   AS max_volume
    ,ROUND(AVG(hit_at_5), 4)                               AS avg_hit_at_5
    ,ROUND(MIN(hit_at_5), 4)                               AS min_hit_at_5
    ,ROUND(MAX(hit_at_5), 4)                               AS max_hit_at_5
    ,ROUND(AVG(ndcg_at_5), 4)                              AS avg_ndcg_at_5
FROM tiered
GROUP BY volume_tier
ORDER BY volume_tier
;

-- ============================================================
-- Q4. Top 10 by Hit@5 within each volume tier
--     Shows whether high volume tiers are also more accurate
-- ============================================================
WITH dx_desc AS (
    SELECT DISTINCT
        trigger_dx
        ,trigger_dx_clean
        ,trigger_ccsr
        ,trigger_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE trigger_dx IS NOT NULL
),
percentiles AS (
    SELECT
        APPROX_QUANTILES(trigger_volume, 100)[OFFSET(33)]  AS p33
        ,APPROX_QUANTILES(trigger_volume, 100)[OFFSET(66)] AS p66
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_diag`
    WHERE model       = 'BERT4Rec'
      AND time_bucket = 'T0_30'
      AND trigger_volume >= 20
),
tiered AS (
    SELECT
        p.trigger_dx
        ,p.trigger_volume
        ,p.hit_at_5
        ,p.ndcg_at_5
        ,p.hit_at_3
        ,CASE
            WHEN p.trigger_volume <= pct.p33 THEN '1_Low'
            WHEN p.trigger_volume <= pct.p66 THEN '2_Medium'
            ELSE                                  '3_High'
        END                                                AS volume_tier
        ,ROW_NUMBER() OVER (
            PARTITION BY
                CASE
                    WHEN p.trigger_volume <= pct.p33 THEN '1_Low'
                    WHEN p.trigger_volume <= pct.p66 THEN '2_Medium'
                    ELSE                                  '3_High'
                END
            ORDER BY p.hit_at_5 DESC
        )                                                  AS rank_within_tier
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_diag` p
    CROSS JOIN percentiles pct
    WHERE p.model       = 'BERT4Rec'
      AND p.time_bucket = 'T0_30'
      AND p.trigger_volume >= 20
)
SELECT
    t.volume_tier
    ,t.rank_within_tier
    ,t.trigger_dx
    ,d.trigger_dx_clean
    ,d.trigger_ccsr
    ,d.trigger_ccsr_desc                                   AS dx_description
    ,t.trigger_volume
    ,ROUND(t.hit_at_5, 4)                                  AS hit_at_5
    ,ROUND(t.ndcg_at_5, 4)                                 AS ndcg_at_5
    ,ROUND(t.hit_at_3, 4)                                  AS hit_at_3
FROM tiered t
LEFT JOIN dx_desc d ON t.trigger_dx = d.trigger_dx
WHERE t.rank_within_tier <= 10
ORDER BY t.volume_tier, t.rank_within_tier
;
