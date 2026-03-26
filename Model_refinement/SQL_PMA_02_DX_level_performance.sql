-- ============================================================
-- SQL_PMA_02 — DX-LEVEL PERFORMANCE SUMMARY
-- Purpose : Per trigger_dx per model per window:
--           volume, TP/FP/FN, hit/ndcg/precision/recall
--           + DX description from ICD9_DIAGNOSIS
--           Top 10 by volume and top 10 by accuracy (min 100 triggers)
-- Sources : A870800_gen_rec_provider_eval_5pct  (SQL_05)
--           edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS
--             join: trigger_dx (with dots) = icd9_dx_cd
-- Output  : A870800_gen_rec_pma_dx_summary_5pct
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_pma_dx_summary_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH

-- 1. Aggregate eval metrics per (trigger_dx, model, time_bucket)
dx_metrics AS (
    SELECT
        e.trigger_dx
        ,e.model
        ,e.time_bucket
        ,COUNT(*)                                        AS n_triggers
        ,SUM(e.tp)                                       AS total_tp
        ,SUM(e.fp)                                       AS total_fp
        ,SUM(e.fn)                                       AS total_fn
        ,ROUND(AVG(e.hit_at_1), 4)                       AS hit_at_1
        ,ROUND(AVG(e.hit_at_3), 4)                       AS hit_at_3
        ,ROUND(AVG(e.hit_at_5), 4)                       AS hit_at_5
        ,ROUND(AVG(e.ndcg_at_1), 4)                      AS ndcg_at_1
        ,ROUND(AVG(e.ndcg_at_3), 4)                      AS ndcg_at_3
        ,ROUND(AVG(e.ndcg_at_5), 4)                      AS ndcg_at_5
        ,ROUND(AVG(e.precision_at_3), 4)                 AS precision_at_3
        ,ROUND(AVG(e.recall_at_3), 4)                    AS recall_at_3
        ,ROUND(SUM(e.tp) / NULLIF(SUM(e.tp) + SUM(e.fp), 0), 4) AS overall_precision
        ,ROUND(SUM(e.tp) / NULLIF(SUM(e.tp) + SUM(e.fn), 0), 4) AS overall_recall
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_5pct` e
    GROUP BY e.trigger_dx, e.model, e.time_bucket
),

-- 2. Join DX descriptions
-- trigger_dx has dots (e.g. I25.10), icd9_dx_cd also has dots
dx_with_desc AS (
    SELECT
        m.*
        ,COALESCE(d.icd9_dx_description, m.trigger_dx)  AS trigger_dx_desc
    FROM dx_metrics m
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` d
        ON m.trigger_dx = d.icd9_dx_cd
),

-- 3. Total volume per trigger_dx across all models and windows
--    Used for volume ranking (model-agnostic)
dx_volume AS (
    SELECT
        trigger_dx
        ,SUM(n_triggers)                                 AS total_volume
    FROM dx_metrics
    GROUP BY trigger_dx
),

-- 4. Volume rank — top 10 by total trigger volume
volume_ranked AS (
    SELECT
        trigger_dx
        ,total_volume
        ,ROW_NUMBER() OVER (ORDER BY total_volume DESC)  AS volume_rank
    FROM dx_volume
),

-- 5. Accuracy rank — top 10 by avg hit@3 across all models
--    Minimum 100 triggers to exclude noise
accuracy_ranked AS (
    SELECT
        trigger_dx
        ,ROUND(AVG(hit_at_3), 4)                         AS avg_hit_at_3_all_models
        ,ROW_NUMBER() OVER (
            ORDER BY AVG(hit_at_3) DESC
        )                                                AS accuracy_rank
    FROM dx_metrics
    WHERE n_triggers >= 100
    GROUP BY trigger_dx
)

-- 6. Final output — all DX metrics with volume + accuracy ranks
SELECT
    d.trigger_dx
    ,d.trigger_dx_desc
    ,d.model
    ,d.time_bucket
    ,d.n_triggers
    ,d.total_tp
    ,d.total_fp
    ,d.total_fn
    ,d.hit_at_1
    ,d.hit_at_3
    ,d.hit_at_5
    ,d.ndcg_at_1
    ,d.ndcg_at_3
    ,d.ndcg_at_5
    ,d.precision_at_3
    ,d.recall_at_3
    ,d.overall_precision
    ,d.overall_recall
    ,vr.total_volume
    ,vr.volume_rank
    ,ar.avg_hit_at_3_all_models
    ,ar.accuracy_rank
FROM dx_with_desc d
LEFT JOIN volume_ranked  vr ON d.trigger_dx = vr.trigger_dx
LEFT JOIN accuracy_ranked ar ON d.trigger_dx = ar.trigger_dx
ORDER BY d.time_bucket, d.model, vr.total_volume DESC
;
