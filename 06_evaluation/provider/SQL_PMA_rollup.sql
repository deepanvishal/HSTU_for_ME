-- ============================================================
-- SQL_PMA_05 — MONTHLY PROVIDER + DX → PROVIDER ROLLUP
-- Purpose : For a reference month (Aug 2024), for each
--           (from_provider, trigger_dx, predicted_to_provider):
--             - expected_transitions: how many times model
--               predicted this path in that month
--             - avg_pred_score: avg confidence of that prediction
--             - actual_transitions: how many times it actually
--               happened (to_provider in true_labels)
-- Grain   : (from_provider, trigger_dx, predicted_to_provider,
--            model, reference_month)
-- Sources : A870800_gen_rec_provider_eval_5pct        SQL_05
--           A870800_gen_rec_provider_model_test_agg_5pct SQL_02
--           A870800_gen_rec_provider_name_lookup       SQL_PMA_00
--           A870800_gen_rec_provider_primary_specialty SQL_01b
--           edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP
--           edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS
-- Filter  : time_bucket = T0_30 (next 30 days only)
--           trigger_date LIKE '2024-08-%' (Aug 2024)
--           model = SASRec (primary analysis model)
-- Notes:
--   trigger_date in eval_5pct is STRING format 'YYYY-MM-DD'
--   true_labels is pipe-delimited STRING of actual provider IDs
--   top5_predictions is pipe-delimited STRING of predicted provider IDs
--   top5_scores is pipe-delimited STRING of prediction scores
--   Deduplication: same as SQL_PMA_03 — QUALIFY for from_provider,
--   pred_pos=score_pos for alignment
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_pma_monthly_provider_dx_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH

-- 1. Filter eval to Aug 2024, T0_30, SASRec
--    Add from_provider via test_agg — many-to-one join, safe
aug_eval AS (
    SELECT
        CAST(e.member_id       AS STRING)                AS member_id
        ,CAST(e.trigger_date   AS STRING)                AS trigger_date
        ,CAST(e.trigger_dx     AS STRING)                AS trigger_dx
        ,CAST(e.member_segment AS STRING)                AS member_segment
        ,e.model
        ,e.top5_predictions
        ,e.top5_scores
        ,e.true_labels
        ,CAST(t.from_provider  AS STRING)                AS from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_5pct` e
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_agg_5pct` t
        ON  CAST(e.member_id       AS STRING) = CAST(t.member_id       AS STRING)
        AND CAST(e.trigger_date    AS STRING) = CAST(t.trigger_date     AS STRING)
        AND CAST(e.trigger_dx      AS STRING) = CAST(t.trigger_dx       AS STRING)
        AND CAST(e.member_segment  AS STRING) = CAST(t.member_segment   AS STRING)
    WHERE e.time_bucket   = 'T0_30'
      AND e.model         = 'SASRec'
      AND e.trigger_date  LIKE '2024-08-%'
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY CAST(e.member_id AS STRING), CAST(e.trigger_date AS STRING),
                     CAST(e.trigger_dx AS STRING), CAST(e.member_segment AS STRING),
                     e.model
        ORDER BY CAST(t.from_provider AS STRING)
    ) = 1
),

-- 2. Explode top5_predictions + scores by position
--    Each row = one prediction event for this trigger
pred_flat AS (
    SELECT
        a.member_id
        ,a.trigger_date
        ,a.trigger_dx
        ,a.from_provider
        ,a.model
        ,a.true_labels
        ,CAST(pred_provider AS STRING)                   AS predicted_provider
        ,CAST(pred_score    AS FLOAT64)                  AS pred_score
    FROM aug_eval a
    CROSS JOIN UNNEST(SPLIT(a.top5_predictions, '|')) AS pred_provider
        WITH OFFSET pred_pos
    CROSS JOIN UNNEST(SPLIT(a.top5_scores, '|'))      AS pred_score
        WITH OFFSET score_pos
    WHERE pred_pos = score_pos
      AND CAST(pred_provider AS STRING) != ''
      AND pred_provider IS NOT NULL
),

-- 3. Explode true_labels to get actual transitions per trigger
--    Each row = one actual provider visited in T0_30 for this trigger
true_flat AS (
    SELECT
        a.member_id
        ,a.trigger_date
        ,a.trigger_dx
        ,a.from_provider
        ,CAST(true_provider AS STRING)                   AS actual_provider
    FROM aug_eval a
    CROSS JOIN UNNEST(SPLIT(a.true_labels, '|')) AS true_provider
    WHERE CAST(true_provider AS STRING) != ''
      AND true_provider IS NOT NULL
),

-- 4. Expected transitions — aggregate predictions to
--    (from_provider, trigger_dx, predicted_provider, model)
expected AS (
    SELECT
        from_provider
        ,trigger_dx
        ,predicted_provider
        ,model
        ,COUNT(*)                                        AS expected_transitions
        ,ROUND(AVG(pred_score), 4)                       AS avg_pred_score
        ,ROUND(MIN(pred_score), 4)                       AS min_pred_score
        ,ROUND(MAX(pred_score), 4)                       AS max_pred_score
    FROM pred_flat
    GROUP BY from_provider, trigger_dx, predicted_provider, model
),

-- 5. Actual transitions — aggregate true labels to
--    (from_provider, trigger_dx, actual_provider)
actual AS (
    SELECT
        from_provider
        ,trigger_dx
        ,actual_provider
        ,COUNT(*)                                        AS actual_transitions
    FROM true_flat
    GROUP BY from_provider, trigger_dx, actual_provider
),

-- 6. Join expected to actual on same path
--    actual_transitions = NULL if this predicted path never happened
combined AS (
    SELECT
        e.from_provider
        ,e.trigger_dx
        ,e.predicted_provider
        ,e.model
        ,'2024-08'                                       AS reference_month
        ,e.expected_transitions
        ,e.avg_pred_score
        ,e.min_pred_score
        ,e.max_pred_score
        ,COALESCE(a.actual_transitions, 0)               AS actual_transitions
        -- Did the model predict a path that actually happened?
        ,CASE WHEN COALESCE(a.actual_transitions, 0) > 0
              THEN 1 ELSE 0 END                          AS path_correct
    FROM expected e
    LEFT JOIN actual a
        ON  CAST(e.from_provider      AS STRING) = CAST(a.from_provider  AS STRING)
        AND CAST(e.trigger_dx         AS STRING) = CAST(a.trigger_dx     AS STRING)
        AND CAST(e.predicted_provider AS STRING) = CAST(a.actual_provider AS STRING)
)

-- 7. Enrich with names and descriptions
SELECT
    c.reference_month
    ,c.model
    ,c.from_provider
    ,COALESCE(fn.provider_name, c.from_provider)         AS from_provider_name
    ,COALESCE(fps.primary_specialty, 'Unknown')           AS from_specialty
    ,COALESCE(fsp.global_lookup_desc, 'Unknown')          AS from_specialty_desc
    ,c.trigger_dx
    ,COALESCE(dx.icd9_dx_description, c.trigger_dx)      AS trigger_dx_desc
    ,c.predicted_provider
    ,COALESCE(tn.provider_name, c.predicted_provider)     AS predicted_provider_name
    ,COALESCE(tps.primary_specialty, 'Unknown')           AS predicted_specialty
    ,COALESCE(tsp.global_lookup_desc, 'Unknown')          AS predicted_specialty_desc
    ,c.expected_transitions
    ,c.avg_pred_score
    ,c.min_pred_score
    ,c.max_pred_score
    ,c.actual_transitions
    ,c.path_correct
    ,ROUND(c.actual_transitions / NULLIF(c.expected_transitions, 0), 4) AS realization_rate
FROM combined c
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_name_lookup` fn
    ON CAST(c.from_provider AS STRING) = CAST(fn.srv_prvdr_id AS STRING)
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_primary_specialty` fps
    ON CAST(c.from_provider AS STRING) = CAST(fps.srv_prvdr_id AS STRING)
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` fsp
    ON fsp.global_lookup_cd = fps.primary_specialty
    AND LOWER(fsp.lookup_column_nm) = 'specialty_ctg_cd'
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_name_lookup` tn
    ON CAST(c.predicted_provider AS STRING) = CAST(tn.srv_prvdr_id AS STRING)
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_primary_specialty` tps
    ON CAST(c.predicted_provider AS STRING) = CAST(tps.srv_prvdr_id AS STRING)
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` tsp
    ON tsp.global_lookup_cd = tps.primary_specialty
    AND LOWER(tsp.lookup_column_nm) = 'specialty_ctg_cd'
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx
    ON CAST(c.trigger_dx AS STRING) = CAST(dx.icd9_dx_cd AS STRING)
ORDER BY c.expected_transitions DESC
;
