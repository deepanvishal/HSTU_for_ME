-- ============================================================
-- SQL_PMA_03 — PROVIDER + DX PREDICTION SUMMARY (Mandatory 5a)
-- Purpose : For each (from_provider, trigger_dx, predicted_provider)
--           how many times predicted, how many correct, avg score,
--           training evidence strength
-- Grain   : (from_provider, trigger_dx, predicted_provider, model, time_bucket)
-- Sources : A870800_gen_rec_provider_eval_5pct            SQL_05
--           A870800_gen_rec_provider_model_test_agg_5pct  SQL_02
--           A870800_gen_rec_provider_markov_train_5pct    SQL_02
--           A870800_gen_rec_provider_name_lookup          SQL_PMA_00
--           A870800_gen_rec_provider_primary_specialty    SQL_01b
--           edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP
--           edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS
-- Deduplication:
--   eval_5pct grain: (member_id, trigger_date, trigger_dx, member_segment,
--                     time_bucket, model)
--   test_agg grain:  (member_id, trigger_date, trigger_dx, member_segment)
--                     one row per trigger -- many-to-one join, safe
--   QUALIFY ROW_NUMBER()=1 handles rare multiple from_provider per trigger
--   UNNEST with pred_pos=score_pos aligns predictions and scores by position
--   Final GROUP BY on (from_provider, trigger_dx, predicted_provider,
--                      model, time_bucket) — COUNT(*) = n triggers predicted
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_pma_provider_dx_summary_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH

-- 1. Join eval to test_agg to get from_provider per trigger
--    Carry from_provider into this CTE explicitly
eval_with_from AS (
    SELECT
        CAST(e.member_id      AS STRING)                 AS member_id
        ,CAST(e.trigger_date  AS STRING)                 AS trigger_date
        ,CAST(e.trigger_dx    AS STRING)                 AS trigger_dx
        ,CAST(e.member_segment AS STRING)                AS member_segment
        ,e.time_bucket
        ,e.model
        ,e.top5_predictions
        ,e.top5_scores
        ,e.true_labels
        ,CAST(t.from_provider AS STRING)                 AS from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_5pct` e
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_agg_5pct` t
        ON  CAST(e.member_id      AS STRING) = CAST(t.member_id      AS STRING)
        AND CAST(e.trigger_date   AS STRING) = CAST(t.trigger_date   AS STRING)
        AND CAST(e.trigger_dx     AS STRING) = CAST(t.trigger_dx     AS STRING)
        AND CAST(e.member_segment AS STRING) = CAST(t.member_segment AS STRING)
    -- Rare case: multiple from_provider entries per trigger in test_agg
    -- Take the lowest NPI string deterministically
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY e.member_id, e.trigger_date, e.trigger_dx,
                     e.member_segment, e.time_bucket, e.model
        ORDER BY CAST(t.from_provider AS STRING)
    ) = 1
),

-- 2. Explode top5_predictions and top5_scores by position
--    from_provider is now in eval_with_from and carried here
pred_exploded AS (
    SELECT
        ef.member_id
        ,ef.trigger_date
        ,ef.trigger_dx
        ,ef.member_segment
        ,ef.time_bucket
        ,ef.model
        ,ef.from_provider
        ,ef.true_labels
        ,CAST(pred_provider AS STRING)                   AS predicted_provider
        ,CAST(pred_score    AS FLOAT64)                  AS pred_score
        -- Hit check: wrap both sides with pipes to avoid partial NPI matches
        ,IF(CONCAT('|', ef.true_labels, '|')
            LIKE CONCAT('%|', CAST(pred_provider AS STRING), '|%'),
            1, 0)                                        AS is_correct
    FROM eval_with_from ef
    CROSS JOIN UNNEST(SPLIT(ef.top5_predictions, '|')) AS pred_provider
        WITH OFFSET pred_pos
    CROSS JOIN UNNEST(SPLIT(ef.top5_scores, '|'))      AS pred_score
        WITH OFFSET score_pos
    WHERE pred_pos = score_pos
      AND CAST(pred_provider AS STRING) != ''
      AND pred_provider IS NOT NULL
),

-- 3. Training evidence per (from_provider, trigger_dx, to_provider)
training_evidence AS (
    SELECT
        CAST(from_provider AS STRING)                    AS from_provider
        ,CAST(trigger_dx   AS STRING)                    AS trigger_dx
        ,CAST(to_provider  AS STRING)                    AS to_provider
        ,SUM(transition_count)                           AS training_transition_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_5pct`
    WHERE from_provider IS NOT NULL
      AND trigger_dx    IS NOT NULL
      AND to_provider   IS NOT NULL
    GROUP BY from_provider, trigger_dx, to_provider
),

-- 4. Aggregate to final grain
--    from_provider already in pred_exploded — no secondary join needed
summary_raw AS (
    SELECT
        p.from_provider
        ,p.trigger_dx
        ,p.predicted_provider
        ,p.model
        ,p.time_bucket
        ,COUNT(*)                                        AS n_times_predicted
        ,SUM(p.is_correct)                               AS n_correct
        ,ROUND(AVG(p.pred_score), 4)                     AS avg_pred_score
        ,ROUND(MIN(p.pred_score), 4)                     AS min_pred_score
        ,ROUND(MAX(p.pred_score), 4)                     AS max_pred_score
    FROM pred_exploded p
    GROUP BY
        p.from_provider
        ,p.trigger_dx
        ,p.predicted_provider
        ,p.model
        ,p.time_bucket
),

-- 5. Enrich with names, specialties, descriptions, training evidence
enriched AS (
    SELECT
        s.from_provider
        ,COALESCE(fn.provider_name,  s.from_provider)    AS from_provider_name
        ,COALESCE(fps.primary_specialty, 'Unknown')       AS from_specialty
        ,COALESCE(fsp.global_lookup_desc, 'Unknown')      AS from_specialty_desc
        ,s.trigger_dx
        ,COALESCE(dx.icd9_dx_description, s.trigger_dx)  AS trigger_dx_desc
        ,s.predicted_provider                             AS to_provider
        ,COALESCE(tn.provider_name,  s.predicted_provider) AS to_provider_name
        ,COALESCE(tps.primary_specialty, 'Unknown')       AS to_specialty
        ,COALESCE(tsp.global_lookup_desc, 'Unknown')      AS to_specialty_desc
        ,s.model
        ,s.time_bucket
        ,s.n_times_predicted
        ,s.n_correct                                      AS n_tp
        ,(s.n_times_predicted - s.n_correct)              AS n_fp
        ,ROUND(s.n_correct / NULLIF(s.n_times_predicted, 0), 4) AS precision_this_pair
        ,s.avg_pred_score
        ,s.min_pred_score
        ,s.max_pred_score
        ,COALESCE(te.training_transition_count, 0)        AS training_transition_count
    FROM summary_raw s
    -- from_provider enrichment
    LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_name_lookup` fn
        ON CAST(s.from_provider AS STRING) = CAST(fn.srv_prvdr_id AS STRING)
    LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_primary_specialty` fps
        ON CAST(s.from_provider AS STRING) = CAST(fps.srv_prvdr_id AS STRING)
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` fsp
        ON fsp.global_lookup_cd = fps.primary_specialty
        AND LOWER(fsp.lookup_column_nm) = 'specialty_ctg_cd'
    -- to_provider enrichment
    LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_name_lookup` tn
        ON CAST(s.predicted_provider AS STRING) = CAST(tn.srv_prvdr_id AS STRING)
    LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_primary_specialty` tps
        ON CAST(s.predicted_provider AS STRING) = CAST(tps.srv_prvdr_id AS STRING)
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` tsp
        ON tsp.global_lookup_cd = tps.primary_specialty
        AND LOWER(tsp.lookup_column_nm) = 'specialty_ctg_cd'
    -- dx description
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx
        ON s.trigger_dx = dx.icd9_dx_cd
    -- training evidence
    LEFT JOIN training_evidence te
        ON  CAST(s.from_provider      AS STRING) = CAST(te.from_provider AS STRING)
        AND CAST(s.trigger_dx         AS STRING) = CAST(te.trigger_dx    AS STRING)
        AND CAST(s.predicted_provider AS STRING) = CAST(te.to_provider   AS STRING)
)

SELECT *
FROM enriched
ORDER BY model, time_bucket, n_times_predicted DESC
;
