-- ============================================================
-- SQL_PMA_03 — PROVIDER + DX PREDICTION SUMMARY (Mandatory 5a)
-- Purpose : For each (from_provider, trigger_dx, predicted_provider)
--           how many times was the prediction made, how many were correct,
--           what was the avg score, how much training evidence exists
-- Grain   : (from_provider, trigger_dx, predicted_provider, model, time_bucket)
-- Sources : A870800_gen_rec_provider_eval_5pct        — eval metrics per trigger
--           A870800_gen_rec_provider_model_test_agg_5pct — from_provider per trigger
--           A870800_gen_rec_provider_markov_train_5pct — training transition evidence
--           A870800_gen_rec_provider_name_lookup       — provider names
--           A870800_gen_rec_provider_primary_specialty — provider specialty
--           edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP     — specialty desc
--           edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS    — dx desc
-- Deduplication notes:
--   eval grain: (member_id, trigger_date, trigger_dx, time_bucket, model)
--   test_agg grain: (member_id, trigger_date, trigger_dx) — many-to-one join, safe
--   After joining from_provider, grain becomes:
--     (member_id, trigger_date, trigger_dx, from_provider, time_bucket, model)
--   SPLIT top5_predictions + UNNEST WITH OFFSET gives:
--     (member_id, trigger_date, trigger_dx, from_provider, time_bucket, model,
--      predicted_provider, pred_position)
--   Aggregation on (from_provider, trigger_dx, predicted_provider, model, time_bucket)
--   uses COUNT(*) = n_times predicted (each trigger is one prediction event)
--   and SUM of is_correct (1 if predicted_provider in true_labels else 0)
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_pma_provider_dx_summary_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH

-- 1. Add from_provider to eval table via many-to-one join on test_agg
--    test_agg has one row per (member_id, trigger_date, trigger_dx)
--    eval has multiple rows per trigger (3 windows × 4 models)
--    Join is safe — no row multiplication
eval_with_from AS (
    SELECT
        e.member_id
        ,CAST(e.trigger_date AS STRING)                  AS trigger_date
        ,e.trigger_dx
        ,e.member_segment
        ,e.time_bucket
        ,e.model
        ,e.top5_predictions
        ,e.top5_scores
        ,e.true_labels
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_eval_5pct` e
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_agg_5pct` t
        ON  e.member_id      = t.member_id
        AND CAST(e.trigger_date AS STRING) = CAST(t.trigger_date AS STRING)
        AND e.trigger_dx     = t.trigger_dx
        AND e.member_segment = t.member_segment
    -- Dedup in case test_agg has multiple from_provider per trigger
    -- (can happen if same member same date same dx visited multiple providers)
    -- We take the first from_provider per trigger deterministically
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY e.member_id, e.trigger_date, e.trigger_dx,
                     e.member_segment, e.time_bucket, e.model
        ORDER BY CAST(t.from_provider AS STRING)
    ) = 1
),

-- 2. Explode top5_predictions and top5_scores to one row per predicted provider
--    UNNEST WITH OFFSET aligns predictions with scores by position
--    true_labels kept as string for per-provider hit check
pred_exploded AS (
    SELECT
        ef.member_id
        ,ef.trigger_date
        ,ef.trigger_dx
        ,ef.member_segment
        ,ef.time_bucket
        ,ef.model
        ,ef.true_labels
        ,CAST(pred_provider AS STRING)                   AS predicted_provider
        ,CAST(pred_score    AS FLOAT64)                  AS pred_score
        -- Is this prediction correct? 1 if in true labels, 0 if not
        ,IF(CONCAT('|', ef.true_labels, '|')
            LIKE CONCAT('%|', CAST(pred_provider AS STRING), '|%'),
            1, 0)                                        AS is_correct
    FROM eval_with_from ef
    -- Explode predictions and scores together by position
    CROSS JOIN UNNEST(SPLIT(ef.top5_predictions, '|')) AS pred_provider
        WITH OFFSET pred_pos
    CROSS JOIN UNNEST(SPLIT(ef.top5_scores, '|'))      AS pred_score
        WITH OFFSET score_pos
    WHERE pred_pos = score_pos                           -- align by position
      AND pred_provider IS NOT NULL
      AND CAST(pred_provider AS STRING) != ''
),

-- 3. Training evidence: total transition count per (from_provider, trigger_dx, to_provider)
--    from markov train table — this is the ground truth evidence from training
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

-- 4. Aggregate to (from_provider, trigger_dx, predicted_provider, model, time_bucket)
--    COUNT(*) = n_triggers where this provider was predicted for this context
--    SUM(is_correct) = n times it was correct (TP for this provider)
summary_raw AS (
    SELECT
        ef.from_provider
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
    -- Join back to eval_with_from to get from_provider
    -- pred_exploded was derived from eval_with_from, so join on trigger key
    INNER JOIN (
        SELECT DISTINCT
            member_id, trigger_date, trigger_dx, member_segment,
            time_bucket, model, from_provider
        FROM eval_with_from
    ) ef
        ON  p.member_id      = ef.member_id
        AND p.trigger_date   = ef.trigger_date
        AND p.trigger_dx     = ef.trigger_dx
        AND p.member_segment = ef.member_segment
        AND p.time_bucket    = ef.time_bucket
        AND p.model          = ef.model
    GROUP BY
        ef.from_provider, p.trigger_dx, p.predicted_provider,
        p.model, p.time_bucket
),

-- 5. Enrich with descriptions and training evidence
enriched AS (
    SELECT
        s.from_provider
        ,COALESCE(fn.provider_name, s.from_provider)     AS from_provider_name
        ,COALESCE(fps.primary_specialty, 'Unknown')       AS from_specialty
        ,COALESCE(fsp.global_lookup_desc, fps.primary_specialty, 'Unknown') AS from_specialty_desc
        ,s.trigger_dx
        ,COALESCE(dx.icd9_dx_description, s.trigger_dx)  AS trigger_dx_desc
        ,s.predicted_provider                             AS to_provider
        ,COALESCE(tn.provider_name, s.predicted_provider) AS to_provider_name
        ,COALESCE(tps.primary_specialty, 'Unknown')       AS to_specialty
        ,COALESCE(tsp.global_lookup_desc, tps.primary_specialty, 'Unknown') AS to_specialty_desc
        ,s.model
        ,s.time_bucket
        ,s.n_times_predicted
        ,s.n_correct                                      AS n_tp
        ,s.n_times_predicted - s.n_correct                AS n_fp
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
        ON  CAST(s.from_provider      AS STRING) = te.from_provider
        AND CAST(s.trigger_dx         AS STRING) = te.trigger_dx
        AND CAST(s.predicted_provider AS STRING) = te.to_provider
)

SELECT *
FROM enriched
ORDER BY
    model
    ,time_bucket
    ,n_times_predicted DESC
;
