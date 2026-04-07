-- ============================================================
-- SQL_PMA_06 — MONTHLY PROVIDER → PROVIDER ROLLUP
-- Purpose : Roll up SQL_PMA_05 by dropping trigger_dx
--           For each (from_provider, predicted_to_provider):
--             - total expected transitions (sum across all DX)
--             - weighted avg prediction score
--             - total actual transitions
--             - realization rate (actual / expected)
-- Grain   : (from_provider, predicted_provider, model, reference_month)
-- Source  : A870800_gen_rec_pma_monthly_provider_dx_5pct  SQL_PMA_05
-- Notes:
--   weighted avg score = sum(expected × avg_score) / sum(expected)
--   realization_rate = total_actual / total_expected
--   A path with high expected but low actual = model over-predicts
--   A path with high actual but low expected = model misses it
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_pma_monthly_provider_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH

-- 1. Roll up from SQL_PMA_05 — drop trigger_dx, sum transitions
provider_rollup AS (
    SELECT
        reference_month
        ,model
        ,from_provider
        ,from_provider_name
        ,from_specialty_desc
        ,predicted_provider
        ,predicted_provider_name
        ,predicted_specialty_desc
        ,COUNT(DISTINCT trigger_dx)                      AS n_dx_codes
        ,SUM(expected_transitions)                       AS total_expected_transitions
        -- Weighted avg score: weight by expected_transitions per DX
        ,ROUND(
            SUM(expected_transitions * avg_pred_score)
            / NULLIF(SUM(expected_transitions), 0)
        , 4)                                             AS weighted_avg_score
        ,SUM(actual_transitions)                         AS total_actual_transitions
        ,SUM(path_correct)                               AS n_dx_paths_correct
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_pma_monthly_provider_dx_5pct`
    GROUP BY
        reference_month, model,
        from_provider, from_provider_name, from_specialty_desc,
        predicted_provider, predicted_provider_name, predicted_specialty_desc
)

SELECT
    reference_month
    ,model
    ,from_provider
    ,from_provider_name
    ,from_specialty_desc
    ,predicted_provider
    ,predicted_provider_name
    ,predicted_specialty_desc
    ,n_dx_codes
    ,total_expected_transitions
    ,weighted_avg_score
    ,total_actual_transitions
    ,n_dx_paths_correct
    -- Realization rate: what fraction of expected actually happened
    ,ROUND(total_actual_transitions
           / NULLIF(total_expected_transitions, 0), 4)   AS realization_rate
    -- Gap: expected - actual (positive = over-predicted, negative = under-predicted)
    ,(total_expected_transitions - total_actual_transitions) AS prediction_gap
    -- Path status
    ,CASE
        WHEN total_actual_transitions  = 0 THEN 'Predicted but never happened'
        WHEN total_expected_transitions = 0 THEN 'Happened but never predicted'
        WHEN total_actual_transitions >= total_expected_transitions THEN 'Under-predicted'
        ELSE 'Over-predicted'
     END                                                 AS path_status
FROM provider_rollup
ORDER BY total_expected_transitions DESC
;
