-- ============================================================
-- ENTROPY COMPARISON — ALL THREE LENSES — ORDER 1
-- ============================================================
WITH lens1 AS (
    SELECT 'Diagnosis to Specialty' AS combination
        ,AVG(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0) AS weighted_avg_entropy
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'Diagnosis to CCSR'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'Diagnosis to Diagnosis'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'Specialty to Specialty'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'Specialty to Diagnosis'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'Specialty to CCSR'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'CCSR to Specialty'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'CCSR to Diagnosis'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'CCSR to CCSR'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
),
lens2 AS (
    SELECT 'Diagnosis to Specialty' AS combination
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0) AS weighted_avg_entropy
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'Diagnosis to CCSR'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'Diagnosis to Diagnosis'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'Specialty to Specialty'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'Specialty to Diagnosis'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'Specialty to CCSR'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'CCSR to Specialty'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'CCSR to Diagnosis'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    UNION ALL
    SELECT 'CCSR to CCSR'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
),
lens3 AS (
    SELECT 'Diagnosis to Specialty' AS combination
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0) AS weighted_avg_entropy
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
      AND trigger_specialty IN ('FP', 'I')
    UNION ALL
    SELECT 'Diagnosis to CCSR'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
      AND trigger_specialty IN ('FP', 'I')
    UNION ALL
    SELECT 'Diagnosis to Diagnosis'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
      AND trigger_specialty IN ('FP', 'I')
    UNION ALL
    SELECT 'Specialty to Specialty'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
      AND trigger_specialty IN ('FP', 'I')
    UNION ALL
    SELECT 'Specialty to Diagnosis'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
      AND trigger_specialty IN ('FP', 'I')
    UNION ALL
    SELECT 'Specialty to CCSR'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
      AND trigger_specialty IN ('FP', 'I')
    UNION ALL
    SELECT 'CCSR to Specialty'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
      AND trigger_specialty IN ('FP', 'I')
    UNION ALL
    SELECT 'CCSR to Diagnosis'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
      AND trigger_specialty IN ('FP', 'I')
    UNION ALL
    SELECT 'CCSR to CCSR'
        ,SUM(conditional_entropy * transition_count) / NULLIF(SUM(transition_count), 0)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
      AND trigger_specialty IN ('FP', 'I')
)
SELECT
    l1.combination
    ,ROUND(l1.weighted_avg_entropy, 4)                   AS lens1_no_filter
    ,ROUND(l2.weighted_avg_entropy, 4)                   AS lens2_first_encounter
    ,ROUND(l3.weighted_avg_entropy, 4)                   AS lens3_fp_first
FROM lens1 l1
JOIN lens2 l2 ON l1.combination = l2.combination
JOIN lens3 l3 ON l1.combination = l3.combination
ORDER BY lens2_first_encounter ASC
