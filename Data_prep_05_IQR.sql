DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_iqr_bounds`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_iqr_bounds`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH bounds AS (
    SELECT 'dx_to_specialty_order1' AS table_name, member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)] AS log_q1
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)] AS log_q3
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'dx_to_dx_order1', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'dx_to_ccsr_order1', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_specialty_order1', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_dx_order1', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_ccsr_order1', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_specialty_order1', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_dx_order1', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_dx_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_ccsr_order1', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_ccsr_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'dx_to_specialty_order2', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'dx_to_dx_order2', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'dx_to_ccsr_order2', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_specialty_order2', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_dx_order2', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_ccsr_order2', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_specialty_order2', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_dx_order2', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_dx_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_ccsr_order2', member_segment
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(1)]
        ,APPROX_QUANTILES(LOG(transition_count), 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_ccsr_order2`
    GROUP BY member_segment
)
SELECT
    table_name
    ,member_segment
    ,log_q1
    ,log_q3
    ,log_q3 - log_q1                                     AS log_iqr
    ,log_q1 - 1.5 * (log_q3 - log_q1)                   AS log_lower_fence
    ,CAST(EXP(log_q1 - 1.5 * (log_q3 - log_q1)) AS INT64) AS lower_fence
FROM bounds
ORDER BY table_name, member_segment
