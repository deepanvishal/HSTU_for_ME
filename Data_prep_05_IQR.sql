DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_iqr_bounds`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_iqr_bounds`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH bounds AS (
    SELECT 'dx_to_specialty_order1' AS table_name, member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)] AS q1
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)] AS q3
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'dx_to_dx_order1', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'dx_to_ccsr_order1', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_specialty_order1', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_dx_order1', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_ccsr_order1', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_specialty_order1', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_dx_order1', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_dx_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_ccsr_order1', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_ccsr_order1`
    GROUP BY member_segment

    UNION ALL

    SELECT 'dx_to_specialty_order2', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'dx_to_dx_order2', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'dx_to_ccsr_order2', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_specialty_order2', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_dx_order2', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'specialty_to_ccsr_order2', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_specialty_order2', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_dx_order2', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_dx_order2`
    GROUP BY member_segment

    UNION ALL

    SELECT 'ccsr_to_ccsr_order2', member_segment
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(1)]
        ,APPROX_QUANTILES(transition_count, 4)[OFFSET(3)]
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_ccsr_order2`
    GROUP BY member_segment
)
SELECT
    table_name
    ,member_segment
    ,q1
    ,q3
    ,q3 - q1                                             AS iqr
    ,GREATEST(q1 - 1.5 * (q3 - q1), 1)                  AS lower_fence
FROM bounds
ORDER BY table_name, member_segment
