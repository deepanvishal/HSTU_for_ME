DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_entropy_summary`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_entropy_summary`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH dx_to_specialty_o1 AS (
    SELECT 'dx_to_specialty' AS combination, 1 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count) AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]                 AS median_entropy
        ,SUM(transition_count)                                               AS total_transitions
        ,COUNT(DISTINCT current_dx)                                          AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
dx_to_dx_o1 AS (
    SELECT 'dx_to_dx' AS combination, 1 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_dx)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
dx_to_ccsr_o1 AS (
    SELECT 'dx_to_ccsr' AS combination, 1 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_dx)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
specialty_to_specialty_o1 AS (
    SELECT 'specialty_to_specialty' AS combination, 1 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_specialty)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
specialty_to_dx_o1 AS (
    SELECT 'specialty_to_dx' AS combination, 1 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_specialty)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
specialty_to_ccsr_o1 AS (
    SELECT 'specialty_to_ccsr' AS combination, 1 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_specialty)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
ccsr_to_specialty_o1 AS (
    SELECT 'ccsr_to_specialty' AS combination, 1 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_ccsr)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
ccsr_to_dx_o1 AS (
    SELECT 'ccsr_to_dx' AS combination, 1 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_ccsr)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_dx_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
ccsr_to_ccsr_o1 AS (
    SELECT 'ccsr_to_ccsr' AS combination, 1 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_ccsr)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_ccsr_order1`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
dx_to_specialty_o2 AS (
    SELECT 'dx_to_specialty' AS combination, 2 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_dx_v1)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order2`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
dx_to_dx_o2 AS (
    SELECT 'dx_to_dx' AS combination, 2 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_dx_v1)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order2`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
dx_to_ccsr_o2 AS (
    SELECT 'dx_to_ccsr' AS combination, 2 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_dx_v1)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order2`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
specialty_to_specialty_o2 AS (
    SELECT 'specialty_to_specialty' AS combination, 2 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_specialty_v1)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order2`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
specialty_to_dx_o2 AS (
    SELECT 'specialty_to_dx' AS combination, 2 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_specialty_v1)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order2`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
specialty_to_ccsr_o2 AS (
    SELECT 'specialty_to_ccsr' AS combination, 2 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_specialty_v1)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order2`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
ccsr_to_specialty_o2 AS (
    SELECT 'ccsr_to_specialty' AS combination, 2 AS markov_order, member_segment
        ,SUM(transition_count * conditional_entropy) / SUM(transition_count)
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)]
        ,SUM(transition_count)
        ,COUNT(DISTINCT current_ccsr_v1)
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order2`
    WHERE transition_count >= 100 AND conditional_entropy IS NOT NULL
    GROUP BY member_segment
