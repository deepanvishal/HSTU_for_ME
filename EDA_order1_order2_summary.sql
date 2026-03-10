-- ============================================================
-- ENTROPY SUMMARY TABLE - ALL 18 COMBINATIONS
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_entropy_summary`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_entropy_summary`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH order1_dx_to_specialty AS (
    SELECT
        'dx_to_specialty'                                AS combination
        ,1                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_dx)                      AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_specialty_order1`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order1_dx_to_dx AS (
    SELECT
        'dx_to_dx'                                       AS combination
        ,1                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_dx)                      AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_dx_order1`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order1_dx_to_ccsr AS (
    SELECT
        'dx_to_ccsr'                                     AS combination
        ,1                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_dx)                      AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_ccsr_order1`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order1_specialty_to_specialty AS (
    SELECT
        'specialty_to_specialty'                         AS combination
        ,1                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_specialty)               AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_specialty_order1`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order1_specialty_to_dx AS (
    SELECT
        'specialty_to_dx'                                AS combination
        ,1                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_specialty)               AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_dx_order1`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order1_specialty_to_ccsr AS (
    SELECT
        'specialty_to_ccsr'                              AS combination
        ,1                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_specialty)               AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_ccsr_order1`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order1_ccsr_to_specialty AS (
    SELECT
        'ccsr_to_specialty'                              AS combination
        ,1                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_ccsr)                    AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_specialty_order1`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order1_ccsr_to_dx AS (
    SELECT
        'ccsr_to_dx'                                     AS combination
        ,1                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_ccsr)                    AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_dx_order1`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order1_ccsr_to_ccsr AS (
    SELECT
        'ccsr_to_ccsr'                                   AS combination
        ,1                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_ccsr)                    AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_ccsr_order1`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order2_dx_to_specialty AS (
    SELECT
        'dx_to_specialty'                                AS combination
        ,2                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_dx_v1)                   AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_specialty_order2`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order2_dx_to_dx AS (
    SELECT
        'dx_to_dx'                                       AS combination
        ,2                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_dx_v1)                   AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_dx_order2`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order2_dx_to_ccsr AS (
    SELECT
        'dx_to_ccsr'                                     AS combination
        ,2                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_dx_v1)                   AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_ccsr_order2`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order2_specialty_to_specialty AS (
    SELECT
        'specialty_to_specialty'                         AS combination
        ,2                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_specialty_v1)            AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_specialty_order2`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order2_specialty_to_dx AS (
    SELECT
        'specialty_to_dx'                                AS combination
        ,2                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_specialty_v1)            AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_dx_order2`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order2_specialty_to_ccsr AS (
    SELECT
        'specialty_to_ccsr'                              AS combination
        ,2                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_specialty_v1)            AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_ccsr_order2`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order2_ccsr_to_specialty AS (
    SELECT
        'ccsr_to_specialty'                              AS combination
        ,2                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_ccsr_v1)                 AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_specialty_order2`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order2_ccsr_to_dx AS (
    SELECT
        'ccsr_to_dx'                                     AS combination
        ,2                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_ccsr_v1)                 AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_dx_order2`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
),
order2_ccsr_to_ccsr AS (
    SELECT
        'ccsr_to_ccsr'                                   AS combination
        ,2                                               AS markov_order
        ,member_segment
        ,SUM(transition_count * conditional_entropy)
            / SUM(transition_count)                      AS weighted_avg_entropy
        ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
        ,SUM(transition_count)                           AS total_transitions
        ,COUNT(DISTINCT current_ccsr_v1)                 AS unique_current_states
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_ccsr_order2`
    WHERE conditional_entropy IS NOT NULL
    GROUP BY member_segment
)
SELECT * FROM order1_dx_to_specialty
UNION ALL SELECT * FROM order1_dx_to_dx
UNION ALL SELECT * FROM order1_dx_to_ccsr
UNION ALL SELECT * FROM order1_specialty_to_specialty
UNION ALL SELECT * FROM order1_specialty_to_dx
UNION ALL SELECT * FROM order1_specialty_to_ccsr
UNION ALL SELECT * FROM order1_ccsr_to_specialty
UNION ALL SELECT * FROM order1_ccsr_to_dx
UNION ALL SELECT * FROM order1_ccsr_to_ccsr
UNION ALL SELECT * FROM order2_dx_to_specialty
UNION ALL SELECT * FROM order2_dx_to_dx
UNION ALL SELECT * FROM order2_dx_to_ccsr
UNION ALL SELECT * FROM order2_specialty_to_specialty
UNION ALL SELECT * FROM order2_specialty_to_dx
UNION ALL SELECT * FROM order2_specialty_to_ccsr
UNION ALL SELECT * FROM order2_ccsr_to_specialty
UNION ALL SELECT * FROM order2_ccsr_to_dx
UNION ALL SELECT * FROM order2_ccsr_to_ccsr
ORDER BY
    weighted_avg_entropy ASC
    ,markov_order ASC
    ,member_segment
