-- ============================================================
-- DX -> SPECIALTY ORDER 1 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_specialty_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_specialty_order1`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_pairs AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,dx_list_raw
        ,LEAD(specialty_codes) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS next_specialty_codes
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`
),
exploded AS (
    SELECT
        member_id
        ,member_segment
        ,current_dx
        ,next_specialty
    FROM visit_pairs
    CROSS JOIN UNNEST(dx_list_raw)                       AS current_dx
    CROSS JOIN UNNEST(next_specialty_codes)              AS next_specialty
    WHERE next_specialty_codes IS NOT NULL
),
transition_counts AS (
    SELECT
        current_dx
        ,next_specialty
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_dx, next_specialty, member_segment
),
dx_totals AS (
    SELECT
        current_dx
        ,member_segment
        ,SUM(transition_count)                           AS dx_total
    FROM transition_counts
    GROUP BY current_dx, member_segment
)
SELECT
    t.current_dx
    ,t.next_specialty
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,d.dx_total
    ,ROUND(t.transition_count / d.dx_total, 4)          AS conditional_probability
    ,ROUND(-SUM(t.transition_count / d.dx_total *
        LOG(t.transition_count / d.dx_total)) OVER (
            PARTITION BY t.current_dx, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN dx_totals d
    ON t.current_dx = d.current_dx
    AND t.member_segment = d.member_segment
ORDER BY
    t.current_dx
    ,t.transition_count DESC
