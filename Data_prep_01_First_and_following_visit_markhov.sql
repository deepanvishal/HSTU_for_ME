-- ============================================================
-- ORDER 1: DX -> SPECIALTY (FILTERED TRANSITIONS)
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order1`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH member_segment AS (
    SELECT
        member_id
        ,visit_date
        ,CASE
            WHEN age_nbr < 18                            THEN 'Children'
            WHEN age_nbr BETWEEN 18 AND 65
                 AND gender_cd = 'M'                     THEN 'Adult_Male'
            WHEN age_nbr BETWEEN 18 AND 65
                 AND gender_cd = 'F'                     THEN 'Adult_Female'
            WHEN age_nbr > 65                            THEN 'Senior'
         END                                             AS member_segment
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`
    GROUP BY member_id, visit_date, member_segment
),
trigger_visits AS (
    -- qualifying trigger claims aggregated to visit level
    SELECT
        member_id
        ,visit_date
        ,visit_number
        ,ARRAY_AGG(DISTINCT dx_raw IGNORE NULLS)         AS dx_list
        ,ARRAY_AGG(DISTINCT specialty_ctg_cd IGNORE NULLS) AS specialty_list
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`
    WHERE visit_flag IN ('first_member_visit', 'new_provider_new_dx', 'known_provider_new_dx')
    GROUP BY member_id, visit_date, visit_number
),
next_visits AS (
    -- next visit after trigger (any visit, no flag filter)
    SELECT
        member_id
        ,visit_date
        ,ARRAY_AGG(DISTINCT specialty_ctg_cd IGNORE NULLS) AS specialty_list
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`
    GROUP BY member_id, visit_date
),
visit_pairs AS (
    SELECT
        t.member_id
        ,t.visit_date                                    AS trigger_date
        ,t.dx_list
        ,n.specialty_list                                AS next_specialty_list
        ,s.member_segment
    FROM trigger_visits t
    JOIN next_visits n
        ON t.member_id = n.member_id
        AND n.visit_date = (
            SELECT MIN(visit_date)
            FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`
            WHERE member_id = t.member_id
                AND visit_date > t.visit_date
        )
    JOIN member_segment s
        ON t.member_id = s.member_id
        AND t.visit_date = s.visit_date
),
exploded AS (
    SELECT
        member_id
        ,member_segment
        ,current_dx
        ,next_specialty
    FROM visit_pairs
    CROSS JOIN UNNEST(dx_list)                           AS current_dx
    CROSS JOIN UNNEST(next_specialty_list)               AS next_specialty
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
