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


-- ============================================================
-- DX -> DX ORDER 1 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_dx_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_dx_order1`
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
        ,LEAD(dx_list_raw) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS next_dx_list
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`
),
exploded AS (
    SELECT
        member_id
        ,member_segment
        ,current_dx
        ,next_dx
    FROM visit_pairs
    CROSS JOIN UNNEST(dx_list_raw)                       AS current_dx
    CROSS JOIN UNNEST(next_dx_list)                      AS next_dx
    WHERE next_dx_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_dx
        ,next_dx
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_dx, next_dx, member_segment
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
    ,t.next_dx
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
    ,t.transition_count DESC;

-- ============================================================
-- DX -> CCSR ORDER 1 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_ccsr_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_ccsr_order1`
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
        ,LEAD(ccsr_list) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS next_ccsr_list
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`
),
exploded AS (
    SELECT
        member_id
        ,member_segment
        ,current_dx
        ,next_ccsr
    FROM visit_pairs
    CROSS JOIN UNNEST(dx_list_raw)                       AS current_dx
    CROSS JOIN UNNEST(next_ccsr_list)                    AS next_ccsr
    WHERE next_ccsr_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_dx
        ,next_ccsr
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_dx, next_ccsr, member_segment
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
    ,t.next_ccsr
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
    ,t.transition_count DESC;

-- ============================================================
-- SPECIALTY -> SPECIALTY ORDER 1 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_specialty_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_specialty_order1`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_pairs AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,specialty_codes
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
        ,current_specialty
        ,next_specialty
    FROM visit_pairs
    CROSS JOIN UNNEST(specialty_codes)                   AS current_specialty
    CROSS JOIN UNNEST(next_specialty_codes)              AS next_specialty
    WHERE next_specialty_codes IS NOT NULL
),
transition_counts AS (
    SELECT
        current_specialty
        ,next_specialty
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_specialty, next_specialty, member_segment
),
specialty_totals AS (
    SELECT
        current_specialty
        ,member_segment
        ,SUM(transition_count)                           AS specialty_total
    FROM transition_counts
    GROUP BY current_specialty, member_segment
)
SELECT
    t.current_specialty
    ,t.next_specialty
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,s.specialty_total
    ,ROUND(t.transition_count / s.specialty_total, 4)   AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.specialty_total *
        LOG(t.transition_count / s.specialty_total)) OVER (
            PARTITION BY t.current_specialty, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN specialty_totals s
    ON t.current_specialty = s.current_specialty
    AND t.member_segment = s.member_segment
ORDER BY
    t.current_specialty
    ,t.transition_count DESC;

-- ============================================================
-- SPECIALTY -> DX ORDER 1 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_dx_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_dx_order1`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_pairs AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,specialty_codes
        ,LEAD(dx_list_raw) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS next_dx_list
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`
),
exploded AS (
    SELECT
        member_id
        ,member_segment
        ,current_specialty
        ,next_dx
    FROM visit_pairs
    CROSS JOIN UNNEST(specialty_codes)                   AS current_specialty
    CROSS JOIN UNNEST(next_dx_list)                      AS next_dx
    WHERE next_dx_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_specialty
        ,next_dx
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_specialty, next_dx, member_segment
),
specialty_totals AS (
    SELECT
        current_specialty
        ,member_segment
        ,SUM(transition_count)                           AS specialty_total
    FROM transition_counts
    GROUP BY current_specialty, member_segment
)
SELECT
    t.current_specialty
    ,t.next_dx
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,s.specialty_total
    ,ROUND(t.transition_count / s.specialty_total, 4)   AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.specialty_total *
        LOG(t.transition_count / s.specialty_total)) OVER (
            PARTITION BY t.current_specialty, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN specialty_totals s
    ON t.current_specialty = s.current_specialty
    AND t.member_segment = s.member_segment
ORDER BY
    t.current_specialty
    ,t.transition_count DESC;

-- ============================================================
-- SPECIALTY -> CCSR ORDER 1 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_ccsr_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_ccsr_order1`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_pairs AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,specialty_codes
        ,LEAD(ccsr_list) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS next_ccsr_list
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`
),
exploded AS (
    SELECT
        member_id
        ,member_segment
        ,current_specialty
        ,next_ccsr
    FROM visit_pairs
    CROSS JOIN UNNEST(specialty_codes)                   AS current_specialty
    CROSS JOIN UNNEST(next_ccsr_list)                    AS next_ccsr
    WHERE next_ccsr_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_specialty
        ,next_ccsr
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_specialty, next_ccsr, member_segment
),
specialty_totals AS (
    SELECT
        current_specialty
        ,member_segment
        ,SUM(transition_count)                           AS specialty_total
    FROM transition_counts
    GROUP BY current_specialty, member_segment
)
SELECT
    t.current_specialty
    ,t.next_ccsr
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,s.specialty_total
    ,ROUND(t.transition_count / s.specialty_total, 4)   AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.specialty_total *
        LOG(t.transition_count / s.specialty_total)) OVER (
            PARTITION BY t.current_specialty, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN specialty_totals s
    ON t.current_specialty = s.current_specialty
    AND t.member_segment = s.member_segment
ORDER BY
    t.current_specialty
    ,t.transition_count DESC;

-- ============================================================
-- CCSR -> SPECIALTY ORDER 1 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_specialty_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_specialty_order1`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_pairs AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,ccsr_list
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
        ,current_ccsr
        ,next_specialty
    FROM visit_pairs
    CROSS JOIN UNNEST(ccsr_list)                         AS current_ccsr
    CROSS JOIN UNNEST(next_specialty_codes)              AS next_specialty
    WHERE next_specialty_codes IS NOT NULL
),
transition_counts AS (
    SELECT
        current_ccsr
        ,next_specialty
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_ccsr, next_specialty, member_segment
),
ccsr_totals AS (
    SELECT
        current_ccsr
        ,member_segment
        ,SUM(transition_count)                           AS ccsr_total
    FROM transition_counts
    GROUP BY current_ccsr, member_segment
)
SELECT
    t.current_ccsr
    ,t.next_specialty
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,c.ccsr_total
    ,ROUND(t.transition_count / c.ccsr_total, 4)        AS conditional_probability
    ,ROUND(-SUM(t.transition_count / c.ccsr_total *
        LOG(t.transition_count / c.ccsr_total)) OVER (
            PARTITION BY t.current_ccsr, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN ccsr_totals c
    ON t.current_ccsr = c.current_ccsr
    AND t.member_segment = c.member_segment
ORDER BY
    t.current_ccsr
    ,t.transition_count DESC;

-- ============================================================
-- CCSR -> DX ORDER 1 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_dx_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_dx_order1`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_pairs AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,ccsr_list
        ,LEAD(dx_list_raw) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS next_dx_list
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`
),
exploded AS (
    SELECT
        member_id
        ,member_segment
        ,current_ccsr
        ,next_dx
    FROM visit_pairs
    CROSS JOIN UNNEST(ccsr_list)                         AS current_ccsr
    CROSS JOIN UNNEST(next_dx_list)                      AS next_dx
    WHERE next_dx_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_ccsr
        ,next_dx
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_ccsr, next_dx, member_segment
),
ccsr_totals AS (
    SELECT
        current_ccsr
        ,member_segment
        ,SUM(transition_count)                           AS ccsr_total
    FROM transition_counts
    GROUP BY current_ccsr, member_segment
)
SELECT
    t.current_ccsr
    ,t.next_dx
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,c.ccsr_total
    ,ROUND(t.transition_count / c.ccsr_total, 4)        AS conditional_probability
    ,ROUND(-SUM(t.transition_count / c.ccsr_total *
        LOG(t.transition_count / c.ccsr_total)) OVER (
            PARTITION BY t.current_ccsr, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN ccsr_totals c
    ON t.current_ccsr = c.current_ccsr
    AND t.member_segment = c.member_segment
ORDER BY
    t.current_ccsr
    ,t.transition_count DESC;

-- ============================================================
-- CCSR -> CCSR ORDER 1 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_ccsr_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_ccsr_order1`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_pairs AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,ccsr_list
        ,LEAD(ccsr_list) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS next_ccsr_list
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`
),
exploded AS (
    SELECT
        member_id
        ,member_segment
        ,current_ccsr
        ,next_ccsr
    FROM visit_pairs
    CROSS JOIN UNNEST(ccsr_list)                         AS current_ccsr
    CROSS JOIN UNNEST(next_ccsr_list)                    AS next_ccsr
    WHERE next_ccsr_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_ccsr
        ,next_ccsr
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_ccsr, next_ccsr, member_segment
),
ccsr_totals AS (
    SELECT
        current_ccsr
        ,member_segment
        ,SUM(transition_count)                           AS ccsr_total
    FROM transition_counts
    GROUP BY current_ccsr, member_segment
)
SELECT
    t.current_ccsr
    ,t.next_ccsr
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,c.ccsr_total
    ,ROUND(t.transition_count / c.ccsr_total, 4)        AS conditional_probability
    ,ROUND(-SUM(t.transition_count / c.ccsr_total *
        LOG(t.transition_count / c.ccsr_total)) OVER (
            PARTITION BY t.current_ccsr, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN ccsr_totals c
    ON t.current_ccsr = c.current_ccsr
    AND t.member_segment = c.member_segment
ORDER BY
    t.current_ccsr
    ,t.transition_count DESC;
