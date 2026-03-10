-- ============================================================
-- DX -> DX ORDER 2 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_dx_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_dx_order2`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_triplets AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,LAG(dx_list_raw) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS dx_v1
        ,dx_list_raw                                     AS dx_v2
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
        ,current_dx_v1
        ,current_dx_v2
        ,next_dx
    FROM visit_triplets
    CROSS JOIN UNNEST(dx_v1)                             AS current_dx_v1
    CROSS JOIN UNNEST(dx_v2)                             AS current_dx_v2
    CROSS JOIN UNNEST(next_dx_list)                      AS next_dx
    WHERE dx_v1 IS NOT NULL
        AND next_dx_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_dx_v1
        ,current_dx_v2
        ,next_dx
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_dx_v1, current_dx_v2, next_dx, member_segment
),
dx_totals AS (
    SELECT
        current_dx_v1
        ,current_dx_v2
        ,member_segment
        ,SUM(transition_count)                           AS dx_total
    FROM transition_counts
    GROUP BY current_dx_v1, current_dx_v2, member_segment
)
SELECT
    t.current_dx_v1
    ,t.current_dx_v2
    ,t.next_dx
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,d.dx_total
    ,ROUND(t.transition_count / d.dx_total, 4)          AS conditional_probability
    ,ROUND(-SUM(t.transition_count / d.dx_total *
        LOG(t.transition_count / d.dx_total)) OVER (
            PARTITION BY t.current_dx_v1, t.current_dx_v2, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN dx_totals d
    ON t.current_dx_v1 = d.current_dx_v1
    AND t.current_dx_v2 = d.current_dx_v2
    AND t.member_segment = d.member_segment
ORDER BY
    t.current_dx_v1
    ,t.current_dx_v2
    ,t.transition_count DESC;

-- ============================================================
-- DX -> SPECIALTY ORDER 2 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_specialty_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_specialty_order2`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_triplets AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,LAG(dx_list_raw) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS dx_v1
        ,dx_list_raw                                     AS dx_v2
        ,LEAD(specialty_codes) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS next_specialty_list
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`
),
exploded AS (
    SELECT
        member_id
        ,member_segment
        ,current_dx_v1
        ,current_dx_v2
        ,next_specialty
    FROM visit_triplets
    CROSS JOIN UNNEST(dx_v1)                             AS current_dx_v1
    CROSS JOIN UNNEST(dx_v2)                             AS current_dx_v2
    CROSS JOIN UNNEST(next_specialty_list)               AS next_specialty
    WHERE dx_v1 IS NOT NULL
        AND next_specialty_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_dx_v1
        ,current_dx_v2
        ,next_specialty
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_dx_v1, current_dx_v2, next_specialty, member_segment
),
dx_totals AS (
    SELECT
        current_dx_v1
        ,current_dx_v2
        ,member_segment
        ,SUM(transition_count)                           AS dx_total
    FROM transition_counts
    GROUP BY current_dx_v1, current_dx_v2, member_segment
)
SELECT
    t.current_dx_v1
    ,t.current_dx_v2
    ,t.next_specialty
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,d.dx_total
    ,ROUND(t.transition_count / d.dx_total, 4)          AS conditional_probability
    ,ROUND(-SUM(t.transition_count / d.dx_total *
        LOG(t.transition_count / d.dx_total)) OVER (
            PARTITION BY t.current_dx_v1, t.current_dx_v2, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN dx_totals d
    ON t.current_dx_v1 = d.current_dx_v1
    AND t.current_dx_v2 = d.current_dx_v2
    AND t.member_segment = d.member_segment
ORDER BY
    t.current_dx_v1
    ,t.current_dx_v2
    ,t.transition_count DESC;

-- ============================================================
-- DX -> CCSR ORDER 2 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_ccsr_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_to_ccsr_order2`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_triplets AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,LAG(dx_list_raw) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS dx_v1
        ,dx_list_raw                                     AS dx_v2
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
        ,current_dx_v1
        ,current_dx_v2
        ,next_ccsr
    FROM visit_triplets
    CROSS JOIN UNNEST(dx_v1)                             AS current_dx_v1
    CROSS JOIN UNNEST(dx_v2)                             AS current_dx_v2
    CROSS JOIN UNNEST(next_ccsr_list)                    AS next_ccsr
    WHERE dx_v1 IS NOT NULL
        AND next_ccsr_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_dx_v1
        ,current_dx_v2
        ,next_ccsr
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_dx_v1, current_dx_v2, next_ccsr, member_segment
),
dx_totals AS (
    SELECT
        current_dx_v1
        ,current_dx_v2
        ,member_segment
        ,SUM(transition_count)                           AS dx_total
    FROM transition_counts
    GROUP BY current_dx_v1, current_dx_v2, member_segment
)
SELECT
    t.current_dx_v1
    ,t.current_dx_v2
    ,t.next_ccsr
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,d.dx_total
    ,ROUND(t.transition_count / d.dx_total, 4)          AS conditional_probability
    ,ROUND(-SUM(t.transition_count / d.dx_total *
        LOG(t.transition_count / d.dx_total)) OVER (
            PARTITION BY t.current_dx_v1, t.current_dx_v2, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN dx_totals d
    ON t.current_dx_v1 = d.current_dx_v1
    AND t.current_dx_v2 = d.current_dx_v2
    AND t.member_segment = d.member_segment
ORDER BY
    t.current_dx_v1
    ,t.current_dx_v2
    ,t.transition_count DESC;

-- ============================================================
-- SPECIALTY -> SPECIALTY ORDER 2 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_specialty_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_specialty_order2`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_triplets AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,LAG(specialty_codes) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS specialty_v1
        ,specialty_codes                                 AS specialty_v2
        ,LEAD(specialty_codes) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS next_specialty_list
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`
),
exploded AS (
    SELECT
        member_id
        ,member_segment
        ,current_specialty_v1
        ,current_specialty_v2
        ,next_specialty
    FROM visit_triplets
    CROSS JOIN UNNEST(specialty_v1)                      AS current_specialty_v1
    CROSS JOIN UNNEST(specialty_v2)                      AS current_specialty_v2
    CROSS JOIN UNNEST(next_specialty_list)               AS next_specialty
    WHERE specialty_v1 IS NOT NULL
        AND next_specialty_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_specialty_v1
        ,current_specialty_v2
        ,next_specialty
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_specialty_v1, current_specialty_v2, next_specialty, member_segment
),
specialty_totals AS (
    SELECT
        current_specialty_v1
        ,current_specialty_v2
        ,member_segment
        ,SUM(transition_count)                           AS specialty_total
    FROM transition_counts
    GROUP BY current_specialty_v1, current_specialty_v2, member_segment
)
SELECT
    t.current_specialty_v1
    ,t.current_specialty_v2
    ,t.next_specialty
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,s.specialty_total
    ,ROUND(t.transition_count / s.specialty_total, 4)   AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.specialty_total *
        LOG(t.transition_count / s.specialty_total)) OVER (
            PARTITION BY t.current_specialty_v1, t.current_specialty_v2, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN specialty_totals s
    ON t.current_specialty_v1 = s.current_specialty_v1
    AND t.current_specialty_v2 = s.current_specialty_v2
    AND t.member_segment = s.member_segment
ORDER BY
    t.current_specialty_v1
    ,t.current_specialty_v2
    ,t.transition_count DESC;

-- ============================================================
-- SPECIALTY -> DX ORDER 2 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_dx_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_dx_order2`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_triplets AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,LAG(specialty_codes) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS specialty_v1
        ,specialty_codes                                 AS specialty_v2
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
        ,current_specialty_v1
        ,current_specialty_v2
        ,next_dx
    FROM visit_triplets
    CROSS JOIN UNNEST(specialty_v1)                      AS current_specialty_v1
    CROSS JOIN UNNEST(specialty_v2)                      AS current_specialty_v2
    CROSS JOIN UNNEST(next_dx_list)                      AS next_dx
    WHERE specialty_v1 IS NOT NULL
        AND next_dx_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_specialty_v1
        ,current_specialty_v2
        ,next_dx
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_specialty_v1, current_specialty_v2, next_dx, member_segment
),
specialty_totals AS (
    SELECT
        current_specialty_v1
        ,current_specialty_v2
        ,member_segment
        ,SUM(transition_count)                           AS specialty_total
    FROM transition_counts
    GROUP BY current_specialty_v1, current_specialty_v2, member_segment
)
SELECT
    t.current_specialty_v1
    ,t.current_specialty_v2
    ,t.next_dx
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,s.specialty_total
    ,ROUND(t.transition_count / s.specialty_total, 4)   AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.specialty_total *
        LOG(t.transition_count / s.specialty_total)) OVER (
            PARTITION BY t.current_specialty_v1, t.current_specialty_v2, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN specialty_totals s
    ON t.current_specialty_v1 = s.current_specialty_v1
    AND t.current_specialty_v2 = s.current_specialty_v2
    AND t.member_segment = s.member_segment
ORDER BY
    t.current_specialty_v1
    ,t.current_specialty_v2
    ,t.transition_count DESC;

-- ============================================================
-- SPECIALTY -> CCSR ORDER 2 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_ccsr_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_to_ccsr_order2`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_triplets AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,LAG(specialty_codes) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS specialty_v1
        ,specialty_codes                                 AS specialty_v2
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
        ,current_specialty_v1
        ,current_specialty_v2
        ,next_ccsr
    FROM visit_triplets
    CROSS JOIN UNNEST(specialty_v1)                      AS current_specialty_v1
    CROSS JOIN UNNEST(specialty_v2)                      AS current_specialty_v2
    CROSS JOIN UNNEST(next_ccsr_list)                    AS next_ccsr
    WHERE specialty_v1 IS NOT NULL
        AND next_ccsr_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_specialty_v1
        ,current_specialty_v2
        ,next_ccsr
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_specialty_v1, current_specialty_v2, next_ccsr, member_segment
),
specialty_totals AS (
    SELECT
        current_specialty_v1
        ,current_specialty_v2
        ,member_segment
        ,SUM(transition_count)                           AS specialty_total
    FROM transition_counts
    GROUP BY current_specialty_v1, current_specialty_v2, member_segment
)
SELECT
    t.current_specialty_v1
    ,t.current_specialty_v2
    ,t.next_ccsr
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,s.specialty_total
    ,ROUND(t.transition_count / s.specialty_total, 4)   AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.specialty_total *
        LOG(t.transition_count / s.specialty_total)) OVER (
            PARTITION BY t.current_specialty_v1, t.current_specialty_v2, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN specialty_totals s
    ON t.current_specialty_v1 = s.current_specialty_v1
    AND t.current_specialty_v2 = s.current_specialty_v2
    AND t.member_segment = s.member_segment
ORDER BY
    t.current_specialty_v1
    ,t.current_specialty_v2
    ,t.transition_count DESC;

-- ============================================================
-- CCSR -> SPECIALTY ORDER 2 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_specialty_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_specialty_order2`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_triplets AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,LAG(ccsr_list) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS ccsr_v1
        ,ccsr_list                                       AS ccsr_v2
        ,LEAD(specialty_codes) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS next_specialty_list
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`
),
exploded AS (
    SELECT
        member_id
        ,member_segment
        ,current_ccsr_v1
        ,current_ccsr_v2
        ,next_specialty
    FROM visit_triplets
    CROSS JOIN UNNEST(ccsr_v1)                           AS current_ccsr_v1
    CROSS JOIN UNNEST(ccsr_v2)                           AS current_ccsr_v2
    CROSS JOIN UNNEST(next_specialty_list)               AS next_specialty
    WHERE ccsr_v1 IS NOT NULL
        AND next_specialty_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_ccsr_v1
        ,current_ccsr_v2
        ,next_specialty
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_ccsr_v1, current_ccsr_v2, next_specialty, member_segment
),
ccsr_totals AS (
    SELECT
        current_ccsr_v1
        ,current_ccsr_v2
        ,member_segment
        ,SUM(transition_count)                           AS ccsr_total
    FROM transition_counts
    GROUP BY current_ccsr_v1, current_ccsr_v2, member_segment
)
SELECT
    t.current_ccsr_v1
    ,t.current_ccsr_v2
    ,t.next_specialty
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,c.ccsr_total
    ,ROUND(t.transition_count / c.ccsr_total, 4)        AS conditional_probability
    ,ROUND(-SUM(t.transition_count / c.ccsr_total *
        LOG(t.transition_count / c.ccsr_total)) OVER (
            PARTITION BY t.current_ccsr_v1, t.current_ccsr_v2, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN ccsr_totals c
    ON t.current_ccsr_v1 = c.current_ccsr_v1
    AND t.current_ccsr_v2 = c.current_ccsr_v2
    AND t.member_segment = c.member_segment
ORDER BY
    t.current_ccsr_v1
    ,t.current_ccsr_v2
    ,t.transition_count DESC;

-- ============================================================
-- CCSR -> DX ORDER 2 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_dx_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_dx_order2`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_triplets AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,LAG(ccsr_list) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS ccsr_v1
        ,ccsr_list                                       AS ccsr_v2
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
        ,current_ccsr_v1
        ,current_ccsr_v2
        ,next_dx
    FROM visit_triplets
    CROSS JOIN UNNEST(ccsr_v1)                           AS current_ccsr_v1
    CROSS JOIN UNNEST(ccsr_v2)                           AS current_ccsr_v2
    CROSS JOIN UNNEST(next_dx_list)                      AS next_dx
    WHERE ccsr_v1 IS NOT NULL
        AND next_dx_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_ccsr_v1
        ,current_ccsr_v2
        ,next_dx
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_ccsr_v1, current_ccsr_v2, next_dx, member_segment
),
ccsr_totals AS (
    SELECT
        current_ccsr_v1
        ,current_ccsr_v2
        ,member_segment
        ,SUM(transition_count)                           AS ccsr_total
    FROM transition_counts
    GROUP BY current_ccsr_v1, current_ccsr_v2, member_segment
)
SELECT
    t.current_ccsr_v1
    ,t.current_ccsr_v2
    ,t.next_dx
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,c.ccsr_total
    ,ROUND(t.transition_count / c.ccsr_total, 4)        AS conditional_probability
    ,ROUND(-SUM(t.transition_count / c.ccsr_total *
        LOG(t.transition_count / c.ccsr_total)) OVER (
            PARTITION BY t.current_ccsr_v1, t.current_ccsr_v2, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN ccsr_totals c
    ON t.current_ccsr_v1 = c.current_ccsr_v1
    AND t.current_ccsr_v2 = c.current_ccsr_v2
    AND t.member_segment = c.member_segment
ORDER BY
    t.current_ccsr_v1
    ,t.current_ccsr_v2
    ,t.transition_count DESC;

-- ============================================================
-- CCSR -> CCSR ORDER 2 TRANSITION
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_ccsr_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_ccsr_to_ccsr_order2`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_triplets AS (
    SELECT
        member_id
        ,member_segment
        ,visit_date
        ,LAG(ccsr_list) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS ccsr_v1
        ,ccsr_list                                       AS ccsr_v2
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
        ,current_ccsr_v1
        ,current_ccsr_v2
        ,next_ccsr
    FROM visit_triplets
    CROSS JOIN UNNEST(ccsr_v1)                           AS current_ccsr_v1
    CROSS JOIN UNNEST(ccsr_v2)                           AS current_ccsr_v2
    CROSS JOIN UNNEST(next_ccsr_list)                    AS next_ccsr
    WHERE ccsr_v1 IS NOT NULL
        AND next_ccsr_list IS NOT NULL
),
transition_counts AS (
    SELECT
        current_ccsr_v1
        ,current_ccsr_v2
        ,next_ccsr
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_ccsr_v1, current_ccsr_v2, next_ccsr, member_segment
),
ccsr_totals AS (
    SELECT
        current_ccsr_v1
        ,current_ccsr_v2
        ,member_segment
        ,SUM(transition_count)                           AS ccsr_total
    FROM transition_counts
    GROUP BY current_ccsr_v1, current_ccsr_v2, member_segment
)
SELECT
    t.current_ccsr_v1
    ,t.current_ccsr_v2
    ,t.next_ccsr
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,c.ccsr_total
    ,ROUND(t.transition_count / c.ccsr_total, 4)        AS conditional_probability
    ,ROUND(-SUM(t.transition_count / c.ccsr_total *
        LOG(t.transition_count / c.ccsr_total)) OVER (
            PARTITION BY t.current_ccsr_v1, t.current_ccsr_v2, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN ccsr_totals c
    ON t.current_ccsr_v1 = c.current_ccsr_v1
    AND t.current_ccsr_v2 = c.current_ccsr_v2
    AND t.member_segment = c.member_segment
ORDER BY
    t.current_ccsr_v1
    ,t.current_ccsr_v2
    ,t.transition_count DESC;
