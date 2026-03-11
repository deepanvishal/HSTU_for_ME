-- ============================================================
-- ORDER 1: DX -> SPECIALTY
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order1`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH next_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_dx, t.trigger_dx_clean
        ,t.trigger_dx_desc, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment
        ,f.specialty_ctg_cd                              AS next_specialty
        ,sp.long_dscrptn                                 AS next_specialty_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON t.member_id = f.member_id
        AND v2.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
        ON f.specialty_ctg_cd = sp.global_lookup_cd
        AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
    WHERE f.specialty_ctg_cd IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,next_specialty, next_specialty_desc
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM next_claims
    GROUP BY
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,next_specialty, next_specialty_desc
        ,member_segment
),
dx_totals AS (
    SELECT trigger_dx, member_segment, SUM(transition_count) AS dx_total
    FROM transition_counts GROUP BY trigger_dx, member_segment
)
SELECT
    t.trigger_dx                                         AS current_dx
    ,t.trigger_dx_desc                                   AS current_dx_desc
    ,t.trigger_ccsr                                      AS current_ccsr
    ,t.trigger_ccsr_desc                                 AS current_ccsr_desc
    ,t.next_specialty, t.next_specialty_desc
    ,t.member_segment, t.transition_count, t.unique_members, d.dx_total
    ,ROUND(t.transition_count / d.dx_total, 4)           AS conditional_probability
    ,ROUND(-SUM(t.transition_count / d.dx_total *
        LOG(t.transition_count / d.dx_total)) OVER (
            PARTITION BY t.trigger_dx, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN dx_totals d ON t.trigger_dx = d.trigger_dx AND t.member_segment = d.member_segment
ORDER BY t.trigger_dx, t.transition_count DESC;


-- ============================================================
-- ORDER 1: DX -> DX
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order1`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH next_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_dx, t.trigger_dx_clean
        ,t.trigger_dx_desc, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment
        ,f.dx_raw                                        AS next_dx
        ,f.dx_clean                                      AS next_dx_clean
        ,dx_desc.icd9_dx_dscrptn                         AS next_dx_desc
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON t.member_id = f.member_id
        AND v2.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
        ON f.dx_raw = dx_desc.icd9_dx_cd
    WHERE f.dx_raw IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM next_claims
    GROUP BY
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
),
dx_totals AS (
    SELECT trigger_dx, member_segment, SUM(transition_count) AS dx_total
    FROM transition_counts GROUP BY trigger_dx, member_segment
)
SELECT
    t.trigger_dx                                         AS current_dx
    ,t.trigger_dx_desc                                   AS current_dx_desc
    ,t.trigger_ccsr                                      AS current_ccsr
    ,t.trigger_ccsr_desc                                 AS current_ccsr_desc
    ,t.next_dx, t.next_dx_desc, t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, d.dx_total
    ,ROUND(t.transition_count / d.dx_total, 4)           AS conditional_probability
    ,ROUND(-SUM(t.transition_count / d.dx_total *
        LOG(t.transition_count / d.dx_total)) OVER (
            PARTITION BY t.trigger_dx, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN dx_totals d ON t.trigger_dx = d.trigger_dx AND t.member_segment = d.member_segment
ORDER BY t.trigger_dx, t.transition_count DESC;


-- ============================================================
-- ORDER 1: DX -> CCSR
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order1`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH next_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_dx, t.trigger_dx_clean
        ,t.trigger_dx_desc, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON t.member_id = f.member_id
        AND v2.visit_date = f.visit_date
    WHERE f.ccsr_category IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,next_ccsr, next_ccsr_desc
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM next_claims
    GROUP BY
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,next_ccsr, next_ccsr_desc, member_segment
),
dx_totals AS (
    SELECT trigger_dx, member_segment, SUM(transition_count) AS dx_total
    FROM transition_counts GROUP BY trigger_dx, member_segment
)
SELECT
    t.trigger_dx                                         AS current_dx
    ,t.trigger_dx_desc                                   AS current_dx_desc
    ,t.trigger_ccsr                                      AS current_ccsr
    ,t.trigger_ccsr_desc                                 AS current_ccsr_desc
    ,t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, d.dx_total
    ,ROUND(t.transition_count / d.dx_total, 4)           AS conditional_probability
    ,ROUND(-SUM(t.transition_count / d.dx_total *
        LOG(t.transition_count / d.dx_total)) OVER (
            PARTITION BY t.trigger_dx, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN dx_totals d ON t.trigger_dx = d.trigger_dx AND t.member_segment = d.member_segment
ORDER BY t.trigger_dx, t.transition_count DESC;


-- ============================================================
-- ORDER 1: SPECIALTY -> SPECIALTY
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order1`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH next_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_specialty, t.trigger_specialty_desc
        ,t.member_segment
        ,f.specialty_ctg_cd                              AS next_specialty
        ,sp.long_dscrptn                                 AS next_specialty_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON t.member_id = f.member_id
        AND v2.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
        ON f.specialty_ctg_cd = sp.global_lookup_cd
        AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
    WHERE f.specialty_ctg_cd IS NOT NULL
        AND t.trigger_specialty IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_specialty, trigger_specialty_desc
        ,next_specialty, next_specialty_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM next_claims
    GROUP BY trigger_specialty, trigger_specialty_desc, next_specialty, next_specialty_desc, member_segment
),
specialty_totals AS (
    SELECT trigger_specialty, member_segment, SUM(transition_count) AS specialty_total
    FROM transition_counts GROUP BY trigger_specialty, member_segment
)
SELECT
    t.trigger_specialty                                  AS current_specialty
    ,t.trigger_specialty_desc                            AS current_specialty_desc
    ,t.next_specialty, t.next_specialty_desc
    ,t.member_segment, t.transition_count, t.unique_members, s.specialty_total
    ,ROUND(t.transition_count / s.specialty_total, 4)    AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.specialty_total *
        LOG(t.transition_count / s.specialty_total)) OVER (
            PARTITION BY t.trigger_specialty, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN specialty_totals s ON t.trigger_specialty = s.trigger_specialty AND t.member_segment = s.member_segment
ORDER BY t.trigger_specialty, t.transition_count DESC;


-- ============================================================
-- ORDER 1: SPECIALTY -> DX
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order1`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH next_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_specialty, t.trigger_specialty_desc
        ,t.member_segment
        ,f.dx_raw                                        AS next_dx
        ,f.dx_clean                                      AS next_dx_clean
        ,dx_desc.icd9_dx_dscrptn                         AS next_dx_desc
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON t.member_id = f.member_id
        AND v2.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
        ON f.dx_raw = dx_desc.icd9_dx_cd
    WHERE f.dx_raw IS NOT NULL
        AND t.trigger_specialty IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_specialty, trigger_specialty_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM next_claims
    GROUP BY
        trigger_specialty, trigger_specialty_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
),
specialty_totals AS (
    SELECT trigger_specialty, member_segment, SUM(transition_count) AS specialty_total
    FROM transition_counts GROUP BY trigger_specialty, member_segment
)
SELECT
    t.trigger_specialty                                  AS current_specialty
    ,t.trigger_specialty_desc                            AS current_specialty_desc
    ,t.next_dx, t.next_dx_desc, t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, s.specialty_total
    ,ROUND(t.transition_count / s.specialty_total, 4)    AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.specialty_total *
        LOG(t.transition_count / s.specialty_total)) OVER (
            PARTITION BY t.trigger_specialty, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN specialty_totals s ON t.trigger_specialty = s.trigger_specialty AND t.member_segment = s.member_segment
ORDER BY t.trigger_specialty, t.transition_count DESC;


-- ============================================================
-- ORDER 1: SPECIALTY -> CCSR
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order1`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH next_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_specialty, t.trigger_specialty_desc
        ,t.member_segment
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON t.member_id = f.member_id
        AND v2.visit_date = f.visit_date
    WHERE f.ccsr_category IS NOT NULL
        AND t.trigger_specialty IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_specialty, trigger_specialty_desc
        ,next_ccsr, next_ccsr_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM next_claims
    GROUP BY trigger_specialty, trigger_specialty_desc, next_ccsr, next_ccsr_desc, member_segment
),
specialty_totals AS (
    SELECT trigger_specialty, member_segment, SUM(transition_count) AS specialty_total
    FROM transition_counts GROUP BY trigger_specialty, member_segment
)
SELECT
    t.trigger_specialty                                  AS current_specialty
    ,t.trigger_specialty_desc                            AS current_specialty_desc
    ,t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, s.specialty_total
    ,ROUND(t.transition_count / s.specialty_total, 4)    AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.specialty_total *
        LOG(t.transition_count / s.specialty_total)) OVER (
            PARTITION BY t.trigger_specialty, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN specialty_totals s ON t.trigger_specialty = s.trigger_specialty AND t.member_segment = s.member_segment
ORDER BY t.trigger_specialty, t.transition_count DESC;


-- ============================================================
-- ORDER 1: CCSR -> SPECIALTY
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order1`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH next_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment
        ,f.specialty_ctg_cd                              AS next_specialty
        ,sp.long_dscrptn                                 AS next_specialty_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON t.member_id = f.member_id
        AND v2.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
        ON f.specialty_ctg_cd = sp.global_lookup_cd
        AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
    WHERE f.specialty_ctg_cd IS NOT NULL
        AND t.trigger_ccsr IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_ccsr, trigger_ccsr_desc
        ,next_specialty, next_specialty_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM next_claims
    GROUP BY trigger_ccsr, trigger_ccsr_desc, next_specialty, next_specialty_desc, member_segment
),
ccsr_totals AS (
    SELECT trigger_ccsr, member_segment, SUM(transition_count) AS ccsr_total
    FROM transition_counts GROUP BY trigger_ccsr, member_segment
)
SELECT
    t.trigger_ccsr                                       AS current_ccsr
    ,t.trigger_ccsr_desc                                 AS current_ccsr_desc
    ,t.next_specialty, t.next_specialty_desc
    ,t.member_segment, t.transition_count, t.unique_members, c.ccsr_total
    ,ROUND(t.transition_count / c.ccsr_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / c.ccsr_total *
        LOG(t.transition_count / c.ccsr_total)) OVER (
            PARTITION BY t.trigger_ccsr, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN ccsr_totals c ON t.trigger_ccsr = c.trigger_ccsr AND t.member_segment = c.member_segment
ORDER BY t.trigger_ccsr, t.transition_count DESC;


-- ============================================================
-- ORDER 1: CCSR -> DX
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_dx_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_dx_order1`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH next_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment
        ,f.dx_raw                                        AS next_dx
        ,f.dx_clean                                      AS next_dx_clean
        ,dx_desc.icd9_dx_dscrptn                         AS next_dx_desc
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON t.member_id = f.member_id
        AND v2.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
        ON f.dx_raw = dx_desc.icd9_dx_cd
    WHERE f.dx_raw IS NOT NULL
        AND t.trigger_ccsr IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_ccsr, trigger_ccsr_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM next_claims
    GROUP BY
        trigger_ccsr, trigger_ccsr_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
),
ccsr_totals AS (
    SELECT trigger_ccsr, member_segment, SUM(transition_count) AS ccsr_total
    FROM transition_counts GROUP BY trigger_ccsr, member_segment
)
SELECT
    t.trigger_ccsr                                       AS current_ccsr
    ,t.trigger_ccsr_desc                                 AS current_ccsr_desc
    ,t.next_dx, t.next_dx_desc, t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, c.ccsr_total
    ,ROUND(t.transition_count / c.ccsr_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / c.ccsr_total *
        LOG(t.transition_count / c.ccsr_total)) OVER (
            PARTITION BY t.trigger_ccsr, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN ccsr_totals c ON t.trigger_ccsr = c.trigger_ccsr AND t.member_segment = c.member_segment
ORDER BY t.trigger_ccsr, t.transition_count DESC;


-- ============================================================
-- ORDER 1: CCSR -> CCSR
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_ccsr_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_ccsr_order1`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH next_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON t.member_id = f.member_id
        AND v2.visit_date = f.visit_date
    WHERE f.ccsr_category IS NOT NULL
        AND t.trigger_ccsr IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_ccsr, trigger_ccsr_desc
        ,next_ccsr, next_ccsr_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM next_claims
    GROUP BY trigger_ccsr, trigger_ccsr_desc, next_ccsr, next_ccsr_desc, member_segment
),
ccsr_totals AS (
    SELECT trigger_ccsr, member_segment, SUM(transition_count) AS ccsr_total
    FROM transition_counts GROUP BY trigger_ccsr, member_segment
)
SELECT
    t.trigger_ccsr                                       AS current_ccsr
    ,t.trigger_ccsr_desc                                 AS current_ccsr_desc
    ,t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, c.ccsr_total
    ,ROUND(t.transition_count / c.ccsr_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / c.ccsr_total *
        LOG(t.transition_count / c.ccsr_total)) OVER (
            PARTITION BY t.trigger_ccsr, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN ccsr_totals c ON t.trigger_ccsr = c.trigger_ccsr AND t.member_segment = c.member_segment
ORDER BY t.trigger_ccsr, t.transition_count DESC;
