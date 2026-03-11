-- ============================================================
-- ORDER 2: DX -> SPECIALTY
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order2`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH v2_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_dx, t.trigger_dx_clean
        ,t.trigger_dx_desc, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment, t.trigger_rank
        ,f.dx_raw                                        AS v2_dx
        ,f.dx_clean                                      AS v2_dx_clean
        ,dx_desc.icd9_dx_dscrptn                         AS v2_dx_desc
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
v3_claims AS (
    SELECT DISTINCT
        v.member_id, v.trigger_date, v.trigger_dx, v.trigger_dx_clean
        ,v.trigger_dx_desc, v.trigger_ccsr, v.trigger_ccsr_desc
        ,v.member_segment, v.v2_dx, v.v2_dx_clean, v.v2_dx_desc
        ,f.specialty_ctg_cd                              AS next_specialty
        ,sp.long_dscrptn                                 AS next_specialty_desc
    FROM v2_claims v
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v3
        ON v.member_id = v3.member_id
        AND v3.visit_rank = v.trigger_rank + 2
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON v.member_id = f.member_id
        AND v3.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
        ON f.specialty_ctg_cd = sp.global_lookup_cd
        AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
    WHERE f.specialty_ctg_cd IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,v2_dx, v2_dx_clean, v2_dx_desc
        ,next_specialty, next_specialty_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM v3_claims
    GROUP BY
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,v2_dx, v2_dx_clean, v2_dx_desc
        ,next_specialty, next_specialty_desc, member_segment
),
pair_totals AS (
    SELECT trigger_dx, v2_dx, member_segment, SUM(transition_count) AS pair_total
    FROM transition_counts GROUP BY trigger_dx, v2_dx, member_segment
)
SELECT
    t.trigger_dx                                         AS current_dx_v1
    ,t.trigger_dx_desc                                   AS current_dx_v1_desc
    ,t.trigger_ccsr                                      AS current_ccsr_v1
    ,t.trigger_ccsr_desc                                 AS current_ccsr_v1_desc
    ,t.v2_dx                                             AS current_dx_v2
    ,t.v2_dx_desc                                        AS current_dx_v2_desc
    ,t.next_specialty, t.next_specialty_desc
    ,t.member_segment, t.transition_count, t.unique_members, p.pair_total
    ,ROUND(t.transition_count / p.pair_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / p.pair_total *
        LOG(t.transition_count / p.pair_total)) OVER (
            PARTITION BY t.trigger_dx, t.v2_dx, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN pair_totals p
    ON t.trigger_dx = p.trigger_dx
    AND t.v2_dx = p.v2_dx
    AND t.member_segment = p.member_segment
ORDER BY t.trigger_dx, t.v2_dx, t.transition_count DESC;


-- ============================================================
-- ORDER 2: DX -> DX
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_dx_order2`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH v2_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_dx, t.trigger_dx_clean
        ,t.trigger_dx_desc, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment, t.trigger_rank
        ,f.dx_raw                                        AS v2_dx
        ,f.dx_clean                                      AS v2_dx_clean
        ,dx_desc.icd9_dx_dscrptn                         AS v2_dx_desc
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
v3_claims AS (
    SELECT DISTINCT
        v.member_id, v.trigger_date, v.trigger_dx, v.trigger_dx_clean
        ,v.trigger_dx_desc, v.trigger_ccsr, v.trigger_ccsr_desc
        ,v.member_segment, v.v2_dx, v.v2_dx_clean, v.v2_dx_desc
        ,f.dx_raw                                        AS next_dx
        ,f.dx_clean                                      AS next_dx_clean
        ,dx_desc.icd9_dx_dscrptn                         AS next_dx_desc
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM v2_claims v
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v3
        ON v.member_id = v3.member_id
        AND v3.visit_rank = v.trigger_rank + 2
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON v.member_id = f.member_id
        AND v3.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
        ON f.dx_raw = dx_desc.icd9_dx_cd
    WHERE f.dx_raw IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,v2_dx, v2_dx_clean, v2_dx_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM v3_claims
    GROUP BY
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,v2_dx, v2_dx_clean, v2_dx_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
),
pair_totals AS (
    SELECT trigger_dx, v2_dx, member_segment, SUM(transition_count) AS pair_total
    FROM transition_counts GROUP BY trigger_dx, v2_dx, member_segment
)
SELECT
    t.trigger_dx                                         AS current_dx_v1
    ,t.trigger_dx_desc                                   AS current_dx_v1_desc
    ,t.trigger_ccsr                                      AS current_ccsr_v1
    ,t.trigger_ccsr_desc                                 AS current_ccsr_v1_desc
    ,t.v2_dx                                             AS current_dx_v2
    ,t.v2_dx_desc                                        AS current_dx_v2_desc
    ,t.next_dx, t.next_dx_desc, t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, p.pair_total
    ,ROUND(t.transition_count / p.pair_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / p.pair_total *
        LOG(t.transition_count / p.pair_total)) OVER (
            PARTITION BY t.trigger_dx, t.v2_dx, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN pair_totals p
    ON t.trigger_dx = p.trigger_dx
    AND t.v2_dx = p.v2_dx
    AND t.member_segment = p.member_segment
ORDER BY t.trigger_dx, t.v2_dx, t.transition_count DESC;


-- ============================================================
-- ORDER 2: DX -> CCSR
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_ccsr_order2`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH v2_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_dx, t.trigger_dx_clean
        ,t.trigger_dx_desc, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment, t.trigger_rank
        ,f.dx_raw                                        AS v2_dx
        ,f.dx_clean                                      AS v2_dx_clean
        ,dx_desc.icd9_dx_dscrptn                         AS v2_dx_desc
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
v3_claims AS (
    SELECT DISTINCT
        v.member_id, v.trigger_date, v.trigger_dx, v.trigger_dx_clean
        ,v.trigger_dx_desc, v.trigger_ccsr, v.trigger_ccsr_desc
        ,v.member_segment, v.v2_dx, v.v2_dx_clean, v.v2_dx_desc
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM v2_claims v
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v3
        ON v.member_id = v3.member_id
        AND v3.visit_rank = v.trigger_rank + 2
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON v.member_id = f.member_id
        AND v3.visit_date = f.visit_date
    WHERE f.ccsr_category IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,v2_dx, v2_dx_clean, v2_dx_desc
        ,next_ccsr, next_ccsr_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM v3_claims
    GROUP BY
        trigger_dx, trigger_dx_clean, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,v2_dx, v2_dx_clean, v2_dx_desc
        ,next_ccsr, next_ccsr_desc, member_segment
),
pair_totals AS (
    SELECT trigger_dx, v2_dx, member_segment, SUM(transition_count) AS pair_total
    FROM transition_counts GROUP BY trigger_dx, v2_dx, member_segment
)
SELECT
    t.trigger_dx                                         AS current_dx_v1
    ,t.trigger_dx_desc                                   AS current_dx_v1_desc
    ,t.trigger_ccsr                                      AS current_ccsr_v1
    ,t.trigger_ccsr_desc                                 AS current_ccsr_v1_desc
    ,t.v2_dx                                             AS current_dx_v2
    ,t.v2_dx_desc                                        AS current_dx_v2_desc
    ,t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, p.pair_total
    ,ROUND(t.transition_count / p.pair_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / p.pair_total *
        LOG(t.transition_count / p.pair_total)) OVER (
            PARTITION BY t.trigger_dx, t.v2_dx, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN pair_totals p
    ON t.trigger_dx = p.trigger_dx
    AND t.v2_dx = p.v2_dx
    AND t.member_segment = p.member_segment
ORDER BY t.trigger_dx, t.v2_dx, t.transition_count DESC;


-- ============================================================
-- ORDER 2: SPECIALTY -> SPECIALTY
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_specialty_order2`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH v2_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_specialty, t.trigger_specialty_desc
        ,t.member_segment, t.trigger_rank
        ,f.specialty_ctg_cd                              AS v2_specialty
        ,sp.long_dscrptn                                 AS v2_specialty_desc
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
v3_claims AS (
    SELECT DISTINCT
        v.member_id, v.trigger_date, v.trigger_specialty, v.trigger_specialty_desc
        ,v.member_segment, v.v2_specialty, v.v2_specialty_desc
        ,f.specialty_ctg_cd                              AS next_specialty
        ,sp.long_dscrptn                                 AS next_specialty_desc
    FROM v2_claims v
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v3
        ON v.member_id = v3.member_id
        AND v3.visit_rank = v.trigger_rank + 2
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON v.member_id = f.member_id
        AND v3.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
        ON f.specialty_ctg_cd = sp.global_lookup_cd
        AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
    WHERE f.specialty_ctg_cd IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_specialty, trigger_specialty_desc
        ,v2_specialty, v2_specialty_desc
        ,next_specialty, next_specialty_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM v3_claims
    GROUP BY
        trigger_specialty, trigger_specialty_desc
        ,v2_specialty, v2_specialty_desc
        ,next_specialty, next_specialty_desc, member_segment
),
pair_totals AS (
    SELECT trigger_specialty, v2_specialty, member_segment, SUM(transition_count) AS pair_total
    FROM transition_counts GROUP BY trigger_specialty, v2_specialty, member_segment
)
SELECT
    t.trigger_specialty                                  AS current_specialty_v1
    ,t.trigger_specialty_desc                            AS current_specialty_v1_desc
    ,t.v2_specialty                                      AS current_specialty_v2
    ,t.v2_specialty_desc                                 AS current_specialty_v2_desc
    ,t.next_specialty, t.next_specialty_desc
    ,t.member_segment, t.transition_count, t.unique_members, p.pair_total
    ,ROUND(t.transition_count / p.pair_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / p.pair_total *
        LOG(t.transition_count / p.pair_total)) OVER (
            PARTITION BY t.trigger_specialty, t.v2_specialty, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN pair_totals p
    ON t.trigger_specialty = p.trigger_specialty
    AND t.v2_specialty = p.v2_specialty
    AND t.member_segment = p.member_segment
ORDER BY t.trigger_specialty, t.v2_specialty, t.transition_count DESC;


-- ============================================================
-- ORDER 2: SPECIALTY -> DX
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_dx_order2`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH v2_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_specialty, t.trigger_specialty_desc
        ,t.member_segment, t.trigger_rank
        ,f.specialty_ctg_cd                              AS v2_specialty
        ,sp.long_dscrptn                                 AS v2_specialty_desc
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
v3_claims AS (
    SELECT DISTINCT
        v.member_id, v.trigger_date, v.trigger_specialty, v.trigger_specialty_desc
        ,v.member_segment, v.v2_specialty, v.v2_specialty_desc
        ,f.dx_raw                                        AS next_dx
        ,f.dx_clean                                      AS next_dx_clean
        ,dx_desc.icd9_dx_dscrptn                         AS next_dx_desc
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM v2_claims v
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v3
        ON v.member_id = v3.member_id
        AND v3.visit_rank = v.trigger_rank + 2
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON v.member_id = f.member_id
        AND v3.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
        ON f.dx_raw = dx_desc.icd9_dx_cd
    WHERE f.dx_raw IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_specialty, trigger_specialty_desc
        ,v2_specialty, v2_specialty_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM v3_claims
    GROUP BY
        trigger_specialty, trigger_specialty_desc
        ,v2_specialty, v2_specialty_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
),
pair_totals AS (
    SELECT trigger_specialty, v2_specialty, member_segment, SUM(transition_count) AS pair_total
    FROM transition_counts GROUP BY trigger_specialty, v2_specialty, member_segment
)
SELECT
    t.trigger_specialty                                  AS current_specialty_v1
    ,t.trigger_specialty_desc                            AS current_specialty_v1_desc
    ,t.v2_specialty                                      AS current_specialty_v2
    ,t.v2_specialty_desc                                 AS current_specialty_v2_desc
    ,t.next_dx, t.next_dx_desc, t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, p.pair_total
    ,ROUND(t.transition_count / p.pair_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / p.pair_total *
        LOG(t.transition_count / p.pair_total)) OVER (
            PARTITION BY t.trigger_specialty, t.v2_specialty, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN pair_totals p
    ON t.trigger_specialty = p.trigger_specialty
    AND t.v2_specialty = p.v2_specialty
    AND t.member_segment = p.member_segment
ORDER BY t.trigger_specialty, t.v2_specialty, t.transition_count DESC;


-- ============================================================
-- ORDER 2: SPECIALTY -> CCSR
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_specialty_to_ccsr_order2`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH v2_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_specialty, t.trigger_specialty_desc
        ,t.member_segment, t.trigger_rank
        ,f.specialty_ctg_cd                              AS v2_specialty
        ,sp.long_dscrptn                                 AS v2_specialty_desc
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
v3_claims AS (
    SELECT DISTINCT
        v.member_id, v.trigger_date, v.trigger_specialty, v.trigger_specialty_desc
        ,v.member_segment, v.v2_specialty, v.v2_specialty_desc
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM v2_claims v
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v3
        ON v.member_id = v3.member_id
        AND v3.visit_rank = v.trigger_rank + 2
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON v.member_id = f.member_id
        AND v3.visit_date = f.visit_date
    WHERE f.ccsr_category IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_specialty, trigger_specialty_desc
        ,v2_specialty, v2_specialty_desc
        ,next_ccsr, next_ccsr_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM v3_claims
    GROUP BY
        trigger_specialty, trigger_specialty_desc
        ,v2_specialty, v2_specialty_desc
        ,next_ccsr, next_ccsr_desc, member_segment
),
pair_totals AS (
    SELECT trigger_specialty, v2_specialty, member_segment, SUM(transition_count) AS pair_total
    FROM transition_counts GROUP BY trigger_specialty, v2_specialty, member_segment
)
SELECT
    t.trigger_specialty                                  AS current_specialty_v1
    ,t.trigger_specialty_desc                            AS current_specialty_v1_desc
    ,t.v2_specialty                                      AS current_specialty_v2
    ,t.v2_specialty_desc                                 AS current_specialty_v2_desc
    ,t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, p.pair_total
    ,ROUND(t.transition_count / p.pair_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / p.pair_total *
        LOG(t.transition_count / p.pair_total)) OVER (
            PARTITION BY t.trigger_specialty, t.v2_specialty, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN pair_totals p
    ON t.trigger_specialty = p.trigger_specialty
    AND t.v2_specialty = p.v2_specialty
    AND t.member_segment = p.member_segment
ORDER BY t.trigger_specialty, t.v2_specialty, t.transition_count DESC;


-- ============================================================
-- ORDER 2: CCSR -> SPECIALTY
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_specialty_order2`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH v2_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment, t.trigger_rank
        ,f.ccsr_category                                 AS v2_ccsr
        ,f.ccsr_category_description                     AS v2_ccsr_desc
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
v3_claims AS (
    SELECT DISTINCT
        v.member_id, v.trigger_date, v.trigger_ccsr, v.trigger_ccsr_desc
        ,v.member_segment, v.v2_ccsr, v.v2_ccsr_desc
        ,f.specialty_ctg_cd                              AS next_specialty
        ,sp.long_dscrptn                                 AS next_specialty_desc
    FROM v2_claims v
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v3
        ON v.member_id = v3.member_id
        AND v3.visit_rank = v.trigger_rank + 2
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON v.member_id = f.member_id
        AND v3.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
        ON f.specialty_ctg_cd = sp.global_lookup_cd
        AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
    WHERE f.specialty_ctg_cd IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_ccsr, trigger_ccsr_desc
        ,v2_ccsr, v2_ccsr_desc
        ,next_specialty, next_specialty_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM v3_claims
    GROUP BY
        trigger_ccsr, trigger_ccsr_desc
        ,v2_ccsr, v2_ccsr_desc
        ,next_specialty, next_specialty_desc, member_segment
),
pair_totals AS (
    SELECT trigger_ccsr, v2_ccsr, member_segment, SUM(transition_count) AS pair_total
    FROM transition_counts GROUP BY trigger_ccsr, v2_ccsr, member_segment
)
SELECT
    t.trigger_ccsr                                       AS current_ccsr_v1
    ,t.trigger_ccsr_desc                                 AS current_ccsr_v1_desc
    ,t.v2_ccsr                                           AS current_ccsr_v2
    ,t.v2_ccsr_desc                                      AS current_ccsr_v2_desc
    ,t.next_specialty, t.next_specialty_desc
    ,t.member_segment, t.transition_count, t.unique_members, p.pair_total
    ,ROUND(t.transition_count / p.pair_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / p.pair_total *
        LOG(t.transition_count / p.pair_total)) OVER (
            PARTITION BY t.trigger_ccsr, t.v2_ccsr, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN pair_totals p
    ON t.trigger_ccsr = p.trigger_ccsr
    AND t.v2_ccsr = p.v2_ccsr
    AND t.member_segment = p.member_segment
ORDER BY t.trigger_ccsr, t.v2_ccsr, t.transition_count DESC;


-- ============================================================
-- ORDER 2: CCSR -> DX
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_dx_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_dx_order2`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH v2_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment, t.trigger_rank
        ,f.ccsr_category                                 AS v2_ccsr
        ,f.ccsr_category_description                     AS v2_ccsr_desc
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
v3_claims AS (
    SELECT DISTINCT
        v.member_id, v.trigger_date, v.trigger_ccsr, v.trigger_ccsr_desc
        ,v.member_segment, v.v2_ccsr, v.v2_ccsr_desc
        ,f.dx_raw                                        AS next_dx
        ,f.dx_clean                                      AS next_dx_clean
        ,dx_desc.icd9_dx_dscrptn                         AS next_dx_desc
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM v2_claims v
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v3
        ON v.member_id = v3.member_id
        AND v3.visit_rank = v.trigger_rank + 2
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON v.member_id = f.member_id
        AND v3.visit_date = f.visit_date
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
        ON f.dx_raw = dx_desc.icd9_dx_cd
    WHERE f.dx_raw IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_ccsr, trigger_ccsr_desc
        ,v2_ccsr, v2_ccsr_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM v3_claims
    GROUP BY
        trigger_ccsr, trigger_ccsr_desc
        ,v2_ccsr, v2_ccsr_desc
        ,next_dx, next_dx_clean, next_dx_desc, next_ccsr, next_ccsr_desc
        ,member_segment
),
pair_totals AS (
    SELECT trigger_ccsr, v2_ccsr, member_segment, SUM(transition_count) AS pair_total
    FROM transition_counts GROUP BY trigger_ccsr, v2_ccsr, member_segment
)
SELECT
    t.trigger_ccsr                                       AS current_ccsr_v1
    ,t.trigger_ccsr_desc                                 AS current_ccsr_v1_desc
    ,t.v2_ccsr                                           AS current_ccsr_v2
    ,t.v2_ccsr_desc                                      AS current_ccsr_v2_desc
    ,t.next_dx, t.next_dx_desc, t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, p.pair_total
    ,ROUND(t.transition_count / p.pair_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / p.pair_total *
        LOG(t.transition_count / p.pair_total)) OVER (
            PARTITION BY t.trigger_ccsr, t.v2_ccsr, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN pair_totals p
    ON t.trigger_ccsr = p.trigger_ccsr
    AND t.v2_ccsr = p.v2_ccsr
    AND t.member_segment = p.member_segment
ORDER BY t.trigger_ccsr, t.v2_ccsr, t.transition_count DESC;


-- ============================================================
-- ORDER 2: CCSR -> CCSR
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_ccsr_order2`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_ccsr_to_ccsr_order2`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH v2_claims AS (
    SELECT DISTINCT
        t.member_id, t.trigger_date, t.trigger_ccsr, t.trigger_ccsr_desc
        ,t.member_segment, t.trigger_rank
        ,f.ccsr_category                                 AS v2_ccsr
        ,f.ccsr_category_description                     AS v2_ccsr_desc
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
v3_claims AS (
    SELECT DISTINCT
        v.member_id, v.trigger_date, v.trigger_ccsr, v.trigger_ccsr_desc
        ,v.member_segment, v.v2_ccsr, v.v2_ccsr_desc
        ,f.ccsr_category                                 AS next_ccsr
        ,f.ccsr_category_description                     AS next_ccsr_desc
    FROM v2_claims v
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v3
        ON v.member_id = v3.member_id
        AND v3.visit_rank = v.trigger_rank + 2
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON v.member_id = f.member_id
        AND v3.visit_date = f.visit_date
    WHERE f.ccsr_category IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_ccsr, trigger_ccsr_desc
        ,v2_ccsr, v2_ccsr_desc
        ,next_ccsr, next_ccsr_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM v3_claims
    GROUP BY
        trigger_ccsr, trigger_ccsr_desc
        ,v2_ccsr, v2_ccsr_desc
        ,next_ccsr, next_ccsr_desc, member_segment
),
pair_totals AS (
    SELECT trigger_ccsr, v2_ccsr, member_segment, SUM(transition_count) AS pair_total
    FROM transition_counts GROUP BY trigger_ccsr, v2_ccsr, member_segment
)
SELECT
    t.trigger_ccsr                                       AS current_ccsr_v1
    ,t.trigger_ccsr_desc                                 AS current_ccsr_v1_desc
    ,t.v2_ccsr                                           AS current_ccsr_v2
    ,t.v2_ccsr_desc                                      AS current_ccsr_v2_desc
    ,t.next_ccsr, t.next_ccsr_desc
    ,t.member_segment, t.transition_count, t.unique_members, p.pair_total
    ,ROUND(t.transition_count / p.pair_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / p.pair_total *
        LOG(t.transition_count / p.pair_total)) OVER (
            PARTITION BY t.trigger_ccsr, t.v2_ccsr, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN pair_totals p
    ON t.trigger_ccsr = p.trigger_ccsr
    AND t.v2_ccsr = p.v2_ccsr
    AND t.member_segment = p.member_segment
ORDER BY t.trigger_ccsr, t.v2_ccsr, t.transition_count DESC;
