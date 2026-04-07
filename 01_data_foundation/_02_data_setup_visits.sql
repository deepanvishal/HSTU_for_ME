-- ============================================================
-- TABLE 3 — A870800_gen_rec_visits
-- All claims deduplicated to visit level
-- Enriched with CCSR + visit rank
-- One row per member + visit_date + specialty + dx
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH claims_deduped AS (
    SELECT
        member_id
        ,srv_start_dt                                    AS visit_date
        ,srv_prvdr_id
        ,specialty_ctg_cd
        ,pri_icd9_dx_cd                                  AS dx_raw
        ,REPLACE(TRIM(pri_icd9_dx_cd), '.', '')          AS dx_clean
        ,plc_srv_cd
        ,med_cost_ctg_cd
        ,age_nbr
        ,gender_cd
        ,SUM(allowed_amt)                                AS allowed_amt
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    WHERE pri_icd9_dx_cd IS NOT NULL
      AND TRIM(pri_icd9_dx_cd) != ''
      AND specialty_ctg_cd IS NOT NULL
      AND TRIM(specialty_ctg_cd) != ''
    GROUP BY
        member_id, srv_start_dt, srv_prvdr_id
        ,specialty_ctg_cd, pri_icd9_dx_cd
        ,REPLACE(TRIM(pri_icd9_dx_cd), '.', '')
        ,plc_srv_cd, med_cost_ctg_cd
        ,age_nbr, gender_cd
),
with_rank AS (
    SELECT
        c.*
        ,DENSE_RANK() OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS visit_rank
        ,CASE
            WHEN age_nbr < 18                            THEN 'Children'
            WHEN age_nbr BETWEEN 18 AND 65
                 AND gender_cd = 'F'                     THEN 'Adult_Female'
            WHEN age_nbr BETWEEN 18 AND 65
                 AND gender_cd = 'M'                     THEN 'Adult_Male'
            WHEN age_nbr > 65                            THEN 'Senior'
            ELSE 'Unknown'
         END                                             AS member_segment
    FROM claims_deduped c
)
SELECT
    r.member_id
    ,r.visit_date
    ,r.visit_rank
    ,r.srv_prvdr_id
    ,r.specialty_ctg_cd
    ,sp.long_dscrptn                                     AS specialty_desc
    ,r.dx_raw
    ,r.dx_clean
    ,c.ccsr_category
    ,c.ccsr_category_description
    ,r.plc_srv_cd
    ,r.med_cost_ctg_cd
    ,r.age_nbr
    ,r.gender_cd
    ,r.member_segment
    ,r.allowed_amt
FROM with_rank r
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mwb_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` c
    ON r.dx_clean = c.icd_10_cm_code
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
    ON r.specialty_ctg_cd = sp.global_lookup_cd
    AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
;

-- ============================================================
-- TABLE 4 — A870800_gen_rec_triggers_qualified
-- Purpose : Identifies first encounter of each diagnosis per member
--           and applies left and right boundary qualification rules
-- Source  : A870800_gen_rec_visits + A870800_gen_rec_member_qualified
-- Output  : One row per member + trigger_date + trigger_dx
--           with qualification flags per time window
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH first_encounters AS (
    SELECT
        member_id
        ,visit_date                                      AS trigger_date
        ,visit_rank                                      AS trigger_rank
        ,dx_raw                                          AS trigger_dx
        ,dx_clean                                        AS trigger_dx_clean
        ,ccsr_category                                   AS trigger_ccsr
        ,ccsr_category_description                       AS trigger_ccsr_desc
        ,specialty_ctg_cd                                AS trigger_specialty
        ,specialty_desc                                  AS trigger_specialty_desc
        ,age_nbr
        ,gender_cd
        ,member_segment
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY member_id, dx_clean
        ORDER BY visit_date
    ) = 1
),
dx_history AS (
    SELECT DISTINCT
        member_id
        ,dx_clean
        ,visit_date
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
),
boundary_checks AS (
    SELECT
        f.*
        ,m.enrollment_start
        ,m.enrollment_end
        ,m.enrolled_months
        ,m.enrollment_window_months

        -- LEFT BOUNDARY
        -- Rule 1: enrolled at least 12 months before trigger
        ,CASE
            WHEN DATE_DIFF(f.trigger_date, m.enrollment_start, DAY) >= 365
            THEN TRUE ELSE FALSE
         END                                             AS rule1_enrolled_12m

        -- Rule 2: DX not seen in 12 months before trigger
        ,CASE
            WHEN EXISTS (
                SELECT 1 FROM dx_history d
                WHERE d.member_id = f.member_id
                  AND d.dx_clean = f.trigger_dx_clean
                  AND d.visit_date >= DATE_SUB(f.trigger_date, INTERVAL 365 DAY)
                  AND d.visit_date < f.trigger_date
            ) THEN FALSE ELSE TRUE
         END                                             AS rule2_dx_not_seen_12m

        -- Informational flag
        ,CASE
            WHEN EXISTS (
                SELECT 1 FROM dx_history d
                WHERE d.member_id = f.member_id
                  AND d.visit_date >= DATE_SUB(f.trigger_date, INTERVAL 365 DAY)
                  AND d.visit_date < f.trigger_date
            ) THEN TRUE ELSE FALSE
         END                                             AS has_claims_12m_before

        -- RIGHT BOUNDARY
        ,CASE
            WHEN f.trigger_date <= DATE '2025-11-30'
             AND m.enrollment_end >= DATE_ADD(f.trigger_date, INTERVAL 30 DAY)
            THEN TRUE ELSE FALSE
         END                                             AS is_t30_right_qualified

        ,CASE
            WHEN f.trigger_date <= DATE '2025-10-31'
             AND m.enrollment_end >= DATE_ADD(f.trigger_date, INTERVAL 60 DAY)
            THEN TRUE ELSE FALSE
         END                                             AS is_t60_right_qualified

        ,CASE
            WHEN f.trigger_date <= DATE '2025-06-30'
             AND m.enrollment_end >= DATE_ADD(f.trigger_date, INTERVAL 180 DAY)
            THEN TRUE ELSE FALSE
         END                                             AS is_t180_right_qualified

    FROM first_encounters f
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified` m
        ON f.member_id = m.member_id
)
SELECT
    member_id
    ,trigger_date
    ,trigger_rank
    ,trigger_dx
    ,trigger_dx_clean
    ,trigger_ccsr
    ,trigger_ccsr_desc
    ,trigger_specialty
    ,trigger_specialty_desc
    ,age_nbr
    ,gender_cd
    ,member_segment
    ,enrollment_start
    ,enrollment_end
    ,enrolled_months
    ,enrollment_window_months
    ,rule1_enrolled_12m
    ,rule2_dx_not_seen_12m
    ,has_claims_12m_before

    -- FINAL QUALIFICATION FLAGS
    ,CASE
        WHEN rule1_enrolled_12m AND rule2_dx_not_seen_12m
        THEN TRUE ELSE FALSE
     END                                                 AS is_left_qualified

    ,CASE
        WHEN rule1_enrolled_12m AND rule2_dx_not_seen_12m
             AND is_t30_right_qualified
        THEN TRUE ELSE FALSE
     END                                                 AS is_t30_qualified

    ,CASE
        WHEN rule1_enrolled_12m AND rule2_dx_not_seen_12m
             AND is_t60_right_qualified
        THEN TRUE ELSE FALSE
     END                                                 AS is_t60_qualified

    ,CASE
        WHEN rule1_enrolled_12m AND rule2_dx_not_seen_12m
             AND is_t180_right_qualified
        THEN TRUE ELSE FALSE
     END                                                 AS is_t180_qualified

FROM boundary_checks
;
