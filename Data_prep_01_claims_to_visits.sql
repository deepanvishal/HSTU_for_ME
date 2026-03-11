-- ============================================================
-- VISIT FLAGS TABLE: visit number + first-DX flags + CCSR
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_flags`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_flags`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH enriched_claims AS (
    SELECT
        member_id
        ,srv_start_dt                                    AS visit_date
        ,specialty_ctg_cd
        ,age_nbr
        ,gender_cd
        ,pri_icd9_dx_cd                                  AS dx_raw
        ,REPLACE(TRIM(pri_icd9_dx_cd), '.', '')          AS dx_clean
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    WHERE pri_icd9_dx_cd IS NOT NULL
        AND TRIM(pri_icd9_dx_cd) != ''
        AND specialty_ctg_cd IS NOT NULL
        AND TRIM(specialty_ctg_cd) != ''
),
ccsr_joined AS (
    SELECT
        e.member_id
        ,e.visit_date
        ,e.specialty_ctg_cd
        ,e.age_nbr
        ,e.gender_cd
        ,e.dx_raw
        ,e.dx_clean
        ,c.ccsr_category
        ,c.ccsr_category_description
    FROM enriched_claims e
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mw_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` c
        ON e.dx_clean = c.icd_10_cm_code
),
first_dx_per_member AS (
    -- first date each member × DX combination appears
    SELECT
        member_id
        ,dx_raw
        ,MIN(visit_date)                                 AS first_dx_date
    FROM ccsr_joined
    GROUP BY member_id, dx_raw
),
visit_level AS (
    SELECT
        c.member_id
        ,c.visit_date
        ,c.age_nbr
        ,c.gender_cd
        ,c.specialty_ctg_cd
        ,c.dx_raw
        ,c.dx_clean
        ,c.ccsr_category
        ,c.ccsr_category_description
        ,CASE WHEN c.visit_date = f.first_dx_date THEN TRUE ELSE FALSE END AS is_first_dx_encounter
    FROM ccsr_joined c
    JOIN first_dx_per_member f
        ON c.member_id = f.member_id
        AND c.dx_raw = f.dx_raw
),
visit_ranked AS (
    SELECT
        member_id
        ,visit_date
        ,age_nbr
        ,gender_cd
        ,specialty_ctg_cd
        ,dx_raw
        ,dx_clean
        ,ccsr_category
        ,ccsr_category_description
        ,is_first_dx_encounter
        ,DENSE_RANK() OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS visit_number
    FROM visit_level
)
SELECT
    member_id
    ,visit_date
    ,visit_number
    ,CASE WHEN visit_number = 1 THEN TRUE ELSE FALSE END AS is_first_member_visit
    ,CASE
        WHEN age_nbr < 18                                THEN 'Children'
        WHEN age_nbr BETWEEN 18 AND 65
             AND gender_cd = 'M'                         THEN 'Adult_Male'
        WHEN age_nbr BETWEEN 18 AND 65
             AND gender_cd = 'F'                         THEN 'Adult_Female'
        WHEN age_nbr > 65                                THEN 'Senior'
     END                                                 AS member_segment
    ,ARRAY_AGG(DISTINCT specialty_ctg_cd IGNORE NULLS)   AS specialty_codes
    ,ARRAY_AGG(DISTINCT dx_raw IGNORE NULLS)             AS dx_list
    ,ARRAY_AGG(DISTINCT dx_clean IGNORE NULLS)           AS dx_list_clean
    ,ARRAY_AGG(DISTINCT ccsr_category IGNORE NULLS)      AS ccsr_list
    ,ARRAY_AGG(DISTINCT ccsr_category_description
        IGNORE NULLS)                                    AS ccsr_desc_list
    -- new DX codes only (first encounter)
    ,ARRAY_AGG(DISTINCT dx_raw
        IGNORE NULLS
        ORDER BY dx_raw)
        FILTER (WHERE is_first_dx_encounter = TRUE)      AS new_dx_list
    ,COUNTIF(is_first_dx_encounter = TRUE)               AS new_dx_count
    ,CASE WHEN COUNTIF(is_first_dx_encounter = TRUE) > 0
          THEN TRUE ELSE FALSE END                       AS has_new_dx
FROM visit_ranked
GROUP BY
    member_id
    ,visit_date
    ,visit_number
    ,member_segment
ORDER BY
    member_id
    ,visit_date
;


SELECT
    is_first_member_visit
    ,has_new_dx
    ,new_dx_count
    ,COUNT(*)                                            AS visit_count
    ,COUNT(DISTINCT member_id)                           AS unique_members
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_flags`
GROUP BY
    is_first_member_visit
    ,has_new_dx
    ,new_dx_count
ORDER BY
    is_first_member_visit DESC
    ,has_new_dx DESC
    ,new_dx_count DESC
;
