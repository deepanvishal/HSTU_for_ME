-- ============================================================
-- BASE TABLE: VISIT TABLE WITH CCSR ENRICHMENT
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_ccsr`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH enriched_claims AS (
    SELECT
        member_id
        ,srv_start_dt
        ,specialty_ctg_cd
        ,age_nbr
        ,gender_cd
        ,pri_icd9_dx_cd                                  AS diagnosis_code_raw
        ,REPLACE(TRIM(pri_icd9_dx_cd), '.', '')          AS diagnosis_code_clean
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    WHERE pri_icd9_dx_cd IS NOT NULL
        AND TRIM(pri_icd9_dx_cd) != ''
        AND specialty_ctg_cd IS NOT NULL
        AND TRIM(specialty_ctg_cd) != ''
),
ccsr_joined AS (
    SELECT
        e.member_id
        ,e.srv_start_dt
        ,e.specialty_ctg_cd
        ,e.age_nbr
        ,e.gender_cd
        ,e.diagnosis_code_raw
        ,e.diagnosis_code_clean
        ,c.ccsr_category
        ,c.ccsr_category_description
    FROM enriched_claims e
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mw_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` c
        ON e.diagnosis_code_clean = c.icd_10_cm_code
    WHERE c.ccsr_category IS NOT NULL
        AND TRIM(c.ccsr_category) != ''
)
SELECT
    member_id
    ,srv_start_dt                                        AS visit_date
    ,CASE
        WHEN age_nbr < 18                                THEN 'Children'
        WHEN age_nbr BETWEEN 18 AND 65
             AND gender_cd = 'M'                         THEN 'Adult_Male'
        WHEN age_nbr BETWEEN 18 AND 65
             AND gender_cd = 'F'                         THEN 'Adult_Female'
        WHEN age_nbr > 65                                THEN 'Senior'
     END                                                 AS member_segment
    ,ARRAY_AGG(DISTINCT specialty_ctg_cd
        IGNORE NULLS)                                    AS specialty_codes
    ,ARRAY_AGG(DISTINCT diagnosis_code_raw
        IGNORE NULLS)                                    AS dx_list_raw
    ,ARRAY_AGG(DISTINCT diagnosis_code_clean
        IGNORE NULLS)                                    AS dx_list_clean
    ,ARRAY_AGG(DISTINCT ccsr_category
        IGNORE NULLS)                                    AS ccsr_list
    ,ARRAY_AGG(DISTINCT ccsr_category_description
        IGNORE NULLS)                                    AS ccsr_desc_list
FROM ccsr_joined
GROUP BY
    member_id
    ,srv_start_dt
    ,member_segment
ORDER BY
    member_id
    ,srv_start_dt
