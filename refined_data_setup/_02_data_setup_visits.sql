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
