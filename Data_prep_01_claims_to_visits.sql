DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH base AS (
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
with_flags AS (
    SELECT
        b.*
        ,DENSE_RANK() OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS visit_number
        ,CASE
            WHEN visit_date = MIN(visit_date) OVER (
                PARTITION BY member_id, dx_raw
            ) THEN TRUE ELSE FALSE
         END                                             AS is_first_dx_encounter
    FROM base b
)
SELECT
    f.member_id
    ,f.visit_date
    ,f.visit_number
    ,f.specialty_ctg_cd
    ,f.age_nbr
    ,f.gender_cd
    ,f.dx_raw
    ,f.dx_clean
    ,f.is_first_dx_encounter
    ,c.ccsr_category
    ,c.ccsr_category_description
FROM with_flags f
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mw_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` c
    ON f.dx_clean = c.icd_10_cm_code
ORDER BY
    f.member_id
    ,f.visit_date
