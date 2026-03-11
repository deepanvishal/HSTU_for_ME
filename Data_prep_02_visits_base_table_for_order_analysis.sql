DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_pairs`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_pairs`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH triggers AS (
    -- one row per member + visit_date + dx, first encounter only
    SELECT DISTINCT
        member_id
        ,visit_date
        ,visit_number
        ,dx_raw
        ,dx_clean
        ,CASE
            WHEN age_nbr < 18                            THEN 'Children'
            WHEN age_nbr BETWEEN 18 AND 65
                 AND gender_cd = 'M'                     THEN 'Adult_Male'
            WHEN age_nbr BETWEEN 18 AND 65
                 AND gender_cd = 'F'                     THEN 'Adult_Female'
            WHEN age_nbr > 65                            THEN 'Senior'
         END                                             AS member_segment
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`
    WHERE is_first_dx_encounter = TRUE
),
next_visit_dates AS (
    -- find next visit date per member using LEAD on distinct visit dates
    SELECT
        member_id
        ,visit_date
        ,LEAD(visit_date) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS next_visit_date
    FROM (
        SELECT DISTINCT member_id, visit_date
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`
    )
),
trigger_with_next AS (
    -- attach next visit date to each trigger
    SELECT
        t.member_id
        ,t.visit_date                                    AS trigger_date
        ,t.visit_number
        ,t.dx_raw                                        AS trigger_dx_raw
        ,t.dx_clean                                      AS trigger_dx_clean
        ,t.member_segment
        ,n.next_visit_date
    FROM triggers t
    JOIN next_visit_dates n
        ON t.member_id = n.member_id
        AND t.visit_date = n.visit_date
    WHERE n.next_visit_date IS NOT NULL
),
next_visit_claims AS (
    -- all claims on the next visit date, no filter
    SELECT DISTINCT
        member_id
        ,visit_date
        ,specialty_ctg_cd
        ,dx_raw                                          AS next_dx_raw
        ,dx_clean                                        AS next_dx_clean
        ,ccsr_category                                   AS next_ccsr
        ,ccsr_category_description                       AS next_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`
)
SELECT
    t.member_id
    ,t.trigger_date
    ,t.visit_number
    ,t.member_segment
    ,t.trigger_dx_raw
    ,t.trigger_dx_clean
    ,dx_desc.icd9_dx_description                         AS trigger_dx_desc
    ,trigger_ccsr.ccsr_category                          AS trigger_ccsr
    ,trigger_ccsr.ccsr_category_description              AS trigger_ccsr_desc
    ,n.visit_date                                        AS next_visit_date
    ,n.specialty_ctg_cd                                  AS next_specialty
    ,sp.global_lookup_desc                               AS next_specialty_desc
    ,n.next_dx_raw
    ,n.next_dx_clean
    ,next_dx_desc.icd9_dx_description                    AS next_dx_desc
    ,n.next_ccsr
    ,n.next_ccsr_desc
FROM trigger_with_next t
JOIN next_visit_claims n
    ON t.member_id = n.member_id
    AND t.next_visit_date = n.visit_date
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
    ON t.trigger_dx_raw = dx_desc.icd9_dx_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mw_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` trigger_ccsr
    ON t.trigger_dx_clean = trigger_ccsr.icd_10_cm_code
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` next_dx_desc
    ON n.next_dx_raw = next_dx_desc.icd9_dx_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
    ON n.specialty_ctg_cd = sp.global_lookup_cd
    AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
ORDER BY
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx_raw
;



-- ============================================================
-- INTERIM 1: VISIT SPINE
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
SELECT DISTINCT
    member_id
    ,visit_date
    ,DENSE_RANK() OVER (
        PARTITION BY member_id
        ORDER BY visit_date
    )                                                    AS visit_rank
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`
ORDER BY member_id, visit_date;


-- ============================================================
-- INTERIM 2: TRIGGERS
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
SELECT DISTINCT
    f.member_id
    ,f.visit_date                                        AS trigger_date
    ,f.dx_raw                                            AS trigger_dx
    ,f.dx_clean                                          AS trigger_dx_clean
    ,dx_desc.icd9_dx_dscrptn                             AS trigger_dx_desc
    ,f.ccsr_category                                     AS trigger_ccsr
    ,f.ccsr_category_description                         AS trigger_ccsr_desc
    ,f.specialty_ctg_cd                                  AS trigger_specialty
    ,sp.long_dscrptn                                     AS trigger_specialty_desc
    ,CASE
        WHEN f.age_nbr < 18                              THEN 'Children'
        WHEN f.age_nbr BETWEEN 18 AND 65
             AND f.gender_cd = 'M'                       THEN 'Adult_Male'
        WHEN f.age_nbr BETWEEN 18 AND 65
             AND f.gender_cd = 'F'                       THEN 'Adult_Female'
        WHEN f.age_nbr > 65                              THEN 'Senior'
     END                                                 AS member_segment
    ,s.visit_rank                                        AS trigger_rank
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` s
    ON f.member_id = s.member_id
    AND f.visit_date = s.visit_date
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
    ON f.dx_raw = dx_desc.icd9_dx_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mwb_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` ccsr
    ON f.dx_clean = ccsr.icd_10_cm_code
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
    ON f.specialty_ctg_cd = sp.global_lookup_cd
    AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
WHERE f.is_first_dx_encounter = TRUE
ORDER BY
    f.member_id
    ,f.visit_date
    ,f.dx_raw;
