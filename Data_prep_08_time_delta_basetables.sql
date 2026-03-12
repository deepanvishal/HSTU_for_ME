DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_base`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_base`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH downstream_visits AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.trigger_dx_desc
        ,t.trigger_ccsr
        ,t.trigger_ccsr_desc
        ,t.trigger_specialty
        ,t.trigger_specialty_desc
        ,t.member_segment
        ,f.visit_date
        ,DATE_DIFF(f.visit_date, t.trigger_date, DAY)   AS days_since_trigger
        ,f.specialty_ctg_cd
        ,sp.long_dscrptn                                 AS specialty_desc
        ,f.dx_raw
        ,f.dx_clean
        ,dx_desc.icd9_dx_dscrptn                         AS dx_desc
        ,f.ccsr_category
        ,f.ccsr_category_description
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON t.member_id = f.member_id
        AND f.visit_date > t.trigger_date
        AND f.visit_date <= DATE_ADD(t.trigger_date, INTERVAL 180 DAY)
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
        ON f.specialty_ctg_cd = sp.global_lookup_cd
        AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
        ON f.dx_raw = dx_desc.icd9_dx_cd
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY t.member_id, t.trigger_date, t.trigger_dx, f.visit_date, f.dx_raw, f.specialty_ctg_cd
        ORDER BY f.visit_date
    ) = 1
)
SELECT
    member_id
    ,trigger_date
    ,trigger_dx
    ,trigger_dx_desc
    ,trigger_ccsr
    ,trigger_ccsr_desc
    ,trigger_specialty
    ,trigger_specialty_desc
    ,member_segment
    ,ARRAY_AGG(
        STRUCT(
            visit_date
            ,days_since_trigger
            ,specialty_ctg_cd                            AS specialty
            ,specialty_desc
            ,dx_raw                                      AS dx
            ,dx_clean
            ,dx_desc
            ,ccsr_category                               AS ccsr
            ,ccsr_category_description                   AS ccsr_desc
        )
        ORDER BY visit_date
    )                                                    AS downstream_visits
FROM downstream_visits
GROUP BY
    member_id, trigger_date, trigger_dx, trigger_dx_desc
    ,trigger_ccsr, trigger_ccsr_desc
    ,trigger_specialty, trigger_specialty_desc
    ,member_segment
