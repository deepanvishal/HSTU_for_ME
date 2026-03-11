DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_order1`
OPTIONS (
    labels=[("owner", "deepan_thulasi_aetna_com")]
)
AS
WITH visit_spine AS (
    SELECT DISTINCT
        member_id
        ,visit_date
        ,DENSE_RANK() OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS visit_rank
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`
),
triggers AS (
    SELECT DISTINCT
        f.member_id
        ,f.visit_date                                    AS trigger_date
        ,f.dx_raw                                        AS trigger_dx
        ,f.dx_clean                                      AS trigger_dx_clean
        ,CASE
            WHEN f.age_nbr < 18                          THEN 'Children'
            WHEN f.age_nbr BETWEEN 18 AND 65
                 AND f.gender_cd = 'M'                   THEN 'Adult_Male'
            WHEN f.age_nbr BETWEEN 18 AND 65
                 AND f.gender_cd = 'F'                   THEN 'Adult_Female'
            WHEN f.age_nbr > 65                          THEN 'Senior'
         END                                             AS member_segment
        ,s.visit_rank                                    AS trigger_rank
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
    JOIN visit_spine s
        ON f.member_id = s.member_id
        AND f.visit_date = s.visit_date
    WHERE f.is_first_dx_encounter = TRUE
),
next_visits AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.trigger_dx_clean
        ,t.member_segment
        ,v2.visit_date                                   AS next_visit_date
    FROM triggers t
    JOIN visit_spine v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
),
next_claims AS (
    SELECT DISTINCT
        n.member_id
        ,n.trigger_date
        ,n.trigger_dx
        ,n.trigger_dx_clean
        ,n.member_segment
        ,f.specialty_ctg_cd                              AS next_specialty
    FROM next_visits n
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
        ON n.member_id = f.member_id
        AND n.next_visit_date = f.visit_date
    WHERE f.specialty_ctg_cd IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_dx
        ,trigger_dx_clean
        ,next_specialty
        ,member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM next_claims
    GROUP BY trigger_dx, trigger_dx_clean, next_specialty, member_segment
),
dx_totals AS (
    SELECT
        trigger_dx
        ,member_segment
        ,SUM(transition_count)                           AS dx_total
    FROM transition_counts
    GROUP BY trigger_dx, member_segment
)
SELECT
    t.trigger_dx                                         AS current_dx
    ,dx_desc.icd9_dx_dscrptn                             AS current_dx_desc
    ,trigger_ccsr.ccsr_category                          AS current_ccsr
    ,trigger_ccsr.ccsr_category_description              AS current_ccsr_desc
    ,t.next_specialty
    ,sp.long_dscrptn                                     AS next_specialty_desc
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,d.dx_total
    ,ROUND(t.transition_count / d.dx_total, 4)          AS conditional_probability
    ,ROUND(-SUM(t.transition_count / d.dx_total *
        LOG(t.transition_count / d.dx_total)) OVER (
            PARTITION BY t.trigger_dx, t.member_segment
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN dx_totals d
    ON t.trigger_dx = d.trigger_dx
    AND t.member_segment = d.member_segment
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
    ON t.trigger_dx = dx_desc.icd9_dx_cd
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mwb_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` trigger_ccsr
    ON t.trigger_dx_clean = trigger_ccsr.icd_10_cm_code
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
    ON t.next_specialty = sp.global_lookup_cd
    AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
ORDER BY
    t.trigger_dx
    ,t.transition_count DESC
;
