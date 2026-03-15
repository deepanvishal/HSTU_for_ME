-- ============================================================
-- SOUTH FLORIDA SCOPE STATISTICS
-- ============================================================
WITH claims_raw AS (
    SELECT
        member_id
        ,srv_prvdr_id
        ,srv_start_dt
        ,pri_icd9_dx_cd
        ,specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
),
visit_spine AS (
    SELECT DISTINCT member_id, srv_start_dt
    FROM claims_raw
),
triggers AS (
    SELECT DISTINCT member_id, trigger_date
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers`
),
trigger_followup AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,v.srv_start_dt                                  AS followup_date
        ,DATE_DIFF(v.srv_start_dt, t.trigger_date, DAY) AS days_after
    FROM triggers t
    JOIN visit_spine v
        ON t.member_id = v.member_id
        AND v.srv_start_dt > t.trigger_date
),
immediate_next AS (
    SELECT member_id, trigger_date
    FROM trigger_followup
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY member_id, trigger_date
        ORDER BY days_after
    ) = 1
)
SELECT
    (SELECT COUNT(*)                       FROM claims_raw)                      AS total_claims
    ,(SELECT COUNT(DISTINCT srv_prvdr_id)  FROM claims_raw)                      AS total_providers
    ,(SELECT COUNT(DISTINCT member_id)     FROM claims_raw)                      AS total_members
    ,(SELECT COUNT(DISTINCT CONCAT(member_id, '_', CAST(srv_start_dt AS STRING)))
      FROM visit_spine)                                                           AS total_visits
    ,(SELECT COUNT(DISTINCT CONCAT(member_id, '_', CAST(trigger_date AS STRING)))
      FROM triggers)                                                              AS trigger_visits_E
    ,(SELECT COUNT(DISTINCT CONCAT(member_id, '_', CAST(trigger_date AS STRING)))
      FROM trigger_followup)                                                      AS triggers_with_followup_F
    ,(SELECT COUNT(*)                      FROM immediate_next)                  AS immediate_next_visit
    ,(SELECT COUNT(DISTINCT CONCAT(member_id, '_', CAST(trigger_date AS STRING)))
      FROM trigger_followup WHERE days_after <= 30)                              AS triggers_with_followup_T30_G
    ,(SELECT COUNT(DISTINCT CONCAT(member_id, '_', CAST(trigger_date AS STRING)))
      FROM trigger_followup WHERE days_after <= 60)                              AS triggers_with_followup_T60_H
    ,(SELECT COUNT(DISTINCT CONCAT(member_id, '_', CAST(trigger_date AS STRING)))
      FROM trigger_followup WHERE days_after <= 180)                             AS triggers_with_followup_T180_I








-- ============================================================
-- DATA QUALITY COVERAGE
-- ============================================================
WITH claims_raw AS (
    SELECT
        member_id
        ,srv_start_dt
        ,pri_icd9_dx_cd
        ,specialty_ctg_cd
        ,REPLACE(pri_icd9_dx_cd, '.', '')               AS dx_clean
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
),
ccsr_mapped AS (
    SELECT DISTINCT icd_10_cm_code
    FROM `edp-prod-hcbstorage.edp_hcb_mwb_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101`
),
total AS (
    SELECT COUNT(*) AS n FROM claims_raw
)
SELECT
    t.n                                                                          AS total_claims
    ,COUNTIF(c.pri_icd9_dx_cd IS NOT NULL
        AND c.pri_icd9_dx_cd != '')                                              AS claims_with_icd10
    ,ROUND(COUNTIF(c.pri_icd9_dx_cd IS NOT NULL
        AND c.pri_icd9_dx_cd != '') / t.n * 100, 2)                             AS pct_with_icd10
    ,COUNTIF(c.specialty_ctg_cd IS NOT NULL
        AND c.specialty_ctg_cd != '')                                            AS claims_with_specialty
    ,ROUND(COUNTIF(c.specialty_ctg_cd IS NOT NULL
        AND c.specialty_ctg_cd != '') / t.n * 100, 2)                           AS pct_with_specialty
    ,COUNTIF(m.icd_10_cm_code IS NOT NULL)                                      AS claims_with_ccsr
    ,ROUND(COUNTIF(m.icd_10_cm_code IS NOT NULL) / t.n * 100, 2)               AS pct_with_ccsr
    ,(SELECT COUNT(DISTINCT member_id) FROM claims_raw
      WHERE member_id IN (
          SELECT member_id FROM claims_raw
          GROUP BY member_id
          HAVING COUNT(DISTINCT srv_start_dt) >= 2
      ))                                                                          AS members_with_2plus_visits
    ,ROUND((SELECT COUNT(DISTINCT member_id) FROM claims_raw
      WHERE member_id IN (
          SELECT member_id FROM claims_raw
          GROUP BY member_id
          HAVING COUNT(DISTINCT srv_start_dt) >= 2
      )) / (SELECT COUNT(DISTINCT member_id) FROM claims_raw) * 100, 2)         AS pct_members_with_2plus_visits
FROM claims_raw c
LEFT JOIN ccsr_mapped m
    ON c.dx_clean = m.icd_10_cm_code
CROSS JOIN total t
GROUP BY t.n
