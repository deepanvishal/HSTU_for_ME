-- ============================================================
-- TABLE 7 — A870800_gen_rec_f_spend_summary
-- Purpose : Allowed amount aggregated by specialty, CCSR,
--           place of service, and medical cost category
--           for trigger visits and downstream visits
-- Source  : A870800_gen_rec_visits_qualified
--           + A870800_gen_rec_triggers_qualified (for trigger visit spend)
-- Output  : One row per grouping + visit_type + time_bucket
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_spend_summary`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_spend_summary`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH trigger_spend AS (
    -- Spend at the trigger visit itself
    SELECT
        v.member_id
        ,v.specialty_ctg_cd
        ,v.specialty_desc
        ,v.ccsr_category
        ,v.ccsr_category_description
        ,v.plc_srv_cd
        ,v.med_cost_ctg_cd
        ,v.member_segment
        ,v.allowed_amt
        ,'trigger'                                       AS visit_type
        ,NULL                                            AS time_bucket
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND t.trigger_date = v.visit_date
        AND t.trigger_dx = v.dx_raw
    WHERE t.is_left_qualified = TRUE
),
downstream_spend AS (
    -- Spend at downstream visits after trigger
    SELECT
        v.member_id
        ,v.specialty_ctg_cd
        ,v.specialty_desc
        ,v.ccsr_category
        ,v.ccsr_category_description
        ,v.plc_srv_cd
        ,v.med_cost_ctg_cd
        ,v.member_segment
        ,v.allowed_amt
        ,CASE WHEN v.is_v2 = TRUE THEN 'v2' ELSE 'downstream' END AS visit_type
        ,CASE
            WHEN v.days_since_trigger <= 30              THEN 'T0_30'
            WHEN v.days_since_trigger <= 60              THEN 'T30_60'
            WHEN v.days_since_trigger <= 180             THEN 'T60_180'
         END                                             AS time_bucket
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    WHERE v.is_left_qualified = TRUE
),
all_visits AS (
    SELECT * FROM trigger_spend
    UNION ALL
    SELECT * FROM downstream_spend
)
-- SUMMARY 1 — BY SPECIALTY
SELECT
    'specialty'                                          AS summary_type
    ,specialty_ctg_cd                                    AS grouping_code
    ,specialty_desc                                      AS grouping_desc
    ,member_segment
    ,visit_type
    ,time_bucket
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,COUNT(*)                                            AS visit_count
    ,ROUND(SUM(allowed_amt), 2)                          AS total_allowed_amt
    ,ROUND(AVG(allowed_amt), 2)                          AS avg_allowed_per_visit
    ,ROUND(SUM(allowed_amt) / COUNT(DISTINCT member_id), 2) AS avg_allowed_per_member
FROM all_visits
WHERE specialty_ctg_cd IS NOT NULL AND specialty_ctg_cd != ''
GROUP BY specialty_ctg_cd, specialty_desc, member_segment, visit_type, time_bucket

UNION ALL

-- SUMMARY 2 — BY CCSR
SELECT
    'ccsr'                                               AS summary_type
    ,ccsr_category                                       AS grouping_code
    ,ccsr_category_description                           AS grouping_desc
    ,member_segment
    ,visit_type
    ,time_bucket
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,COUNT(*)                                            AS visit_count
    ,ROUND(SUM(allowed_amt), 2)                          AS total_allowed_amt
    ,ROUND(AVG(allowed_amt), 2)                          AS avg_allowed_per_visit
    ,ROUND(SUM(allowed_amt) / COUNT(DISTINCT member_id), 2) AS avg_allowed_per_member
FROM all_visits
WHERE ccsr_category IS NOT NULL AND ccsr_category != ''
GROUP BY ccsr_category, ccsr_category_description, member_segment, visit_type, time_bucket

UNION ALL

-- SUMMARY 3 — BY PLACE OF SERVICE
SELECT
    'place_of_service'                                   AS summary_type
    ,CAST(plc_srv_cd AS STRING)                          AS grouping_code
    ,CAST(plc_srv_cd AS STRING)                          AS grouping_desc
    ,member_segment
    ,visit_type
    ,time_bucket
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,COUNT(*)                                            AS visit_count
    ,ROUND(SUM(allowed_amt), 2)                          AS total_allowed_amt
    ,ROUND(AVG(allowed_amt), 2)                          AS avg_allowed_per_visit
    ,ROUND(SUM(allowed_amt) / COUNT(DISTINCT member_id), 2) AS avg_allowed_per_member
FROM all_visits
WHERE plc_srv_cd IS NOT NULL AND CAST(plc_srv_cd AS STRING) != ''
GROUP BY plc_srv_cd, member_segment, visit_type, time_bucket

UNION ALL

-- SUMMARY 4 — BY MEDICAL COST CATEGORY
SELECT
    'med_cost_category'                                  AS summary_type
    ,CAST(med_cost_ctg_cd AS STRING)                     AS grouping_code
    ,CAST(med_cost_ctg_cd AS STRING)                     AS grouping_desc
    ,member_segment
    ,visit_type
    ,time_bucket
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,COUNT(*)                                            AS visit_count
    ,ROUND(SUM(allowed_amt), 2)                          AS total_allowed_amt
    ,ROUND(AVG(allowed_amt), 2)                          AS avg_allowed_per_visit
    ,ROUND(SUM(allowed_amt) / COUNT(DISTINCT member_id), 2) AS avg_allowed_per_member
FROM all_visits
WHERE med_cost_ctg_cd IS NOT NULL AND CAST(med_cost_ctg_cd AS STRING) != ''
GROUP BY med_cost_ctg_cd, member_segment, visit_type, time_bucket

ORDER BY summary_type, visit_type, time_bucket, total_allowed_amt DESC
