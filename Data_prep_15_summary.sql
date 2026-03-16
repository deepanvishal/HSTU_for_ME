-- ============================================================
-- SPEND SUMMARY — TRIGGER VISIT AND FOLLOWING VISITS
-- ============================================================

-- STEP 1: Deduplicated claim level allowed_amt
-- One allowed_amt per claim_id — no double counting
WITH claims_deduped AS (
    SELECT DISTINCT
        claim_id
        ,member_id
        ,srv_start_dt
        ,srv_prvdr_id
        ,pri_icd9_dx_cd
        ,REPLACE(pri_icd9_dx_cd, '.', '')               AS dx_clean
        ,specialty_ctg_cd
        ,plc_srv_cd
        ,med_cost_ctg_cd
        ,allowed_amt
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY claim_id
        ORDER BY srv_start_dt
    ) = 1
),
-- STEP 2: Enrich with CCSR
claims_enriched AS (
    SELECT
        c.*
        ,ccsr.ccsr_category
        ,ccsr.ccsr_category_description
    FROM claims_deduped c
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mwb_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` ccsr
        ON c.dx_clean = ccsr.icd_10_cm_code
),
-- STEP 3: Tag trigger visits
trigger_visits AS (
    SELECT
        c.*
        ,'trigger'                                       AS visit_type
        ,t.trigger_date
    FROM claims_enriched c
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
        ON c.member_id = t.member_id
        AND c.dx_clean = t.trigger_dx_clean
        AND c.srv_start_dt = t.trigger_date
),
-- STEP 4: Tag V2 visits (immediate next visit after trigger)
v2_visits AS (
    SELECT
        c.*
        ,'v2'                                            AS visit_type
        ,t.trigger_date
    FROM claims_enriched c
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
        ON c.member_id = t.member_id
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
    WHERE c.srv_start_dt = v2.visit_date
),
-- STEP 5: Tag T180 downstream visits
downstream_visits AS (
    SELECT
        c.*
        ,'downstream'                                    AS visit_type
        ,t.trigger_date
    FROM claims_enriched c
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` t
        ON c.member_id = t.member_id
        AND c.srv_start_dt > t.trigger_date
        AND c.srv_start_dt <= DATE_ADD(t.trigger_date, INTERVAL 180 DAY)
),
-- STEP 6: Union all visit types
all_visits AS (
    SELECT * FROM trigger_visits
    UNION ALL
    SELECT * FROM v2_visits
    UNION ALL
    SELECT * FROM downstream_visits
)

-- ============================================================
-- SUMMARY 1 — BY SPECIALTY
-- ============================================================
SELECT
    'specialty'                                          AS summary_type
    ,specialty_ctg_cd                                    AS grouping_code
    ,sp.long_dscrptn                                     AS grouping_desc
    ,visit_type
    ,COUNT(DISTINCT claim_id)                            AS claim_count
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,ROUND(SUM(allowed_amt), 2)                          AS total_allowed_amt
    ,ROUND(AVG(allowed_amt), 2)                          AS avg_allowed_per_claim
    ,ROUND(SUM(allowed_amt) / COUNT(DISTINCT member_id), 2) AS avg_allowed_per_member
FROM all_visits
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp
    ON specialty_ctg_cd = sp.global_lookup_cd
    AND LOWER(sp.lookup_column_nm) = 'specialty_ctg_cd'
WHERE specialty_ctg_cd IS NOT NULL AND specialty_ctg_cd != ''
GROUP BY specialty_ctg_cd, sp.long_dscrptn, visit_type

UNION ALL

-- ============================================================
-- SUMMARY 2 — BY CCSR
-- ============================================================
SELECT
    'ccsr'                                               AS summary_type
    ,ccsr_category                                       AS grouping_code
    ,ccsr_category_description                           AS grouping_desc
    ,visit_type
    ,COUNT(DISTINCT claim_id)                            AS claim_count
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,ROUND(SUM(allowed_amt), 2)                          AS total_allowed_amt
    ,ROUND(AVG(allowed_amt), 2)                          AS avg_allowed_per_claim
    ,ROUND(SUM(allowed_amt) / COUNT(DISTINCT member_id), 2) AS avg_allowed_per_member
FROM all_visits
WHERE ccsr_category IS NOT NULL AND ccsr_category != ''
GROUP BY ccsr_category, ccsr_category_description, visit_type

UNION ALL

-- ============================================================
-- SUMMARY 3 — BY PLACE OF SERVICE
-- ============================================================
SELECT
    'place_of_service'                                   AS summary_type
    ,plc_srv_cd                                          AS grouping_code
    ,CAST(plc_srv_cd AS STRING)                          AS grouping_desc
    ,visit_type
    ,COUNT(DISTINCT claim_id)                            AS claim_count
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,ROUND(SUM(allowed_amt), 2)                          AS total_allowed_amt
    ,ROUND(AVG(allowed_amt), 2)                          AS avg_allowed_per_claim
    ,ROUND(SUM(allowed_amt) / COUNT(DISTINCT member_id), 2) AS avg_allowed_per_member
FROM all_visits
WHERE plc_srv_cd IS NOT NULL AND plc_srv_cd != ''
GROUP BY plc_srv_cd, visit_type

UNION ALL

-- ============================================================
-- SUMMARY 4 — BY MEDICAL COST CATEGORY
-- ============================================================
SELECT
    'med_cost_category'                                  AS summary_type
    ,med_cost_ctg_cd                                     AS grouping_code
    ,CAST(med_cost_ctg_cd AS STRING)                     AS grouping_desc
    ,visit_type
    ,COUNT(DISTINCT claim_id)                            AS claim_count
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,ROUND(SUM(allowed_amt), 2)                          AS total_allowed_amt
    ,ROUND(AVG(allowed_amt), 2)                          AS avg_allowed_per_claim
    ,ROUND(SUM(allowed_amt) / COUNT(DISTINCT member_id), 2) AS avg_allowed_per_member
FROM all_visits
WHERE med_cost_ctg_cd IS NOT NULL AND med_cost_ctg_cd != ''
GROUP BY med_cost_ctg_cd, visit_type

ORDER BY summary_type, visit_type, total_allowed_amt DESC
