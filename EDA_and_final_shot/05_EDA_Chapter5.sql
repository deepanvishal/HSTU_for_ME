-- ============================================================
-- CHAPTER 5: Raw vs Qualified — Side by Side
-- Story: Did our qualification rules introduce bias?
-- Sources: A870800_claims_gen_rec_2022_2025_sfl (raw)
--          A870800_gen_rec_visits (all visits)
--          A870800_gen_rec_triggers_qualified (qualified)
--          A870800_gen_rec_visits_qualified (qualified visits)
--          A870800_gen_rec_member_qualified (membership)
-- ============================================================

-- ============================================================
-- SECTION 1: Member Funnel
-- Enrolled → Claimants → Trigger Members → T180 Qualified
-- ============================================================

-- ------------------------------------------------------------
-- Q1. Full member funnel — one number per step
-- ------------------------------------------------------------
SELECT
    'Enrolled Members'                                     AS population
    ,COUNT(DISTINCT member_id)                             AS members
    ,NULL                                                  AS pct_of_enrolled
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`

UNION ALL

SELECT
    'Claimants'
    ,COUNT(DISTINCT member_id)
    ,ROUND(100.0 * COUNT(DISTINCT member_id) / (
        SELECT COUNT(DISTINCT member_id)
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
    ), 2)
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`

UNION ALL

SELECT
    'Members with Triggers'
    ,COUNT(DISTINCT member_id)
    ,ROUND(100.0 * COUNT(DISTINCT member_id) / (
        SELECT COUNT(DISTINCT member_id)
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
    ), 2)
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`

UNION ALL

SELECT
    'Left Qualified Members'
    ,COUNT(DISTINCT member_id)
    ,ROUND(100.0 * COUNT(DISTINCT member_id) / (
        SELECT COUNT(DISTINCT member_id)
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
    ), 2)
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
WHERE is_left_qualified = TRUE

UNION ALL

SELECT
    'T30 Qualified Members'
    ,COUNT(DISTINCT member_id)
    ,ROUND(100.0 * COUNT(DISTINCT member_id) / (
        SELECT COUNT(DISTINCT member_id)
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
    ), 2)
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
WHERE is_t30_qualified = TRUE

UNION ALL

SELECT
    'T60 Qualified Members'
    ,COUNT(DISTINCT member_id)
    ,ROUND(100.0 * COUNT(DISTINCT member_id) / (
        SELECT COUNT(DISTINCT member_id)
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
    ), 2)
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
WHERE is_t60_qualified = TRUE

UNION ALL

SELECT
    'T180 Qualified Members'
    ,COUNT(DISTINCT member_id)
    ,ROUND(100.0 * COUNT(DISTINCT member_id) / (
        SELECT COUNT(DISTINCT member_id)
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
    ), 2)
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
WHERE is_t180_qualified = TRUE
;

-- ============================================================
-- SECTION 2: Key volume and distribution metrics
-- Raw claims vs T180 qualified visits — side by side
-- ============================================================

-- ------------------------------------------------------------
-- Q2. Volume comparison — claims, visits, members, allowed amt
-- ------------------------------------------------------------
SELECT
    'Raw Claims'                                           AS dataset
    ,COUNT(*)                                              AS total_rows
    ,COUNT(DISTINCT member_id)                             AS members
    ,COUNT(DISTINCT CONCAT(member_id,'_',
        CAST(srv_start_dt AS STRING)))                     AS distinct_visits
    ,ROUND(SUM(allowed_amt), 0)                            AS total_allowed_amt
    ,ROUND(AVG(allowed_amt), 2)                            AS avg_allowed_per_claim
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`

UNION ALL

SELECT
    'T180 Qualified Visits'
    ,COUNT(*)
    ,COUNT(DISTINCT member_id)
    ,COUNT(DISTINCT CONCAT(member_id,'_',
        CAST(visit_date AS STRING)))
    ,ROUND(SUM(allowed_amt), 0)
    ,ROUND(AVG(allowed_amt), 2)
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
WHERE is_t180_qualified = TRUE
;

-- ------------------------------------------------------------
-- Q3. Visits per member distribution — Raw vs T180 Qualified
-- ------------------------------------------------------------
SELECT
    'Raw'                                                  AS dataset
    ,MIN(visit_count)                                      AS min_visits
    ,MAX(visit_count)                                      AS max_visits
    ,ROUND(AVG(visit_count), 1)                            AS mean_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(25)]        AS p25
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(50)]        AS median_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(75)]        AS p75
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(90)]        AS p90
FROM (
    SELECT member_id, COUNT(DISTINCT srv_start_dt) AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    GROUP BY member_id
)

UNION ALL

SELECT
    'T180 Qualified'
    ,MIN(visit_count)
    ,MAX(visit_count)
    ,ROUND(AVG(visit_count), 1)
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(25)]
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(50)]
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(75)]
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(90)]
FROM (
    SELECT member_id, COUNT(DISTINCT visit_date) AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
    WHERE is_t180_qualified = TRUE
    GROUP BY member_id
)
;

-- ------------------------------------------------------------
-- Q4. Median days between visits — Raw vs T180 Qualified
-- ------------------------------------------------------------
SELECT
    'Raw'                                                  AS dataset
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(50)]           AS median_days_between_visits
    ,ROUND(AVG(days_gap), 1)                               AS mean_days_between_visits
FROM (
    SELECT
        member_id
        ,DATE_DIFF(srv_start_dt,
            LAG(srv_start_dt) OVER (
                PARTITION BY member_id ORDER BY srv_start_dt
            ), DAY)                                        AS days_gap
    FROM (
        SELECT DISTINCT member_id, srv_start_dt
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    )
)
WHERE days_gap IS NOT NULL

UNION ALL

SELECT
    'T180 Qualified'
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(50)]
    ,ROUND(AVG(days_gap), 1)
FROM (
    SELECT
        member_id
        ,DATE_DIFF(visit_date,
            LAG(visit_date) OVER (
                PARTITION BY member_id ORDER BY visit_date
            ), DAY)                                        AS days_gap
    FROM (
        SELECT DISTINCT member_id, visit_date
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
        WHERE is_t180_qualified = TRUE
    )
)
WHERE days_gap IS NOT NULL
;

-- ============================================================
-- SECTION 3: Demographic and clinical mix
-- Is the qualified population representative?
-- ============================================================

-- ------------------------------------------------------------
-- Q5. Member segment mix — Raw vs T180 Qualified
-- ------------------------------------------------------------
SELECT
    'Raw'                                                  AS dataset
    ,member_segment
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct
FROM (
    SELECT DISTINCT member_id, member_segment
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
)
GROUP BY dataset, member_segment

UNION ALL

SELECT
    'T180 Qualified'
    ,member_segment
    ,COUNT(DISTINCT member_id)
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)
FROM (
    SELECT DISTINCT member_id, member_segment
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
    WHERE is_t180_qualified = TRUE
)
GROUP BY dataset, member_segment
ORDER BY dataset, pct DESC
;

-- ------------------------------------------------------------
-- Q6. Submarket mix — Raw vs T180 Qualified
-- ------------------------------------------------------------
SELECT
    'Raw'                                                  AS dataset
    ,submarket
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
GROUP BY dataset, submarket

UNION ALL

SELECT
    'T180 Qualified'
    ,m.submarket
    ,COUNT(DISTINCT t.member_id)
    ,ROUND(100.0 * COUNT(DISTINCT t.member_id) /
        SUM(COUNT(DISTINCT t.member_id)) OVER (), 2)
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified` m
    ON t.member_id = m.member_id
WHERE t.is_t180_qualified = TRUE
GROUP BY dataset, m.submarket
ORDER BY dataset, pct DESC
;

-- ------------------------------------------------------------
-- Q7. Specialty mix — Raw vs T180 Qualified
-- PCP vs Specialist vs Inpatient vs Other
-- ------------------------------------------------------------
SELECT
    'Raw'                                                  AS dataset
    ,CASE
        WHEN specialty_ctg_cd IN ('FP','I')
            THEN 'PCP'
        WHEN specialty_ctg_cd IN (
            'CARD','ENDO','NEPH','NEUR','ONCO',
            'OPTH','ORTH','PEDS','PSYC','PULM',
            'RHEU','UROL','DERM','GAST','HEMA',
            'INFD','OBGY','OTOL','SURG','VASC'
        )   THEN 'Specialist'
        WHEN plc_srv_cd = 'I'
            THEN 'Inpatient'
        ELSE    'Other'
    END                                                    AS visit_type
    ,COUNT(*)                                              AS visits
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
GROUP BY dataset, visit_type

UNION ALL

SELECT
    'T180 Qualified'
    ,CASE
        WHEN specialty_ctg_cd IN ('FP','I')
            THEN 'PCP'
        WHEN specialty_ctg_cd IN (
            'CARD','ENDO','NEPH','NEUR','ONCO',
            'OPTH','ORTH','PEDS','PSYC','PULM',
            'RHEU','UROL','DERM','GAST','HEMA',
            'INFD','OBGY','OTOL','SURG','VASC'
        )   THEN 'Specialist'
        WHEN plc_srv_cd = 'I'
            THEN 'Inpatient'
        ELSE    'Other'
    END
    ,COUNT(*)
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
WHERE is_t180_qualified = TRUE
GROUP BY dataset, visit_type
ORDER BY dataset, pct DESC
;
