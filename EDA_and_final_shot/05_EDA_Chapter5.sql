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
WITH enrolled AS (
    SELECT COUNT(DISTINCT member_id) AS total_enrolled
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
)
SELECT
    'L1_Enrolled Members'                                  AS population
    ,COUNT(DISTINCT m.member_id)                           AS members
    ,CAST(100.0 AS FLOAT64)                                AS pct_of_enrolled
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified` m
CROSS JOIN enrolled e

UNION ALL

SELECT
    'L2_Claimants'                                         AS population
    ,COUNT(DISTINCT c.member_id)                           AS members
    ,ROUND(100.0 * COUNT(DISTINCT c.member_id) / e.total_enrolled, 2) AS pct_of_enrolled
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl` c
CROSS JOIN enrolled e

UNION ALL

SELECT
    'L3_Members with Triggers'                             AS population
    ,COUNT(DISTINCT t.member_id)                           AS members
    ,ROUND(100.0 * COUNT(DISTINCT t.member_id) / e.total_enrolled, 2) AS pct_of_enrolled
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
CROSS JOIN enrolled e

UNION ALL

SELECT
    'L4_Left Qualified Members'                            AS population
    ,COUNT(DISTINCT t.member_id)                           AS members
    ,ROUND(100.0 * COUNT(DISTINCT t.member_id) / e.total_enrolled, 2) AS pct_of_enrolled
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
CROSS JOIN enrolled e
WHERE t.is_left_qualified = TRUE

UNION ALL

SELECT
    'L5_T30 Qualified Members'                             AS population
    ,COUNT(DISTINCT t.member_id)                           AS members
    ,ROUND(100.0 * COUNT(DISTINCT t.member_id) / e.total_enrolled, 2) AS pct_of_enrolled
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
CROSS JOIN enrolled e
WHERE t.is_t30_qualified = TRUE

UNION ALL

SELECT
    'L6_T60 Qualified Members'                             AS population
    ,COUNT(DISTINCT t.member_id)                           AS members
    ,ROUND(100.0 * COUNT(DISTINCT t.member_id) / e.total_enrolled, 2) AS pct_of_enrolled
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
CROSS JOIN enrolled e
WHERE t.is_t60_qualified = TRUE

UNION ALL

SELECT
    'L7_T180 Qualified Members'                            AS population
    ,COUNT(DISTINCT t.member_id)                           AS members
    ,ROUND(100.0 * COUNT(DISTINCT t.member_id) / e.total_enrolled, 2) AS pct_of_enrolled
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
CROSS JOIN enrolled e
WHERE t.is_t180_qualified = TRUE

ORDER BY population
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
    'T180 Qualified Visits'                                AS dataset
    ,COUNT(*)                                              AS total_rows
    ,COUNT(DISTINCT member_id)                             AS members
    ,COUNT(DISTINCT CONCAT(member_id,'_',
        CAST(visit_date AS STRING)))                       AS distinct_visits
    ,ROUND(SUM(allowed_amt), 0)                            AS total_allowed_amt
    ,ROUND(AVG(allowed_amt), 2)                            AS avg_allowed_per_claim
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
    'T180 Qualified'                                       AS dataset
    ,MIN(visit_count)                                      AS min_visits
    ,MAX(visit_count)                                      AS max_visits
    ,ROUND(AVG(visit_count), 1)                            AS mean_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(25)]        AS p25
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(50)]        AS median_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(75)]        AS p75
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(90)]        AS p90
FROM (
    SELECT member_id, COUNT(DISTINCT visit_date) AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
    WHERE is_t180_qualified = TRUE
    GROUP BY member_id
)

ORDER BY dataset
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
        DATE_DIFF(srv_start_dt,
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
    'T180 Qualified'                                       AS dataset
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(50)]           AS median_days_between_visits
    ,ROUND(AVG(days_gap), 1)                               AS mean_days_between_visits
FROM (
    SELECT
        DATE_DIFF(visit_date,
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

ORDER BY dataset
;

-- ============================================================
-- SECTION 3: Demographic and clinical mix
-- Is the qualified population representative?
-- ============================================================

-- ------------------------------------------------------------
-- Q5. Member segment mix — Raw vs T180 Qualified
-- pct computed per dataset independently — each sums to 100%
-- ------------------------------------------------------------
SELECT
    'Raw'                                                  AS dataset
    ,member_segment                                        AS member_segment
    ,members                                               AS members
    ,ROUND(100.0 * members / SUM(members) OVER (), 2)      AS pct
FROM (
    SELECT
        member_segment
        ,COUNT(DISTINCT member_id)                         AS members
    FROM (
        SELECT DISTINCT member_id, member_segment
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    )
    GROUP BY member_segment
)

UNION ALL

SELECT
    'T180 Qualified'                                       AS dataset
    ,member_segment                                        AS member_segment
    ,members                                               AS members
    ,ROUND(100.0 * members / SUM(members) OVER (), 2)      AS pct
FROM (
    SELECT
        member_segment
        ,COUNT(DISTINCT member_id)                         AS members
    FROM (
        SELECT DISTINCT member_id, member_segment
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
        WHERE is_t180_qualified = TRUE
    )
    GROUP BY member_segment
)

ORDER BY dataset, pct DESC
;

-- ------------------------------------------------------------
-- Q6. Submarket mix — Raw vs T180 Qualified
-- ------------------------------------------------------------
SELECT
    'Raw'                                                  AS dataset
    ,submarket                                             AS submarket
    ,members                                               AS members
    ,ROUND(100.0 * members / SUM(members) OVER (), 2)      AS pct
FROM (
    SELECT
        submarket
        ,COUNT(DISTINCT member_id)                         AS members
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
    GROUP BY submarket
)

UNION ALL

SELECT
    'T180 Qualified'                                       AS dataset
    ,submarket                                             AS submarket
    ,members                                               AS members
    ,ROUND(100.0 * members / SUM(members) OVER (), 2)      AS pct
FROM (
    SELECT
        m.submarket                                        AS submarket
        ,COUNT(DISTINCT t.member_id)                       AS members
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified` m
        ON t.member_id = m.member_id
    WHERE t.is_t180_qualified = TRUE
    GROUP BY m.submarket
)

ORDER BY dataset, pct DESC
;

-- ------------------------------------------------------------
-- Q7. Specialty mix — Raw vs T180 Qualified
-- PCP vs Specialist vs Inpatient vs Other
-- ------------------------------------------------------------
SELECT
    'Raw'                                                  AS dataset
    ,visit_type                                            AS visit_type
    ,visits                                                AS visits
    ,ROUND(100.0 * visits / SUM(visits) OVER (), 2)        AS pct
FROM (
    SELECT
        CASE
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
        END                                                AS visit_type
        ,COUNT(*)                                          AS visits
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY visit_type
)

UNION ALL

SELECT
    'T180 Qualified'                                       AS dataset
    ,visit_type                                            AS visit_type
    ,visits                                                AS visits
    ,ROUND(100.0 * visits / SUM(visits) OVER (), 2)        AS pct
FROM (
    SELECT
        CASE
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
        END                                                AS visit_type
        ,COUNT(*)                                          AS visits
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
    WHERE is_t180_qualified = TRUE
    GROUP BY visit_type
)

ORDER BY dataset, pct DESC
;
