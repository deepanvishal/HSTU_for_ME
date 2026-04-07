-- ============================================================
-- CHAPTER 3: Visit Facts
-- Source: A870800_gen_rec_visits
-- Unit: member + visit_date (already deduplicated)
-- ============================================================

-- ============================================================
-- ACT 1: What does a member's visit history look like?
-- ============================================================

-- ------------------------------------------------------------
-- Q1. Total distinct visits
-- ------------------------------------------------------------
SELECT
    COUNT(*)                                               AS total_visit_rows
    ,COUNT(DISTINCT CONCAT(member_id, '_', CAST(visit_date AS STRING)))
                                                           AS total_distinct_visits
    ,COUNT(DISTINCT member_id)                             AS total_members_with_visits
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
;

-- ------------------------------------------------------------
-- Q2. Visits per member distribution
-- ------------------------------------------------------------
SELECT
    MIN(visit_count)                                       AS min_visits
    ,MAX(visit_count)                                      AS max_visits
    ,ROUND(AVG(visit_count), 1)                            AS mean_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(10)]        AS p10
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(25)]        AS p25
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(50)]        AS median
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(75)]        AS p75
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(90)]        AS p90
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(99)]        AS p99
FROM (
    SELECT
        member_id
        ,COUNT(DISTINCT visit_date)                        AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY member_id
)
;

-- ------------------------------------------------------------
-- Q3. Visit count buckets — full distribution
-- Shows where members cluster
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN visit_count = 1        THEN '01_exactly_1'
        WHEN visit_count = 2        THEN '02_exactly_2'
        WHEN visit_count BETWEEN 3 AND 5   THEN '03_3_to_5'
        WHEN visit_count BETWEEN 6 AND 10  THEN '04_6_to_10'
        WHEN visit_count BETWEEN 11 AND 20 THEN '05_11_to_20'
        WHEN visit_count > 20       THEN '06_20_plus'
    END                                                    AS visit_bucket
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct
FROM (
    SELECT
        member_id
        ,COUNT(DISTINCT visit_date)                        AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY member_id
)
GROUP BY visit_bucket
ORDER BY visit_bucket
;

-- ------------------------------------------------------------
-- Q3b. Place of service distribution by visit count bucket
-- For each visit bucket — what % of visits are O, I, etc.
-- ------------------------------------------------------------
WITH member_visit_counts AS (
    SELECT member_id, COUNT(DISTINCT visit_date) AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY member_id
),
visits_with_bucket AS (
    SELECT
        v.plc_srv_cd
        ,CASE
            WHEN vc.visit_count = 1                THEN '01_exactly_1'
            WHEN vc.visit_count BETWEEN 2 AND 3    THEN '02_2_to_3'
            WHEN vc.visit_count BETWEEN 4 AND 5    THEN '03_4_to_5'
            WHEN vc.visit_count BETWEEN 6 AND 10   THEN '04_6_to_10'
            WHEN vc.visit_count BETWEEN 11 AND 20  THEN '05_11_to_20'
            WHEN vc.visit_count > 20               THEN '06_20_plus'
        END                                                AS visit_bucket
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    JOIN member_visit_counts vc ON v.member_id = vc.member_id
),
aggregated AS (
    SELECT
        visit_bucket
        ,plc_srv_cd
        ,COUNT(*)                                          AS visit_rows
        ,SUM(COUNT(*)) OVER (PARTITION BY visit_bucket)   AS total_in_bucket
    FROM visits_with_bucket
    GROUP BY visit_bucket, plc_srv_cd
)
SELECT
    visit_bucket
    ,plc_srv_cd
    ,visit_rows
    ,total_in_bucket
    ,ROUND(100.0 * visit_rows / total_in_bucket, 2)        AS pct_within_bucket
FROM aggregated
ORDER BY visit_bucket, pct_within_bucket DESC
;
-- The sequence modeling threshold question
-- ------------------------------------------------------------
SELECT
    COUNTIF(visit_count = 1)                               AS members_1_visit
    ,ROUND(100.0 * COUNTIF(visit_count = 1)
        / COUNT(*), 2)                                     AS pct_1_visit
    ,COUNTIF(visit_count >= 6)                             AS members_6plus_visits
    ,ROUND(100.0 * COUNTIF(visit_count >= 6)
        / COUNT(*), 2)                                     AS pct_6plus_visits
    ,COUNT(*)                                              AS total_members
FROM (
    SELECT
        member_id
        ,COUNT(DISTINCT visit_date)                        AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY member_id
)
;

-- ============================================================
-- ACT 2: How far apart are visits?
-- ============================================================

-- ------------------------------------------------------------
-- Q5. Inter-visit gap distribution — consecutive visit pairs
-- ------------------------------------------------------------
SELECT
    MIN(days_gap)                                          AS min_gap
    ,MAX(days_gap)                                         AS max_gap
    ,ROUND(AVG(days_gap), 1)                               AS mean_gap
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(10)]           AS p10
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(25)]           AS p25
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(50)]           AS median_gap
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(75)]           AS p75
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(90)]           AS p90
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(99)]           AS p99
FROM (
    SELECT
        member_id
        ,visit_date
        ,LAG(visit_date) OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                  AS prev_visit_date
        ,DATE_DIFF(
            visit_date,
            LAG(visit_date) OVER (
                PARTITION BY member_id ORDER BY visit_date
            ),
            DAY
        )                                                  AS days_gap
    FROM (
        SELECT DISTINCT member_id, visit_date
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    )
)
WHERE days_gap IS NOT NULL
;

-- ------------------------------------------------------------
-- Q6. Gap buckets — distribution of inter-visit gaps
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN days_gap = 1               THEN '01_1_day'
        WHEN days_gap BETWEEN 2 AND 7   THEN '02_2_to_7d'
        WHEN days_gap BETWEEN 8 AND 30  THEN '03_8_to_30d'
        WHEN days_gap BETWEEN 31 AND 90 THEN '04_31_to_90d'
        WHEN days_gap BETWEEN 91 AND 180 THEN '05_91_to_180d'
        WHEN days_gap > 180             THEN '06_180d_plus'
    END                                                    AS gap_bucket
    ,COUNT(*)                                              AS visit_pairs
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct
FROM (
    SELECT
        member_id
        ,DATE_DIFF(
            visit_date,
            LAG(visit_date) OVER (
                PARTITION BY member_id ORDER BY visit_date
            ),
            DAY
        )                                                  AS days_gap
    FROM (
        SELECT DISTINCT member_id, visit_date
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    )
)
WHERE days_gap IS NOT NULL
GROUP BY gap_bucket
ORDER BY gap_bucket
;

-- ============================================================
-- ACT 3: What is causing 1-day gaps?
-- ============================================================

-- ------------------------------------------------------------
-- Q7. % of visit pairs with 1-day gap
-- And among those — what % involve inpatient (plc_srv_cd = 'I')
-- Fix: deduplicate to one row per member per date first
--      multiple rows per day exist (diff specialty/dx) — LAG on raw creates 0-day gaps
--      MAX(plc_srv_cd) picks dominant place of service per day
-- ------------------------------------------------------------
WITH deduped_visits AS (
    SELECT
        member_id
        ,visit_date
        ,MAX(plc_srv_cd)                                   AS plc_srv_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY member_id, visit_date
),
visit_pairs AS (
    SELECT
        member_id
        ,visit_date                                        AS current_visit
        ,plc_srv_cd                                        AS current_plc_srv_cd
        ,LAG(visit_date) OVER (
            PARTITION BY member_id ORDER BY visit_date
        )                                                  AS prev_visit
        ,LAG(plc_srv_cd) OVER (
            PARTITION BY member_id ORDER BY visit_date
        )                                                  AS prev_plc_srv_cd
        ,DATE_DIFF(
            visit_date,
            LAG(visit_date) OVER (
                PARTITION BY member_id ORDER BY visit_date
            ),
            DAY
        )                                                  AS days_gap
    FROM deduped_visits
)
SELECT
    COUNT(*)                                               AS total_pairs
    ,COUNTIF(days_gap = 1)                                 AS pairs_1day_gap
    ,ROUND(100.0 * COUNTIF(days_gap = 1)
        / COUNT(*), 2)                                     AS pct_1day_gap
    ,COUNTIF(days_gap = 1
        AND (current_plc_srv_cd = 'I'
             OR prev_plc_srv_cd = 'I'))                    AS pairs_1day_inpatient
    ,ROUND(100.0 * COUNTIF(days_gap = 1
        AND (current_plc_srv_cd = 'I'
             OR prev_plc_srv_cd = 'I'))
        / NULLIF(COUNTIF(days_gap = 1), 0), 2)            AS pct_1day_gap_inpatient
FROM visit_pairs
WHERE days_gap IS NOT NULL
;

-- ------------------------------------------------------------
-- Q7b. Place of service transition pairs
-- For every consecutive visit pair — what is prev → current combo?
-- Key question: what % are I→I (same inpatient stay noise)
-- Fix: deduplicate to one row per member per date before LAG
-- ------------------------------------------------------------
WITH deduped_visits AS (
    SELECT
        member_id
        ,visit_date
        ,MAX(plc_srv_cd)                                   AS plc_srv_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY member_id, visit_date
),
visit_pairs AS (
    SELECT
        member_id
        ,plc_srv_cd                                        AS current_plc
        ,LAG(plc_srv_cd) OVER (
            PARTITION BY member_id ORDER BY visit_date
        )                                                  AS prev_plc
        ,DATE_DIFF(
            visit_date,
            LAG(visit_date) OVER (
                PARTITION BY member_id ORDER BY visit_date
            ),
            DAY
        )                                                  AS days_gap
    FROM deduped_visits
)
SELECT
    CONCAT(
        COALESCE(prev_plc, 'NULL')
        ,' → '
        ,COALESCE(current_plc, 'NULL')
    )                                                      AS plc_transition
    ,CASE WHEN days_gap = 1 THEN '1_day_gap'
          ELSE 'other_gap' END                             AS gap_type
    ,COUNT(*)                                              AS pair_count
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct_of_all_pairs
FROM visit_pairs
WHERE days_gap IS NOT NULL
GROUP BY plc_transition, gap_type
ORDER BY pair_count DESC
;

-- ------------------------------------------------------------
-- Q8. Overall % of visit rows that are inpatient
-- ------------------------------------------------------------
SELECT
    COUNTIF(plc_srv_cd = 'I')                    AS inpatient_visit_rows
    ,COUNT(*)                                              AS total_visit_rows
    ,ROUND(100.0 * COUNTIF(plc_srv_cd = 'I')
        / COUNT(*), 2)                                     AS pct_inpatient
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
;
