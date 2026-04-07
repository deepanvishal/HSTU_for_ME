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
-- Q4. Members with 1 visit vs 6+ visits
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
-- And among those — what % involve inpatient (plc_srv_ctg_cd = 'I')
-- ------------------------------------------------------------
WITH visit_pairs AS (
    SELECT
        v.member_id
        ,v.visit_date                                      AS current_visit
        ,v.plc_srv_cd                                      AS current_plc_srv_cd
        ,LAG(v.visit_date) OVER (
            PARTITION BY v.member_id ORDER BY v.visit_date
        )                                                  AS prev_visit
        ,LAG(v.plc_srv_cd) OVER (
            PARTITION BY v.member_id ORDER BY v.visit_date
        )                                                  AS prev_plc_srv_cd
        ,DATE_DIFF(
            v.visit_date,
            LAG(v.visit_date) OVER (
                PARTITION BY v.member_id ORDER BY v.visit_date
            ),
            DAY
        )                                                  AS days_gap
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
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
-- Q8. Overall % of visit rows that are inpatient
-- ------------------------------------------------------------
SELECT
    COUNTIF(plc_srv_ctg_cd = 'I')                    AS inpatient_visit_rows
    ,COUNT(*)                                              AS total_visit_rows
    ,ROUND(100.0 * COUNTIF(plc_srv_ctg_cd = 'I')
        / COUNT(*), 2)                                     AS pct_inpatient
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
;

-- ============================================================
-- ACT 4: Should we collapse inpatient clusters?
-- ============================================================

-- ------------------------------------------------------------
-- Q9. Inpatient cluster identification
-- Consecutive inpatient days per member = one admission
-- ------------------------------------------------------------
WITH inpatient_visits AS (
    SELECT DISTINCT
        member_id
        ,visit_date
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE plc_srv_ctg_cd = 'I'
),
with_gap AS (
    SELECT
        member_id
        ,visit_date
        ,DATE_DIFF(
            visit_date,
            LAG(visit_date) OVER (
                PARTITION BY member_id ORDER BY visit_date
            ),
            DAY
        )                                                  AS days_from_prev
    FROM inpatient_visits
),
with_cluster AS (
    SELECT
        member_id
        ,visit_date
        ,SUM(CASE WHEN days_from_prev > 1
                  OR days_from_prev IS NULL
             THEN 1 ELSE 0 END)
            OVER (PARTITION BY member_id ORDER BY visit_date)
                                                           AS cluster_id
    FROM with_gap
)
SELECT
    COUNT(DISTINCT CONCAT(member_id,'_',CAST(cluster_id AS STRING)))
                                                           AS total_inpatient_admissions
    ,COUNT(DISTINCT member_id)                             AS members_with_inpatient
    ,ROUND(AVG(days_in_cluster), 1)                        AS avg_stay_days
    ,APPROX_QUANTILES(days_in_cluster, 100)[OFFSET(50)]    AS median_stay_days
    ,APPROX_QUANTILES(days_in_cluster, 100)[OFFSET(90)]    AS p90_stay_days
    ,MAX(days_in_cluster)                                  AS max_stay_days
    ,COUNTIF(days_in_cluster = 1)                          AS single_day_admissions
    ,COUNTIF(days_in_cluster > 1)                          AS multi_day_admissions
FROM (
    SELECT
        member_id
        ,cluster_id
        ,COUNT(*)                                          AS days_in_cluster
    FROM with_cluster
    GROUP BY member_id, cluster_id
)
;

-- ------------------------------------------------------------
-- Q10. Before vs After collapsing inpatient clusters
-- Visits per member distribution — how does it shift?
-- ------------------------------------------------------------
WITH inpatient_collapsed AS (
    -- Inpatient: collapse consecutive days to single visit (first day of admission)
    WITH inpatient_visits AS (
        SELECT DISTINCT member_id, visit_date
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
        WHERE plc_srv_ctg_cd = 'I'
    ),
    with_gap AS (
        SELECT
            member_id
            ,visit_date
            ,DATE_DIFF(visit_date,
                LAG(visit_date) OVER (PARTITION BY member_id ORDER BY visit_date),
                DAY)                                       AS days_from_prev
        FROM inpatient_visits
    ),
    with_cluster AS (
        SELECT
            member_id
            ,visit_date
            ,SUM(CASE WHEN days_from_prev > 1
                      OR days_from_prev IS NULL
                 THEN 1 ELSE 0 END)
                OVER (PARTITION BY member_id ORDER BY visit_date)
                                                           AS cluster_id
        FROM with_gap
    )
    SELECT member_id, MIN(visit_date) AS visit_date
    FROM with_cluster
    GROUP BY member_id, cluster_id
),
non_inpatient AS (
    SELECT DISTINCT member_id, visit_date
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE plc_srv_ctg_cd != 'I'
),
combined AS (
    SELECT member_id, visit_date FROM inpatient_collapsed
    UNION DISTINCT
    SELECT member_id, visit_date FROM non_inpatient
)
SELECT
    'After Collapse'                                       AS version
    ,MIN(visit_count)                                      AS min_visits
    ,MAX(visit_count)                                      AS max_visits
    ,ROUND(AVG(visit_count), 1)                            AS mean_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(50)]        AS median_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(90)]        AS p90_visits
FROM (
    SELECT member_id, COUNT(DISTINCT visit_date) AS visit_count
    FROM combined GROUP BY member_id
)

UNION ALL

SELECT
    'Before Collapse'                                      AS version
    ,MIN(visit_count)                                      AS min_visits
    ,MAX(visit_count)                                      AS max_visits
    ,ROUND(AVG(visit_count), 1)                            AS mean_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(50)]        AS median_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(90)]        AS p90_visits
FROM (
    SELECT member_id, COUNT(DISTINCT visit_date) AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY member_id
)
;
