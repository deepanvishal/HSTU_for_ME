-- ============================================================
-- EDA_trigger_sequence_and_followup.sql
-- Purpose : Two analyses for qualified triggers
--   1. Visit history distribution before trigger (0-6, 7-20, 20+)
--   2. Days between trigger and next visit — full distro + T windows
-- Source  : A870800_gen_rec_triggers_qualified
--           A870800_gen_rec_visits
-- ============================================================

-- ------------------------------------------------------------
-- Q1. Pre-trigger visit history distribution
-- For each left-qualified trigger — how many distinct visit dates
-- did the member have BEFORE that trigger date?
-- Bucketed into 0-6, 7-20, 20+
-- Should sum to total left-qualified triggers (~1.4M)
-- ------------------------------------------------------------
WITH pre_trigger_counts AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx_clean
        -- Count distinct visit dates before THIS specific trigger
        -- Grouped by member_id + trigger_date to avoid cross-trigger bleed
        ,COUNT(DISTINCT v.visit_date)                      AS visits_before_trigger
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date < t.trigger_date
    WHERE t.is_left_qualified = TRUE
    GROUP BY t.member_id, t.trigger_date, t.trigger_dx_clean
)
SELECT
    CASE
        WHEN visits_before_trigger BETWEEN 0 AND 6  THEN '01_0_to_6_visits'
        WHEN visits_before_trigger BETWEEN 7 AND 20 THEN '02_7_to_20_visits'
        WHEN visits_before_trigger > 20             THEN '03_20_plus_visits'
    END                                                    AS visit_bucket
    ,COUNT(*)                                              AS trigger_count
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2)     AS pct_of_total
    ,ROUND(AVG(visits_before_trigger), 1)                  AS avg_visits_in_bucket
    ,MIN(visits_before_trigger)                            AS min_visits
    ,MAX(visits_before_trigger)                            AS max_visits
FROM pre_trigger_counts
GROUP BY visit_bucket
ORDER BY visit_bucket
;

-- ------------------------------------------------------------
-- Q2. Days between trigger and next visit — full distribution
-- For each left-qualified trigger — how many days until the
-- member's next visit after the trigger date?
-- Shows overall distro + breakdown by T30 / T60 / T180 windows
-- ------------------------------------------------------------
WITH next_visit_gaps AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx_clean
        ,MIN(v.visit_date)                                 AS next_visit_date
        ,DATE_DIFF(MIN(v.visit_date), t.trigger_date, DAY) AS days_to_next_visit
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date > t.trigger_date
    WHERE t.is_left_qualified = TRUE
    GROUP BY t.member_id, t.trigger_date, t.trigger_dx_clean
)
SELECT
    CASE
        WHEN days_to_next_visit BETWEEN 1  AND 30  THEN '01_T30_1_to_30d'
        WHEN days_to_next_visit BETWEEN 31 AND 60  THEN '02_T60_31_to_60d'
        WHEN days_to_next_visit BETWEEN 61 AND 180 THEN '03_T180_61_to_180d'
        ELSE                                            '04_Beyond_180d'
    END                                                    AS time_window
    ,COUNT(*)                                              AS trigger_count
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2)     AS pct_of_total
    ,MIN(days_to_next_visit)                               AS min_days
    ,MAX(days_to_next_visit)                               AS max_days
    ,ROUND(AVG(days_to_next_visit), 1)                     AS mean_days
    ,APPROX_QUANTILES(days_to_next_visit, 100)[OFFSET(25)] AS p25_days
    ,APPROX_QUANTILES(days_to_next_visit, 100)[OFFSET(50)] AS median_days
    ,APPROX_QUANTILES(days_to_next_visit, 100)[OFFSET(75)] AS p75_days
FROM next_visit_gaps
WHERE days_to_next_visit IS NOT NULL
  AND days_to_next_visit > 0
GROUP BY time_window
ORDER BY time_window
;

-- ------------------------------------------------------------
-- Q2b. Days distribution — day-level counts within each window
-- Use this to understand shape of distribution per window
-- ------------------------------------------------------------
WITH next_visit_gaps AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx_clean
        ,DATE_DIFF(MIN(v.visit_date), t.trigger_date, DAY) AS days_to_next_visit
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date > t.trigger_date
    WHERE t.is_left_qualified = TRUE
    GROUP BY t.member_id, t.trigger_date, t.trigger_dx_clean
)
SELECT
    days_to_next_visit
    ,CASE
        WHEN days_to_next_visit BETWEEN 1  AND 30  THEN '01_T30'
        WHEN days_to_next_visit BETWEEN 31 AND 60  THEN '02_T60'
        WHEN days_to_next_visit BETWEEN 61 AND 180 THEN '03_T180'
        ELSE                                            '04_Beyond'
    END                                                    AS time_window
    ,COUNT(*)                                              AS trigger_count
FROM next_visit_gaps
WHERE days_to_next_visit IS NOT NULL
  AND days_to_next_visit > 0
  AND days_to_next_visit <= 180
GROUP BY days_to_next_visit, time_window
ORDER BY days_to_next_visit
;
