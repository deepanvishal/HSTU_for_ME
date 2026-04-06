-- ============================================================
-- CHAPTER 1: Who is in our Universe?
-- Source: A870800_gen_rec_member_qualified
--         A870800_gen_rec_member_demographics
-- ============================================================

-- ------------------------------------------------------------
-- Q1. Total enrolled members
-- ------------------------------------------------------------
SELECT
    COUNT(DISTINCT member_id)                               AS total_members
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
;

-- ------------------------------------------------------------
-- Q2. Members by year
-- ------------------------------------------------------------
SELECT
    membership_year
    ,COUNT(DISTINCT member_id)                             AS members
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_demographics`
GROUP BY membership_year
ORDER BY membership_year
;

-- ------------------------------------------------------------
-- Q3. Members by submarket
-- ------------------------------------------------------------
SELECT
    submarket
    ,COUNT(DISTINCT member_id)                             AS members
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
GROUP BY submarket
ORDER BY members DESC
;

-- ------------------------------------------------------------
-- Q4. Age group distribution (0-18, 18-65, 65+)
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN age_nbr < 18               THEN '1_0_to_18'
        WHEN age_nbr BETWEEN 18 AND 65  THEN '2_18_to_65'
        WHEN age_nbr > 65               THEN '3_65_plus'
        ELSE                                 'Unknown'
    END                                                    AS age_group
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct
FROM (
    SELECT member_id, MAX(age_nbr) AS age_nbr
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_demographics`
    GROUP BY member_id
)
GROUP BY age_group
ORDER BY age_group
;

-- ------------------------------------------------------------
-- Q5. Gender split (Male / Female)
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN gender_cd = 'M' THEN 'Male'
        WHEN gender_cd = 'F' THEN 'Female'
        ELSE                      'Unknown'
    END                                                    AS gender
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct
FROM (
    SELECT member_id, MAX(gender_cd) AS gender_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_demographics`
    GROUP BY member_id
)
GROUP BY gender
ORDER BY members DESC
;

-- ------------------------------------------------------------
-- Q5b. Member segment distribution
-- Segments: Children, Adult_Male, Adult_Female, Senior
-- (exact values from _02_data_setup_visits.sql)
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN age_nbr < 18                               THEN 'Children'
        WHEN age_nbr BETWEEN 18 AND 65 AND gender_cd = 'M' THEN 'Adult_Male'
        WHEN age_nbr BETWEEN 18 AND 65 AND gender_cd = 'F' THEN 'Adult_Female'
        WHEN age_nbr > 65                               THEN 'Senior'
        ELSE                                                 'Unknown'
    END                                                    AS member_segment
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct
FROM (
    SELECT member_id, MAX(age_nbr) AS age_nbr, MAX(gender_cd) AS gender_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_demographics`
    GROUP BY member_id
)
GROUP BY member_segment
ORDER BY members DESC
;

-- ------------------------------------------------------------
-- Q6. Enrollment window distribution (enrollment_window_months)
-- ------------------------------------------------------------
SELECT
    MIN(enrollment_window_months)                          AS window_min
    ,MAX(enrollment_window_months)                         AS window_max
    ,ROUND(AVG(enrollment_window_months), 1)               AS window_mean
    ,APPROX_QUANTILES(enrollment_window_months, 100)[OFFSET(10)]  AS window_p10
    ,APPROX_QUANTILES(enrollment_window_months, 100)[OFFSET(25)]  AS window_p25
    ,APPROX_QUANTILES(enrollment_window_months, 100)[OFFSET(50)]  AS window_median
    ,APPROX_QUANTILES(enrollment_window_months, 100)[OFFSET(75)]  AS window_p75
    ,APPROX_QUANTILES(enrollment_window_months, 100)[OFFSET(90)]  AS window_p90
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
;

-- ------------------------------------------------------------
-- Q7. Enrolled months distribution (enrolled_months)
-- Note: enrolled_months = COUNT(DISTINCT eff_dt) — actual active months
--       vs enrollment_window_months = MAX - MIN in months (calendar span)
--       Gap between the two = months with no coverage within window
-- ------------------------------------------------------------
SELECT
    MIN(enrolled_months)                                   AS enrolled_min
    ,MAX(enrolled_months)                                  AS enrolled_max
    ,ROUND(AVG(enrolled_months), 1)                        AS enrolled_mean
    ,APPROX_QUANTILES(enrolled_months, 100)[OFFSET(25)]    AS enrolled_p25
    ,APPROX_QUANTILES(enrolled_months, 100)[OFFSET(50)]    AS enrolled_median
    ,APPROX_QUANTILES(enrolled_months, 100)[OFFSET(75)]    AS enrolled_p75
    ,APPROX_QUANTILES(enrolled_months, 100)[OFFSET(90)]    AS enrolled_p90
    ,COUNTIF(enrolled_months < enrollment_window_months)   AS members_with_gaps
    ,ROUND(100.0 * COUNTIF(enrolled_months < enrollment_window_months)
        / COUNT(*), 2)                                     AS pct_with_gaps
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
;

-- ------------------------------------------------------------
-- Q8. Members by enrollment window bucket
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN enrollment_window_months < 6   THEN '1_LT_6m'
        WHEN enrollment_window_months < 12  THEN '2_6_to_12m'
        WHEN enrollment_window_months < 24  THEN '3_12_to_24m'
        WHEN enrollment_window_months < 36  THEN '4_24_to_36m'
        ELSE                                     '5_36m_plus'
    END                                                    AS window_bucket
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
GROUP BY window_bucket
ORDER BY window_bucket
;

-- ------------------------------------------------------------
-- Q9. What % of members have 12+ months enrollment?
-- Justifies Rule 1 — is the 12m cutoff losing too many members?
-- ------------------------------------------------------------
SELECT
    COUNTIF(enrollment_window_months >= 12)                AS members_12m_plus
    ,COUNT(*)                                              AS total_members
    ,ROUND(100.0 * COUNTIF(enrollment_window_months >= 12)
        / COUNT(*), 2)                                     AS pct_12m_plus
    ,COUNTIF(enrollment_window_months < 12)                AS members_lost_by_rule1
    ,ROUND(100.0 * COUNTIF(enrollment_window_months < 12)
        / COUNT(*), 2)                                     AS pct_lost_by_rule1
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
;

-- ------------------------------------------------------------
-- Q10. Members with full 36-month continuous coverage
-- Best cohort — observed full 2022-2025 window
-- ------------------------------------------------------------
SELECT
    COUNTIF(enrollment_window_months >= 36)                AS members_full_36m
    ,COUNTIF(enrolled_months >= 36)                        AS members_active_36m
    ,COUNT(*)                                              AS total_members
    ,ROUND(100.0 * COUNTIF(enrollment_window_months >= 36)
        / COUNT(*), 2)                                     AS pct_full_36m
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
;
