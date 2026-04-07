-- ============================================================
-- CHAPTER 4: Triggers — Qualifying True New Onsets
-- Source: A870800_gen_rec_triggers_qualified
--         A870800_gen_rec_member_qualified
-- ============================================================

-- ============================================================
-- ACT 1: What is the trigger population?
-- Raw first encounters before any qualification rules
-- ============================================================

-- ------------------------------------------------------------
-- Q1. Total raw triggers and members with triggers
-- ------------------------------------------------------------
SELECT
    COUNT(*)                                               AS total_triggers
    ,COUNT(DISTINCT member_id)                             AS members_with_triggers
    ,ROUND(AVG(trigger_count), 1)                          AS avg_triggers_per_member
    ,APPROX_QUANTILES(trigger_count, 100)[OFFSET(50)]      AS median_triggers_per_member
FROM (
    SELECT
        member_id
        ,COUNT(*)                                          AS trigger_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
    GROUP BY member_id
)
;

-- ------------------------------------------------------------
-- Q2. Triggers per member distribution
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN trigger_count = 1          THEN '01_exactly_1'
        WHEN trigger_count = 2          THEN '02_exactly_2'
        WHEN trigger_count BETWEEN 3 AND 5  THEN '03_3_to_5'
        WHEN trigger_count BETWEEN 6 AND 10 THEN '04_6_to_10'
        WHEN trigger_count > 10         THEN '05_10_plus'
    END                                                    AS trigger_bucket
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct
FROM (
    SELECT
        member_id
        ,COUNT(*)                                          AS trigger_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
    GROUP BY member_id
)
GROUP BY trigger_bucket
ORDER BY trigger_bucket
;

-- ------------------------------------------------------------
-- Q3. Where in the member journey does the trigger fall?
-- Distribution of trigger_rank — is it early or deep in history?
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN trigger_rank = 1           THEN '01_first_visit'
        WHEN trigger_rank = 2           THEN '02_second_visit'
        WHEN trigger_rank BETWEEN 3 AND 5   THEN '03_3rd_to_5th'
        WHEN trigger_rank BETWEEN 6 AND 10  THEN '04_6th_to_10th'
        WHEN trigger_rank > 10          THEN '05_beyond_10th'
    END                                                    AS trigger_rank_bucket
    ,COUNT(*)                                              AS triggers
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
GROUP BY trigger_rank_bucket
ORDER BY trigger_rank_bucket
;

-- ============================================================
-- ACT 2: Are we capturing true new onsets?
-- Justifying Rule 1 (12m enrollment) and Rule 2 (DX not seen)
-- ============================================================

-- ------------------------------------------------------------
-- Q4. Rule 1 — 12 months enrollment before trigger
-- How many triggers have 12m of history?
-- How many are lost without this rule?
-- ------------------------------------------------------------
SELECT
    COUNTIF(rule1_enrolled_12m = TRUE)                     AS triggers_pass_rule1
    ,COUNTIF(rule1_enrolled_12m = FALSE)                   AS triggers_fail_rule1
    ,COUNT(*)                                              AS total_triggers
    ,ROUND(100.0 * COUNTIF(rule1_enrolled_12m = TRUE)
        / COUNT(*), 2)                                     AS pct_pass_rule1
    ,ROUND(100.0 * COUNTIF(rule1_enrolled_12m = FALSE)
        / COUNT(*), 2)                                     AS pct_lost_rule1
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
;

-- ------------------------------------------------------------
-- Q5. Rule 2 — DX not seen in prior 12 months
-- Among triggers that pass Rule 1 — how many had
-- the same DX already active in the prior year?
-- ------------------------------------------------------------
SELECT
    COUNTIF(rule1_enrolled_12m = TRUE
        AND rule2_dx_not_seen_12m = TRUE)                  AS triggers_pass_both
    ,COUNTIF(rule1_enrolled_12m = TRUE
        AND rule2_dx_not_seen_12m = FALSE)                 AS triggers_fail_rule2
    ,COUNTIF(rule1_enrolled_12m = TRUE)                    AS triggers_pass_rule1
    ,ROUND(100.0 * COUNTIF(rule1_enrolled_12m = TRUE
        AND rule2_dx_not_seen_12m = FALSE)
        / NULLIF(COUNTIF(rule1_enrolled_12m = TRUE), 0)
        , 2)                                               AS pct_dx_already_active
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
;

-- ------------------------------------------------------------
-- Q6. Full qualification funnel — trigger level
-- Shows attrition at each rule step
-- ------------------------------------------------------------
SELECT
    COUNT(*)                                               AS total_raw_triggers
    ,COUNTIF(rule1_enrolled_12m = TRUE)                    AS after_rule1
    ,COUNTIF(is_left_qualified = TRUE)                     AS after_rule1_and_rule2
    ,COUNTIF(is_t30_qualified = TRUE)                      AS t30_qualified
    ,COUNTIF(is_t60_qualified = TRUE)                      AS t60_qualified
    ,COUNTIF(is_t180_qualified = TRUE)                     AS t180_qualified
    ,ROUND(100.0 * COUNTIF(rule1_enrolled_12m = TRUE)
        / COUNT(*), 2)                                     AS pct_after_rule1
    ,ROUND(100.0 * COUNTIF(is_left_qualified = TRUE)
        / COUNT(*), 2)                                     AS pct_left_qualified
    ,ROUND(100.0 * COUNTIF(is_t30_qualified = TRUE)
        / COUNT(*), 2)                                     AS pct_t30_qualified
    ,ROUND(100.0 * COUNTIF(is_t60_qualified = TRUE)
        / COUNT(*), 2)                                     AS pct_t60_qualified
    ,ROUND(100.0 * COUNTIF(is_t180_qualified = TRUE)
        / COUNT(*), 2)                                     AS pct_t180_qualified
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
;

-- ------------------------------------------------------------
-- Q7. Member level funnel — how many members survive each step
-- ------------------------------------------------------------
SELECT
    COUNT(DISTINCT member_id)                              AS total_members_with_triggers
    ,COUNT(DISTINCT CASE WHEN rule1_enrolled_12m = TRUE
        THEN member_id END)                                AS after_rule1
    ,COUNT(DISTINCT CASE WHEN is_left_qualified = TRUE
        THEN member_id END)                                AS after_rule1_and_rule2
    ,COUNT(DISTINCT CASE WHEN is_t30_qualified = TRUE
        THEN member_id END)                                AS t30_qualified
    ,COUNT(DISTINCT CASE WHEN is_t60_qualified = TRUE
        THEN member_id END)                                AS t60_qualified
    ,COUNT(DISTINCT CASE WHEN is_t180_qualified = TRUE
        THEN member_id END)                                AS t180_qualified
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
;

-- ============================================================
-- ACT 3: How quickly does the next event happen after trigger?
-- Justifies T30 / T60 / T180 prediction windows
-- ============================================================

-- ------------------------------------------------------------
-- Q8. Days from enrollment start to first trigger
-- How long does a member need before generating a new diagnosis?
-- ------------------------------------------------------------
SELECT
    MIN(days_to_first_trigger)                             AS min_days
    ,MAX(days_to_first_trigger)                            AS max_days
    ,ROUND(AVG(days_to_first_trigger), 1)                  AS mean_days
    ,APPROX_QUANTILES(days_to_first_trigger, 100)[OFFSET(25)]  AS p25
    ,APPROX_QUANTILES(days_to_first_trigger, 100)[OFFSET(50)]  AS median_days
    ,APPROX_QUANTILES(days_to_first_trigger, 100)[OFFSET(75)]  AS p75
    ,APPROX_QUANTILES(days_to_first_trigger, 100)[OFFSET(90)]  AS p90
FROM (
    SELECT
        t.member_id
        ,DATE_DIFF(MIN(t.trigger_date), m.enrollment_start, DAY)
                                                           AS days_to_first_trigger
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified` m
        ON t.member_id = m.member_id
    GROUP BY t.member_id, m.enrollment_start
)
;

-- ------------------------------------------------------------
-- Q9. Days to next visit after qualified trigger
-- Distribution across T windows — justifies T30/T60/T180
-- Only among left qualified triggers
-- ------------------------------------------------------------
WITH next_visit_gap AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,MIN(v.visit_date)                                 AS next_visit_date
        ,DATE_DIFF(MIN(v.visit_date), t.trigger_date, DAY) AS days_to_next_visit
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date > t.trigger_date
    WHERE t.is_left_qualified = TRUE
    GROUP BY t.member_id, t.trigger_date
)
SELECT
    COUNTIF(days_to_next_visit BETWEEN 1 AND 30)           AS next_visit_t30
    ,COUNTIF(days_to_next_visit BETWEEN 31 AND 60)         AS next_visit_t31_60
    ,COUNTIF(days_to_next_visit BETWEEN 61 AND 180)        AS next_visit_t61_180
    ,COUNTIF(days_to_next_visit > 180)                     AS next_visit_beyond_180
    ,COUNT(*)                                              AS total_qualified_triggers
    ,ROUND(100.0 * COUNTIF(days_to_next_visit BETWEEN 1 AND 30)
        / COUNT(*), 2)                                     AS pct_t30
    ,ROUND(100.0 * COUNTIF(days_to_next_visit BETWEEN 31 AND 60)
        / COUNT(*), 2)                                     AS pct_t31_60
    ,ROUND(100.0 * COUNTIF(days_to_next_visit BETWEEN 61 AND 180)
        / COUNT(*), 2)                                     AS pct_t61_180
    ,ROUND(100.0 * COUNTIF(days_to_next_visit > 180)
        / COUNT(*), 2)                                     AS pct_beyond_180
FROM next_visit_gap
;

-- ------------------------------------------------------------
-- Q10. Triggers by member segment
-- Are some segments generating more triggers than others?
-- ------------------------------------------------------------
SELECT
    member_segment
    ,COUNT(*)                                              AS total_triggers
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(AVG(trigger_count), 1)                          AS avg_triggers_per_member
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct_of_triggers
FROM (
    SELECT
        member_id
        ,member_segment
        ,COUNT(*)                                          AS trigger_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
    GROUP BY member_id, member_segment
)
GROUP BY member_segment
ORDER BY total_triggers DESC
;
