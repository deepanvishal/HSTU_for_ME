-- ============================================================
-- CHAPTER 4: Visits → Triggers
-- Story: How raw visits were transformed into qualified triggers
-- Sources: A870800_gen_rec_visits
--          A870800_claims_gen_rec_2022_2025_sfl
--          A870800_gen_rec_triggers_qualified
--          A870800_gen_rec_member_qualified
-- ============================================================

-- ============================================================
-- ACT 1: Data quality filters applied to visits
-- What got dropped before first encounter logic was applied?
-- ============================================================

-- ------------------------------------------------------------
-- Q1. Total visit rows vs visits with valid dx and specialty
-- Shows % of visits lost by NULL dx and NULL specialty filters
-- ------------------------------------------------------------
SELECT
    COUNT(*)                                               AS total_visit_rows
    ,COUNTIF(pri_icd9_dx_cd IS NOT NULL
        AND TRIM(pri_icd9_dx_cd) != '')                    AS visits_with_dx
    ,COUNTIF(specialty_ctg_cd IS NOT NULL
        AND TRIM(specialty_ctg_cd) != '')                  AS visits_with_specialty
    ,COUNTIF(pri_icd9_dx_cd IS NOT NULL
        AND TRIM(pri_icd9_dx_cd) != ''
        AND specialty_ctg_cd IS NOT NULL
        AND TRIM(specialty_ctg_cd) != '')                  AS visits_with_both
    ,ROUND(100.0 * COUNTIF(pri_icd9_dx_cd IS NULL
        OR TRIM(pri_icd9_dx_cd) = '')
        / COUNT(*), 2)                                     AS pct_lost_null_dx
    ,ROUND(100.0 * COUNTIF(specialty_ctg_cd IS NULL
        OR TRIM(specialty_ctg_cd) = '')
        / COUNT(*), 2)                                     AS pct_lost_null_specialty
    ,ROUND(100.0 * COUNTIF(pri_icd9_dx_cd IS NOT NULL
        AND TRIM(pri_icd9_dx_cd) != ''
        AND specialty_ctg_cd IS NOT NULL
        AND TRIM(specialty_ctg_cd) != '')
        / COUNT(*), 2)                                     AS pct_visits_surviving
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
;

-- ------------------------------------------------------------
-- Q2. Where are NULL dx concentrated?
-- By place of service
-- ------------------------------------------------------------
SELECT
    plc_srv_cd
    ,COUNT(*)                                              AS total_visits
    ,COUNTIF(pri_icd9_dx_cd IS NULL
        OR TRIM(pri_icd9_dx_cd) = '')                      AS null_dx_visits
    ,ROUND(100.0 * COUNTIF(pri_icd9_dx_cd IS NULL
        OR TRIM(pri_icd9_dx_cd) = '')
        / COUNT(*), 2)                                     AS pct_null_dx
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
GROUP BY plc_srv_cd
ORDER BY null_dx_visits DESC
;

-- ------------------------------------------------------------
-- Q3. Where are NULL specialties concentrated?
-- By place of service
-- ------------------------------------------------------------
SELECT
    plc_srv_cd
    ,COUNT(*)                                              AS total_visits
    ,COUNTIF(specialty_ctg_cd IS NULL
        OR TRIM(specialty_ctg_cd) = '')                    AS null_specialty_visits
    ,ROUND(100.0 * COUNTIF(specialty_ctg_cd IS NULL
        OR TRIM(specialty_ctg_cd) = '')
        / COUNT(*), 2)                                     AS pct_null_specialty
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
GROUP BY plc_srv_cd
ORDER BY null_specialty_visits DESC
;

-- ============================================================
-- ACT 2: First encounter logic
-- Why first encounter = trigger, and what does the data show?
-- ============================================================

-- ------------------------------------------------------------
-- Q4. How many unique dx codes per member?
-- Members encounter many distinct diagnoses over time
-- ------------------------------------------------------------
SELECT
    MIN(unique_dx)                                         AS min_unique_dx
    ,MAX(unique_dx)                                        AS max_unique_dx
    ,ROUND(AVG(unique_dx), 1)                              AS mean_unique_dx
    ,APPROX_QUANTILES(unique_dx, 100)[OFFSET(25)]          AS p25
    ,APPROX_QUANTILES(unique_dx, 100)[OFFSET(50)]          AS median_unique_dx
    ,APPROX_QUANTILES(unique_dx, 100)[OFFSET(75)]          AS p75
    ,APPROX_QUANTILES(unique_dx, 100)[OFFSET(90)]          AS p90
FROM (
    SELECT
        member_id
        ,COUNT(DISTINCT REPLACE(TRIM(pri_icd9_dx_cd), '.', ''))
                                                           AS unique_dx
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE pri_icd9_dx_cd IS NOT NULL
      AND TRIM(pri_icd9_dx_cd) != ''
    GROUP BY member_id
)
;

-- ------------------------------------------------------------
-- Q5. How often does the same dx repeat per member?
-- High repeat rate confirms chronic conditions recur —
-- justifies keeping only first encounter as the true new onset
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN dx_occurrences = 1              THEN '01_seen_once'
        WHEN dx_occurrences = 2              THEN '02_seen_twice'
        WHEN dx_occurrences BETWEEN 3 AND 5  THEN '03_3_to_5x'
        WHEN dx_occurrences BETWEEN 6 AND 10 THEN '04_6_to_10x'
        WHEN dx_occurrences > 10             THEN '05_10x_plus'
    END                                                    AS repeat_bucket
    ,COUNT(*)                                              AS member_dx_pairs
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct
FROM (
    SELECT
        member_id
        ,REPLACE(TRIM(pri_icd9_dx_cd), '.', '')            AS dx_clean
        ,COUNT(*)                                          AS dx_occurrences
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE pri_icd9_dx_cd IS NOT NULL
      AND TRIM(pri_icd9_dx_cd) != ''
    GROUP BY member_id, dx_clean
)
GROUP BY repeat_bucket
ORDER BY repeat_bucket
;

-- ------------------------------------------------------------
-- Q6. First encounters vs repeat encounters — overall split
-- What % of all visit rows are first encounters?
-- ------------------------------------------------------------
WITH ranked AS (
    SELECT
        member_id
        ,REPLACE(TRIM(pri_icd9_dx_cd), '.', '')            AS dx_clean
        ,visit_date
        ,ROW_NUMBER() OVER (
            PARTITION BY member_id
                ,REPLACE(TRIM(pri_icd9_dx_cd), '.', '')
            ORDER BY visit_date
        )                                                  AS dx_encounter_rank
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE pri_icd9_dx_cd IS NOT NULL
      AND TRIM(pri_icd9_dx_cd) != ''
)
SELECT
    COUNTIF(dx_encounter_rank = 1)                         AS first_encounters
    ,COUNTIF(dx_encounter_rank > 1)                        AS repeat_encounters
    ,COUNT(*)                                              AS total_visits
    ,ROUND(100.0 * COUNTIF(dx_encounter_rank = 1)
        / COUNT(*), 2)                                     AS pct_first_encounters
    ,ROUND(100.0 * COUNTIF(dx_encounter_rank > 1)
        / COUNT(*), 2)                                     AS pct_repeat_encounters
FROM ranked
;

-- ============================================================
-- ACT 3: What is the trigger population?
-- Raw first encounters before any qualification rules
-- ============================================================

-- ------------------------------------------------------------
-- Q7. Total raw triggers and members with triggers
-- ------------------------------------------------------------
SELECT
    SUM(trigger_count)                                     AS total_triggers
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
-- Q8. Triggers per member distribution
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN trigger_count = 1              THEN '01_exactly_1'
        WHEN trigger_count = 2              THEN '02_exactly_2'
        WHEN trigger_count BETWEEN 3 AND 5  THEN '03_3_to_5'
        WHEN trigger_count BETWEEN 6 AND 10 THEN '04_6_to_10'
        WHEN trigger_count > 10             THEN '05_10_plus'
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
-- Q9. Where in the member journey does the trigger fall?
-- Distribution of trigger_rank
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN trigger_rank = 1               THEN '01_first_visit'
        WHEN trigger_rank = 2               THEN '02_second_visit'
        WHEN trigger_rank BETWEEN 3 AND 5   THEN '03_3rd_to_5th'
        WHEN trigger_rank BETWEEN 6 AND 10  THEN '04_6th_to_10th'
        WHEN trigger_rank > 10              THEN '05_beyond_10th'
    END                                                    AS trigger_rank_bucket
    ,COUNT(*)                                              AS triggers
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
GROUP BY trigger_rank_bucket
ORDER BY trigger_rank_bucket
;

-- ============================================================
-- ACT 4: Are we capturing true new onsets?
-- Justifying Rule 1 (12m enrollment) and Rule 2 (DX not seen)
-- ============================================================

-- ------------------------------------------------------------
-- Q10. Rule 1 — 12 months enrollment before trigger
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
-- Q11. Rule 2 — DX not seen in prior 12 months
-- Among triggers passing Rule 1 — how many had DX already active?
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
-- Q12. Full qualification funnel — trigger level
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
-- Q13. Member level funnel
-- How many members survive each qualification step
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
-- ACT 5: How quickly does the next event happen after trigger?
-- Justifies T30 / T60 / T180 prediction windows
-- ============================================================

-- ------------------------------------------------------------
-- Q14. Days from enrollment start to first trigger
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
-- Q15. Days to next visit after qualified trigger
-- Justifies T30 / T60 / T180 windows
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
-- Q16. Triggers by member segment
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
