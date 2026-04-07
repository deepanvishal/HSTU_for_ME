-- ============================================================
-- CHAPTER 6: Sequence Signal for SASRec
-- Story: Does the qualified population have enough history
--        and diversity for sequential modeling to work?
-- Source: A870800_gen_rec_visits_qualified (T180 qualified)
--         A870800_gen_rec_visits (all visits for sequence context)
-- ============================================================

-- ============================================================
-- ACT 1: Is the sequence long enough?
-- ============================================================

-- ------------------------------------------------------------
-- Q1. Visit sequence length distribution
-- Among T180 qualified members — how many visits do they have
-- in their FULL history (not just the qualified window)
-- ------------------------------------------------------------
SELECT
    MIN(visit_count)                                       AS min_visits
    ,MAX(visit_count)                                      AS max_visits
    ,ROUND(AVG(visit_count), 1)                            AS mean_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(10)]        AS p10
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(25)]        AS p25
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(50)]        AS median_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(75)]        AS p75
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(90)]        AS p90
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(99)]        AS p99
FROM (
    SELECT
        v.member_id
        ,COUNT(DISTINCT v.visit_date)                      AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    INNER JOIN (
        SELECT DISTINCT member_id
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
        WHERE is_t180_qualified = TRUE
    ) q ON v.member_id = q.member_id
    GROUP BY v.member_id
)
;

-- ------------------------------------------------------------
-- Q2. Sequence length buckets
-- Full distribution — shows where the population clusters
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN visit_count = 1                THEN '01_exactly_1'
        WHEN visit_count BETWEEN 2 AND 3    THEN '02_2_to_3'
        WHEN visit_count BETWEEN 4 AND 5    THEN '03_4_to_5'
        WHEN visit_count BETWEEN 6 AND 10   THEN '04_6_to_10'
        WHEN visit_count BETWEEN 11 AND 20  THEN '05_11_to_20'
        WHEN visit_count > 20               THEN '06_20_plus'
    END                                                    AS sequence_bucket
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct
FROM (
    SELECT
        v.member_id
        ,COUNT(DISTINCT v.visit_date)                      AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    INNER JOIN (
        SELECT DISTINCT member_id
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
        WHERE is_t180_qualified = TRUE
    ) q ON v.member_id = q.member_id
    GROUP BY v.member_id
)
GROUP BY sequence_bucket
ORDER BY sequence_bucket
;

-- ------------------------------------------------------------
-- Q3. Cumulative % by sequence length threshold
-- What % of members have 3+, 6+, 10+, 15+, 20+ visits?
-- This defines the trainable population at each threshold
-- ------------------------------------------------------------
SELECT
    threshold
    ,members_at_or_above
    ,total_members
    ,ROUND(100.0 * members_at_or_above / total_members, 2) AS pct_of_qualified
FROM (
    SELECT
        t.threshold
        ,COUNTIF(visit_count >= t.threshold)               AS members_at_or_above
        ,COUNT(*)                                          AS total_members
    FROM (
        SELECT
            v.member_id
            ,COUNT(DISTINCT v.visit_date)                  AS visit_count
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        INNER JOIN (
            SELECT DISTINCT member_id
            FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
            WHERE is_t180_qualified = TRUE
        ) q ON v.member_id = q.member_id
        GROUP BY v.member_id
    )
    CROSS JOIN UNNEST([3, 6, 10, 15, 20]) AS t(threshold)
    GROUP BY t.threshold
)
ORDER BY threshold
;

-- ============================================================
-- ACT 2: Is there enough time between visits to matter?
-- ============================================================

-- ------------------------------------------------------------
-- Q4. Inter-visit gap — 6+ visit members vs < 6 visit members
-- Do longer-sequence members visit more regularly?
-- ------------------------------------------------------------
WITH member_visits AS (
    SELECT
        v.member_id
        ,v.visit_date
        ,COUNT(DISTINCT v.visit_date) OVER (
            PARTITION BY v.member_id
        )                                                  AS total_visits
        ,DATE_DIFF(
            v.visit_date,
            LAG(v.visit_date) OVER (
                PARTITION BY v.member_id ORDER BY v.visit_date
            ),
            DAY
        )                                                  AS days_gap
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    INNER JOIN (
        SELECT DISTINCT member_id
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
        WHERE is_t180_qualified = TRUE
    ) q ON v.member_id = q.member_id
)
SELECT
    CASE
        WHEN total_visits >= 6  THEN '6_plus_visits'
        ELSE                         'less_than_6_visits'
    END                                                    AS sequence_group
    ,COUNT(*)                                              AS total_pairs
    ,ROUND(AVG(days_gap), 1)                               AS mean_gap_days
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(25)]           AS p25_gap
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(50)]           AS median_gap_days
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(75)]           AS p75_gap
    ,APPROX_QUANTILES(days_gap, 100)[OFFSET(90)]           AS p90_gap
FROM member_visits
WHERE days_gap IS NOT NULL
GROUP BY sequence_group
ORDER BY sequence_group
;

-- ============================================================
-- ACT 3: Is there enough specialty diversity to predict?
-- ============================================================

-- ------------------------------------------------------------
-- Q5. Unique specialties per member distribution
-- Among T180 qualified members
-- ------------------------------------------------------------
SELECT
    MIN(unique_specialties)                                AS min_specialties
    ,MAX(unique_specialties)                               AS max_specialties
    ,ROUND(AVG(unique_specialties), 1)                     AS mean_specialties
    ,APPROX_QUANTILES(unique_specialties, 100)[OFFSET(25)] AS p25
    ,APPROX_QUANTILES(unique_specialties, 100)[OFFSET(50)] AS median_specialties
    ,APPROX_QUANTILES(unique_specialties, 100)[OFFSET(75)] AS p75
    ,APPROX_QUANTILES(unique_specialties, 100)[OFFSET(90)] AS p90
FROM (
    SELECT
        v.member_id
        ,COUNT(DISTINCT v.specialty_ctg_cd)                AS unique_specialties
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    INNER JOIN (
        SELECT DISTINCT member_id
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
        WHERE is_t180_qualified = TRUE
    ) q ON v.member_id = q.member_id
    GROUP BY v.member_id
)
;

-- ------------------------------------------------------------
-- Q6. Members with only 1 unique specialty — trivial sequences
-- Model has nothing to learn from these members
-- ------------------------------------------------------------
SELECT
    COUNTIF(unique_specialties = 1)                        AS single_specialty_members
    ,COUNTIF(unique_specialties > 1)                       AS multi_specialty_members
    ,COUNT(*)                                              AS total_members
    ,ROUND(100.0 * COUNTIF(unique_specialties = 1)
        / COUNT(*), 2)                                     AS pct_single_specialty
    ,ROUND(100.0 * COUNTIF(unique_specialties > 1)
        / COUNT(*), 2)                                     AS pct_multi_specialty
FROM (
    SELECT
        v.member_id
        ,COUNT(DISTINCT v.specialty_ctg_cd)                AS unique_specialties
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    INNER JOIN (
        SELECT DISTINCT member_id
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
        WHERE is_t180_qualified = TRUE
    ) q ON v.member_id = q.member_id
    GROUP BY v.member_id
)
;

-- ============================================================
-- ACT 4: Is the self-loop rate too high?
-- ============================================================

-- ------------------------------------------------------------
-- Q7. Self-loop rate — consecutive visits with same specialty
-- High self-loop = next visit = same specialty is trivial answer
-- Among T180 qualified members only
-- ------------------------------------------------------------
WITH transitions AS (
    SELECT
        v.member_id
        ,v.specialty_ctg_cd                                AS current_specialty
        ,LAG(v.specialty_ctg_cd) OVER (
            PARTITION BY v.member_id ORDER BY v.visit_date
        )                                                  AS prev_specialty
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    INNER JOIN (
        SELECT DISTINCT member_id
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
        WHERE is_t180_qualified = TRUE
    ) q ON v.member_id = q.member_id
)
SELECT
    COUNT(*)                                               AS total_transitions
    ,COUNTIF(current_specialty = prev_specialty)           AS self_loops
    ,COUNTIF(current_specialty != prev_specialty)          AS specialty_changes
    ,ROUND(100.0 * COUNTIF(current_specialty = prev_specialty)
        / COUNT(*), 2)                                     AS self_loop_rate_pct
    ,ROUND(100.0 * COUNTIF(current_specialty != prev_specialty)
        / COUNT(*), 2)                                     AS transition_rate_pct
FROM transitions
WHERE prev_specialty IS NOT NULL
;

-- ------------------------------------------------------------
-- Q8. Self-loop rate by specialty
-- Which specialties are most likely to repeat?
-- High repeat = chronic condition management
-- Low repeat = acute / referral visit
-- ------------------------------------------------------------
WITH transitions AS (
    SELECT
        v.member_id
        ,v.specialty_ctg_cd                                AS current_specialty
        ,LAG(v.specialty_ctg_cd) OVER (
            PARTITION BY v.member_id ORDER BY v.visit_date
        )                                                  AS prev_specialty
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    INNER JOIN (
        SELECT DISTINCT member_id
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
        WHERE is_t180_qualified = TRUE
    ) q ON v.member_id = q.member_id
)
SELECT
    current_specialty
    ,COUNT(*)                                              AS total_transitions
    ,COUNTIF(current_specialty = prev_specialty)           AS self_loops
    ,ROUND(100.0 * COUNTIF(current_specialty = prev_specialty)
        / COUNT(*), 2)                                     AS self_loop_rate_pct
FROM transitions
WHERE prev_specialty IS NOT NULL
GROUP BY current_specialty
ORDER BY self_loop_rate_pct DESC
LIMIT 20
;
