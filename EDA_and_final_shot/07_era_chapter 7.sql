-- ============================================================
-- CHAPTER 7: Provider Level Analysis & Self-Loop
-- Story: At the provider level, how loyal are members?
--        Does same-provider repeat visits dominate the sequence?
-- Source: A870800_gen_rec_visits (all visits)
--         A870800_gen_rec_triggers_qualified (T180 qualified)
-- ============================================================

-- ============================================================
-- SECTION 1: Provider basics
-- ============================================================

-- ------------------------------------------------------------
-- Q1. Total distinct providers in the visit data
-- ------------------------------------------------------------
SELECT
    COUNT(DISTINCT srv_prvdr_id)                           AS total_providers
    ,COUNT(DISTINCT member_id)                             AS total_members
    ,COUNT(*)                                              AS total_visit_rows
    ,ROUND(COUNT(*) * 1.0 /
        COUNT(DISTINCT srv_prvdr_id), 1)                   AS avg_visits_per_provider
    ,ROUND(COUNT(*) * 1.0 /
        COUNT(DISTINCT member_id), 1)                      AS avg_visits_per_member
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
;

-- ------------------------------------------------------------
-- Q2. Visits per provider distribution
-- How concentrated is visit volume?
-- ------------------------------------------------------------
SELECT
    MIN(visit_count)                                       AS min_visits
    ,MAX(visit_count)                                      AS max_visits
    ,ROUND(AVG(visit_count), 1)                            AS mean_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(25)]        AS p25
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(50)]        AS median_visits
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(75)]        AS p75
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(90)]        AS p90
    ,APPROX_QUANTILES(visit_count, 100)[OFFSET(99)]        AS p99
FROM (
    SELECT
        srv_prvdr_id
        ,COUNT(*)                                          AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY srv_prvdr_id
)
;

-- ------------------------------------------------------------
-- Q3. Members per provider distribution
-- How many unique members does each provider see?
-- ------------------------------------------------------------
SELECT
    MIN(member_count)                                      AS min_members
    ,MAX(member_count)                                     AS max_members
    ,ROUND(AVG(member_count), 1)                           AS mean_members
    ,APPROX_QUANTILES(member_count, 100)[OFFSET(25)]       AS p25
    ,APPROX_QUANTILES(member_count, 100)[OFFSET(50)]       AS median_members
    ,APPROX_QUANTILES(member_count, 100)[OFFSET(75)]       AS p75
    ,APPROX_QUANTILES(member_count, 100)[OFFSET(90)]       AS p90
FROM (
    SELECT
        srv_prvdr_id
        ,COUNT(DISTINCT member_id)                         AS member_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY srv_prvdr_id
)
;

-- ------------------------------------------------------------
-- Q4. Providers per member distribution
-- How many unique providers does each member see?
-- ------------------------------------------------------------
SELECT
    MIN(provider_count)                                    AS min_providers
    ,MAX(provider_count)                                   AS max_providers
    ,ROUND(AVG(provider_count), 1)                         AS mean_providers
    ,APPROX_QUANTILES(provider_count, 100)[OFFSET(25)]     AS p25
    ,APPROX_QUANTILES(provider_count, 100)[OFFSET(50)]     AS median_providers
    ,APPROX_QUANTILES(provider_count, 100)[OFFSET(75)]     AS p75
    ,APPROX_QUANTILES(provider_count, 100)[OFFSET(90)]     AS p90
FROM (
    SELECT
        member_id
        ,COUNT(DISTINCT srv_prvdr_id)                      AS provider_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY member_id
)
;

-- ------------------------------------------------------------
-- Q5. Members who see only 1 provider vs many
-- High single-provider % = loyalty, low = care fragmentation
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN provider_count = 1             THEN '01_single_provider'
        WHEN provider_count BETWEEN 2 AND 3 THEN '02_2_to_3'
        WHEN provider_count BETWEEN 4 AND 5 THEN '03_4_to_5'
        WHEN provider_count BETWEEN 6 AND 10 THEN '04_6_to_10'
        WHEN provider_count > 10            THEN '05_10_plus'
    END                                                    AS provider_bucket
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct
FROM (
    SELECT
        member_id
        ,COUNT(DISTINCT srv_prvdr_id)                      AS provider_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY member_id
)
GROUP BY provider_bucket
ORDER BY provider_bucket
;

-- ============================================================
-- SECTION 2: Provider self-loop analysis
-- Among T180 qualified members — how often do consecutive
-- visits go to the same provider?
-- ============================================================

-- ------------------------------------------------------------
-- Q6. Overall provider self-loop rate
-- Compare with specialty self-loop rate from Chapter 6
-- ------------------------------------------------------------
WITH provider_transitions AS (
    SELECT
        v.member_id
        ,v.srv_prvdr_id                                    AS current_provider
        ,v.specialty_ctg_cd                                AS current_specialty
        ,LAG(v.srv_prvdr_id) OVER (
            PARTITION BY v.member_id ORDER BY v.visit_date
        )                                                  AS prev_provider
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
    -- Provider level
    ,COUNTIF(current_provider = prev_provider)             AS provider_self_loops
    ,ROUND(100.0 * COUNTIF(current_provider = prev_provider)
        / COUNT(*), 2)                                     AS provider_self_loop_pct
    -- Specialty level for comparison
    ,COUNTIF(current_specialty = prev_specialty)           AS specialty_self_loops
    ,ROUND(100.0 * COUNTIF(current_specialty = prev_specialty)
        / COUNT(*), 2)                                     AS specialty_self_loop_pct
    -- Same provider AND same specialty
    ,COUNTIF(current_provider = prev_provider
        AND current_specialty = prev_specialty)            AS both_same
    ,ROUND(100.0 * COUNTIF(current_provider = prev_provider
        AND current_specialty = prev_specialty)
        / COUNT(*), 2)                                     AS both_same_pct
    -- Same specialty but different provider
    ,COUNTIF(current_provider != prev_provider
        AND current_specialty = prev_specialty)            AS diff_provider_same_specialty
    ,ROUND(100.0 * COUNTIF(current_provider != prev_provider
        AND current_specialty = prev_specialty)
        / COUNT(*), 2)                                     AS diff_provider_same_specialty_pct
FROM provider_transitions
WHERE prev_provider IS NOT NULL
;

-- ------------------------------------------------------------
-- Q7. Provider self-loop rate by specialty
-- Which specialties have highest provider loyalty?
-- PCP likely highest — specialist referrals likely lowest
-- ------------------------------------------------------------
WITH provider_transitions AS (
    SELECT
        v.member_id
        ,v.srv_prvdr_id                                    AS current_provider
        ,v.specialty_ctg_cd                                AS current_specialty
        ,LAG(v.srv_prvdr_id) OVER (
            PARTITION BY v.member_id ORDER BY v.visit_date
        )                                                  AS prev_provider
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
    ,COUNTIF(current_provider = prev_provider
        AND current_specialty = prev_specialty)            AS same_provider_same_specialty
    ,COUNTIF(current_provider != prev_provider
        AND current_specialty = prev_specialty)            AS diff_provider_same_specialty
    ,ROUND(100.0 * COUNTIF(current_provider = prev_provider
        AND current_specialty = prev_specialty)
        / NULLIF(COUNTIF(current_specialty = prev_specialty), 0)
        , 2)                                               AS provider_loyalty_within_specialty_pct
FROM provider_transitions
WHERE prev_provider IS NOT NULL
  AND prev_specialty IS NOT NULL
GROUP BY current_specialty
ORDER BY provider_loyalty_within_specialty_pct DESC
LIMIT 20
;

-- ------------------------------------------------------------
-- Q8. Provider self-loop rate by sequence length bucket
-- Do members with longer histories show higher provider loyalty?
-- ------------------------------------------------------------
WITH member_visit_counts AS (
    SELECT
        member_id
        ,COUNT(DISTINCT visit_date)                        AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    GROUP BY member_id
),
provider_transitions AS (
    SELECT
        v.member_id
        ,v.srv_prvdr_id                                    AS current_provider
        ,LAG(v.srv_prvdr_id) OVER (
            PARTITION BY v.member_id ORDER BY v.visit_date
        )                                                  AS prev_provider
        ,vc.visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    INNER JOIN (
        SELECT DISTINCT member_id
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
        WHERE is_t180_qualified = TRUE
    ) q ON v.member_id = q.member_id
    JOIN member_visit_counts vc ON v.member_id = vc.member_id
)
SELECT
    CASE
        WHEN visit_count BETWEEN 1 AND 3    THEN '01_1_to_3'
        WHEN visit_count BETWEEN 4 AND 5    THEN '02_4_to_5'
        WHEN visit_count BETWEEN 6 AND 10   THEN '03_6_to_10'
        WHEN visit_count BETWEEN 11 AND 20  THEN '04_11_to_20'
        WHEN visit_count > 20               THEN '05_20_plus'
    END                                                    AS sequence_bucket
    ,COUNT(*)                                              AS total_transitions
    ,ROUND(100.0 * COUNTIF(current_provider = prev_provider)
        / COUNT(*), 2)                                     AS provider_self_loop_pct
FROM provider_transitions
WHERE prev_provider IS NOT NULL
GROUP BY sequence_bucket
ORDER BY sequence_bucket
;
