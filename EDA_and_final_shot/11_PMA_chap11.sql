-- ============================================================
-- NB_PMA_02_pos_distribution.sql
-- Purpose : Inpatient vs Outpatient distribution
--           for top 10 predictable specialties
-- Primary metric : Hit@5, T0_30, BERT4Rec
-- Source  : A870800_gen_rec_analysis_perf_by_ending_specialty
--           A870800_gen_rec_visits_qualified
--           A870800_gen_rec_markov_train (for descriptions)
-- ============================================================

-- ============================================================
-- Q1. Top 10 specialties — full POS breakdown
--     One row per specialty × plc_srv_cd
--     Deduped: one record per member + visit_date + specialty
-- ============================================================
WITH specialty_desc AS (
    SELECT DISTINCT next_specialty, next_specialty_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE next_specialty IS NOT NULL
),
top10 AS (
    SELECT
        ending_specialty
        ,ROUND(hit_rate_at_5, 4)                           AS hit_rate_at_5
        ,total_appearances
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_ending_specialty`
    WHERE model        = 'BERT4Rec'
      AND time_bucket  = 'T0_30'
      AND total_appearances >= 20
    ORDER BY hit_rate_at_5 DESC
    LIMIT 10
),
deduped_visits AS (
    -- One row per member + visit_date + specialty
    -- collapse same-day multi-claim rows
    SELECT
        member_id
        ,visit_date
        ,specialty_ctg_cd
        ,MAX(plc_srv_cd)                                   AS plc_srv_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
    WHERE is_left_qualified = TRUE
      AND (is_t30_qualified = TRUE OR is_t60_qualified = TRUE OR is_t180_qualified = TRUE)
      AND specialty_ctg_cd IS NOT NULL
    GROUP BY member_id, visit_date, specialty_ctg_cd
),
pos_detail AS (
    SELECT
        v.specialty_ctg_cd                                 AS ending_specialty
        ,v.plc_srv_cd
        ,COUNT(*)                                          AS visit_count
        ,SUM(COUNT(*)) OVER (
            PARTITION BY v.specialty_ctg_cd
        )                                                  AS total_for_specialty
    FROM deduped_visits v
    JOIN top10 t ON v.specialty_ctg_cd = t.ending_specialty
    GROUP BY v.specialty_ctg_cd, v.plc_srv_cd
)
SELECT
    p.ending_specialty
    ,d.next_specialty_desc                                 AS specialty_desc
    ,t.hit_rate_at_5
    ,t.total_appearances
    ,p.plc_srv_cd
    ,p.visit_count
    ,p.total_for_specialty
    ,ROUND(100.0 * p.visit_count / p.total_for_specialty, 2) AS pct_within_specialty
FROM pos_detail p
JOIN top10 t     ON p.ending_specialty = t.ending_specialty
LEFT JOIN specialty_desc d ON p.ending_specialty = d.next_specialty
ORDER BY t.hit_rate_at_5 DESC, p.visit_count DESC
;

-- ============================================================
-- Q2. Inpatient vs Outpatient binary split
--     One row per specialty — clean I vs non-I summary
-- ============================================================
WITH specialty_desc AS (
    SELECT DISTINCT next_specialty, next_specialty_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE next_specialty IS NOT NULL
),
top10 AS (
    SELECT
        ending_specialty
        ,ROUND(hit_rate_at_5, 4)                           AS hit_rate_at_5
        ,total_appearances
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_ending_specialty`
    WHERE model        = 'BERT4Rec'
      AND time_bucket  = 'T0_30'
      AND total_appearances >= 20
    ORDER BY hit_rate_at_5 DESC
    LIMIT 10
),
deduped_visits AS (
    SELECT
        member_id
        ,visit_date
        ,specialty_ctg_cd
        ,MAX(plc_srv_cd)                                   AS plc_srv_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
    WHERE is_left_qualified = TRUE
      AND (is_t30_qualified = TRUE OR is_t60_qualified = TRUE OR is_t180_qualified = TRUE)
      AND specialty_ctg_cd IS NOT NULL
    GROUP BY member_id, visit_date, specialty_ctg_cd
)
SELECT
    v.specialty_ctg_cd                                     AS ending_specialty
    ,d.next_specialty_desc                                 AS specialty_desc
    ,t.hit_rate_at_5
    ,t.total_appearances
    ,COUNT(*)                                              AS total_visits
    ,COUNTIF(v.plc_srv_cd = 'I')                           AS inpatient_visits
    ,COUNTIF(v.plc_srv_cd != 'I')                          AS outpatient_visits
    ,ROUND(100.0 * COUNTIF(v.plc_srv_cd = 'I')
        / COUNT(*), 2)                                     AS pct_inpatient
    ,ROUND(100.0 * COUNTIF(v.plc_srv_cd != 'I')
        / COUNT(*), 2)                                     AS pct_outpatient
FROM deduped_visits v
JOIN top10 t ON v.specialty_ctg_cd = t.ending_specialty
LEFT JOIN specialty_desc d ON v.specialty_ctg_cd = d.next_specialty
GROUP BY
    v.specialty_ctg_cd, d.next_specialty_desc
    ,t.hit_rate_at_5, t.total_appearances
ORDER BY t.hit_rate_at_5 DESC
;

-- ============================================================
-- Q3. Inpatient % vs Hit@5 — is higher inpatient % correlated
--     with better or worse predictability?
--     All specialties with >= 20 appearances, not just top 10
-- ============================================================
WITH specialty_desc AS (
    SELECT DISTINCT next_specialty, next_specialty_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE next_specialty IS NOT NULL
),
deduped_visits AS (
    SELECT
        member_id
        ,visit_date
        ,specialty_ctg_cd
        ,MAX(plc_srv_cd)                                   AS plc_srv_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
    WHERE is_left_qualified = TRUE
      AND (is_t30_qualified = TRUE OR is_t60_qualified = TRUE OR is_t180_qualified = TRUE)
      AND specialty_ctg_cd IS NOT NULL
    GROUP BY member_id, visit_date, specialty_ctg_cd
),
pos_summary AS (
    SELECT
        specialty_ctg_cd                                   AS ending_specialty
        ,COUNT(*)                                          AS total_visits
        ,ROUND(100.0 * COUNTIF(plc_srv_cd = 'I')
            / COUNT(*), 2)                                 AS pct_inpatient
    FROM deduped_visits
    GROUP BY specialty_ctg_cd
)
SELECT
    p.ending_specialty
    ,d.next_specialty_desc                                 AS specialty_desc
    ,pos.total_visits
    ,pos.pct_inpatient
    ,ROUND(perf.hit_rate_at_5, 4)                          AS hit_rate_at_5
    ,perf.total_appearances
    ,CASE
        WHEN pos.pct_inpatient = 0          THEN '1_No_Inpatient'
        WHEN pos.pct_inpatient < 25         THEN '2_Low_Inpatient_lt25pct'
        WHEN pos.pct_inpatient BETWEEN 25 AND 50 THEN '3_Medium_25_to_50pct'
        WHEN pos.pct_inpatient > 50         THEN '4_High_Inpatient_gt50pct'
    END                                                    AS inpatient_tier
FROM pos_summary pos
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_ending_specialty` perf
    ON pos.ending_specialty = perf.ending_specialty
    AND perf.model       = 'BERT4Rec'
    AND perf.time_bucket = 'T0_30'
    AND perf.total_appearances >= 20
LEFT JOIN specialty_desc d ON pos.ending_specialty = d.next_specialty
JOIN (
    -- Alias workaround for inpatient_tier in ORDER BY
    SELECT ending_specialty AS p_ending_specialty
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_analysis_perf_by_ending_specialty`
    WHERE model = 'BERT4Rec' AND time_bucket = 'T0_30'
) p ON pos.ending_specialty = p.p_ending_specialty
ORDER BY pos.pct_inpatient DESC
;
