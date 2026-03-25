-- ============================================================
-- QUERY 1 — SEQUENCE TOKEN EXPANSION
-- How many tokens per trigger: provider-level vs specialty-level
-- Current model: one row per (member, date, specialty, dx) from A870800_gen_rec_visits
-- Provider model: one row per (member, date, provider) after dedup
-- ============================================================

WITH triggers AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    WHERE t.is_left_qualified = TRUE
      AND t.has_claims_12m_before = TRUE
),

-- Current: how the specialty sequence table counts tokens
-- Grain = (member, date, specialty) from visits table joined to trigger
specialty_tokens AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,COUNT(*) AS token_count_specialty_raw
        ,COUNT(DISTINCT CONCAT(CAST(v.visit_date AS STRING), '|', v.specialty_ctg_cd))
                                                             AS token_count_specialty_deduped
    FROM triggers t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date < t.trigger_date
        AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
    WHERE v.specialty_ctg_cd IS NOT NULL
      AND v.specialty_ctg_cd != ''
    GROUP BY t.member_id, t.trigger_date, t.trigger_dx
),

-- Provider: proposed grain = (member, date, provider)
provider_tokens AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,COUNT(*) AS token_count_provider_raw
        ,COUNT(DISTINCT CONCAT(CAST(v.visit_date AS STRING), '|', v.srv_prvdr_id))
                                                             AS token_count_provider_deduped
    FROM triggers t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date < t.trigger_date
        AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
    WHERE v.srv_prvdr_id IS NOT NULL
    GROUP BY t.member_id, t.trigger_date, t.trigger_dx
)

SELECT
    COUNT(*)                                                 AS total_triggers
    ,ROUND(AVG(s.token_count_specialty_raw), 1)              AS avg_tokens_specialty_raw
    ,ROUND(AVG(s.token_count_specialty_deduped), 1)          AS avg_tokens_specialty_deduped
    ,ROUND(AVG(p.token_count_provider_raw), 1)               AS avg_tokens_provider_raw
    ,ROUND(AVG(p.token_count_provider_deduped), 1)           AS avg_tokens_provider_deduped
    ,ROUND(APPROX_QUANTILES(s.token_count_specialty_deduped, 100)[OFFSET(50)], 1)
                                                             AS p50_specialty
    ,ROUND(APPROX_QUANTILES(p.token_count_provider_deduped, 100)[OFFSET(50)], 1)
                                                             AS p50_provider
    ,ROUND(APPROX_QUANTILES(s.token_count_specialty_deduped, 100)[OFFSET(95)], 1)
                                                             AS p95_specialty
    ,ROUND(APPROX_QUANTILES(p.token_count_provider_deduped, 100)[OFFSET(95)], 1)
                                                             AS p95_provider
    ,ROUND(APPROX_QUANTILES(s.token_count_specialty_deduped, 100)[OFFSET(99)], 1)
                                                             AS p99_specialty
    ,ROUND(APPROX_QUANTILES(p.token_count_provider_deduped, 100)[OFFSET(99)], 1)
                                                             AS p99_provider
    -- Expansion factor: how much more tokens does provider grain produce
    ,ROUND(AVG(p.token_count_provider_deduped * 1.0
               / NULLIF(s.token_count_specialty_deduped, 0)), 2)
                                                             AS avg_expansion_factor
FROM specialty_tokens s
JOIN provider_tokens p
    ON s.member_id = p.member_id
    AND s.trigger_date = p.trigger_date
    AND s.trigger_dx = p.trigger_dx
;


-- ============================================================
-- QUERY 2 — LABEL EXPANSION
-- How many distinct providers vs specialties per trigger per window
-- Source: A870800_gen_rec_visits_qualified (post-trigger visits)
-- ============================================================

WITH label_counts AS (
    SELECT
        v.member_id
        ,v.trigger_date
        ,v.trigger_dx
        ,CASE
            WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
            WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
            WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
         END                                                 AS time_bucket
        ,COUNT(DISTINCT v.specialty_ctg_cd)                  AS n_specialties
        ,COUNT(DISTINCT v.srv_prvdr_id)                      AS n_providers
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    WHERE v.is_left_qualified = TRUE
      AND v.is_v2 = FALSE
      AND v.specialty_ctg_cd IS NOT NULL
      AND v.srv_prvdr_id IS NOT NULL
      AND v.days_since_trigger <= 180
      AND (
          (v.days_since_trigger <= 30  AND v.is_t30_qualified  = TRUE)
       OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND v.is_t60_qualified  = TRUE)
       OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND v.is_t180_qualified = TRUE)
      )
    GROUP BY v.member_id, v.trigger_date, v.trigger_dx, time_bucket
)

SELECT
    time_bucket
    ,COUNT(*)                                                AS trigger_count
    ,ROUND(AVG(n_specialties), 1)                            AS avg_specialties
    ,ROUND(AVG(n_providers), 1)                              AS avg_providers
    ,ROUND(AVG(n_providers * 1.0 / NULLIF(n_specialties, 0)), 2)
                                                             AS avg_expansion_factor
    ,APPROX_QUANTILES(n_specialties, 100)[OFFSET(50)]        AS p50_specialties
    ,APPROX_QUANTILES(n_providers, 100)[OFFSET(50)]          AS p50_providers
    ,APPROX_QUANTILES(n_specialties, 100)[OFFSET(95)]        AS p95_specialties
    ,APPROX_QUANTILES(n_providers, 100)[OFFSET(95)]          AS p95_providers
    ,APPROX_QUANTILES(n_specialties, 100)[OFFSET(99)]        AS p99_specialties
    ,APPROX_QUANTILES(n_providers, 100)[OFFSET(99)]          AS p99_providers
FROM label_counts
GROUP BY time_bucket
ORDER BY time_bucket
;


-- ============================================================
-- QUERY 3 — PROVIDER FREQUENCY DISTRIBUTION
-- How many members per provider — the long tail
-- Source: A870800_gen_rec_visits (full population, pre-filter)
-- ============================================================

WITH provider_freq AS (
    SELECT
        srv_prvdr_id
        ,COUNT(DISTINCT member_id)                           AS n_members
        ,COUNT(*)                                            AS n_visits
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE srv_prvdr_id IS NOT NULL
    GROUP BY srv_prvdr_id
)

SELECT
    COUNT(*)                                                 AS total_providers
    ,COUNTIF(n_members >= 100)                               AS providers_100plus
    ,COUNTIF(n_members >= 50)                                AS providers_50plus
    ,COUNTIF(n_members >= 20)                                AS providers_20plus
    ,COUNTIF(n_members >= 10)                                AS providers_10plus
    ,COUNTIF(n_members >= 5)                                 AS providers_5plus
    ,COUNTIF(n_members < 5)                                  AS providers_under_5
    ,COUNTIF(n_members = 1)                                  AS providers_single_member
    ,ROUND(AVG(n_members), 1)                                AS avg_members_per_provider
    ,APPROX_QUANTILES(n_members, 100)[OFFSET(50)]            AS p50_members
    ,APPROX_QUANTILES(n_members, 100)[OFFSET(75)]            AS p75_members
    ,APPROX_QUANTILES(n_members, 100)[OFFSET(90)]            AS p90_members
    ,APPROX_QUANTILES(n_members, 100)[OFFSET(95)]            AS p95_members
    ,APPROX_QUANTILES(n_members, 100)[OFFSET(99)]            AS p99_members
    ,MAX(n_members)                                          AS max_members
FROM provider_freq
;
