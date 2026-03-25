-- ============================================================
-- QUERY A — FIND THE 80% CUMULATIVE VOLUME CUTOFF
-- Rank providers by claim volume, find where cumulative hits 80%
-- ============================================================

WITH provider_volume AS (
    SELECT
        srv_prvdr_id
        ,COUNT(*)                                            AS claim_count
        ,COUNT(DISTINCT member_id)                           AS member_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE srv_prvdr_id IS NOT NULL
    GROUP BY srv_prvdr_id
),
ranked AS (
    SELECT
        srv_prvdr_id
        ,claim_count
        ,member_count
        ,SUM(claim_count) OVER (ORDER BY claim_count DESC)   AS cumulative_claims
        ,SUM(claim_count) OVER ()                            AS total_claims
        ,ROW_NUMBER() OVER (ORDER BY claim_count DESC)       AS provider_rank
        ,COUNT(*) OVER ()                                    AS total_providers
    FROM provider_volume
)
SELECT
    provider_rank                                            AS nth_provider
    ,total_providers
    ,claim_count                                             AS claims_at_cutoff
    ,member_count                                            AS members_at_cutoff
    ,cumulative_claims
    ,total_claims
    ,ROUND(cumulative_claims * 100.0 / total_claims, 2)      AS cumulative_pct
FROM ranked
WHERE cumulative_claims >= total_claims * 0.75
  AND cumulative_claims <= total_claims * 0.85
ORDER BY provider_rank
LIMIT 20
;


-- ============================================================
-- QUERY B — PROVIDER STATS AT COMMON CUTOFFS
-- Summary row per cumulative % threshold
-- ============================================================

WITH provider_volume AS (
    SELECT
        srv_prvdr_id
        ,COUNT(*)                                            AS claim_count
        ,COUNT(DISTINCT member_id)                           AS member_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE srv_prvdr_id IS NOT NULL
    GROUP BY srv_prvdr_id
),
ranked AS (
    SELECT
        srv_prvdr_id
        ,claim_count
        ,member_count
        ,SUM(claim_count) OVER (ORDER BY claim_count DESC)   AS cumulative_claims
        ,SUM(claim_count) OVER ()                            AS total_claims
        ,ROW_NUMBER() OVER (ORDER BY claim_count DESC)       AS provider_rank
    FROM provider_volume
)
SELECT
    threshold
    ,COUNT(*)                                                AS n_providers
    ,MIN(r.claim_count)                                      AS min_claims_in_set
    ,ROUND(AVG(r.member_count), 1)                           AS avg_members
    ,MIN(r.member_count)                                     AS min_members_in_set
FROM ranked r
CROSS JOIN UNNEST([0.70, 0.75, 0.80, 0.85, 0.90]) AS threshold
WHERE r.cumulative_claims <= r.total_claims * threshold
GROUP BY threshold
ORDER BY threshold
;


-- ============================================================
-- QUERY C — LABEL EXPLOSION RE-CHECK WITH 80% CUTOFF
-- Same as original Query 2 but only counting providers
-- that fall within the top-80%-volume set
-- ============================================================

WITH provider_volume AS (
    SELECT
        srv_prvdr_id
        ,COUNT(*) AS claim_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE srv_prvdr_id IS NOT NULL
    GROUP BY srv_prvdr_id
),
ranked AS (
    SELECT
        srv_prvdr_id
        ,SUM(claim_count) OVER (ORDER BY claim_count DESC)   AS cumulative_claims
        ,SUM(claim_count) OVER ()                            AS total_claims
    FROM provider_volume
),
top80_providers AS (
    SELECT srv_prvdr_id
    FROM ranked
    WHERE cumulative_claims <= total_claims * 0.80
),

label_counts AS (
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
        ,COUNT(DISTINCT v.srv_prvdr_id)                      AS n_providers_all
        ,COUNT(DISTINCT CASE
            WHEN v.srv_prvdr_id IN (SELECT srv_prvdr_id FROM top80_providers)
            THEN v.srv_prvdr_id
         END)                                                AS n_providers_top80
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
    ,ROUND(AVG(n_providers_all), 1)                          AS avg_providers_all
    ,ROUND(AVG(n_providers_top80), 1)                        AS avg_providers_top80
    -- What % of labels survive the cutoff
    ,ROUND(AVG(n_providers_top80 * 100.0
               / NULLIF(n_providers_all, 0)), 1)             AS avg_pct_labels_retained
    -- Triggers that lose ALL provider labels
    ,COUNTIF(n_providers_top80 = 0)                          AS triggers_with_zero_labels
    ,ROUND(COUNTIF(n_providers_top80 = 0) * 100.0
           / COUNT(*), 2)                                    AS pct_triggers_zero_labels
    ,APPROX_QUANTILES(n_providers_top80, 100)[OFFSET(50)]    AS p50_providers_top80
    ,APPROX_QUANTILES(n_providers_top80, 100)[OFFSET(95)]    AS p95_providers_top80
    ,APPROX_QUANTILES(n_providers_top80, 100)[OFFSET(99)]    AS p99_providers_top80
FROM label_counts
GROUP BY time_bucket
ORDER BY time_bucket
;


-- ============================================================
-- QUERY D — SEQUENCE TOKEN CHECK WITH 80% CUTOFF
-- Input sequences: all providers stay (rare ones map to UNK)
-- But how many tokens per trigger are top80 vs UNK?
-- ============================================================

WITH provider_volume AS (
    SELECT
        srv_prvdr_id
        ,COUNT(*) AS claim_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE srv_prvdr_id IS NOT NULL
    GROUP BY srv_prvdr_id
),
ranked AS (
    SELECT
        srv_prvdr_id
        ,SUM(claim_count) OVER (ORDER BY claim_count DESC)   AS cumulative_claims
        ,SUM(claim_count) OVER ()                            AS total_claims
    FROM provider_volume
),
top80_providers AS (
    SELECT srv_prvdr_id
    FROM ranked
    WHERE cumulative_claims <= total_claims * 0.80
),

triggers AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    WHERE t.is_left_qualified = TRUE
      AND t.has_claims_12m_before = TRUE
),

token_breakdown AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,COUNT(DISTINCT CONCAT(CAST(v.visit_date AS STRING), '|', v.srv_prvdr_id))
                                                             AS total_tokens
        ,COUNT(DISTINCT CASE
            WHEN v.srv_prvdr_id IN (SELECT srv_prvdr_id FROM top80_providers)
            THEN CONCAT(CAST(v.visit_date AS STRING), '|', v.srv_prvdr_id)
         END)                                                AS known_tokens
        ,COUNT(DISTINCT CASE
            WHEN v.srv_prvdr_id NOT IN (SELECT srv_prvdr_id FROM top80_providers)
            THEN CONCAT(CAST(v.visit_date AS STRING), '|', v.srv_prvdr_id)
         END)                                                AS unk_tokens
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
    ,ROUND(AVG(total_tokens), 1)                             AS avg_total_tokens
    ,ROUND(AVG(known_tokens), 1)                             AS avg_known_tokens
    ,ROUND(AVG(unk_tokens), 1)                               AS avg_unk_tokens
    ,ROUND(AVG(known_tokens * 100.0
               / NULLIF(total_tokens, 0)), 1)                AS avg_pct_known
    ,ROUND(AVG(unk_tokens * 100.0
               / NULLIF(total_tokens, 0)), 1)                AS avg_pct_unk
    -- Triggers where ALL tokens would be UNK
    ,COUNTIF(known_tokens = 0)                               AS triggers_all_unk
    ,ROUND(COUNTIF(known_tokens = 0) * 100.0
           / COUNT(*), 2)                                    AS pct_triggers_all_unk
FROM token_breakdown
;
