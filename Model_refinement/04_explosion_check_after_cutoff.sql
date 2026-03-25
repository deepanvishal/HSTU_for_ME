-- ============================================================
-- QUERY C — LABEL EXPLOSION CHECK
-- % of label providers surviving transition-based 80% cutoff
-- % of triggers losing ALL labels per window
-- ============================================================

WITH label_counts AS (
    SELECT
        v.member_id
        ,v.trigger_date
        ,v.trigger_dx
        ,CASE
            WHEN v.days_since_trigger <= 30  THEN 'T0_30'
            WHEN v.days_since_trigger <= 60  THEN 'T30_60'
            WHEN v.days_since_trigger <= 180 THEN 'T60_180'
         END                                             AS time_bucket
        ,COUNT(DISTINCT v.srv_prvdr_id)                  AS n_providers_all
        ,COUNT(DISTINCT CASE
            WHEN p.is_top80 = TRUE
            THEN v.srv_prvdr_id
         END)                                            AS n_providers_top80
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_vocab` p
        ON v.srv_prvdr_id = p.srv_prvdr_id
    WHERE v.is_left_qualified = TRUE
      AND v.is_v2 = FALSE
      AND v.srv_prvdr_id IS NOT NULL
      AND v.days_since_trigger <= 180
      AND (
          (v.days_since_trigger <= 30  AND v.is_t30_qualified  = TRUE)
       OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30 AND v.is_t60_qualified  = TRUE)
       OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60 AND v.is_t180_qualified = TRUE)
      )
    GROUP BY v.member_id, v.trigger_date, v.trigger_dx, time_bucket
)

SELECT
    time_bucket
    ,COUNT(*)                                            AS trigger_count
    ,ROUND(AVG(n_providers_all), 1)                      AS avg_providers_all
    ,ROUND(AVG(n_providers_top80), 1)                    AS avg_providers_top80
    ,ROUND(AVG(n_providers_top80 * 100.0
        / NULLIF(n_providers_all, 0)), 1)                AS avg_pct_labels_retained
    ,COUNTIF(n_providers_top80 = 0)                      AS triggers_zero_labels
    ,ROUND(COUNTIF(n_providers_top80 = 0) * 100.0
        / COUNT(*), 2)                                   AS pct_triggers_zero_labels
FROM label_counts
GROUP BY time_bucket
ORDER BY time_bucket
;


-- ============================================================
-- QUERY D — SEQUENCE TOKEN CHECK
-- % of input tokens known vs UNK per trigger
-- ============================================================

WITH triggers AS (
    SELECT DISTINCT
        member_id
        ,trigger_date
        ,trigger_dx
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
    WHERE is_left_qualified = TRUE
      AND has_claims_12m_before = TRUE
),

token_breakdown AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,COUNT(DISTINCT CONCAT(CAST(v.visit_date AS STRING), '|', v.srv_prvdr_id))
                                                         AS total_tokens
        ,COUNT(DISTINCT CASE
            WHEN p.is_top80 = TRUE
            THEN CONCAT(CAST(v.visit_date AS STRING), '|', v.srv_prvdr_id)
         END)                                            AS known_tokens
    FROM triggers t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date < t.trigger_date
        AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
    LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_vocab` p
        ON v.srv_prvdr_id = p.srv_prvdr_id
    WHERE v.srv_prvdr_id IS NOT NULL
    GROUP BY t.member_id, t.trigger_date, t.trigger_dx
)

SELECT
    COUNT(*)                                             AS total_triggers
    ,ROUND(AVG(total_tokens), 1)                         AS avg_total_tokens
    ,ROUND(AVG(known_tokens), 1)                         AS avg_known_tokens
    ,ROUND(AVG(known_tokens * 100.0
        / NULLIF(total_tokens, 0)), 1)                   AS avg_pct_known
    ,COUNTIF(known_tokens = 0)                           AS triggers_all_unk
    ,ROUND(COUNTIF(known_tokens = 0) * 100.0
        / COUNT(*), 2)                                   AS pct_triggers_all_unk
FROM token_breakdown
;
