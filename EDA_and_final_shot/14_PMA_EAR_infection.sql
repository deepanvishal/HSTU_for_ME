-- ============================================================
-- EDA_h6691_transitions.sql
-- Diagnosis : H66.91 — Otitis media unspecified, right ear
-- Logic:
--   Step 1 — Qualified triggers: left + right boundary both TRUE
--   Step 2 — Hit visits table for all visits AFTER trigger date
--             (any visit, any date, any specialty)
--   Step 3 — Count transitions per member_segment + specialty
--   Step 4 — Check if BERT4Rec predicted that specialty in top 5
--   Step 5 — Top 5 transitions per member_segment
-- ============================================================

WITH specialty_desc AS (
    SELECT DISTINCT next_specialty, next_specialty_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE next_specialty IS NOT NULL
),
dx_desc AS (
    SELECT DISTINCT trigger_dx, trigger_ccsr_desc AS trigger_dx_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE trigger_dx = 'H66.91'
),
-- Step 1: Qualified triggers — left AND right boundary active
qualified_triggers AS (
    SELECT DISTINCT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
    WHERE trigger_dx       = 'H66.91'
      AND is_left_qualified = TRUE
      AND is_t180_qualified = TRUE
),
-- Step 2: Immediate next visit date after trigger
-- Find MIN(visit_date) > trigger_date, then get all visits on that date
following_visits AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,v.visit_date
        ,v.specialty_ctg_cd
    FROM qualified_triggers t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON  t.member_id  = v.member_id
        AND v.visit_date = (
            SELECT MIN(v2.visit_date)
            FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v2
            WHERE v2.member_id  = t.member_id
              AND v2.visit_date > t.trigger_date
        )
        AND v.specialty_ctg_cd IS NOT NULL
),
-- Step 3: BERT4Rec predictions for H66.91 triggers
bert_preds AS (
    SELECT
        member_id
        ,trigger_date                                      AS trigger_date_str
        ,trigger_dx
        ,top5_predictions
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_trigger_scores`
    WHERE model       = 'BERT4Rec'
      AND trigger_dx  = 'H66.91'
      AND time_bucket = 'T0_30'
),
-- Step 4: Join predictions — check if specialty was in top 5
joined AS (
    SELECT
        f.member_segment
        ,f.trigger_dx
        ,f.specialty_ctg_cd                                AS following_specialty
        ,f.days_since_trigger
        ,f.plc_srv_cd
        ,CASE
            WHEN b.top5_predictions IS NULL                THEN NULL
            WHEN f.specialty_ctg_cd
                IN UNNEST(SPLIT(b.top5_predictions, '|')) THEN 1
            ELSE 0
         END                                               AS bert_correct
    FROM following_visits f
    LEFT JOIN bert_preds b
        ON  f.member_id                    = b.member_id
        AND CAST(f.trigger_date AS STRING) = b.trigger_date_str
        AND f.trigger_dx                   = b.trigger_dx
),
-- Step 5: Aggregate + rank top 5 per segment
ranked AS (
    SELECT
        member_segment
        ,trigger_dx
        ,following_specialty
        ,COUNT(*)                                          AS total_following_visits
        ,ROUND(AVG(days_since_trigger), 1)                 AS avg_days_since_trigger
        ,APPROX_QUANTILES(days_since_trigger, 100)[OFFSET(50)]
                                                           AS median_days_since_trigger
        ,COUNTIF(plc_srv_cd = 'I')                         AS inpatient_visits
        ,ROUND(100.0 * COUNTIF(plc_srv_cd = 'I')
            / COUNT(*), 2)                                 AS pct_inpatient
        ,COUNTIF(bert_correct = 1)                         AS bert_correct
        ,COUNTIF(bert_correct = 0)                         AS bert_missed
        ,COUNTIF(bert_correct IS NULL)                     AS no_prediction
        ,ROUND(100.0 * COUNTIF(bert_correct = 1)
            / NULLIF(COUNTIF(bert_correct IS NOT NULL), 0)
            , 2)                                           AS bert_accuracy_pct
        ,ROW_NUMBER() OVER (
            PARTITION BY member_segment
            ORDER BY COUNT(*) DESC
        )                                                  AS rank_within_segment
    FROM joined
    GROUP BY member_segment, trigger_dx, following_specialty
)
SELECT
    r.member_segment
    ,r.rank_within_segment                                 AS rank
    ,r.trigger_dx
    ,dx.trigger_dx_desc
    ,r.following_specialty
    ,sp.next_specialty_desc                                AS following_specialty_desc
    ,r.total_following_visits
    ,r.avg_days_since_trigger
    ,r.median_days_since_trigger
    ,r.inpatient_visits
    ,r.pct_inpatient
    ,r.bert_correct
    ,r.bert_missed
    ,r.no_prediction
    ,r.bert_accuracy_pct
FROM ranked r
LEFT JOIN dx_desc dx        ON r.trigger_dx          = dx.trigger_dx
LEFT JOIN specialty_desc sp ON r.following_specialty  = sp.next_specialty
WHERE r.rank_within_segment <= 5
ORDER BY r.member_segment, r.rank_within_segment
;
