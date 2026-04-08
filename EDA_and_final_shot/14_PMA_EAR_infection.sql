-- ============================================================
-- EDA_h6691_transitions.sql
-- Diagnosis : H66.91 — Otitis media unspecified, right ear
-- Source of TRUE transitions : A870800_gen_rec_visits_qualified
--   is_v2 = TRUE            → immediate next visit after trigger
--   is_left_qualified = TRUE → 12m enrollment + dx not seen
--   is_t30_qualified  = TRUE → right boundary active
--   trigger_dx = 'H66.91'
--   Columns used:
--     member_id, trigger_date, trigger_dx, member_segment
--     specialty_ctg_cd  → true next visit specialty
-- Model correctness:
--   Join on member_id + CAST(trigger_date AS STRING) + trigger_dx
--   Check specialty_ctg_cd IN SPLIT(top5_predictions,'|')
-- Descriptions from A870800_gen_rec_markov_train
-- ============================================================

-- ------------------------------------------------------------
-- TABLE 1: True transitions + Markov correctness
-- ------------------------------------------------------------
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
true_transitions AS (
    -- One row per actual V2 visit after H66.91 trigger
    SELECT
        v.member_id
        ,v.trigger_date
        ,v.trigger_dx
        ,v.member_segment
        ,v.specialty_ctg_cd                                AS true_next_specialty
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    WHERE v.trigger_dx       = 'H66.91'
      AND v.is_v2            = TRUE
      AND v.is_left_qualified = TRUE
      AND v.is_t30_qualified  = TRUE
      AND v.specialty_ctg_cd IS NOT NULL
),
markov_preds AS (
    SELECT
        member_id
        ,trigger_date                                      AS trigger_date_str
        ,trigger_dx
        ,top5_predictions
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_trigger_scores`
    WHERE trigger_dx  = 'H66.91'
      AND time_bucket = 'T0_30'
),
joined AS (
    SELECT
        t.member_segment
        ,t.trigger_dx
        ,t.true_next_specialty
        ,CASE
            WHEN m.top5_predictions IS NULL                THEN NULL
            WHEN t.true_next_specialty
                IN UNNEST(SPLIT(m.top5_predictions, '|')) THEN 1
            ELSE 0
         END                                               AS markov_correct
    FROM true_transitions t
    LEFT JOIN markov_preds m
        ON  t.member_id                        = m.member_id
        AND CAST(t.trigger_date AS STRING)     = m.trigger_date_str
        AND t.trigger_dx                       = m.trigger_dx
),
ranked AS (
    SELECT
        member_segment
        ,trigger_dx
        ,true_next_specialty
        ,COUNT(*)                                          AS total_transitions
        ,COUNTIF(markov_correct = 1)                       AS markov_correct
        ,COUNTIF(markov_correct = 0)                       AS markov_missed
        ,COUNTIF(markov_correct IS NULL)                   AS no_prediction
        ,ROUND(100.0 * COUNTIF(markov_correct = 1)
            / NULLIF(COUNTIF(markov_correct IS NOT NULL), 0)
            , 2)                                           AS markov_accuracy_pct
        ,ROW_NUMBER() OVER (
            PARTITION BY member_segment
            ORDER BY COUNT(*) DESC
        )                                                  AS rank_within_segment
    FROM joined
    GROUP BY member_segment, trigger_dx, true_next_specialty
)
SELECT
    r.member_segment
    ,r.rank_within_segment                                 AS rank
    ,r.trigger_dx
    ,dx.trigger_dx_desc
    ,r.true_next_specialty
    ,sp.next_specialty_desc
    ,r.total_transitions
    ,r.markov_correct
    ,r.markov_missed
    ,r.no_prediction
    ,r.markov_accuracy_pct
FROM ranked r
LEFT JOIN dx_desc dx        ON r.trigger_dx         = dx.trigger_dx
LEFT JOIN specialty_desc sp ON r.true_next_specialty = sp.next_specialty
WHERE r.rank_within_segment <= 5
ORDER BY r.member_segment, r.rank_within_segment
;

-- ------------------------------------------------------------
-- TABLE 2: True transitions + BERT4Rec correctness
-- ------------------------------------------------------------
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
true_transitions AS (
    SELECT
        v.member_id
        ,v.trigger_date
        ,v.trigger_dx
        ,v.member_segment
        ,v.specialty_ctg_cd                                AS true_next_specialty
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    WHERE v.trigger_dx       = 'H66.91'
      AND v.is_v2            = TRUE
      AND v.is_left_qualified = TRUE
      AND v.is_t30_qualified  = TRUE
      AND v.specialty_ctg_cd IS NOT NULL
),
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
joined AS (
    SELECT
        t.member_segment
        ,t.trigger_dx
        ,t.true_next_specialty
        ,CASE
            WHEN b.top5_predictions IS NULL                THEN NULL
            WHEN t.true_next_specialty
                IN UNNEST(SPLIT(b.top5_predictions, '|')) THEN 1
            ELSE 0
         END                                               AS bert_correct
    FROM true_transitions t
    LEFT JOIN bert_preds b
        ON  t.member_id                    = b.member_id
        AND CAST(t.trigger_date AS STRING) = b.trigger_date_str
        AND t.trigger_dx                   = b.trigger_dx
),
ranked AS (
    SELECT
        member_segment
        ,trigger_dx
        ,true_next_specialty
        ,COUNT(*)                                          AS total_transitions
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
    GROUP BY member_segment, trigger_dx, true_next_specialty
)
SELECT
    r.member_segment
    ,r.rank_within_segment                                 AS rank
    ,r.trigger_dx
    ,dx.trigger_dx_desc
    ,r.true_next_specialty
    ,sp.next_specialty_desc
    ,r.total_transitions
    ,r.bert_correct
    ,r.bert_missed
    ,r.no_prediction
    ,r.bert_accuracy_pct
FROM ranked r
LEFT JOIN dx_desc dx        ON r.trigger_dx         = dx.trigger_dx
LEFT JOIN specialty_desc sp ON r.true_next_specialty = sp.next_specialty
WHERE r.rank_within_segment <= 5
ORDER BY r.member_segment, r.rank_within_segment
;
