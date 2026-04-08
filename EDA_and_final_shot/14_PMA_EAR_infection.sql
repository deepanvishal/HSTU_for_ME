-- ============================================================
-- EDA_h6691_transitions.sql
-- Diagnosis : H66.91 — Otitis media unspecified, right ear
-- Two tables:
--   Table 1 — Ground truth transitions + Markov correctness
--   Table 2 — Ground truth transitions + BERT4Rec correctness
-- Both: top 5 transitions per member_segment with descriptions
-- Descriptions from A870800_gen_rec_markov_train:
--   trigger_dx     → trigger_ccsr_desc
--   next_specialty → next_specialty_desc
-- ============================================================

-- ------------------------------------------------------------
-- TABLE 1: Ground truth transitions + Markov prediction accuracy
-- ------------------------------------------------------------
WITH dx_desc AS (
    -- Trigger dx description
    SELECT DISTINCT
        trigger_dx
        ,trigger_ccsr_desc                                 AS trigger_dx_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE trigger_dx = 'H66.91'
),
specialty_desc AS (
    -- Next specialty description
    SELECT DISTINCT
        next_specialty
        ,next_specialty_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE next_specialty IS NOT NULL
),
test_triggers AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.label_specialty                                 AS true_next_specialty
        ,t.time_bucket
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test` t
    WHERE t.trigger_dx     = 'H66.91'
      AND t.label_specialty IS NOT NULL
      AND t.time_bucket    = 'T0_30'
),
markov_preds AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,time_bucket
        ,top5_predictions                                  AS predicted_pipe
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
            WHEN t.true_next_specialty IN UNNEST(SPLIT(m.predicted_pipe, '|'))
            THEN 1 ELSE 0
         END                                               AS markov_correct
    FROM test_triggers t
    LEFT JOIN markov_preds m
        ON  t.member_id    = m.member_id
        AND t.trigger_date = m.trigger_date
        AND t.trigger_dx   = m.trigger_dx
),
transition_counts AS (
    SELECT
        member_segment
        ,trigger_dx
        ,true_next_specialty
        ,COUNT(*)                                          AS total_transitions
        ,SUM(markov_correct)                               AS markov_predicted_correct
        ,ROUND(100.0 * SUM(markov_correct)
            / COUNT(*), 2)                                 AS markov_accuracy_pct
        ,ROW_NUMBER() OVER (
            PARTITION BY member_segment
            ORDER BY COUNT(*) DESC
        )                                                  AS rank_within_segment
    FROM joined
    GROUP BY member_segment, trigger_dx, true_next_specialty
)
SELECT
    tc.member_segment
    ,tc.rank_within_segment                                AS rank
    ,tc.trigger_dx
    ,dx.trigger_dx_desc
    ,tc.true_next_specialty
    ,sp.next_specialty_desc
    ,tc.total_transitions
    ,tc.markov_predicted_correct
    ,tc.markov_accuracy_pct
FROM transition_counts tc
LEFT JOIN dx_desc dx        ON tc.trigger_dx         = dx.trigger_dx
LEFT JOIN specialty_desc sp ON tc.true_next_specialty = sp.next_specialty
WHERE tc.rank_within_segment <= 5
ORDER BY tc.member_segment, tc.rank_within_segment
;

-- ------------------------------------------------------------
-- TABLE 2: Ground truth transitions + BERT4Rec prediction accuracy
-- ------------------------------------------------------------
WITH dx_desc AS (
    SELECT DISTINCT
        trigger_dx
        ,trigger_ccsr_desc                                 AS trigger_dx_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE trigger_dx = 'H66.91'
),
specialty_desc AS (
    SELECT DISTINCT
        next_specialty
        ,next_specialty_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
    WHERE next_specialty IS NOT NULL
),
test_triggers AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.label_specialty                                 AS true_next_specialty
        ,t.time_bucket
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test` t
    WHERE t.trigger_dx     = 'H66.91'
      AND t.label_specialty IS NOT NULL
      AND t.time_bucket    = 'T0_30'
),
bert_preds AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,time_bucket
        ,top5_predictions                                  AS predicted_pipe
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
            WHEN t.true_next_specialty IN UNNEST(SPLIT(b.predicted_pipe, '|'))
            THEN 1 ELSE 0
         END                                               AS bert_correct
    FROM test_triggers t
    LEFT JOIN bert_preds b
        ON  t.member_id    = b.member_id
        AND t.trigger_date = b.trigger_date
        AND t.trigger_dx   = b.trigger_dx
),
transition_counts AS (
    SELECT
        member_segment
        ,trigger_dx
        ,true_next_specialty
        ,COUNT(*)                                          AS total_transitions
        ,SUM(bert_correct)                                 AS bert_predicted_correct
        ,ROUND(100.0 * SUM(bert_correct)
            / COUNT(*), 2)                                 AS bert_accuracy_pct
        ,ROW_NUMBER() OVER (
            PARTITION BY member_segment
            ORDER BY COUNT(*) DESC
        )                                                  AS rank_within_segment
    FROM joined
    GROUP BY member_segment, trigger_dx, true_next_specialty
)
SELECT
    tc.member_segment
    ,tc.rank_within_segment                                AS rank
    ,tc.trigger_dx
    ,dx.trigger_dx_desc
    ,tc.true_next_specialty
    ,sp.next_specialty_desc
    ,tc.total_transitions
    ,tc.bert_predicted_correct
    ,tc.bert_accuracy_pct
FROM transition_counts tc
LEFT JOIN dx_desc dx        ON tc.trigger_dx         = dx.trigger_dx
LEFT JOIN specialty_desc sp ON tc.true_next_specialty = sp.next_specialty
WHERE tc.rank_within_segment <= 5
ORDER BY tc.member_segment, tc.rank_within_segment
;
