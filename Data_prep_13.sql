-- ============================================================
-- MARKOV BASELINE — TRAIN/TEST SPLIT
-- TRAIN: triggers before 2024
-- TEST: triggers 2024 onwards
-- ============================================================

-- STEP 1 — TRAIN: build transition probabilities from pre-2024 triggers
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_train`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH train_data AS (
    SELECT *
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_summary`
    WHERE trigger_date < '2024-01-01'
      AND transition_count >= 100
      AND conditional_entropy IS NOT NULL
),
pair_totals AS (
    SELECT
        trigger_dx
        ,v2_dx
        ,member_segment
        ,time_window
        ,SUM(transition_count)                           AS pair_total
    FROM train_data
    GROUP BY trigger_dx, v2_dx, member_segment, time_window
)
SELECT
    t.trigger_dx
    ,t.trigger_dx_desc
    ,t.v2_dx
    ,t.v2_dx_desc
    ,t.next_specialty
    ,t.next_specialty_desc
    ,t.member_segment
    ,t.time_window
    ,t.transition_count
    ,p.pair_total
    ,ROUND(t.transition_count / p.pair_total, 4)         AS train_probability
    ,ROW_NUMBER() OVER (
        PARTITION BY t.trigger_dx, t.v2_dx, t.member_segment, t.time_window
        ORDER BY t.transition_count DESC
    )                                                    AS specialty_rank
FROM train_data t
JOIN pair_totals p
    ON t.trigger_dx = p.trigger_dx
    AND t.v2_dx = p.v2_dx
    AND t.member_segment = p.member_segment
    AND t.time_window = p.time_window;


-- STEP 2 — TEST: actual next specialty for 2024+ triggers
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_test`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    member_id
    ,trigger_date
    ,trigger_dx
    ,trigger_dx_desc
    ,v2_dx
    ,v2_dx_desc
    ,next_specialty                                      AS actual_specialty
    ,member_segment
    ,time_window
    ,median_days_to_specialty
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_summary`
WHERE trigger_date >= '2024-01-01'
  AND transition_count >= 1
  AND next_specialty IS NOT NULL;


-- STEP 3 — JOIN TEST TO TRAIN PREDICTIONS (top 5 per pair)
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_predictions`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.v2_dx
    ,t.actual_specialty
    ,t.member_segment
    ,t.time_window
    ,p.next_specialty                                    AS predicted_specialty
    ,p.train_probability
    ,p.specialty_rank
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_test` t
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_train` p
    ON t.trigger_dx = p.trigger_dx
    AND t.v2_dx = p.v2_dx
    AND t.member_segment = p.member_segment
    AND t.time_window = p.time_window
WHERE p.specialty_rank <= 5
