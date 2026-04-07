-- ============================================================
-- FILE 2 — MODEL TABLES PER SAMPLE SIZE
-- Purpose : Create train, test, markov, and RL tables
--           for 1%, 5%, 10% stratified samples
-- Sources : A870800_gen_rec_triggers_qualified (small)
--           A870800_gen_rec_visits_qualified   (medium)
--           A870800_gen_rec_sample_members_Xpct (tiny)
-- Critical: Does NOT read from model_input_sequences (3TB ARRAY)
--           Labels derived from visits_qualified directly
--           Only columns needed for model training selected
-- ============================================================


-- ══════════════════════════════════════════════════════════════════════════════
-- 1 PCT TABLES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── MODEL TRAIN 1PCT ─────────────────────────────────────────────────────────
-- Source: triggers_qualified + visits_qualified — no 3TB table
-- Labels: downstream specialty visits per trigger per window
-- Columns: keys + model input features + window masks + label only
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_train_1pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_train_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date < DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.specialty_ctg_cd IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );


-- ── MODEL TEST 1PCT ──────────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test_1pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date >= DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.specialty_ctg_cd IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );


-- ── MARKOV TRAIN 1PCT ────────────────────────────────────────────────────────
-- Transition counts only — smallest possible grain
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train_1pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.trigger_dx
    ,t.member_segment
    ,v.specialty_ctg_cd                                  AS next_specialty
    ,COUNT(*)                                            AS transition_count
    ,COUNT(DISTINCT t.member_id)                         AS unique_members
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
    AND v.is_v2 = TRUE
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date < DATE '2024-01-01'
  AND v.specialty_ctg_cd IS NOT NULL
GROUP BY
    t.trigger_dx
    ,t.member_segment
    ,v.specialty_ctg_cd;


-- ── RL LABELS TRAIN 1PCT ─────────────────────────────────────────────────────
-- Keys + reward signals only — provider, specialty, dx, window
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_train_1pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_train_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider_id
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,v.dx_raw                                            AS label_dx
    ,v.days_since_trigger
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date < DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.srv_prvdr_id IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );


-- ── RL LABELS TEST 1PCT ──────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_test_1pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_test_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider_id
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,v.dx_raw                                            AS label_dx
    ,v.days_since_trigger
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date >= DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.srv_prvdr_id IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );


-- ══════════════════════════════════════════════════════════════════════════════
-- 5 PCT TABLES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── MODEL TRAIN 5PCT ─────────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_train_5pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_train_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date < DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.specialty_ctg_cd IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );


-- ── MODEL TEST 5PCT ──────────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test_5pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date >= DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.specialty_ctg_cd IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );


-- ── MARKOV TRAIN 5PCT ────────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train_5pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.trigger_dx
    ,t.member_segment
    ,v.specialty_ctg_cd                                  AS next_specialty
    ,COUNT(*)                                            AS transition_count
    ,COUNT(DISTINCT t.member_id)                         AS unique_members
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
    AND v.is_v2 = TRUE
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date < DATE '2024-01-01'
  AND v.specialty_ctg_cd IS NOT NULL
GROUP BY
    t.trigger_dx
    ,t.member_segment
    ,v.specialty_ctg_cd;


-- ── RL LABELS TRAIN 5PCT ─────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_train_5pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_train_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider_id
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,v.dx_raw                                            AS label_dx
    ,v.days_since_trigger
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date < DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.srv_prvdr_id IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );


-- ── RL LABELS TEST 5PCT ──────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_test_5pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_test_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider_id
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,v.dx_raw                                            AS label_dx
    ,v.days_since_trigger
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date >= DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.srv_prvdr_id IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );


-- ══════════════════════════════════════════════════════════════════════════════
-- 10 PCT TABLES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── MODEL TRAIN 10PCT ────────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_train_10pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_train_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date < DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.specialty_ctg_cd IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );


-- ── MODEL TEST 10PCT ─────────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test_10pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date >= DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.specialty_ctg_cd IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );


-- ── MARKOV TRAIN 10PCT ───────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train_10pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.trigger_dx
    ,t.member_segment
    ,v.specialty_ctg_cd                                  AS next_specialty
    ,COUNT(*)                                            AS transition_count
    ,COUNT(DISTINCT t.member_id)                         AS unique_members
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
    AND v.is_v2 = TRUE
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date < DATE '2024-01-01'
  AND v.specialty_ctg_cd IS NOT NULL
GROUP BY
    t.trigger_dx
    ,t.member_segment
    ,v.specialty_ctg_cd;


-- ── RL LABELS TRAIN 10PCT ────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_train_10pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_train_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider_id
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,v.dx_raw                                            AS label_dx
    ,v.days_since_trigger
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date < DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.srv_prvdr_id IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );


-- ── RL LABELS TEST 10PCT ─────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_test_10pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_test_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider_id
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,v.dx_raw                                            AS label_dx
    ,v.days_since_trigger
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
    ON t.member_id = s.member_id
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date >= DATE '2024-01-01'
  AND t.has_claims_12m_before = TRUE
  AND v.is_v2 = FALSE
  AND v.srv_prvdr_id IS NOT NULL
  AND v.days_since_trigger <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30  AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60  AND t.is_t180_qualified = TRUE)
  );
