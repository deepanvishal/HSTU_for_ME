-- ============================================================
-- EMBEDDING CORPUS
-- Purpose : Raw corpus for training DX, specialty and
--           provider embeddings
-- Cutoff  : Pre-2024 claims only — no leakage into test set
-- Used by : DX embeddings, specialty embeddings,
--           provider embeddings (Node2Vec / random walks)
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_embeddings_train_corpus`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_embeddings_train_corpus`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    member_id
    ,srv_start_dt                                        AS visit_date
    ,srv_prvdr_id
    ,specialty_ctg_cd
    ,REPLACE(TRIM(pri_icd9_dx_cd), '.', '')              AS dx_clean
    ,plc_srv_cd
    ,med_cost_ctg_cd
    ,age_nbr
    ,gender_cd
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
WHERE srv_start_dt < DATE '2024-01-01'
  AND pri_icd9_dx_cd IS NOT NULL
  AND TRIM(pri_icd9_dx_cd) != ''
  AND specialty_ctg_cd IS NOT NULL
  AND TRIM(specialty_ctg_cd) != ''
  AND srv_prvdr_id IS NOT NULL;


-- ============================================================
-- MARKOV TRAIN
-- Purpose : Transition counts for Markov probability
--           computation — pre-2024 triggers only
-- Critical: Never include test triggers here — leakage risk
-- Used by : Markov baseline model only
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_markov_train`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.trigger_dx
    ,t.trigger_dx_clean
    ,t.trigger_ccsr
    ,t.trigger_ccsr_desc
    ,t.trigger_specialty
    ,t.member_segment
    ,v.specialty_ctg_cd                                  AS next_specialty
    ,v.specialty_desc                                    AS next_specialty_desc
    ,v.dx_raw                                            AS next_dx
    ,v.dx_clean                                          AS next_dx_clean
    ,v.ccsr_category                                     AS next_ccsr
    ,v.ccsr_category_description                         AS next_ccsr_desc
    ,COUNT(*)                                            AS transition_count
    ,COUNT(DISTINCT t.member_id)                         AS unique_members
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
    AND v.is_v2 = TRUE
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date < DATE '2024-01-01'
  AND v.specialty_ctg_cd IS NOT NULL
GROUP BY
    t.trigger_dx, t.trigger_dx_clean
    ,t.trigger_ccsr, t.trigger_ccsr_desc
    ,t.trigger_specialty, t.member_segment
    ,v.specialty_ctg_cd, v.specialty_desc
    ,v.dx_raw, v.dx_clean
    ,v.ccsr_category, v.ccsr_category_description;


-- ============================================================
-- MODEL TRAIN
-- Purpose : Training sequences and specialty labels
--           for all sequence models
-- Cutoff  : Triggers before 2024-01-01
--           has_claims_12m_before = TRUE ensures non-empty
--           input sequence for sequence models
-- Used by : SASRec, BERT4Rec, HSTU, HSTU+embeddings,
--           BaseModel
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_train`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_train`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT *
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_input_sequences`
WHERE trigger_date < DATE '2024-01-01'
  AND has_claims_12m_before = TRUE;


-- ============================================================
-- MODEL TEST
-- Purpose : Test sequences and specialty labels
--           for all sequence models
-- Cutoff  : Triggers on or after 2024-01-01
--           has_claims_12m_before = TRUE ensures non-empty
--           input sequence for sequence models
-- Used by : SASRec, BERT4Rec, HSTU, HSTU+embeddings,
--           BaseModel, Markov evaluation
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT *
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_input_sequences`
WHERE trigger_date >= DATE '2024-01-01'
  AND has_claims_12m_before = TRUE;


-- ============================================================
-- RL LABELS TRAIN
-- Purpose : Provider-level labels and reward context
--           for reinforcement learning — train set
-- Cutoff  : Triggers before 2024-01-01
-- Grain   : One row per member + trigger + downstream visit
--           + time bucket
-- Reward hierarchy : provider match > DX match >
--                    specialty match > none
-- Used by : RL model only
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_train`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_train`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.trigger_dx_clean
    ,t.trigger_ccsr
    ,t.trigger_specialty
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,t.has_claims_12m_before
    ,t.visit_sequence
    ,v.srv_prvdr_id                                      AS label_provider_id
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,v.specialty_desc                                    AS label_specialty_desc
    ,v.dx_raw                                            AS label_dx
    ,v.dx_clean                                          AS label_dx_clean
    ,v.visit_date                                        AS label_visit_date
    ,v.days_since_trigger
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_train` t
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE v.srv_prvdr_id IS NOT NULL
  AND v.days_since_trigger <= 180
  AND v.is_v2 = FALSE
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60
       AND v.days_since_trigger > 30
       AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180
       AND v.days_since_trigger > 60
       AND t.is_t180_qualified = TRUE)
  );


-- ============================================================
-- RL LABELS TEST
-- Purpose : Provider-level labels and reward context
--           for reinforcement learning — test set
-- Cutoff  : Triggers on or after 2024-01-01
-- Used by : RL model evaluation only
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_test`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_rl_labels_test`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.trigger_dx_clean
    ,t.trigger_ccsr
    ,t.trigger_specialty
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,t.has_claims_12m_before
    ,t.visit_sequence
    ,v.srv_prvdr_id                                      AS label_provider_id
    ,v.specialty_ctg_cd                                  AS label_specialty
    ,v.specialty_desc                                    AS label_specialty_desc
    ,v.dx_raw                                            AS label_dx
    ,v.dx_clean                                          AS label_dx_clean
    ,v.visit_date                                        AS label_visit_date
    ,v.days_since_trigger
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_test` t
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON t.member_id = v.member_id
    AND t.trigger_date = v.trigger_date
    AND t.trigger_dx = v.trigger_dx
WHERE v.srv_prvdr_id IS NOT NULL
  AND v.days_since_trigger <= 180
  AND v.is_v2 = FALSE
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60
       AND v.days_since_trigger > 30
       AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180
       AND v.days_since_trigger > 60
       AND t.is_t180_qualified = TRUE)
  );
