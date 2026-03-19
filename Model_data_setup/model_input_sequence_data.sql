-- ============================================================
-- FILE 3 — SEQUENCE TABLES PER SAMPLE SIZE
-- Purpose : Pre-materialize pre-trigger visit sequences
--           for train and test per sample size
--           Python reads SELECT * — no BQ joins at runtime
--           Same sequences guaranteed across all models
-- Sources : A870800_gen_rec_triggers_qualified
--           A870800_gen_rec_visits
--           A870800_gen_rec_sample_members_{X}pct
-- Notes   : recency_rank=1 is most recent visit before trigger
--           Capped at 20 visits (MAX_SEQ_LEN)
--           Only specialty_ctg_cd stored — no ARRAY columns
--           train = trigger < 2024, test = trigger >= 2024
-- ============================================================


-- ══════════════════════════════════════════════════════════════════════════════
-- HELPER MACRO — sequence logic is identical across all 6 tables
-- Only the sample table and train/test date filter changes
-- ══════════════════════════════════════════════════════════════════════════════


-- ── TRAIN SEQUENCES 1PCT ─────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_train_sequences_1pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_train_sequences_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH triggers AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date < DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
),
ranked AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,v.specialty_ctg_cd
        ,ROW_NUMBER() OVER (
            PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
            ORDER BY v.visit_date DESC
        )                                                AS recency_rank
    FROM triggers t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date < t.trigger_date
        AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
    WHERE v.specialty_ctg_cd IS NOT NULL
      AND v.specialty_ctg_cd != ''
)
SELECT *
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank;


-- ── TEST SEQUENCES 1PCT ──────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_test_sequences_1pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_test_sequences_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH triggers AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date >= DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
),
ranked AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,v.specialty_ctg_cd
        ,ROW_NUMBER() OVER (
            PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
            ORDER BY v.visit_date DESC
        )                                                AS recency_rank
    FROM triggers t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date < t.trigger_date
        AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
    WHERE v.specialty_ctg_cd IS NOT NULL
      AND v.specialty_ctg_cd != ''
)
SELECT *
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank;


-- ══════════════════════════════════════════════════════════════════════════════
-- 5 PCT SEQUENCES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── TRAIN SEQUENCES 5PCT ─────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_train_sequences_5pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_train_sequences_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH triggers AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date < DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
),
ranked AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,v.specialty_ctg_cd
        ,ROW_NUMBER() OVER (
            PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
            ORDER BY v.visit_date DESC
        )                                                AS recency_rank
    FROM triggers t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date < t.trigger_date
        AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
    WHERE v.specialty_ctg_cd IS NOT NULL
      AND v.specialty_ctg_cd != ''
)
SELECT *
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank;


-- ── TEST SEQUENCES 5PCT ──────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_test_sequences_5pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_test_sequences_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH triggers AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date >= DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
),
ranked AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,v.specialty_ctg_cd
        ,ROW_NUMBER() OVER (
            PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
            ORDER BY v.visit_date DESC
        )                                                AS recency_rank
    FROM triggers t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date < t.trigger_date
        AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
    WHERE v.specialty_ctg_cd IS NOT NULL
      AND v.specialty_ctg_cd != ''
)
SELECT *
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank;


-- ══════════════════════════════════════════════════════════════════════════════
-- 10 PCT SEQUENCES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── TRAIN SEQUENCES 10PCT ────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_train_sequences_10pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_train_sequences_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH triggers AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date < DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
),
ranked AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,v.specialty_ctg_cd
        ,ROW_NUMBER() OVER (
            PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
            ORDER BY v.visit_date DESC
        )                                                AS recency_rank
    FROM triggers t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date < t.trigger_date
        AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
    WHERE v.specialty_ctg_cd IS NOT NULL
      AND v.specialty_ctg_cd != ''
)
SELECT *
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank;


-- ── TEST SEQUENCES 10PCT ─────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_test_sequences_10pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_test_sequences_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH triggers AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date >= DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
),
ranked AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,v.specialty_ctg_cd
        ,ROW_NUMBER() OVER (
            PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
            ORDER BY v.visit_date DESC
        )                                                AS recency_rank
    FROM triggers t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date < t.trigger_date
        AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
    WHERE v.specialty_ctg_cd IS NOT NULL
      AND v.specialty_ctg_cd != ''
)
SELECT *
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank;
