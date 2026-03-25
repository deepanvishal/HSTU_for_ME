-- ============================================================
-- SQL_02 — PROVIDER MODEL TABLES PER SAMPLE SIZE
-- Purpose : Create provider-level train, test, and Markov
--           tables for 1%, 5%, 10% stratified samples
-- Mirrors : Model_train_test_split.sql
-- Sources : A870800_gen_rec_triggers_qualified
--           A870800_gen_rec_visits              (from_provider join)
--           A870800_gen_rec_visits_qualified     (labels)
--           A870800_gen_rec_provider_vocab       (is_top80 filter)
--           A870800_gen_rec_sample_members_{pct}
-- Changes vs existing:
--   label = srv_prvdr_id (not specialty_ctg_cd)
--   train labels filtered to is_top80 = TRUE only
--   test  labels kept ALL providers including tail
--   from_provider added via triple join to visits table
--   trigger_dx_clean + trigger_specialty carried as metadata
--   Markov grain: (trigger_dx, member_segment, from_provider, to_provider)
-- ============================================================

-- ══════════════════════════════════════════════════════════════════════════════
-- SHARED CTE LOGIC (applied per sample — repeated for BQ compatibility)
-- from_provider: join visits on (member + trigger_date + trigger_dx + trigger_specialty)
-- Triple join needed — visits grain = (member, date, provider, specialty, dx)
-- trigger_specialty disambiguates when multiple providers bill same DX same day
-- ══════════════════════════════════════════════════════════════════════════════


-- ══════════════════════════════════════════════════════════════════════════════
-- 1 PCT TABLES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── PROVIDER MODEL TRAIN 1PCT ─────────────────────────────────────────────────
-- Labels: all downstream provider visits per trigger per window
-- Filtered: is_top80 = TRUE only — tail providers excluded from training labels
-- Zero-label triggers: excluded (triggers whose post-visits are all tail providers)
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_train_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH from_provider AS (
    -- Get trigger visit provider via triple join
    -- visits grain = (member, date, provider, specialty, dx)
    -- trigger_specialty disambiguates multiple providers billing same DX same day
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,vi.srv_prvdr_id                                 AS from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
        ON t.member_id = s.member_id
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` vi
        ON  vi.member_id        = t.member_id
        AND vi.visit_date       = t.trigger_date
        AND vi.dx_raw           = t.trigger_dx
        AND vi.specialty_ctg_cd = t.trigger_specialty
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date < DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
)
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.trigger_dx_clean
    ,t.trigger_specialty                                 -- metadata only, not model input
    ,fp.from_provider
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
    ON t.member_id = s.member_id
JOIN from_provider fp
    ON  fp.member_id    = t.member_id
    AND fp.trigger_date = t.trigger_date
    AND fp.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON  v.member_id    = t.member_id
    AND v.trigger_date = t.trigger_date
    AND v.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_vocab` pv
    ON  pv.srv_prvdr_id = v.srv_prvdr_id
    AND pv.is_top80     = TRUE                           -- train: top80 labels only
WHERE t.is_left_qualified      = TRUE
  AND t.trigger_date           < DATE '2024-01-01'
  AND t.has_claims_12m_before  = TRUE
  AND v.is_v2                  = FALSE
  AND v.srv_prvdr_id           IS NOT NULL
  AND v.days_since_trigger     <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30
                                    AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60
                                    AND t.is_t180_qualified = TRUE)
  )
;


-- ── PROVIDER MODEL TEST 1PCT ──────────────────────────────────────────────────
-- Labels: ALL downstream providers including tail (test = real world)
-- No is_top80 filter — model evaluated against full true label set
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH from_provider AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,vi.srv_prvdr_id                                 AS from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
        ON t.member_id = s.member_id
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` vi
        ON  vi.member_id        = t.member_id
        AND vi.visit_date       = t.trigger_date
        AND vi.dx_raw           = t.trigger_dx
        AND vi.specialty_ctg_cd = t.trigger_specialty
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date >= DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
)
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.trigger_dx_clean
    ,t.trigger_specialty
    ,fp.from_provider
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
    ON t.member_id = s.member_id
JOIN from_provider fp
    ON  fp.member_id    = t.member_id
    AND fp.trigger_date = t.trigger_date
    AND fp.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON  v.member_id    = t.member_id
    AND v.trigger_date = t.trigger_date
    AND v.trigger_dx   = t.trigger_dx
WHERE t.is_left_qualified      = TRUE
  AND t.trigger_date           >= DATE '2024-01-01'
  AND t.has_claims_12m_before  = TRUE
  AND v.is_v2                  = FALSE
  AND v.srv_prvdr_id           IS NOT NULL
  AND v.days_since_trigger     <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30
                                    AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60
                                    AND t.is_t180_qualified = TRUE)
  )
;


-- ── PROVIDER MARKOV TRAIN 1PCT ────────────────────────────────────────────────
-- Grain: (trigger_dx, member_segment, from_provider, to_provider)
-- to_provider = immediate next visit provider (is_v2 = TRUE)
-- from_provider = trigger visit provider (triple join to visits)
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH from_provider AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,vi.srv_prvdr_id                                 AS from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
        ON t.member_id = s.member_id
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` vi
        ON  vi.member_id        = t.member_id
        AND vi.visit_date       = t.trigger_date
        AND vi.dx_raw           = t.trigger_dx
        AND vi.specialty_ctg_cd = t.trigger_specialty
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date < DATE '2024-01-01'
)
SELECT
    t.trigger_dx
    ,t.member_segment
    ,fp.from_provider
    ,v.srv_prvdr_id                                      AS to_provider
    ,COUNT(*)                                            AS transition_count
    ,COUNT(DISTINCT t.member_id)                         AS unique_members
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
    ON t.member_id = s.member_id
JOIN from_provider fp
    ON  fp.member_id    = t.member_id
    AND fp.trigger_date = t.trigger_date
    AND fp.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON  v.member_id    = t.member_id
    AND v.trigger_date = t.trigger_date
    AND v.trigger_dx   = t.trigger_dx
    AND v.is_v2        = TRUE                            -- immediate next visit only
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date      < DATE '2024-01-01'
  AND v.srv_prvdr_id      IS NOT NULL
GROUP BY
    t.trigger_dx
    ,t.member_segment
    ,fp.from_provider
    ,v.srv_prvdr_id
;


-- ══════════════════════════════════════════════════════════════════════════════
-- 5 PCT TABLES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── PROVIDER MODEL TRAIN 5PCT ─────────────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_train_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH from_provider AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,vi.srv_prvdr_id                                 AS from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
        ON t.member_id = s.member_id
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` vi
        ON  vi.member_id        = t.member_id
        AND vi.visit_date       = t.trigger_date
        AND vi.dx_raw           = t.trigger_dx
        AND vi.specialty_ctg_cd = t.trigger_specialty
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date < DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
)
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.trigger_dx_clean
    ,t.trigger_specialty
    ,fp.from_provider
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
    ON t.member_id = s.member_id
JOIN from_provider fp
    ON  fp.member_id    = t.member_id
    AND fp.trigger_date = t.trigger_date
    AND fp.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON  v.member_id    = t.member_id
    AND v.trigger_date = t.trigger_date
    AND v.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_vocab` pv
    ON  pv.srv_prvdr_id = v.srv_prvdr_id
    AND pv.is_top80     = TRUE
WHERE t.is_left_qualified      = TRUE
  AND t.trigger_date           < DATE '2024-01-01'
  AND t.has_claims_12m_before  = TRUE
  AND v.is_v2                  = FALSE
  AND v.srv_prvdr_id           IS NOT NULL
  AND v.days_since_trigger     <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30
                                    AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60
                                    AND t.is_t180_qualified = TRUE)
  )
;


-- ── PROVIDER MODEL TEST 5PCT ──────────────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH from_provider AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,vi.srv_prvdr_id                                 AS from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
        ON t.member_id = s.member_id
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` vi
        ON  vi.member_id        = t.member_id
        AND vi.visit_date       = t.trigger_date
        AND vi.dx_raw           = t.trigger_dx
        AND vi.specialty_ctg_cd = t.trigger_specialty
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date >= DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
)
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.trigger_dx_clean
    ,t.trigger_specialty
    ,fp.from_provider
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
    ON t.member_id = s.member_id
JOIN from_provider fp
    ON  fp.member_id    = t.member_id
    AND fp.trigger_date = t.trigger_date
    AND fp.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON  v.member_id    = t.member_id
    AND v.trigger_date = t.trigger_date
    AND v.trigger_dx   = t.trigger_dx
WHERE t.is_left_qualified      = TRUE
  AND t.trigger_date           >= DATE '2024-01-01'
  AND t.has_claims_12m_before  = TRUE
  AND v.is_v2                  = FALSE
  AND v.srv_prvdr_id           IS NOT NULL
  AND v.days_since_trigger     <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30
                                    AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60
                                    AND t.is_t180_qualified = TRUE)
  )
;


-- ── PROVIDER MARKOV TRAIN 5PCT ────────────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH from_provider AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,vi.srv_prvdr_id                                 AS from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
        ON t.member_id = s.member_id
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` vi
        ON  vi.member_id        = t.member_id
        AND vi.visit_date       = t.trigger_date
        AND vi.dx_raw           = t.trigger_dx
        AND vi.specialty_ctg_cd = t.trigger_specialty
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date < DATE '2024-01-01'
)
SELECT
    t.trigger_dx
    ,t.member_segment
    ,fp.from_provider
    ,v.srv_prvdr_id                                      AS to_provider
    ,COUNT(*)                                            AS transition_count
    ,COUNT(DISTINCT t.member_id)                         AS unique_members
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
    ON t.member_id = s.member_id
JOIN from_provider fp
    ON  fp.member_id    = t.member_id
    AND fp.trigger_date = t.trigger_date
    AND fp.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON  v.member_id    = t.member_id
    AND v.trigger_date = t.trigger_date
    AND v.trigger_dx   = t.trigger_dx
    AND v.is_v2        = TRUE
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date      < DATE '2024-01-01'
  AND v.srv_prvdr_id      IS NOT NULL
GROUP BY
    t.trigger_dx
    ,t.member_segment
    ,fp.from_provider
    ,v.srv_prvdr_id
;


-- ══════════════════════════════════════════════════════════════════════════════
-- 10 PCT TABLES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── PROVIDER MODEL TRAIN 10PCT ────────────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_train_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH from_provider AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,vi.srv_prvdr_id                                 AS from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
        ON t.member_id = s.member_id
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` vi
        ON  vi.member_id        = t.member_id
        AND vi.visit_date       = t.trigger_date
        AND vi.dx_raw           = t.trigger_dx
        AND vi.specialty_ctg_cd = t.trigger_specialty
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date < DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
)
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.trigger_dx_clean
    ,t.trigger_specialty
    ,fp.from_provider
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
    ON t.member_id = s.member_id
JOIN from_provider fp
    ON  fp.member_id    = t.member_id
    AND fp.trigger_date = t.trigger_date
    AND fp.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON  v.member_id    = t.member_id
    AND v.trigger_date = t.trigger_date
    AND v.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_vocab` pv
    ON  pv.srv_prvdr_id = v.srv_prvdr_id
    AND pv.is_top80     = TRUE
WHERE t.is_left_qualified      = TRUE
  AND t.trigger_date           < DATE '2024-01-01'
  AND t.has_claims_12m_before  = TRUE
  AND v.is_v2                  = FALSE
  AND v.srv_prvdr_id           IS NOT NULL
  AND v.days_since_trigger     <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30
                                    AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60
                                    AND t.is_t180_qualified = TRUE)
  )
;


-- ── PROVIDER MODEL TEST 10PCT ─────────────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_model_test_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH from_provider AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,vi.srv_prvdr_id                                 AS from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
        ON t.member_id = s.member_id
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` vi
        ON  vi.member_id        = t.member_id
        AND vi.visit_date       = t.trigger_date
        AND vi.dx_raw           = t.trigger_dx
        AND vi.specialty_ctg_cd = t.trigger_specialty
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date >= DATE '2024-01-01'
      AND t.has_claims_12m_before = TRUE
)
SELECT
    t.member_id
    ,t.trigger_date
    ,t.trigger_dx
    ,t.trigger_dx_clean
    ,t.trigger_specialty
    ,fp.from_provider
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,v.srv_prvdr_id                                      AS label_provider
    ,CASE
        WHEN v.days_since_trigger <= 30                  THEN 'T0_30'
        WHEN v.days_since_trigger <= 60                  THEN 'T30_60'
        WHEN v.days_since_trigger <= 180                 THEN 'T60_180'
     END                                                 AS time_bucket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
    ON t.member_id = s.member_id
JOIN from_provider fp
    ON  fp.member_id    = t.member_id
    AND fp.trigger_date = t.trigger_date
    AND fp.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON  v.member_id    = t.member_id
    AND v.trigger_date = t.trigger_date
    AND v.trigger_dx   = t.trigger_dx
WHERE t.is_left_qualified      = TRUE
  AND t.trigger_date           >= DATE '2024-01-01'
  AND t.has_claims_12m_before  = TRUE
  AND v.is_v2                  = FALSE
  AND v.srv_prvdr_id           IS NOT NULL
  AND v.days_since_trigger     <= 180
  AND (
      (v.days_since_trigger <= 30  AND t.is_t30_qualified  = TRUE)
   OR (v.days_since_trigger <= 60  AND v.days_since_trigger > 30
                                    AND t.is_t60_qualified  = TRUE)
   OR (v.days_since_trigger <= 180 AND v.days_since_trigger > 60
                                    AND t.is_t180_qualified = TRUE)
  )
;


-- ── PROVIDER MARKOV TRAIN 10PCT ───────────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_markov_train_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH from_provider AS (
    SELECT DISTINCT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,vi.srv_prvdr_id                                 AS from_provider
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
        ON t.member_id = s.member_id
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` vi
        ON  vi.member_id        = t.member_id
        AND vi.visit_date       = t.trigger_date
        AND vi.dx_raw           = t.trigger_dx
        AND vi.specialty_ctg_cd = t.trigger_specialty
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_date < DATE '2024-01-01'
)
SELECT
    t.trigger_dx
    ,t.member_segment
    ,fp.from_provider
    ,v.srv_prvdr_id                                      AS to_provider
    ,COUNT(*)                                            AS transition_count
    ,COUNT(DISTINCT t.member_id)                         AS unique_members
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
    ON t.member_id = s.member_id
JOIN from_provider fp
    ON  fp.member_id    = t.member_id
    AND fp.trigger_date = t.trigger_date
    AND fp.trigger_dx   = t.trigger_dx
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    ON  v.member_id    = t.member_id
    AND v.trigger_date = t.trigger_date
    AND v.trigger_dx   = t.trigger_dx
    AND v.is_v2        = TRUE
WHERE t.is_left_qualified = TRUE
  AND t.trigger_date      < DATE '2024-01-01'
  AND v.srv_prvdr_id      IS NOT NULL
GROUP BY
    t.trigger_dx
    ,t.member_segment
    ,fp.from_provider
    ,v.srv_prvdr_id
;
