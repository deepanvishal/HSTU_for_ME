-- ============================================================
-- SQL_03 — PROVIDER SEQUENCE TABLES PER SAMPLE SIZE
-- Purpose : Pre-materialize pre-trigger provider visit sequences
--           for train and test per sample size
--           Python reads SELECT * — no BQ joins at runtime
--           Same sequences guaranteed across all models
-- Mirrors : model_input_sequence_data.sql
-- Sources : A870800_gen_rec_triggers_qualified
--           A870800_gen_rec_visits
--           A870800_gen_rec_sample_members_{X}pct
-- Changes vs existing:
--   Grain: (member, trigger, visit_date, provider) — one row per provider per day
--          Existing was (member, trigger, visit_date, specialty)
--   Added: srv_prvdr_id — primary sequence token
--   Added: delta_t_bucket — for HSTU temporal attention
--          Computed chronologically (ASC) before recency ranking (DESC)
--   Kept:  specialty_ctg_cd — for composite embedding emb(provider)+emb(specialty)
--   Tiebreaker: ROW_NUMBER ORDER BY visit_date DESC, srv_prvdr_id ASC
--               Ensures deterministic cuts when multiple providers share a day
-- Notes:
--   recency_rank=1 is most recent provider visit before trigger
--   Capped at 20 visits (MAX_SEQ_LEN) — changing requires rerunning this SQL
--   Tail providers kept — UNK mapping happens in Python (NB_02/NB_03)
--   train = trigger < 2024-01-01 | test = trigger >= 2024-01-01
-- ============================================================


-- ══════════════════════════════════════════════════════════════════════════════
-- CTE STRUCTURE (identical across all 6 tables):
--
-- Step 1 — triggers CTE
--   Filter qualified triggers for sample + date window
--
-- Step 2 — provider_days CTE
--   Dedup visits to ONE row per (member, visit_date, provider)
--   Multiple DX codes per provider per day → take first by dx_raw ASC
--   Specialty is provider-level, not DX-level, so it's stable per row
--
-- Step 3 — with_delta_t CTE
--   Compute delta_t_bucket via LAG on visit_date ASC (chronological)
--   Must be chronological — delta_t = days since PRIOR visit
--   floor(log(max(1, days_since_prior)) / 0.301) — same as HSTU paper
--
-- Step 4 — ranked CTE
--   ROW_NUMBER ORDER BY visit_date DESC, srv_prvdr_id ASC
--   DESC for recency (rank 1 = most recent)
--   srv_prvdr_id ASC as tiebreaker for same-day determinism
--
-- Step 5 — SELECT WHERE recency_rank <= 20
-- ══════════════════════════════════════════════════════════════════════════════


-- ══════════════════════════════════════════════════════════════════════════════
-- 1 PCT TABLES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── PROVIDER TRAIN SEQUENCES 1PCT ────────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_train_sequences_1pct`
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
    WHERE t.is_left_qualified      = TRUE
      AND t.trigger_date           < DATE '2024-01-01'
      AND t.has_claims_12m_before  = TRUE
),

provider_days AS (
    -- Dedup to one row per (member, visit_date, provider)
    -- Multiple DX codes per provider per day in visits table → take one
    -- Specialty is provider-level and stable — first by dx_raw ASC is fine
    SELECT
        v.member_id
        ,v.visit_date
        ,v.srv_prvdr_id
        ,v.specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    WHERE v.srv_prvdr_id      IS NOT NULL
      AND v.specialty_ctg_cd  IS NOT NULL
      AND v.specialty_ctg_cd  != ''
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY v.member_id, v.visit_date, v.srv_prvdr_id
        ORDER BY v.dx_raw ASC
    ) = 1
),

with_delta_t AS (
    -- Compute delta_t_bucket chronologically (ASC) per (member, trigger)
    -- Must be ASC — delta_t = days since PRIOR chronological visit
    -- NULL for the first visit in the 365-day window (no prior visit)
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,p.visit_date
        ,p.srv_prvdr_id
        ,p.specialty_ctg_cd
        ,CAST(FLOOR(
            LOG(GREATEST(1,
                DATE_DIFF(p.visit_date,
                    LAG(p.visit_date) OVER (
                        PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
                        ORDER BY p.visit_date ASC, p.srv_prvdr_id ASC
                    ),
                DAY))) / 0.301
        ) AS INT64)                                      AS delta_t_bucket
    FROM triggers t
    JOIN provider_days p
        ON  p.member_id  = t.member_id
        AND p.visit_date < t.trigger_date
        AND p.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
),

ranked AS (
    -- Rank by recency (most recent = rank 1)
    -- Tiebreaker srv_prvdr_id ASC ensures determinism when multiple providers
    -- share the same visit_date at the sequence boundary
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,visit_date
        ,srv_prvdr_id
        ,specialty_ctg_cd
        ,delta_t_bucket
        ,ROW_NUMBER() OVER (
            PARTITION BY member_id, trigger_date, trigger_dx
            ORDER BY visit_date DESC, srv_prvdr_id ASC
        )                                                AS recency_rank
    FROM with_delta_t
)

SELECT
    member_id
    ,trigger_date
    ,trigger_dx
    ,member_segment
    ,is_t30_qualified
    ,is_t60_qualified
    ,is_t180_qualified
    ,visit_date
    ,srv_prvdr_id
    ,specialty_ctg_cd
    ,delta_t_bucket
    ,recency_rank
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank
;


-- ── PROVIDER TEST SEQUENCES 1PCT ─────────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_test_sequences_1pct`
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
    WHERE t.is_left_qualified      = TRUE
      AND t.trigger_date           >= DATE '2024-01-01'
      AND t.has_claims_12m_before  = TRUE
),

provider_days AS (
    SELECT
        v.member_id
        ,v.visit_date
        ,v.srv_prvdr_id
        ,v.specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    WHERE v.srv_prvdr_id      IS NOT NULL
      AND v.specialty_ctg_cd  IS NOT NULL
      AND v.specialty_ctg_cd  != ''
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY v.member_id, v.visit_date, v.srv_prvdr_id
        ORDER BY v.dx_raw ASC
    ) = 1
),

with_delta_t AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,p.visit_date
        ,p.srv_prvdr_id
        ,p.specialty_ctg_cd
        ,CAST(FLOOR(
            LOG(GREATEST(1,
                DATE_DIFF(p.visit_date,
                    LAG(p.visit_date) OVER (
                        PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
                        ORDER BY p.visit_date ASC, p.srv_prvdr_id ASC
                    ),
                DAY))) / 0.301
        ) AS INT64)                                      AS delta_t_bucket
    FROM triggers t
    JOIN provider_days p
        ON  p.member_id  = t.member_id
        AND p.visit_date < t.trigger_date
        AND p.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
),

ranked AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,visit_date
        ,srv_prvdr_id
        ,specialty_ctg_cd
        ,delta_t_bucket
        ,ROW_NUMBER() OVER (
            PARTITION BY member_id, trigger_date, trigger_dx
            ORDER BY visit_date DESC, srv_prvdr_id ASC
        )                                                AS recency_rank
    FROM with_delta_t
)

SELECT
    member_id
    ,trigger_date
    ,trigger_dx
    ,member_segment
    ,is_t30_qualified
    ,is_t60_qualified
    ,is_t180_qualified
    ,visit_date
    ,srv_prvdr_id
    ,specialty_ctg_cd
    ,delta_t_bucket
    ,recency_rank
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank
;


-- ══════════════════════════════════════════════════════════════════════════════
-- 5 PCT TABLES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── PROVIDER TRAIN SEQUENCES 5PCT ────────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_train_sequences_5pct`
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
    WHERE t.is_left_qualified      = TRUE
      AND t.trigger_date           < DATE '2024-01-01'
      AND t.has_claims_12m_before  = TRUE
),

provider_days AS (
    SELECT
        v.member_id
        ,v.visit_date
        ,v.srv_prvdr_id
        ,v.specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    WHERE v.srv_prvdr_id      IS NOT NULL
      AND v.specialty_ctg_cd  IS NOT NULL
      AND v.specialty_ctg_cd  != ''
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY v.member_id, v.visit_date, v.srv_prvdr_id
        ORDER BY v.dx_raw ASC
    ) = 1
),

with_delta_t AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,p.visit_date
        ,p.srv_prvdr_id
        ,p.specialty_ctg_cd
        ,CAST(FLOOR(
            LOG(GREATEST(1,
                DATE_DIFF(p.visit_date,
                    LAG(p.visit_date) OVER (
                        PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
                        ORDER BY p.visit_date ASC, p.srv_prvdr_id ASC
                    ),
                DAY))) / 0.301
        ) AS INT64)                                      AS delta_t_bucket
    FROM triggers t
    JOIN provider_days p
        ON  p.member_id  = t.member_id
        AND p.visit_date < t.trigger_date
        AND p.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
),

ranked AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,visit_date
        ,srv_prvdr_id
        ,specialty_ctg_cd
        ,delta_t_bucket
        ,ROW_NUMBER() OVER (
            PARTITION BY member_id, trigger_date, trigger_dx
            ORDER BY visit_date DESC, srv_prvdr_id ASC
        )                                                AS recency_rank
    FROM with_delta_t
)

SELECT
    member_id
    ,trigger_date
    ,trigger_dx
    ,member_segment
    ,is_t30_qualified
    ,is_t60_qualified
    ,is_t180_qualified
    ,visit_date
    ,srv_prvdr_id
    ,specialty_ctg_cd
    ,delta_t_bucket
    ,recency_rank
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank
;


-- ── PROVIDER TEST SEQUENCES 5PCT ─────────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_test_sequences_5pct`
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
    WHERE t.is_left_qualified      = TRUE
      AND t.trigger_date           >= DATE '2024-01-01'
      AND t.has_claims_12m_before  = TRUE
),

provider_days AS (
    SELECT
        v.member_id
        ,v.visit_date
        ,v.srv_prvdr_id
        ,v.specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    WHERE v.srv_prvdr_id      IS NOT NULL
      AND v.specialty_ctg_cd  IS NOT NULL
      AND v.specialty_ctg_cd  != ''
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY v.member_id, v.visit_date, v.srv_prvdr_id
        ORDER BY v.dx_raw ASC
    ) = 1
),

with_delta_t AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,p.visit_date
        ,p.srv_prvdr_id
        ,p.specialty_ctg_cd
        ,CAST(FLOOR(
            LOG(GREATEST(1,
                DATE_DIFF(p.visit_date,
                    LAG(p.visit_date) OVER (
                        PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
                        ORDER BY p.visit_date ASC, p.srv_prvdr_id ASC
                    ),
                DAY))) / 0.301
        ) AS INT64)                                      AS delta_t_bucket
    FROM triggers t
    JOIN provider_days p
        ON  p.member_id  = t.member_id
        AND p.visit_date < t.trigger_date
        AND p.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
),

ranked AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,visit_date
        ,srv_prvdr_id
        ,specialty_ctg_cd
        ,delta_t_bucket
        ,ROW_NUMBER() OVER (
            PARTITION BY member_id, trigger_date, trigger_dx
            ORDER BY visit_date DESC, srv_prvdr_id ASC
        )                                                AS recency_rank
    FROM with_delta_t
)

SELECT
    member_id
    ,trigger_date
    ,trigger_dx
    ,member_segment
    ,is_t30_qualified
    ,is_t60_qualified
    ,is_t180_qualified
    ,visit_date
    ,srv_prvdr_id
    ,specialty_ctg_cd
    ,delta_t_bucket
    ,recency_rank
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank
;


-- ══════════════════════════════════════════════════════════════════════════════
-- 10 PCT TABLES
-- ══════════════════════════════════════════════════════════════════════════════

-- ── PROVIDER TRAIN SEQUENCES 10PCT ───────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_train_sequences_10pct`
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
    WHERE t.is_left_qualified      = TRUE
      AND t.trigger_date           < DATE '2024-01-01'
      AND t.has_claims_12m_before  = TRUE
),

provider_days AS (
    SELECT
        v.member_id
        ,v.visit_date
        ,v.srv_prvdr_id
        ,v.specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    WHERE v.srv_prvdr_id      IS NOT NULL
      AND v.specialty_ctg_cd  IS NOT NULL
      AND v.specialty_ctg_cd  != ''
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY v.member_id, v.visit_date, v.srv_prvdr_id
        ORDER BY v.dx_raw ASC
    ) = 1
),

with_delta_t AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,p.visit_date
        ,p.srv_prvdr_id
        ,p.specialty_ctg_cd
        ,CAST(FLOOR(
            LOG(GREATEST(1,
                DATE_DIFF(p.visit_date,
                    LAG(p.visit_date) OVER (
                        PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
                        ORDER BY p.visit_date ASC, p.srv_prvdr_id ASC
                    ),
                DAY))) / 0.301
        ) AS INT64)                                      AS delta_t_bucket
    FROM triggers t
    JOIN provider_days p
        ON  p.member_id  = t.member_id
        AND p.visit_date < t.trigger_date
        AND p.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
),

ranked AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,visit_date
        ,srv_prvdr_id
        ,specialty_ctg_cd
        ,delta_t_bucket
        ,ROW_NUMBER() OVER (
            PARTITION BY member_id, trigger_date, trigger_dx
            ORDER BY visit_date DESC, srv_prvdr_id ASC
        )                                                AS recency_rank
    FROM with_delta_t
)

SELECT
    member_id
    ,trigger_date
    ,trigger_dx
    ,member_segment
    ,is_t30_qualified
    ,is_t60_qualified
    ,is_t180_qualified
    ,visit_date
    ,srv_prvdr_id
    ,specialty_ctg_cd
    ,delta_t_bucket
    ,recency_rank
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank
;


-- ── PROVIDER TEST SEQUENCES 10PCT ────────────────────────────────────────────
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_test_sequences_10pct`
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
    WHERE t.is_left_qualified      = TRUE
      AND t.trigger_date           >= DATE '2024-01-01'
      AND t.has_claims_12m_before  = TRUE
),

provider_days AS (
    SELECT
        v.member_id
        ,v.visit_date
        ,v.srv_prvdr_id
        ,v.specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    WHERE v.srv_prvdr_id      IS NOT NULL
      AND v.specialty_ctg_cd  IS NOT NULL
      AND v.specialty_ctg_cd  != ''
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY v.member_id, v.visit_date, v.srv_prvdr_id
        ORDER BY v.dx_raw ASC
    ) = 1
),

with_delta_t AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.member_segment
        ,t.is_t30_qualified
        ,t.is_t60_qualified
        ,t.is_t180_qualified
        ,p.visit_date
        ,p.srv_prvdr_id
        ,p.specialty_ctg_cd
        ,CAST(FLOOR(
            LOG(GREATEST(1,
                DATE_DIFF(p.visit_date,
                    LAG(p.visit_date) OVER (
                        PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
                        ORDER BY p.visit_date ASC, p.srv_prvdr_id ASC
                    ),
                DAY))) / 0.301
        ) AS INT64)                                      AS delta_t_bucket
    FROM triggers t
    JOIN provider_days p
        ON  p.member_id  = t.member_id
        AND p.visit_date < t.trigger_date
        AND p.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 365 DAY)
),

ranked AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,member_segment
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,visit_date
        ,srv_prvdr_id
        ,specialty_ctg_cd
        ,delta_t_bucket
        ,ROW_NUMBER() OVER (
            PARTITION BY member_id, trigger_date, trigger_dx
            ORDER BY visit_date DESC, srv_prvdr_id ASC
        )                                                AS recency_rank
    FROM with_delta_t
)

SELECT
    member_id
    ,trigger_date
    ,trigger_dx
    ,member_segment
    ,is_t30_qualified
    ,is_t60_qualified
    ,is_t180_qualified
    ,visit_date
    ,srv_prvdr_id
    ,specialty_ctg_cd
    ,delta_t_bucket
    ,recency_rank
FROM ranked
WHERE recency_rank <= 20
ORDER BY member_id, trigger_date, trigger_dx, recency_rank
;
