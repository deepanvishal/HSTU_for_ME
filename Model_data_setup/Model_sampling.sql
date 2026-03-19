-- ============================================================
-- FILE 1 — STRATIFIED MEMBER SAMPLING AND QA
-- Purpose : Create stratified member samples at 1%, 5%, 10%
--           Stratified on:
--             1. Member segment (proportional)
--             2. DX code coverage (100+ member DX codes)
--             3. Specialty coverage (all specialties)
--             4. T30/T60/T180 time bucket proportions
-- Output  : A870800_gen_rec_sample_members_1pct
--           A870800_gen_rec_sample_members_5pct
--           A870800_gen_rec_sample_members_10pct
-- ============================================================


-- ══════════════════════════════════════════════════════════════════════════════
-- SECTION 0 — POPULATION STATISTICS (reference for QA)
-- ══════════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH member_segments AS (
    SELECT
        member_id
        ,member_segment
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
    WHERE is_left_qualified = TRUE
    QUALIFY ROW_NUMBER() OVER (PARTITION BY member_id ORDER BY trigger_date) = 1
),
trigger_windows AS (
    SELECT
        member_id
        ,COUNTIF(is_t30_qualified  = TRUE) AS t30_triggers
        ,COUNTIF(is_t60_qualified  = TRUE) AS t60_triggers
        ,COUNTIF(is_t180_qualified = TRUE) AS t180_triggers
        ,COUNT(*)                           AS total_triggers
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
    WHERE is_left_qualified = TRUE
    GROUP BY member_id
)
SELECT
    m.member_id
    ,m.member_segment
    ,COALESCE(t.t30_triggers,  0) AS t30_triggers
    ,COALESCE(t.t60_triggers,  0) AS t60_triggers
    ,COALESCE(t.t180_triggers, 0) AS t180_triggers
    ,COALESCE(t.total_triggers, 0) AS total_triggers
FROM member_segments m
LEFT JOIN trigger_windows t ON m.member_id = t.member_id;


-- ══════════════════════════════════════════════════════════════════════════════
-- SECTION 1 — DX CODES WITH 100+ MEMBERS (coverage requirement)
-- ══════════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_coverage_targets`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    trigger_dx
    ,COUNT(DISTINCT member_id)                           AS member_count
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
WHERE is_left_qualified = TRUE
GROUP BY trigger_dx
HAVING COUNT(DISTINCT member_id) >= 100;


-- ══════════════════════════════════════════════════════════════════════════════
-- SECTION 2 — SPECIALTY COVERAGE TARGETS
-- ══════════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_coverage_targets`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT DISTINCT
    specialty_ctg_cd
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
WHERE specialty_ctg_cd IS NOT NULL
  AND specialty_ctg_cd != ''
  AND is_left_qualified = TRUE;


-- ══════════════════════════════════════════════════════════════════════════════
-- SECTION 3 — BEST MEMBER PER DX CODE (richest signal)
-- One member per DX code — the one with most triggers for that DX
-- ══════════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_representative_members`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH dx_member_counts AS (
    SELECT
        trigger_dx
        ,member_id
        ,COUNT(*) AS trigger_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
    WHERE is_left_qualified = TRUE
      AND trigger_dx IN (
          SELECT trigger_dx
          FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_coverage_targets`
      )
    GROUP BY trigger_dx, member_id
)
SELECT
    trigger_dx
    ,member_id
FROM dx_member_counts
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY trigger_dx
    ORDER BY trigger_count DESC
) = 1;


-- ══════════════════════════════════════════════════════════════════════════════
-- SECTION 4 — BEST MEMBER PER SPECIALTY (richest signal)
-- ══════════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_representative_members`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH specialty_member_counts AS (
    SELECT
        specialty_ctg_cd
        ,member_id
        ,COUNT(*) AS visit_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
    WHERE specialty_ctg_cd IS NOT NULL
      AND specialty_ctg_cd != ''
      AND is_left_qualified = TRUE
    GROUP BY specialty_ctg_cd, member_id
)
SELECT
    specialty_ctg_cd
    ,member_id
FROM specialty_member_counts
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY specialty_ctg_cd
    ORDER BY visit_count DESC
) = 1;


-- ══════════════════════════════════════════════════════════════════════════════
-- MACRO — STRATIFIED SAMPLE BUILDER
-- Called three times with different sample rates
-- ══════════════════════════════════════════════════════════════════════════════

-- ── 1% SAMPLE ────────────────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH
-- Layer 1: stratified base sample by member_segment
base_sample AS (
    SELECT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`
    WHERE FARM_FINGERPRINT(CAST(member_id AS STRING)) < 0.01 * 0x7FFFFFFFFFFFFFFF
),
-- Layer 2: force-include one member per qualifying DX code
dx_forced AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_representative_members`
),
-- Layer 3: force-include one member per specialty
specialty_forced AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_representative_members`
),
-- Layer 4: top up T180 coverage
-- Find members with T180 triggers not already in sample
t180_topup AS (
    SELECT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`
    WHERE t180_triggers > 0
      AND member_id NOT IN (SELECT member_id FROM base_sample)
    ORDER BY t180_triggers DESC
    LIMIT (
        SELECT CAST(COUNT(*) * 0.001 AS INT64)
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`
    )
),
-- Union all layers
all_members AS (
    SELECT member_id FROM base_sample
    UNION DISTINCT
    SELECT member_id FROM dx_forced
    UNION DISTINCT
    SELECT member_id FROM specialty_forced
    UNION DISTINCT
    SELECT member_id FROM t180_topup
)
SELECT DISTINCT
    a.member_id
    ,p.member_segment
    ,p.t30_triggers
    ,p.t60_triggers
    ,p.t180_triggers
    ,p.total_triggers
FROM all_members a
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats` p
    ON a.member_id = p.member_id;


-- ── 5% SAMPLE ────────────────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH
base_sample AS (
    SELECT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`
    WHERE FARM_FINGERPRINT(CAST(member_id AS STRING)) < 0.05 * 0x7FFFFFFFFFFFFFFF
),
dx_forced AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_representative_members`
),
specialty_forced AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_representative_members`
),
t180_topup AS (
    SELECT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`
    WHERE t180_triggers > 0
      AND member_id NOT IN (SELECT member_id FROM base_sample)
    ORDER BY t180_triggers DESC
    LIMIT (
        SELECT CAST(COUNT(*) * 0.005 AS INT64)
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`
    )
),
all_members AS (
    SELECT member_id FROM base_sample
    UNION DISTINCT
    SELECT member_id FROM dx_forced
    UNION DISTINCT
    SELECT member_id FROM specialty_forced
    UNION DISTINCT
    SELECT member_id FROM t180_topup
)
SELECT DISTINCT
    a.member_id
    ,p.member_segment
    ,p.t30_triggers
    ,p.t60_triggers
    ,p.t180_triggers
    ,p.total_triggers
FROM all_members a
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats` p
    ON a.member_id = p.member_id;


-- ── 10% SAMPLE ───────────────────────────────────────────────────────────────
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH
base_sample AS (
    SELECT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`
    WHERE FARM_FINGERPRINT(CAST(member_id AS STRING)) < 0.10 * 0x7FFFFFFFFFFFFFFF
),
dx_forced AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_representative_members`
),
specialty_forced AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_representative_members`
),
t180_topup AS (
    SELECT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`
    WHERE t180_triggers > 0
      AND member_id NOT IN (SELECT member_id FROM base_sample)
    ORDER BY t180_triggers DESC
    LIMIT (
        SELECT CAST(COUNT(*) * 0.01 AS INT64)
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`
    )
),
all_members AS (
    SELECT member_id FROM base_sample
    UNION DISTINCT
    SELECT member_id FROM dx_forced
    UNION DISTINCT
    SELECT member_id FROM specialty_forced
    UNION DISTINCT
    SELECT member_id FROM t180_topup
)
SELECT DISTINCT
    a.member_id
    ,p.member_segment
    ,p.t30_triggers
    ,p.t60_triggers
    ,p.t180_triggers
    ,p.total_triggers
FROM all_members a
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats` p
    ON a.member_id = p.member_id;


-- ══════════════════════════════════════════════════════════════════════════════
-- SECTION 5 — QA VALIDATION
-- Run for each sample to confirm representativeness
-- ══════════════════════════════════════════════════════════════════════════════

-- ── QA MACRO — run for each sample size ──────────────────────────────────────
-- Replace SAMPLE_TABLE with actual table name to validate

-- QA 1: Member count and effective sample rate
SELECT
    'Population'                                         AS dataset
    ,COUNT(*)                                            AS member_count
    ,SUM(total_triggers)                                 AS total_triggers
    ,ROUND(AVG(t30_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t30_qualified
    ,ROUND(AVG(t60_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t60_qualified
    ,ROUND(AVG(t180_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t180_qualified
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`

UNION ALL

SELECT
    '1pct Sample'                                        AS dataset
    ,COUNT(*)                                            AS member_count
    ,SUM(total_triggers)                                 AS total_triggers
    ,ROUND(AVG(t30_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t30_qualified
    ,ROUND(AVG(t60_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t60_qualified
    ,ROUND(AVG(t180_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t180_qualified
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct`

UNION ALL

SELECT
    '5pct Sample'                                        AS dataset
    ,COUNT(*)                                            AS member_count
    ,SUM(total_triggers)                                 AS total_triggers
    ,ROUND(AVG(t30_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t30_qualified
    ,ROUND(AVG(t60_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t60_qualified
    ,ROUND(AVG(t180_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t180_qualified
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct`

UNION ALL

SELECT
    '10pct Sample'                                       AS dataset
    ,COUNT(*)                                            AS member_count
    ,SUM(total_triggers)                                 AS total_triggers
    ,ROUND(AVG(t30_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t30_qualified
    ,ROUND(AVG(t60_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t60_qualified
    ,ROUND(AVG(t180_triggers / NULLIF(total_triggers,0)) * 100, 2) AS pct_t180_qualified
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct`

ORDER BY dataset;


-- QA 2: Member segment distribution vs population
SELECT
    member_segment
    ,SUM(CASE WHEN dataset = 'Population' THEN member_count END) AS pop_count
    ,SUM(CASE WHEN dataset = '1pct'       THEN member_count END) AS s1_count
    ,SUM(CASE WHEN dataset = '5pct'       THEN member_count END) AS s5_count
    ,SUM(CASE WHEN dataset = '10pct'      THEN member_count END) AS s10_count
    ,ROUND(SUM(CASE WHEN dataset = 'Population' THEN pct END), 2) AS pop_pct
    ,ROUND(SUM(CASE WHEN dataset = '1pct'       THEN pct END), 2) AS s1_pct
    ,ROUND(SUM(CASE WHEN dataset = '5pct'       THEN pct END), 2) AS s5_pct
    ,ROUND(SUM(CASE WHEN dataset = '10pct'      THEN pct END), 2) AS s10_pct
FROM (
    SELECT 'Population' AS dataset, member_segment
        ,COUNT(*) AS member_count
        ,COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_population_stats`
    GROUP BY member_segment

    UNION ALL

    SELECT '1pct' AS dataset, member_segment
        ,COUNT(*) AS member_count
        ,COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct`
    GROUP BY member_segment

    UNION ALL

    SELECT '5pct' AS dataset, member_segment
        ,COUNT(*) AS member_count
        ,COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct`
    GROUP BY member_segment

    UNION ALL

    SELECT '10pct' AS dataset, member_segment
        ,COUNT(*) AS member_count
        ,COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct`
    GROUP BY member_segment
)
GROUP BY member_segment
ORDER BY member_segment;


-- QA 3: DX code coverage — what % of 100+ member DX codes are in each sample
WITH pop_dx AS (
    SELECT DISTINCT trigger_dx
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_dx_coverage_targets`
),
s1_dx AS (
    SELECT DISTINCT trigger_dx
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
),
s5_dx AS (
    SELECT DISTINCT trigger_dx
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
),
s10_dx AS (
    SELECT DISTINCT trigger_dx
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
)
SELECT
    COUNT(DISTINCT p.trigger_dx)                         AS total_dx_codes
    ,COUNT(DISTINCT s1.trigger_dx)                       AS s1_dx_covered
    ,COUNT(DISTINCT s5.trigger_dx)                       AS s5_dx_covered
    ,COUNT(DISTINCT s10.trigger_dx)                      AS s10_dx_covered
    ,ROUND(COUNT(DISTINCT s1.trigger_dx)  * 100.0 / COUNT(DISTINCT p.trigger_dx), 2) AS s1_coverage_pct
    ,ROUND(COUNT(DISTINCT s5.trigger_dx)  * 100.0 / COUNT(DISTINCT p.trigger_dx), 2) AS s5_coverage_pct
    ,ROUND(COUNT(DISTINCT s10.trigger_dx) * 100.0 / COUNT(DISTINCT p.trigger_dx), 2) AS s10_coverage_pct
FROM pop_dx p
LEFT JOIN s1_dx  s1  ON p.trigger_dx = s1.trigger_dx
LEFT JOIN s5_dx  s5  ON p.trigger_dx = s5.trigger_dx
LEFT JOIN s10_dx s10 ON p.trigger_dx = s10.trigger_dx;


-- QA 4: Specialty coverage — what % of specialties are in each sample
WITH pop_spec AS (
    SELECT DISTINCT specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_specialty_coverage_targets`
),
s1_spec AS (
    SELECT DISTINCT v.specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
        ON v.member_id = s.member_id
    WHERE v.is_left_qualified = TRUE
),
s5_spec AS (
    SELECT DISTINCT v.specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
        ON v.member_id = s.member_id
    WHERE v.is_left_qualified = TRUE
),
s10_spec AS (
    SELECT DISTINCT v.specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
        ON v.member_id = s.member_id
    WHERE v.is_left_qualified = TRUE
)
SELECT
    COUNT(DISTINCT p.specialty_ctg_cd)                   AS total_specialties
    ,COUNT(DISTINCT s1.specialty_ctg_cd)                 AS s1_spec_covered
    ,COUNT(DISTINCT s5.specialty_ctg_cd)                 AS s5_spec_covered
    ,COUNT(DISTINCT s10.specialty_ctg_cd)                AS s10_spec_covered
    ,ROUND(COUNT(DISTINCT s1.specialty_ctg_cd)  * 100.0 / COUNT(DISTINCT p.specialty_ctg_cd), 2) AS s1_coverage_pct
    ,ROUND(COUNT(DISTINCT s5.specialty_ctg_cd)  * 100.0 / COUNT(DISTINCT p.specialty_ctg_cd), 2) AS s5_coverage_pct
    ,ROUND(COUNT(DISTINCT s10.specialty_ctg_cd) * 100.0 / COUNT(DISTINCT p.specialty_ctg_cd), 2) AS s10_coverage_pct
FROM pop_spec p
LEFT JOIN s1_spec  s1  ON p.specialty_ctg_cd = s1.specialty_ctg_cd
LEFT JOIN s5_spec  s5  ON p.specialty_ctg_cd = s5.specialty_ctg_cd
LEFT JOIN s10_spec s10 ON p.specialty_ctg_cd = s10.specialty_ctg_cd;


-- QA 5: Train vs test trigger split per sample
-- Confirms split is consistent with full population
WITH pop_split AS (
    SELECT
        'Population' AS dataset
        ,COUNTIF(trigger_date < DATE '2024-01-01') AS train_triggers
        ,COUNTIF(trigger_date >= DATE '2024-01-01') AS test_triggers
        ,COUNT(*) AS total_triggers
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
    WHERE is_left_qualified = TRUE
),
s1_split AS (
    SELECT
        '1pct Sample' AS dataset
        ,COUNTIF(t.trigger_date < DATE '2024-01-01') AS train_triggers
        ,COUNTIF(t.trigger_date >= DATE '2024-01-01') AS test_triggers
        ,COUNT(*) AS total_triggers
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_1pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
),
s5_split AS (
    SELECT
        '5pct Sample' AS dataset
        ,COUNTIF(t.trigger_date < DATE '2024-01-01') AS train_triggers
        ,COUNTIF(t.trigger_date >= DATE '2024-01-01') AS test_triggers
        ,COUNT(*) AS total_triggers
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_5pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
),
s10_split AS (
    SELECT
        '10pct Sample' AS dataset
        ,COUNTIF(t.trigger_date < DATE '2024-01-01') AS train_triggers
        ,COUNTIF(t.trigger_date >= DATE '2024-01-01') AS test_triggers
        ,COUNT(*) AS total_triggers
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_sample_members_10pct` s
        ON t.member_id = s.member_id
    WHERE t.is_left_qualified = TRUE
)
SELECT
    dataset
    ,train_triggers
    ,test_triggers
    ,total_triggers
    ,ROUND(train_triggers * 100.0 / NULLIF(total_triggers, 0), 2) AS train_pct
    ,ROUND(test_triggers  * 100.0 / NULLIF(total_triggers, 0), 2) AS test_pct
FROM (
    SELECT * FROM pop_split
    UNION ALL SELECT * FROM s1_split
    UNION ALL SELECT * FROM s5_split
    UNION ALL SELECT * FROM s10_split
)
ORDER BY dataset;
