-- ============================================================
-- ENTROPY SUMMARY — ORDER 1
-- All 9 combinations across 3 lenses
-- Lens 1: Any sequential visits
-- Lens 2: First encounter of diagnosis
-- Lens 3: FP/I first encounter
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_entropy_order1`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_entropy_order1`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

-- ── LENS 1: ANY SEQUENTIAL VISITS ────────────────────────────────────────────
WITH any_pairs AS (
    SELECT
        v1.member_id
        ,v1.member_segment
        ,v1.specialty_ctg_cd                             AS current_specialty
        ,v1.specialty_desc                               AS current_specialty_desc
        ,v1.dx_raw                                       AS current_dx
        ,v1.ccsr_category                                AS current_ccsr
        ,v1.ccsr_category_description                    AS current_ccsr_desc
        ,v2.specialty_ctg_cd                             AS next_specialty
        ,v2.specialty_desc                               AS next_specialty_desc
        ,v2.dx_raw                                       AS next_dx
        ,v2.ccsr_category                                AS next_ccsr
        ,v2.ccsr_category_description                    AS next_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v1
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v2
        ON v1.member_id = v2.member_id
        AND v2.visit_rank = v1.visit_rank + 1
),
first_enc_pairs AS (
    SELECT
        t.member_id
        ,t.member_segment
        ,t.trigger_specialty                             AS current_specialty
        ,t.trigger_specialty_desc                        AS current_specialty_desc
        ,t.trigger_dx                                    AS current_dx
        ,t.trigger_ccsr                                  AS current_ccsr
        ,t.trigger_ccsr_desc                             AS current_ccsr_desc
        ,v.specialty_ctg_cd                              AS next_specialty
        ,v.specialty_desc                                AS next_specialty_desc
        ,v.dx_raw                                        AS next_dx
        ,v.ccsr_category                                 AS next_ccsr
        ,v.ccsr_category_description                     AS next_ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
        ON t.member_id = v.member_id
        AND t.trigger_date = v.trigger_date
        AND t.trigger_dx = v.trigger_dx
        AND v.is_v2 = TRUE
    WHERE t.is_left_qualified = TRUE
),
fp_pairs AS (
    SELECT * FROM first_enc_pairs
    WHERE current_specialty IN ('FP', 'I')
),

-- ── ENTROPY COMPUTATION PER COMBINATION PER LENS ─────────────────────────────
-- dx_to_specialty
dx_spec_any AS (
    SELECT 'dx_to_specialty' AS combination, 'any_visits' AS lens
        ,member_segment, current_dx AS current_state, next_specialty AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM any_pairs WHERE current_dx IS NOT NULL AND next_specialty IS NOT NULL
    GROUP BY member_segment, current_dx, next_specialty
),
dx_spec_fe AS (
    SELECT 'dx_to_specialty' AS combination, 'first_encounter' AS lens
        ,member_segment, current_dx AS current_state, next_specialty AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM first_enc_pairs WHERE current_dx IS NOT NULL AND next_specialty IS NOT NULL
    GROUP BY member_segment, current_dx, next_specialty
),
dx_spec_fp AS (
    SELECT 'dx_to_specialty' AS combination, 'fp_first' AS lens
        ,member_segment, current_dx AS current_state, next_specialty AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM fp_pairs WHERE current_dx IS NOT NULL AND next_specialty IS NOT NULL
    GROUP BY member_segment, current_dx, next_specialty
),
-- dx_to_dx
dx_dx_any AS (
    SELECT 'dx_to_dx' AS combination, 'any_visits' AS lens
        ,member_segment, current_dx AS current_state, next_dx AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM any_pairs WHERE current_dx IS NOT NULL AND next_dx IS NOT NULL
    GROUP BY member_segment, current_dx, next_dx
),
dx_dx_fe AS (
    SELECT 'dx_to_dx' AS combination, 'first_encounter' AS lens
        ,member_segment, current_dx AS current_state, next_dx AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM first_enc_pairs WHERE current_dx IS NOT NULL AND next_dx IS NOT NULL
    GROUP BY member_segment, current_dx, next_dx
),
dx_dx_fp AS (
    SELECT 'dx_to_dx' AS combination, 'fp_first' AS lens
        ,member_segment, current_dx AS current_state, next_dx AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM fp_pairs WHERE current_dx IS NOT NULL AND next_dx IS NOT NULL
    GROUP BY member_segment, current_dx, next_dx
),
-- dx_to_ccsr
dx_ccsr_any AS (
    SELECT 'dx_to_ccsr' AS combination, 'any_visits' AS lens
        ,member_segment, current_dx AS current_state, next_ccsr AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM any_pairs WHERE current_dx IS NOT NULL AND next_ccsr IS NOT NULL
    GROUP BY member_segment, current_dx, next_ccsr
),
dx_ccsr_fe AS (
    SELECT 'dx_to_ccsr' AS combination, 'first_encounter' AS lens
        ,member_segment, current_dx AS current_state, next_ccsr AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM first_enc_pairs WHERE current_dx IS NOT NULL AND next_ccsr IS NOT NULL
    GROUP BY member_segment, current_dx, next_ccsr
),
dx_ccsr_fp AS (
    SELECT 'dx_to_ccsr' AS combination, 'fp_first' AS lens
        ,member_segment, current_dx AS current_state, next_ccsr AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM fp_pairs WHERE current_dx IS NOT NULL AND next_ccsr IS NOT NULL
    GROUP BY member_segment, current_dx, next_ccsr
),
-- specialty_to_specialty
spec_spec_any AS (
    SELECT 'specialty_to_specialty' AS combination, 'any_visits' AS lens
        ,member_segment, current_specialty AS current_state, next_specialty AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM any_pairs WHERE current_specialty IS NOT NULL AND next_specialty IS NOT NULL
    GROUP BY member_segment, current_specialty, next_specialty
),
spec_spec_fe AS (
    SELECT 'specialty_to_specialty' AS combination, 'first_encounter' AS lens
        ,member_segment, current_specialty AS current_state, next_specialty AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM first_enc_pairs WHERE current_specialty IS NOT NULL AND next_specialty IS NOT NULL
    GROUP BY member_segment, current_specialty, next_specialty
),
spec_spec_fp AS (
    SELECT 'specialty_to_specialty' AS combination, 'fp_first' AS lens
        ,member_segment, current_specialty AS current_state, next_specialty AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM fp_pairs WHERE current_specialty IS NOT NULL AND next_specialty IS NOT NULL
    GROUP BY member_segment, current_specialty, next_specialty
),
-- specialty_to_dx
spec_dx_any AS (
    SELECT 'specialty_to_dx' AS combination, 'any_visits' AS lens
        ,member_segment, current_specialty AS current_state, next_dx AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM any_pairs WHERE current_specialty IS NOT NULL AND next_dx IS NOT NULL
    GROUP BY member_segment, current_specialty, next_dx
),
spec_dx_fe AS (
    SELECT 'specialty_to_dx' AS combination, 'first_encounter' AS lens
        ,member_segment, current_specialty AS current_state, next_dx AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM first_enc_pairs WHERE current_specialty IS NOT NULL AND next_dx IS NOT NULL
    GROUP BY member_segment, current_specialty, next_dx
),
spec_dx_fp AS (
    SELECT 'specialty_to_dx' AS combination, 'fp_first' AS lens
        ,member_segment, current_specialty AS current_state, next_dx AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM fp_pairs WHERE current_specialty IS NOT NULL AND next_dx IS NOT NULL
    GROUP BY member_segment, current_specialty, next_dx
),
-- specialty_to_ccsr
spec_ccsr_any AS (
    SELECT 'specialty_to_ccsr' AS combination, 'any_visits' AS lens
        ,member_segment, current_specialty AS current_state, next_ccsr AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM any_pairs WHERE current_specialty IS NOT NULL AND next_ccsr IS NOT NULL
    GROUP BY member_segment, current_specialty, next_ccsr
),
spec_ccsr_fe AS (
    SELECT 'specialty_to_ccsr' AS combination, 'first_encounter' AS lens
        ,member_segment, current_specialty AS current_state, next_ccsr AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM first_enc_pairs WHERE current_specialty IS NOT NULL AND next_ccsr IS NOT NULL
    GROUP BY member_segment, current_specialty, next_ccsr
),
spec_ccsr_fp AS (
    SELECT 'specialty_to_ccsr' AS combination, 'fp_first' AS lens
        ,member_segment, current_specialty AS current_state, next_ccsr AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM fp_pairs WHERE current_specialty IS NOT NULL AND next_ccsr IS NOT NULL
    GROUP BY member_segment, current_specialty, next_ccsr
),
-- ccsr_to_specialty
ccsr_spec_any AS (
    SELECT 'ccsr_to_specialty' AS combination, 'any_visits' AS lens
        ,member_segment, current_ccsr AS current_state, next_specialty AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM any_pairs WHERE current_ccsr IS NOT NULL AND next_specialty IS NOT NULL
    GROUP BY member_segment, current_ccsr, next_specialty
),
ccsr_spec_fe AS (
    SELECT 'ccsr_to_specialty' AS combination, 'first_encounter' AS lens
        ,member_segment, current_ccsr AS current_state, next_specialty AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM first_enc_pairs WHERE current_ccsr IS NOT NULL AND next_specialty IS NOT NULL
    GROUP BY member_segment, current_ccsr, next_specialty
),
ccsr_spec_fp AS (
    SELECT 'ccsr_to_specialty' AS combination, 'fp_first' AS lens
        ,member_segment, current_ccsr AS current_state, next_specialty AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM fp_pairs WHERE current_ccsr IS NOT NULL AND next_specialty IS NOT NULL
    GROUP BY member_segment, current_ccsr, next_specialty
),
-- ccsr_to_dx
ccsr_dx_any AS (
    SELECT 'ccsr_to_dx' AS combination, 'any_visits' AS lens
        ,member_segment, current_ccsr AS current_state, next_dx AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM any_pairs WHERE current_ccsr IS NOT NULL AND next_dx IS NOT NULL
    GROUP BY member_segment, current_ccsr, next_dx
),
ccsr_dx_fe AS (
    SELECT 'ccsr_to_dx' AS combination, 'first_encounter' AS lens
        ,member_segment, current_ccsr AS current_state, next_dx AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM first_enc_pairs WHERE current_ccsr IS NOT NULL AND next_dx IS NOT NULL
    GROUP BY member_segment, current_ccsr, next_dx
),
ccsr_dx_fp AS (
    SELECT 'ccsr_to_dx' AS combination, 'fp_first' AS lens
        ,member_segment, current_ccsr AS current_state, next_dx AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM fp_pairs WHERE current_ccsr IS NOT NULL AND next_dx IS NOT NULL
    GROUP BY member_segment, current_ccsr, next_dx
),
-- ccsr_to_ccsr
ccsr_ccsr_any AS (
    SELECT 'ccsr_to_ccsr' AS combination, 'any_visits' AS lens
        ,member_segment, current_ccsr AS current_state, next_ccsr AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM any_pairs WHERE current_ccsr IS NOT NULL AND next_ccsr IS NOT NULL
    GROUP BY member_segment, current_ccsr, next_ccsr
),
ccsr_ccsr_fe AS (
    SELECT 'ccsr_to_ccsr' AS combination, 'first_encounter' AS lens
        ,member_segment, current_ccsr AS current_state, next_ccsr AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM first_enc_pairs WHERE current_ccsr IS NOT NULL AND next_ccsr IS NOT NULL
    GROUP BY member_segment, current_ccsr, next_ccsr
),
ccsr_ccsr_fp AS (
    SELECT 'ccsr_to_ccsr' AS combination, 'fp_first' AS lens
        ,member_segment, current_ccsr AS current_state, next_ccsr AS next_state
        ,COUNT(*) AS transition_count, COUNT(DISTINCT member_id) AS unique_members
    FROM fp_pairs WHERE current_ccsr IS NOT NULL AND next_ccsr IS NOT NULL
    GROUP BY member_segment, current_ccsr, next_ccsr
),
all_transitions AS (
    SELECT * FROM dx_spec_any   UNION ALL SELECT * FROM dx_spec_fe   UNION ALL SELECT * FROM dx_spec_fp
    UNION ALL SELECT * FROM dx_dx_any    UNION ALL SELECT * FROM dx_dx_fe    UNION ALL SELECT * FROM dx_dx_fp
    UNION ALL SELECT * FROM dx_ccsr_any  UNION ALL SELECT * FROM dx_ccsr_fe  UNION ALL SELECT * FROM dx_ccsr_fp
    UNION ALL SELECT * FROM spec_spec_any UNION ALL SELECT * FROM spec_spec_fe UNION ALL SELECT * FROM spec_spec_fp
    UNION ALL SELECT * FROM spec_dx_any  UNION ALL SELECT * FROM spec_dx_fe  UNION ALL SELECT * FROM spec_dx_fp
    UNION ALL SELECT * FROM spec_ccsr_any UNION ALL SELECT * FROM spec_ccsr_fe UNION ALL SELECT * FROM spec_ccsr_fp
    UNION ALL SELECT * FROM ccsr_spec_any UNION ALL SELECT * FROM ccsr_spec_fe UNION ALL SELECT * FROM ccsr_spec_fp
    UNION ALL SELECT * FROM ccsr_dx_any  UNION ALL SELECT * FROM ccsr_dx_fe  UNION ALL SELECT * FROM ccsr_dx_fp
    UNION ALL SELECT * FROM ccsr_ccsr_any UNION ALL SELECT * FROM ccsr_ccsr_fe UNION ALL SELECT * FROM ccsr_ccsr_fp
),
state_totals AS (
    SELECT combination, lens, member_segment, current_state
        ,SUM(transition_count) AS state_total
    FROM all_transitions
    GROUP BY combination, lens, member_segment, current_state
),
with_entropy AS (
    SELECT
        t.combination, t.lens, t.member_segment
        ,t.current_state, t.next_state
        ,t.transition_count, t.unique_members
        ,s.state_total
        ,ROUND(t.transition_count / s.state_total, 4)   AS conditional_probability
        ,ROUND(-SUM(t.transition_count / s.state_total *
            LOG(t.transition_count / s.state_total)) OVER (
                PARTITION BY t.combination, t.lens, t.member_segment, t.current_state
            ), 4)                                        AS conditional_entropy
    FROM all_transitions t
    JOIN state_totals s
        ON t.combination = s.combination
        AND t.lens = s.lens
        AND t.member_segment = s.member_segment
        AND t.current_state = s.current_state
    WHERE t.transition_count >= 100
)
SELECT
    combination
    ,lens
    ,member_segment
    ,SUM(transition_count * conditional_entropy) / SUM(transition_count) AS weighted_avg_entropy
    ,APPROX_QUANTILES(conditional_entropy, 2)[OFFSET(1)] AS median_entropy
    ,SUM(transition_count)                               AS total_transitions
    ,COUNT(DISTINCT current_state)                       AS unique_current_states
FROM with_entropy
GROUP BY combination, lens, member_segment
ORDER BY lens, weighted_avg_entropy ASC
