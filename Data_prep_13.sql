-- ============================================================
-- STEP 1 — MEMBER LEVEL UNNESTED BASE — TRAIN AND TEST
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_eval`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH triggers_dated AS (
    SELECT
        member_id
        ,trigger_dx
        ,trigger_date
        ,member_segment
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers`
),
unnested AS (
    SELECT
        b.member_id
        ,b.trigger_dx
        ,b.trigger_dx_desc
        ,b.trigger_ccsr
        ,b.trigger_ccsr_desc
        ,b.member_segment
        ,t.trigger_date
        ,v.specialty                                     AS visit_specialty
        ,v.specialty_desc                                AS visit_specialty_desc
        ,v.dx                                            AS visit_dx
        ,v.days_since_trigger
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_base` b
    JOIN triggers_dated t
        ON b.member_id = t.member_id
        AND b.trigger_dx = t.trigger_dx
    ,UNNEST(b.downstream_visits) AS v
    WHERE v.specialty IS NOT NULL
      AND v.specialty != ''
),
v2 AS (
    SELECT
        member_id
        ,trigger_dx
        ,trigger_date
        ,member_segment
        ,visit_dx                                        AS v2_dx
        ,days_since_trigger                              AS v2_days
    FROM unnested
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY member_id, trigger_dx, trigger_date
        ORDER BY days_since_trigger
    ) = 1
),
first_specialist AS (
    SELECT
        u.member_id
        ,u.trigger_dx
        ,u.trigger_dx_desc
        ,u.trigger_ccsr
        ,u.trigger_ccsr_desc
        ,u.member_segment
        ,u.trigger_date
        ,v.v2_dx
        ,u.visit_specialty                               AS actual_specialty
        ,u.visit_specialty_desc                          AS actual_specialty_desc
        ,u.days_since_trigger
        ,CASE
            WHEN u.days_since_trigger <= 30  THEN 'T30'
            WHEN u.days_since_trigger <= 60  THEN 'T60'
            WHEN u.days_since_trigger <= 180 THEN 'T180'
        END                                              AS time_window
    FROM unnested u
    JOIN v2 v
        ON u.member_id = v.member_id
        AND u.trigger_dx = v.trigger_dx
        AND u.trigger_date = v.trigger_date
    WHERE u.days_since_trigger > v.v2_days
      AND u.visit_specialty IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY u.member_id, u.trigger_dx, u.trigger_date, 
                     CASE WHEN u.days_since_trigger <= 30  THEN 'T30'
                          WHEN u.days_since_trigger <= 60  THEN 'T60'
                          WHEN u.days_since_trigger <= 180 THEN 'T180' END
        ORDER BY u.days_since_trigger
    ) = 1
)
SELECT
    *
    ,CASE WHEN trigger_date < '2024-01-01' THEN 'train' ELSE 'test' END AS split
FROM first_specialist
WHERE time_window IS NOT NULL;


-- ============================================================
-- STEP 2 — TRAIN: BUILD TRANSITION PROBABILITIES
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_train`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH transition_counts AS (
    SELECT
        trigger_dx, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,v2_dx
        ,actual_specialty                                AS next_specialty
        ,actual_specialty_desc                           AS next_specialty_desc
        ,member_segment, time_window
        ,COUNT(DISTINCT member_id)                       AS transition_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_eval`
    WHERE split = 'train'
    GROUP BY
        trigger_dx, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,v2_dx
        ,actual_specialty, actual_specialty_desc
        ,member_segment, time_window
    HAVING COUNT(DISTINCT member_id) >= 100
),
pair_totals AS (
    SELECT
        trigger_dx, v2_dx, member_segment, time_window
        ,SUM(transition_count)                           AS pair_total
    FROM transition_counts
    GROUP BY trigger_dx, v2_dx, member_segment, time_window
)
SELECT
    t.*
    ,p.pair_total
    ,ROUND(t.transition_count / p.pair_total, 4)         AS train_probability
    ,ROW_NUMBER() OVER (
        PARTITION BY t.trigger_dx, t.v2_dx, t.member_segment, t.time_window
        ORDER BY t.transition_count DESC
    )                                                    AS specialty_rank
FROM transition_counts t
JOIN pair_totals p
    ON t.trigger_dx = p.trigger_dx
    AND t.v2_dx = p.v2_dx
    AND t.member_segment = p.member_segment
    AND t.time_window = p.time_window;


-- ============================================================
-- STEP 3 — TEST: JOIN PREDICTIONS TO ACTUALS
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_predictions`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    e.member_id
    ,e.trigger_date
    ,e.trigger_dx
    ,e.trigger_dx_desc
    ,e.v2_dx
    ,e.actual_specialty
    ,e.member_segment
    ,e.time_window
    ,p.next_specialty                                    AS predicted_specialty
    ,p.train_probability
    ,p.specialty_rank
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_eval` e
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_train` p
    ON e.trigger_dx = p.trigger_dx
    AND e.v2_dx = p.v2_dx
    AND e.member_segment = p.member_segment
    AND e.time_window = p.time_window
WHERE e.split = 'test'
  AND p.specialty_rank <= 5
