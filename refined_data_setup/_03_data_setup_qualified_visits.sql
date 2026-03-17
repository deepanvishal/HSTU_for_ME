-- ============================================================
-- TABLE 6 — A870800_gen_rec_model_input_sequences
-- Purpose : Model training input — pre-trigger visit sequence
--           paired with post-trigger specialty labels
-- Source  : A870800_gen_rec_triggers_qualified
--           + A870800_gen_rec_visits
--           + A870800_gen_rec_visits_qualified
-- Output  : One row per member + trigger + downstream specialty visit
--           visit_sequence = T180 visits BEFORE trigger as ARRAY STRUCT
--           label_specialty = specialty visited after trigger
--           time_bucket = T0_30 / T30_60 / T60_180
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_input_sequences`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_input_sequences`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH pre_trigger_flat AS (
    -- All visits within T180 BEFORE trigger date
    -- Flat rows with delta_t_bucket computed via LAG before aggregation
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
        ,v.visit_date
        ,DATE_DIFF(t.trigger_date, v.visit_date, DAY)    AS days_before_trigger
        ,v.specialty_ctg_cd
        ,v.specialty_desc
        ,v.dx_raw
        ,v.dx_clean                                      AS visit_dx_clean
        ,v.ccsr_category
        ,v.ccsr_category_description
        -- Log-scale delta_t bucket per HSTU paper
        -- floor(log(max(1, days_since_prior_visit)) / 0.301)
        -- NULL for first visit in sequence — no prior visit
        ,CAST(FLOOR(
            LOG(GREATEST(1,
                DATE_DIFF(v.visit_date,
                    LAG(v.visit_date) OVER (
                        PARTITION BY t.member_id, t.trigger_date, t.trigger_dx
                        ORDER BY v.visit_date
                    ),
                DAY))) / 0.301
        ) AS INT64)                                      AS delta_t_bucket
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
        ON t.member_id = v.member_id
        AND v.visit_date >= DATE_SUB(t.trigger_date, INTERVAL 180 DAY)
        AND v.visit_date < t.trigger_date
    WHERE t.is_left_qualified = TRUE
),
pre_trigger AS (
    -- Aggregate flat pre-trigger rows into ARRAY STRUCT per member + trigger
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,trigger_dx_clean
        ,trigger_ccsr
        ,trigger_specialty
        ,member_segment
        ,age_nbr
        ,gender_cd
        ,is_t30_qualified
        ,is_t60_qualified
        ,is_t180_qualified
        ,has_claims_12m_before
        ,ARRAY_AGG(
            STRUCT(
                visit_date
                ,days_before_trigger
                ,specialty_ctg_cd                        AS specialty
                ,specialty_desc
                ,dx_raw                                  AS dx
                ,visit_dx_clean                          AS dx_clean
                ,ccsr_category                           AS ccsr
                ,ccsr_category_description               AS ccsr_desc
                ,delta_t_bucket
            )
            ORDER BY visit_date
        )                                                AS visit_sequence
    FROM pre_trigger_flat
    GROUP BY
        member_id, trigger_date, trigger_dx, trigger_dx_clean
        ,trigger_ccsr, trigger_specialty, member_segment
        ,age_nbr, gender_cd
        ,is_t30_qualified, is_t60_qualified, is_t180_qualified
        ,has_claims_12m_before
),
post_trigger_labels AS (
    -- All downstream specialty visits after trigger within T180
    -- One row per member + trigger + specialty visit
    -- time_bucket assigned based on days_since_trigger
    -- Only include buckets the trigger is qualified for
    SELECT DISTINCT
        v.member_id
        ,v.trigger_date
        ,v.trigger_dx
        ,v.specialty_ctg_cd                              AS label_specialty
        ,v.specialty_desc                                AS label_specialty_desc
        ,v.days_since_trigger
        ,CASE
            WHEN v.days_since_trigger <= 30              THEN 'T0_30'
            WHEN v.days_since_trigger <= 60              THEN 'T30_60'
            WHEN v.days_since_trigger <= 180             THEN 'T60_180'
         END                                             AS time_bucket
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    WHERE v.specialty_ctg_cd IS NOT NULL
      AND v.is_v2 = FALSE
      -- Only include rows where trigger qualifies for that bucket
      AND (
          (v.days_since_trigger <= 30  AND v.is_t30_qualified  = TRUE)
       OR (v.days_since_trigger <= 60
           AND v.days_since_trigger > 30
           AND v.is_t60_qualified  = TRUE)
       OR (v.days_since_trigger <= 180
           AND v.days_since_trigger > 60
           AND v.is_t180_qualified = TRUE)
      )
)
SELECT
    p.member_id
    ,p.trigger_date
    ,p.trigger_dx
    ,p.trigger_dx_clean
    ,p.trigger_ccsr
    ,p.trigger_specialty
    ,p.member_segment
    ,p.age_nbr
    ,p.gender_cd
    ,p.is_t30_qualified
    ,p.is_t60_qualified
    ,p.is_t180_qualified
    ,p.has_claims_12m_before
    ,p.visit_sequence
    ,l.label_specialty
    ,l.label_specialty_desc
    ,l.time_bucket
    ,l.days_since_trigger                                AS days_to_specialty
FROM pre_trigger p
JOIN post_trigger_labels l
    ON p.member_id = l.member_id
    AND p.trigger_date = l.trigger_date
    AND p.trigger_dx = l.trigger_dx
