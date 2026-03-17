-- ============================================================
-- TABLE 6 — A870800_gen_rec_model_input_sequences
-- Purpose : Model training input — pre-trigger visit sequence
--           and post-trigger specialty labels per window
-- Source  : A870800_gen_rec_triggers_qualified
--           + A870800_gen_rec_visits
--           + A870800_gen_rec_visits_qualified
-- Output  : One row per member + trigger
--           visit_sequence = T180 visits BEFORE trigger as ARRAY
--           label_t30/t60/t180 = first specialist AFTER trigger
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_input_sequences`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_model_input_sequences`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH pre_trigger_flat AS (
    -- All visits within T180 BEFORE trigger — flat rows with delta_t computed
    -- delta_t_bucket computed here using LAG before aggregation
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
        -- delta_t = days since prior visit in sequence
        -- log-scale bucketed per HSTU paper: floor(log(max(1, delta_t)) / 0.301)
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
    -- Aggregate flat rows into ARRAY STRUCT per member + trigger
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
-- First specialist label per window
-- Uses earliest visit date after V2 within window
-- to avoid alphabetical MIN issue on specialty code
first_post_v2_visit AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,time_window
        ,specialty_ctg_cd
        ,specialty_desc
    FROM (
        SELECT
            v.member_id
            ,v.trigger_date
            ,v.trigger_dx
            ,v.specialty_ctg_cd
            ,v.specialty_desc
            ,'T30'                                       AS time_window
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
        WHERE v.is_v2 = FALSE
          AND v.days_since_trigger <= 30
          AND v.specialty_ctg_cd IS NOT NULL
          AND v.is_t30_qualified = TRUE
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY v.member_id, v.trigger_date, v.trigger_dx
            ORDER BY v.days_since_trigger
        ) = 1

        UNION ALL

        SELECT
            v.member_id
            ,v.trigger_date
            ,v.trigger_dx
            ,v.specialty_ctg_cd
            ,v.specialty_desc
            ,'T60'                                       AS time_window
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
        WHERE v.is_v2 = FALSE
          AND v.days_since_trigger <= 60
          AND v.specialty_ctg_cd IS NOT NULL
          AND v.is_t60_qualified = TRUE
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY v.member_id, v.trigger_date, v.trigger_dx
            ORDER BY v.days_since_trigger
        ) = 1

        UNION ALL

        SELECT
            v.member_id
            ,v.trigger_date
            ,v.trigger_dx
            ,v.specialty_ctg_cd
            ,v.specialty_desc
            ,'T180'                                      AS time_window
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
        WHERE v.is_v2 = FALSE
          AND v.days_since_trigger <= 180
          AND v.specialty_ctg_cd IS NOT NULL
          AND v.is_t180_qualified = TRUE
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY v.member_id, v.trigger_date, v.trigger_dx
            ORDER BY v.days_since_trigger
        ) = 1
    )
),
labels AS (
    -- Pivot time windows into columns
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,MAX(CASE WHEN time_window = 'T30'  THEN specialty_ctg_cd END) AS label_t30
        ,MAX(CASE WHEN time_window = 'T60'  THEN specialty_ctg_cd END) AS label_t60
        ,MAX(CASE WHEN time_window = 'T180' THEN specialty_ctg_cd END) AS label_t180
        ,MAX(CASE WHEN time_window = 'T30'  THEN specialty_desc END)   AS label_t30_desc
        ,MAX(CASE WHEN time_window = 'T60'  THEN specialty_desc END)   AS label_t60_desc
        ,MAX(CASE WHEN time_window = 'T180' THEN specialty_desc END)   AS label_t180_desc
    FROM first_post_v2_visit
    GROUP BY member_id, trigger_date, trigger_dx
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
    ,l.label_t30
    ,l.label_t30_desc
    ,l.label_t60
    ,l.label_t60_desc
    ,l.label_t180
    ,l.label_t180_desc
FROM pre_trigger p
LEFT JOIN labels l
    ON p.member_id = l.member_id
    AND p.trigger_date = l.trigger_date
    AND p.trigger_dx = l.trigger_dx
