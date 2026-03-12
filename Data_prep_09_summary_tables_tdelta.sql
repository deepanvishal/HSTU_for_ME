DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_summary`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_summary`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH unnested AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,trigger_dx_desc
        ,trigger_ccsr
        ,trigger_ccsr_desc
        ,trigger_specialty
        ,trigger_specialty_desc
        ,member_segment
        ,v.visit_date
        ,v.days_since_trigger
        ,v.specialty
        ,v.specialty_desc
        ,v.dx
        ,v.dx_desc
        ,v.ccsr
        ,v.ccsr_desc
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_base`
    ,UNNEST(downstream_visits) AS v
),
v2 AS (
    SELECT
        member_id
        ,trigger_date
        ,trigger_dx
        ,dx                                              AS v2_dx
        ,dx_desc                                         AS v2_dx_desc
        ,days_since_trigger                              AS v2_days
    FROM unnested
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY member_id, trigger_date, trigger_dx
        ORDER BY days_since_trigger
    ) = 1
),
post_v2 AS (
    SELECT
        u.member_id
        ,u.trigger_date
        ,u.trigger_dx
        ,u.trigger_dx_desc
        ,u.trigger_ccsr
        ,u.trigger_ccsr_desc
        ,u.trigger_specialty
        ,u.member_segment
        ,v.v2_dx
        ,v.v2_dx_desc
        ,u.specialty                                     AS next_specialty
        ,u.specialty_desc                                AS next_specialty_desc
        ,u.days_since_trigger
    FROM unnested u
    JOIN v2 v
        ON u.member_id = v.member_id
        AND u.trigger_date = v.trigger_date
        AND u.trigger_dx = v.trigger_dx
    WHERE u.days_since_trigger > v.v2_days
      AND u.specialty IS NOT NULL
),
first_post_v2_by_window AS (
    SELECT * FROM (
        SELECT *, 'T30'  AS time_window FROM post_v2 WHERE days_since_trigger <= 30
        UNION ALL
        SELECT *, 'T60'  AS time_window FROM post_v2 WHERE days_since_trigger <= 60
        UNION ALL
        SELECT *, 'T180' AS time_window FROM post_v2 WHERE days_since_trigger <= 180
    )
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY member_id, trigger_date, trigger_dx, v2_dx, time_window
        ORDER BY days_since_trigger
    ) = 1
),
transition_counts AS (
    SELECT
        trigger_dx, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,trigger_specialty
        ,v2_dx, v2_dx_desc
        ,next_specialty, next_specialty_desc
        ,member_segment, time_window
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
        ,ROUND(AVG(days_since_trigger), 1)               AS avg_days_to_specialty
        ,ROUND(APPROX_QUANTILES(days_since_trigger, 2)[OFFSET(1)], 1) AS median_days_to_specialty
    FROM first_post_v2_by_window
    GROUP BY
        trigger_dx, trigger_dx_desc
        ,trigger_ccsr, trigger_ccsr_desc
        ,trigger_specialty
        ,v2_dx, v2_dx_desc
        ,next_specialty, next_specialty_desc
        ,member_segment, time_window
),
pair_totals AS (
    SELECT
        trigger_dx, v2_dx, member_segment, time_window
        ,SUM(transition_count)                           AS pair_total
    FROM transition_counts
    GROUP BY trigger_dx, v2_dx, member_segment, time_window
)
SELECT
    t.trigger_dx, t.trigger_dx_desc
    ,t.trigger_ccsr, t.trigger_ccsr_desc
    ,t.trigger_specialty
    ,t.v2_dx, t.v2_dx_desc
    ,t.next_specialty, t.next_specialty_desc
    ,t.member_segment, t.time_window
    ,t.transition_count, t.unique_members
    ,p.pair_total
    ,t.avg_days_to_specialty
    ,t.median_days_to_specialty
    ,ROUND(t.transition_count / p.pair_total, 4)         AS conditional_probability
    ,ROUND(-SUM(t.transition_count / p.pair_total *
        LOG(t.transition_count / p.pair_total)) OVER (
            PARTITION BY t.trigger_dx, t.v2_dx, t.member_segment, t.time_window
        ), 4)                                            AS conditional_entropy
FROM transition_counts t
JOIN pair_totals p
    ON t.trigger_dx = p.trigger_dx
    AND t.v2_dx = p.v2_dx
    AND t.member_segment = p.member_segment
    AND t.time_window = p.time_window
ORDER BY t.time_window, t.trigger_dx, t.v2_dx, t.transition_count DESC
