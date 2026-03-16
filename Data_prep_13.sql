CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_train`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH train_data AS (
    SELECT *
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_summary`
    WHERE trigger_date < '2024-01-01'
      AND next_specialty IS NOT NULL
),
transition_counts AS (
    SELECT
        trigger_dx
        ,trigger_dx_desc
        ,v2_dx
        ,v2_dx_desc
        ,next_specialty
        ,next_specialty_desc
        ,member_segment
        ,time_window
        ,COUNT(DISTINCT member_id)                       AS transition_count
    FROM train_data
    GROUP BY
        trigger_dx, trigger_dx_desc
        ,v2_dx, v2_dx_desc
        ,next_specialty, next_specialty_desc
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
    AND t.time_window = p.time_window
