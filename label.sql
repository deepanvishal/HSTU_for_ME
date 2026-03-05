-- ============================================================
-- TABLE 4: LABEL TABLE
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_label` AS

WITH visit_pairs AS (
    SELECT
        a.member_id
        ,a.visit_date
        ,a.visit_seq_num
        ,b.visit_date                               AS future_visit_date
        ,b.specialty_codes                          AS future_specialty_codes
        ,b.provider_ids                             AS future_provider_ids
        ,b.dx_list                                  AS future_dx_list
        ,DATE_DIFF(b.visit_date, a.visit_date, DAY) AS days_to_future_visit
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence` a
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence` b
        ON a.member_id = b.member_id
        AND b.visit_date > a.visit_date
        AND DATE_DIFF(b.visit_date, a.visit_date, DAY) <= 180
)

,specialties_unpacked AS (
    SELECT
        member_id
        ,visit_date
        ,visit_seq_num
        ,days_to_future_visit
        ,specialty
    FROM visit_pairs
    CROSS JOIN UNNEST(future_specialty_codes) AS specialty
)

,providers_unpacked AS (
    SELECT
        member_id
        ,visit_date
        ,visit_seq_num
        ,days_to_future_visit
        ,provider
    FROM visit_pairs
    CROSS JOIN UNNEST(future_provider_ids) AS provider
)

,dx_unpacked AS (
    SELECT
        member_id
        ,visit_date
        ,visit_seq_num
        ,days_to_future_visit
        ,dx
    FROM visit_pairs
    CROSS JOIN UNNEST(future_dx_list) AS dx
)

,specialty_labels AS (
    SELECT
        member_id
        ,visit_date
        ,visit_seq_num
        ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 30  THEN specialty END IGNORE NULLS) AS specialties_30
        ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 60  THEN specialty END IGNORE NULLS) AS specialties_60
        ,ARRAY_AGG(DISTINCT specialty IGNORE NULLS)                                                AS specialties_180
    FROM specialties_unpacked
    GROUP BY member_id, visit_date, visit_seq_num
)

,provider_labels AS (
    SELECT
        member_id
        ,visit_date
        ,visit_seq_num
        ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 30  THEN provider END IGNORE NULLS) AS providers_30
        ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 60  THEN provider END IGNORE NULLS) AS providers_60
        ,ARRAY_AGG(DISTINCT provider IGNORE NULLS)                                                AS providers_180
    FROM providers_unpacked
    GROUP BY member_id, visit_date, visit_seq_num
)

,dx_labels AS (
    SELECT
        member_id
        ,visit_date
        ,visit_seq_num
        ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 30  THEN dx END IGNORE NULLS) AS dx_30
        ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 60  THEN dx END IGNORE NULLS) AS dx_60
        ,ARRAY_AGG(DISTINCT dx IGNORE NULLS)                                                AS dx_180
    FROM dx_unpacked
    GROUP BY member_id, visit_date, visit_seq_num
)

SELECT
    s.member_id
    ,s.visit_date
    ,s.visit_seq_num
    ,s.specialties_30
    ,s.specialties_60
    ,s.specialties_180
    ,p.providers_30
    ,p.providers_60
    ,p.providers_180
    ,d.dx_30
    ,d.dx_60
    ,d.dx_180
FROM specialty_labels s
INNER JOIN provider_labels p
    ON s.member_id = p.member_id
    AND s.visit_date = p.visit_date
    AND s.visit_seq_num = p.visit_seq_num
INNER JOIN dx_labels d
    ON s.member_id = d.member_id
    AND s.visit_date = d.visit_date
    AND s.visit_seq_num = d.visit_seq_num
ORDER BY
    s.member_id
    ,s.visit_seq_num
