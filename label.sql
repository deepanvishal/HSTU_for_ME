CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.label_table` AS

WITH visit_pairs AS (
    SELECT
        a.member_id
        ,a.visit_date
        ,a.visit_seq_num
        ,b.visit_date                                   AS future_visit_date
        ,b.specialty_codes                              AS future_specialty_codes
        ,b.provider_ids                                 AS future_provider_ids
        ,b.dx_list                                      AS future_dx_list
        ,DATE_DIFF(b.visit_date, a.visit_date, DAY)     AS days_to_future_visit
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.sequence_table` a
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.sequence_table` b
        ON a.member_id = b.member_id
        AND b.visit_date > a.visit_date
        AND DATE_DIFF(b.visit_date, a.visit_date, DAY) <= 180
)

SELECT
    member_id
    ,visit_date
    ,visit_seq_num

    -- specialty labels
    ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 30  THEN specialty END IGNORE NULLS) AS specialties_30
    ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 60  THEN specialty END IGNORE NULLS) AS specialties_60
    ,ARRAY_AGG(DISTINCT specialty IGNORE NULLS)                                                AS specialties_180

    -- provider labels
    ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 30  THEN provider END IGNORE NULLS) AS providers_30
    ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 60  THEN provider END IGNORE NULLS) AS providers_60
    ,ARRAY_AGG(DISTINCT provider IGNORE NULLS)                                                AS providers_180

    -- diagnosis labels
    ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 30  THEN dx END IGNORE NULLS)       AS dx_30
    ,ARRAY_AGG(DISTINCT CASE WHEN days_to_future_visit <= 60  THEN dx END IGNORE NULLS)       AS dx_60
    ,ARRAY_AGG(DISTINCT dx IGNORE NULLS)                                                      AS dx_180

FROM visit_pairs
CROSS JOIN UNNEST(future_specialty_codes)               AS specialty
CROSS JOIN UNNEST(future_provider_ids)                  AS provider
CROSS JOIN UNNEST(future_dx_list)                       AS dx
GROUP BY
    member_id
    ,visit_date
    ,visit_seq_num
ORDER BY
    member_id
    ,visit_seq_num
