CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.sequence_table` AS

WITH visit_with_lag AS (
    SELECT
        member_id
        ,visit_date
        ,provider_ids
        ,specialty_codes
        ,dx_list
        ,procedure_codes
        ,place_of_service
        ,gender_cd
        ,ROW_NUMBER() OVER (PARTITION BY member_id ORDER BY visit_date)            AS visit_seq_num
        ,DATE_DIFF(visit_date, LAG(visit_date) OVER (PARTITION BY member_id ORDER BY visit_date), DAY) AS delta_t
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.visit_table`
)

SELECT
    v.member_id
    ,v.visit_date
    ,v.visit_seq_num
    ,COALESCE(v.delta_t, 0)                            AS delta_t
    ,FLOOR(LOG10(GREATEST(1, ABS(COALESCE(delta_t, 0)))) / 0.301) AS delta_t_bucket
    ,v.provider_ids
    ,v.specialty_codes
    ,v.dx_list
    ,v.procedure_codes
    ,v.place_of_service
    ,v.gender_cd
    ,m.prior_dx_list
FROM visit_with_lag v
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.member_profile_snapshot` m
    ON v.member_id = m.member_id
    AND v.visit_date = m.visit_date
ORDER BY
    v.member_id
    ,v.visit_seq_num
