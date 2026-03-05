-- ============================================================
-- TABLE 2: MEMBER PROFILE SNAPSHOT
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_member_profile_snapshot` AS

WITH member_visits AS (
    SELECT DISTINCT
        member_id
        ,visit_date
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_table`
)

,prior_claims AS (
    SELECT
        mv.member_id
        ,mv.visit_date
        ,c.pri_icd9_dx_ccd
        ,c.gender_cd
    FROM member_visits mv
    INNER JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl` c
        ON mv.member_id = c.member_id
        AND c.srv_start_dt < mv.visit_date
)

SELECT
    member_id
    ,visit_date
    ,ARRAY_AGG(DISTINCT pri_icd9_dx_ccd IGNORE NULLS) AS prior_dx_list
    ,MAX(gender_cd)                                    AS gender_cd
FROM prior_claims
GROUP BY
    member_id
    ,visit_date
ORDER BY
    member_id
    ,visit_date
