CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.visit_table` AS

WITH filtered_claims AS (
    SELECT
        member_id
        ,srv_start_dt
        ,srv_prvdr_id
        ,specialty_ctg_cd
        ,pri_icd9_dx_ccd
        ,plc_srv_cd
        ,prcdr_cd
        ,gender_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    WHERE specialty_ctg_cd IN (
            'CARD', 'ENDO', 'NEPH', 'NEUR', 'ONCO'
            ,'OPTH', 'ORTH', 'PEDS', 'PSYC', 'PULM'
            ,'RHEU', 'UROL', 'DERM', 'GAST', 'HEMA'
            ,'INFD', 'OBGY', 'OTOL', 'SURG', 'VASC'
            ,'PCP'
        )
        AND plc_srv_cd NOT IN ('E')
)

,member_visit_counts AS (
    SELECT
        member_id
        ,COUNT(DISTINCT srv_start_dt) AS total_visits
    FROM filtered_claims
    GROUP BY member_id
    HAVING total_visits >= 10
)

SELECT
    f.member_id
    ,f.srv_start_dt                          AS visit_date
    ,ARRAY_AGG(DISTINCT f.srv_prvdr_id)      AS provider_ids
    ,ARRAY_AGG(DISTINCT f.specialty_ctg_cd)  AS specialty_codes
    ,ARRAY_AGG(DISTINCT f.pri_icd9_dx_ccd)   AS dx_list
    ,ARRAY_AGG(DISTINCT f.prcdr_cd)          AS procedure_codes
    ,ARRAY_AGG(DISTINCT f.plc_srv_cd)        AS place_of_service
    ,MAX(f.gender_cd)                        AS gender_cd
FROM filtered_claims f
INNER JOIN member_visit_counts m
    ON f.member_id = m.member_id
GROUP BY
    f.member_id
    ,f.srv_start_dt
ORDER BY
    f.member_id
    ,f.srv_start_dt
