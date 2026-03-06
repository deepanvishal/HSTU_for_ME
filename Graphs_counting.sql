-- ============================================================
-- GRAPH 1: PROVIDER CO-OCCURRENCE EDGES
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_provider_edges` AS

WITH sampled_members AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence`
    WHERE RAND() < 0.1
)
,exploded AS (
    SELECT
        p1 AS provider_1
        ,p2 AS provider_2
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence` s
    INNER JOIN sampled_members m ON s.member_id = m.member_id
    CROSS JOIN UNNEST(s.provider_ids) AS p1
    CROSS JOIN UNNEST(s.provider_ids) AS p2
    WHERE p1 < p2
)
SELECT
    provider_1
    ,provider_2
    ,COUNT(*) AS weight
FROM exploded
GROUP BY provider_1, provider_2

-- ============================================================
-- GRAPH 2: SPECIALTY CO-OCCURRENCE EDGES
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_specialty_edges` AS

WITH sampled_members AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence`
    WHERE RAND() < 0.1
)
,exploded AS (
    SELECT
        s1 AS specialty_1
        ,s2 AS specialty_2
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence` s
    INNER JOIN sampled_members m ON s.member_id = m.member_id
    CROSS JOIN UNNEST(s.specialty_codes) AS s1
    CROSS JOIN UNNEST(s.specialty_codes) AS s2
    WHERE s1 < s2
)
SELECT
    specialty_1
    ,specialty_2
    ,COUNT(*) AS weight
FROM exploded
GROUP BY specialty_1, specialty_2

-- ============================================================
-- GRAPH 3: DIAGNOSIS CO-OCCURRENCE EDGES
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_dx_edges` AS

WITH sampled_members AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence`
    WHERE RAND() < 0.1
)
,exploded AS (
    SELECT
        d1 AS dx_1
        ,d2 AS dx_2
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence` s
    INNER JOIN sampled_members m ON s.member_id = m.member_id
    CROSS JOIN UNNEST(s.dx_list) AS d1
    CROSS JOIN UNNEST(s.dx_list) AS d2
    WHERE d1 < d2
)
SELECT
    dx_1
    ,dx_2
    ,COUNT(*) AS weight
FROM exploded
GROUP BY dx_1, dx_2


-- ============================================================
-- GRAPH 4: PROCEDURE CO-OCCURRENCE EDGES
-- ============================================================
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_procedure_edges` AS

WITH sampled_members AS (
    SELECT DISTINCT member_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence`
    WHERE RAND() < 0.1
)
,exploded AS (
    SELECT
        p1 AS procedure_1
        ,p2 AS procedure_2
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence` s
    INNER JOIN sampled_members m ON s.member_id = m.member_id
    CROSS JOIN UNNEST(s.procedure_codes) AS p1
    CROSS JOIN UNNEST(s.procedure_codes) AS p2
    WHERE p1 < p2
)
SELECT
    procedure_1
    ,procedure_2
    ,COUNT(*) AS weight
FROM exploded
GROUP BY procedure_1, procedure_2


SELECT
    specialty_ctg_cd AS specialty
    ,ROW_NUMBER() OVER (ORDER BY specialty_ctg_cd) - 1 AS idx
FROM (
    SELECT DISTINCT specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    WHERE specialty_ctg_cd IS NOT NULL
)
ORDER BY specialty_ctg_cd
