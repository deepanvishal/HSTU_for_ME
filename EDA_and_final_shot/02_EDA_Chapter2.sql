-- ============================================================
-- CHAPTER 2: Claims Facts
-- Source: A870800_claims_gen_rec_2022_2025_sfl
-- ============================================================

-- Specialty classification used throughout this chapter:
-- PCP        = FP, I
-- Specialist = CARD, ENDO, NEPH, NEUR, ONCO, OPTH, ORTH, PEDS,
--              PSYC, PULM, RHEU, UROL, DERM, GAST, HEMA, INFD,
--              OBGY, OTOL, SURG, VASC
-- Other      = everything else

-- ------------------------------------------------------------
-- Q1. Total claim lines
-- ------------------------------------------------------------
SELECT
    COUNT(*)                                               AS total_claim_lines
    ,COUNT(DISTINCT member_id)                             AS total_claimants
    ,COUNT(DISTINCT srv_prvdr_id)                          AS total_providers
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
;

-- ------------------------------------------------------------
-- Q2. Claim lines by year and year-month
-- ------------------------------------------------------------
SELECT
    EXTRACT(YEAR FROM srv_start_dt)                        AS claim_year
    ,FORMAT_DATE('%Y-%m', srv_start_dt)                    AS claim_month
    ,COUNT(*)                                              AS claim_lines
    ,COUNT(DISTINCT member_id)                             AS claimants
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
GROUP BY claim_year, claim_month
ORDER BY claim_month
;

-- ------------------------------------------------------------
-- Q3. Claims per claimant distribution
-- ------------------------------------------------------------
SELECT
    MIN(claim_count)                                       AS min_claims
    ,MAX(claim_count)                                      AS max_claims
    ,ROUND(AVG(claim_count), 1)                            AS mean_claims
    ,APPROX_QUANTILES(claim_count, 100)[OFFSET(10)]        AS p10
    ,APPROX_QUANTILES(claim_count, 100)[OFFSET(25)]        AS p25
    ,APPROX_QUANTILES(claim_count, 100)[OFFSET(50)]        AS median
    ,APPROX_QUANTILES(claim_count, 100)[OFFSET(75)]        AS p75
    ,APPROX_QUANTILES(claim_count, 100)[OFFSET(90)]        AS p90
    ,APPROX_QUANTILES(claim_count, 100)[OFFSET(99)]        AS p99
FROM (
    SELECT
        member_id
        ,COUNT(*)                                          AS claim_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    GROUP BY member_id
)
;

-- ------------------------------------------------------------
-- Q4. Allowed amount stats
-- ------------------------------------------------------------
SELECT
    ROUND(SUM(allowed_amt), 0)                             AS total_allowed
    ,ROUND(AVG(allowed_amt), 2)                            AS mean_allowed
    ,APPROX_QUANTILES(allowed_amt, 100)[OFFSET(50)]        AS median_allowed
    ,APPROX_QUANTILES(allowed_amt, 100)[OFFSET(90)]        AS p90_allowed
    ,APPROX_QUANTILES(allowed_amt, 100)[OFFSET(99)]        AS p99_allowed
    ,MIN(allowed_amt)                                      AS min_allowed
    ,MAX(allowed_amt)                                      AS max_allowed
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
;

-- ------------------------------------------------------------
-- Q5. Place of service distribution — Inpatient vs Outpatient
-- ------------------------------------------------------------
SELECT
    plc_srv_ctg_cd
    ,med_cost_ctg_cd
    ,COUNT(*)                                              AS claim_lines
    ,COUNT(DISTINCT member_id)                             AS claimants
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct_claims
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
GROUP BY plc_srv_ctg_cd, med_cost_ctg_cd
ORDER BY claim_lines DESC
;

-- ------------------------------------------------------------
-- Q6. Top 20 specialties by claim volume
-- ------------------------------------------------------------
SELECT
    specialty_ctg_cd
    ,specialty_ctg_cd_desc
    ,COUNT(*)                                              AS claim_lines
    ,COUNT(DISTINCT member_id)                             AS claimants
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct_claims
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
GROUP BY specialty_ctg_cd, specialty_ctg_cd_desc
ORDER BY claim_lines DESC
LIMIT 20
;

-- ------------------------------------------------------------
-- Q7. Top 20 CCSR categories by claim volume
-- Source: visits table which has CCSR joined in
-- ------------------------------------------------------------
SELECT
    ccsr_category
    ,ccsr_category_description
    ,COUNT(*)                                              AS claim_lines
    ,COUNT(DISTINCT member_id)                             AS claimants
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct_claims
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
WHERE ccsr_category IS NOT NULL
GROUP BY ccsr_category, ccsr_category_description
ORDER BY claim_lines DESC
LIMIT 20
;

-- ------------------------------------------------------------
-- Q8. Visit type classification — PCP vs Specialist vs Other
-- Overall summary
-- ------------------------------------------------------------
SELECT
    CASE
        WHEN specialty_ctg_cd IN ('FP', 'I')
            THEN 'PCP'
        WHEN specialty_ctg_cd IN (
            'CARD', 'ENDO', 'NEPH', 'NEUR', 'ONCO',
            'OPTH', 'ORTH', 'PEDS', 'PSYC', 'PULM',
            'RHEU', 'UROL', 'DERM', 'GAST', 'HEMA',
            'INFD', 'OBGY', 'OTOL', 'SURG', 'VASC'
        )   THEN 'Specialist'
        ELSE    'Other'
    END                                                    AS visit_type
    ,COUNT(*)                                              AS claim_lines
    ,COUNT(DISTINCT member_id)                             AS claimants
    ,ROUND(SUM(allowed_amt), 0)                            AS total_allowed
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct_claims
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct_claimants
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
GROUP BY visit_type
ORDER BY claim_lines DESC
;

-- ------------------------------------------------------------
-- Q9. Specialist breakdown — claims and claimants
-- per each of the 20 specialist codes
-- ------------------------------------------------------------
SELECT
    specialty_ctg_cd
    ,specialty_ctg_cd_desc
    ,COUNT(*)                                              AS claim_lines
    ,COUNT(DISTINCT member_id)                             AS claimants
    ,ROUND(SUM(allowed_amt), 0)                            AS total_allowed
    ,ROUND(AVG(allowed_amt), 2)                            AS avg_allowed_per_claim
    ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct_of_specialist_claims
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
WHERE specialty_ctg_cd IN (
    'CARD', 'ENDO', 'NEPH', 'NEUR', 'ONCO',
    'OPTH', 'ORTH', 'PEDS', 'PSYC', 'PULM',
    'RHEU', 'UROL', 'DERM', 'GAST', 'HEMA',
    'INFD', 'OBGY', 'OTOL', 'SURG', 'VASC'
)
GROUP BY specialty_ctg_cd, specialty_ctg_cd_desc
ORDER BY claim_lines DESC
;

-- ------------------------------------------------------------
-- Q10. Claimant journey type — all combinations
-- 4 visit categories:
--   PCP        = FP, I
--   Specialist = 20 specialist codes
--   Inpatient  = specialty_ctg_cd starting with 'W'
--   Other      = everything else
-- Shows every observed combination and % of claimants
-- ------------------------------------------------------------
SELECT
    CONCAT(
        CASE WHEN has_pcp        = 1 THEN 'PCP '       ELSE '' END
        ,CASE WHEN has_specialist = 1 THEN 'Specialist ' ELSE '' END
        ,CASE WHEN has_inpatient  = 1 THEN 'Inpatient '  ELSE '' END
        ,CASE WHEN has_other      = 1 THEN 'Other'       ELSE '' END
    )                                                      AS journey_combination
    ,COUNT(DISTINCT member_id)                             AS members
    ,ROUND(100.0 * COUNT(DISTINCT member_id) /
        SUM(COUNT(DISTINCT member_id)) OVER (), 2)         AS pct
FROM (
    SELECT
        member_id
        ,MAX(CASE WHEN specialty_ctg_cd IN ('FP','I')
            THEN 1 ELSE 0 END)                             AS has_pcp
        ,MAX(CASE WHEN specialty_ctg_cd IN (
            'CARD', 'ENDO', 'NEPH', 'NEUR', 'ONCO',
            'OPTH', 'ORTH', 'PEDS', 'PSYC', 'PULM',
            'RHEU', 'UROL', 'DERM', 'GAST', 'HEMA',
            'INFD', 'OBGY', 'OTOL', 'SURG', 'VASC'
        ) THEN 1 ELSE 0 END)                               AS has_specialist
        ,MAX(CASE WHEN specialty_ctg_cd LIKE 'W%'
            THEN 1 ELSE 0 END)                             AS has_inpatient
        ,MAX(CASE WHEN specialty_ctg_cd NOT IN (
                'FP','I',
                'CARD', 'ENDO', 'NEPH', 'NEUR', 'ONCO',
                'OPTH', 'ORTH', 'PEDS', 'PSYC', 'PULM',
                'RHEU', 'UROL', 'DERM', 'GAST', 'HEMA',
                'INFD', 'OBGY', 'OTOL', 'SURG', 'VASC'
            )
            AND specialty_ctg_cd NOT LIKE 'W%'
            THEN 1 ELSE 0 END)                             AS has_other
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    GROUP BY member_id
)
GROUP BY journey_combination
ORDER BY members DESC
;
