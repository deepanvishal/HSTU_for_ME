WITH dx_codes AS (
    SELECT 
        pri_icd9_dx_ccd AS diagnosis_code,
        COUNT(*) AS claim_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    WHERE srv_start_dt BETWEEN '2023-01-01' AND '2025-12-31'
        AND pri_icd9_dx_ccd IS NOT NULL
    GROUP BY pri_icd9_dx_ccd
),
matched AS (
    SELECT
        d.diagnosis_code,
        d.claim_count,
        CASE WHEN c.icd_10_cm_code IS NOT NULL THEN 1 ELSE 0 END AS matched
    FROM dx_codes d
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mw_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` c
        ON d.diagnosis_code = c.icd_10_cm_code
)
SELECT
    SUM(claim_count)                                                            AS total_claims,
    SUM(CASE WHEN matched = 1 THEN claim_count ELSE 0 END)                     AS matched_claims,
    SUM(CASE WHEN matched = 0 THEN claim_count ELSE 0 END)                     AS unmatched_claims,
    ROUND(SUM(CASE WHEN matched = 1 THEN claim_count ELSE 0 END) * 100.0 
          / SUM(claim_count), 2)                                                AS match_rate_pct
FROM matched

;

SELECT 
    d.diagnosis_code,
    COUNT(*) AS claim_count
FROM (
    SELECT pri_icd9_dx_ccd AS diagnosis_code
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    WHERE srv_start_dt BETWEEN '2023-01-01' AND '2025-12-31'
        AND pri_icd9_dx_ccd IS NOT NULL
) d
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_mw_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` c
    ON REPLACE(d.diagnosis_code, '.', '') = c.icd_10_cm_code
WHERE c.icd_10_cm_code IS NULL
GROUP BY d.diagnosis_code
ORDER BY claim_count DESC
LIMIT 50
;



-- ============================================================
-- ICD10 Diagnosis Code -> Next Visit Specialty Transition
-- ============================================================
-- current_dx           : raw ICD-10 diagnosis code from current visit
-- next_specialty       : specialty code of next visit
-- transition_count     : how many times this dx -> specialty transition occurred
-- unique_members       : how many distinct members drove this transition
-- dx_total             : total transitions originating from this dx code
-- conditional_probability : given this dx, probability next visit is this specialty
-- member_pct           : % of members with this dx who transition to this specialty
WITH enriched_claims AS (
    SELECT
        member_id
        ,srv_start_dt
        ,specialty_ctg_cd
        ,REPLACE(pri_icd9_dx_cd, '.', '')               AS diagnosis_code
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
    WHERE pri_icd9_dx_cd IS NOT NULL
        AND specialty_ctg_cd IS NOT NULL
),

visits AS (
    SELECT
        member_id
        ,srv_start_dt                                    AS visit_date
        ,ARRAY_AGG(DISTINCT diagnosis_code)              AS dx_list
        ,ARRAY_AGG(DISTINCT specialty_ctg_cd)            AS specialty_list
    FROM enriched_claims
    GROUP BY member_id, srv_start_dt
),

visit_pairs AS (
    SELECT
        v.member_id
        ,v.dx_list                                       AS current_dx_list
        ,LEAD(v.specialty_list) OVER (
            PARTITION BY v.member_id ORDER BY v.visit_date
        )                                                AS next_specialty_list
    FROM visits v
),

exploded AS (
    SELECT
        member_id
        ,current_dx
        ,next_specialty
    FROM visit_pairs
    CROSS JOIN UNNEST(current_dx_list)      AS current_dx
    CROSS JOIN UNNEST(next_specialty_list)  AS next_specialty
    WHERE next_specialty_list IS NOT NULL
),

transition_counts AS (
    SELECT
        current_dx
        ,next_specialty
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
    FROM exploded
    GROUP BY current_dx, next_specialty
),

dx_totals AS (
    SELECT
        current_dx
        ,SUM(transition_count)                           AS dx_total
    FROM transition_counts
    GROUP BY current_dx
)

SELECT
    t.current_dx
    ,t.next_specialty
    ,t.transition_count
    ,t.unique_members
    ,d.dx_total
    ,ROUND(t.transition_count / d.dx_total, 4)          AS conditional_probability
    ,ROUND(t.unique_members * 100.0 / 
        SUM(t.unique_members) OVER (
            PARTITION BY t.current_dx
        ), 2)                                            AS member_pct
FROM transition_counts t
JOIN dx_totals d ON t.current_dx = d.current_dx
ORDER BY t.transition_count DESC
LIMIT 100






















