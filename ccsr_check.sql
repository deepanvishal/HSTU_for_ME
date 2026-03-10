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
