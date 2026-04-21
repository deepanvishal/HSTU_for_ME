-- ============================================================
-- STAGING TABLE: BENEFICIARY COUNTS BY FLORIDA COUNTY
-- SOURCE: CMS MA STATE/COUNTY PENETRATION FILE
-- GRAIN: county_fips x county_name
-- FILTERED TO: FLORIDA (state_fips = '12')
-- NOTE: penetration is in xx.xx% string format, cast to decimal
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_beneficiaries`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH florida_penetration AS (
  -- --------------------------------------------------------
  -- FILTER TO FLORIDA COUNTIES ONLY
  -- --------------------------------------------------------
  SELECT *
  FROM `anbc-hcb-prod.provider_ds_netconf_data_hcb_prod.cms_medicare_penetration`
  WHERE LEFT(fipscnty, 2) = '12'
)

SELECT
  p.fipscnty                                                        AS county_fips,
  p.county_name,
  p.eligibles                                                        AS eligible_beneficiaries,
  p.enrolled                                                         AS ma_enrolled,
  SAFE_CAST(REPLACE(p.penetration, '%', '') AS FLOAT64) / 100       AS penetration_rate,
  c.county_type,
  c.compliance_threshold,
  c.pop_density
FROM florida_penetration p
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_county_classification` c
  ON p.fipscnty = c.county_fips
ORDER BY p.county_name
