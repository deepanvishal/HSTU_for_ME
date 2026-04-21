-- ============================================================
-- STAGING TABLE: BENEFICIARY COUNTS BY FLORIDA COUNTY
-- SOURCE: CMS MA STATE/COUNTY PENETRATION FILE
-- GRAIN: county_fips x county_name
-- FILTERED TO: FLORIDA (fipsst = '12') + LATEST INGEST ONLY
-- NOTE: penetration is in xx.xx% string format, cast to decimal
-- NOTE: fipscnty zero-padded to 3 digits + fipsst to build full 5-digit FIPS
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_beneficiaries`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH latest_period AS (
  -- --------------------------------------------------------
  -- GET MOST RECENT INGEST DATE
  -- --------------------------------------------------------
  SELECT MAX(ingest_time) AS max_ingest
  FROM `anbc-hcb-prod.provider_ds_netconf_data_hcb_prod.cms_medicare_penetration`
),

florida_penetration AS (
  -- --------------------------------------------------------
  -- FILTER TO FLORIDA + LATEST INGEST ONLY
  -- BUILD FULL 5-DIGIT COUNTY FIPS
  -- --------------------------------------------------------
  SELECT
    CONCAT(
      LPAD(CAST(fipsst   AS STRING), 2, '0'),
      LPAD(CAST(fipscnty AS STRING), 3, '0')
    )                                                                AS county_fips,
    county_name,
    eligibles,
    enrolled,
    penetration,
    ingest_time
  FROM `anbc-hcb-prod.provider_ds_netconf_data_hcb_prod.cms_medicare_penetration`
  CROSS JOIN latest_period
  WHERE fipsst = '12'
    AND ingest_time = latest_period.max_ingest
)

SELECT
  p.county_fips,
  p.county_name,
  p.eligibles                                                        AS eligible_beneficiaries,
  p.enrolled                                                         AS ma_enrolled,
  SAFE_CAST(REPLACE(p.penetration, '%', '') AS FLOAT64) / 100       AS penetration_rate,
  p.ingest_time                                                      AS data_as_of,
  c.county_type,
  c.compliance_threshold,
  c.pop_density
FROM florida_penetration p
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_county_classification` c
  ON p.county_fips = c.county_fips
ORDER BY p.county_name
