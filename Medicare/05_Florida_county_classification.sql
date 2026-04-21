-- ============================================================
-- REF TABLE: FLORIDA COUNTY CLASSIFICATION
-- SOURCE: bigquery-public-data.geo_us_boundaries.counties
--         + bigquery-public-data.census_bureau_acs.county_2020_5yr
-- GRAIN: county_fips x county_name
-- COUNTY TYPE LOGIC PER 42 CFR 422.116
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_county_classification`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH raw_counties AS (
  -- --------------------------------------------------------
  -- PULL FLORIDA COUNTY BOUNDARIES + AREA FROM PUBLIC GEO TABLE
  -- --------------------------------------------------------
  SELECT
    geo.county_fips_code                                             AS county_fips,
    geo.county_name,
    geo.state_fips_code,
    geo.area_land_meters / 2589988.11                               AS area_sq_miles
  FROM `bigquery-public-data.geo_us_boundaries.counties` geo
  WHERE geo.state_fips_code = '12'
),

population AS (
  -- --------------------------------------------------------
  -- PULL FLORIDA COUNTY POPULATION FROM ACS 5-YEAR ESTIMATES
  -- --------------------------------------------------------
  SELECT
    geo_id                                                           AS county_fips,
    total_pop
  FROM `bigquery-public-data.census_bureau_acs.county_2020_5yr`
  WHERE state_fips_code = '12'
),

joined AS (
  -- --------------------------------------------------------
  -- JOIN GEO + POPULATION
  -- --------------------------------------------------------
  SELECT
    r.county_fips,
    r.county_name,
    p.total_pop                                                      AS population,
    r.area_sq_miles,
    ROUND(p.total_pop / NULLIF(r.area_sq_miles, 0), 2)             AS pop_density
  FROM raw_counties r
  LEFT JOIN population p
    ON r.county_fips = p.county_fips
),

classified AS (
  -- --------------------------------------------------------
  -- APPLY 42 CFR 422.116 COUNTY TYPE CLASSIFICATION RULES
  -- PRIORITY ORDER: LARGE METRO > METRO > MICRO > RURAL > CEAC
  -- --------------------------------------------------------
  SELECT
    county_fips,
    county_name,
    population,
    area_sq_miles,
    pop_density,
    CASE
      -- LARGE METRO
      WHEN (population >= 1000000 AND pop_density >= 1000)
        OR (population >= 500000  AND pop_density >= 1500)
        OR (pop_density >= 5000)                                     THEN 'Large Metro'
      -- METRO
      WHEN (population >= 1000000 AND pop_density >= 10)
        OR (population >= 500000  AND pop_density >= 10)
        OR (population >= 200000  AND pop_density >= 10)
        OR (population >= 50000   AND pop_density >= 100)
        OR (population >= 10000   AND pop_density >= 1000)          THEN 'Metro'
      -- MICRO
      WHEN (population >= 50000   AND pop_density >= 10)
        OR (population >= 10000   AND pop_density >= 50)            THEN 'Micro'
      -- CEAC (check before rural - density < 10 wins regardless of pop)
      WHEN pop_density < 10                                          THEN 'CEAC'
      -- RURAL
      WHEN (population >= 10000   AND pop_density >= 10)
        OR (population < 10000    AND pop_density >= 50)            THEN 'Rural'
      ELSE 'Rural'
    END                                                              AS county_type
  FROM joined
)

SELECT
  county_fips,
  county_name,
  population,
  area_sq_miles,
  pop_density,
  county_type,
  -- --------------------------------------------------------
  -- COMPLIANCE THRESHOLD PER 422.116 d(4)
  -- 90% for Large Metro + Metro
  -- 85% for Micro, Rural, CEAC
  -- --------------------------------------------------------
  CASE
    WHEN county_type IN ('Large Metro', 'Metro') THEN 0.90
    ELSE 0.85
  END                                                                AS compliance_threshold
FROM classified
ORDER BY county_type, county_name
