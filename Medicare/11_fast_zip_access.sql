-- ============================================================
-- TABLE 9: fact_zip_access
-- PURPOSE: FOR EACH BENEFICIARY ZIP x SPECIALTY x PLAN TYPE
--          COUNT PROVIDERS WITHIN CMS DISTANCE THRESHOLD
--          TAG ZIP AS HAS_ACCESS OR NO_ACCESS
-- SOURCE:  stg_beneficiaries
--          stg_providers
--          ref_zip_reference
--          ref_time_distance
-- GRAIN:   bene_zip x cms_specialty x plan_type
-- NOTE:    THRESHOLD BASED ON BENEFICIARY COUNTY TYPE
--          ST_DISTANCE in meters converted to miles
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_fact_zip_access`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH zip_provider_pairs AS (
  -- --------------------------------------------------------
  -- FOR EACH BENE ZIP x PROVIDER x SPECIALTY x PLAN TYPE
  -- FILTER TO PAIRS WITHIN CMS DISTANCE THRESHOLD
  -- THRESHOLD USES BENEFICIARY COUNTY TYPE
  -- --------------------------------------------------------
  SELECT
    b.zip_code                                                       AS bene_zip,
    b.county_fips                                                    AS bene_county_fips,
    b.county_name                                                    AS bene_county_name,
    b.county_type                                                    AS bene_county_type,
    b.compliance_threshold,
    b.total_population                                               AS bene_zip_population,
    b.zip_radius_miles                                               AS bene_zip_radius,
    p.provider_id,
    p.cms_specialty,
    p.plan_type,
    p.inflated,
    p.match_type,
    t.max_distance_miles,
    ROUND(
      ST_DISTANCE(
        ST_GEOGPOINT(bene_zip.zip_long, bene_zip.zip_lat),
        ST_GEOGPOINT(p.zip_long,        p.zip_lat)
      ) / 1609.34
    , 2)                                                             AS distance_miles
  FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_beneficiaries` b

  -- get bene zip lat/long from ref_zip_reference
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_zip_reference` bene_zip
    ON b.zip_code = bene_zip.zip_code

  -- cross join providers
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers` p
    ON TRUE

  -- threshold lookup using bene county type
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_time_distance` t
    ON t.cms_specialty = p.cms_specialty
    AND t.county_type  = b.county_type

  -- core filter: only pairs within threshold survive
  WHERE ST_DISTANCE(
          ST_GEOGPOINT(bene_zip.zip_long, bene_zip.zip_lat),
          ST_GEOGPOINT(p.zip_long,        p.zip_lat)
        ) / 1609.34 <= t.max_distance_miles
)

-- --------------------------------------------------------
-- AGGREGATE TO GRAIN: bene_zip x cms_specialty x plan_type
-- COUNT DISTINCT PROVIDERS WITHIN THRESHOLD
-- TAG HAS_ACCESS
-- --------------------------------------------------------
SELECT
  bene_zip,
  bene_county_fips,
  bene_county_name,
  bene_county_type,
  compliance_threshold,
  bene_zip_population,
  cms_specialty,
  plan_type,
  inflated,
  match_type,
  max_distance_miles,
  COUNT(DISTINCT provider_id)                                        AS provider_count_within_threshold,
  CASE
    WHEN COUNT(DISTINCT provider_id) >= 1 THEN TRUE
    ELSE FALSE
  END                                                                AS has_access
FROM zip_provider_pairs
GROUP BY
  bene_zip,
  bene_county_fips,
  bene_county_name,
  bene_county_type,
  compliance_threshold,
  bene_zip_population,
  cms_specialty,
  plan_type,
  inflated,
  match_type,
  max_distance_miles;


-- ============================================================
-- TABLE 10: fact_gap_analysis
-- PURPOSE: COUNTY LEVEL COMPLIANCE ROLLUP
--          % BENEFICIARIES WITH ACCESS vs CMS THRESHOLD
--          ACTUAL vs REQUIRED PROVIDER COUNT
-- SOURCE:  fact_zip_access
--          ref_min_ratio
--          stg_beneficiaries (county_eligibles denominator)
-- GRAIN:   county x cms_specialty x plan_type
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_fact_gap_analysis`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH county_totals AS (
  -- --------------------------------------------------------
  -- GET TOTAL POPULATION AND ELIGIBLES PER COUNTY
  -- USED AS DENOMINATOR FOR PCT_COVERED
  -- --------------------------------------------------------
  SELECT
    county_fips,
    county_name,
    county_type,
    compliance_threshold,
    SUM(total_population)                                            AS total_zip_population,
    MAX(county_eligibles)                                            AS county_eligibles
  FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_beneficiaries`
  GROUP BY
    county_fips,
    county_name,
    county_type,
    compliance_threshold
),

zip_access_with_nulls AS (
  -- --------------------------------------------------------
  -- INCLUDE ZIPS WITH NO PROVIDERS (NO_ACCESS)
  -- LEFT JOIN FROM BENEFICIARIES TO ENSURE ALL ZIPS INCLUDED
  -- EVEN THOSE WITH ZERO PROVIDERS WITHIN THRESHOLD
  -- --------------------------------------------------------
  SELECT
    b.zip_code,
    b.county_fips,
    b.county_name,
    b.county_type,
    b.compliance_threshold,
    b.total_population,
    b.county_eligibles,
    sc.cms_specialty,
    sc.aetna_cd,
    pt.plan_type,
    -- if zip has access record use it, else default to no access
    COALESCE(za.provider_count_within_threshold, 0)                 AS provider_count,
    COALESCE(za.has_access, FALSE)                                   AS has_access,
    COALESCE(za.inflated, FALSE)                                     AS inflated,
    COALESCE(za.match_type, sc.match_type)                          AS match_type
  FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_beneficiaries` b

  -- cross join specialty list to ensure all specialties represented
  CROSS JOIN (
    SELECT DISTINCT cms_specialty, aetna_cd, match_type, inflated
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_specialty_crosswalk`
  ) sc

  -- cross join plan types
  CROSS JOIN (
    SELECT DISTINCT plan_type
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers`
  ) pt

  -- left join access results
  LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_fact_zip_access` za
    ON b.zip_code      = za.bene_zip
    AND sc.cms_specialty = za.cms_specialty
    AND pt.plan_type     = za.plan_type
),

county_rollup AS (
  -- --------------------------------------------------------
  -- ROLL UP TO COUNTY x SPECIALTY x PLAN TYPE
  -- --------------------------------------------------------
  SELECT
    county_fips,
    county_name,
    county_type,
    compliance_threshold,
    cms_specialty,
    plan_type,
    inflated,
    match_type,
    county_eligibles,
    SUM(total_population)                                            AS total_county_population,
    -- numerator: population in zips with at least 1 provider
    SUM(CASE WHEN has_access THEN total_population ELSE 0 END)      AS population_with_access,
    -- pct beneficiaries covered
    ROUND(
      SUM(CASE WHEN has_access THEN total_population ELSE 0 END)
      / NULLIF(SUM(total_population), 0)
    , 4)                                                             AS pct_covered,
    -- actual provider count: distinct providers in county within threshold
    SUM(provider_count)                                              AS actual_provider_count
  FROM zip_access_with_nulls
  GROUP BY
    county_fips,
    county_name,
    county_type,
    compliance_threshold,
    cms_specialty,
    plan_type,
    inflated,
    match_type,
    county_eligibles
)

-- --------------------------------------------------------
-- FINAL GAP ANALYSIS WITH COMPLIANCE FLAGS
-- --------------------------------------------------------
SELECT
  r.county_fips,
  r.county_name,
  r.county_type,
  r.cms_specialty,
  r.plan_type,
  r.inflated,
  r.match_type,
  r.county_eligibles,
  r.total_county_population,
  r.population_with_access,
  r.pct_covered,
  r.compliance_threshold,
  -- required provider count per 422.116
  CEIL(m.min_ratio_per_1000 * r.county_eligibles / 1000)           AS required_provider_count,
  r.actual_provider_count,
  -- gap: negative = surplus, positive = shortage
  CEIL(m.min_ratio_per_1000 * r.county_eligibles / 1000)
    - r.actual_provider_count                                        AS provider_gap,
  -- compliance flags
  CASE
    WHEN r.pct_covered >= r.compliance_threshold THEN TRUE
    ELSE FALSE
  END                                                                AS access_compliant,
  CASE
    WHEN r.actual_provider_count >=
         CEIL(m.min_ratio_per_1000 * r.county_eligibles / 1000)    THEN TRUE
    ELSE FALSE
  END                                                                AS count_compliant,
  -- overall compliance: both tests must pass
  CASE
    WHEN r.pct_covered >= r.compliance_threshold
    AND  r.actual_provider_count >=
         CEIL(m.min_ratio_per_1000 * r.county_eligibles / 1000)    THEN 'COMPLIANT'
    ELSE 'NON-COMPLIANT'
  END                                                                AS compliance_status
FROM county_rollup r
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_min_ratio` m
  ON m.cms_specialty = r.cms_specialty
  AND m.county_type  = r.county_type
ORDER BY
  r.county_name,
  r.cms_specialty,
  r.plan_type
