SELECT DISTINCT
  county_nm,
  state
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_mbr_with_zip`
WHERE state = 'FL'
ORDER BY county_nm;

SELECT DISTINCT
  county_name,
  county_fips
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_county_classification`
ORDER BY county_name;

SELECT
  DISTINCT prod_type,
  LENGTH(zip_cd) AS zip_len,
  MIN(zip_cd) AS zip_min,
  MAX(zip_cd) AS zip_max
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_mbr_with_zip`
GROUP BY prod_type


-- ============================================================
-- TABLE: stg_providers
-- PURPOSE: SUPPLY SIDE - AETNA CONTRACTED PROVIDERS FOR FLORIDA
-- SOURCE:  A870800_medicare_supply_demand_mbr_with_zip
--          ref_specialty_crosswalk
--          ref_county_name_crosswalk
--          ref_zip_reference
-- GRAIN:   prvdr_id_no x cms_specialty x prod_type x zip_cd
-- NOTE:    ONE AETNA SPECIALTY → MULTIPLE CMS SPECIALTIES (fan out)
--          prod_type: HMO IVL = MA-HMO, PPO IVL = MA-PPO
--          zip_cd already 5 digits, no padding needed
--          snapshot table, no date filter needed
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH florida_providers AS (
  -- --------------------------------------------------------
  -- FILTER TO FLORIDA PROVIDERS ONLY
  -- --------------------------------------------------------
  SELECT
    prvdr_id_no                                                      AS provider_id,
    tin_owner_nm                                                     AS provider_name,
    tax_id_no,
    specialty_ctg_cd,
    county_nm,
    zip_cd,
    prod_type,
    market,
    submarket,
    -- map prod_type to CMS plan type
    CASE
      WHEN prod_type = 'HMO IVL' THEN 'MA-HMO'
      WHEN prod_type = 'PPO IVL' THEN 'MA-PPO'
      ELSE prod_type
    END                                                              AS plan_type
  FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_mbr_with_zip`
  WHERE state = 'FL'
),

mapped_specialty AS (
  -- --------------------------------------------------------
  -- JOIN TO SPECIALTY CROSSWALK
  -- ONE AETNA CODE → MULTIPLE CMS SPECIALTIES (intentional fan out)
  -- E.G. VVRH → Physical Therapy, Occupational Therapy, Speech Therapy
  -- --------------------------------------------------------
  SELECT
    p.provider_id,
    p.provider_name,
    p.tax_id_no,
    p.specialty_ctg_cd                                               AS aetna_specialty_cd,
    s.cms_specialty,
    s.match_type,
    s.inflated,
    p.county_nm,
    p.zip_cd,
    p.plan_type,
    p.market,
    p.submarket
  FROM florida_providers p
  LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_specialty_crosswalk` s
    ON p.specialty_ctg_cd = s.aetna_cd
),

mapped_county AS (
  -- --------------------------------------------------------
  -- JOIN TO COUNTY NAME CROSSWALK
  -- RESOLVES AETNA NAME MISMATCHES (Desoto, Saint Johns, Saint Lucie)
  -- PROVIDERS IN no_coverage COUNTIES WILL HAVE NULL county_fips
  -- --------------------------------------------------------
  SELECT
    m.provider_id,
    m.provider_name,
    m.tax_id_no,
    m.aetna_specialty_cd,
    m.cms_specialty,
    m.match_type,
    m.inflated,
    m.county_nm                                                      AS aetna_county_nm,
    c.census_county_nm,
    c.county_fips,
    m.zip_cd,
    m.plan_type,
    m.market,
    m.submarket
  FROM mapped_specialty m
  LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_county_name_crosswalk` c
    ON m.county_nm = c.aetna_county_nm
),

mapped_zip AS (
  -- --------------------------------------------------------
  -- JOIN TO ZIP REFERENCE FOR LAT/LONG + COUNTY TYPE
  -- --------------------------------------------------------
  SELECT
    m.provider_id,
    m.provider_name,
    m.tax_id_no,
    m.aetna_specialty_cd,
    m.cms_specialty,
    m.match_type,
    m.inflated,
    m.aetna_county_nm,
    m.census_county_nm,
    m.county_fips,
    m.zip_cd,
    z.zip_lat,
    z.zip_long,
    z.zip_centroid,
    z.zip_radius_miles,
    z.county_type,
    m.plan_type,
    m.market,
    m.submarket
  FROM mapped_county m
  LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_zip_reference` z
    ON m.zip_cd = z.zip_code
)

-- --------------------------------------------------------
-- DEDUP TO GRAIN: provider_id x cms_specialty x plan_type x zip_cd
-- HANDLES DUPLICATE ROWS FROM SOURCE TABLE
-- --------------------------------------------------------
SELECT DISTINCT
  provider_id,
  provider_name,
  tax_id_no,
  aetna_specialty_cd,
  cms_specialty,
  match_type,
  inflated,
  aetna_county_nm,
  census_county_nm,
  county_fips,
  zip_cd,
  zip_lat,
  zip_long,
  zip_centroid,
  zip_radius_miles,
  county_type,
  plan_type,
  market,
  submarket
FROM mapped_zip
WHERE cms_specialty IS NOT NULL  -- exclude unmapped specialties
ORDER BY provider_id, cms_specialty, plan_type
