
-- ============================================================
-- TABLE 7: ref_county_name_crosswalk
-- PURPOSE: MAP AETNA COUNTY NAMES TO CENSUS COUNTY NAMES + FIPS
-- SOURCE:  mbr_with_zip (aetna) vs ref_county_classification (census)
-- GRAIN:   aetna_county_nm
-- KNOWN MISMATCHES:
--   Desoto      → DeSoto
--   Saint Johns → St. Johns
--   Saint Lucie → St. Lucie
-- NOTE: 41 Aetna counties vs 67 Census counties
--       26 counties have no Aetna providers - flagged as no_coverage
-- ============================================================
 
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_county_name_crosswalk`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT * FROM UNNEST([
  -- --------------------------------------------------------
  -- EXACT MATCHES (38 counties)
  -- --------------------------------------------------------
  STRUCT('Alachua'      AS aetna_county_nm, 'Alachua'      AS census_county_nm, '12001' AS county_fips, 'exact'      AS match_type),
  STRUCT('Baker',                           'Baker',                             '12003',               'exact'),
  STRUCT('Brevard',                         'Brevard',                           '12009',               'exact'),
  STRUCT('Broward',                         'Broward',                           '12011',               'exact'),
  STRUCT('Charlotte',                       'Charlotte',                         '12015',               'exact'),
  STRUCT('Citrus',                          'Citrus',                            '12017',               'exact'),
  STRUCT('Clay',                            'Clay',                              '12019',               'exact'),
  STRUCT('Collier',                         'Collier',                           '12021',               'exact'),
  STRUCT('Columbia',                        'Columbia',                          '12023',               'exact'),
  STRUCT('Duval',                           'Duval',                             '12031',               'exact'),
  STRUCT('Escambia',                        'Escambia',                          '12033',               'exact'),
  STRUCT('Flagler',                         'Flagler',                           '12035',               'exact'),
  STRUCT('Hernando',                        'Hernando',                          '12053',               'exact'),
  STRUCT('Highlands',                       'Highlands',                         '12055',               'exact'),
  STRUCT('Hillsborough',                    'Hillsborough',                      '12057',               'exact'),
  STRUCT('Indian River',                    'Indian River',                      '12061',               'exact'),
  STRUCT('Lake',                            'Lake',                              '12069',               'exact'),
  STRUCT('Lee',                             'Lee',                               '12071',               'exact'),
  STRUCT('Levy',                            'Levy',                              '12075',               'exact'),
  STRUCT('Manatee',                         'Manatee',                           '12081',               'exact'),
  STRUCT('Marion',                          'Marion',                            '12083',               'exact'),
  STRUCT('Martin',                          'Martin',                            '12085',               'exact'),
  STRUCT('Miami-Dade',                      'Miami-Dade',                        '12086',               'exact'),
  STRUCT('Nassau',                          'Nassau',                            '12089',               'exact'),
  STRUCT('Okaloosa',                        'Okaloosa',                          '12091',               'exact'),
  STRUCT('Orange',                          'Orange',                            '12095',               'exact'),
  STRUCT('Osceola',                         'Osceola',                           '12097',               'exact'),
  STRUCT('Palm Beach',                      'Palm Beach',                        '12099',               'exact'),
  STRUCT('Pasco',                           'Pasco',                             '12101',               'exact'),
  STRUCT('Pinellas',                        'Pinellas',                          '12103',               'exact'),
  STRUCT('Polk',                            'Polk',                              '12105',               'exact'),
  STRUCT('Putnam',                          'Putnam',                            '12107',               'exact'),
  STRUCT('Santa Rosa',                      'Santa Rosa',                        '12113',               'exact'),
  STRUCT('Sarasota',                        'Sarasota',                          '12115',               'exact'),
  STRUCT('Seminole',                        'Seminole',                          '12117',               'exact'),
  STRUCT('Sumter',                          'Sumter',                            '12119',               'exact'),
  STRUCT('Volusia',                         'Volusia',                           '12127',               'exact'),
  STRUCT('Walton',                          'Walton',                            '12131',               'exact'),
  -- --------------------------------------------------------
  -- NAME MISMATCH FIXES (3 counties)
  -- --------------------------------------------------------
  STRUCT('Desoto',                          'DeSoto',                            '12027',               'name_fix'),
  STRUCT('Saint Johns',                     'St. Johns',                         '12109',               'name_fix'),
  STRUCT('Saint Lucie',                     'St. Lucie',                         '12111',               'name_fix'),
  -- --------------------------------------------------------
  -- NO AETNA COVERAGE (26 counties)
  -- PRESENT IN CENSUS NOT IN AETNA PROVIDER FILE
  -- --------------------------------------------------------
  STRUCT(NULL,                              'Bay',                               '12005',               'no_coverage'),
  STRUCT(NULL,                              'Bradford',                          '12007',               'no_coverage'),
  STRUCT(NULL,                              'Calhoun',                           '12013',               'no_coverage'),
  STRUCT(NULL,                              'Dixie',                             '12029',               'no_coverage'),
  STRUCT(NULL,                              'Franklin',                          '12037',               'no_coverage'),
  STRUCT(NULL,                              'Gadsden',                           '12039',               'no_coverage'),
  STRUCT(NULL,                              'Gilchrist',                         '12041',               'no_coverage'),
  STRUCT(NULL,                              'Glades',                            '12043',               'no_coverage'),
  STRUCT(NULL,                              'Gulf',                              '12045',               'no_coverage'),
  STRUCT(NULL,                              'Hamilton',                          '12047',               'no_coverage'),
  STRUCT(NULL,                              'Hardee',                            '12049',               'no_coverage'),
  STRUCT(NULL,                              'Hendry',                            '12051',               'no_coverage'),
  STRUCT(NULL,                              'Holmes',                            '12059',               'no_coverage'),
  STRUCT(NULL,                              'Jackson',                           '12063',               'no_coverage'),
  STRUCT(NULL,                              'Jefferson',                         '12065',               'no_coverage'),
  STRUCT(NULL,                              'Lafayette',                         '12067',               'no_coverage'),
  STRUCT(NULL,                              'Leon',                              '12073',               'no_coverage'),
  STRUCT(NULL,                              'Liberty',                           '12077',               'no_coverage'),
  STRUCT(NULL,                              'Madison',                           '12079',               'no_coverage'),
  STRUCT(NULL,                              'Monroe',                            '12087',               'no_coverage'),
  STRUCT(NULL,                              'Okeechobee',                        '12093',               'no_coverage'),
  STRUCT(NULL,                              'Suwannee',                          '12121',               'no_coverage'),
  STRUCT(NULL,                              'Taylor',                            '12123',               'no_coverage'),
  STRUCT(NULL,                              'Union',                             '12125',               'no_coverage'),
  STRUCT(NULL,                              'Wakulla',                           '12129',               'no_coverage'),
  STRUCT(NULL,                              'Washington',                        '12133',               'no_coverage')
]);
 
 
-- ============================================================
-- TABLE 8: stg_providers
-- PURPOSE: SUPPLY SIDE - AETNA CONTRACTED PROVIDERS FOR FLORIDA
-- SOURCE:  A870800_medicare_supply_demand_mbr_with_zip
--          ref_specialty_crosswalk
--          ref_county_name_crosswalk
--          ref_zip_reference
-- GRAIN:   provider_id x cms_specialty x plan_type x zip_cd
-- NOTE:    ONE AETNA SPECIALTY → MULTIPLE CMS SPECIALTIES (fan out)
--          prod_type: HMO IVL = MA-HMO, PPO IVL = MA-PPO
--          zip_cd already 5 digits, no padding needed
--          snapshot table, no date filter needed
--          lat/long from ref_zip_reference via zip_cd join
-- ============================================================
 
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
 
WITH florida_providers AS (
  -- --------------------------------------------------------
  -- FILTER TO FLORIDA ONLY
  -- MAP prod_type TO CMS PLAN TYPE LABELS
  -- --------------------------------------------------------
  SELECT
    prvdr_id_no                                                      AS provider_id,
    tin_owner_nm                                                     AS provider_name,
    tax_id_no,
    specialty_ctg_cd,
    county_nm,
    zip_cd,
    market,
    submarket,
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
  -- INTENTIONAL FAN OUT: ONE AETNA CODE → MULTIPLE CMS SPECIALTIES
  -- E.G. VVRH → Physical Therapy + Occupational Therapy + Speech Therapy
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
  -- no_coverage COUNTIES → county_fips WILL BE NULL
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
  -- lat/long used in fact_distance_matrix for ST_DISTANCE calc
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
-- FINAL SELECT
-- DEDUP TO GRAIN: provider_id x cms_specialty x plan_type x zip_cd
-- EXCLUDE UNMAPPED SPECIALTIES
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
WHERE cms_specialty IS NOT NULL
ORDER BY provider_id, cms_specialty, plan_type
 
