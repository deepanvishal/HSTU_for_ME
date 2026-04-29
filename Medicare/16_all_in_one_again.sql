-- ============================================================
-- MEDICARE NETWORK ADEQUACY & CAPACITY MODELING
-- MASTER SQL FILE - ALL STEPS IN EXECUTION ORDER
--
-- PROJECT:  anbc-hcb-dev
-- DATASET:  provider_ds_netconf_data_hcb_dev
-- PREFIX:   A870800_medicare_supply_demand_
-- AUTHOR:   deepan_thulasi_aetna_com
-- SOURCE:   42 CFR 422.116, CMS 2026 HSD Reference File
-- SCOPE:    Florida only
--
-- EXECUTION ORDER:
--   STEP 1:  ref_specialty_crosswalk
--   STEP 2:  ref_time_distance
--   STEP 3:  ref_county_classification
--   STEP 4:  ref_zip_reference
--   STEP 5:  ref_county_name_crosswalk
--   STEP 6:  ref_hsd_required_counts
--   STEP 7:  stg_beneficiaries
--   STEP 8:  stg_providers_multi_specialty
--   STEP 9:  fact_zip_access
--   STEP 10: fact_gap_analysis
-- ============================================================


-- ============================================================
-- STEP 1: ref_specialty_crosswalk
-- WHAT:   Maps CMS 422.116 specialty names to Aetna internal
--         specialty codes. One CMS specialty can map to multiple
--         Aetna codes and vice versa.
-- WHY:    Aetna and CMS use different specialty coding systems.
--         This crosswalk is the bridge for all downstream joins.
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_specialty_crosswalk`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT * FROM UNNEST([
  STRUCT('Primary Care'                  AS cms_specialty, 'FP'   AS aetna_cd),
  STRUCT('Primary Care',                                   'I'),
  STRUCT('Allergy and Immunology',                         'A'),
  STRUCT('Cardiology',                                     'C'),
  STRUCT('Chiropractor',                                   'VVCH'),
  STRUCT('Clinical Psychology',                            'VVMH'),
  STRUCT('Clinical Social Work',                           'VVMH'),
  STRUCT('Dermatology',                                    'D'),
  STRUCT('Endocrinology',                                  'E'),
  STRUCT('ENT/Otolaryngology',                             'EN'),
  STRUCT('Gastroenterology',                               'G'),
  STRUCT('General Surgery',                                'S'),
  STRUCT('Gynecology OB/GYN',                              'OG'),
  STRUCT('Infectious Diseases',                            'II'),
  STRUCT('Nephrology',                                     'N'),
  STRUCT('Neurology',                                      'NE'),
  STRUCT('Neurosurgery',                                   'NS'),
  STRUCT('Oncology Medical/Surgical',                      'H'),
  STRUCT('Oncology Radiation',                             'RO'),
  STRUCT('Ophthalmology',                                  'O'),
  STRUCT('Orthopedic Surgery',                             'OR'),
  STRUCT('Physiatry Rehabilitative Med',                   'VVRH'),
  STRUCT('Plastic Surgery',                                'PS'),
  STRUCT('Podiatry',                                       'VVPD'),
  STRUCT('Psychiatry',                                     'PY'),
  STRUCT('Pulmonology',                                    'PD'),
  STRUCT('Rheumatology',                                   'RH'),
  STRUCT('Urology',                                        'U'),
  STRUCT('Vascular Surgery',                               'VS'),
  STRUCT('Cardiothoracic Surgery',                         'CS'),
  STRUCT('Acute Inpatient Hospitals',                      'WHOS'),
  STRUCT('Cardiac Surgery Program',                        'CS'),
  STRUCT('Cardiac Catheterization',                        'C'),
  STRUCT('Critical Care ICU',                              'VICU'),
  STRUCT('Surgical Services ASC',                          'WASF'),
  STRUCT('Skilled Nursing Facility',                       'WLTC'),
  STRUCT('Diagnostic Radiology',                           'WRAD'),
  STRUCT('Mammography',                                    'VRAD'),
  STRUCT('Physical Therapy',                               'VVRH'),
  STRUCT('Occupational Therapy',                           'VVRH'),
  STRUCT('Speech Therapy',                                 'VVRH'),
  STRUCT('Inpatient Psychiatric',                          'WBHF'),
  STRUCT('Outpatient Infusion/Chemo',                      'WHOS'),
  STRUCT('Outpatient Behavioral Health',                   'WBHF')
]);


-- ============================================================
-- STEP 2: ref_time_distance
-- WHAT:   Maximum time and distance thresholds per specialty
--         per county type. Directly from 42 CFR 422.116 Table 1.
-- WHY:    Used in fact_zip_access to filter provider-beneficiary
--         pairs. Only pairs within this threshold survive.
-- NOTE:   Threshold uses BENEFICIARY county type, not provider.
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_time_distance`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT * FROM UNNEST([
  -- Primary Care
  STRUCT('Primary Care' AS cms_specialty, 'Large Metro' AS county_type, 10 AS max_time_min, 5   AS max_distance_miles),
  STRUCT('Primary Care', 'Metro',  15,  10),
  STRUCT('Primary Care', 'Micro',  30,  20),
  STRUCT('Primary Care', 'Rural',  40,  30),
  STRUCT('Primary Care', 'CEAC',   70,  60),
  -- Allergy and Immunology
  STRUCT('Allergy and Immunology', 'Large Metro',  30,  15),
  STRUCT('Allergy and Immunology', 'Metro',        45,  30),
  STRUCT('Allergy and Immunology', 'Micro',        80,  60),
  STRUCT('Allergy and Immunology', 'Rural',        90,  75),
  STRUCT('Allergy and Immunology', 'CEAC',        125, 110),
  -- Cardiology
  STRUCT('Cardiology', 'Large Metro', 20, 10),
  STRUCT('Cardiology', 'Metro',       30, 20),
  STRUCT('Cardiology', 'Micro',       50, 35),
  STRUCT('Cardiology', 'Rural',       75, 60),
  STRUCT('Cardiology', 'CEAC',        95, 85),
  -- Chiropractor
  STRUCT('Chiropractor', 'Large Metro',  30,  15),
  STRUCT('Chiropractor', 'Metro',        45,  30),
  STRUCT('Chiropractor', 'Micro',        80,  60),
  STRUCT('Chiropractor', 'Rural',        90,  75),
  STRUCT('Chiropractor', 'CEAC',        125, 110),
  -- Clinical Psychology
  STRUCT('Clinical Psychology', 'Large Metro',  20,  10),
  STRUCT('Clinical Psychology', 'Metro',        45,  30),
  STRUCT('Clinical Psychology', 'Micro',        60,  45),
  STRUCT('Clinical Psychology', 'Rural',        75,  60),
  STRUCT('Clinical Psychology', 'CEAC',        145, 130),
  -- Clinical Social Work
  STRUCT('Clinical Social Work', 'Large Metro',  20,  10),
  STRUCT('Clinical Social Work', 'Metro',        30,  20),
  STRUCT('Clinical Social Work', 'Micro',        50,  35),
  STRUCT('Clinical Social Work', 'Rural',        75,  60),
  STRUCT('Clinical Social Work', 'CEAC',        125, 110),
  -- Dermatology
  STRUCT('Dermatology', 'Large Metro',  20,  10),
  STRUCT('Dermatology', 'Metro',        45,  30),
  STRUCT('Dermatology', 'Micro',        60,  45),
  STRUCT('Dermatology', 'Rural',        75,  60),
  STRUCT('Dermatology', 'CEAC',        110, 100),
  -- Endocrinology
  STRUCT('Endocrinology', 'Large Metro',  30,  15),
  STRUCT('Endocrinology', 'Metro',        60,  40),
  STRUCT('Endocrinology', 'Micro',       100,  75),
  STRUCT('Endocrinology', 'Rural',       110,  90),
  STRUCT('Endocrinology', 'CEAC',        145, 130),
  -- ENT/Otolaryngology
  STRUCT('ENT/Otolaryngology', 'Large Metro',  30,  15),
  STRUCT('ENT/Otolaryngology', 'Metro',        45,  30),
  STRUCT('ENT/Otolaryngology', 'Micro',        80,  60),
  STRUCT('ENT/Otolaryngology', 'Rural',        90,  75),
  STRUCT('ENT/Otolaryngology', 'CEAC',        125, 110),
  -- Gastroenterology
  STRUCT('Gastroenterology', 'Large Metro',  20,  10),
  STRUCT('Gastroenterology', 'Metro',        45,  30),
  STRUCT('Gastroenterology', 'Micro',        60,  45),
  STRUCT('Gastroenterology', 'Rural',        75,  60),
  STRUCT('Gastroenterology', 'CEAC',        110, 100),
  -- General Surgery
  STRUCT('General Surgery', 'Large Metro', 20, 10),
  STRUCT('General Surgery', 'Metro',       30, 20),
  STRUCT('General Surgery', 'Micro',       50, 35),
  STRUCT('General Surgery', 'Rural',       75, 60),
  STRUCT('General Surgery', 'CEAC',        95, 85),
  -- Gynecology OB/GYN
  STRUCT('Gynecology OB/GYN', 'Large Metro',  30,  15),
  STRUCT('Gynecology OB/GYN', 'Metro',        45,  30),
  STRUCT('Gynecology OB/GYN', 'Micro',        80,  60),
  STRUCT('Gynecology OB/GYN', 'Rural',        90,  75),
  STRUCT('Gynecology OB/GYN', 'CEAC',        125, 110),
  -- Infectious Diseases
  STRUCT('Infectious Diseases', 'Large Metro',  30,  15),
  STRUCT('Infectious Diseases', 'Metro',        60,  40),
  STRUCT('Infectious Diseases', 'Micro',       100,  75),
  STRUCT('Infectious Diseases', 'Rural',       110,  90),
  STRUCT('Infectious Diseases', 'CEAC',        145, 130),
  -- Nephrology
  STRUCT('Nephrology', 'Large Metro',  30,  15),
  STRUCT('Nephrology', 'Metro',        45,  30),
  STRUCT('Nephrology', 'Micro',        80,  60),
  STRUCT('Nephrology', 'Rural',        90,  75),
  STRUCT('Nephrology', 'CEAC',        125, 110),
  -- Neurology
  STRUCT('Neurology', 'Large Metro',  20,  10),
  STRUCT('Neurology', 'Metro',        45,  30),
  STRUCT('Neurology', 'Micro',        60,  45),
  STRUCT('Neurology', 'Rural',        75,  60),
  STRUCT('Neurology', 'CEAC',        110, 100),
  -- Neurosurgery
  STRUCT('Neurosurgery', 'Large Metro',  30,  15),
  STRUCT('Neurosurgery', 'Metro',        60,  40),
  STRUCT('Neurosurgery', 'Micro',       100,  75),
  STRUCT('Neurosurgery', 'Rural',       110,  90),
  STRUCT('Neurosurgery', 'CEAC',        145, 130),
  -- Oncology Medical/Surgical
  STRUCT('Oncology Medical/Surgical', 'Large Metro',  20,  10),
  STRUCT('Oncology Medical/Surgical', 'Metro',        45,  30),
  STRUCT('Oncology Medical/Surgical', 'Micro',        60,  45),
  STRUCT('Oncology Medical/Surgical', 'Rural',        75,  60),
  STRUCT('Oncology Medical/Surgical', 'CEAC',        110, 100),
  -- Oncology Radiation
  STRUCT('Oncology Radiation', 'Large Metro',  30,  15),
  STRUCT('Oncology Radiation', 'Metro',        60,  40),
  STRUCT('Oncology Radiation', 'Micro',       100,  75),
  STRUCT('Oncology Radiation', 'Rural',       110,  90),
  STRUCT('Oncology Radiation', 'CEAC',        145, 130),
  -- Ophthalmology
  STRUCT('Ophthalmology', 'Large Metro', 20, 10),
  STRUCT('Ophthalmology', 'Metro',       30, 20),
  STRUCT('Ophthalmology', 'Micro',       50, 35),
  STRUCT('Ophthalmology', 'Rural',       75, 60),
  STRUCT('Ophthalmology', 'CEAC',        95, 85),
  -- Orthopedic Surgery
  STRUCT('Orthopedic Surgery', 'Large Metro', 20, 10),
  STRUCT('Orthopedic Surgery', 'Metro',       30, 20),
  STRUCT('Orthopedic Surgery', 'Micro',       50, 35),
  STRUCT('Orthopedic Surgery', 'Rural',       75, 60),
  STRUCT('Orthopedic Surgery', 'CEAC',        95, 85),
  -- Physiatry Rehabilitative Med
  STRUCT('Physiatry Rehabilitative Med', 'Large Metro',  30,  15),
  STRUCT('Physiatry Rehabilitative Med', 'Metro',        45,  30),
  STRUCT('Physiatry Rehabilitative Med', 'Micro',        80,  60),
  STRUCT('Physiatry Rehabilitative Med', 'Rural',        90,  75),
  STRUCT('Physiatry Rehabilitative Med', 'CEAC',        125, 110),
  -- Plastic Surgery
  STRUCT('Plastic Surgery', 'Large Metro',  30,  15),
  STRUCT('Plastic Surgery', 'Metro',        60,  40),
  STRUCT('Plastic Surgery', 'Micro',       100,  75),
  STRUCT('Plastic Surgery', 'Rural',       110,  90),
  STRUCT('Plastic Surgery', 'CEAC',        145, 130),
  -- Podiatry
  STRUCT('Podiatry', 'Large Metro',  20,  10),
  STRUCT('Podiatry', 'Metro',        45,  30),
  STRUCT('Podiatry', 'Micro',        60,  45),
  STRUCT('Podiatry', 'Rural',        75,  60),
  STRUCT('Podiatry', 'CEAC',        110, 100),
  -- Psychiatry
  STRUCT('Psychiatry', 'Large Metro',  20,  10),
  STRUCT('Psychiatry', 'Metro',        45,  30),
  STRUCT('Psychiatry', 'Micro',        60,  45),
  STRUCT('Psychiatry', 'Rural',        75,  60),
  STRUCT('Psychiatry', 'CEAC',        110, 100),
  -- Pulmonology
  STRUCT('Pulmonology', 'Large Metro',  20,  10),
  STRUCT('Pulmonology', 'Metro',        45,  30),
  STRUCT('Pulmonology', 'Micro',        60,  45),
  STRUCT('Pulmonology', 'Rural',        75,  60),
  STRUCT('Pulmonology', 'CEAC',        110, 100),
  -- Rheumatology
  STRUCT('Rheumatology', 'Large Metro',  30,  15),
  STRUCT('Rheumatology', 'Metro',        60,  40),
  STRUCT('Rheumatology', 'Micro',       100,  75),
  STRUCT('Rheumatology', 'Rural',       110,  90),
  STRUCT('Rheumatology', 'CEAC',        145, 130),
  -- Urology
  STRUCT('Urology', 'Large Metro',  20,  10),
  STRUCT('Urology', 'Metro',        45,  30),
  STRUCT('Urology', 'Micro',        60,  45),
  STRUCT('Urology', 'Rural',        75,  60),
  STRUCT('Urology', 'CEAC',        110, 100),
  -- Vascular Surgery
  STRUCT('Vascular Surgery', 'Large Metro',  30,  15),
  STRUCT('Vascular Surgery', 'Metro',        60,  40),
  STRUCT('Vascular Surgery', 'Micro',       100,  75),
  STRUCT('Vascular Surgery', 'Rural',       110,  90),
  STRUCT('Vascular Surgery', 'CEAC',        145, 130),
  -- Cardiothoracic Surgery
  STRUCT('Cardiothoracic Surgery', 'Large Metro',  30,  15),
  STRUCT('Cardiothoracic Surgery', 'Metro',        60,  40),
  STRUCT('Cardiothoracic Surgery', 'Micro',       100,  75),
  STRUCT('Cardiothoracic Surgery', 'Rural',       110,  90),
  STRUCT('Cardiothoracic Surgery', 'CEAC',        145, 130),
  -- Acute Inpatient Hospitals
  STRUCT('Acute Inpatient Hospitals', 'Large Metro', 20,  10),
  STRUCT('Acute Inpatient Hospitals', 'Metro',       45,  30),
  STRUCT('Acute Inpatient Hospitals', 'Micro',       80,  60),
  STRUCT('Acute Inpatient Hospitals', 'Rural',       75,  60),
  STRUCT('Acute Inpatient Hospitals', 'CEAC',       110, 100),
  -- Cardiac Surgery Program
  STRUCT('Cardiac Surgery Program', 'Large Metro',  30,  15),
  STRUCT('Cardiac Surgery Program', 'Metro',        60,  40),
  STRUCT('Cardiac Surgery Program', 'Micro',       160, 120),
  STRUCT('Cardiac Surgery Program', 'Rural',       145, 120),
  STRUCT('Cardiac Surgery Program', 'CEAC',        155, 140),
  -- Cardiac Catheterization
  STRUCT('Cardiac Catheterization', 'Large Metro',  30,  15),
  STRUCT('Cardiac Catheterization', 'Metro',        60,  40),
  STRUCT('Cardiac Catheterization', 'Micro',       160, 120),
  STRUCT('Cardiac Catheterization', 'Rural',       145, 120),
  STRUCT('Cardiac Catheterization', 'CEAC',        155, 140),
  -- Critical Care ICU
  STRUCT('Critical Care ICU', 'Large Metro',  20,  10),
  STRUCT('Critical Care ICU', 'Metro',        45,  30),
  STRUCT('Critical Care ICU', 'Micro',       160, 120),
  STRUCT('Critical Care ICU', 'Rural',       145, 120),
  STRUCT('Critical Care ICU', 'CEAC',        155, 140),
  -- Surgical Services ASC
  STRUCT('Surgical Services ASC', 'Large Metro', 20,  10),
  STRUCT('Surgical Services ASC', 'Metro',       45,  30),
  STRUCT('Surgical Services ASC', 'Micro',       80,  60),
  STRUCT('Surgical Services ASC', 'Rural',       75,  60),
  STRUCT('Surgical Services ASC', 'CEAC',       110, 100),
  -- Skilled Nursing Facility
  STRUCT('Skilled Nursing Facility', 'Large Metro', 20, 10),
  STRUCT('Skilled Nursing Facility', 'Metro',       45, 30),
  STRUCT('Skilled Nursing Facility', 'Micro',       80, 60),
  STRUCT('Skilled Nursing Facility', 'Rural',       75, 60),
  STRUCT('Skilled Nursing Facility', 'CEAC',        95, 85),
  -- Diagnostic Radiology
  STRUCT('Diagnostic Radiology', 'Large Metro', 20,  10),
  STRUCT('Diagnostic Radiology', 'Metro',       45,  30),
  STRUCT('Diagnostic Radiology', 'Micro',       80,  60),
  STRUCT('Diagnostic Radiology', 'Rural',       75,  60),
  STRUCT('Diagnostic Radiology', 'CEAC',       110, 100),
  -- Mammography
  STRUCT('Mammography', 'Large Metro', 20,  10),
  STRUCT('Mammography', 'Metro',       45,  30),
  STRUCT('Mammography', 'Micro',       80,  60),
  STRUCT('Mammography', 'Rural',       75,  60),
  STRUCT('Mammography', 'CEAC',       110, 100),
  -- Physical Therapy
  STRUCT('Physical Therapy', 'Large Metro', 20,  10),
  STRUCT('Physical Therapy', 'Metro',       45,  30),
  STRUCT('Physical Therapy', 'Micro',       80,  60),
  STRUCT('Physical Therapy', 'Rural',       75,  60),
  STRUCT('Physical Therapy', 'CEAC',       110, 100),
  -- Occupational Therapy
  STRUCT('Occupational Therapy', 'Large Metro', 20,  10),
  STRUCT('Occupational Therapy', 'Metro',       45,  30),
  STRUCT('Occupational Therapy', 'Micro',       80,  60),
  STRUCT('Occupational Therapy', 'Rural',       75,  60),
  STRUCT('Occupational Therapy', 'CEAC',       110, 100),
  -- Speech Therapy
  STRUCT('Speech Therapy', 'Large Metro', 20,  10),
  STRUCT('Speech Therapy', 'Metro',       45,  30),
  STRUCT('Speech Therapy', 'Micro',       80,  60),
  STRUCT('Speech Therapy', 'Rural',       75,  60),
  STRUCT('Speech Therapy', 'CEAC',       110, 100),
  -- Inpatient Psychiatric
  STRUCT('Inpatient Psychiatric', 'Large Metro',  30,  15),
  STRUCT('Inpatient Psychiatric', 'Metro',        70,  45),
  STRUCT('Inpatient Psychiatric', 'Micro',       100,  75),
  STRUCT('Inpatient Psychiatric', 'Rural',        90,  75),
  STRUCT('Inpatient Psychiatric', 'CEAC',        155, 140),
  -- Outpatient Infusion/Chemo
  STRUCT('Outpatient Infusion/Chemo', 'Large Metro', 20,  10),
  STRUCT('Outpatient Infusion/Chemo', 'Metro',       45,  30),
  STRUCT('Outpatient Infusion/Chemo', 'Micro',       80,  60),
  STRUCT('Outpatient Infusion/Chemo', 'Rural',       75,  60),
  STRUCT('Outpatient Infusion/Chemo', 'CEAC',       110, 100),
  -- Outpatient Behavioral Health
  STRUCT('Outpatient Behavioral Health', 'Large Metro', 20,  10),
  STRUCT('Outpatient Behavioral Health', 'Metro',       40,  25),
  STRUCT('Outpatient Behavioral Health', 'Micro',       55,  40),
  STRUCT('Outpatient Behavioral Health', 'Rural',       60,  50),
  STRUCT('Outpatient Behavioral Health', 'CEAC',       110, 100)
]);


-- ============================================================
-- STEP 3: ref_county_classification
-- WHAT:   Classifies all 67 Florida counties into CMS county
--         types: Large Metro, Metro, Micro, Rural, CEAC.
--         Also stores compliance threshold (90% or 85%) and
--         county radius for confidence interval calculation.
-- WHY:    County type drives which distance + ratio thresholds
--         apply per 42 CFR 422.116. Compliance threshold
--         determines pass/fail in fact_gap_analysis.
-- SOURCE: bigquery-public-data.geo_us_boundaries.counties
--         bigquery-public-data.census_bureau_acs.county_2020_5yr
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_county_classification`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH raw_counties AS (
  SELECT
    geo_id                                                           AS county_fips,
    county_name,
    area_land_meters / 2589988.11                                   AS area_sq_miles,
    int_point_geom                                                   AS county_centroid
  FROM `bigquery-public-data.geo_us_boundaries.counties`
  WHERE state_fips_code = '12'
),

population AS (
  SELECT
    geo_id                                                           AS county_fips,
    total_pop
  FROM `bigquery-public-data.census_bureau_acs.county_2020_5yr`
  WHERE LEFT(geo_id, 2) = '12'
),

joined AS (
  SELECT
    r.county_fips,
    r.county_name,
    r.area_sq_miles,
    p.total_pop                                                      AS population,
    ROUND(p.total_pop / NULLIF(r.area_sq_miles, 0), 2)             AS pop_density,
    ROUND(SQRT(r.area_sq_miles / ACOS(-1)), 2)                     AS county_radius_miles
  FROM raw_counties r
  LEFT JOIN population p USING (county_fips)
),

classified AS (
  SELECT
    *,
    CASE
      WHEN (population >= 1000000 AND pop_density >= 1000)
        OR (population >= 500000  AND pop_density >= 1500)
        OR (pop_density >= 5000)                                     THEN 'Large Metro'
      WHEN (population >= 1000000 AND pop_density >= 10)
        OR (population >= 500000  AND pop_density >= 10)
        OR (population >= 200000  AND pop_density >= 10)
        OR (population >= 50000   AND pop_density >= 100)
        OR (population >= 10000   AND pop_density >= 1000)          THEN 'Metro'
      WHEN (population >= 50000   AND pop_density >= 10)
        OR (population >= 10000   AND pop_density >= 50)            THEN 'Micro'
      WHEN pop_density < 10                                          THEN 'CEAC'
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
  county_radius_miles,
  county_type,
  CASE
    WHEN county_type IN ('Large Metro', 'Metro') THEN 0.90
    ELSE 0.85
  END                                                                AS compliance_threshold
FROM classified
ORDER BY county_type, county_name;


-- ============================================================
-- STEP 4: ref_zip_reference
-- WHAT:   Master geographic lookup for all Florida zip codes.
--         Contains zip centroid lat/long, area, radius, and
--         county mapping via spatial intersection.
-- WHY:    Single source of truth for all geo information used
--         in distance calculations. Both stg_beneficiaries and
--         stg_providers_multi_specialty join here for lat/long.
--         Zip radius used for confidence interval calculation.
-- SOURCE: bigquery-public-data.geo_us_boundaries.zip_codes
--         bigquery-public-data.census_bureau_acs.zip_codes_2018_5yr
--         bigquery-public-data.geo_us_boundaries.counties
-- NOTE:   ST_INTERSECTS used to catch border zips (GA/AL)
--         that serve FL members — intentional design decision.
--         zip_centroid excluded from SELECT (GEOGRAPHY type
--         incompatible with GROUP BY). Reconstructed at query
--         time via ST_GEOGPOINT(zip_long, zip_lat).
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_zip_reference`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH florida_zips AS (
  SELECT
    z.zip_code,
    z.area_land_meters / 2589988.11                                 AS area_sq_miles,
    ROUND(SQRT((z.area_land_meters / 2589988.11) / ACOS(-1)), 2)   AS zip_radius_miles,
    ST_Y(z.internal_point_geom)                                     AS zip_lat,
    ST_X(z.internal_point_geom)                                     AS zip_long,
    z.zip_code_geom
  FROM `bigquery-public-data.geo_us_boundaries.zip_codes` z
  WHERE EXISTS (
    SELECT 1
    FROM `bigquery-public-data.geo_us_boundaries.counties` c
    WHERE c.state_fips_code = '12'
      AND ST_INTERSECTS(z.zip_code_geom, c.county_geom)
  )
),

zip_population AS (
  SELECT
    geo_id                                                           AS zip_code,
    total_pop
  FROM `bigquery-public-data.census_bureau_acs.zip_codes_2018_5yr`
),

zip_to_county AS (
  SELECT
    z.zip_code,
    c.geo_id                                                         AS county_fips,
    c.county_name,
    ST_AREA(ST_INTERSECTION(z.zip_code_geom, c.county_geom))        AS intersection_area,
    ROW_NUMBER() OVER (
      PARTITION BY z.zip_code
      ORDER BY ST_AREA(ST_INTERSECTION(z.zip_code_geom, c.county_geom)) DESC
    )                                                                AS rnk
  FROM florida_zips z
  JOIN `bigquery-public-data.geo_us_boundaries.counties` c
    ON c.state_fips_code = '12'
    AND ST_INTERSECTS(z.zip_code_geom, c.county_geom)
)

SELECT
  z.zip_code,
  z.area_sq_miles,
  z.zip_radius_miles,
  z.zip_lat,
  z.zip_long,
  p.total_pop                                                        AS zip_population,
  m.county_fips,
  m.county_name,
  cc.county_type,
  cc.compliance_threshold,
  cc.county_radius_miles
FROM florida_zips z
LEFT JOIN zip_population p
  ON z.zip_code = p.zip_code
LEFT JOIN zip_to_county m
  ON z.zip_code = m.zip_code
  AND m.rnk = 1
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_county_classification` cc
  ON m.county_fips = cc.county_fips
ORDER BY z.zip_code;


-- ============================================================
-- STEP 5: ref_county_name_crosswalk
-- WHAT:   Maps Aetna county names to Census county names and
--         FIPS codes. Handles 3 known name mismatches.
--         Flags 26 Florida counties with no Aetna coverage.
-- WHY:    Aetna uses different county name formats than Census.
--         Without this crosswalk, county joins fail silently.
-- KNOWN MISMATCHES:
--   Desoto      → DeSoto   (12027)
--   Saint Johns → St. Johns (12109)
--   Saint Lucie → St. Lucie (12111)
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_county_name_crosswalk`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT * FROM UNNEST([
  -- exact matches (38 counties)
  STRUCT('Alachua'      AS aetna_county_nm, 'Alachua'      AS census_county_nm, '12001' AS county_fips),
  STRUCT('Baker',                           'Baker',                             '12003'),
  STRUCT('Brevard',                         'Brevard',                           '12009'),
  STRUCT('Broward',                         'Broward',                           '12011'),
  STRUCT('Charlotte',                       'Charlotte',                         '12015'),
  STRUCT('Citrus',                          'Citrus',                            '12017'),
  STRUCT('Clay',                            'Clay',                              '12019'),
  STRUCT('Collier',                         'Collier',                           '12021'),
  STRUCT('Columbia',                        'Columbia',                          '12023'),
  STRUCT('Duval',                           'Duval',                             '12031'),
  STRUCT('Escambia',                        'Escambia',                          '12033'),
  STRUCT('Flagler',                         'Flagler',                           '12035'),
  STRUCT('Hernando',                        'Hernando',                          '12053'),
  STRUCT('Highlands',                       'Highlands',                         '12055'),
  STRUCT('Hillsborough',                    'Hillsborough',                      '12057'),
  STRUCT('Indian River',                    'Indian River',                      '12061'),
  STRUCT('Lake',                            'Lake',                              '12069'),
  STRUCT('Lee',                             'Lee',                               '12071'),
  STRUCT('Levy',                            'Levy',                              '12075'),
  STRUCT('Manatee',                         'Manatee',                           '12081'),
  STRUCT('Marion',                          'Marion',                            '12083'),
  STRUCT('Martin',                          'Martin',                            '12085'),
  STRUCT('Miami-Dade',                      'Miami-Dade',                        '12086'),
  STRUCT('Nassau',                          'Nassau',                            '12089'),
  STRUCT('Okaloosa',                        'Okaloosa',                          '12091'),
  STRUCT('Orange',                          'Orange',                            '12095'),
  STRUCT('Osceola',                         'Osceola',                           '12097'),
  STRUCT('Palm Beach',                      'Palm Beach',                        '12099'),
  STRUCT('Pasco',                           'Pasco',                             '12101'),
  STRUCT('Pinellas',                        'Pinellas',                          '12103'),
  STRUCT('Polk',                            'Polk',                              '12105'),
  STRUCT('Putnam',                          'Putnam',                            '12107'),
  STRUCT('Santa Rosa',                      'Santa Rosa',                        '12113'),
  STRUCT('Sarasota',                        'Sarasota',                          '12115'),
  STRUCT('Seminole',                        'Seminole',                          '12117'),
  STRUCT('Sumter',                          'Sumter',                            '12119'),
  STRUCT('Volusia',                         'Volusia',                           '12127'),
  STRUCT('Walton',                          'Walton',                            '12131'),
  -- name mismatch fixes (3 counties)
  STRUCT('Desoto',                          'DeSoto',                            '12027'),
  STRUCT('Saint Johns',                     'St. Johns',                         '12109'),
  STRUCT('Saint Lucie',                     'St. Lucie',                         '12111'),
  -- no Aetna coverage (26 counties)
  STRUCT(NULL,                              'Bay',                               '12005'),
  STRUCT(NULL,                              'Bradford',                          '12007'),
  STRUCT(NULL,                              'Calhoun',                           '12013'),
  STRUCT(NULL,                              'Dixie',                             '12029'),
  STRUCT(NULL,                              'Franklin',                          '12037'),
  STRUCT(NULL,                              'Gadsden',                           '12039'),
  STRUCT(NULL,                              'Gilchrist',                         '12041'),
  STRUCT(NULL,                              'Glades',                            '12043'),
  STRUCT(NULL,                              'Gulf',                              '12045'),
  STRUCT(NULL,                              'Hamilton',                          '12047'),
  STRUCT(NULL,                              'Hardee',                            '12049'),
  STRUCT(NULL,                              'Hendry',                            '12051'),
  STRUCT(NULL,                              'Holmes',                            '12059'),
  STRUCT(NULL,                              'Jackson',                           '12063'),
  STRUCT(NULL,                              'Jefferson',                         '12065'),
  STRUCT(NULL,                              'Lafayette',                         '12067'),
  STRUCT(NULL,                              'Leon',                              '12073'),
  STRUCT(NULL,                              'Liberty',                           '12077'),
  STRUCT(NULL,                              'Madison',                           '12079'),
  STRUCT(NULL,                              'Monroe',                            '12087'),
  STRUCT(NULL,                              'Okeechobee',                        '12093'),
  STRUCT(NULL,                              'Suwannee',                          '12121'),
  STRUCT(NULL,                              'Taylor',                            '12123'),
  STRUCT(NULL,                              'Union',                             '12125'),
  STRUCT(NULL,                              'Wakulla',                           '12129'),
  STRUCT(NULL,                              'Washington',                        '12133')
]);


-- ============================================================
-- STEP 6: ref_hsd_required_counts
-- WHAT:   Exact required provider and facility counts per
--         Florida county per specialty. Sourced directly from
--         CMS 2026 HSD Reference File (published 12-17-2025).
-- WHY:    Replaces approximated ratio-based calculation.
--         CMS uses 95th percentile base population ratio which
--         we cannot derive from available data. This table
--         gives us the exact numbers CMS uses for compliance.
-- SOURCE: CMS 2026 HSD Reference File
--         https://www.cms.gov/medicare/health-drug-plans/
--         medicare-advantage-application
-- NOTE:   Loaded via load_hsd_to_bq.ipynb notebook.
--         Run notebook before running this step if table
--         does not exist.
-- ============================================================

-- ref_hsd_required_counts was loaded via notebook.
-- Verify it exists before proceeding:
SELECT
  COUNT(*)                                                           AS total_rows,
  COUNT(DISTINCT county_name)                                        AS counties,
  COUNT(DISTINCT cms_specialty)                                      AS specialties
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_hsd_required_counts`;
-- Expected: 2881 rows, 67 counties, 43 specialties


-- ============================================================
-- STEP 7: stg_beneficiaries
-- WHAT:   Demand side table. One row per Florida zip code with
--         total population (ACS 2018) and county context
--         including CMS Medicare eligible counts.
-- WHY:    Population is the demand measure. Used in:
--         - fact_zip_access: bene zip is the center point
--           for distance calculation
--         - fact_gap_analysis: population denominator for
--           pct_covered calculation
-- NOTE:   lat/long NOT stored here. Joined from ref_zip_reference
--         at distance calculation time only.
--         county_eligibles from CMS penetration file used as
--         denominator for HSD required count validation only.
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_beneficiaries`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH latest_penetration AS (
  SELECT MAX(ingest_time) AS max_ingest
  FROM `anbc-hcb-prod.provider_ds_netconf_data_hcb_prod.cms_medicare_penetration`
),

county_penetration AS (
  SELECT
    CONCAT(
      LPAD(CAST(fipsst   AS STRING), 2, '0'),
      LPAD(CAST(fipscnty AS STRING), 3, '0')
    )                                                                AS county_fips,
    SAFE_CAST(REPLACE(CAST(eligibles AS STRING), ',', '') AS FLOAT64) AS county_eligibles,
    enrolled                                                         AS county_ma_enrolled,
    SAFE_CAST(REPLACE(penetration, '%', '') AS FLOAT64) / 100       AS county_penetration_rate,
    ingest_time                                                      AS data_as_of
  FROM `anbc-hcb-prod.provider_ds_netconf_data_hcb_prod.cms_medicare_penetration`
  CROSS JOIN latest_penetration
  WHERE fipsst = '12'
    AND ingest_time = latest_penetration.max_ingest
)

SELECT
  z.zip_code,
  z.zip_population                                                   AS total_population,
  z.zip_radius_miles,
  z.county_fips,
  z.county_name,
  z.county_type,
  z.compliance_threshold,
  p.county_eligibles,
  p.county_ma_enrolled,
  p.county_penetration_rate,
  p.data_as_of
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_zip_reference` z
LEFT JOIN county_penetration p
  ON z.county_fips = p.county_fips
ORDER BY z.zip_code;


-- ============================================================
-- STEP 8: stg_providers_multi_specialty
-- WHAT:   Supply side table. One row per provider per specialty
--         per plan type. Uses network ID explosion to get ALL
--         specialties per provider, not just primary specialty.
-- WHY:    A multi-specialty provider (e.g. hospital) counts
--         toward multiple CMS specialty requirements. Using
--         only primary specialty undercounts supply.
-- SOURCE: A870800_medicare_supply_demand_mbr_with_zip
--         edp-prod-hcbstorage.edp_hcb_core_cnsv.RPDB_RPNPRAC
--         edp-prod-hcbstorage.edp_hcb_core_srcv.EPDB_PRVDR
--         edp-prod-hcbstorage.edp_hcb_core_srcv.RPDB_RINPR
--         edp-prod-hcbstorage.edp_hcb_core_cnsv.PRVDR_TY_X_SPCLTY
--         edp-prod-hcbstorage.edp_hcb_core_srcv.GLOBAL_LOOKUP
-- NOTE:   SAFE_CAST used throughout - data quality is poor.
--         NULL guards on join keys prevent phantom matches
--         from CAST(NULL AS STRING) = 'null' in BigQuery.
--         Providers with no cms_specialty or zip_lat match
--         are excluded (out of state / unmapped specialties).
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers_multi_specialty`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH mbr_exploded AS (
  -- explode network_id on '-' delimiter to get individual network IDs
  -- SAFE_CAST protects against empty or non-numeric segments
  SELECT DISTINCT
    CAST(prvdr_id_no AS INT64)                                       AS pin,
    SAFE_CAST(TRIM(ntwk_id_exploded) AS INT64)                      AS ntwk_id_no,
    CAST(prvdr_id_no AS STRING)                                     AS provider_id,
    tin_owner_nm                                                     AS provider_name,
    tax_id_no,
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
  CROSS JOIN UNNEST(SPLIT(network_id, '-'))                         AS ntwk_id_exploded
  WHERE state = 'FL'
    AND network_id IS NOT NULL
    AND TRIM(ntwk_id_exploded) != ''
),

rpnprac AS (
  -- get all specialties per provider via network join
  -- left join retains providers with no RPDB match
  -- specialty_cd derived from major + category + subclass codes
  SELECT DISTINCT
    a.pin,
    a.provider_id,
    a.provider_name,
    a.tax_id_no,
    a.county_nm,
    a.zip_cd,
    a.plan_type,
    a.market,
    a.submarket,
    b.rpnp_prvdr_type_cd,
    CASE
      WHEN TRIM(c.prvdr_info_ty_cd) = 'N'
        THEN COALESCE(
          SAFE_CAST(SAFE_CAST(CONCAT(
            COALESCE(CAST(b.rpnp_spcl_majcl_cd AS STRING), ''),
            COALESCE(CAST(b.rpnp_spcl_ctgry_cd AS STRING), ''),
            COALESCE(CAST(b.rpnp_spcl_sbcls_cd AS STRING), '')
          ) AS INT64) AS STRING),
          CAST(c.prvdr_type_cd AS STRING))
      ELSE COALESCE(
          SAFE_CAST(SAFE_CAST(CONCAT(
            COALESCE(CAST(b.rpnp_spcl_majcl_cd AS STRING), ''),
            COALESCE(CAST(b.rpnp_spcl_ctgry_cd AS STRING), ''),
            COALESCE(CAST(b.rpnp_spcl_sbcls_cd AS STRING), '')
          ) AS INT64) AS STRING),
          CAST(d.rip_prvdr_type_cd AS STRING))
    END                                                              AS specialty_cd
  FROM mbr_exploded a
  LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.RPDB_RPNPRAC` b
    ON CAST(a.ntwk_id_no AS INT64) = CAST(b.ntwk_id_no AS INT64)
    AND CAST(a.pin AS INT64)       = CAST(b.prvdr_id_no AS INT64) * 100 + 9
  LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.EPDB_PRVDR` c
    ON CAST(a.pin AS INT64) = CAST(c.prvdr_id_no AS INT64) * 100 + 9
  LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.RPDB_RINPR` d
    ON CAST(a.pin AS INT64) = CAST(d.prvdr_id_no AS INT64) * 100 + 9
),

specialty_mapped AS (
  -- map specialty_cd to specialty_ctg_cd via PRVDR_TY_X_SPCLTY
  -- get descriptions from GLOBAL_LOOKUP
  -- null guards prevent phantom matches from NULL casts
  SELECT
    a.pin,
    a.provider_id,
    a.provider_name,
    a.tax_id_no,
    a.county_nm,
    a.zip_cd,
    a.plan_type,
    a.market,
    a.submarket,
    a.rpnp_prvdr_type_cd,
    a.specialty_cd,
    TRIM(b.specialty_ctg_cd)                                        AS specialty_ctg_cd,
    c.short_dscrptn                                                  AS specialty_cd_desc,
    d.short_dscrptn                                                  AS specialty_ctg_cd_desc,
    e.short_dscrptn                                                  AS prvdr_type_desc
  FROM rpnprac a
  LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.PRVDR_TY_X_SPCLTY` b
    ON a.rpnp_prvdr_type_cd IS NOT NULL
    AND a.specialty_cd IS NOT NULL
    AND TRIM(CAST(a.rpnp_prvdr_type_cd AS STRING)) = TRIM(CAST(b.provider_type_cd AS STRING))
    AND TRIM(CAST(a.specialty_cd AS STRING))        = TRIM(CAST(b.specialty_cd AS STRING))
  LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.GLOBAL_LOOKUP` c
    ON TRIM(CAST(a.specialty_cd AS STRING))         = TRIM(CAST(c.global_lookup_cd AS STRING))
    AND c.lookup_column_nm                          = 'SPECIALTY_CD'
  LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.GLOBAL_LOOKUP` d
    ON b.specialty_ctg_cd IS NOT NULL
    AND TRIM(CAST(b.specialty_ctg_cd AS STRING))    = TRIM(CAST(d.global_lookup_cd AS STRING))
    AND d.lookup_column_nm                          = 'SPECIALTY_CTG_CD'
  LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.GLOBAL_LOOKUP` e
    ON TRIM(CAST(a.rpnp_prvdr_type_cd AS STRING))   = TRIM(CAST(e.global_lookup_cd AS STRING))
    AND e.lookup_column_nm                          = 'PROVIDER_TYPE_CD'
)

SELECT
  s.provider_id,
  s.provider_name,
  s.tax_id_no,
  s.rpnp_prvdr_type_cd,
  s.prvdr_type_desc,
  s.specialty_cd,
  s.specialty_cd_desc,
  s.specialty_ctg_cd                                                AS aetna_specialty_cd,
  s.specialty_ctg_cd_desc,
  sc.cms_specialty,
  s.county_nm                                                        AS aetna_county_nm,
  c.census_county_nm,
  c.county_fips,
  s.zip_cd,
  z.zip_lat,
  z.zip_long,
  z.zip_radius_miles,
  z.county_type,
  s.plan_type,
  s.market,
  s.submarket
FROM specialty_mapped s
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_specialty_crosswalk` sc
  ON TRIM(CAST(s.specialty_ctg_cd AS STRING)) = TRIM(CAST(sc.aetna_cd AS STRING))
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_county_name_crosswalk` c
  ON TRIM(CAST(s.county_nm AS STRING)) = TRIM(CAST(c.aetna_county_nm AS STRING))
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_zip_reference` z
  ON TRIM(CAST(s.zip_cd AS STRING)) = TRIM(CAST(z.zip_code AS STRING))
WHERE sc.cms_specialty IS NOT NULL
  AND z.zip_lat IS NOT NULL
GROUP BY ALL
ORDER BY s.provider_id, sc.cms_specialty, s.plan_type;


-- ============================================================
-- STEP 9: fact_zip_access
-- WHAT:   For each beneficiary zip × CMS specialty × plan type,
--         counts how many contracted providers exist within
--         the CMS maximum distance threshold.
--         Tags each zip as has_access = TRUE/FALSE.
-- WHY:    This is the core distance compliance computation.
--         CMS requires that X% of beneficiaries in each county
--         have at least 1 provider within the distance threshold.
--         This table answers "does this zip have access?"
--         fact_gap_analysis then rolls this up to county level.
-- NOTE:   SPARSE TABLE - only rows where at least 1 provider
--         exists within threshold are stored.
--         Zips with zero access are handled in fact_gap_analysis
--         via LEFT JOIN from all_combinations CTE.
--         Threshold lookup uses BENEFICIARY county type per
--         42 CFR 422.116 - not provider county type.
--         ST_DISTANCE returns meters, divided by 1609.34 for miles.
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_fact_zip_access`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH zip_provider_pairs AS (
  -- for each bene zip × provider × specialty × plan type
  -- apply distance filter: only pairs within CMS threshold survive
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
    t.max_distance_miles,
    ROUND(
      ST_DISTANCE(
        ST_GEOGPOINT(bene_zip.zip_long, bene_zip.zip_lat),
        ST_GEOGPOINT(p.zip_long,        p.zip_lat)
      ) / 1609.34
    , 2)                                                             AS distance_miles
  FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_beneficiaries` b
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_zip_reference` bene_zip
    ON b.zip_code = bene_zip.zip_code
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers_multi_specialty` p
    ON TRUE
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_time_distance` t
    ON t.cms_specialty = p.cms_specialty
    AND t.county_type  = b.county_type
  WHERE ST_DISTANCE(
          ST_GEOGPOINT(bene_zip.zip_long, bene_zip.zip_lat),
          ST_GEOGPOINT(p.zip_long,        p.zip_lat)
        ) / 1609.34 <= t.max_distance_miles
)

SELECT
  bene_zip,
  bene_county_fips,
  bene_county_name,
  bene_county_type,
  compliance_threshold,
  bene_zip_population,
  bene_zip_radius,
  cms_specialty,
  plan_type,
  max_distance_miles,
  COUNT(DISTINCT provider_id)                                        AS provider_count_within_threshold,
  TRUE                                                               AS has_access
FROM zip_provider_pairs
GROUP BY
  bene_zip,
  bene_county_fips,
  bene_county_name,
  bene_county_type,
  compliance_threshold,
  bene_zip_population,
  bene_zip_radius,
  cms_specialty,
  plan_type,
  max_distance_miles;


-- ============================================================
-- STEP 10: fact_gap_analysis
-- WHAT:   Final county-level compliance output.
--         For each county × specialty × plan type:
--         - % beneficiaries with at least 1 provider in range
--         - actual provider count vs CMS required count
--         - compliance status (COMPLIANT / NON-COMPLIANT)
-- WHY:    CMS evaluates compliance at county level.
--         Two tests must both pass per 42 CFR 422.116:
--         Test 1: pct_covered >= 90% (Large Metro/Metro)
--                              >= 85% (Micro/Rural/CEAC)
--         Test 2: actual_provider_count >= required_count
--                 (from CMS 2026 HSD Reference File)
-- NOTE:   all_combinations CTE creates complete grid of
--         county × specialty × plan_type to ensure zips with
--         zero providers are included as NO_ACCESS = FALSE.
--         required_count from ref_hsd_required_counts — exact
--         CMS numbers, no approximation.
--         county_eligibles used for context only.
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_fact_gap_analysis`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH all_combinations AS (
  -- build complete grid: all bene zips × all specialties × all plan types
  -- ensures zips with no providers are not silently dropped
  SELECT
    b.zip_code,
    b.county_fips,
    b.county_name,
    b.county_type,
    b.compliance_threshold,
    b.total_population,
    sc.cms_specialty,
    pt.plan_type
  FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_beneficiaries` b
  CROSS JOIN (
    SELECT DISTINCT cms_specialty
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_specialty_crosswalk`
  ) sc
  CROSS JOIN (
    SELECT DISTINCT plan_type
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers_multi_specialty`
  ) pt
),

zip_access_complete AS (
  -- left join fact_zip_access to all_combinations
  -- fills in zeros for zips with no providers within threshold
  SELECT
    a.zip_code,
    a.county_fips,
    a.county_name,
    a.county_type,
    a.compliance_threshold,
    a.total_population,
    a.cms_specialty,
    a.plan_type,
    COALESCE(z.provider_count_within_threshold, 0)                  AS provider_count,
    COALESCE(z.has_access, FALSE)                                    AS has_access
  FROM all_combinations a
  LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_fact_zip_access` z
    ON a.zip_code       = z.bene_zip
    AND a.cms_specialty = z.cms_specialty
    AND a.plan_type     = z.plan_type
),

county_rollup AS (
  -- roll up zip level to county level
  -- pct_covered = population with access / total county population
  SELECT
    county_fips,
    county_name,
    county_type,
    compliance_threshold,
    cms_specialty,
    plan_type,
    SUM(total_population)                                            AS total_county_population,
    SUM(CASE WHEN has_access THEN total_population ELSE 0 END)      AS population_with_access,
    ROUND(
      SUM(CASE WHEN has_access THEN total_population ELSE 0 END)
      / NULLIF(SUM(total_population), 0)
    , 4)                                                             AS pct_covered,
    SUM(provider_count)                                              AS actual_provider_count
  FROM zip_access_complete
  GROUP BY
    county_fips,
    county_name,
    county_type,
    compliance_threshold,
    cms_specialty,
    plan_type,
),

hospital_beds AS (
  -- --------------------------------------------------------
  -- SUM CONTRACTED BEDS PER COUNTY FOR ACUTE INPATIENT ONLY
  -- SOURCE: hosp_list_cmi (Pin = provider_id, Beds = bed count)
  -- CMS requires 12.2 beds per 1,000 beneficiaries
  -- NULL beds excluded - unknown bed count not credited
  -- --------------------------------------------------------
  SELECT
    p.county_fips,
    p.plan_type,
    SUM(CAST(h.Beds AS INT64))                                       AS total_contracted_beds
  FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers_multi_specialty` p
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.hosp_list_cmi` h
    ON CAST(p.provider_id AS STRING) = CAST(h.Pin AS STRING)
  WHERE p.cms_specialty = 'Acute Inpatient Hospitals'
    AND h.Beds IS NOT NULL
  GROUP BY p.county_fips, p.plan_type
)

SELECT
  r.county_fips,
  r.county_name,
  r.county_type,
  r.cms_specialty,
  r.plan_type,
  hsd.total_beneficiaries                                            AS county_total_beneficiaries,
  hsd.beneficiaries_required_to_cover,
  hsd.ratio_95th_percentile,
  r.total_county_population,
  r.population_with_access,
  r.pct_covered,
  r.compliance_threshold,
  hsd.required_count                                                 AS required_provider_count,
  -- beds column: populated only for Acute Inpatient Hospitals, NULL for all others
  CASE
    WHEN r.cms_specialty = 'Acute Inpatient Hospitals'
      THEN COALESCE(b.total_contracted_beds, 0)
    ELSE NULL
  END                                                                AS total_contracted_beds,
  -- actual count: beds for hospitals, provider count for everything else
  CASE
    WHEN r.cms_specialty = 'Acute Inpatient Hospitals'
      THEN COALESCE(b.total_contracted_beds, 0)
    ELSE r.actual_provider_count
  END                                                                AS actual_count,
  -- gap: beds vs required for hospitals, providers vs required for others
  CASE
    WHEN r.cms_specialty = 'Acute Inpatient Hospitals'
      THEN hsd.required_count - COALESCE(b.total_contracted_beds, 0)
    ELSE hsd.required_count - r.actual_provider_count
  END                                                                AS provider_gap,
  -- test 1: % beneficiaries with access >= compliance threshold
  CASE
    WHEN r.pct_covered >= r.compliance_threshold THEN TRUE
    ELSE FALSE
  END                                                                AS access_compliant,
  -- test 2: beds >= required for hospitals, providers >= required for others
  CASE
    WHEN r.cms_specialty = 'Acute Inpatient Hospitals'
      THEN COALESCE(b.total_contracted_beds, 0) >= hsd.required_count
    ELSE r.actual_provider_count >= hsd.required_count
  END                                                                AS count_compliant,
  -- overall: both tests must pass per 42 CFR 422.116
  CASE
    WHEN r.pct_covered >= r.compliance_threshold
    AND (
      CASE
        WHEN r.cms_specialty = 'Acute Inpatient Hospitals'
          THEN COALESCE(b.total_contracted_beds, 0) >= hsd.required_count
        ELSE r.actual_provider_count >= hsd.required_count
      END
    )                                                                THEN 'COMPLIANT'
    ELSE 'NON-COMPLIANT'
  END                                                                AS compliance_status

FROM county_rollup r
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_hsd_required_counts` hsd
  ON hsd.county_name    = r.county_name
  AND hsd.cms_specialty = r.cms_specialty
LEFT JOIN hospital_beds b
  ON r.county_fips  = b.county_fips
  AND r.plan_type   = b.plan_type
ORDER BY
  r.county_name,
  r.cms_specialty,
  r.plan_type;
