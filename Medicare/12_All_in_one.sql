-- ============================================================
-- MEDICARE NETWORK ADEQUACY & CAPACITY MODELING
-- ALL REFERENCE + STAGING TABLES
-- PROJECT:  anbc-hcb-dev
-- DATASET:  provider_ds_netconf_data_hcb_dev
-- PREFIX:   A870800_medicare_supply_demand_
-- AUTHOR:   deepan_thulasi_aetna_com
-- DATE:     2026-04-21
-- SCOPE:    FLORIDA COUNTIES ONLY
-- ============================================================
-- https://www.cms.gov/files/document/2026-hsd-reference-file-updated-12-17-2025.xlsx

-- ============================================================
-- TABLE 1: ref_specialty_crosswalk
-- PURPOSE: MAP CMS 422.116 SPECIALTIES TO AETNA SPECIALTY CODES
-- FLAGS:   match_type = exact/proxy
--          inflated   = TRUE if aetna_cd maps to multiple cms specialties
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_specialty_crosswalk`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT * FROM UNNEST([
  STRUCT('Primary Care'                  AS cms_specialty, 'FP'   AS aetna_cd, 'exact'  AS match_type, FALSE AS inflated),
  STRUCT('Primary Care',                                   'I',               'exact',               FALSE),
  STRUCT('Allergy and Immunology',                         'A',               'exact',               FALSE),
  STRUCT('Cardiology',                                     'C',               'exact',               FALSE),
  STRUCT('Chiropractor',                                   'VVCH',            'exact',               FALSE),
  STRUCT('Clinical Psychology',                            'VVMH',            'proxy',               TRUE),
  STRUCT('Clinical Social Work',                           'VVMH',            'proxy',               TRUE),
  STRUCT('Dermatology',                                    'D',               'exact',               FALSE),
  STRUCT('Endocrinology',                                  'E',               'exact',               FALSE),
  STRUCT('ENT/Otolaryngology',                             'EN',              'exact',               FALSE),
  STRUCT('Gastroenterology',                               'G',               'exact',               FALSE),
  STRUCT('General Surgery',                                'S',               'exact',               FALSE),
  STRUCT('Gynecology OB/GYN',                              'OG',              'exact',               FALSE),
  STRUCT('Infectious Diseases',                            'II',              'exact',               FALSE),
  STRUCT('Nephrology',                                     'N',               'exact',               FALSE),
  STRUCT('Neurology',                                      'NE',              'exact',               FALSE),
  STRUCT('Neurosurgery',                                   'NS',              'exact',               FALSE),
  STRUCT('Oncology Medical/Surgical',                      'H',               'exact',               FALSE),
  STRUCT('Oncology Radiation',                             'RO',              'exact',               FALSE),
  STRUCT('Ophthalmology',                                  'O',               'exact',               FALSE),
  STRUCT('Orthopedic Surgery',                             'OR',              'exact',               FALSE),
  STRUCT('Physiatry Rehabilitative Med',                   'VVRH',            'proxy',               TRUE),
  STRUCT('Plastic Surgery',                                'PS',              'exact',               FALSE),
  STRUCT('Podiatry',                                       'VVPD',            'exact',               FALSE),
  STRUCT('Psychiatry',                                     'PY',              'exact',               FALSE),
  STRUCT('Pulmonology',                                    'PD',              'exact',               FALSE),
  STRUCT('Rheumatology',                                   'RH',              'exact',               FALSE),
  STRUCT('Urology',                                        'U',               'exact',               FALSE),
  STRUCT('Vascular Surgery',                               'VS',              'exact',               FALSE),
  STRUCT('Cardiothoracic Surgery',                         'CS',              'exact',               FALSE),
  STRUCT('Acute Inpatient Hospitals',                      'WHOS',            'exact',               FALSE),
  STRUCT('Cardiac Surgery Program',                        'CS',              'proxy',               TRUE),
  STRUCT('Cardiac Catheterization',                        'C',               'proxy',               TRUE),
  STRUCT('Critical Care ICU',                              'VICU',            'exact',               FALSE),
  STRUCT('Surgical Services ASC',                          'WASF',            'exact',               FALSE),
  STRUCT('Skilled Nursing Facility',                       'WLTC',            'exact',               FALSE),
  STRUCT('Diagnostic Radiology',                           'WRAD',            'exact',               FALSE),
  STRUCT('Mammography',                                    'VRAD',            'proxy',               TRUE),
  STRUCT('Physical Therapy',                               'VVRH',            'proxy',               TRUE),
  STRUCT('Occupational Therapy',                           'VVRH',            'proxy',               TRUE),
  STRUCT('Speech Therapy',                                 'VVRH',            'proxy',               TRUE),
  STRUCT('Inpatient Psychiatric',                          'WBHF',            'proxy',               TRUE),
  STRUCT('Outpatient Infusion/Chemo',                      'WHOS',            'proxy',               TRUE),
  STRUCT('Outpatient Behavioral Health',                   'WBHF',            'proxy',               TRUE)
]);


-- ============================================================
-- TABLE 2: ref_time_distance
-- PURPOSE: MAX TIME + DISTANCE THRESHOLDS PER 42 CFR 422.116
-- SOURCE:  42 CFR 422.116 TABLE 1
-- GRAIN:   cms_specialty x county_type
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
-- TABLE 3: ref_min_ratio
-- PURPOSE: MINIMUM PROVIDER RATIO PER 1,000 BENEFICIARIES
-- SOURCE:  42 CFR 422.116 TABLE 2
-- GRAIN:   cms_specialty x county_type
-- NOTE:    ACUTE HOSPITAL = beds per 1,000 not providers
-- NOTE:    ALL OTHER FACILITY TYPES = minimum 1 per county
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_min_ratio`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT * FROM UNNEST([
  -- Primary Care
  STRUCT('Primary Care' AS cms_specialty, 'Large Metro' AS county_type, 1.67 AS min_ratio_per_1000),
  STRUCT('Primary Care', 'Metro',  1.67),
  STRUCT('Primary Care', 'Micro',  1.42),
  STRUCT('Primary Care', 'Rural',  1.42),
  STRUCT('Primary Care', 'CEAC',   1.42),
  -- Allergy and Immunology
  STRUCT('Allergy and Immunology', 'Large Metro', 0.05),
  STRUCT('Allergy and Immunology', 'Metro',       0.05),
  STRUCT('Allergy and Immunology', 'Micro',       0.04),
  STRUCT('Allergy and Immunology', 'Rural',       0.04),
  STRUCT('Allergy and Immunology', 'CEAC',        0.04),
  -- Cardiology
  STRUCT('Cardiology', 'Large Metro', 0.27),
  STRUCT('Cardiology', 'Metro',       0.27),
  STRUCT('Cardiology', 'Micro',       0.23),
  STRUCT('Cardiology', 'Rural',       0.23),
  STRUCT('Cardiology', 'CEAC',        0.23),
  -- Chiropractor
  STRUCT('Chiropractor', 'Large Metro', 0.10),
  STRUCT('Chiropractor', 'Metro',       0.10),
  STRUCT('Chiropractor', 'Micro',       0.09),
  STRUCT('Chiropractor', 'Rural',       0.09),
  STRUCT('Chiropractor', 'CEAC',        0.09),
  -- Clinical Psychology
  STRUCT('Clinical Psychology', 'Large Metro', 0.15),
  STRUCT('Clinical Psychology', 'Metro',       0.15),
  STRUCT('Clinical Psychology', 'Micro',       0.13),
  STRUCT('Clinical Psychology', 'Rural',       0.13),
  STRUCT('Clinical Psychology', 'CEAC',        0.13),
  -- Clinical Social Work
  STRUCT('Clinical Social Work', 'Large Metro', 0.25),
  STRUCT('Clinical Social Work', 'Metro',       0.25),
  STRUCT('Clinical Social Work', 'Micro',       0.22),
  STRUCT('Clinical Social Work', 'Rural',       0.22),
  STRUCT('Clinical Social Work', 'CEAC',        0.22),
  -- Dermatology
  STRUCT('Dermatology', 'Large Metro', 0.16),
  STRUCT('Dermatology', 'Metro',       0.16),
  STRUCT('Dermatology', 'Micro',       0.14),
  STRUCT('Dermatology', 'Rural',       0.14),
  STRUCT('Dermatology', 'CEAC',        0.14),
  -- Endocrinology
  STRUCT('Endocrinology', 'Large Metro', 0.04),
  STRUCT('Endocrinology', 'Metro',       0.04),
  STRUCT('Endocrinology', 'Micro',       0.03),
  STRUCT('Endocrinology', 'Rural',       0.03),
  STRUCT('Endocrinology', 'CEAC',        0.03),
  -- ENT/Otolaryngology
  STRUCT('ENT/Otolaryngology', 'Large Metro', 0.06),
  STRUCT('ENT/Otolaryngology', 'Metro',       0.06),
  STRUCT('ENT/Otolaryngology', 'Micro',       0.05),
  STRUCT('ENT/Otolaryngology', 'Rural',       0.05),
  STRUCT('ENT/Otolaryngology', 'CEAC',        0.05),
  -- Gastroenterology
  STRUCT('Gastroenterology', 'Large Metro', 0.12),
  STRUCT('Gastroenterology', 'Metro',       0.12),
  STRUCT('Gastroenterology', 'Micro',       0.10),
  STRUCT('Gastroenterology', 'Rural',       0.10),
  STRUCT('Gastroenterology', 'CEAC',        0.10),
  -- General Surgery
  STRUCT('General Surgery', 'Large Metro', 0.28),
  STRUCT('General Surgery', 'Metro',       0.28),
  STRUCT('General Surgery', 'Micro',       0.24),
  STRUCT('General Surgery', 'Rural',       0.24),
  STRUCT('General Surgery', 'CEAC',        0.24),
  -- Gynecology OB/GYN
  STRUCT('Gynecology OB/GYN', 'Large Metro', 0.04),
  STRUCT('Gynecology OB/GYN', 'Metro',       0.04),
  STRUCT('Gynecology OB/GYN', 'Micro',       0.03),
  STRUCT('Gynecology OB/GYN', 'Rural',       0.03),
  STRUCT('Gynecology OB/GYN', 'CEAC',        0.03),
  -- Infectious Diseases
  STRUCT('Infectious Diseases', 'Large Metro', 0.03),
  STRUCT('Infectious Diseases', 'Metro',       0.03),
  STRUCT('Infectious Diseases', 'Micro',       0.03),
  STRUCT('Infectious Diseases', 'Rural',       0.03),
  STRUCT('Infectious Diseases', 'CEAC',        0.03),
  -- Nephrology
  STRUCT('Nephrology', 'Large Metro', 0.09),
  STRUCT('Nephrology', 'Metro',       0.09),
  STRUCT('Nephrology', 'Micro',       0.08),
  STRUCT('Nephrology', 'Rural',       0.08),
  STRUCT('Nephrology', 'CEAC',        0.08),
  -- Neurology
  STRUCT('Neurology', 'Large Metro', 0.12),
  STRUCT('Neurology', 'Metro',       0.12),
  STRUCT('Neurology', 'Micro',       0.10),
  STRUCT('Neurology', 'Rural',       0.10),
  STRUCT('Neurology', 'CEAC',        0.10),
  -- Neurosurgery
  STRUCT('Neurosurgery', 'Large Metro', 0.01),
  STRUCT('Neurosurgery', 'Metro',       0.01),
  STRUCT('Neurosurgery', 'Micro',       0.01),
  STRUCT('Neurosurgery', 'Rural',       0.01),
  STRUCT('Neurosurgery', 'CEAC',        0.01),
  -- Oncology Medical/Surgical
  STRUCT('Oncology Medical/Surgical', 'Large Metro', 0.19),
  STRUCT('Oncology Medical/Surgical', 'Metro',       0.19),
  STRUCT('Oncology Medical/Surgical', 'Micro',       0.16),
  STRUCT('Oncology Medical/Surgical', 'Rural',       0.16),
  STRUCT('Oncology Medical/Surgical', 'CEAC',        0.16),
  -- Oncology Radiation
  STRUCT('Oncology Radiation', 'Large Metro', 0.06),
  STRUCT('Oncology Radiation', 'Metro',       0.06),
  STRUCT('Oncology Radiation', 'Micro',       0.05),
  STRUCT('Oncology Radiation', 'Rural',       0.05),
  STRUCT('Oncology Radiation', 'CEAC',        0.05),
  -- Ophthalmology
  STRUCT('Ophthalmology', 'Large Metro', 0.24),
  STRUCT('Ophthalmology', 'Metro',       0.24),
  STRUCT('Ophthalmology', 'Micro',       0.20),
  STRUCT('Ophthalmology', 'Rural',       0.20),
  STRUCT('Ophthalmology', 'CEAC',        0.20),
  -- Orthopedic Surgery
  STRUCT('Orthopedic Surgery', 'Large Metro', 0.20),
  STRUCT('Orthopedic Surgery', 'Metro',       0.20),
  STRUCT('Orthopedic Surgery', 'Micro',       0.17),
  STRUCT('Orthopedic Surgery', 'Rural',       0.17),
  STRUCT('Orthopedic Surgery', 'CEAC',        0.17),
  -- Physiatry Rehabilitative Med
  STRUCT('Physiatry Rehabilitative Med', 'Large Metro', 0.04),
  STRUCT('Physiatry Rehabilitative Med', 'Metro',       0.04),
  STRUCT('Physiatry Rehabilitative Med', 'Micro',       0.03),
  STRUCT('Physiatry Rehabilitative Med', 'Rural',       0.03),
  STRUCT('Physiatry Rehabilitative Med', 'CEAC',        0.03),
  -- Plastic Surgery
  STRUCT('Plastic Surgery', 'Large Metro', 0.01),
  STRUCT('Plastic Surgery', 'Metro',       0.01),
  STRUCT('Plastic Surgery', 'Micro',       0.01),
  STRUCT('Plastic Surgery', 'Rural',       0.01),
  STRUCT('Plastic Surgery', 'CEAC',        0.01),
  -- Podiatry
  STRUCT('Podiatry', 'Large Metro', 0.19),
  STRUCT('Podiatry', 'Metro',       0.19),
  STRUCT('Podiatry', 'Micro',       0.16),
  STRUCT('Podiatry', 'Rural',       0.16),
  STRUCT('Podiatry', 'CEAC',        0.16),
  -- Psychiatry
  STRUCT('Psychiatry', 'Large Metro', 0.14),
  STRUCT('Psychiatry', 'Metro',       0.14),
  STRUCT('Psychiatry', 'Micro',       0.12),
  STRUCT('Psychiatry', 'Rural',       0.12),
  STRUCT('Psychiatry', 'CEAC',        0.12),
  -- Pulmonology
  STRUCT('Pulmonology', 'Large Metro', 0.13),
  STRUCT('Pulmonology', 'Metro',       0.13),
  STRUCT('Pulmonology', 'Micro',       0.11),
  STRUCT('Pulmonology', 'Rural',       0.11),
  STRUCT('Pulmonology', 'CEAC',        0.11),
  -- Rheumatology
  STRUCT('Rheumatology', 'Large Metro', 0.07),
  STRUCT('Rheumatology', 'Metro',       0.07),
  STRUCT('Rheumatology', 'Micro',       0.06),
  STRUCT('Rheumatology', 'Rural',       0.06),
  STRUCT('Rheumatology', 'CEAC',        0.06),
  -- Urology
  STRUCT('Urology', 'Large Metro', 0.12),
  STRUCT('Urology', 'Metro',       0.12),
  STRUCT('Urology', 'Micro',       0.10),
  STRUCT('Urology', 'Rural',       0.10),
  STRUCT('Urology', 'CEAC',        0.10),
  -- Vascular Surgery
  STRUCT('Vascular Surgery', 'Large Metro', 0.02),
  STRUCT('Vascular Surgery', 'Metro',       0.02),
  STRUCT('Vascular Surgery', 'Micro',       0.02),
  STRUCT('Vascular Surgery', 'Rural',       0.02),
  STRUCT('Vascular Surgery', 'CEAC',        0.02),
  -- Cardiothoracic Surgery
  STRUCT('Cardiothoracic Surgery', 'Large Metro', 0.01),
  STRUCT('Cardiothoracic Surgery', 'Metro',       0.01),
  STRUCT('Cardiothoracic Surgery', 'Micro',       0.01),
  STRUCT('Cardiothoracic Surgery', 'Rural',       0.01),
  STRUCT('Cardiothoracic Surgery', 'CEAC',        0.01),
  -- Acute Inpatient Hospitals (beds per 1,000 not providers)
  STRUCT('Acute Inpatient Hospitals', 'Large Metro', 12.2),
  STRUCT('Acute Inpatient Hospitals', 'Metro',       12.2),
  STRUCT('Acute Inpatient Hospitals', 'Micro',       12.2),
  STRUCT('Acute Inpatient Hospitals', 'Rural',       12.2),
  STRUCT('Acute Inpatient Hospitals', 'CEAC',        12.2),
  -- All other facility types: minimum = 1 per county per 422.116
  STRUCT('Cardiac Surgery Program',      'Large Metro', 1.0),
  STRUCT('Cardiac Surgery Program',      'Metro',       1.0),
  STRUCT('Cardiac Surgery Program',      'Micro',       1.0),
  STRUCT('Cardiac Surgery Program',      'Rural',       1.0),
  STRUCT('Cardiac Surgery Program',      'CEAC',        1.0),
  STRUCT('Cardiac Catheterization',      'Large Metro', 1.0),
  STRUCT('Cardiac Catheterization',      'Metro',       1.0),
  STRUCT('Cardiac Catheterization',      'Micro',       1.0),
  STRUCT('Cardiac Catheterization',      'Rural',       1.0),
  STRUCT('Cardiac Catheterization',      'CEAC',        1.0),
  STRUCT('Critical Care ICU',            'Large Metro', 1.0),
  STRUCT('Critical Care ICU',            'Metro',       1.0),
  STRUCT('Critical Care ICU',            'Micro',       1.0),
  STRUCT('Critical Care ICU',            'Rural',       1.0),
  STRUCT('Critical Care ICU',            'CEAC',        1.0),
  STRUCT('Surgical Services ASC',        'Large Metro', 1.0),
  STRUCT('Surgical Services ASC',        'Metro',       1.0),
  STRUCT('Surgical Services ASC',        'Micro',       1.0),
  STRUCT('Surgical Services ASC',        'Rural',       1.0),
  STRUCT('Surgical Services ASC',        'CEAC',        1.0),
  STRUCT('Skilled Nursing Facility',     'Large Metro', 1.0),
  STRUCT('Skilled Nursing Facility',     'Metro',       1.0),
  STRUCT('Skilled Nursing Facility',     'Micro',       1.0),
  STRUCT('Skilled Nursing Facility',     'Rural',       1.0),
  STRUCT('Skilled Nursing Facility',     'CEAC',        1.0),
  STRUCT('Diagnostic Radiology',         'Large Metro', 1.0),
  STRUCT('Diagnostic Radiology',         'Metro',       1.0),
  STRUCT('Diagnostic Radiology',         'Micro',       1.0),
  STRUCT('Diagnostic Radiology',         'Rural',       1.0),
  STRUCT('Diagnostic Radiology',         'CEAC',        1.0),
  STRUCT('Mammography',                  'Large Metro', 1.0),
  STRUCT('Mammography',                  'Metro',       1.0),
  STRUCT('Mammography',                  'Micro',       1.0),
  STRUCT('Mammography',                  'Rural',       1.0),
  STRUCT('Mammography',                  'CEAC',        1.0),
  STRUCT('Physical Therapy',             'Large Metro', 1.0),
  STRUCT('Physical Therapy',             'Metro',       1.0),
  STRUCT('Physical Therapy',             'Micro',       1.0),
  STRUCT('Physical Therapy',             'Rural',       1.0),
  STRUCT('Physical Therapy',             'CEAC',        1.0),
  STRUCT('Occupational Therapy',         'Large Metro', 1.0),
  STRUCT('Occupational Therapy',         'Metro',       1.0),
  STRUCT('Occupational Therapy',         'Micro',       1.0),
  STRUCT('Occupational Therapy',         'Rural',       1.0),
  STRUCT('Occupational Therapy',         'CEAC',        1.0),
  STRUCT('Speech Therapy',               'Large Metro', 1.0),
  STRUCT('Speech Therapy',               'Metro',       1.0),
  STRUCT('Speech Therapy',               'Micro',       1.0),
  STRUCT('Speech Therapy',               'Rural',       1.0),
  STRUCT('Speech Therapy',               'CEAC',        1.0),
  STRUCT('Inpatient Psychiatric',        'Large Metro', 1.0),
  STRUCT('Inpatient Psychiatric',        'Metro',       1.0),
  STRUCT('Inpatient Psychiatric',        'Micro',       1.0),
  STRUCT('Inpatient Psychiatric',        'Rural',       1.0),
  STRUCT('Inpatient Psychiatric',        'CEAC',        1.0),
  STRUCT('Outpatient Infusion/Chemo',    'Large Metro', 1.0),
  STRUCT('Outpatient Infusion/Chemo',    'Metro',       1.0),
  STRUCT('Outpatient Infusion/Chemo',    'Micro',       1.0),
  STRUCT('Outpatient Infusion/Chemo',    'Rural',       1.0),
  STRUCT('Outpatient Infusion/Chemo',    'CEAC',        1.0),
  STRUCT('Outpatient Behavioral Health', 'Large Metro', 1.0),
  STRUCT('Outpatient Behavioral Health', 'Metro',       1.0),
  STRUCT('Outpatient Behavioral Health', 'Micro',       1.0),
  STRUCT('Outpatient Behavioral Health', 'Rural',       1.0),
  STRUCT('Outpatient Behavioral Health', 'CEAC',        1.0)
]);


-- ============================================================
-- TABLE 4: ref_county_classification
-- PURPOSE: FLORIDA COUNTY TYPE CLASSIFICATION PER 42 CFR 422.116
-- SOURCE:  bigquery-public-data.geo_us_boundaries.counties
--          bigquery-public-data.census_bureau_acs.county_2020_5yr
-- GRAIN:   county_fips x county_name
-- KEY COLS: int_point_geom = centroid, county_geom = polygon
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_county_classification`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH raw_counties AS (
  -- --------------------------------------------------------
  -- FLORIDA COUNTIES: GEO + AREA + CENTROID
  -- --------------------------------------------------------
  SELECT
    geo_id                                                           AS county_fips,
    county_name,
    area_land_meters / 2589988.11                                   AS area_sq_miles,
    int_point_geom                                                   AS county_centroid
  FROM `bigquery-public-data.geo_us_boundaries.counties`
  WHERE state_fips_code = '12'
),

population AS (
  -- --------------------------------------------------------
  -- FLORIDA COUNTY POPULATION FROM ACS 2020 5-YEAR
  -- --------------------------------------------------------
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
    r.county_centroid,
    ST_Y(r.county_centroid)                                         AS county_lat,
    ST_X(r.county_centroid)                                         AS county_long,
    p.total_pop                                                      AS population,
    ROUND(p.total_pop / NULLIF(r.area_sq_miles, 0), 2)             AS pop_density,
    -- county radius for confidence interval: approximate circle radius
    ROUND(SQRT(r.area_sq_miles / ACOS(-1)), 2)                     AS county_radius_miles
  FROM raw_counties r
  LEFT JOIN population p USING (county_fips)
),

classified AS (
  -- --------------------------------------------------------
  -- APPLY 42 CFR 422.116 COUNTY TYPE RULES
  -- PRIORITY ORDER: LARGE METRO > METRO > MICRO > CEAC > RURAL
  -- --------------------------------------------------------
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
  county_lat,
  county_long,
  county_centroid,
  county_type,
  -- compliance threshold per 422.116 d(4)
  CASE
    WHEN county_type IN ('Large Metro', 'Metro') THEN 0.90
    ELSE 0.85
  END                                                                AS compliance_threshold
FROM classified
ORDER BY county_type, county_name;


-- ============================================================
-- TABLE 5: ref_zip_reference
-- PURPOSE: ZIP CODE GEO REFERENCE FOR FLORIDA
--          CENTROID LAT/LONG, AREA, RADIUS, COUNTY MAPPING
--          POPULATION FROM ACS 2018 5-YEAR
-- SOURCE:  bigquery-public-data.geo_us_boundaries.zip_codes
--          bigquery-public-data.census_bureau_acs.zip_codes_2018_5yr
--          bigquery-public-data.geo_us_boundaries.counties
-- GRAIN:   zip_code
-- KEY COLS: internal_point_geom = centroid, zip_code_geom = polygon
--           int_point_geom = county centroid, county_geom = county polygon
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_zip_reference`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH florida_zips AS (
  -- --------------------------------------------------------
  -- GET ALL ZIPS THAT INTERSECT FLORIDA COUNTIES
  -- USE ST_INTERSECTS AGAINST COUNTY POLYGONS
  -- --------------------------------------------------------
  SELECT
    z.zip_code,
    z.area_land_meters / 2589988.11                                 AS area_sq_miles,
    ROUND(SQRT((z.area_land_meters / 2589988.11) / ACOS(-1)), 2)   AS zip_radius_miles,
    ST_Y(z.internal_point_geom)                                     AS zip_lat,
    ST_X(z.internal_point_geom)                                     AS zip_long,
    z.internal_point_geom                                           AS zip_centroid,
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
  -- --------------------------------------------------------
  -- ZIP LEVEL POPULATION FROM ACS 2018 5-YEAR
  -- NOTE: geo_id = 5 digit zip code
  -- --------------------------------------------------------
  SELECT
    geo_id                                                           AS zip_code,
    total_pop
  FROM `bigquery-public-data.census_bureau_acs.zip_codes_2018_5yr`
),

zip_to_county AS (
  -- --------------------------------------------------------
  -- MAP EACH ZIP TO ITS PRIMARY COUNTY
  -- USING LARGEST INTERSECTION AREA WHERE ZIP CROSSES BOUNDARY
  -- --------------------------------------------------------
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
-- TABLE 6: stg_beneficiaries
-- PURPOSE: BENEFICIARY DEMAND BY ZIP CODE FOR FLORIDA
-- SOURCE:  bigquery-public-data.census_bureau_acs.zip_codes_2018_5yr
--          A870800_medicare_supply_demand_ref_zip_reference
--          anbc-hcb-prod cms_medicare_penetration (county validation)
-- GRAIN:   zip_code
-- NOTE:    PRIMARY DEMAND = ACS 2018 zip population
--          CMS PENETRATION = county level validation/context only
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_beneficiaries`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH latest_penetration AS (
  -- --------------------------------------------------------
  -- GET MOST RECENT CMS PENETRATION FILE
  -- USED FOR COUNTY LEVEL VALIDATION CONTEXT ONLY
  -- --------------------------------------------------------
  SELECT MAX(ingest_time) AS max_ingest
  FROM `anbc-hcb-prod.provider_ds_netconf_data_hcb_prod.cms_medicare_penetration`
),

county_penetration AS (
  -- --------------------------------------------------------
  -- FLORIDA COUNTY LEVEL MA PENETRATION
  -- BUILD FULL 5-DIGIT FIPS FROM fipsst + fipscnty
  -- --------------------------------------------------------
  SELECT
    CONCAT(
      LPAD(CAST(fipsst   AS STRING), 2, '0'),
      LPAD(CAST(fipscnty AS STRING), 3, '0')
    )                                                                AS county_fips,
    county_name,
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
  -- county level cms context for gap calculation denominator only
  p.county_eligibles,
  p.county_ma_enrolled,
  p.county_penetration_rate,
  p.data_as_of
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_zip_reference` z
LEFT JOIN county_penetration p
  ON z.county_fips = p.county_fips
ORDER BY z.zip_code;


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
  -- lat/long used in fact_zip_access for ST_DISTANCE calc
  -- zip_centroid excluded - reconstructed at query time via ST_GEOGPOINT
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
-- EXCLUDE UNMAPPED SPECIALTIES + OUT OF STATE PROVIDERS
-- GROUP BY ALL instead of DISTINCT - avoids GEOGRAPHY type issues
-- --------------------------------------------------------
SELECT
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
  zip_radius_miles,
  county_type,
  plan_type,
  market,
  submarket
FROM mapped_zip
WHERE cms_specialty IS NOT NULL
  AND zip_lat IS NOT NULL
GROUP BY ALL
ORDER BY provider_id, cms_specialty, plan_type


-- ============================================================
-- TABLE 9: fact_zip_access
-- PURPOSE: FOR EACH BENEFICIARY ZIP x SPECIALTY x PLAN TYPE
--          COUNT PROVIDERS WITHIN CMS DISTANCE THRESHOLD
--          TAG ZIP AS HAS_ACCESS (TRUE/FALSE)
-- SOURCE:  stg_beneficiaries
--          stg_providers
--          ref_zip_reference
--          ref_time_distance
-- GRAIN:   bene_zip x cms_specialty x plan_type
-- NOTE:    SPARSE TABLE - ONLY ZIPS WITH AT LEAST 1 PROVIDER SURVIVE
--          ZEROS HANDLED IN fact_gap_analysis VIA LEFT JOIN
--          ST_DISTANCE in meters converted to miles (/1609.34)
--          THRESHOLD USES BENEFICIARY COUNTY TYPE PER 422.116
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_fact_zip_access`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH zip_provider_pairs AS (
  -- --------------------------------------------------------
  -- CROSS JOIN BENE ZIPS x PROVIDERS
  -- FILTER TO PAIRS WITHIN CMS DISTANCE THRESHOLD
  -- THRESHOLD LOOKUP USES BENEFICIARY COUNTY TYPE
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
    -- distance in miles between bene zip centroid and provider zip centroid
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

  -- cross join to all providers
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers` p
    ON TRUE

  -- threshold lookup: uses beneficiary county type not provider county type
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_time_distance` t
    ON t.cms_specialty = p.cms_specialty
    AND t.county_type  = b.county_type

  -- core filter: only provider zips within CMS threshold survive
  WHERE ST_DISTANCE(
          ST_GEOGPOINT(bene_zip.zip_long, bene_zip.zip_lat),
          ST_GEOGPOINT(p.zip_long,        p.zip_lat)
        ) / 1609.34 <= t.max_distance_miles
)

-- --------------------------------------------------------
-- AGGREGATE TO GRAIN: bene_zip x cms_specialty x plan_type
-- COUNT DISTINCT PROVIDERS WITHIN THRESHOLD
-- --------------------------------------------------------
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
  inflated,
  match_type,
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
  inflated,
  match_type,
  max_distance_miles;


-- ============================================================
-- TABLE 10: fact_gap_analysis
-- PURPOSE: COUNTY LEVEL COMPLIANCE ROLLUP
-- SOURCE:  fact_zip_access
--          ref_hsd_required_counts (CMS 2026 HSD - exact required counts)
--          stg_beneficiaries
--          ref_specialty_crosswalk
-- GRAIN:   county_fips x cms_specialty x plan_type
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_fact_gap_analysis`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH all_combinations AS (
  -- --------------------------------------------------------
  -- BUILD COMPLETE GRID:
  -- ALL BENE ZIPS x ALL CMS SPECIALTIES x ALL PLAN TYPES
  -- --------------------------------------------------------
  SELECT
    b.zip_code,
    b.county_fips,
    b.county_name,
    b.county_type,
    b.compliance_threshold,
    b.total_population,
    sc.cms_specialty,
    sc.match_type,
    sc.inflated,
    pt.plan_type
  FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_beneficiaries` b
  CROSS JOIN (
    SELECT DISTINCT cms_specialty, match_type, inflated
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_specialty_crosswalk`
  ) sc
  CROSS JOIN (
    SELECT DISTINCT plan_type
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers`
  ) pt
),

zip_access_complete AS (
  -- --------------------------------------------------------
  -- LEFT JOIN FACT_ZIP_ACCESS
  -- FILLS IN ZEROS FOR ZIPS WITH NO PROVIDERS WITHIN THRESHOLD
  -- --------------------------------------------------------
  SELECT
    a.zip_code,
    a.county_fips,
    a.county_name,
    a.county_type,
    a.compliance_threshold,
    a.total_population,
    a.cms_specialty,
    a.match_type,
    a.inflated,
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
    inflated,
    match_type
)

-- --------------------------------------------------------
-- FINAL OUTPUT
-- required_count from CMS HSD file directly
-- no approximation, no ratio calculation
-- --------------------------------------------------------
SELECT
  r.county_fips,
  r.county_name,
  r.county_type,
  r.cms_specialty,
  r.plan_type,
  r.inflated,
  r.match_type,
  hsd.total_beneficiaries                                            AS county_total_beneficiaries,
  hsd.beneficiaries_required_to_cover,
  hsd.ratio_95th_percentile,
  r.total_county_population,
  r.population_with_access,
  r.pct_covered,
  r.compliance_threshold,
  hsd.required_count                                                 AS required_provider_count,
  r.actual_provider_count,
  hsd.required_count - r.actual_provider_count                      AS provider_gap,
  -- test 1: access
  CASE
    WHEN r.pct_covered >= r.compliance_threshold THEN TRUE
    ELSE FALSE
  END                                                                AS access_compliant,
  -- test 2: count
  CASE
    WHEN r.actual_provider_count >= hsd.required_count THEN TRUE
    ELSE FALSE
  END                                                                AS count_compliant,
  -- overall
  CASE
    WHEN r.pct_covered >= r.compliance_threshold
    AND  r.actual_provider_count >= hsd.required_count               THEN 'COMPLIANT'
    ELSE 'NON-COMPLIANT'
  END                                                                AS compliance_status

FROM county_rollup r
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_hsd_required_counts` hsd
  ON hsd.county_name  = r.county_name
  AND hsd.cms_specialty = r.cms_specialty
ORDER BY
  r.county_name,
  r.cms_specialty,
  r.plan_type
