-- ============================================================
-- REF TABLE: MINIMUM PROVIDER RATIO PER 1,000 BENEFICIARIES
-- SOURCE: 42 CFR 422.116 TABLE 2
-- GRAIN: cms_specialty x county_type
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_min_ratio`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT * FROM UNNEST([

  -- --------------------------------------------------------
  -- PRIMARY CARE
  -- --------------------------------------------------------
  STRUCT('Primary Care' AS cms_specialty, 'Large Metro' AS county_type, 1.67 AS min_ratio_per_1000),
  STRUCT('Primary Care', 'Metro',       1.67),
  STRUCT('Primary Care', 'Micro',       1.42),
  STRUCT('Primary Care', 'Rural',       1.42),
  STRUCT('Primary Care', 'CEAC',        1.42),

  -- --------------------------------------------------------
  -- ALLERGY AND IMMUNOLOGY
  -- --------------------------------------------------------
  STRUCT('Allergy and Immunology', 'Large Metro', 0.05),
  STRUCT('Allergy and Immunology', 'Metro',       0.05),
  STRUCT('Allergy and Immunology', 'Micro',       0.04),
  STRUCT('Allergy and Immunology', 'Rural',       0.04),
  STRUCT('Allergy and Immunology', 'CEAC',        0.04),

  -- --------------------------------------------------------
  -- CARDIOLOGY
  -- --------------------------------------------------------
  STRUCT('Cardiology', 'Large Metro', 0.27),
  STRUCT('Cardiology', 'Metro',       0.27),
  STRUCT('Cardiology', 'Micro',       0.23),
  STRUCT('Cardiology', 'Rural',       0.23),
  STRUCT('Cardiology', 'CEAC',        0.23),

  -- --------------------------------------------------------
  -- CHIROPRACTOR
  -- --------------------------------------------------------
  STRUCT('Chiropractor', 'Large Metro', 0.10),
  STRUCT('Chiropractor', 'Metro',       0.10),
  STRUCT('Chiropractor', 'Micro',       0.09),
  STRUCT('Chiropractor', 'Rural',       0.09),
  STRUCT('Chiropractor', 'CEAC',        0.09),

  -- --------------------------------------------------------
  -- CLINICAL PSYCHOLOGY
  -- --------------------------------------------------------
  STRUCT('Clinical Psychology', 'Large Metro', 0.15),
  STRUCT('Clinical Psychology', 'Metro',       0.15),
  STRUCT('Clinical Psychology', 'Micro',       0.13),
  STRUCT('Clinical Psychology', 'Rural',       0.13),
  STRUCT('Clinical Psychology', 'CEAC',        0.13),

  -- --------------------------------------------------------
  -- CLINICAL SOCIAL WORK
  -- --------------------------------------------------------
  STRUCT('Clinical Social Work', 'Large Metro', 0.25),
  STRUCT('Clinical Social Work', 'Metro',       0.25),
  STRUCT('Clinical Social Work', 'Micro',       0.22),
  STRUCT('Clinical Social Work', 'Rural',       0.22),
  STRUCT('Clinical Social Work', 'CEAC',        0.22),

  -- --------------------------------------------------------
  -- DERMATOLOGY
  -- --------------------------------------------------------
  STRUCT('Dermatology', 'Large Metro', 0.16),
  STRUCT('Dermatology', 'Metro',       0.16),
  STRUCT('Dermatology', 'Micro',       0.14),
  STRUCT('Dermatology', 'Rural',       0.14),
  STRUCT('Dermatology', 'CEAC',        0.14),

  -- --------------------------------------------------------
  -- ENDOCRINOLOGY
  -- --------------------------------------------------------
  STRUCT('Endocrinology', 'Large Metro', 0.04),
  STRUCT('Endocrinology', 'Metro',       0.04),
  STRUCT('Endocrinology', 'Micro',       0.03),
  STRUCT('Endocrinology', 'Rural',       0.03),
  STRUCT('Endocrinology', 'CEAC',        0.03),

  -- --------------------------------------------------------
  -- ENT/OTOLARYNGOLOGY
  -- --------------------------------------------------------
  STRUCT('ENT/Otolaryngology', 'Large Metro', 0.06),
  STRUCT('ENT/Otolaryngology', 'Metro',       0.06),
  STRUCT('ENT/Otolaryngology', 'Micro',       0.05),
  STRUCT('ENT/Otolaryngology', 'Rural',       0.05),
  STRUCT('ENT/Otolaryngology', 'CEAC',        0.05),

  -- --------------------------------------------------------
  -- GASTROENTEROLOGY
  -- --------------------------------------------------------
  STRUCT('Gastroenterology', 'Large Metro', 0.12),
  STRUCT('Gastroenterology', 'Metro',       0.12),
  STRUCT('Gastroenterology', 'Micro',       0.10),
  STRUCT('Gastroenterology', 'Rural',       0.10),
  STRUCT('Gastroenterology', 'CEAC',        0.10),

  -- --------------------------------------------------------
  -- GENERAL SURGERY
  -- --------------------------------------------------------
  STRUCT('General Surgery', 'Large Metro', 0.28),
  STRUCT('General Surgery', 'Metro',       0.28),
  STRUCT('General Surgery', 'Micro',       0.24),
  STRUCT('General Surgery', 'Rural',       0.24),
  STRUCT('General Surgery', 'CEAC',        0.24),

  -- --------------------------------------------------------
  -- GYNECOLOGY OB/GYN
  -- --------------------------------------------------------
  STRUCT('Gynecology OB/GYN', 'Large Metro', 0.04),
  STRUCT('Gynecology OB/GYN', 'Metro',       0.04),
  STRUCT('Gynecology OB/GYN', 'Micro',       0.03),
  STRUCT('Gynecology OB/GYN', 'Rural',       0.03),
  STRUCT('Gynecology OB/GYN', 'CEAC',        0.03),

  -- --------------------------------------------------------
  -- INFECTIOUS DISEASES
  -- --------------------------------------------------------
  STRUCT('Infectious Diseases', 'Large Metro', 0.03),
  STRUCT('Infectious Diseases', 'Metro',       0.03),
  STRUCT('Infectious Diseases', 'Micro',       0.03),
  STRUCT('Infectious Diseases', 'Rural',       0.03),
  STRUCT('Infectious Diseases', 'CEAC',        0.03),

  -- --------------------------------------------------------
  -- NEPHROLOGY
  -- --------------------------------------------------------
  STRUCT('Nephrology', 'Large Metro', 0.09),
  STRUCT('Nephrology', 'Metro',       0.09),
  STRUCT('Nephrology', 'Micro',       0.08),
  STRUCT('Nephrology', 'Rural',       0.08),
  STRUCT('Nephrology', 'CEAC',        0.08),

  -- --------------------------------------------------------
  -- NEUROLOGY
  -- --------------------------------------------------------
  STRUCT('Neurology', 'Large Metro', 0.12),
  STRUCT('Neurology', 'Metro',       0.12),
  STRUCT('Neurology', 'Micro',       0.10),
  STRUCT('Neurology', 'Rural',       0.10),
  STRUCT('Neurology', 'CEAC',        0.10),

  -- --------------------------------------------------------
  -- NEUROSURGERY
  -- --------------------------------------------------------
  STRUCT('Neurosurgery', 'Large Metro', 0.01),
  STRUCT('Neurosurgery', 'Metro',       0.01),
  STRUCT('Neurosurgery', 'Micro',       0.01),
  STRUCT('Neurosurgery', 'Rural',       0.01),
  STRUCT('Neurosurgery', 'CEAC',        0.01),

  -- --------------------------------------------------------
  -- ONCOLOGY MEDICAL/SURGICAL
  -- --------------------------------------------------------
  STRUCT('Oncology Medical/Surgical', 'Large Metro', 0.19),
  STRUCT('Oncology Medical/Surgical', 'Metro',       0.19),
  STRUCT('Oncology Medical/Surgical', 'Micro',       0.16),
  STRUCT('Oncology Medical/Surgical', 'Rural',       0.16),
  STRUCT('Oncology Medical/Surgical', 'CEAC',        0.16),

  -- --------------------------------------------------------
  -- ONCOLOGY RADIATION
  -- --------------------------------------------------------
  STRUCT('Oncology Radiation', 'Large Metro', 0.06),
  STRUCT('Oncology Radiation', 'Metro',       0.06),
  STRUCT('Oncology Radiation', 'Micro',       0.05),
  STRUCT('Oncology Radiation', 'Rural',       0.05),
  STRUCT('Oncology Radiation', 'CEAC',        0.05),

  -- --------------------------------------------------------
  -- OPHTHALMOLOGY
  -- --------------------------------------------------------
  STRUCT('Ophthalmology', 'Large Metro', 0.24),
  STRUCT('Ophthalmology', 'Metro',       0.24),
  STRUCT('Ophthalmology', 'Micro',       0.20),
  STRUCT('Ophthalmology', 'Rural',       0.20),
  STRUCT('Ophthalmology', 'CEAC',        0.20),

  -- --------------------------------------------------------
  -- ORTHOPEDIC SURGERY
  -- --------------------------------------------------------
  STRUCT('Orthopedic Surgery', 'Large Metro', 0.20),
  STRUCT('Orthopedic Surgery', 'Metro',       0.20),
  STRUCT('Orthopedic Surgery', 'Micro',       0.17),
  STRUCT('Orthopedic Surgery', 'Rural',       0.17),
  STRUCT('Orthopedic Surgery', 'CEAC',        0.17),

  -- --------------------------------------------------------
  -- PHYSIATRY REHABILITATIVE MED
  -- --------------------------------------------------------
  STRUCT('Physiatry Rehabilitative Med', 'Large Metro', 0.04),
  STRUCT('Physiatry Rehabilitative Med', 'Metro',       0.04),
  STRUCT('Physiatry Rehabilitative Med', 'Micro',       0.03),
  STRUCT('Physiatry Rehabilitative Med', 'Rural',       0.03),
  STRUCT('Physiatry Rehabilitative Med', 'CEAC',        0.03),

  -- --------------------------------------------------------
  -- PLASTIC SURGERY
  -- --------------------------------------------------------
  STRUCT('Plastic Surgery', 'Large Metro', 0.01),
  STRUCT('Plastic Surgery', 'Metro',       0.01),
  STRUCT('Plastic Surgery', 'Micro',       0.01),
  STRUCT('Plastic Surgery', 'Rural',       0.01),
  STRUCT('Plastic Surgery', 'CEAC',        0.01),

  -- --------------------------------------------------------
  -- PODIATRY
  -- --------------------------------------------------------
  STRUCT('Podiatry', 'Large Metro', 0.19),
  STRUCT('Podiatry', 'Metro',       0.19),
  STRUCT('Podiatry', 'Micro',       0.16),
  STRUCT('Podiatry', 'Rural',       0.16),
  STRUCT('Podiatry', 'CEAC',        0.16),

  -- --------------------------------------------------------
  -- PSYCHIATRY
  -- --------------------------------------------------------
  STRUCT('Psychiatry', 'Large Metro', 0.14),
  STRUCT('Psychiatry', 'Metro',       0.14),
  STRUCT('Psychiatry', 'Micro',       0.12),
  STRUCT('Psychiatry', 'Rural',       0.12),
  STRUCT('Psychiatry', 'CEAC',        0.12),

  -- --------------------------------------------------------
  -- PULMONOLOGY
  -- --------------------------------------------------------
  STRUCT('Pulmonology', 'Large Metro', 0.13),
  STRUCT('Pulmonology', 'Metro',       0.13),
  STRUCT('Pulmonology', 'Micro',       0.11),
  STRUCT('Pulmonology', 'Rural',       0.11),
  STRUCT('Pulmonology', 'CEAC',        0.11),

  -- --------------------------------------------------------
  -- RHEUMATOLOGY
  -- --------------------------------------------------------
  STRUCT('Rheumatology', 'Large Metro', 0.07),
  STRUCT('Rheumatology', 'Metro',       0.07),
  STRUCT('Rheumatology', 'Micro',       0.06),
  STRUCT('Rheumatology', 'Rural',       0.06),
  STRUCT('Rheumatology', 'CEAC',        0.06),

  -- --------------------------------------------------------
  -- UROLOGY
  -- --------------------------------------------------------
  STRUCT('Urology', 'Large Metro', 0.12),
  STRUCT('Urology', 'Metro',       0.12),
  STRUCT('Urology', 'Micro',       0.10),
  STRUCT('Urology', 'Rural',       0.10),
  STRUCT('Urology', 'CEAC',        0.10),

  -- --------------------------------------------------------
  -- VASCULAR SURGERY
  -- --------------------------------------------------------
  STRUCT('Vascular Surgery', 'Large Metro', 0.02),
  STRUCT('Vascular Surgery', 'Metro',       0.02),
  STRUCT('Vascular Surgery', 'Micro',       0.02),
  STRUCT('Vascular Surgery', 'Rural',       0.02),
  STRUCT('Vascular Surgery', 'CEAC',        0.02),

  -- --------------------------------------------------------
  -- CARDIOTHORACIC SURGERY
  -- --------------------------------------------------------
  STRUCT('Cardiothoracic Surgery', 'Large Metro', 0.01),
  STRUCT('Cardiothoracic Surgery', 'Metro',       0.01),
  STRUCT('Cardiothoracic Surgery', 'Micro',       0.01),
  STRUCT('Cardiothoracic Surgery', 'Rural',       0.01),
  STRUCT('Cardiothoracic Surgery', 'CEAC',        0.01),

  -- --------------------------------------------------------
  -- ACUTE INPATIENT HOSPITALS
  -- NOTE: ratio = beds per 1,000 beneficiaries not providers
  -- --------------------------------------------------------
  STRUCT('Acute Inpatient Hospitals', 'Large Metro', 12.2),
  STRUCT('Acute Inpatient Hospitals', 'Metro',       12.2),
  STRUCT('Acute Inpatient Hospitals', 'Micro',       12.2),
  STRUCT('Acute Inpatient Hospitals', 'Rural',       12.2),
  STRUCT('Acute Inpatient Hospitals', 'CEAC',        12.2),

  -- --------------------------------------------------------
  -- FACILITY TYPES: MIN = 1 PER COUNTY (per 422.116)
  -- --------------------------------------------------------
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

])
