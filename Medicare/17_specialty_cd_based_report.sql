-- ============================================================
-- TABLE: ref_specialty_crosswalk_expanded
-- PURPOSE: EXPANDED CMS SPECIALTY TO AETNA SPECIALTY_CD MAPPING
--          MAPS ON specialty_cd (raw code from RPDB_RPNPRAC)
--          NOT specialty_ctg_cd (category code)
--          442 ROWS - ALL KNOWN AETNA CODES PER CMS SPECIALTY
-- SOURCE:  Global Lookup Table (SPECIALTY_CD)
--          CMS 42 CFR 422.116 specialty definitions
-- NOTE:    ref_specialty_crosswalk kept as backup (specialty_ctg_cd)
--          Review bad matches before production use
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_specialty_crosswalk_expanded`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT * FROM UNNEST([
  STRUCT('Primary Care' AS cms_specialty, '10101' AS aetna_code, 'General Practice' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '10201' AS aetna_code, 'Family Practice' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '10202' AS aetna_code, 'Geriatric Medicine/Family Prac' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '10301' AS aetna_code, 'Internal Medicine' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '10308' AS aetna_code, 'Geriatric Medicine/Internal Me' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '10433' AS aetna_code, 'Sports Medicine/Pediatrics' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '10438' AS aetna_code, 'Pediatrics Hospice and Pallia' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '30609' AS aetna_code, 'Otolaryngology (Pediatrics)' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '50101' AS aetna_code, 'General Practice - Dental' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '90360' AS aetna_code, 'Oral Surgery (Pediatrics)' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '91151' AS aetna_code, 'Sleep Medicine-Family Practice' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '91154' AS aetna_code, 'Obesity Medicine-Pediatrics' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '91209' AS aetna_code, 'Config-Primary Care' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '91210' AS aetna_code, 'Config-Primary Care Attestatio' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '2IM' AS aetna_code, 'Internal Medicine' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '2P' AS aetna_code, 'Pediatrics' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '2FP' AS aetna_code, 'Family Practice' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '2GP' AS aetna_code, 'General Practice' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '2I' AS aetna_code, 'Internal Medicine' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '10336' AS aetna_code, 'Internal Medicine Hospice' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '10401' AS aetna_code, 'Pediatrics' AS aetna_description),
  STRUCT('Primary Care' AS cms_specialty, '10421' AS aetna_code, 'Pediatric Internal Medicine' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '10326' AS aetna_code, 'Allergy/Immunology' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '10501' AS aetna_code, 'Allergy & Immunology' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '10603' AS aetna_code, 'Dermatological Immunology/Diag' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '90003' AS aetna_code, 'Allergy' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '90004' AS aetna_code, 'Allergy (Pediatric)' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '90335' AS aetna_code, 'Immunology' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '90386' AS aetna_code, 'Otolaryngology/Allergy' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '91124' AS aetna_code, 'Transplant & Immunology' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '2A' AS aetna_code, 'Allergy' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '2AIM' AS aetna_code, 'Immunology' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '2AIMP' AS aetna_code, 'Immunology (Pediatric)' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '2AP' AS aetna_code, 'Allergy (Pediatric)' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '2ENA' AS aetna_code, 'Otolaryngology/Allergy' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '2ENA' AS aetna_code, 'Otolaryngology/Allergy' AS aetna_description),
  STRUCT('Allergy and Immunology' AS cms_specialty, '10411' AS aetna_code, 'Pediatric Allergy & Immunology' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '10302' AS aetna_code, 'Cardiac Electrophysiology' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '10303' AS aetna_code, 'Cardiovascular Disease' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '10322' AS aetna_code, 'Cardiology' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '40312' AS aetna_code, 'Nuclear Cardiology' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '90313' AS aetna_code, 'Cardiology (Invasive)' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '91046' AS aetna_code, 'Cardiac Valve Replacement' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '91205' AS aetna_code, 'Cardiac Monitoring Service' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '2C' AS aetna_code, 'Cardiology' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '2CC' AS aetna_code, 'Cardiology (Pediatric)' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '2CEP' AS aetna_code, 'Cardiac Electrophysiology' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '2CI' AS aetna_code, 'Cardiology (Invasive)' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '2CS' AS aetna_code, 'Cardiothoracic/Cardiovascular' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '10332' AS aetna_code, 'Interventional Cardiology' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '10339' AS aetna_code, 'Cardiology Adv Heart Failure/' AS aetna_description),
  STRUCT('Cardiology' AS cms_specialty, '10403' AS aetna_code, 'Pediatric Cardiology' AS aetna_description),
  STRUCT('Chiropractor' AS cms_specialty, '91146' AS aetna_code, 'Chiropractics' AS aetna_description),
  STRUCT('Chiropractor' AS cms_specialty, '2CH' AS aetna_code, 'Chiropractics' AS aetna_description),
  STRUCT('Chiropractor' AS cms_specialty, 'DC' AS aetna_code, 'Chiropractor' AS aetna_description),
  STRUCT('Clinical Psychology' AS cms_specialty, '90305' AS aetna_code, 'Adolescent Psychology' AS aetna_description),
  STRUCT('Clinical Psychology' AS cms_specialty, '90314' AS aetna_code, 'Child Psychology' AS aetna_description),
  STRUCT('Clinical Psychology' AS cms_specialty, '91018' AS aetna_code, 'Psychological Testing' AS aetna_description),
  STRUCT('Clinical Psychology' AS cms_specialty, '91029' AS aetna_code, 'Neuropsych Testing' AS aetna_description),
  STRUCT('Clinical Psychology' AS cms_specialty, 'CP' AS aetna_code, 'Clinical Psychologist' AS aetna_description),
  STRUCT('Clinical Psychology' AS cms_specialty, 'NPS' AS aetna_code, 'Neuropsychologist' AS aetna_description),
  STRUCT('Clinical Psychology' AS cms_specialty, '2NPH' AS aetna_code, 'Neuropsychology' AS aetna_description),
  STRUCT('Clinical Psychology' AS cms_specialty, '2PHA' AS aetna_code, 'Adolescent Psychology' AS aetna_description),
  STRUCT('Clinical Psychology' AS cms_specialty, '2PHGR' AS aetna_code, 'Geriatric Psychology' AS aetna_description),
  STRUCT('Clinical Psychology' AS cms_specialty, '2PHP' AS aetna_code, 'Child Psychology' AS aetna_description),
  STRUCT('Clinical Social Work' AS cms_specialty, '90371' AS aetna_code, 'Psychiatric Social Worker' AS aetna_description),
  STRUCT('Clinical Social Work' AS cms_specialty, '91207' AS aetna_code, 'Certified Social Work' AS aetna_description),
  STRUCT('Clinical Social Work' AS cms_specialty, '2MLS' AS aetna_code, 'Social Worker Masters Licensed' AS aetna_description),
  STRUCT('Clinical Social Work' AS cms_specialty, '2MUS' AS aetna_code, 'Social Worker(Masters w/o Lic)' AS aetna_description),
  STRUCT('Clinical Social Work' AS cms_specialty, 'SW' AS aetna_code, 'Clinical Social Worker' AS aetna_description),
  STRUCT('Clinical Social Work' AS cms_specialty, '2PYSW' AS aetna_code, 'Psychiatric Social Worker' AS aetna_description),
  STRUCT('Dermatology' AS cms_specialty, '10430' AS aetna_code, 'Pediatric Dermatology' AS aetna_description),
  STRUCT('Dermatology' AS cms_specialty, '10601' AS aetna_code, 'Dermatology' AS aetna_description),
  STRUCT('Dermatology' AS cms_specialty, '10602' AS aetna_code, 'Dermatopathology/Dermatology' AS aetna_description),
  STRUCT('Dermatology' AS cms_specialty, '40207' AS aetna_code, 'Dermatopathology/Pathology' AS aetna_description),
  STRUCT('Dermatology' AS cms_specialty, '2D' AS aetna_code, 'Dermatology' AS aetna_description),
  STRUCT('Dermatology' AS cms_specialty, '2DP' AS aetna_code, 'Dermatopathology' AS aetna_description),
  STRUCT('Dermatology' AS cms_specialty, '2DPD' AS aetna_code, 'Dermatology (Pediatric)' AS aetna_description),
  STRUCT('Endocrinology' AS cms_specialty, '10306' AS aetna_code, 'Endocrinology Diabetes & Meta' AS aetna_description),
  STRUCT('Endocrinology' AS cms_specialty, '10319' AS aetna_code, 'Endocrinology' AS aetna_description),
  STRUCT('Endocrinology' AS cms_specialty, '20105' AS aetna_code, 'Endocrinology Reproductive' AS aetna_description),
  STRUCT('Endocrinology' AS cms_specialty, '2E' AS aetna_code, 'Endocrinology' AS aetna_description),
  STRUCT('Endocrinology' AS cms_specialty, '2PE' AS aetna_code, 'Endocrinology (Pediatric)' AS aetna_description),
  STRUCT('Endocrinology' AS cms_specialty, '10405' AS aetna_code, 'Pediatric Endocrinology' AS aetna_description),
  STRUCT('Endocrinology' AS cms_specialty, '91059' AS aetna_code, 'Endocrine Surgery' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '30601' AS aetna_code, 'Otolaryngology' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '30603' AS aetna_code, 'Otorhinolaryngology & Oro-Faci' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '30604' AS aetna_code, 'Otorhinolaryngology/Plastic Su' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '30605' AS aetna_code, 'Otology' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '30607' AS aetna_code, 'Otorhinolaryngology' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '30608' AS aetna_code, 'Otology/Neurotology' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '30609' AS aetna_code, 'Otolaryngology (Pediatrics)' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '30806' AS aetna_code, 'Surgery Head & Neck' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '90386' AS aetna_code, 'Otolaryngology/Allergy' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '91117' AS aetna_code, 'Sleep Medicine (Otolaryngology)' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '2EN' AS aetna_code, 'Otolaryngology' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '2ENA' AS aetna_code, 'Otolaryngology/Allergy' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '2PEN' AS aetna_code, 'Otolaryngology (Pediatric)' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '2ENA' AS aetna_code, 'Otolaryngology/Allergy' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '2ENHN' AS aetna_code, 'Otolaryngology(Head&Neck) Su' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '2ENN' AS aetna_code, 'Neuro-Otology' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '10417' AS aetna_code, 'Pediatric Otolaryngology' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '14601' AS aetna_code, 'Neurotology' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '91061' AS aetna_code, 'ENT Trauma' AS aetna_description),
  STRUCT('ENT/Otolaryngology' AS cms_specialty, '91078' AS aetna_code, 'Otolaryngology (ENT) Cancer Su' AS aetna_description),
  STRUCT('Gastroenterology' AS cms_specialty, '10307' AS aetna_code, 'Gastroenterology' AS aetna_description),
  STRUCT('Gastroenterology' AS cms_specialty, '91045' AS aetna_code, 'Capsule Endoscopy' AS aetna_description),
  STRUCT('Gastroenterology' AS cms_specialty, '2PG' AS aetna_code, 'Gastroenterology (Pediatric)' AS aetna_description),
  STRUCT('Gastroenterology' AS cms_specialty, '2G' AS aetna_code, 'Gastroenterology' AS aetna_description),
  STRUCT('Gastroenterology' AS cms_specialty, '2PG' AS aetna_code, 'Gastroenterology (Pediatric)' AS aetna_description),
  STRUCT('Gastroenterology' AS cms_specialty, '10406' AS aetna_code, 'Pediatric Gastroenterology' AS aetna_description),
  STRUCT('Gastroenterology' AS cms_specialty, '91060' AS aetna_code, 'Endoscopic Ultrasound' AS aetna_description),
  STRUCT('Gastroenterology' AS cms_specialty, '91063' AS aetna_code, 'Endoscopic Retrograde Cholangi' AS aetna_description),
  STRUCT('Gastroenterology' AS cms_specialty, '91065' AS aetna_code, 'Esophageal Motility Disorders' AS aetna_description),
  STRUCT('General Surgery' AS cms_specialty, '30502' AS aetna_code, 'Surgery Hand/Orthopedic' AS aetna_description),
  STRUCT('General Surgery' AS cms_specialty, '30702' AS aetna_code, 'Surgery Hand/Plastic' AS aetna_description),
  STRUCT('General Surgery' AS cms_specialty, '30803' AS aetna_code, 'Surgery Critical care' AS aetna_description),
  STRUCT('General Surgery' AS cms_specialty, '30804' AS aetna_code, 'Surgery General Vascular' AS aetna_description),
  STRUCT('General Surgery' AS cms_specialty, '30809' AS aetna_code, 'Surgery Hand' AS aetna_description),
  STRUCT('General Surgery' AS cms_specialty, '30810' AS aetna_code, 'Surgery Oncology' AS aetna_description),
  STRUCT('General Surgery' AS cms_specialty, '30811' AS aetna_code, 'Surgery Hospice and Palliativ' AS aetna_description),
  STRUCT('General Surgery' AS cms_specialty, '2S' AS aetna_code, 'Surgery (General)' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '20104' AS aetna_code, 'Maternal & Fetal Medicine' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '20106' AS aetna_code, 'Gynecology' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '20107' AS aetna_code, 'Perinatology' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '20108' AS aetna_code, 'Obstetrics & Gynecology - CA P' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '20109' AS aetna_code, 'Obstetrics/Gynecology Hospice' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '20110' AS aetna_code, 'Female Pelvic Medicine & Recon' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '30807' AS aetna_code, 'Surgery Obstetrics & Gynecolo' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '90069' AS aetna_code, 'Perinatology/PF' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '90304' AS aetna_code, 'Adolescent Gynecology' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '90355' AS aetna_code, 'Obstetrics' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '90398' AS aetna_code, 'Uro-Gynecology' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '91097' AS aetna_code, 'Pediatric Gynecology' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '91102' AS aetna_code, 'Pediatric Uro-Gynecology' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '2PAOG' AS aetna_code, 'Physicians Assistant Ob/Gyn' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '2NPOG' AS aetna_code, 'Nurse practitioner (ob/gyn)' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '2OG' AS aetna_code, 'Ob/Gyn' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '2OGA' AS aetna_code, 'Adolescent Gynecology' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '2OGOB' AS aetna_code, 'Obstetrics' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '2OH' AS aetna_code, 'Perinatology' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '2UGY' AS aetna_code, 'Uro-gynecology' AS aetna_description),
  STRUCT('Gynecology OB/GYN' AS cms_specialty, '20191' AS aetna_code, 'Obstetrics & Gynecology' AS aetna_description),
  STRUCT('Infectious Diseases' AS cms_specialty, '10310' AS aetna_code, 'Infectious Disease' AS aetna_description),
  STRUCT('Infectious Diseases' AS cms_specialty, '91158' AS aetna_code, 'Infectious Disease Focus' AS aetna_description),
  STRUCT('Infectious Diseases' AS cms_specialty, '2III' AS aetna_code, 'Infectious Disease' AS aetna_description),
  STRUCT('Infectious Diseases' AS cms_specialty, '2IIP' AS aetna_code, 'Infectious Diseases(Pediatric)' AS aetna_description),
  STRUCT('Infectious Diseases' AS cms_specialty, '10412' AS aetna_code, 'Pediatric Infectious Disease' AS aetna_description),
  STRUCT('Nephrology' AS cms_specialty, '10312' AS aetna_code, 'Nephrology' AS aetna_description),
  STRUCT('Nephrology' AS cms_specialty, '91217' AS aetna_code, 'Hemodialysis' AS aetna_description),
  STRUCT('Nephrology' AS cms_specialty, '91220' AS aetna_code, 'Kidney Transplant Program' AS aetna_description),
  STRUCT('Nephrology' AS cms_specialty, '2N' AS aetna_code, 'Nephrology' AS aetna_description),
  STRUCT('Nephrology' AS cms_specialty, 'DI' AS aetna_code, 'Dialysis Center' AS aetna_description),
  STRUCT('Nephrology' AS cms_specialty, '2NP' AS aetna_code, 'Nephrology (Pediatric)' AS aetna_description),
  STRUCT('Nephrology' AS cms_specialty, '2HD' AS aetna_code, 'Hemodialysis' AS aetna_description),
  STRUCT('Nephrology' AS cms_specialty, '10408' AS aetna_code, 'Pediatric Nephrology' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '10806' AS aetna_code, 'Neuromuscular Medicine Physica' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '91044' AS aetna_code, 'Botox injections Neurology' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '91149' AS aetna_code, 'Sleep Medicine-Neurology' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '2NE' AS aetna_code, 'Neurology' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '2PN' AS aetna_code, 'Neurology (Pediatric)' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '10334' AS aetna_code, 'Vascular Neurology' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '10422' AS aetna_code, 'Pediatric Neurology' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '11002' AS aetna_code, 'Neurology' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '11003' AS aetna_code, 'Neurology Child' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '11006' AS aetna_code, 'Neurology & Psychiatry' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '11008' AS aetna_code, 'Neurology Chemical' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '11009' AS aetna_code, 'Child Neurology' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '11014' AS aetna_code, 'Neurology/Psychiatry Hospice' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '11015' AS aetna_code, 'Sleep Medicine - Neurology' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '11016' AS aetna_code, 'Neurocritical Care' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '11102' AS aetna_code, 'Neuromuscular Medicine Psychia' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '11103' AS aetna_code, 'Epilepsy' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '91062' AS aetna_code, 'Epilepsy Surgery' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '91081' AS aetna_code, 'Movement Disorders' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '91082' AS aetna_code, 'Multiple Sclerosis' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '91084' AS aetna_code, 'Neuromuscular Medicine' AS aetna_description),
  STRUCT('Neurology' AS cms_specialty, '91086' AS aetna_code, 'Neurovascular Surgery' AS aetna_description),
  STRUCT('Neurosurgery' AS cms_specialty, '10803' AS aetna_code, 'Spinal Cord Injury Medicine' AS aetna_description),
  STRUCT('Neurosurgery' AS cms_specialty, '90347' AS aetna_code, 'Neurosurgery (Pediatric)' AS aetna_description),
  STRUCT('Neurosurgery' AS cms_specialty, '90348' AS aetna_code, 'Neurosurgery (Spine)' AS aetna_description),
  STRUCT('Neurosurgery' AS cms_specialty, '91118' AS aetna_code, 'Spinal Cord Stimulation' AS aetna_description),
  STRUCT('Neurosurgery' AS cms_specialty, '91119' AS aetna_code, 'Stereotactic & Functional Neur' AS aetna_description),
  STRUCT('Neurosurgery' AS cms_specialty, '2NS' AS aetna_code, 'Neurosurgery' AS aetna_description),
  STRUCT('Neurosurgery' AS cms_specialty, '2NSP' AS aetna_code, 'Neurosurgery (Pediatric)' AS aetna_description),
  STRUCT('Neurosurgery' AS cms_specialty, '2NSS' AS aetna_code, 'Neurosurgery (Spine)' AS aetna_description),
  STRUCT('Oncology Medical/Surgical' AS cms_specialty, '10311' AS aetna_code, 'Oncology Medical' AS aetna_description),
  STRUCT('Oncology Medical/Surgical' AS cms_specialty, '10315' AS aetna_code, 'Hematology/Oncology' AS aetna_description),
  STRUCT('Oncology Medical/Surgical' AS cms_specialty, '20103' AS aetna_code, 'Oncology Gynecologic' AS aetna_description),
  STRUCT('Oncology Medical/Surgical' AS cms_specialty, '30810' AS aetna_code, 'Surgery Oncology' AS aetna_description),
  STRUCT('Oncology Medical/Surgical' AS cms_specialty, '90372' AS aetna_code, 'Radiation Oncology (Pediatric)' AS aetna_description),
  STRUCT('Oncology Medical/Surgical' AS cms_specialty, '91126' AS aetna_code, 'Urologic Oncology' AS aetna_description),
  STRUCT('Oncology Medical/Surgical' AS cms_specialty, '91129' AS aetna_code, 'Surgery Carcinoid' AS aetna_description),
  STRUCT('Oncology Medical/Surgical' AS cms_specialty, '2ROP' AS aetna_code, 'Radiation Oncology (Pediatric)' AS aetna_description),
  STRUCT('Oncology Medical/Surgical' AS cms_specialty, '91085' AS aetna_code, 'Neuro-Oncology' AS aetna_description),
  STRUCT('Oncology Radiation' AS cms_specialty, '40303' AS aetna_code, 'Radiation Oncology' AS aetna_description),
  STRUCT('Oncology Radiation' AS cms_specialty, '40304' AS aetna_code, 'Radiological Physics' AS aetna_description),
  STRUCT('Oncology Radiation' AS cms_specialty, '40310' AS aetna_code, 'Therapeutic Radiology' AS aetna_description),
  STRUCT('Oncology Radiation' AS cms_specialty, '40316' AS aetna_code, 'Radiation Therapy' AS aetna_description),
  STRUCT('Oncology Radiation' AS cms_specialty, '40318' AS aetna_code, 'Radium Therapy' AS aetna_description),
  STRUCT('Oncology Radiation' AS cms_specialty, '90372' AS aetna_code, 'Radiation Oncology (Pediatric)' AS aetna_description),
  STRUCT('Oncology Radiation' AS cms_specialty, '2RO' AS aetna_code, 'Radiation Therapy' AS aetna_description),
  STRUCT('Oncology Radiation' AS cms_specialty, '2ROP' AS aetna_code, 'Radiation Oncology (Pediatric)' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '30401' AS aetna_code, 'Opthalmology' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '30402' AS aetna_code, 'Retinal Opthalmology' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '30403' AS aetna_code, 'Sleep Medicine-Ophthalmology/O' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '90089' AS aetna_code, 'Retinal Specialist' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '90311' AS aetna_code, 'Anterior Segment (Glaucoma)' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '90315' AS aetna_code, 'Corneal Specialist' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '90343' AS aetna_code, 'Neuro-Ophthalmology' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '90356' AS aetna_code, 'Oculoplastic Surgery' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '91155' AS aetna_code, 'Pediatric Ophthalmology' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '2PEO' AS aetna_code, 'Ophthalmology (Pediatric)' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '2O' AS aetna_code, 'Ophthalmology' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '2OAG' AS aetna_code, 'Anterior Segment (Glaucoma)' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '2OC' AS aetna_code, 'Corneal Specialist' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '2PSOC' AS aetna_code, 'Oculoplastic Surgery' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '2RS' AS aetna_code, 'Retinal Specialist' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '10414' AS aetna_code, 'Pediatric Opthalmology' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '91070' AS aetna_code, 'Glaucoma Service' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '91087' AS aetna_code, 'Ophthamologic Cancer' AS aetna_description),
  STRUCT('Ophthalmology' AS cms_specialty, '91088' AS aetna_code, 'Orbital Surgery' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '10317' AS aetna_code, 'Oncology Orthopedic' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '30501' AS aetna_code, 'Surgery Orthopedic' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '30502' AS aetna_code, 'Surgery Hand/Orthopedic' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '30503' AS aetna_code, 'Surgery Knee' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '90361' AS aetna_code, 'Orthopedics (Foot & Ankle)' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '90362' AS aetna_code, 'Orthopedics (Joint Replacement' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '90365' AS aetna_code, 'Orthopedics Surgery (Spine)' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '91101' AS aetna_code, 'Pediatric Orthopedic Oncology' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '2OR' AS aetna_code, 'Orthopedics' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '2ORFA' AS aetna_code, 'Orthopedics (Foot & Ankle)' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '2ORON' AS aetna_code, 'Orthopedics (Oncology)' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '2ORR' AS aetna_code, 'Orthopedics (Joint Replacement' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '2ORS' AS aetna_code, 'Orthopedics Surgery (Spine)' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '2ORSM' AS aetna_code, 'Orthopedics (Sports Medicine)' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '2POR' AS aetna_code, 'Orthopedics (Pediatric)' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '10418' AS aetna_code, 'Pediatric Orthopedic' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '91092' AS aetna_code, 'Orthopedic Elbow Replacement' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '91093' AS aetna_code, 'Orthopedic Trauma' AS aetna_description),
  STRUCT('Orthopedic Surgery' AS cms_specialty, '91094' AS aetna_code, 'Orthopedic Shoulder' AS aetna_description),
  STRUCT('Physiatry Rehabilitative Med' AS cms_specialty, '10801' AS aetna_code, 'Physical Medicine & Rehabilita' AS aetna_description),
  STRUCT('Physiatry Rehabilitative Med' AS cms_specialty, '10802' AS aetna_code, 'Rehabilitation Medicine' AS aetna_description),
  STRUCT('Physiatry Rehabilitative Med' AS cms_specialty, '10805' AS aetna_code, 'Physical Medicine Hospice and' AS aetna_description),
  STRUCT('Physiatry Rehabilitative Med' AS cms_specialty, '10807' AS aetna_code, 'Pediatric Physical Medicine an' AS aetna_description),
  STRUCT('Physiatry Rehabilitative Med' AS cms_specialty, '2PM' AS aetna_code, 'Physical Medicine' AS aetna_description),
  STRUCT('Physiatry Rehabilitative Med' AS cms_specialty, '2RM' AS aetna_code, 'Rehab Medicine' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '10428' AS aetna_code, 'Pediatric Plastic Surgery' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '30602' AS aetna_code, 'Surgery Oro-Facial Plastic' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '90308' AS aetna_code, 'Facial Plastic and Reconstruct' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '90316' AS aetna_code, 'Craniofacial Surgery' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '90317' AS aetna_code, 'Craniofacial Surgery (Pediatri' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '90356' AS aetna_code, 'Oculoplastic Surgery' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '91111' AS aetna_code, 'Reconstructive Breast Surgery' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '91112' AS aetna_code, 'Reconstructive Breast Surgery' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '91113' AS aetna_code, 'Reconstructive Breast Surgery' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '2PS' AS aetna_code, 'Plastic Surgery' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '2PSCF' AS aetna_code, 'Craniofacial Surgery' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '2PSCP' AS aetna_code, 'Craniofacial Surgery (Pediatri' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '2PSOC' AS aetna_code, 'Oculoplastic Surgery' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '2PSP' AS aetna_code, 'Plastic Surgery (Pediatric)' AS aetna_description),
  STRUCT('Plastic Surgery' AS cms_specialty, '91054' AS aetna_code, 'Craniofacial Plastics' AS aetna_description),
  STRUCT('Podiatry' AS cms_specialty, '91213' AS aetna_code, 'Foot and Ankle Surgery' AS aetna_description),
  STRUCT('Podiatry' AS cms_specialty, '91214' AS aetna_code, 'Foot Surgery' AS aetna_description),
  STRUCT('Podiatry' AS cms_specialty, 'DP' AS aetna_code, 'Podiatrist' AS aetna_description),
  STRUCT('Podiatry' AS cms_specialty, '2PO' AS aetna_code, 'Podiatry' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '91244' AS aetna_code, 'Psychiatry Autism Spectrum' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '91245' AS aetna_code, 'Psychiatry Child & Adolescent' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '91246' AS aetna_code, 'Psychiatry Child & Adolescent' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '91247' AS aetna_code, 'Psychiatry Child & Adolestcent' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '91249' AS aetna_code, 'Psychiatry Child & Adolescent' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '91250' AS aetna_code, 'Psychiatry Home Based Services' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '91252' AS aetna_code, 'Psychiatry Trauma/Crisis' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '2PPY' AS aetna_code, 'Psychiatry (Pediatric)' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '2PY' AS aetna_code, 'Psychiatry' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '2PYGR' AS aetna_code, 'Geriatric Psychiatry' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '11001' AS aetna_code, 'Psychiatry' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '11004' AS aetna_code, 'Psychiatry Child & Adolescent' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '11005' AS aetna_code, 'Psychiatry Geriatric' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '11006' AS aetna_code, 'Neurology & Psychiatry' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '11007' AS aetna_code, 'Addictionology' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '11010' AS aetna_code, 'Child Psychiatry' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '11011' AS aetna_code, 'Addiction Psychiatry' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '11013' AS aetna_code, 'Forensic Medicine' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '11014' AS aetna_code, 'Neurology/Psychiatry Hospice' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '11015' AS aetna_code, 'Sleep Medicine - Neurology' AS aetna_description),
  STRUCT('Psychiatry' AS cms_specialty, '11101' AS aetna_code, 'Psychomatic Medicine' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '10304' AS aetna_code, 'Critical Care Medicine' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '10313' AS aetna_code, 'Pulmonary Disease' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '10318' AS aetna_code, 'Medical Diseases of Chest' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '20102' AS aetna_code, 'Critical Care Medicine/Obstetr' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '30102' AS aetna_code, 'Critical Care Medicine/Anesthe' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '30302' AS aetna_code, 'Critical Care Medicine Neurolo' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '91139' AS aetna_code, 'Sleep Medicine - Pulmonology' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '2CCM' AS aetna_code, 'Critical Care Medicine' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '2CCMP' AS aetna_code, 'Critical Care Medicine (Pediat' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '2PD' AS aetna_code, 'Pulmonary Disease' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '2PPD' AS aetna_code, 'Pulmonary Diseases (Pediatric)' AS aetna_description),
  STRUCT('Pulmonology' AS cms_specialty, '10409' AS aetna_code, 'Pediatric Pulmonology' AS aetna_description),
  STRUCT('Rheumatology' AS cms_specialty, '10314' AS aetna_code, 'Rheumatology' AS aetna_description),
  STRUCT('Rheumatology' AS cms_specialty, '91041' AS aetna_code, 'Arthritis Reconstruction' AS aetna_description),
  STRUCT('Rheumatology' AS cms_specialty, '2RH' AS aetna_code, 'Rheumatology' AS aetna_description),
  STRUCT('Rheumatology' AS cms_specialty, '2RHP' AS aetna_code, 'Rheumatology (Pediatric)' AS aetna_description),
  STRUCT('Rheumatology' AS cms_specialty, '10420' AS aetna_code, 'Pediatric Rheumatology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '30301' AS aetna_code, 'Surgery Neurological' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '30808' AS aetna_code, 'Surgery Urological' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '31001' AS aetna_code, 'Urology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '90379' AS aetna_code, 'Urology (Male Infertility)' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '90398' AS aetna_code, 'Uro-Gynecology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '91044' AS aetna_code, 'Botox injections Neurology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '91102' AS aetna_code, 'Pediatric Uro-Gynecology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '91126' AS aetna_code, 'Urologic Oncology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '91127' AS aetna_code, 'UROLOGICTR' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '91149' AS aetna_code, 'Sleep Medicine-Neurology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '91220' AS aetna_code, 'Kidney Transplant Program' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '2NE' AS aetna_code, 'Neurology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '2PN' AS aetna_code, 'Neurology (Pediatric)' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '2PU' AS aetna_code, 'Urology (Pediatric)' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '2U' AS aetna_code, 'Urology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '2UGY' AS aetna_code, 'Uro-gynecology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '2UMI' AS aetna_code, 'Urology (Male Infertility)' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '10334' AS aetna_code, 'Vascular Neurology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '10415' AS aetna_code, 'Pediatric Urology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '10422' AS aetna_code, 'Pediatric Neurology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '11002' AS aetna_code, 'Neurology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '11003' AS aetna_code, 'Neurology Child' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '11006' AS aetna_code, 'Neurology & Psychiatry' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '11008' AS aetna_code, 'Neurology Chemical' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '11009' AS aetna_code, 'Child Neurology' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '11014' AS aetna_code, 'Neurology/Psychiatry Hospice' AS aetna_description),
  STRUCT('Urology' AS cms_specialty, '11015' AS aetna_code, 'Sleep Medicine - Neurology' AS aetna_description),
  STRUCT('Vascular Surgery' AS cms_specialty, '40317' AS aetna_code, 'Vascular & Interventional Radi' AS aetna_description),
  STRUCT('Vascular Surgery' AS cms_specialty, '40319' AS aetna_code, 'Angiography and Interventional' AS aetna_description),
  STRUCT('Vascular Surgery' AS cms_specialty, '90071' AS aetna_code, 'Peripheral Vascular Disease' AS aetna_description),
  STRUCT('Vascular Surgery' AS cms_specialty, '2IY' AS aetna_code, 'Peripheral Vascular Disease' AS aetna_description),
  STRUCT('Vascular Surgery' AS cms_specialty, '2VS' AS aetna_code, 'Vascular Surgery' AS aetna_description),
  STRUCT('Vascular Surgery' AS cms_specialty, '10334' AS aetna_code, 'Vascular Neurology' AS aetna_description),
  STRUCT('Vascular Surgery' AS cms_specialty, '91086' AS aetna_code, 'Neurovascular Surgery' AS aetna_description),
  STRUCT('Cardiothoracic Surgery' AS cms_specialty, '30805' AS aetna_code, 'Surgery Thoracic Cardiovascul' AS aetna_description),
  STRUCT('Cardiothoracic Surgery' AS cms_specialty, '30812' AS aetna_code, 'Surgery Congenital Cardiac/Th' AS aetna_description),
  STRUCT('Cardiothoracic Surgery' AS cms_specialty, '30901' AS aetna_code, 'Surgery Thoracic' AS aetna_description),
  STRUCT('Cardiothoracic Surgery' AS cms_specialty, '91206' AS aetna_code, 'Cardiac Surgery Program' AS aetna_description),
  STRUCT('Cardiothoracic Surgery' AS cms_specialty, '91215' AS aetna_code, 'Heart Transplant Program' AS aetna_description),
  STRUCT('Cardiothoracic Surgery' AS cms_specialty, '2CS' AS aetna_code, 'Cardiothoracic/Cardiovascular' AS aetna_description),
  STRUCT('Cardiothoracic Surgery' AS cms_specialty, '2TS' AS aetna_code, 'Thoracic Surgery' AS aetna_description),
  STRUCT('Cardiothoracic Surgery' AS cms_specialty, '10426' AS aetna_code, 'Pediatric Thoracic Surgery' AS aetna_description),
  STRUCT('Acute Inpatient Hospitals' AS cms_specialty, '2OT' AS aetna_code, 'Hospital Outpatient' AS aetna_description),
  STRUCT('Acute Inpatient Hospitals' AS cms_specialty, 'CH' AS aetna_code, 'Children\'s Hospital' AS aetna_description),
  STRUCT('Acute Inpatient Hospitals' AS cms_specialty, 'HO' AS aetna_code, 'Acute Short Term Hospital' AS aetna_description),
  STRUCT('Acute Inpatient Hospitals' AS cms_specialty, 'HSLT' AS aetna_code, 'Hospitalist' AS aetna_description),
  STRUCT('Acute Inpatient Hospitals' AS cms_specialty, 'LHO' AS aetna_code, 'Long Term Acute Care Hospital' AS aetna_description),
  STRUCT('Acute Inpatient Hospitals' AS cms_specialty, '2HSLT' AS aetna_code, 'Hospitalist' AS aetna_description),
  STRUCT('Acute Inpatient Hospitals' AS cms_specialty, '2SH' AS aetna_code, 'Specialty Hospital' AS aetna_description),
  STRUCT('Acute Inpatient Hospitals' AS cms_specialty, '91002' AS aetna_code, 'Hospitalist' AS aetna_description),
  STRUCT('Cardiac Surgery Program' AS cms_specialty, '91046' AS aetna_code, 'Cardiac Valve Replacement' AS aetna_description),
  STRUCT('Cardiac Surgery Program' AS cms_specialty, '91205' AS aetna_code, 'Cardiac Monitoring Service' AS aetna_description),
  STRUCT('Cardiac Surgery Program' AS cms_specialty, '91206' AS aetna_code, 'Cardiac Surgery Program' AS aetna_description),
  STRUCT('Cardiac Surgery Program' AS cms_specialty, '2CS' AS aetna_code, 'Cardiothoracic/Cardiovascular' AS aetna_description),
  STRUCT('Cardiac Catheterization' AS cms_specialty, '10302' AS aetna_code, 'Cardiac Electrophysiology' AS aetna_description),
  STRUCT('Cardiac Catheterization' AS cms_specialty, '91205' AS aetna_code, 'Cardiac Monitoring Service' AS aetna_description),
  STRUCT('Cardiac Catheterization' AS cms_specialty, '2CEP' AS aetna_code, 'Cardiac Electrophysiology' AS aetna_description),
  STRUCT('Cardiac Catheterization' AS cms_specialty, '10332' AS aetna_code, 'Interventional Cardiology' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '10304' AS aetna_code, 'Critical Care Medicine' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '10432' AS aetna_code, 'Pediatric Intensive Care' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '20102' AS aetna_code, 'Critical Care Medicine/Obstetr' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '30102' AS aetna_code, 'Critical Care Medicine/Anesthe' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '30302' AS aetna_code, 'Critical Care Medicine Neurolo' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '30803' AS aetna_code, 'Surgery Critical care' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '91125' AS aetna_code, 'Trauma Surgical Critical Care' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '91165' AS aetna_code, 'Intensive Care Coordination' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '2CCM' AS aetna_code, 'Critical Care Medicine' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '2CCMP' AS aetna_code, 'Critical Care Medicine (Pediat' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '10404' AS aetna_code, 'Pediatric Critical Care' AS aetna_description),
  STRUCT('Critical Care ICU' AS cms_specialty, '11016' AS aetna_code, 'Neurocritical Care' AS aetna_description),
  STRUCT('Surgical Services ASC' AS cms_specialty, '91235' AS aetna_code, 'Outpatient Surgery' AS aetna_description),
  STRUCT('Surgical Services ASC' AS cms_specialty, 'AC' AS aetna_code, 'Ambulatory Surgicenter' AS aetna_description),
  STRUCT('Surgical Services ASC' AS cms_specialty, 'FEC' AS aetna_code, 'Freestanding Emergency Center' AS aetna_description),
  STRUCT('Surgical Services ASC' AS cms_specialty, '2FS' AS aetna_code, 'Free Standing Surgical Unit' AS aetna_description),
  STRUCT('Skilled Nursing Facility' AS cms_specialty, '91287' AS aetna_code, 'Assisted Living Center' AS aetna_description),
  STRUCT('Skilled Nursing Facility' AS cms_specialty, '91294' AS aetna_code, 'Skilled Nursing Facilities' AS aetna_description),
  STRUCT('Skilled Nursing Facility' AS cms_specialty, '91301' AS aetna_code, 'Nursing Facility Transition Di' AS aetna_description),
  STRUCT('Skilled Nursing Facility' AS cms_specialty, '91302' AS aetna_code, 'Recuperative Care' AS aetna_description),
  STRUCT('Skilled Nursing Facility' AS cms_specialty, 'ALC' AS aetna_code, 'Assisted Living Center' AS aetna_description),
  STRUCT('Skilled Nursing Facility' AS cms_specialty, 'LSS' AS aetna_code, 'Long-Term Services and Support' AS aetna_description),
  STRUCT('Skilled Nursing Facility' AS cms_specialty, 'SK' AS aetna_code, 'Skilled Nursing Facility' AS aetna_description),
  STRUCT('Skilled Nursing Facility' AS cms_specialty, '2SNF' AS aetna_code, 'Skilled Nursing Facility' AS aetna_description),
  STRUCT('Diagnostic Radiology' AS cms_specialty, '40306' AS aetna_code, 'Diagnostic Roentgenology' AS aetna_description),
  STRUCT('Diagnostic Radiology' AS cms_specialty, '40311' AS aetna_code, 'Neuroradiology' AS aetna_description),
  STRUCT('Diagnostic Radiology' AS cms_specialty, '40313' AS aetna_code, 'Nuclear Imaging and Therapy' AS aetna_description),
  STRUCT('Diagnostic Radiology' AS cms_specialty, '40315' AS aetna_code, 'Diagnostic Ultrasound' AS aetna_description),
  STRUCT('Diagnostic Radiology' AS cms_specialty, '40320' AS aetna_code, 'Body Imaging' AS aetna_description),
  STRUCT('Diagnostic Radiology' AS cms_specialty, '91224' AS aetna_code, 'Medical Imaging' AS aetna_description),
  STRUCT('Diagnostic Radiology' AS cms_specialty, 'RFA' AS aetna_code, 'Radiology Center' AS aetna_description),
  STRUCT('Mammography' AS cms_specialty, '91223' AS aetna_code, 'Mammography' AS aetna_description),
  STRUCT('Physical Therapy' AS cms_specialty, '90331' AS aetna_code, 'Hand Rehabilitation' AS aetna_description),
  STRUCT('Physical Therapy' AS cms_specialty, '90375' AS aetna_code, 'Physical Therapy (Pediatric)' AS aetna_description),
  STRUCT('Physical Therapy' AS cms_specialty, '91141' AS aetna_code, 'Physical Therapy' AS aetna_description),
  STRUCT('Physical Therapy' AS cms_specialty, '2HR' AS aetna_code, 'Hand Rehabilitation' AS aetna_description),
  STRUCT('Physical Therapy' AS cms_specialty, '2PT' AS aetna_code, 'Physical Therapy' AS aetna_description),
  STRUCT('Occupational Therapy' AS cms_specialty, '90374' AS aetna_code, 'Occupational Therapy (Pediatri' AS aetna_description),
  STRUCT('Occupational Therapy' AS cms_specialty, '91142' AS aetna_code, 'Occupational Therapy' AS aetna_description),
  STRUCT('Occupational Therapy' AS cms_specialty, '2TO' AS aetna_code, 'Occupational Therapy' AS aetna_description),
  STRUCT('Speech Therapy' AS cms_specialty, '90373' AS aetna_code, 'Speech Therapy (Pediatric)' AS aetna_description),
  STRUCT('Speech Therapy' AS cms_specialty, '91143' AS aetna_code, 'Speech Therapy' AS aetna_description),
  STRUCT('Speech Therapy' AS cms_specialty, '91257' AS aetna_code, 'Speech/Hearing' AS aetna_description),
  STRUCT('Speech Therapy' AS cms_specialty, '91258' AS aetna_code, 'Speech/Hearing Therapy' AS aetna_description),
  STRUCT('Speech Therapy' AS cms_specialty, '91259' AS aetna_code, 'Speech/Language/Hearing Therap' AS aetna_description),
  STRUCT('Speech Therapy' AS cms_specialty, 'SH' AS aetna_code, 'Speech Pathologist' AS aetna_description),
  STRUCT('Speech Therapy' AS cms_specialty, 'ST' AS aetna_code, 'Speech Therapist' AS aetna_description),
  STRUCT('Speech Therapy' AS cms_specialty, '2TT' AS aetna_code, 'Speech Therapy' AS aetna_description),
  STRUCT('Inpatient Psychiatric' AS cms_specialty, 'RTF' AS aetna_code, 'Residential Treatment Facility' AS aetna_description),
  STRUCT('Inpatient Psychiatric' AS cms_specialty, '2PLMD' AS aetna_code, 'Palliative Medicine' AS aetna_description),
  STRUCT('Inpatient Psychiatric' AS cms_specialty, '91001' AS aetna_code, 'Palliative Medicine' AS aetna_description),
  STRUCT('Inpatient Psychiatric' AS cms_specialty, '91003' AS aetna_code, 'Psychotic Disorders' AS aetna_description),
  STRUCT('Outpatient Infusion/Chemo' AS cms_specialty, '91180' AS aetna_code, 'Antibiotic Infusion' AS aetna_description),
  STRUCT('Outpatient Infusion/Chemo' AS cms_specialty, '91218' AS aetna_code, 'Home Infusion Therapy for HIV' AS aetna_description),
  STRUCT('Outpatient Infusion/Chemo' AS cms_specialty, '91234' AS aetna_code, 'Outpatient Infusion/Chemothera' AS aetna_description),
  STRUCT('Outpatient Infusion/Chemo' AS cms_specialty, 'HI' AS aetna_code, 'Home Infusion' AS aetna_description),
  STRUCT('Outpatient Infusion/Chemo' AS cms_specialty, 'IC' AS aetna_code, 'Infusion Center' AS aetna_description),
  STRUCT('Outpatient Infusion/Chemo' AS cms_specialty, '2IC' AS aetna_code, 'Infusion Center' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '10204' AS aetna_code, 'Addiction Medicine' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '90001' AS aetna_code, 'Addictions Counselor' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '91032' AS aetna_code, 'Applied Behavioral Analysis' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '91134' AS aetna_code, 'Behavioral Health Rehabilitati' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '91174' AS aetna_code, 'Mobile Crisis Intervention (MC' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '91175' AS aetna_code, 'Behavioral Health Services Tel' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '91278' AS aetna_code, 'Applied Behavioral Analysis (A' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '2AC' AS aetna_code, 'Addictions Counselor' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '2MH' AS aetna_code, 'Mental Health-Substance Abuse' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, 'ABA' AS aetna_code, 'Applied Behavioral Analysis' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, 'BHR' AS aetna_code, 'Behavioral Health Rehabilitati' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, 'CAC' AS aetna_code, 'Certified Addictions Counselor' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, 'CMC' AS aetna_code, 'Community Mental Health Center' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, 'MH' AS aetna_code, 'Mental Health - Substance Abus' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, 'SA' AS aetna_code, 'Substance Abuse Facility' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '11007' AS aetna_code, 'Addictionology' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '11011' AS aetna_code, 'Addiction Psychiatry' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '90428' AS aetna_code, 'Mental Health' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '91005' AS aetna_code, 'Dialectic Behavioral Therapy' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '91006' AS aetna_code, 'Cognitive Behavioral Therapy' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '91011' AS aetna_code, 'Substance Abuse Professional' AS aetna_description),
  STRUCT('Outpatient Behavioral Health' AS cms_specialty, '91012' AS aetna_code, 'Crisis Intervention' AS aetna_description)
]);-- ============================================================
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
  -- Primary Care (min_ratio: LM/M=1.67, Micro/Rural/CEAC=1.42)
  STRUCT('Primary Care' AS cms_specialty, 'Large Metro' AS county_type, 10 AS max_time_min, 5   AS max_distance_miles, 1.67 AS min_ratio_per_1000),
  STRUCT('Primary Care', 'Metro',  15,  10, 1.67),
  STRUCT('Primary Care', 'Micro',  30,  20, 1.42),
  STRUCT('Primary Care', 'Rural',  40,  30, 1.42),
  STRUCT('Primary Care', 'CEAC',   70,  60, 1.42),
  -- Allergy and Immunology (min_ratio: LM/M=0.05, Micro/Rural/CEAC=0.04)
  STRUCT('Allergy and Immunology', 'Large Metro',  30,  15, 0.05),
  STRUCT('Allergy and Immunology', 'Metro',        45,  30, 0.05),
  STRUCT('Allergy and Immunology', 'Micro',        80,  60, 0.04),
  STRUCT('Allergy and Immunology', 'Rural',        90,  75, 0.04),
  STRUCT('Allergy and Immunology', 'CEAC',        125, 110, 0.04),
  -- Cardiology (min_ratio: LM/M=0.27, Micro/Rural/CEAC=0.23)
  STRUCT('Cardiology', 'Large Metro', 20, 10, 0.27),
  STRUCT('Cardiology', 'Metro',       30, 20, 0.27),
  STRUCT('Cardiology', 'Micro',       50, 35, 0.23),
  STRUCT('Cardiology', 'Rural',       75, 60, 0.23),
  STRUCT('Cardiology', 'CEAC',        95, 85, 0.23),
  -- Chiropractor (min_ratio: LM/M=0.10, Micro/Rural/CEAC=0.09)
  STRUCT('Chiropractor', 'Large Metro',  30,  15, 0.10),
  STRUCT('Chiropractor', 'Metro',        45,  30, 0.10),
  STRUCT('Chiropractor', 'Micro',        80,  60, 0.09),
  STRUCT('Chiropractor', 'Rural',        90,  75, 0.09),
  STRUCT('Chiropractor', 'CEAC',        125, 110, 0.09),
  -- Clinical Psychology (min_ratio: LM/M=0.15, Micro/Rural/CEAC=0.13)
  STRUCT('Clinical Psychology', 'Large Metro',  20,  10, 0.15),
  STRUCT('Clinical Psychology', 'Metro',        45,  30, 0.15),
  STRUCT('Clinical Psychology', 'Micro',        60,  45, 0.13),
  STRUCT('Clinical Psychology', 'Rural',        75,  60, 0.13),
  STRUCT('Clinical Psychology', 'CEAC',        145, 130, 0.13),
  -- Clinical Social Work (min_ratio: LM/M=0.25, Micro/Rural/CEAC=0.22)
  STRUCT('Clinical Social Work', 'Large Metro',  20,  10, 0.25),
  STRUCT('Clinical Social Work', 'Metro',        30,  20, 0.25),
  STRUCT('Clinical Social Work', 'Micro',        50,  35, 0.22),
  STRUCT('Clinical Social Work', 'Rural',        75,  60, 0.22),
  STRUCT('Clinical Social Work', 'CEAC',        125, 110, 0.22),
  -- Dermatology (min_ratio: LM/M=0.16, Micro/Rural/CEAC=0.14)
  STRUCT('Dermatology', 'Large Metro',  20,  10, 0.16),
  STRUCT('Dermatology', 'Metro',        45,  30, 0.16),
  STRUCT('Dermatology', 'Micro',        60,  45, 0.14),
  STRUCT('Dermatology', 'Rural',        75,  60, 0.14),
  STRUCT('Dermatology', 'CEAC',        110, 100, 0.14),
  -- Endocrinology (min_ratio: LM/M=0.04, Micro/Rural/CEAC=0.03)
  STRUCT('Endocrinology', 'Large Metro',  30,  15, 0.04),
  STRUCT('Endocrinology', 'Metro',        60,  40, 0.04),
  STRUCT('Endocrinology', 'Micro',       100,  75, 0.03),
  STRUCT('Endocrinology', 'Rural',       110,  90, 0.03),
  STRUCT('Endocrinology', 'CEAC',        145, 130, 0.03),
  -- ENT/Otolaryngology (min_ratio: LM/M=0.06, Micro/Rural/CEAC=0.05)
  STRUCT('ENT/Otolaryngology', 'Large Metro',  30,  15, 0.06),
  STRUCT('ENT/Otolaryngology', 'Metro',        45,  30, 0.06),
  STRUCT('ENT/Otolaryngology', 'Micro',        80,  60, 0.05),
  STRUCT('ENT/Otolaryngology', 'Rural',        90,  75, 0.05),
  STRUCT('ENT/Otolaryngology', 'CEAC',        125, 110, 0.05),
  -- Gastroenterology (min_ratio: LM/M=0.12, Micro/Rural/CEAC=0.10)
  STRUCT('Gastroenterology', 'Large Metro',  20,  10, 0.12),
  STRUCT('Gastroenterology', 'Metro',        45,  30, 0.12),
  STRUCT('Gastroenterology', 'Micro',        60,  45, 0.10),
  STRUCT('Gastroenterology', 'Rural',        75,  60, 0.10),
  STRUCT('Gastroenterology', 'CEAC',        110, 100, 0.10),
  -- General Surgery (min_ratio: LM/M=0.28, Micro/Rural/CEAC=0.24)
  STRUCT('General Surgery', 'Large Metro', 20, 10, 0.28),
  STRUCT('General Surgery', 'Metro',       30, 20, 0.28),
  STRUCT('General Surgery', 'Micro',       50, 35, 0.24),
  STRUCT('General Surgery', 'Rural',       75, 60, 0.24),
  STRUCT('General Surgery', 'CEAC',        95, 85, 0.24),
  -- Gynecology OB/GYN (min_ratio: LM/M=0.04, Micro/Rural/CEAC=0.03)
  STRUCT('Gynecology OB/GYN', 'Large Metro',  30,  15, 0.04),
  STRUCT('Gynecology OB/GYN', 'Metro',        45,  30, 0.04),
  STRUCT('Gynecology OB/GYN', 'Micro',        80,  60, 0.03),
  STRUCT('Gynecology OB/GYN', 'Rural',        90,  75, 0.03),
  STRUCT('Gynecology OB/GYN', 'CEAC',        125, 110, 0.03),
  -- Infectious Diseases (min_ratio: LM/M=0.03, Micro/Rural/CEAC=0.03)
  STRUCT('Infectious Diseases', 'Large Metro',  30,  15, 0.03),
  STRUCT('Infectious Diseases', 'Metro',        60,  40, 0.03),
  STRUCT('Infectious Diseases', 'Micro',       100,  75, 0.03),
  STRUCT('Infectious Diseases', 'Rural',       110,  90, 0.03),
  STRUCT('Infectious Diseases', 'CEAC',        145, 130, 0.03),
  -- Nephrology (min_ratio: LM/M=0.09, Micro/Rural/CEAC=0.08)
  STRUCT('Nephrology', 'Large Metro',  30,  15, 0.09),
  STRUCT('Nephrology', 'Metro',        45,  30, 0.09),
  STRUCT('Nephrology', 'Micro',        80,  60, 0.08),
  STRUCT('Nephrology', 'Rural',        90,  75, 0.08),
  STRUCT('Nephrology', 'CEAC',        125, 110, 0.08),
  -- Neurology (min_ratio: LM/M=0.12, Micro/Rural/CEAC=0.10)
  STRUCT('Neurology', 'Large Metro',  20,  10, 0.12),
  STRUCT('Neurology', 'Metro',        45,  30, 0.12),
  STRUCT('Neurology', 'Micro',        60,  45, 0.10),
  STRUCT('Neurology', 'Rural',        75,  60, 0.10),
  STRUCT('Neurology', 'CEAC',        110, 100, 0.10),
  -- Neurosurgery (min_ratio: all=0.01)
  STRUCT('Neurosurgery', 'Large Metro',  30,  15, 0.01),
  STRUCT('Neurosurgery', 'Metro',        60,  40, 0.01),
  STRUCT('Neurosurgery', 'Micro',       100,  75, 0.01),
  STRUCT('Neurosurgery', 'Rural',       110,  90, 0.01),
  STRUCT('Neurosurgery', 'CEAC',        145, 130, 0.01),
  -- Oncology Medical/Surgical (min_ratio: LM/M=0.19, Micro/Rural/CEAC=0.16)
  STRUCT('Oncology Medical/Surgical', 'Large Metro',  20,  10, 0.19),
  STRUCT('Oncology Medical/Surgical', 'Metro',        45,  30, 0.19),
  STRUCT('Oncology Medical/Surgical', 'Micro',        60,  45, 0.16),
  STRUCT('Oncology Medical/Surgical', 'Rural',        75,  60, 0.16),
  STRUCT('Oncology Medical/Surgical', 'CEAC',        110, 100, 0.16),
  -- Oncology Radiation (min_ratio: LM/M=0.06, Micro/Rural/CEAC=0.05)
  STRUCT('Oncology Radiation', 'Large Metro',  30,  15, 0.06),
  STRUCT('Oncology Radiation', 'Metro',        60,  40, 0.06),
  STRUCT('Oncology Radiation', 'Micro',       100,  75, 0.05),
  STRUCT('Oncology Radiation', 'Rural',       110,  90, 0.05),
  STRUCT('Oncology Radiation', 'CEAC',        145, 130, 0.05),
  -- Ophthalmology (min_ratio: LM/M=0.24, Micro/Rural/CEAC=0.20)
  STRUCT('Ophthalmology', 'Large Metro', 20, 10, 0.24),
  STRUCT('Ophthalmology', 'Metro',       30, 20, 0.24),
  STRUCT('Ophthalmology', 'Micro',       50, 35, 0.20),
  STRUCT('Ophthalmology', 'Rural',       75, 60, 0.20),
  STRUCT('Ophthalmology', 'CEAC',        95, 85, 0.20),
  -- Orthopedic Surgery (min_ratio: LM/M=0.20, Micro/Rural/CEAC=0.17)
  STRUCT('Orthopedic Surgery', 'Large Metro', 20, 10, 0.20),
  STRUCT('Orthopedic Surgery', 'Metro',       30, 20, 0.20),
  STRUCT('Orthopedic Surgery', 'Micro',       50, 35, 0.17),
  STRUCT('Orthopedic Surgery', 'Rural',       75, 60, 0.17),
  STRUCT('Orthopedic Surgery', 'CEAC',        95, 85, 0.17),
  -- Physiatry Rehabilitative Med (min_ratio: LM/M=0.04, Micro/Rural/CEAC=0.03)
  STRUCT('Physiatry Rehabilitative Med', 'Large Metro',  30,  15, 0.04),
  STRUCT('Physiatry Rehabilitative Med', 'Metro',        45,  30, 0.04),
  STRUCT('Physiatry Rehabilitative Med', 'Micro',        80,  60, 0.03),
  STRUCT('Physiatry Rehabilitative Med', 'Rural',        90,  75, 0.03),
  STRUCT('Physiatry Rehabilitative Med', 'CEAC',        125, 110, 0.03),
  -- Plastic Surgery (min_ratio: all=0.01)
  STRUCT('Plastic Surgery', 'Large Metro',  30,  15, 0.01),
  STRUCT('Plastic Surgery', 'Metro',        60,  40, 0.01),
  STRUCT('Plastic Surgery', 'Micro',       100,  75, 0.01),
  STRUCT('Plastic Surgery', 'Rural',       110,  90, 0.01),
  STRUCT('Plastic Surgery', 'CEAC',        145, 130, 0.01),
  -- Podiatry (min_ratio: LM/M=0.19, Micro/Rural/CEAC=0.16)
  STRUCT('Podiatry', 'Large Metro',  20,  10, 0.19),
  STRUCT('Podiatry', 'Metro',        45,  30, 0.19),
  STRUCT('Podiatry', 'Micro',        60,  45, 0.16),
  STRUCT('Podiatry', 'Rural',        75,  60, 0.16),
  STRUCT('Podiatry', 'CEAC',        110, 100, 0.16),
  -- Psychiatry (min_ratio: LM/M=0.14, Micro/Rural/CEAC=0.12)
  STRUCT('Psychiatry', 'Large Metro',  20,  10, 0.14),
  STRUCT('Psychiatry', 'Metro',        45,  30, 0.14),
  STRUCT('Psychiatry', 'Micro',        60,  45, 0.12),
  STRUCT('Psychiatry', 'Rural',        75,  60, 0.12),
  STRUCT('Psychiatry', 'CEAC',        110, 100, 0.12),
  -- Pulmonology (min_ratio: LM/M=0.13, Micro/Rural/CEAC=0.11)
  STRUCT('Pulmonology', 'Large Metro',  20,  10, 0.13),
  STRUCT('Pulmonology', 'Metro',        45,  30, 0.13),
  STRUCT('Pulmonology', 'Micro',        60,  45, 0.11),
  STRUCT('Pulmonology', 'Rural',        75,  60, 0.11),
  STRUCT('Pulmonology', 'CEAC',        110, 100, 0.11),
  -- Rheumatology (min_ratio: LM/M=0.07, Micro/Rural/CEAC=0.06)
  STRUCT('Rheumatology', 'Large Metro',  30,  15, 0.07),
  STRUCT('Rheumatology', 'Metro',        60,  40, 0.07),
  STRUCT('Rheumatology', 'Micro',       100,  75, 0.06),
  STRUCT('Rheumatology', 'Rural',       110,  90, 0.06),
  STRUCT('Rheumatology', 'CEAC',        145, 130, 0.06),
  -- Urology (min_ratio: LM/M=0.12, Micro/Rural/CEAC=0.10)
  STRUCT('Urology', 'Large Metro',  20,  10, 0.12),
  STRUCT('Urology', 'Metro',        45,  30, 0.12),
  STRUCT('Urology', 'Micro',        60,  45, 0.10),
  STRUCT('Urology', 'Rural',        75,  60, 0.10),
  STRUCT('Urology', 'CEAC',        110, 100, 0.10),
  -- Vascular Surgery (min_ratio: all=0.02)
  STRUCT('Vascular Surgery', 'Large Metro',  30,  15, 0.02),
  STRUCT('Vascular Surgery', 'Metro',        60,  40, 0.02),
  STRUCT('Vascular Surgery', 'Micro',       100,  75, 0.02),
  STRUCT('Vascular Surgery', 'Rural',       110,  90, 0.02),
  STRUCT('Vascular Surgery', 'CEAC',        145, 130, 0.02),
  -- Cardiothoracic Surgery (min_ratio: all=0.01)
  STRUCT('Cardiothoracic Surgery', 'Large Metro',  30,  15, 0.01),
  STRUCT('Cardiothoracic Surgery', 'Metro',        60,  40, 0.01),
  STRUCT('Cardiothoracic Surgery', 'Micro',       100,  75, 0.01),
  STRUCT('Cardiothoracic Surgery', 'Rural',       110,  90, 0.01),
  STRUCT('Cardiothoracic Surgery', 'CEAC',        145, 130, 0.01),
  -- Acute Inpatient Hospitals (min_ratio: all=12.2 beds per 1,000)
  STRUCT('Acute Inpatient Hospitals', 'Large Metro', 20,  10, 12.2),
  STRUCT('Acute Inpatient Hospitals', 'Metro',       45,  30, 12.2),
  STRUCT('Acute Inpatient Hospitals', 'Micro',       80,  60, 12.2),
  STRUCT('Acute Inpatient Hospitals', 'Rural',       75,  60, 12.2),
  STRUCT('Acute Inpatient Hospitals', 'CEAC',       110, 100, 12.2),
  -- Cardiac Surgery Program (facility min=1, ratio=NULL)
  STRUCT('Cardiac Surgery Program', 'Large Metro',  30,  15, CAST(NULL AS FLOAT64)),
  STRUCT('Cardiac Surgery Program', 'Metro',        60,  40, CAST(NULL AS FLOAT64)),
  STRUCT('Cardiac Surgery Program', 'Micro',       160, 120, CAST(NULL AS FLOAT64)),
  STRUCT('Cardiac Surgery Program', 'Rural',       145, 120, CAST(NULL AS FLOAT64)),
  STRUCT('Cardiac Surgery Program', 'CEAC',        155, 140, CAST(NULL AS FLOAT64)),
  -- Cardiac Catheterization (facility min=1)
  STRUCT('Cardiac Catheterization', 'Large Metro',  30,  15, CAST(NULL AS FLOAT64)),
  STRUCT('Cardiac Catheterization', 'Metro',        60,  40, CAST(NULL AS FLOAT64)),
  STRUCT('Cardiac Catheterization', 'Micro',       160, 120, CAST(NULL AS FLOAT64)),
  STRUCT('Cardiac Catheterization', 'Rural',       145, 120, CAST(NULL AS FLOAT64)),
  STRUCT('Cardiac Catheterization', 'CEAC',        155, 140, CAST(NULL AS FLOAT64)),
  -- Critical Care ICU (facility min=1)
  STRUCT('Critical Care ICU', 'Large Metro',  20,  10, CAST(NULL AS FLOAT64)),
  STRUCT('Critical Care ICU', 'Metro',        45,  30, CAST(NULL AS FLOAT64)),
  STRUCT('Critical Care ICU', 'Micro',       160, 120, CAST(NULL AS FLOAT64)),
  STRUCT('Critical Care ICU', 'Rural',       145, 120, CAST(NULL AS FLOAT64)),
  STRUCT('Critical Care ICU', 'CEAC',        155, 140, CAST(NULL AS FLOAT64)),
  -- Surgical Services ASC (facility min=1)
  STRUCT('Surgical Services ASC', 'Large Metro', 20,  10, CAST(NULL AS FLOAT64)),
  STRUCT('Surgical Services ASC', 'Metro',       45,  30, CAST(NULL AS FLOAT64)),
  STRUCT('Surgical Services ASC', 'Micro',       80,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Surgical Services ASC', 'Rural',       75,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Surgical Services ASC', 'CEAC',       110, 100, CAST(NULL AS FLOAT64)),
  -- Skilled Nursing Facility (facility min=1)
  STRUCT('Skilled Nursing Facility', 'Large Metro', 20, 10, CAST(NULL AS FLOAT64)),
  STRUCT('Skilled Nursing Facility', 'Metro',       45, 30, CAST(NULL AS FLOAT64)),
  STRUCT('Skilled Nursing Facility', 'Micro',       80, 60, CAST(NULL AS FLOAT64)),
  STRUCT('Skilled Nursing Facility', 'Rural',       75, 60, CAST(NULL AS FLOAT64)),
  STRUCT('Skilled Nursing Facility', 'CEAC',        95, 85, CAST(NULL AS FLOAT64)),
  -- Diagnostic Radiology (facility min=1)
  STRUCT('Diagnostic Radiology', 'Large Metro', 20,  10, CAST(NULL AS FLOAT64)),
  STRUCT('Diagnostic Radiology', 'Metro',       45,  30, CAST(NULL AS FLOAT64)),
  STRUCT('Diagnostic Radiology', 'Micro',       80,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Diagnostic Radiology', 'Rural',       75,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Diagnostic Radiology', 'CEAC',       110, 100, CAST(NULL AS FLOAT64)),
  -- Mammography (facility min=1)
  STRUCT('Mammography', 'Large Metro', 20,  10, CAST(NULL AS FLOAT64)),
  STRUCT('Mammography', 'Metro',       45,  30, CAST(NULL AS FLOAT64)),
  STRUCT('Mammography', 'Micro',       80,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Mammography', 'Rural',       75,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Mammography', 'CEAC',       110, 100, CAST(NULL AS FLOAT64)),
  -- Physical Therapy (facility min=1)
  STRUCT('Physical Therapy', 'Large Metro', 20,  10, CAST(NULL AS FLOAT64)),
  STRUCT('Physical Therapy', 'Metro',       45,  30, CAST(NULL AS FLOAT64)),
  STRUCT('Physical Therapy', 'Micro',       80,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Physical Therapy', 'Rural',       75,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Physical Therapy', 'CEAC',       110, 100, CAST(NULL AS FLOAT64)),
  -- Occupational Therapy (facility min=1)
  STRUCT('Occupational Therapy', 'Large Metro', 20,  10, CAST(NULL AS FLOAT64)),
  STRUCT('Occupational Therapy', 'Metro',       45,  30, CAST(NULL AS FLOAT64)),
  STRUCT('Occupational Therapy', 'Micro',       80,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Occupational Therapy', 'Rural',       75,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Occupational Therapy', 'CEAC',       110, 100, CAST(NULL AS FLOAT64)),
  -- Speech Therapy (facility min=1)
  STRUCT('Speech Therapy', 'Large Metro', 20,  10, CAST(NULL AS FLOAT64)),
  STRUCT('Speech Therapy', 'Metro',       45,  30, CAST(NULL AS FLOAT64)),
  STRUCT('Speech Therapy', 'Micro',       80,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Speech Therapy', 'Rural',       75,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Speech Therapy', 'CEAC',       110, 100, CAST(NULL AS FLOAT64)),
  -- Inpatient Psychiatric (facility min=1)
  STRUCT('Inpatient Psychiatric', 'Large Metro',  30,  15, CAST(NULL AS FLOAT64)),
  STRUCT('Inpatient Psychiatric', 'Metro',        70,  45, CAST(NULL AS FLOAT64)),
  STRUCT('Inpatient Psychiatric', 'Micro',       100,  75, CAST(NULL AS FLOAT64)),
  STRUCT('Inpatient Psychiatric', 'Rural',        90,  75, CAST(NULL AS FLOAT64)),
  STRUCT('Inpatient Psychiatric', 'CEAC',        155, 140, CAST(NULL AS FLOAT64)),
  -- Outpatient Infusion/Chemo (facility min=1)
  STRUCT('Outpatient Infusion/Chemo', 'Large Metro', 20,  10, CAST(NULL AS FLOAT64)),
  STRUCT('Outpatient Infusion/Chemo', 'Metro',       45,  30, CAST(NULL AS FLOAT64)),
  STRUCT('Outpatient Infusion/Chemo', 'Micro',       80,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Outpatient Infusion/Chemo', 'Rural',       75,  60, CAST(NULL AS FLOAT64)),
  STRUCT('Outpatient Infusion/Chemo', 'CEAC',       110, 100, CAST(NULL AS FLOAT64)),
  -- Outpatient Behavioral Health (facility min=1)
  STRUCT('Outpatient Behavioral Health', 'Large Metro', 20,  10, CAST(NULL AS FLOAT64)),
  STRUCT('Outpatient Behavioral Health', 'Metro',       40,  25, CAST(NULL AS FLOAT64)),
  STRUCT('Outpatient Behavioral Health', 'Micro',       55,  40, CAST(NULL AS FLOAT64)),
  STRUCT('Outpatient Behavioral Health', 'Rural',       60,  50, CAST(NULL AS FLOAT64)),
  STRUCT('Outpatient Behavioral Health', 'CEAC',       110, 100, CAST(NULL AS FLOAT64))
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

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers_multi_specialty_v2`
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
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_specialty_crosswalk_expanded` sc
  ON TRIM(CAST(s.specialty_cd AS STRING)) = TRIM(CAST(sc.aetna_code AS STRING))
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

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_fact_zip_access_v2`
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
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers_multi_specialty_v2` p
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

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_fact_gap_analysis_v2`
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
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers_multi_specialty_v2`
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
  LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_fact_zip_access_v2` z
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
    , 4)                                                             AS pct_covered
  FROM zip_access_complete
  GROUP BY
    county_fips,
    county_name,
    county_type,
    compliance_threshold,
    cms_specialty,
    plan_type
),

distinct_providers AS (
  -- --------------------------------------------------------
  -- COUNT DISTINCT PROVIDERS PER COUNTY × SPECIALTY × PLAN TYPE
  -- PER 422.116(e)(1)(i): PROVIDER MUST BE WITHIN THRESHOLD
  -- OF AT LEAST ONE BENEFICIARY TO COUNT
  -- FIXES DOUBLE COUNT BUG:
  --   OLD: SUM(provider_count_per_zip) counts same provider multiple times
  --   NEW: COUNT(DISTINCT provider_id) — each provider counted once per county
  -- --------------------------------------------------------
  SELECT
    b.county_fips,
    p.cms_specialty,
    p.plan_type,
    COUNT(DISTINCT p.provider_id)                                    AS actual_provider_count
  FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_beneficiaries` b
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_zip_reference` bene_zip
    ON b.zip_code = bene_zip.zip_code
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers_multi_specialty_v2` p
    ON TRUE
  JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_time_distance` t
    ON t.cms_specialty = p.cms_specialty
    AND t.county_type  = b.county_type
  WHERE ST_DISTANCE(
          ST_GEOGPOINT(bene_zip.zip_long, bene_zip.zip_lat),
          ST_GEOGPOINT(p.zip_long,        p.zip_lat)
        ) / 1609.34 <= t.max_distance_miles
  GROUP BY
    b.county_fips,
    p.cms_specialty,
    p.plan_type
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
  FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers_multi_specialty_v2` p
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
  t.min_ratio_per_1000,
  t.max_distance_miles,
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
    ELSE COALESCE(dp.actual_provider_count, 0)
  END                                                                AS actual_count,
  -- gap: beds vs required for hospitals, providers vs required for others
  CASE
    WHEN r.cms_specialty = 'Acute Inpatient Hospitals'
      THEN hsd.required_count - COALESCE(b.total_contracted_beds, 0)
    ELSE hsd.required_count - COALESCE(dp.actual_provider_count, 0)
  END                                                                AS provider_gap,
  -- test 1: % population with access >= compliance threshold
  CASE
    WHEN r.pct_covered >= r.compliance_threshold THEN TRUE
    ELSE FALSE
  END                                                                AS access_compliant,
  -- test 2: beds >= required for hospitals, providers >= required for others
  CASE
    WHEN r.cms_specialty = 'Acute Inpatient Hospitals'
      THEN COALESCE(b.total_contracted_beds, 0) >= hsd.required_count
    ELSE COALESCE(dp.actual_provider_count, 0)  >= hsd.required_count
  END                                                                AS count_compliant,
  -- overall: both tests must pass per 42 CFR 422.116
  CASE
    WHEN r.pct_covered >= r.compliance_threshold
    AND (
      CASE
        WHEN r.cms_specialty = 'Acute Inpatient Hospitals'
          THEN COALESCE(b.total_contracted_beds, 0) >= hsd.required_count
        ELSE COALESCE(dp.actual_provider_count, 0)  >= hsd.required_count
      END
    )                                                                THEN 'COMPLIANT'
    ELSE 'NON-COMPLIANT'
  END                                                                AS compliance_status

FROM county_rollup r
LEFT JOIN distinct_providers dp
  ON r.county_fips    = dp.county_fips
  AND r.cms_specialty = dp.cms_specialty
  AND r.plan_type     = dp.plan_type
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_hsd_required_counts` hsd
  ON hsd.county_name    = r.county_name
  AND hsd.cms_specialty = r.cms_specialty
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_ref_time_distance` t
  ON t.cms_specialty = r.cms_specialty
  AND t.county_type  = r.county_type
LEFT JOIN hospital_beds b
  ON r.county_fips  = b.county_fips
  AND r.plan_type   = b.plan_type
ORDER BY
  r.county_name,
  r.cms_specialty,
  r.plan_type;
