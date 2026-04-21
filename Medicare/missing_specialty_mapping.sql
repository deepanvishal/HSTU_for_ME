WITH procedure_groups AS (
  SELECT
    specialty_ctg_cd,
    CASE
      WHEN prcdr_cd IN ('90791','90834','90837','90847')                         THEN 'Clinical Psychology'
      WHEN prcdr_cd IN ('90832','90833','90838','98960')                         THEN 'Clinical Social Work'
      WHEN prcdr_cd BETWEEN '97010' AND '97799'                                 THEN 'Physical Therapy'
      WHEN prcdr_cd IN ('97165','97166','97167','97168','97530','97535')         THEN 'Occupational Therapy'
      WHEN prcdr_cd IN ('92507','92508','92521','92522','92523','92524')         THEN 'Speech Therapy'
      WHEN prcdr_cd BETWEEN '99304' AND '99310'                                 THEN 'Skilled Nursing Facility'
      WHEN prcdr_cd IN ('90801','90802','99221','99222','99223')                 THEN 'Inpatient Psychiatric'
      WHEN prcdr_cd BETWEEN '96360' AND '96417'                                 THEN 'Outpatient Infusion/Chemo'
      WHEN prcdr_cd BETWEEN '33400' AND '33999'                                 THEN 'Cardiac Surgery Program'
      WHEN prcdr_cd BETWEEN '93451' AND '93572'                                 THEN 'Cardiac Catheterization'
      WHEN prcdr_cd IN ('77065','77066','77067')                                 THEN 'Mammography'
    END AS cms_specialty_group,
    COUNT(*) AS claim_count
  FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
  WHERE prcdr_cd IS NOT NULL
  GROUP BY specialty_ctg_cd, cms_specialty_group
)

SELECT
  cms_specialty_group,
  specialty_ctg_cd,
  claim_count,
  RANK() OVER (PARTITION BY cms_specialty_group ORDER BY claim_count DESC) AS rnk
FROM procedure_groups
WHERE cms_specialty_group IS NOT NULL
QUALIFY rnk <= 3
ORDER BY cms_specialty_group, rnk
