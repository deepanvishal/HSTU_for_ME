WITH procedure_groups AS (
  SELECT
    specialty_ctg_cd,
    CASE
      WHEN prcdr_cd IN ('90791','90792','90832','90833','90834','90835','90836','90837','90838',
                        '96101','96102','96103','96105','96110','96112','96113','96116','96121',
                        '96125','96130','96131','96132','96133','96136','96137','96138','96139') 
                                                          THEN 'Clinical Psychology'
      WHEN prcdr_cd IN ('90832','90833','90834','90835','90836','90837','90838','90839','90840') 
                                                          THEN 'Clinical Social Work'
      WHEN prcdr_cd IN ('97110','97112','97116','97140','97150',
                        '97161','97162','97163','97164','97165','97166','97167','97168',
                        '97530','97535','97542','97750','97755','97760','97761','97763') 
                                                          THEN 'Physical Therapy'
      WHEN prcdr_cd IN ('97165','97166','97167','97168','97530',
                        '97535','97542','97750','97755','97760','97761','97763') 
                                                          THEN 'Occupational Therapy'
      WHEN prcdr_cd IN ('92507','92508','92521','92522','92523','92524',
                        '92526','92610','92611','92612','92614','92616') 
                                                          THEN 'Speech Therapy'
      WHEN prcdr_cd BETWEEN '99304' AND '99310'           THEN 'Skilled Nursing Facility'
      WHEN prcdr_cd IN ('90801','90802','99221','99222','99223','99231','99232','99233') 
                                                          THEN 'Inpatient Psychiatric'
      WHEN prcdr_cd BETWEEN '96360' AND '96549'           THEN 'Outpatient Infusion/Chemo'
      WHEN prcdr_cd BETWEEN '33400' AND '33999'           THEN 'Cardiac Surgery Program'
      WHEN prcdr_cd IN ('93451','93452','93453','93454','93455','93456','93457','93458',
                        '93459','93460','93461','93462','93463','93464','93566','93567',
                        '93568','93569','93573','93574','93575') 
                                                          THEN 'Cardiac Catheterization'
      WHEN prcdr_cd IN ('77065','77066','77067','G0202','G0204','G0206') 
                                                          THEN 'Mammography'
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
