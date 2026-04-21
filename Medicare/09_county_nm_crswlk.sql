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
