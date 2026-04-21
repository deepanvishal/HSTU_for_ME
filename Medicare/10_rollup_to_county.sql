-- 
SELECT
  COUNT(*)                        AS total_rows,
  COUNTIF(cms_specialty IS NULL)  AS null_specialty,
  COUNTIF(county_fips IS NULL)    AS null_county,
  COUNTIF(zip_lat IS NULL)        AS null_zip_lat
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers`
