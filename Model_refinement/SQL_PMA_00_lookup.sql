-- ============================================================
-- SQL_PMA_00 — PROVIDER NAME LOOKUP
-- Purpose : Extract srv_prvdr_id + pin_name from large sfl table
--           into a small lookup table used by all PMA SQLs
--           Run ONCE — avoids repeated scans of the large sfl table
-- Source  : A870800_claims_gen_rec_2022_2025_sfl
-- Output  : A870800_gen_rec_provider_name_lookup
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_name_lookup`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    srv_prvdr_id
    ,MAX(pin_name)                                       AS provider_name
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`
WHERE srv_prvdr_id IS NOT NULL
  AND pin_name     IS NOT NULL
  AND TRIM(pin_name) != ''
GROUP BY srv_prvdr_id
;
