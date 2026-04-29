-- ============================================================
-- TABLE: stg_providers_multi_specialty
-- PURPOSE: GET ALL SPECIALTIES PER PROVIDER (NOT JUST PRIMARY)
--          VIA NETWORK ID EXPLOSION + RPDB_RPNPRAC JOIN
-- SOURCE:  A870800_medicare_supply_demand_mbr_with_zip
--          edp-prod-hcbstorage.edp_hcb_core_cnsv.RPDB_RPNPRAC
--          edp-prod-hcbstorage.edp_hcb_core_srcv.EPDB_PRVDR
--          edp-prod-hcbstorage.edp_hcb_core_srcv.RPDB_RINPR
--          edp-prod-hcbstorage.edp_hcb_core_cnsv.PRVDR_TY_X_SPCLTY
--          edp-prod-hcbstorage.edp_hcb_core_srcv.GLOBAL_LOOKUP
-- GRAIN:   provider_id x specialty_ctg_cd x plan_type (LONG FORMAT)
-- FIXES:   1. SAFE_CAST on network_id explosion (empty segment protection)
--          2. NULL guard on PRVDR_TY_X_SPCLTY join keys
--          3. NULL guard on GLOBAL_LOOKUP specialty_ctg_cd join
--          4. WHERE cms_specialty + zip_lat filters commented out
--             to retain unmatched providers for investigation
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_medicare_supply_demand_stg_providers_multi_specialty`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH mbr_exploded AS (
  -- --------------------------------------------------------
  -- EXPLODE NETWORK_ID STRING INTO INDIVIDUAL NETWORK IDS
  -- SPLIT ON '-' DELIMITER
  -- SAFE_CAST PROTECTS AGAINST EMPTY/NON-NUMERIC SEGMENTS
  -- --------------------------------------------------------
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
  -- --------------------------------------------------------
  -- GET ALL SPECIALTIES PER PROVIDER VIA NETWORK JOIN
  -- LEFT JOIN ON RPDB_RPNPRAC TO RETAIN UNMATCHED PROVIDERS
  -- CASE: prvdr_info_ty_cd = 'N' → individual provider
  --       ELSE → organization/facility
  -- --------------------------------------------------------
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
          SAFE_CAST(
            SAFE_CAST(
              CONCAT(
                COALESCE(CAST(b.rpnp_spcl_majcl_cd AS STRING), ''),
                COALESCE(CAST(b.rpnp_spcl_ctgry_cd AS STRING), ''),
                COALESCE(CAST(b.rpnp_spcl_sbcls_cd AS STRING), '')
              ) AS INT64
            ) AS STRING
          ),
          CAST(c.prvdr_type_cd AS STRING)
        )
      ELSE COALESCE(
          SAFE_CAST(
            SAFE_CAST(
              CONCAT(
                COALESCE(CAST(b.rpnp_spcl_majcl_cd AS STRING), ''),
                COALESCE(CAST(b.rpnp_spcl_ctgry_cd AS STRING), ''),
                COALESCE(CAST(b.rpnp_spcl_sbcls_cd AS STRING), '')
              ) AS INT64
            ) AS STRING
          ),
          CAST(d.rip_prvdr_type_cd AS STRING)
        )
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
  -- --------------------------------------------------------
  -- MAP specialty_cd → specialty_ctg_cd VIA PRVDR_TY_X_SPCLTY
  -- NULL GUARD ON JOIN KEYS TO PREVENT PHANTOM MATCHES
  -- --------------------------------------------------------
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
  -- specialty_cd description
  LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.GLOBAL_LOOKUP` c
    ON TRIM(CAST(a.specialty_cd AS STRING))         = TRIM(CAST(c.global_lookup_cd AS STRING))
    AND c.lookup_column_nm                          = 'SPECIALTY_CD'
  -- specialty_ctg_cd description
  LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.GLOBAL_LOOKUP` d
    ON b.specialty_ctg_cd IS NOT NULL
    AND TRIM(CAST(b.specialty_ctg_cd AS STRING))    = TRIM(CAST(d.global_lookup_cd AS STRING))
    AND d.lookup_column_nm                          = 'SPECIALTY_CTG_CD'
  -- provider type description
  LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_srcv.GLOBAL_LOOKUP` e
    ON TRIM(CAST(a.rpnp_prvdr_type_cd AS STRING))   = TRIM(CAST(e.global_lookup_cd AS STRING))
    AND e.lookup_column_nm                          = 'PROVIDER_TYPE_CD'
)

-- --------------------------------------------------------
-- FINAL: LONG FORMAT
-- ONE ROW PER provider_id x specialty_ctg_cd x plan_type
-- --------------------------------------------------------
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
  sc.match_type,
  sc.inflated,
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
-- WHERE sc.cms_specialty IS NOT NULL  -- commented out: retain unmatched for investigation
-- AND z.zip_lat IS NOT NULL           -- commented out: retain unmatched for investigation
GROUP BY ALL
ORDER BY s.provider_id, sc.cms_specialty, s.plan_type
