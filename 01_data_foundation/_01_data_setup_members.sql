DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_demographics`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_demographics`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    member_id
    ,eff_year                                            AS membership_year
    ,MAX(age_nbr)                                        AS age_nbr
    ,MAX(gender_cd)                                      AS gender_cd
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_members`
GROUP BY member_id, eff_year
;

DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_member_qualified`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    member_id
    ,MIN(eff_dt)                                         AS enrollment_start
    ,MAX(eff_dt)                                         AS enrollment_end
    ,DATE_DIFF(MAX(eff_dt), MIN(eff_dt), MONTH)          AS enrollment_window_months
    ,COUNT(DISTINCT eff_dt)                              AS enrolled_months
    ,MAX(zip_cd)                                         AS zip_cd
    ,MAX(state_postal_cd)                                AS state_postal_cd
    ,MAX(county_cd)                                      AS county_cd
    ,MAX(county_nm)                                      AS county_nm
    ,MAX(market)                                         AS market
    ,MAX(submarket)                                      AS submarket
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_members`
GROUP BY member_id
