-- ============================================================
-- TRACK 1 PENETRATION SUMMARY — T30, T60, T180 WITH BINARY ENTROPY
-- Purpose : Penetration rate + binary entropy per trigger DX
--           + specialty + time window
-- Source  : A870800_gen_rec_visits_qualified
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_penetration`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_penetration`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH windowed AS (
    SELECT
        v.member_id
        ,v.trigger_dx
        ,v.trigger_ccsr                                  AS trigger_ccsr
        ,v.trigger_ccsr_desc                             AS trigger_ccsr_desc
        ,v.trigger_specialty
        ,v.member_segment
        ,v.specialty_ctg_cd                              AS visit_specialty
        ,v.specialty_desc                                AS visit_specialty_desc
        ,CASE
            WHEN v.days_since_trigger <= 30              THEN 'T30'
            WHEN v.days_since_trigger <= 60              THEN 'T60'
            WHEN v.days_since_trigger <= 180             THEN 'T180'
         END                                             AS time_window
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
    WHERE v.is_left_qualified = TRUE
      AND v.days_since_trigger <= 180
      AND v.specialty_ctg_cd IS NOT NULL
      AND v.specialty_ctg_cd != ''
),
member_trigger_totals AS (
    SELECT
        trigger_dx
        ,member_segment
        ,time_window
        ,COUNT(DISTINCT member_id)                       AS total_members
    FROM windowed
    GROUP BY trigger_dx, member_segment, time_window
),
specialty_visits AS (
    SELECT
        trigger_dx
        ,trigger_ccsr
        ,trigger_ccsr_desc
        ,trigger_specialty
        ,member_segment
        ,visit_specialty
        ,visit_specialty_desc
        ,time_window
        ,COUNT(DISTINCT member_id)                       AS members_visited
        ,COUNT(*)                                        AS visit_count
    FROM windowed
    GROUP BY
        trigger_dx, trigger_ccsr, trigger_ccsr_desc
        ,trigger_specialty, member_segment
        ,visit_specialty, visit_specialty_desc
        ,time_window
),
with_penetration AS (
    SELECT
        s.trigger_dx
        ,COALESCE(dx_desc.icd9_dx_dscrptn, s.trigger_dx) AS trigger_dx_desc
        ,s.trigger_ccsr
        ,s.trigger_ccsr_desc
        ,s.trigger_specialty
        ,sp_trig.long_dscrptn                            AS trigger_specialty_desc
        ,s.member_segment
        ,s.visit_specialty
        ,s.visit_specialty_desc
        ,s.time_window
        ,s.members_visited
        ,s.visit_count
        ,t.total_members
        ,ROUND(s.members_visited / t.total_members, 4)   AS penetration_rate
    FROM specialty_visits s
    JOIN member_trigger_totals t
        ON s.trigger_dx = t.trigger_dx
        AND s.member_segment = t.member_segment
        AND s.time_window = t.time_window
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
        ON s.trigger_dx = dx_desc.icd9_dx_cd
    LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` sp_trig
        ON s.trigger_specialty = sp_trig.global_lookup_cd
        AND LOWER(sp_trig.lookup_column_nm) = 'specialty_ctg_cd'
    WHERE t.total_members >= 100
)
SELECT
    *
    ,ROUND(
        -penetration_rate * LOG(penetration_rate)
        - (1 - penetration_rate) * LOG(1 - penetration_rate)
    , 4)                                                 AS binary_entropy
FROM with_penetration
WHERE penetration_rate > 0
  AND penetration_rate < 1
ORDER BY time_window, trigger_dx, penetration_rate DESC
