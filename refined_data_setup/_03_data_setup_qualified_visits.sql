-- ============================================================
-- TABLE 5 — A870800_gen_rec_visits_qualified
-- Purpose : All downstream visits after a qualified trigger
--           within T180 days — flat row level
-- Source  : A870800_gen_rec_triggers_qualified
--           + A870800_gen_rec_visits
-- Output  : One row per member + trigger + downstream visit date
--           + specialty + dx
--           Slice by days_since_trigger for T30/T60/T180
--           Slice by trigger_specialty for FP/I analysis
--           Slice by is_t180_qualified etc for window eligibility
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
SELECT
    -- Trigger context
    t.member_id
    ,t.trigger_date
    ,t.trigger_rank
    ,t.trigger_dx
    ,t.trigger_dx_clean
    ,t.trigger_ccsr
    ,t.trigger_ccsr_desc
    ,t.trigger_specialty
    ,t.trigger_specialty_desc
    ,t.member_segment
    ,t.age_nbr
    ,t.gender_cd

    -- Window qualification flags — carried through for downstream filtering
    ,t.is_left_qualified
    ,t.is_t30_qualified
    ,t.is_t60_qualified
    ,t.is_t180_qualified
    ,t.has_claims_12m_before

    -- Downstream visit attributes
    ,v.visit_date
    ,v.visit_rank                                        AS downstream_visit_rank
    ,DATE_DIFF(v.visit_date, t.trigger_date, DAY)        AS days_since_trigger
    ,v.srv_prvdr_id
    ,v.specialty_ctg_cd
    ,v.specialty_desc
    ,v.dx_raw
    ,v.dx_clean
    ,v.ccsr_category
    ,v.ccsr_category_description
    ,v.plc_srv_cd
    ,v.med_cost_ctg_cd
    ,v.allowed_amt

    -- V2 flag — immediate next visit after trigger
    ,CASE
        WHEN v.visit_rank = t.trigger_rank + 1
        THEN TRUE ELSE FALSE
     END                                                 AS is_v2

FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v
    ON t.member_id = v.member_id
    AND v.visit_date > t.trigger_date
    AND v.visit_date <= DATE_ADD(t.trigger_date, INTERVAL 180 DAY)
WHERE t.is_left_qualified = TRUE
