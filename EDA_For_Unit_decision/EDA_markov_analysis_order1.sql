-- ============================================================
-- DX TO SPECIALTY — ANY SEQUENTIAL VISITS
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_any`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_any`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH pairs AS (
    SELECT
        v1.member_id
        ,v1.member_segment
        ,v1.dx_raw                                       AS current_dx
        ,v1.dx_clean                                     AS current_dx_clean
        ,v1.ccsr_category                                AS current_ccsr
        ,v1.ccsr_category_description                    AS current_ccsr_desc
        ,v2.specialty_ctg_cd                             AS next_specialty
        ,v2.specialty_desc                               AS next_specialty_desc
        ,v2.allowed_amt                                  AS next_visit_allowed_amt
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v1
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits` v2
        ON v1.member_id = v2.member_id
        AND v2.visit_rank = v1.visit_rank + 1
    WHERE v1.dx_raw IS NOT NULL
      AND v2.specialty_ctg_cd IS NOT NULL
),
transition_counts AS (
    SELECT
        current_dx, current_dx_clean, current_ccsr, current_ccsr_desc
        ,next_specialty, next_specialty_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
        ,ROUND(SUM(next_visit_allowed_amt), 2)           AS total_allowed_amt
        ,ROUND(SUM(next_visit_allowed_amt)
            / NULLIF(COUNT(*), 0), 2)                    AS avg_allowed_per_transition
        ,ROUND(SUM(next_visit_allowed_amt)
            / NULLIF(COUNT(DISTINCT member_id), 0), 2)   AS avg_allowed_per_member
    FROM pairs
    GROUP BY
        current_dx, current_dx_clean, current_ccsr, current_ccsr_desc
        ,next_specialty, next_specialty_desc, member_segment
),
state_totals AS (
    SELECT current_dx, member_segment
        ,SUM(transition_count)                           AS dx_total
    FROM transition_counts
    GROUP BY current_dx, member_segment
)
SELECT
    t.current_dx
    ,dx_desc.icd9_dx_dscrptn                             AS current_dx_desc
    ,t.current_ccsr
    ,t.current_ccsr_desc
    ,t.next_specialty
    ,t.next_specialty_desc
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,s.dx_total
    ,ROUND(t.transition_count / s.dx_total, 4)           AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.dx_total *
        LOG(t.transition_count / s.dx_total)) OVER (
            PARTITION BY t.current_dx, t.member_segment
        ), 4)                                            AS conditional_entropy
    ,t.total_allowed_amt
    ,t.avg_allowed_per_transition
    ,t.avg_allowed_per_member
FROM transition_counts t
JOIN state_totals s
    ON t.current_dx = s.current_dx
    AND t.member_segment = s.member_segment
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
    ON t.current_dx = dx_desc.icd9_dx_cd
WHERE t.transition_count >= 100
ORDER BY t.current_dx, t.transition_count DESC;


-- ============================================================
-- DX TO SPECIALTY — FIRST ENCOUNTER OF DIAGNOSIS
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_first_encounter`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_first_encounter`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH pairs AS (
    SELECT
        t.member_id
        ,t.member_segment
        ,t.trigger_dx                                    AS current_dx
        ,t.trigger_dx_clean                              AS current_dx_clean
        ,t.trigger_ccsr                                  AS current_ccsr
        ,t.trigger_ccsr_desc                             AS current_ccsr_desc
        ,v.specialty_ctg_cd                              AS next_specialty
        ,v.specialty_desc                                AS next_specialty_desc
        ,v.allowed_amt                                   AS next_visit_allowed_amt
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
        ON t.member_id = v.member_id
        AND t.trigger_date = v.trigger_date
        AND t.trigger_dx = v.trigger_dx
        AND v.is_v2 = TRUE
    WHERE t.is_left_qualified = TRUE
      AND v.specialty_ctg_cd IS NOT NULL
),
transition_counts AS (
    SELECT
        current_dx, current_dx_clean, current_ccsr, current_ccsr_desc
        ,next_specialty, next_specialty_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
        ,ROUND(SUM(next_visit_allowed_amt), 2)           AS total_allowed_amt
        ,ROUND(SUM(next_visit_allowed_amt)
            / NULLIF(COUNT(*), 0), 2)                    AS avg_allowed_per_transition
        ,ROUND(SUM(next_visit_allowed_amt)
            / NULLIF(COUNT(DISTINCT member_id), 0), 2)   AS avg_allowed_per_member
    FROM pairs
    GROUP BY
        current_dx, current_dx_clean, current_ccsr, current_ccsr_desc
        ,next_specialty, next_specialty_desc, member_segment
),
state_totals AS (
    SELECT current_dx, member_segment
        ,SUM(transition_count)                           AS dx_total
    FROM transition_counts
    GROUP BY current_dx, member_segment
)
SELECT
    t.current_dx
    ,dx_desc.icd9_dx_dscrptn                             AS current_dx_desc
    ,t.current_ccsr
    ,t.current_ccsr_desc
    ,t.next_specialty
    ,t.next_specialty_desc
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,s.dx_total
    ,ROUND(t.transition_count / s.dx_total, 4)           AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.dx_total *
        LOG(t.transition_count / s.dx_total)) OVER (
            PARTITION BY t.current_dx, t.member_segment
        ), 4)                                            AS conditional_entropy
    ,t.total_allowed_amt
    ,t.avg_allowed_per_transition
    ,t.avg_allowed_per_member
FROM transition_counts t
JOIN state_totals s
    ON t.current_dx = s.current_dx
    AND t.member_segment = s.member_segment
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
    ON t.current_dx = dx_desc.icd9_dx_cd
WHERE t.transition_count >= 100
ORDER BY t.current_dx, t.transition_count DESC;


-- ============================================================
-- DX TO SPECIALTY — FP/I FIRST ENCOUNTER
-- ============================================================
DROP TABLE IF EXISTS `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_fp`;
CREATE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_dx_to_specialty_fp`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS
WITH pairs AS (
    SELECT
        t.member_id
        ,t.member_segment
        ,t.trigger_dx                                    AS current_dx
        ,t.trigger_dx_clean                              AS current_dx_clean
        ,t.trigger_ccsr                                  AS current_ccsr
        ,t.trigger_ccsr_desc                             AS current_ccsr_desc
        ,v.specialty_ctg_cd                              AS next_specialty
        ,v.specialty_desc                                AS next_specialty_desc
        ,v.allowed_amt                                   AS next_visit_allowed_amt
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified` t
    JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits_qualified` v
        ON t.member_id = v.member_id
        AND t.trigger_date = v.trigger_date
        AND t.trigger_dx = v.trigger_dx
        AND v.is_v2 = TRUE
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_specialty IN ('FP', 'I')
      AND v.specialty_ctg_cd IS NOT NULL
),
transition_counts AS (
    SELECT
        current_dx, current_dx_clean, current_ccsr, current_ccsr_desc
        ,next_specialty, next_specialty_desc, member_segment
        ,COUNT(*)                                        AS transition_count
        ,COUNT(DISTINCT member_id)                       AS unique_members
        ,ROUND(SUM(next_visit_allowed_amt), 2)           AS total_allowed_amt
        ,ROUND(SUM(next_visit_allowed_amt)
            / NULLIF(COUNT(*), 0), 2)                    AS avg_allowed_per_transition
        ,ROUND(SUM(next_visit_allowed_amt)
            / NULLIF(COUNT(DISTINCT member_id), 0), 2)   AS avg_allowed_per_member
    FROM pairs
    GROUP BY
        current_dx, current_dx_clean, current_ccsr, current_ccsr_desc
        ,next_specialty, next_specialty_desc, member_segment
),
state_totals AS (
    SELECT current_dx, member_segment
        ,SUM(transition_count)                           AS dx_total
    FROM transition_counts
    GROUP BY current_dx, member_segment
)
SELECT
    t.current_dx
    ,dx_desc.icd9_dx_dscrptn                             AS current_dx_desc
    ,t.current_ccsr
    ,t.current_ccsr_desc
    ,t.next_specialty
    ,t.next_specialty_desc
    ,t.member_segment
    ,t.transition_count
    ,t.unique_members
    ,s.dx_total
    ,ROUND(t.transition_count / s.dx_total, 4)           AS conditional_probability
    ,ROUND(-SUM(t.transition_count / s.dx_total *
        LOG(t.transition_count / s.dx_total)) OVER (
            PARTITION BY t.current_dx, t.member_segment
        ), 4)                                            AS conditional_entropy
    ,t.total_allowed_amt
    ,t.avg_allowed_per_transition
    ,t.avg_allowed_per_member
FROM transition_counts t
JOIN state_totals s
    ON t.current_dx = s.current_dx
    AND t.member_segment = s.member_segment
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` dx_desc
    ON t.current_dx = dx_desc.icd9_dx_cd
WHERE t.transition_count >= 100
ORDER BY t.current_dx, t.transition_count DESC;
