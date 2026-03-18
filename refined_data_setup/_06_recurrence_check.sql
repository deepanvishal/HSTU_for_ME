-- Count 1: Current qualified triggers
SELECT COUNT(*) AS first_occurrence_triggers
    ,COUNT(DISTINCT member_id) AS unique_members
    ,COUNT(DISTINCT trigger_dx_clean) AS unique_dx_codes
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
WHERE is_left_qualified = TRUE;

-- Count 2: Additional triggers recurrence logic would add
SELECT COUNT(*) AS recurrence_triggers
    ,COUNT(DISTINCT member_id) AS unique_members
    ,COUNT(DISTINCT dx_clean) AS unique_dx_codes
FROM (
    SELECT
        member_id
        ,dx_clean
        ,visit_date
        ,LAG(visit_date) OVER (
            PARTITION BY member_id, dx_clean
            ORDER BY visit_date
        ) AS prior_dx_date
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
)
WHERE prior_dx_date IS NOT NULL
  AND DATE_DIFF(visit_date, prior_dx_date, DAY) > 365;
