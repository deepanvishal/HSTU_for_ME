WITH visit_spine AS (
    SELECT DISTINCT
        member_id
        ,visit_date
        ,DENSE_RANK() OVER (
            PARTITION BY member_id
            ORDER BY visit_date
        )                                                AS visit_rank
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged`
    WHERE member_id = '<your_member_id>'
),
triggers AS (
    SELECT DISTINCT
        f.member_id
        ,f.visit_date                                    AS trigger_date
        ,f.visit_number
        ,f.dx_raw                                        AS trigger_dx
        ,s.visit_rank                                    AS trigger_rank
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` f
    JOIN visit_spine s
        ON f.member_id = s.member_id
        AND f.visit_date = s.visit_date
    WHERE f.is_first_dx_encounter = TRUE
        AND f.member_id = '<your_member_id>'
),
chained AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.trigger_rank
        ,v2.visit_date                                   AS v2_date
        ,v3.visit_date                                   AS v3_date
        ,v4.visit_date                                   AS v4_date
    FROM triggers t
    LEFT JOIN visit_spine v2
        ON t.member_id = v2.member_id
        AND v2.visit_rank = t.trigger_rank + 1
    LEFT JOIN visit_spine v3
        ON t.member_id = v3.member_id
        AND v3.visit_rank = t.trigger_rank + 2
    LEFT JOIN visit_spine v4
        ON t.member_id = v4.member_id
        AND v4.visit_rank = t.trigger_rank + 3
)
SELECT
    c.member_id
    ,c.trigger_date
    ,c.trigger_dx
    ,c.v2_date
    ,c.v3_date
    ,c.v4_date
    -- v2 claims
    ,v2.dx_raw                                           AS v2_dx
    ,v2.specialty_ctg_cd                                 AS v2_specialty
    ,v2.ccsr_category                                    AS v2_ccsr
    -- v3 claims
    ,v3.dx_raw                                           AS v3_dx
    ,v3.specialty_ctg_cd                                 AS v3_specialty
    ,v3.ccsr_category                                    AS v3_ccsr
    -- v4 claims
    ,v4.dx_raw                                           AS v4_dx
    ,v4.specialty_ctg_cd                                 AS v4_specialty
    ,v4.ccsr_category                                    AS v4_ccsr
FROM chained c
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` v2
    ON c.member_id = v2.member_id
    AND c.v2_date = v2.visit_date
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` v3
    ON c.member_id = v3.member_id
    AND c.v3_date = v3.visit_date
LEFT JOIN `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` v4
    ON c.member_id = v4.member_id
    AND c.v4_date = v4.visit_date
ORDER BY
    c.trigger_date
    ,c.trigger_dx
    ,c.v2_date
    ,c.v3_date
