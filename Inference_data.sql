-- inference_members table
-- one row per member, last 20 visits only
CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_inference_sequences` AS

WITH ranked_visits AS (
    SELECT
        member_id
        ,visit_seq_num
        ,delta_t_bucket
        ,provider_ids
        ,specialty_codes
        ,dx_list
        ,procedure_codes
        ,ROW_NUMBER() OVER (
            PARTITION BY member_id
            ORDER BY visit_seq_num DESC
        ) AS recency_rank
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence`
)

SELECT
    member_id
    ,visit_seq_num
    ,delta_t_bucket
    ,provider_ids
    ,specialty_codes
    ,dx_list
    ,procedure_codes
    ,recency_rank
FROM ranked_visits
WHERE recency_rank <= 20
ORDER BY member_id, visit_seq_num ASC



CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_inference_labels` AS

SELECT
    l.member_id
    ,l.visit_seq_num
    ,l.specialties_30
    ,l.specialties_60
    ,l.specialties_180
    ,l.providers_30
    ,l.providers_60
    ,l.providers_180
    ,l.dx_30
    ,l.dx_60
    ,l.dx_180
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_label` l
INNER JOIN (
    SELECT
        member_id
        ,MAX(visit_seq_num) AS second_last_seq_num
    FROM (
        SELECT
            member_id
            ,visit_seq_num
            ,ROW_NUMBER() OVER (
                PARTITION BY member_id
                ORDER BY visit_seq_num DESC
            ) AS rn
        FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence`
    )
    WHERE rn = 2
    GROUP BY member_id
) last
    ON  l.member_id     = last.member_id
    AND l.visit_seq_num = last.second_last_seq_num
