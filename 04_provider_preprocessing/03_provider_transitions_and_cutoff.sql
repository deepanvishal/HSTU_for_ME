-- ============================================================
-- TABLE 1 — A870800_gen_rec_provider_transitions
-- Purpose : All cross-day provider-to-provider transition pairs
--           with transition counts across all members
-- Source  : A870800_gen_rec_visits
-- Grain   : (from_provider, to_provider)
-- Notes   : 
--   - visit_rank (DENSE_RANK on visit_date) already computed in source
--   - Dedup to (member, visit_rank, provider) first — source grain is
--     (member, date, provider, specialty, dx)
--   - Transition = provider at rank N -> provider at rank N+1
--   - Same-day transitions excluded naturally via visit_rank
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_transitions`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH provider_days AS (
    -- Deduplicate to (member, visit_rank, provider)
    -- visit_rank is DENSE_RANK on visit_date — same rank = same day
    -- This is the cheapest dedup — no window function needed
    SELECT DISTINCT
        member_id
        ,visit_rank
        ,srv_prvdr_id
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE srv_prvdr_id IS NOT NULL
),

transitions_raw AS (
    -- Self-join on member_id + consecutive visit_rank
    -- f = from (rank N), t = to (rank N+1)
    -- This captures ALL cross-day provider pairs including many-to-many
    SELECT
        f.srv_prvdr_id                                   AS from_provider
        ,t.srv_prvdr_id                                  AS to_provider
    FROM provider_days f
    JOIN provider_days t
        ON  f.member_id  = t.member_id
        AND t.visit_rank = f.visit_rank + 1
)

SELECT
    from_provider
    ,to_provider
    ,COUNT(*)                                            AS transition_count
FROM transitions_raw
GROUP BY
    from_provider
    ,to_provider
;


-- ============================================================
-- TABLE 2 — A870800_gen_rec_provider_vocab
-- Purpose : Provider vocab with inbound + outbound transition
--           volume, cumulative ranking, and 80% cutoff flag
-- Source  : A870800_gen_rec_provider_transitions
-- Grain   : One row per srv_prvdr_id
-- Output  : Use is_top80 flag for SASRec label vocab,
--           sequence UNK mapping, and downstream model tables
-- ============================================================

CREATE OR REPLACE TABLE `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_vocab`
OPTIONS (labels=[("owner", "deepan_thulasi_aetna_com")])
AS

WITH outbound AS (
    -- Total outbound transitions per provider (as source)
    SELECT
        from_provider                                    AS srv_prvdr_id
        ,SUM(transition_count)                           AS outbound_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_transitions`
    GROUP BY from_provider
),

inbound AS (
    -- Total inbound transitions per provider (as destination)
    SELECT
        to_provider                                      AS srv_prvdr_id
        ,SUM(transition_count)                           AS inbound_count
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_provider_transitions`
    GROUP BY to_provider
),

combined AS (
    -- Full outer join — provider may be source-only or destination-only
    SELECT
        COALESCE(o.srv_prvdr_id, i.srv_prvdr_id)         AS srv_prvdr_id
        ,COALESCE(o.outbound_count, 0)                   AS outbound_count
        ,COALESCE(i.inbound_count, 0)                    AS inbound_count
        ,COALESCE(o.outbound_count, 0)
            + COALESCE(i.inbound_count, 0)               AS total_transition_count
    FROM outbound o
    FULL OUTER JOIN inbound i
        ON o.srv_prvdr_id = i.srv_prvdr_id
),

ranked AS (
    SELECT
        srv_prvdr_id
        ,outbound_count
        ,inbound_count
        ,total_transition_count
        ,ROW_NUMBER() OVER (
            ORDER BY total_transition_count DESC
        )                                                AS provider_rank
        ,SUM(total_transition_count) OVER (
            ORDER BY total_transition_count DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                                AS cumulative_count
        ,SUM(total_transition_count) OVER ()             AS grand_total
    FROM combined
)

SELECT
    srv_prvdr_id
    ,provider_rank
    ,outbound_count
    ,inbound_count
    ,total_transition_count
    ,cumulative_count
    ,grand_total
    ,ROUND(cumulative_count * 100.0 / grand_total, 4)    AS cumulative_pct
    ,CASE
        WHEN cumulative_count <= grand_total * 0.80
        THEN TRUE ELSE FALSE
     END                                                 AS is_top80
FROM ranked
ORDER BY provider_rank
;
