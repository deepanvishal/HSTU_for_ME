-- ============================================================
-- EDA_self_loop_specialty.sql
-- Purpose : Accurate self-loop analysis at specialty level
-- Approach:
--   Step 1 — Get qualified triggers (left + right boundary)
--   Step 2 — Find immediate next visit of a DIFFERENT date
--             using MIN(visit_date) > trigger_date
--             No LAG/LEAD used
--   Step 3 — Valid pair = trigger + immediate next visit
--   Step 4 — Self loop = trigger specialty = next visit specialty
--   Step 5 — Count total triggers, valid pairs,
--             self loops, non self loops
-- ============================================================

-- ------------------------------------------------------------
-- STEP 1 + 2 + 3: Build valid trigger → next visit pairs
-- Dedup next visit to one row per member + visit_date
-- taking MAX(specialty_ctg_cd) to get one specialty per day
-- ------------------------------------------------------------
WITH qualified_triggers AS (
    SELECT DISTINCT
        member_id
        ,trigger_date
        ,trigger_dx
        ,trigger_specialty                                 AS trigger_specialty
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
    WHERE is_left_qualified = TRUE
      AND (
          is_t30_qualified  = TRUE
       OR is_t60_qualified  = TRUE
       OR is_t180_qualified = TRUE
      )
),
-- Deduplicate visits to one row per member + visit_date
-- Multiple specialties can occur on same day — take MAX for determinism
deduped_visits AS (
    SELECT
        member_id
        ,visit_date
        ,MAX(specialty_ctg_cd)                             AS specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE specialty_ctg_cd IS NOT NULL
    GROUP BY member_id, visit_date
),
-- For each trigger — find the immediate next visit date
next_visit_date AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.trigger_specialty
        ,MIN(v.visit_date)                                 AS next_visit_date
    FROM qualified_triggers t
    JOIN deduped_visits v
        ON t.member_id   = v.member_id
        AND v.visit_date > t.trigger_date
    GROUP BY t.member_id, t.trigger_date, t.trigger_dx, t.trigger_specialty
),
-- Join back to get the specialty of that next visit date
valid_pairs AS (
    SELECT
        n.member_id
        ,n.trigger_date
        ,n.trigger_dx
        ,n.trigger_specialty
        ,n.next_visit_date
        ,v.specialty_ctg_cd                                AS next_specialty
    FROM next_visit_date n
    JOIN deduped_visits v
        ON n.member_id      = v.member_id
        AND n.next_visit_date = v.visit_date
)

-- ------------------------------------------------------------
-- STEP 4 + 5: Count total, valid pairs, self loops, non self loops
-- ------------------------------------------------------------
SELECT
    -- Total qualified triggers (denominator)
    COUNT(DISTINCT CONCAT(t.member_id,'_',CAST(t.trigger_date AS STRING),'_',t.trigger_dx))
                                                           AS total_qualified_triggers

    -- Valid pairs — triggers that have an immediate next visit
    ,COUNT(p.trigger_date)                                 AS triggers_with_next_visit

    -- Triggers with no next visit found
    ,COUNT(DISTINCT CONCAT(t.member_id,'_',CAST(t.trigger_date AS STRING),'_',t.trigger_dx))
        - COUNT(p.trigger_date)                            AS triggers_no_next_visit

    -- Self loop — same specialty
    ,COUNTIF(p.trigger_specialty = p.next_specialty)       AS self_loops

    -- Non self loop — different specialty
    ,COUNTIF(p.trigger_specialty != p.next_specialty)      AS non_self_loops

    -- Rates
    ,ROUND(100.0 * COUNT(p.trigger_date) /
        COUNT(DISTINCT CONCAT(t.member_id,'_',CAST(t.trigger_date AS STRING),'_',t.trigger_dx))
        , 2)                                               AS pct_with_next_visit

    ,ROUND(100.0 * COUNTIF(p.trigger_specialty = p.next_specialty) /
        NULLIF(COUNT(p.trigger_date), 0)
        , 2)                                               AS self_loop_pct

    ,ROUND(100.0 * COUNTIF(p.trigger_specialty != p.next_specialty) /
        NULLIF(COUNT(p.trigger_date), 0)
        , 2)                                               AS non_self_loop_pct

FROM qualified_triggers t
LEFT JOIN valid_pairs p
    ON t.member_id    = p.member_id
    AND t.trigger_date = p.trigger_date
    AND t.trigger_dx   = p.trigger_dx
;

-- ------------------------------------------------------------
-- Self loop breakdown by trigger specialty
-- Which specialties have highest self-loop rate?
-- ------------------------------------------------------------
WITH qualified_triggers AS (
    SELECT DISTINCT
        member_id
        ,trigger_date
        ,trigger_dx
        ,trigger_specialty
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers_qualified`
    WHERE is_left_qualified = TRUE
      AND (
          is_t30_qualified  = TRUE
       OR is_t60_qualified  = TRUE
       OR is_t180_qualified = TRUE
      )
),
deduped_visits AS (
    SELECT
        member_id
        ,visit_date
        ,MAX(specialty_ctg_cd)                             AS specialty_ctg_cd
    FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visits`
    WHERE specialty_ctg_cd IS NOT NULL
    GROUP BY member_id, visit_date
),
next_visit_date AS (
    SELECT
        t.member_id
        ,t.trigger_date
        ,t.trigger_dx
        ,t.trigger_specialty
        ,MIN(v.visit_date)                                 AS next_visit_date
    FROM qualified_triggers t
    JOIN deduped_visits v
        ON t.member_id    = v.member_id
        AND v.visit_date  > t.trigger_date
    GROUP BY t.member_id, t.trigger_date, t.trigger_dx, t.trigger_specialty
),
valid_pairs AS (
    SELECT
        n.trigger_specialty
        ,v.specialty_ctg_cd                                AS next_specialty
    FROM next_visit_date n
    JOIN deduped_visits v
        ON n.member_id       = v.member_id
        AND n.next_visit_date = v.visit_date
)
SELECT
    trigger_specialty
    ,COUNT(*)                                              AS total_pairs
    ,COUNTIF(trigger_specialty = next_specialty)           AS self_loops
    ,COUNTIF(trigger_specialty != next_specialty)          AS non_self_loops
    ,ROUND(100.0 * COUNTIF(trigger_specialty = next_specialty)
        / COUNT(*), 2)                                     AS self_loop_pct
    ,ROUND(100.0 * COUNTIF(trigger_specialty != next_specialty)
        / COUNT(*), 2)                                     AS non_self_loop_pct
FROM valid_pairs
GROUP BY trigger_specialty
ORDER BY self_loop_pct DESC
;
