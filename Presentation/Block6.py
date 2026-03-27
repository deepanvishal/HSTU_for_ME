# ============================================================
# Block 6a — Dollar Funnel (corrected)
# Level 1: Total $ from all SFL claims
# Level 2: $ on trigger day only (member + trigger_date)
# Level 3: $ on immediate next visit day (V2)
# Level 4: $ for all claims within T180 (deduplicated)
# Requires: client, DS
# ============================================================
from IPython.display import display, Markdown

# ── Level 1: Total spend ──────────────────────────────────────
level1 = client.query(f"""
    SELECT
        SUM(CAST(allowed_amt AS FLOAT64))                AS total_spend
        ,COUNT(*)                                        AS total_claims
    FROM `{DS}.A870800_claims_gen_rec_2022_2025_sfl`
""").to_dataframe().iloc[0]

L1_SPEND = f"${float(level1['total_spend']) / 1e9:.1f}B"
L1_CLAIMS = f"{float(level1['total_claims']):,.0f}"

# ── Level 2: Trigger-day spend ────────────────────────────────
level2 = client.query(f"""
    WITH trigger_days AS (
        SELECT DISTINCT
            member_id
            ,trigger_date
        FROM `{DS}.A870800_gen_rec_triggers_qualified`
        WHERE is_left_qualified = TRUE
    )
    SELECT
        SUM(CAST(c.allowed_amt AS FLOAT64))              AS trigger_day_spend
        ,COUNT(*)                                        AS trigger_day_claims
    FROM trigger_days t
    JOIN `{DS}.A870800_claims_gen_rec_2022_2025_sfl` c
        ON t.member_id = c.member_id
        AND t.trigger_date = c.srv_start_dt
""").to_dataframe().iloc[0]

L2_SPEND = f"${float(level2['trigger_day_spend']) / 1e9:.1f}B"
L2_CLAIMS = f"{float(level2['trigger_day_claims']):,.0f}"

# ── Level 3: Immediate next visit (V2) spend ──────────────────
level3 = client.query(f"""
    WITH v2_dates AS (
        SELECT DISTINCT
            member_id
            ,trigger_date
            ,trigger_dx
            ,MIN(visit_date)                             AS v2_date
        FROM `{DS}.A870800_gen_rec_visits_qualified`
        WHERE days_since_trigger > 0
          AND specialty_ctg_cd IS NOT NULL
        GROUP BY 1, 2, 3
    ),
    v2_qualified AS (
        SELECT DISTINCT
            v.member_id
            ,v.v2_date
        FROM v2_dates v
        JOIN `{DS}.A870800_gen_rec_triggers_qualified` t
            ON v.member_id = t.member_id
            AND v.trigger_date = t.trigger_date
            AND v.trigger_dx = t.trigger_dx
        WHERE t.is_left_qualified = TRUE
    )
    SELECT
        SUM(CAST(c.allowed_amt AS FLOAT64))              AS v2_spend
        ,COUNT(*)                                        AS v2_claims
    FROM v2_qualified v
    JOIN `{DS}.A870800_claims_gen_rec_2022_2025_sfl` c
        ON v.member_id = c.member_id
        AND v.v2_date = c.srv_start_dt
""").to_dataframe().iloc[0]

L3_SPEND = f"${float(level3['v2_spend']) / 1e9:.1f}B"
L3_CLAIMS = f"{float(level3['v2_claims']):,.0f}"

# ── Level 4: All claims within T180 (deduplicated) ────────────
level4 = client.query(f"""
    WITH member_windows AS (
        SELECT
            member_id
            ,DATE_ADD(MIN(trigger_date), INTERVAL 1 DAY) AS earliest_start
            ,DATE_ADD(MAX(trigger_date), INTERVAL 180 DAY) AS latest_end
        FROM `{DS}.A870800_gen_rec_triggers_qualified`
        WHERE is_left_qualified = TRUE
        GROUP BY member_id
    )
    SELECT
        SUM(CAST(c.allowed_amt AS FLOAT64))              AS t180_spend
        ,COUNT(*)                                        AS t180_claims
    FROM member_windows w
    JOIN `{DS}.A870800_claims_gen_rec_2022_2025_sfl` c
        ON w.member_id = c.member_id
        AND c.srv_start_dt >= w.earliest_start
        AND c.srv_start_dt <= w.latest_end
""").to_dataframe().iloc[0]

L4_SPEND = f"${float(level4['t180_spend']) / 1e9:.1f}B"
L4_CLAIMS = f"{float(level4['t180_claims']):,.0f}"

display(Markdown(f"""
### Dollar Funnel (Corrected, Deduplicated)
| Level | Description | Spend | Claims |
|---|---|---|---|
| 1 | All claims in dataset | {L1_SPEND} | {L1_CLAIMS} |
| 2 | Trigger-day claims | {L2_SPEND} | {L2_CLAIMS} |
| 3 | Immediate next visit (V2) claims | {L3_SPEND} | {L3_CLAIMS} |
| 4 | All claims within T180 of trigger (approx) | {L4_SPEND} | {L4_CLAIMS} |
"""))

print("Block 6a done.")
