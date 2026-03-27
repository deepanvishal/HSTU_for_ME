# ============================================================
# Block 2 — Triggers + Population Facts
# FACT-07: 5% sample member count
# FACT-08: Max sequence length (constant = 20)
# FACT-14: Qualified trigger count
# FACT-30: Train trigger count
# FACT-31: Test trigger count
# FACT-36: Member segment distribution
# FACT-37: Triggers before vs after boundary filtering
# Sources: triggers_qualified, population_stats, sample_members_5pct,
#          model_train, model_test
# ============================================================
from IPython.display import display, Markdown

FACT_08 = 20

# ── Query 1: Trigger counts + boundary impact ────────────────
q1 = client.query(f"""
    SELECT
        COUNT(*)                                         AS total_triggers
        ,COUNTIF(is_left_qualified = TRUE)               AS qualified_triggers
    FROM `{DS}.A870800_gen_rec_triggers_qualified`
""").to_dataframe().iloc[0]

FACT_14 = f"{q1['qualified_triggers']:,.0f}"
FACT_37_TOTAL = f"{q1['total_triggers']:,.0f}"
FACT_37_QUALIFIED = FACT_14
FACT_37_PCT = f"{q1['qualified_triggers'] / q1['total_triggers'] * 100:.1f}"

# ── Query 2: Train / Test / Sample / Segments ─────────────────
q2 = client.query(f"""
    SELECT
        (SELECT COUNT(*) FROM `{DS}.A870800_gen_rec_model_train`)   AS train_count
        ,(SELECT COUNT(*) FROM `{DS}.A870800_gen_rec_model_test`)   AS test_count
        ,(SELECT COUNT(DISTINCT member_id)
          FROM `{DS}.A870800_gen_rec_sample_members_5pct`)          AS sample_5pct_members
""").to_dataframe().iloc[0]

FACT_07 = f"{q2['sample_5pct_members']:,.0f}"
FACT_30 = f"{q2['train_count']:,.0f}"
FACT_31 = f"{q2['test_count']:,.0f}"

# ── Query 3: Segment distribution ─────────────────────────────
fact_36 = client.query(f"""
    SELECT
        member_segment
        ,COUNT(*)                                        AS member_count
        ,ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
    FROM `{DS}.A870800_gen_rec_population_stats`
    GROUP BY member_segment
    ORDER BY member_count DESC
""").to_dataframe()

display(Markdown(f"""
| FACT | Value |
|---|---|
| FACT-07 5% sample members | {FACT_07} |
| FACT-08 Max sequence length | {FACT_08} |
| FACT-14 Qualified triggers | {FACT_14} |
| FACT-30 Train triggers | {FACT_30} |
| FACT-31 Test triggers | {FACT_31} |
| FACT-37 Total triggers (pre-filter) | {FACT_37_TOTAL} |
| FACT-37 Pass rate | {FACT_37_PCT}% |
"""))

display(Markdown("### FACT-36: Member Segment Distribution"))
display(fact_36)

print("Block 2 done.")
