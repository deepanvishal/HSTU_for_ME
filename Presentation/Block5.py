# ============================================================
# Block 5 — Sequence + Confidence Facts
# FACT-09: Hit@3 at T30 by sequence length bucket
# FACT-24: Trigger distribution by confidence bucket
# FACT-25: Annual trigger volume in high-confidence
# FACT-27: Conditions above deployment threshold
# FACT-28: Trigger volume for deployable conditions
# Sources: trigger_scores, test_sequences_5pct,
#          analysis_perf_by_diag
# Requires: Block 3 (best_model, fact_19_raw)
# ============================================================
from IPython.display import display, Markdown

# ── Query 1: Hit@3 by sequence length bucket ─────────────────
fact_09 = client.query(f"""
    WITH seq_lengths AS (
        SELECT
            member_id
            ,trigger_date
            ,trigger_dx
            ,MAX(recency_rank)                           AS seq_length
        FROM `{DS}.A870800_gen_rec_test_sequences_5pct`
        GROUP BY 1, 2, 3
    ),
    joined AS (
        SELECT
            s.hit_at_3
            ,sl.seq_length
            ,CASE
                WHEN sl.seq_length < 5  THEN '<5'
                WHEN sl.seq_length < 10 THEN '5-9'
                WHEN sl.seq_length < 15 THEN '10-14'
                ELSE '15-20'
            END                                          AS seq_bucket
        FROM `{DS}.A870800_gen_rec_trigger_scores` s
        JOIN seq_lengths sl
            ON s.member_id = sl.member_id
            AND s.trigger_date = sl.trigger_date
            AND s.trigger_dx = sl.trigger_dx
        WHERE s.time_bucket = 'T0_30'
          AND s.model = '{best_model}'
    )
    SELECT
        seq_bucket
        ,COUNT(*)                                        AS n_triggers
        ,ROUND(AVG(hit_at_3), 4)                         AS hit_at_3
    FROM joined
    GROUP BY seq_bucket
    ORDER BY
        CASE seq_bucket
            WHEN '<5'   THEN 1
            WHEN '5-9'  THEN 2
            WHEN '10-14' THEN 3
            WHEN '15-20' THEN 4
        END
""").to_dataframe()

display(Markdown("### FACT-09: Hit@3 at T30 by Sequence Length"))
display(fact_09)

FACT_09_SHORT = f"{float(fact_09[fact_09['seq_bucket'] == '<5']['hit_at_3'].values[0]) * 100:.1f}" if len(fact_09[fact_09['seq_bucket'] == '<5']) > 0 else "N/A"
FACT_09_LONG = f"{float(fact_09[fact_09['seq_bucket'] == '15-20']['hit_at_3'].values[0]) * 100:.1f}" if len(fact_09[fact_09['seq_bucket'] == '15-20']) > 0 else "N/A"

display(Markdown(f"""
- Shortest sequences (<5): **{FACT_09_SHORT}%**
- Longest sequences (15-20): **{FACT_09_LONG}%**
"""))

# ── Query 2: Confidence distribution ──────────────────────────
fact_24 = client.query(f"""
    SELECT
        hit_at_3
        ,COUNT(*)                                        AS n_triggers
    FROM `{DS}.A870800_gen_rec_trigger_scores`
    WHERE time_bucket = 'T0_30'
      AND model = '{best_model}'
    GROUP BY hit_at_3
""").to_dataframe()

total_triggers_scored = float(fact_24['n_triggers'].sum())
high_conf = float(fact_24[fact_24['hit_at_3'] == 1]['n_triggers'].sum())
low_conf = total_triggers_scored - high_conf

FACT_24_HIGH_PCT = f"{high_conf / total_triggers_scored * 100:.1f}"
FACT_24_LOW_PCT = f"{low_conf / total_triggers_scored * 100:.1f}"

# FACT-25: Annualize — test set is 2024-2025 (~2 years)
FACT_25 = f"{high_conf / 2:,.0f}"

display(Markdown(f"""
### FACT-24: Confidence Distribution at T30
| Bucket | Triggers | % |
|---|---|---|
| High confidence (Hit@3 = 1) | {high_conf:,.0f} | {FACT_24_HIGH_PCT}% |
| Low confidence (Hit@3 = 0) | {low_conf:,.0f} | {FACT_24_LOW_PCT}% |

### FACT-25: Annual High-Confidence Triggers
- {FACT_25} triggers/year
"""))

# ── FACT-27, FACT-28: Deployable conditions ───────────────────
try:
    DEPLOY_THRESHOLD = 0.5
    deployable = fact_19_raw[fact_19_raw["hit_at_3"] >= DEPLOY_THRESHOLD]
    FACT_27 = len(deployable)
    FACT_28 = f"{int(deployable['trigger_volume'].sum()):,}"

    display(Markdown(f"""
### FACT-27 + FACT-28: Deployable Conditions (Hit@3 >= {DEPLOY_THRESHOLD})
- {FACT_27} conditions above threshold
- {FACT_28} triggers covered
"""))
except NameError:
    print("WARNING: fact_19_raw not found — run Block 3 first")
    FACT_27 = "TBD"
    FACT_28 = "TBD"

print("Block 5 done.")
