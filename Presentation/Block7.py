# ============================================================
# Block 7 — Corrected Facts at K=5
# FACT-09: Hit@5 at T30 by sequence length bucket
# FACT-24: Confidence distribution at Hit@5
# FACT-25: Annual high-confidence triggers
# FACT-26: Dollar value in high-confidence (using corrected T180 $)
# FACT-27: Deployable conditions (Hit@5 >= 0.5)
# FACT-28: Trigger volume for deployable conditions
# Requires: client, DS, best_model from Block 3, L4 spend from Block 6a
# ============================================================
from IPython.display import display, Markdown
import pandas as pd

# ── FACT-09: Hit@5 at T30 by sequence length ──────────────────
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
            s.hit_at_5
            ,sl.seq_length
            ,CASE
                WHEN sl.seq_length < 5  THEN '<5'
                WHEN sl.seq_length < 10 THEN '5-9'
                WHEN sl.seq_length < 15 THEN '10-14'
                ELSE '15-20'
            END                                          AS seq_bucket
        FROM `{DS}.A870800_gen_rec_trigger_scores` s
        JOIN seq_lengths sl
            ON CAST(s.member_id AS STRING) = CAST(sl.member_id AS STRING)
            AND CAST(s.trigger_date AS DATE) = sl.trigger_date
            AND s.trigger_dx = sl.trigger_dx
        WHERE s.time_bucket = 'T0_30'
          AND s.model = '{best_model}'
          AND s.true_labels IS NOT NULL AND s.true_labels != ''
          AND s.top5_predictions IS NOT NULL AND s.top5_predictions != ''
    )
    SELECT
        seq_bucket
        ,COUNT(*)                                        AS n_triggers
        ,ROUND(AVG(hit_at_5), 4)                         AS hit_at_5
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

display(Markdown("### FACT-09: Hit@5 at T30 by Sequence Length"))
display(fact_09)

# ── FACT-24: Confidence distribution at Hit@5 ─────────────────
fact_24 = client.query(f"""
    SELECT
        hit_at_5
        ,COUNT(*)                                        AS n_triggers
    FROM `{DS}.A870800_gen_rec_trigger_scores`
    WHERE time_bucket = 'T0_30'
      AND model = '{best_model}'
      AND true_labels IS NOT NULL AND true_labels != ''
      AND top5_predictions IS NOT NULL AND top5_predictions != ''
    GROUP BY hit_at_5
""").to_dataframe()

total_scored = float(fact_24['n_triggers'].sum())
high_conf = float(fact_24[fact_24['hit_at_5'] == 1]['n_triggers'].sum())
low_conf = total_scored - high_conf

FACT_24_HIGH_PCT = f"{high_conf / total_scored * 100:.1f}"
FACT_24_LOW_PCT = f"{low_conf / total_scored * 100:.1f}"

# FACT-25: Annualize — test set is 2024-2025 (~2 years)
FACT_25 = f"{high_conf / 2:,.0f}"

display(Markdown(f"""
### FACT-24: Confidence Distribution at T30 (Hit@5)
| Bucket | Triggers | % |
|---|---|---|
| High confidence (Hit@5 = 1) | {high_conf:,.0f} | {FACT_24_HIGH_PCT}% |
| Low confidence (Hit@5 = 0) | {low_conf:,.0f} | {FACT_24_LOW_PCT}% |

### FACT-25: Annual High-Confidence Triggers (Hit@5)
- {FACT_25} triggers/year
"""))

# ── FACT-27 + FACT-28: Deployable conditions at Hit@5 ─────────
deploy = client.query(f"""
    WITH all_scores AS (
        SELECT trigger_dx, hit_at_5
        FROM `{DS}.A870800_gen_rec_trigger_scores`
        WHERE time_bucket = 'T0_30'
          AND model = '{best_model}'
          AND true_labels IS NOT NULL AND true_labels != ''
          AND top5_predictions IS NOT NULL AND top5_predictions != ''
        UNION ALL
        SELECT trigger_dx, hit_at_5
        FROM `{DS}.A870800_gen_rec_markov_trigger_scores_5pct`
        WHERE time_bucket = 'T0_30'
          AND model = 'Markov'
          AND true_labels IS NOT NULL AND true_labels != ''
          AND top5_predictions IS NOT NULL AND top5_predictions != ''
    ),
    by_dx AS (
        SELECT
            trigger_dx
            ,COUNT(*)                                    AS trigger_volume
            ,ROUND(AVG(hit_at_5), 4)                     AS hit_at_5
        FROM all_scores
        GROUP BY trigger_dx
        HAVING COUNT(*) >= 20
    )
    SELECT
        COUNT(*)                                         AS n_conditions
        ,SUM(trigger_volume)                             AS total_triggers
    FROM by_dx
    WHERE hit_at_5 >= 0.5
""").to_dataframe().iloc[0]

FACT_27 = int(deploy['n_conditions'])
FACT_28 = f"{int(deploy['total_triggers']):,}"

display(Markdown(f"""
### FACT-27 + FACT-28: Deployable Conditions (Hit@5 >= 0.5)
- {FACT_27} conditions above threshold
- {FACT_28} triggers covered
"""))

# ── FACT-26: High-confidence dollar value (corrected) ─────────
# Uses corrected T180 spend from Block 6a
# Proportion of high-confidence triggers × T180 spend
try:
    t180_spend = float(level4['t180_spend'])
    high_conf_pct = high_conf / total_scored
    FACT_26_RAW = t180_spend * high_conf_pct
    FACT_26 = f"${FACT_26_RAW / 1e9:.1f}B"
    display(Markdown(f"""
### FACT-26: High-Confidence Dollar Value (approx, Hit@5)
- {FACT_24_HIGH_PCT}% of triggers are high-confidence
- T180 downstream spend: ${t180_spend / 1e9:.1f}B
- Estimated high-confidence spend: {FACT_26}
- Note: approximation — assumes spend proportional to trigger confidence rate
"""))
except NameError:
    FACT_26 = "TBD — run Block 6a first"
    print(f"WARNING: level4 not found. {FACT_26}")

print("Block 7 done.")
