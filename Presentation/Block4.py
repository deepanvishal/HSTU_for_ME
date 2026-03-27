# ============================================================
# Block 4 — Spend + Dollar Value Facts
# FACT-12: Total dollar value of all claims
# FACT-15: Dollar value within T180 of triggers
# FACT-16: Downstream visit count within T180
# FACT-20: Penetration rates for top conditions
# FACT-26: Dollar value in high-confidence bucket (approx)
# FACT-32: Spend per specialty (for quadrant chart)
# Sources: f_spend_summary, markov_train
# ============================================================
from IPython.display import display, Markdown

# ── Query 1: Total spend + downstream spend ───────────────────
spend_totals = client.query(f"""
    SELECT
        SUM(CASE WHEN visit_type = 'trigger'
            THEN total_allowed_amt ELSE 0 END)           AS trigger_spend
        ,SUM(CASE WHEN visit_type IN ('downstream', 'v2')
            THEN total_allowed_amt ELSE 0 END)           AS downstream_spend
        ,SUM(total_allowed_amt)                          AS total_spend
        ,SUM(CASE WHEN visit_type IN ('downstream', 'v2')
            THEN visit_count ELSE 0 END)                 AS downstream_visits
    FROM `{DS}.A870800_gen_rec_f_spend_summary`
    WHERE summary_type = 'specialty'
""").to_dataframe().iloc[0]

FACT_12 = f"${spend_totals['total_spend'] / 1e9:.2f}B"
FACT_12_RAW = spend_totals['total_spend']
FACT_15 = f"${spend_totals['downstream_spend'] / 1e9:.2f}B"
FACT_15_RAW = spend_totals['downstream_spend']
FACT_16 = f"{spend_totals['downstream_visits']:,.0f}"

display(Markdown(f"""
| FACT | Value |
|---|---|
| FACT-12 Total spend | {FACT_12} |
| FACT-15 Downstream spend (T180) | {FACT_15} |
| FACT-16 Downstream visits | {FACT_16} |
"""))

# ── Query 2: Spend per specialty (for quadrant VIS-11) ────────
fact_32 = client.query(f"""
    SELECT
        s.grouping_code                                  AS ending_specialty
        ,s.grouping_desc                                 AS specialty_desc
        ,SUM(s.total_allowed_amt)                        AS total_allowed_amt
        ,SUM(s.visit_count)                              AS visit_count
        ,SUM(s.unique_members)                           AS member_count
    FROM `{DS}.A870800_gen_rec_f_spend_summary` s
    WHERE s.summary_type = 'specialty'
      AND s.visit_type IN ('downstream', 'v2')
      AND s.grouping_code IS NOT NULL
      AND s.grouping_code != ''
    GROUP BY 1, 2
    ORDER BY total_allowed_amt DESC
""").to_dataframe()

display(Markdown("### FACT-32: Spend per Specialty (for quadrant chart)"))
display(fact_32)

# ── Query 3: Penetration rates for top conditions ─────────────
# Uses top 5 conditions from FACT-19 (requires Block 3 to have run)
try:
    top5_dx = fact_19_top10.head(5)["trigger_dx"].tolist()
    dx_list = ", ".join([f"'{dx}'" for dx in top5_dx])

    fact_20 = client.query(f"""
        SELECT
            trigger_dx
            ,trigger_ccsr_desc
            ,next_specialty
            ,next_specialty_desc
            ,transition_count
            ,unique_members
        FROM `{DS}.A870800_gen_rec_markov_train`
        WHERE trigger_dx IN ({dx_list})
          AND next_specialty IS NOT NULL
        ORDER BY trigger_dx, transition_count DESC
    """).to_dataframe()

    display(Markdown("### FACT-20: Top Condition Referral Patterns"))
    for dx in top5_dx:
        sub = fact_20[fact_20["trigger_dx"] == dx].head(5)
        desc = sub.iloc[0]["trigger_ccsr_desc"] if len(sub) > 0 else dx
        display(Markdown(f"**{desc} ({dx}):**"))
        display(sub[["next_specialty_desc", "transition_count", "unique_members"]])
except NameError:
    print("WARNING: fact_19_top10 not found — run Block 3 first")
    fact_20 = None

# ── FACT-26: Approximate dollar value for high-confidence ─────
# Rough: proportion of downstream spend attributable to
# conditions where hit_at_3 > average
try:
    high_conf_dx = fact_19_raw[fact_19_raw["hit_at_3"] > fact_19_raw["hit_at_3"].median()]["trigger_dx"].tolist()
    high_conf_pct = len(high_conf_dx) / len(fact_19_raw)
    FACT_26_RAW = FACT_15_RAW * high_conf_pct
    FACT_26 = f"${FACT_26_RAW / 1e9:.2f}B"
    display(Markdown(f"""
### FACT-26: High-Confidence Dollar Value (approx)
- {len(high_conf_dx)} of {len(fact_19_raw)} conditions above median Hit@3
- Estimated downstream spend in high-confidence pathways: {FACT_26}
- Note: approximation — assumes spend proportional to condition count
"""))
except NameError:
    print("WARNING: fact_19_raw not found — run Block 3 first")
    FACT_26 = "TBD"

print("Block 4 done.")
