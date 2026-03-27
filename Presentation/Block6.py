# ============================================================
# Block 6 — Narrative Corrections
#
# 1. Dollar funnel: total claims $ → qualified trigger pairs $ → downstream $
# 2. Switch to Hit@5
# 3. Model consistency across segments, dx volume, specialties
# 4. Average visits following a trigger (justifies K)
#
# Requires: client, DS from Block 1
# ============================================================
from IPython.display import display, Markdown
import pandas as pd

# ── 1. DOLLAR FUNNEL (corrected) ──────────────────────────────
# Level 1: Total $ across ALL claims
# Level 2: $ for claims where the member+date is a qualified trigger
# Level 3: $ for downstream visits within T180 of those triggers

funnel = client.query(f"""
    WITH
    -- Total claims spend
    total AS (
        SELECT
            SUM(CAST(alw_amt AS FLOAT64))                AS total_spend
            ,COUNT(*)                                    AS total_claims
        FROM `{DS}.A870800_claims_gen_rec_2022_2025_sfl`
    ),
    -- Trigger-day spend: claims on trigger_date for qualified triggers
    trigger_spend AS (
        SELECT
            SUM(CAST(c.alw_amt AS FLOAT64))              AS trigger_day_spend
            ,COUNT(*)                                    AS trigger_day_claims
        FROM `{DS}.A870800_gen_rec_triggers_qualified` t
        JOIN `{DS}.A870800_claims_gen_rec_2022_2025_sfl` c
            ON t.member_id = c.member_id
            AND t.trigger_date = c.srv_start_dt
        WHERE t.is_left_qualified = TRUE
    ),
    -- Downstream spend: claims within T180 after trigger
    downstream_spend AS (
        SELECT
            SUM(CAST(c.alw_amt AS FLOAT64))              AS downstream_spend
            ,COUNT(*)                                    AS downstream_claims
        FROM `{DS}.A870800_gen_rec_triggers_qualified` t
        JOIN `{DS}.A870800_claims_gen_rec_2022_2025_sfl` c
            ON t.member_id = c.member_id
            AND c.srv_start_dt > t.trigger_date
            AND c.srv_start_dt <= DATE_ADD(t.trigger_date, INTERVAL 180 DAY)
        WHERE t.is_left_qualified = TRUE
          AND t.is_t180_qualified = TRUE
    )
    SELECT
        t.total_spend
        ,t.total_claims
        ,ts.trigger_day_spend
        ,ts.trigger_day_claims
        ,ds.downstream_spend
        ,ds.downstream_claims
    FROM total t, trigger_spend ts, downstream_spend ds
""").to_dataframe().iloc[0]

FUNNEL_TOTAL = f"${float(funnel['total_spend']) / 1e9:.1f}B"
FUNNEL_TRIGGER = f"${float(funnel['trigger_day_spend']) / 1e9:.1f}B"
FUNNEL_DOWNSTREAM = f"${float(funnel['downstream_spend']) / 1e9:.1f}B"

display(Markdown(f"""
### Corrected Dollar Funnel
| Level | Spend | Claims |
|---|---|---|
| All claims | {FUNNEL_TOTAL} | {float(funnel['total_claims']):,.0f} |
| Trigger-day claims | {FUNNEL_TRIGGER} | {float(funnel['trigger_day_claims']):,.0f} |
| Downstream within T180 | {FUNNEL_DOWNSTREAM} | {float(funnel['downstream_claims']):,.0f} |
"""))


# ── 2. AVERAGE VISITS PER TRIGGER (justifies K) ──────────────
avg_visits = client.query(f"""
    SELECT
        AVG(visit_count)                                 AS avg_visits
        ,APPROX_QUANTILES(visit_count, 100)[OFFSET(50)] AS median_visits
        ,APPROX_QUANTILES(visit_count, 100)[OFFSET(75)] AS p75_visits
        ,APPROX_QUANTILES(visit_count, 100)[OFFSET(90)] AS p90_visits
        ,MAX(visit_count)                                AS max_visits
    FROM (
        SELECT
            t.member_id
            ,t.trigger_date
            ,t.trigger_dx
            ,COUNT(DISTINCT v.specialty_ctg_cd)          AS visit_count
        FROM `{DS}.A870800_gen_rec_triggers_qualified` t
        JOIN `{DS}.A870800_gen_rec_visits_qualified` v
            ON t.member_id = v.member_id
            AND t.trigger_date = v.trigger_date
            AND t.trigger_dx = v.trigger_dx
        WHERE t.is_left_qualified = TRUE
          AND t.is_t180_qualified = TRUE
          AND v.days_since_trigger > 0
          AND v.days_since_trigger <= 180
        GROUP BY 1, 2, 3
    )
""").to_dataframe().iloc[0]

display(Markdown(f"""
### Average Distinct Specialties Visited After Trigger (T180)
| Stat | Value |
|---|---|
| Mean | {float(avg_visits['avg_visits']):.1f} |
| Median | {int(avg_visits['median_visits'])} |
| P75 | {int(avg_visits['p75_visits'])} |
| P90 | {int(avg_visits['p90_visits'])} |
| Max | {int(avg_visits['max_visits'])} |
"""))


# ── 3. HIT@5 — ALL MODELS, ALL WINDOWS, ALL SEGMENTS ─────────
hit5_overall = client.query(f"""
    SELECT model, time_bucket, member_segment
        ,hit_at_5, ndcg_at_5, n_triggers
    FROM `{DS}.A870800_gen_rec_analysis_perf_overall`
    ORDER BY model, time_bucket, member_segment
""").to_dataframe()

display(Markdown("### Hit@5 — All Models, All Segments, T30"))
t30 = hit5_overall[hit5_overall["time_bucket"] == "T0_30"].copy()
display(t30[["model", "member_segment", "hit_at_5", "n_triggers"]])

display(Markdown("### Hit@5 — Overall (ALL segments)"))
all_seg = t30[t30["member_segment"] == "ALL"][["model", "hit_at_5"]]
display(all_seg)


# ── 4. MODEL CONSISTENCY ANALYSIS ─────────────────────────────
# Across: segments, diagnosis volume tiers, ending specialties

# 4a. Hit@5 by segment per model at T30
seg_pivot = (t30[t30["member_segment"] != "ALL"]
    .pivot(index="model", columns="member_segment", values="hit_at_5"))
seg_pivot["range"] = seg_pivot.max(axis=1) - seg_pivot.min(axis=1)
seg_pivot["std"] = seg_pivot.drop(columns="range").std(axis=1)

display(Markdown("### 4a. Hit@5 by Segment (T30) — Consistency"))
display(seg_pivot)

# 4b. Hit@5 by diagnosis volume tier
dx_consistency = client.query(f"""
    WITH volume_tiers AS (
        SELECT
            model
            ,trigger_dx
            ,trigger_volume
            ,hit_at_5
            ,CASE
                WHEN trigger_volume >= 1000 THEN 'High (1000+)'
                WHEN trigger_volume >= 100  THEN 'Med (100-999)'
                ELSE 'Low (20-99)'
            END                                          AS volume_tier
        FROM `{DS}.A870800_gen_rec_analysis_perf_by_diag`
        WHERE time_bucket = 'T0_30'
          AND trigger_volume >= 20
    )
    SELECT
        model
        ,volume_tier
        ,COUNT(*)                                        AS n_dx_codes
        ,ROUND(AVG(hit_at_5), 4)                         AS avg_hit_at_5
        ,ROUND(STDDEV(hit_at_5), 4)                      AS std_hit_at_5
    FROM volume_tiers
    GROUP BY model, volume_tier
    ORDER BY model,
        CASE volume_tier
            WHEN 'High (1000+)' THEN 1
            WHEN 'Med (100-999)' THEN 2
            ELSE 3
        END
""").to_dataframe()

display(Markdown("### 4b. Hit@5 by Diagnosis Volume Tier (T30)"))
display(dx_consistency)

# 4c. Hit@5 by ending specialty — variance per model
spec_consistency = client.query(f"""
    SELECT
        model
        ,COUNT(*)                                        AS n_specialties
        ,ROUND(AVG(hit_rate_at_5), 4)                    AS avg_hit_rate_5
        ,ROUND(STDDEV(hit_rate_at_5), 4)                 AS std_hit_rate_5
        ,ROUND(MIN(hit_rate_at_5), 4)                    AS min_hit_rate_5
        ,ROUND(MAX(hit_rate_at_5), 4)                    AS max_hit_rate_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_by_ending_specialty`
    WHERE time_bucket = 'T0_30'
      AND total_appearances >= 20
    GROUP BY model
    ORDER BY std_hit_rate_5 ASC
""").to_dataframe()

display(Markdown("### 4c. Hit@5 by Ending Specialty — Variance per Model (T30)"))
display(spec_consistency)

# 4d. Summary: which model is most consistent?
display(Markdown("""
### Consistency Summary
**Most consistent model** = lowest standard deviation across segments, dx volume tiers, and ending specialties.
Review the three tables above. The model with the smallest range and std across all three dimensions is the most reliable for deployment.
"""))

print("Block 6 done.")
