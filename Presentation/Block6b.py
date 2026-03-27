# ============================================================
# Block 6b — Avg Visits + Consistency Analysis (K=5)
# 1. Average distinct specialties per trigger (justifies K)
# 2. Hit@5 all models, all windows, all segments
# 3. Consistency: by segment, dx volume tier, ending specialty
# Requires: client, DS, best_model from Block 3
# Source: analysis_perf_full, triggers_qualified, visits_qualified,
#         analysis_perf_by_diag, analysis_perf_by_ending_specialty
# ============================================================
from IPython.display import display, Markdown
import pandas as pd

# ── 1. AVERAGE VISITS PER TRIGGER (justifies K=5) ────────────
avg_visits = client.query(f"""
    SELECT
        AVG(specialty_count)                             AS avg_specialties
        ,APPROX_QUANTILES(specialty_count, 100)[OFFSET(50)] AS median
        ,APPROX_QUANTILES(specialty_count, 100)[OFFSET(75)] AS p75
        ,APPROX_QUANTILES(specialty_count, 100)[OFFSET(90)] AS p90
        ,MAX(specialty_count)                            AS max_val
    FROM (
        SELECT
            t.member_id
            ,t.trigger_date
            ,t.trigger_dx
            ,COUNT(DISTINCT v.specialty_ctg_cd)          AS specialty_count
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
| Mean | {float(avg_visits['avg_specialties']):.1f} |
| Median | {int(avg_visits['median'])} |
| P75 | {int(avg_visits['p75'])} |
| P90 | {int(avg_visits['p90'])} |
| Max | {int(avg_visits['max_val'])} |

**K=5 justification:** Median member visits {int(avg_visits['median'])} distinct specialties within 180 days of a new diagnosis. P75 is {int(avg_visits['p75'])}. K=5 covers the majority of actual visit patterns.
"""))


# ── 2. HIT@5 — ALL MODELS, ALL WINDOWS, ALL SEGMENTS ─────────
hit5_full = client.query(f"""
    SELECT model, time_bucket, member_segment
        ,n_triggers, hit_at_5, ndcg_at_5
    FROM `{DS}.A870800_gen_rec_analysis_perf_full`
    ORDER BY model, time_bucket, member_segment
""").to_dataframe()

display(Markdown("### Hit@5 — Overall (ALL segments)"))
overall = hit5_full[
    (hit5_full["member_segment"] == "ALL")
    & (hit5_full["time_bucket"] == "T0_30")
][["model", "hit_at_5", "n_triggers"]]
display(overall)

display(Markdown("### Hit@5 — All Segments at T30"))
t30 = hit5_full[hit5_full["time_bucket"] == "T0_30"].copy()
display(t30[["model", "member_segment", "hit_at_5", "n_triggers"]])


# ── 3. CONSISTENCY ANALYSIS ───────────────────────────────────

# 3a. Hit@5 by segment per model at T30
display(Markdown("---\n### 3a. Consistency — Hit@5 by Segment (T30)"))
seg_data = t30[t30["member_segment"] != "ALL"].copy()
seg_pivot = seg_data.pivot(index="model", columns="member_segment", values="hit_at_5")
seg_pivot["range"] = seg_pivot.max(axis=1) - seg_pivot.min(axis=1)
seg_pivot["std"] = seg_pivot.drop(columns="range").std(axis=1)
display(seg_pivot.round(4))

# 3b. Hit@5 by diagnosis volume tier
display(Markdown("---\n### 3b. Consistency — Hit@5 by Diagnosis Volume Tier (T30)"))
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
display(dx_consistency)

# 3c. Hit@5 by ending specialty — variance per model
display(Markdown("---\n### 3c. Consistency — Hit@5 by Ending Specialty (T30)"))
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
display(spec_consistency)

# ── 3d. Summary: rank models by consistency ───────────────────
display(Markdown("---\n### 3d. Consistency Summary"))

models_list = seg_pivot.index.tolist()
summary_rows = []
for model in models_list:
    seg_std = seg_pivot.loc[model, "std"] if model in seg_pivot.index else None
    dx_row = dx_consistency[dx_consistency["model"] == model]
    dx_std = dx_row["std_hit_at_5"].mean() if len(dx_row) > 0 else None
    spec_row = spec_consistency[spec_consistency["model"] == model]
    spec_std = float(spec_row["std_hit_rate_5"].values[0]) if len(spec_row) > 0 else None
    avg_std = sum(filter(None, [seg_std, dx_std, spec_std])) / sum(1 for x in [seg_std, dx_std, spec_std] if x is not None)
    summary_rows.append({
        "model": model,
        "segment_std": round(seg_std, 4) if seg_std else None,
        "dx_volume_std": round(dx_std, 4) if dx_std else None,
        "specialty_std": round(spec_std, 4) if spec_std else None,
        "avg_std": round(avg_std, 4)
    })

summary_df = pd.DataFrame(summary_rows).sort_values("avg_std")
display(summary_df)

most_consistent = summary_df.iloc[0]["model"]
display(Markdown(f"""
**Most consistent model: {most_consistent}**
Lowest average standard deviation across segments, diagnosis volume tiers, and ending specialties.
"""))

print("Block 6b done.")
