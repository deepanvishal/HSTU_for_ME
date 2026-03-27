# ============================================================
# Block 3 — Model Performance Facts (K=5)
# FACT-03: Hit@5 at T30/T60/T180 — Markov
# FACT-04: Hit@5 at T30/T60/T180 — best approach
# FACT-05: Hit@5 by ending specialty at T30
# FACT-06: Overall average Hit@5 at T30
# FACT-18: Specialties above/below average
# FACT-19: Pathway consistency top/bottom 10 conditions
# FACT-22: Hit@5 by member_segment
# FACT-33: Specialty code → name mapping
# FACT-38: All models Hit@5 at T30
# Sources: analysis_perf_overall, analysis_perf_by_ending_specialty,
#          analysis_perf_by_diag, markov_train
# ============================================================
from IPython.display import display, Markdown

# ── Query 1: Overall performance — all models, all windows ────
perf_overall = client.query(f"""
    SELECT model, time_bucket, member_segment
        ,hit_at_5, ndcg_at_5, n_triggers
    FROM `{DS}.A870800_gen_rec_analysis_perf_overall`
    ORDER BY model, time_bucket, member_segment
""").to_dataframe()

all_seg = perf_overall[perf_overall["member_segment"] == "ALL"]

# FACT-38: All models at T30
fact_38 = all_seg[all_seg["time_bucket"] == "T0_30"][["model", "hit_at_5"]].copy()
display(Markdown("### FACT-38: All Models Hit@5 at T30"))
display(fact_38)

# FACT-03: Markov per window
markov = all_seg[all_seg["model"] == "Markov"].set_index("time_bucket")["hit_at_5"]
FACT_03_T30 = f"{markov.get('T0_30', 0) * 100:.1f}"
FACT_03_T60 = f"{markov.get('T30_60', 0) * 100:.1f}"
FACT_03_T180 = f"{markov.get('T60_180', 0) * 100:.1f}"

# FACT-04: Best approach per window
best_model = fact_38.loc[fact_38["hit_at_5"].idxmax(), "model"]
best = all_seg[all_seg["model"] == best_model].set_index("time_bucket")["hit_at_5"]
FACT_04_T30 = f"{best.get('T0_30', 0) * 100:.1f}"
FACT_04_T60 = f"{best.get('T30_60', 0) * 100:.1f}"
FACT_04_T180 = f"{best.get('T60_180', 0) * 100:.1f}"

# FACT-06: Overall average Hit@5 at T30 (best model)
FACT_06 = FACT_04_T30

# FACT-22: Hit@5 by segment at T30 (best model)
fact_22 = (perf_overall[
    (perf_overall["model"] == best_model)
    & (perf_overall["time_bucket"] == "T0_30")
    & (perf_overall["member_segment"] != "ALL")
][["member_segment", "hit_at_5", "n_triggers"]]
.sort_values("hit_at_5", ascending=False)
.copy())

display(Markdown(f"""
| FACT | Value |
|---|---|
| FACT-03 Markov T30 / T60 / T180 | {FACT_03_T30}% / {FACT_03_T60}% / {FACT_03_T180}% |
| FACT-04 Best ({best_model}) T30 / T60 / T180 | {FACT_04_T30}% / {FACT_04_T60}% / {FACT_04_T180}% |
| FACT-06 Overall Hit@5 T30 | {FACT_06}% |
"""))
display(Markdown("### FACT-22: Hit@5 by Segment at T30"))
display(fact_22)

# ── Query 2: Performance by ending specialty at T30 ───────────
fact_05_raw = client.query(f"""
    SELECT
        ending_specialty
        ,total_appearances
        ,predicted_at_5
        ,ROUND(SAFE_DIVIDE(predicted_at_5, total_appearances), 4) AS hit_rate_at_5
        ,avg_ndcg_at_3 AS avg_ndcg
    FROM `{DS}.A870800_gen_rec_analysis_perf_by_ending_specialty`
    WHERE time_bucket = 'T0_30'
      AND model = '{best_model}'
      AND total_appearances >= 20
    ORDER BY hit_rate_at_5 DESC
""").to_dataframe()

# FACT-33: Specialty descriptions
fact_33 = client.query(f"""
    SELECT DISTINCT
        next_specialty                                   AS specialty_code
        ,next_specialty_desc                             AS specialty_name
    FROM `{DS}.A870800_gen_rec_markov_train`
    WHERE next_specialty IS NOT NULL
""").to_dataframe()

fact_05 = fact_05_raw.merge(
    fact_33, left_on="ending_specialty", right_on="specialty_code", how="left"
)
fact_05["display_name"] = fact_05["specialty_name"].fillna(fact_05["ending_specialty"])

FACT_05_TOP1 = fact_05.iloc[0]["display_name"]
FACT_05_TOP1_VAL = f"{fact_05.iloc[0]['hit_rate_at_5'] * 100:.1f}"
FACT_05_BOT1 = fact_05.iloc[-1]["display_name"]
FACT_05_BOT1_VAL = f"{fact_05.iloc[-1]['hit_rate_at_5'] * 100:.1f}"

avg_hit5 = fact_05["hit_rate_at_5"].mean()
FACT_18_ABOVE = len(fact_05[fact_05["hit_rate_at_5"] > avg_hit5])
FACT_18_BELOW = len(fact_05[fact_05["hit_rate_at_5"] <= avg_hit5])

display(Markdown(f"""
### FACT-05: Hit@5 by Ending Specialty at T30
| Position | Specialty | Hit@5 |
|---|---|---|
| Top 1 | {FACT_05_TOP1} | {FACT_05_TOP1_VAL}% |
| Bottom 1 | {FACT_05_BOT1} | {FACT_05_BOT1_VAL}% |

### FACT-18: Specialties Above/Below Average
- Above average: {FACT_18_ABOVE}
- Below average: {FACT_18_BELOW}
"""))
display(fact_05[["display_name", "total_appearances", "hit_rate_at_5"]])

# ── Query 3: Performance by diagnosis — pathway consistency ───
fact_19_raw = client.query(f"""
    SELECT
        d.trigger_dx
        ,d.trigger_volume
        ,d.hit_at_5
        ,m.trigger_ccsr
        ,m.trigger_ccsr_desc
    FROM `{DS}.A870800_gen_rec_analysis_perf_by_diag` d
    LEFT JOIN (
        SELECT DISTINCT trigger_dx, trigger_ccsr, trigger_ccsr_desc
        FROM `{DS}.A870800_gen_rec_markov_train`
    ) m ON d.trigger_dx = m.trigger_dx
    WHERE d.time_bucket = 'T0_30'
      AND d.model = '{best_model}'
      AND d.trigger_volume >= 20
    ORDER BY d.hit_at_5 DESC
""").to_dataframe()

fact_19_raw["display_name"] = fact_19_raw["trigger_ccsr_desc"].fillna(fact_19_raw["trigger_dx"])

fact_19_top10 = fact_19_raw.head(10)
fact_19_bot10 = fact_19_raw.tail(10)

FACT_19_TOP1 = fact_19_top10.iloc[0]["display_name"]
FACT_19_TOP1_VAL = f"{fact_19_top10.iloc[0]['hit_at_5'] * 100:.1f}"
FACT_19_BOT1 = fact_19_bot10.iloc[-1]["display_name"]
FACT_19_BOT1_VAL = f"{fact_19_bot10.iloc[-1]['hit_at_5'] * 100:.1f}"

display(Markdown(f"""
### FACT-19: Pathway Consistency (Hit@5)
| Position | Condition | Hit@5 |
|---|---|---|
| Most consistent | {FACT_19_TOP1} | {FACT_19_TOP1_VAL}% |
| Least consistent | {FACT_19_BOT1} | {FACT_19_BOT1_VAL}% |
"""))
display(Markdown("**Top 10:**"))
display(fact_19_top10[["display_name", "trigger_volume", "hit_at_5"]])
display(Markdown("**Bottom 10:**"))
display(fact_19_bot10[["display_name", "trigger_volume", "hit_at_5"]])

print(f"\nBest model: {best_model}")
print("Block 3 done.")
