# ============================================================
# Block 8b — Diagnosis Density on Trigger Day
# How many distinct diagnoses accompany a qualified trigger?
# Step 1: qualified triggers (is_left + is_t180)
# Step 2: join SFL on member + date, count distinct pri_icd9_dx_cd
# Requires: client, DS
# ============================================================
from IPython.display import display, Markdown

dx_density = client.query(f"""
    WITH qualified AS (
        SELECT DISTINCT member_id, trigger_date
        FROM `{DS}.A870800_gen_rec_triggers_qualified`
        WHERE is_left_qualified = TRUE
          AND is_t180_qualified = TRUE
    ),
    joined AS (
        SELECT
            q.member_id
            ,q.trigger_date
            ,COUNT(DISTINCT REPLACE(TRIM(c.pri_icd9_dx_cd), '.', '')) AS n_dx
        FROM qualified q
        JOIN `{DS}.A870800_claims_gen_rec_2022_2025_sfl` c
            ON q.member_id = c.member_id
            AND q.trigger_date = c.srv_start_dt
        WHERE c.pri_icd9_dx_cd IS NOT NULL
          AND TRIM(c.pri_icd9_dx_cd) != ''
        GROUP BY 1, 2
    )
    SELECT
        n_dx
        ,COUNT(*) AS n_trigger_days
    FROM joined
    GROUP BY 1
    ORDER BY 1
""").to_dataframe()

display(Markdown("### Diagnosis Density on Trigger Day"))
display(dx_density)

total = dx_density["n_trigger_days"].sum()
stats = client.query(f"""
    WITH qualified AS (
        SELECT DISTINCT member_id, trigger_date
        FROM `{DS}.A870800_gen_rec_triggers_qualified`
        WHERE is_left_qualified = TRUE
          AND is_t180_qualified = TRUE
    ),
    joined AS (
        SELECT q.member_id, q.trigger_date
            ,COUNT(DISTINCT REPLACE(TRIM(c.pri_icd9_dx_cd), '.', '')) AS n_dx
        FROM qualified q
        JOIN `{DS}.A870800_claims_gen_rec_2022_2025_sfl` c
            ON q.member_id = c.member_id
            AND q.trigger_date = c.srv_start_dt
        WHERE c.pri_icd9_dx_cd IS NOT NULL AND TRIM(c.pri_icd9_dx_cd) != ''
        GROUP BY 1, 2
    )
    SELECT
        ROUND(AVG(n_dx), 2) AS mean_dx
        ,APPROX_QUANTILES(n_dx, 100)[OFFSET(50)] AS median_dx
        ,APPROX_QUANTILES(n_dx, 100)[OFFSET(75)] AS p75_dx
        ,APPROX_QUANTILES(n_dx, 100)[OFFSET(90)] AS p90_dx
        ,MAX(n_dx) AS max_dx
        ,COUNT(*) AS total_trigger_days
    FROM joined
""").to_dataframe().iloc[0]

display(Markdown(f"""
### Summary Statistics
| Stat | Value |
|---|---|
| Total trigger-days | {int(stats['total_trigger_days']):,} |
| Mean diagnoses per trigger day | {float(stats['mean_dx']):.1f} |
| Median | {int(stats['median_dx'])} |
| P75 | {int(stats['p75_dx'])} |
| P90 | {int(stats['p90_dx'])} |
| Max | {int(stats['max_dx'])} |
"""))

print("Block 8b done.")
