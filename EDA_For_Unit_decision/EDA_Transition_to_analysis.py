# ============================================================
# NB_05 — Transition to Analysis
# Purpose : Bridge from data setup to EDA and modeling.
#           Summarizes the qualified population and defines
#           the three analytical lenses used in all downstream
#           analysis and model training.
# Sources : A870800_gen_rec_triggers_qualified
# ============================================================
from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, Markdown

client = bigquery.Client(project="anbc-hcb-dev")
DATASET = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
TQ = f"`{DATASET}.A870800_gen_rec_triggers_qualified`"

def fmt_count(x):
    return f"{int(x):,}"

def fmt_pct(x):
    return f"{x:.1f}%"


display(Markdown("""
---
# NB 05 — Transition to Analysis
## From Data Setup to EDA and Modeling

This notebook closes the data setup phase and bridges into the
analytical phase of the project.

Notebooks NB_01 through NB_04 established:
- The raw data characteristics of the South Florida market
- The boundary rules that define a valid trigger
- The quantitative impact of those rules on the population
- The profile of the qualified population

This notebook summarizes those findings in one place and defines
the three analytical lenses used in all downstream EDA and modeling.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — POPULATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 1 — Population Summary

A consolidated view of the qualified population across all time windows.
"""))

summary = client.query(f"""
SELECT
    COUNT(*)                                             AS all_first_encounters
    ,COUNTIF(is_left_qualified = TRUE)                   AS left_qualified
    ,COUNTIF(is_t30_qualified = TRUE)                    AS t30_qualified
    ,COUNTIF(is_t60_qualified = TRUE)                    AS t60_qualified
    ,COUNTIF(is_t180_qualified = TRUE)                   AS t180_qualified
    ,COUNTIF(is_t180_qualified = TRUE
        AND has_claims_12m_before = TRUE)                AS t180_model_ready
    ,COUNT(DISTINCT member_id)                           AS all_members
    ,COUNT(DISTINCT CASE WHEN is_left_qualified = TRUE
        THEN member_id END)                              AS left_qualified_members
    ,COUNT(DISTINCT CASE WHEN is_t30_qualified = TRUE
        THEN member_id END)                              AS t30_members
    ,COUNT(DISTINCT CASE WHEN is_t60_qualified = TRUE
        THEN member_id END)                              AS t60_members
    ,COUNT(DISTINCT CASE WHEN is_t180_qualified = TRUE
        THEN member_id END)                              AS t180_members
    ,COUNT(DISTINCT CASE WHEN is_t180_qualified = TRUE
        AND has_claims_12m_before = TRUE
        THEN member_id END)                              AS t180_model_ready_members
FROM {TQ}
""").to_dataframe()

s = summary.iloc[0]
base = s["all_first_encounters"]

display(Markdown(f"""
| Stage | Triggers | Members | % of All First Encounters |
|---|---|---|---|
| All first encounters | {fmt_count(s['all_first_encounters'])} | {fmt_count(s['all_members'])} | 100% |
| Left qualified | {fmt_count(s['left_qualified'])} | {fmt_count(s['left_qualified_members'])} | {fmt_pct(s['left_qualified']/base*100)} |
| T30 qualified | {fmt_count(s['t30_qualified'])} | {fmt_count(s['t30_members'])} | {fmt_pct(s['t30_qualified']/base*100)} |
| T60 qualified | {fmt_count(s['t60_qualified'])} | {fmt_count(s['t60_members'])} | {fmt_pct(s['t60_qualified']/base*100)} |
| T180 qualified | {fmt_count(s['t180_qualified'])} | {fmt_count(s['t180_members'])} | {fmt_pct(s['t180_qualified']/base*100)} |
| T180 model ready | {fmt_count(s['t180_model_ready'])} | {fmt_count(s['t180_model_ready_members'])} | {fmt_pct(s['t180_model_ready']/base*100)} |
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — THREE ANALYTICAL LENSES
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 2 — Three Analytical Lenses

All downstream EDA and transition analysis is conducted through
three distinct analytical lenses. Each lens asks a slightly different
question and uses a different subset of the data.

---

### Lens 1 — Any Sequential Visits

**Source:** All visits in the data — no trigger filter applied
**Grain:** Any two consecutive visits for any member
**Question:** Across all visits in the population, what diagnosis
typically precedes what specialty?

**Use:** Establishes baseline transition patterns across the full
claims population. Serves as the broadest reference point.

---

### Lens 2 — First Encounter of Diagnosis

**Source:** Left-qualified triggers + immediate next visit
**Grain:** First occurrence of each ICD-10 code per member → next visit
**Question:** When a member presents with a new diagnosis for the first time,
where do they go next?

**Use:** The primary analytical lens. Captures new diagnosis routing
decisions — the most clinically meaningful signal for prediction.

---

### Lens 3 — FP/I First Encounter

**Source:** Left-qualified triggers where trigger specialty is FP or I + next visit
**Grain:** First occurrence of each ICD-10 code at a primary care visit → next visit
**Question:** When a primary care physician diagnoses a condition for the first time,
where does the member get referred next?

**Use:** The most clinically focused lens. Primary care referral decisions
represent the clearest care navigation signal — the physician is actively
routing the member to a specialist.

---
"""))

# lens population sizes
lens_counts = client.query(f"""
WITH any_pairs AS (
    SELECT COUNT(*) AS pair_count
    FROM `{DATASET}.A870800_gen_rec_visits` v1
    JOIN `{DATASET}.A870800_gen_rec_visits` v2
        ON v1.member_id = v2.member_id
        AND v2.visit_rank = v1.visit_rank + 1
),
fe_pairs AS (
    SELECT COUNT(*) AS pair_count
    FROM {TQ} t
    JOIN `{DATASET}.A870800_gen_rec_visits_qualified` v
        ON t.member_id = v.member_id
        AND t.trigger_date = v.trigger_date
        AND t.trigger_dx = v.trigger_dx
        AND v.is_v2 = TRUE
    WHERE t.is_left_qualified = TRUE
),
fp_pairs AS (
    SELECT COUNT(*) AS pair_count
    FROM {TQ} t
    JOIN `{DATASET}.A870800_gen_rec_visits_qualified` v
        ON t.member_id = v.member_id
        AND t.trigger_date = v.trigger_date
        AND t.trigger_dx = v.trigger_dx
        AND v.is_v2 = TRUE
    WHERE t.is_left_qualified = TRUE
      AND t.trigger_specialty IN ('FP', 'I')
)
SELECT
    (SELECT pair_count FROM any_pairs)   AS any_visits_pairs
    ,(SELECT pair_count FROM fe_pairs)   AS first_encounter_pairs
    ,(SELECT pair_count FROM fp_pairs)   AS fp_first_pairs
""").to_dataframe()

lc = lens_counts.iloc[0]

display(Markdown(f"""
### Lens Population Sizes

| Lens | Transition Pairs |
|---|---|
| Any sequential visits | {fmt_count(lc['any_visits_pairs'])} |
| First encounter of diagnosis | {fmt_count(lc['first_encounter_pairs'])} |
| FP/I first encounter | {fmt_count(lc['fp_first_pairs'])} |
"""))

fig, ax = plt.subplots(figsize=(12, 5))
lenses = ["Any Sequential\nVisits", "First Encounter\nof Diagnosis", "FP/I First\nEncounter"]
counts = [lc["any_visits_pairs"], lc["first_encounter_pairs"], lc["fp_first_pairs"]]
colors = ["#CCCCCC", "#4C9BE8", "#F4845F"]
bars = ax.bar(lenses, counts, color=colors, alpha=0.85, width=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            fmt_count(count), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Transition Pairs", fontsize=10)
ax.set_title("Transition Pairs Available Per Analytical Lens",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("lens_populations.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — THREE TIME WINDOWS
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 3 — Three Time Windows

All transition analysis and model training uses three incremental
time windows after the trigger date.

| Window | Definition | Trigger Cutoff | Question |
|---|---|---|---|
| T0_30 | Days 1 to 30 after trigger | Trigger on or before Nov 30 2025 | Where does the member go in the first month? |
| T30_60 | Days 31 to 60 after trigger | Trigger on or before Oct 31 2025 | Where does the member go in the second month? |
| T60_180 | Days 61 to 180 after trigger | Trigger on or before Jun 30 2025 | Where does the member go in months 2 to 6? |

**Windows are incremental — not cumulative.**
A visit on day 45 belongs to T30_60 only — not to T0_30 or T60_180.

**Partial qualification is invalid.**
A trigger that qualifies for T0_30 but not T30_60 is excluded from
T30_60 analysis entirely.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WHAT COMES NEXT
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 4 — What Comes Next

The data setup phase is complete. The following analysis phases build on
the qualified population defined in NB_01 through NB_04.

---

### EDA Phase

**Entropy Analysis**
Compute weighted average entropy for all 9 transition combinations
(DX to Specialty, DX to DX, DX to CCSR, Specialty to Specialty, etc.)
across all three lenses. This justifies DX to Specialty as the
prediction target with the strongest and most consistent signal.

**DX to Specialty Transition Tables**
Build three transition tables — one per analytical lens — with
conditional probability, entropy, and cost metrics per transition pair.

**EDA Notebooks**
Three Python notebooks — one per lens — with entropy bar charts,
transition heatmaps, bipartite network graphs, and cost analysis.

**Pareto and Cost Analysis**
Identify the DX to Specialty transitions that drive 80% of total spend.
Surface the most impactful diagnoses and specialties by volume and cost.

---

### Modeling Phase

**Markov Baseline**
Order 1 and Order 2 Markov chain transition models.
Train on pre-2024 data, evaluate on 2024 and beyond.
Establishes the baseline prediction accuracy to beat.

**Sequential Models**
SASRec → BERT4Rec → HSTU → HSTU with reinforcement learning.
Each model reads from the pre-trigger visit sequence in
A870800_gen_rec_model_input_sequences and predicts label_specialty
per time bucket.

**Evaluation**
Hit@K, Precision@K, Recall@K, NDCG@K for K = 1, 3, 5
across T0_30, T30_60, and T60_180 windows.

---
"""))
