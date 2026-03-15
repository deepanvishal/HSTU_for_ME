# Provider Network — Exploratory Data Analysis
## South Florida Claims Data — 2022 to 2025

---

## Purpose

This EDA investigates whether **predictive signal exists in claims data** to anticipate
where a member will seek care next, given their first encounter with a new diagnosis.

The EDA is not modeling. It is evidence gathering. Every analytical choice made in this
document is driven by one question:

> **Is there enough signal in the data to justify building a sequential prediction model —
> and if so, where is that signal strongest?**

The outputs of this EDA are **data decisions** — conclusions that directly determine
the scope, input features, target variable, and time window of the prediction model.

---

## Data Decisions This EDA Must Answer

1. **What is the right prediction target?**
   Diagnosis code, CCSR clinical category, or provider specialty?

2. **What is the right starting point?**
   Any first encounter, or only first encounters at a primary care provider?

3. **What is the right time window?**
   Immediate next visit, or a time-bounded window (T30, T60, T180)?

4. **What is the right scope?**
   All diagnosis codes, or a focused set of high-signal CCSR domains?

5. **Does member profile shift the signal?**
   Do routing patterns differ meaningfully by cohort for the same diagnosis?

---

## 1. Data Scope

**Source:** CVS/Aetna claims data 2022–2025
**Geography:** South Florida
**Dataset:** `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`

### 1.1 South Florida — Summary Statistics

| Metric | Count |
|---|---|
| Total unique providers | TBD |
| Total unique members | TBD |
| Total claims | TBD |
| Total visits | TBD |
| Visits with first encounter of a diagnosis (E) | TBD |
| (E) visits with at least one follow-up visit (F) | TBD |
| Immediate next visit after (E) | TBD |
| Visits within T30 days (G) | TBD |
| Visits within T60 days (H) | TBD |
| Visits within T180 days (I) | TBD |

### 1.2 Data Quality Coverage

| Metric | Coverage |
|---|---|
| Claims with valid ICD-10 codes | TBD |
| Claims with specialty codes | TBD |
| Claims with CCSR mappings | TBD |
| Members with at least 2 visits | TBD |

### 1.3 Findings

TBD

---

## 2. Clinical Grouping — CCSR

ICD-10 codes are grouped into approximately 530 CCSR clinical categories using AHRQ mapping.
CCSR is used **only within this EDA** to reduce dimensionality and identify which clinical
domains carry the strongest signal. It is not a modeling feature.

The CCSR mapping allows transition analysis at the domain level — surfacing which
clinical areas have predictable care pathways without being overwhelmed by the volume
of individual ICD-10 codes.

---

## 3. What Is the Right Prediction Target?

### Method
Nine transition combinations evaluated across three starting units
(Diagnosis Code, CCSR Category, Provider Specialty) and three prediction units
(Diagnosis Code, CCSR Category, Provider Specialty).

Signal measured using **conditional entropy**: `H = -SUM(p * LOG(p))`

Lower entropy = stronger predictive signal.

Only transitions with 100 or more occurrences included to ensure statistical reliability.

### Results

TBD

> Full results: `EDA_01_entropy_summary.ipynb`

### Decision

**Starting unit:** Diagnosis Code (ICD-10)
**Prediction unit:** Provider Specialty

**Rationale:**
- Lowest weighted average entropy across all cohorts
- Most clinically actionable — directly maps to provider matching and care navigation
- Diagnosis code is observable at the point of first encounter

---

## 4. Sequence Analysis — Three Analytical Lenses

### 4.1 Lens 1 — No Filters (All Encounters)

**Question:** Given any visit sequence, which transitions are highly predictable?

TBD

> Full results: `EDA_02_lens1_all_encounters.ipynb`

---

### 4.2 Lens 2 — First Encounter of a Diagnosis

**Question:** Given a member's first encounter with a new diagnosis, which next visit
sequences are highly predictable?

**Trigger definition:** First date a member presents with a specific ICD-10 code.

TBD

> Full results: `EDA_03_lens2_first_encounter.ipynb`

---

### 4.3 Lens 3 — First Encounter at a Primary Care Provider

**Question:** Given that the first encounter occurred at a Family Practice or Internal
Medicine provider, which next visit sequences are highly predictable?

**Trigger filter:** `trigger_specialty IN ('FP', 'I')`

**Rationale:** FP/I visits represent the most clinically coherent entry point —
the member came to a generalist first, who then directed care forward.
Non-PCP triggers (specialist-first, ER, lab-first) represent noisier entry points
where the care pathway is already in progress.

TBD

> Full results: `EDA_04_lens3_pcp_first.ipynb`

---

### 4.4 Common Patterns Observed

**FP visit prior to first diagnosis encounter:**
On investigation, a significant portion of first diagnosis encounters are preceded
by an FP visit — suggesting the FP visit generated the referral or lab order
that led to the diagnosis. This means the trigger may not be the true start
of the care journey.

TBD

**Trade-off — known limitation:**
Defining the trigger as the first encounter of a diagnosis is a practical approximation
and is sufficient for predicting the next visit in sequence. However, it is not necessarily
the first encounter of a condition for the member. A PCP visit prior to the trigger
may represent the true clinical starting point. This is flagged as a known limitation —
not a blocker, but important for clinical interpretation.

---

## 5. Why Time Window Over Sequence

### Problem

Sequential visit analysis (Order 1/2) treats the immediate next visit as the prediction
target. In practice, members generate multiple unrelated visits between clinically
related visits — labs, follow-ups, chronic condition management on adjacent days.

A cardiology visit on day 5 may be unrelated to the trigger diagnosis. Treating it
as the prediction target introduces noise.

### Method

Instead of the immediate next visit, capture the **first specialist seen within a
defined window** after the second visit (V2). Three windows evaluated:

- **T30** — First specialist within 30 days after V2
- **T60** — First specialist within 60 days after V2
- **T180** — First specialist within 180 days after V2

Signal comparison: entropy of Order 1 next-visit transitions vs T30/T60/T180
transitions for the same diagnosis pairs.

Entropy trend per diagnosis pair across T30, T60, T180 reveals:
- Dropping entropy — signal strengthens over time, longer window is better
- Flat entropy — signal established early, T30 is sufficient
- Rising entropy — routing scatters over time, shorter window is better

### Results

TBD

TBD

> Full results: `EDA_05_time_window_analysis.ipynb`

### Decision

TBD

---

## 6. Time Window Analysis — FP First Encounters

### 6.1 T30 — First Specialist Within 30 Days

**Question:** Given a first encounter at FP/I, which specialty does the member
visit within 30 days — and how predictably?

TBD

> Full results: `EDA_06_t30_fp.ipynb`

---

### 6.2 T60 — First Specialist Within 60 Days

TBD

> Full results: `EDA_07_t60_fp.ipynb`

---

### 6.3 T180 — First Specialist Within 180 Days

TBD

> Full results: `EDA_08_t180_fp.ipynb`

---

## 7. What Is the Right Scope?

### Problem

Over 70,000 ICD-10 codes exist on the trigger side. Most have insufficient evidence
or high entropy — no predictable routing pattern. Building a model on the full scope
dilutes performance.

### Method
- Compute weighted average entropy by CCSR domain
- Apply minimum transition threshold of 100 per diagnosis pair
- Identify CCSR domains with consistently low entropy across cohorts and time windows

### Results

TBD

> Full results: `EDA_09_scope_selection.ipynb`

### Decision

TBD

Coverage validation: percentage of members whose trigger diagnosis falls within
the selected scope.

TBD

---

## 8. Does Member Profile Shift the Signal?

### Method
Compare entropy and conditional probability distributions for the same diagnosis pairs
across Adult Female, Adult Male, Senior, and Children cohorts.

### Results

TBD

> Full results: `EDA_10_cohort_analysis.ipynb`

### Decision

TBD

---

## 9. Markov Baseline

Transition probabilities from this EDA used directly as a naive prediction baseline.

**Full scope baseline** — all diagnosis pairs with 100+ transitions across all time windows
**Scoped baseline** — high-signal CCSR domains only (Section 7)

The scoped baseline is expected to outperform the full scope baseline because
high-entropy diagnosis pairs drag down overall accuracy. This comparison justifies
narrowing the model scope before training HSTU.

### Results

TBD

> Full results: `EDA_11_markov_baseline.ipynb`

**Note:** Test split must be identical to HSTU model evaluation.
Baseline construction finalized during model building phase.

Evaluation metrics — same as HSTU:
- NDCG at K
- Precision at K
- Recall at K
- Hit at K

For K = 1, 3, 5 across T30, T60, T180.

---

## 10. Business Value

Given a member's trigger diagnosis, V2 diagnosis, member segment, and time window,
the model produces predictions at three levels:

| Level | Output |
|---|---|
| Member level | TBD |
| Specialty level | TBD |
| Provider level | TBD |

**Use cases:**
- **Care navigation** — proactively connect member to in-network specialist before self-referral
- **Network adequacy** — identify specialties with high predicted demand by geography and condition
- **Gap in care detection** — members with high-signal diagnosis pairs who never complete the predicted visit

---

## 11. Known Limitations

### PCP Prior Visit Trade-off
Trigger is defined as the first encounter of a diagnosis — not the first encounter
of a condition. A PCP visit generating the referral may precede the trigger.
Acceptable for prediction. Important for clinical interpretation.

### Right-Censoring
Members triggering after June 2024 may lack complete T180 follow-up within
the 2022–2025 dataset. T180 counts for recent triggers are understated.
T30 and T60 are less affected.

### Multi-Condition Noise
Members with multiple active conditions generate visits across unrelated care pathways
on adjacent days. Time window analysis reduces but does not eliminate this noise.

---

## 12. Deliverables

| Deliverable | Notebook | Status |
|---|---|---|
| Claims flagged table | — | Done |
| Visit spine and triggers | — | Done |
| 9 Order 1 transition tables | — | Done |
| 9 Order 2 transition tables | — | Done |
| Entropy summary table | — | Done |
| Sequence EDA — Order 1 Parts 1, 2, 3 | EDA_01_entropy_summary.ipynb | Done |
| Track 1 base table (ARRAY STRUCT) | — | In Progress |
| Track 1 summary table | — | In Progress |
| Time window EDA — T30, T60, T180 | EDA_05_time_window_analysis.ipynb | In Progress |
| South Florida scope stats | — | Pending |
| Data quality coverage | — | Pending |
| Scope selection — CCSR domain analysis | EDA_09_scope_selection.ipynb | Pending |
| Cohort analysis | EDA_10_cohort_analysis.ipynb | Pending |
| Markov baseline — full scope | EDA_11_markov_baseline.ipynb | Pending |
| Markov baseline — scoped | EDA_11_markov_baseline.ipynb | Pending |
| EDA summary document | This document | In Progress |
| 2-page milestone slide deck | — | Pending |
