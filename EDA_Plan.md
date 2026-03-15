# Provider Network Transition Analysis — Project Plan
## Milestone Meeting — March 2026

---

## 1. Project Objective

Build a predictive system that, given a member's first encounter with a new diagnosis,
predicts the next provider specialty (and eventually provider) they will visit.

This analysis establishes the **empirical foundation** — proving predictive signal exists
in claims data before committing to a sequential model architecture.

---

## 2. Data Scope

**Source:** CVS/Aetna claims data 2022–2025
**Geography:** South Florida
**Dataset:** `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl`

### 2.1 South Florida — Summary Statistics
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

### 2.2 Data Quality Coverage
| Metric | Coverage % |
|---|---|
| Claims with valid ICD-10 codes | TBD |
| Claims with specialty codes | TBD |
| Claims with CCSR mappings | TBD |
| Members with at least 2 visits | TBD |

---

## 3. Data Preparation

### 3.1 Reference Tables
| Table | Purpose |
|---|---|
| `A870800_gen_rec_claims_flagged` | Claims enriched with visit flags, CCSR, first encounter flags |
| `A870800_gen_rec_visit_spine` | One row per member + visit date with visit rank |
| `A870800_gen_rec_triggers` | One row per member + trigger date + trigger DX (first encounter) |
| `A870800_gen_rec_f_track1_base` | Member + trigger + downstream visits as ARRAY<STRUCT>, T180 window |
| `A870800_gen_rec_f_track1_summary` | Aggregated transitions with entropy, probability, median days |

### 3.2 Visit Flagging Logic
- `is_first_dx_encounter` — first time member presents with a specific ICD-10 code
- `visit_flag` — `first_member_visit`, `new_provider_new_dx`, `new_provider_known_dx`,
  `known_provider_new_dx`, `known_provider_known_dx`

### 3.3 Member Segments
| Segment | Definition |
|---|---|
| Children | Age < 18 |
| Adult Female | Age 18-65, Gender F |
| Adult Male | Age 18-65, Gender M |
| Senior | Age > 65 |

### 3.4 Clinical Grouping — CCSR
ICD-10 codes are grouped into ~530 CCSR clinical categories using AHRQ mapping.
CCSR is used **only for EDA** to reduce dimensionality and identify clinical domains
with strong predictive signal.

---

## 4. Analytical Framework

### 4.1 Why Diagnosis → Specialty?

Nine transition combinations were evaluated across three starting units
(Diagnosis Code, CCSR Category, Provider Specialty) and three prediction units
(Diagnosis Code, CCSR Category, Provider Specialty).

**Selection criterion:** Conditional entropy — lower entropy = higher predictive power.

**Winner:** Diagnosis Code → Provider Specialty
- Most clinically actionable
- Lowest weighted average entropy across all cohorts
- Direct input to provider matching and care navigation

### 4.2 Markov Chain Framework

**Order 1:** V1 (trigger) → V2 (next visit)
**Order 2:** V1 + V2 → V3

Transition probability: `P(next_specialty | current_dx)`
Conditional entropy: `H = -SUM(p * LOG(p))` per current state

Lower entropy = more predictable routing from that diagnosis.

---

## 5. EDA — Three Analytical Lenses

### Lens 1 — No Filters (All Encounters)
**Question:** Given any visit sequence in the data, which transitions are highly predictable?

**Units analyzed:**
- Diagnosis Code → Provider Specialty
- Diagnosis Code → CCSR Category
- Diagnosis Code → Diagnosis Code
- CCSR → Provider Specialty
- CCSR → CCSR
- Specialty → Specialty
- Specialty → Diagnosis Code
- Specialty → CCSR

**Tables:** `A870800_gen_rec_f_[combination]_order1/2`

---

### Lens 2 — First Encounter of a Diagnosis
**Question:** Given a member's first encounter with a new diagnosis, which next visit
sequences are highly predictable?

**Trigger definition:** First date a member presents with a specific ICD-10 code.

**Tables:** `A870800_gen_rec_f_[combination]_order1/2` (filtered from triggers table)

---

### Lens 3 — First Encounter with FP or I (Potential PCP)
**Question:** Given that the first encounter of a diagnosis occurred at a Family Practice
or Internal Medicine provider, which next visit sequences are highly predictable?

**Trigger filter:** `trigger_specialty IN ('FP', 'I')`

**Tables:** Same as Lens 2, filtered on `trigger_specialty`

---

## 6. EDA Justifications

### 6.1 Why T-Delta Over Sequence

**Problem with sequence analysis (Order 1/2):**
Sequential visit analysis treats the next single visit as the prediction target.
In practice, members generate multiple unrelated visits between clinically related visits —
labs, follow-ups, chronic condition management on adjacent days.
This creates noise — a cardiology visit on day 5 may be unrelated to the trigger diagnosis.

**Evidence:**
- Compare entropy of Order 1/2 transitions vs T30/T60/T180 transitions
- For the same `(trigger_dx, v2_dx)` pairs — does entropy drop significantly
  when moving from next-visit to time-window?
- If yes — time window removes noise and strengthens signal

**Decision criterion:** If weighted average entropy is meaningfully lower in T30
compared to Order 1 for the same DX pairs → T-delta is the right framing.

---

### 6.2 Starting Point Selection — FP Only vs Any Visit

**Question:** Does filtering to FP/I triggers improve predictive signal compared
to all first encounters?

**Evidence:**
- Side-by-side entropy comparison — all triggers vs FP/I triggers only
- For same `(trigger_dx, v2_dx)` pairs — is entropy lower in FP/I subset?
- Volume check — is the FP/I subset large enough to model reliably?

**Expected finding:**
FP/I triggers represent the most clinically coherent entry point —
the member came to a generalist first, who then directed care.
Non-FP triggers (specialist-first, ER, lab-first) represent noisier entry points
where the care pathway is already in progress.

**Decision criterion:**
- If FP/I entropy < all-triggers entropy by meaningful margin → scope to FP/I
- If volume drops below modeling threshold → reconsider

---

### 6.3 Diagnosis and CCSR Scope Selection

**Question:** Should we limit the model to certain diagnosis codes or CCSR categories?

**Problem with full scope:**
- 70,000+ ICD-10 codes on the trigger side
- Rare diagnoses have insufficient evidence (transition_count < 100)
- Many diagnoses have high entropy — no predictable routing pattern

**Evidence:**
- Entropy distribution by CCSR domain — identify low-entropy domains
- Transition volume distribution — identify well-evidenced DX codes
- Penetration rate by CCSR — identify domains where specialist referral
  is common and predictable

**Selection criteria:**
1. CCSR domains with weighted average entropy below threshold
2. Minimum transition count of 100+ per DX pair
3. Specialist penetration rate > X% within T30

**Expected outcome:**
A focused set of 5-10 CCSR domains (chronic conditions — diabetes,
hypertension, cardiovascular, musculoskeletal, mental health)
representing the highest-signal modeling candidates.

---

## 7. Markov Baseline Model

### 7.1 Full Scope Baseline
Built for all DX pairs, all time windows, all cohorts.

- Input: `(trigger_dx, v2_dx, member_segment, time_window)`
- Output: `next_specialty` = argmax(conditional_probability)
- Coverage: all DX pairs with transition_count >= 100

### 7.2 Scoped Baseline
Built for high-signal CCSR domains only (selected from Section 6.3).

- Same input/output structure
- Restricted to selected CCSR domains

### 7.3 Why Scoped Baseline Wins
Compare full scope vs scoped baseline on:
- Average conditional entropy
- Top-1 accuracy on held-out transitions
- Coverage — % of members whose trigger DX falls within scope

**Expected finding:**
Scoped baseline has meaningfully higher accuracy because high-entropy
DX pairs in the full scope drag down overall performance.
This justifies limiting the sequential model to the high-signal scope.

### 7.4 Evaluation Metrics (Model Building Phase)
Same test split as HSTU model:
- NDCG@K
- Precision@K
- Recall@K
- Hit@K

For K = 1, 3, 5 across T30, T60, T180.

---

## 8. Known Limitations

### 8.1 PCP Prior Visit Trade-off
Defining the trigger as the first encounter of a diagnosis is a practical approximation.
In many cases there is an FP visit **prior** to the diagnosis trigger — the FP visit
generated the referral or lab order that led to the diagnosis.

**Evidence gathered:**
- Volume of cases where an FP visit precedes the trigger within T30
- % of trigger visits where `visit_flag = known_provider_new_dx`

**Impact:** The trigger is not always the true start of the care journey.
Acceptable for prediction purposes but important for clinical interpretation.

### 8.2 Right-Censoring
Members who triggered in late 2024 may not have complete T180 follow-up
within the 2022–2025 dataset window.

**Impact:** T180 transition counts for recent triggers are understated.
T30 and T60 are less affected.

**Mitigation:** Flag trigger dates after June 2024 as potentially right-censored
in all T180 analyses.

### 8.3 Multi-Condition Noise
Members with multiple active conditions generate visits across unrelated care pathways
on the same or adjacent days.

**Mitigation:** Time window analysis reduces but does not eliminate this noise.
Future work: condition-specific cohort filtering.

---

## 9. Business Value

### 9.1 Prediction Outputs
Given a member's `(trigger_dx, v2_dx, member_segment, time_window)`:

| Level | Output |
|---|---|
| Member level | Predicted next specialty for this specific member |
| Specialty level | Probability distribution across specialties for this DX pair |
| Provider level | Recommended in-network provider within predicted specialty |

### 9.2 Use Cases
- **Care navigation** — proactively connect member to in-network specialist before self-referral
- **Network adequacy** — identify specialties with high predicted demand by geography and condition
- **Gap in care detection** — members with high-signal DX pairs who never complete the predicted visit

---

## 10. Provider Recommendation Hypothesis (Next Phase)

**Hypothesis:** For the same `(trigger_dx, v2_dx)` pair, the next specialty varies
significantly by member profile — suggesting providers implicitly personalize referrals.

**Validation approach:**
- Fix: `trigger_dx + v2_dx` pair
- Vary: `member_segment`, age, gender
- Measure: KL divergence or Chi-square on specialty distribution across cohorts

**If validated:** Member profile must be included as model input.
**If not validated:** Simpler model without member profile is sufficient.

---

## 11. Deliverables

| Deliverable | Status |
|---|---|
| Claims flagged table | Done |
| Visit spine + triggers | Done |
| 9 Order 1 transition tables | Done |
| 9 Order 2 transition tables | Done |
| Entropy summary table | Done |
| Python EDA — Order 1 (Parts 1/2/3) | Done |
| Track 1 base table (ARRAY<STRUCT>) | In Progress |
| Track 1 summary table | In Progress |
| Python EDA — Track 1 (T30/T60/T180) | In Progress |
| South Florida scope stats SQL | Pending |
| Data quality coverage SQL | Pending |
| Markov baseline — full scope | Pending |
| Markov baseline — scoped | Pending |
| Milestone MD document | This document |
| 2-page slide deck | Pending |

---

## 12. Next Steps

1. Complete Track 1 base + summary tables
2. Run Track 1 Python EDA
3. South Florida scope statistics + data quality SQL
4. EDA justifications — entropy comparisons (Sections 6.1, 6.2, 6.3)
5. Markov baseline — full scope + scoped
6. 2-page milestone slide deck
7. Provider recommendation hypothesis (next phase)
8. Markov vs HSTU baseline comparison (model building phase)
