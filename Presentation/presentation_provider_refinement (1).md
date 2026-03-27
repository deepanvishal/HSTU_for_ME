# Provider-Level Prediction — Refinement Slides
## 4 Slides (filled with actual results)

---

## R1: Setup — What Changed

**Title:** "Provider-level prediction extends specialty prediction to 31,634 individual providers"

**Visual:** Side-by-side comparison table (PROV-VIS-01)

| Dimension | Specialty Model | Provider Model |
|---|---|---|
| Prediction target | ~70 specialties | 31,634 providers (top 80% of 747K) |
| Input per visit | specialty code | provider ID + specialty code |
| Loss function | BCE (multi-hot labels) | InfoNCE (dot-product + negatives) |
| Markov grain | (trigger_dx, segment) → specialty | (trigger_dx, segment, from_provider) → to_provider |
| Output | Top 5 specialties | Top 5 providers |
| Models compared | Markov, SASRec, BERT4Rec | Markov, SASRec, BERT4Rec, HSTU |

**Body:**
- Same 5% member sample (968K members), same train/test split (pre-2024 / 2024+).
- Provider vocabulary: top 80% by transition volume (31,634 of 747K). Tail providers mapped to UNK.
- HSTU added — uses time gaps between visits as a native input.

---

## R2: Overall Performance — All Models

**Title:** "Provider-level prediction achieves 46.1% accuracy at 30 days — sequence models are 2x the baseline"

**Visual:** Grouped bar chart — 4 models × 3 windows at Hit@5 (PROV-VIS-02)

| Model | T30 | T60 | T180 |
|---|---|---|---|
| SASRec | 46.1% | 45.8% | 69.3% |
| HSTU | 44.8% | 44.5% | 68.5% |
| BERT4Rec | 43.4% | 43.3% | 67.6% |
| Markov | 20.3% | 19.1% | 37.4% |

**Body:**
- Baseline (Markov): 20.3% at T30. Transition sparsity at 31K providers collapses simple frequency-based prediction.
- Best approach (SASRec): 46.1% at T30, 69.3% at T180. Sequence models generalize from visit patterns where Markov cannot.
- Compared to specialty prediction (85.3%), provider prediction is harder — 31K targets vs 70 — but the 2x lift over Markov confirms sequence models add substantial value.
- Hit@5 by segment (SASRec, T30): Senior 47.2%, Adult Female 46.4%, Adult Male 44.1%, Children 28.8%.

---

## R3: Where It Works — Inbound and Outbound

**Title:** "Outbound: top providers predicted at 100% accuracy. Inbound: top destinations predicted with 97–100% precision."

**Visual:** Two horizontal bar charts side by side (PROV-VIS-03a, PROV-VIS-03b)

**Left — Outbound (SASRec, T30, excl Lab):**
"When a member leaves this provider, how often can we predict who they see next?"

| Provider | Specialty | Triggers | Hit@5 |
|---|---|---|---|
| Onelis Vega | Nurse Practitioner | 53 | 100% |
| Edward Noguera | Anesthesia | 20 | 100% |
| Hien D. Liu | Hematology/Oncology | 22 | 100% |
| Anurag Agarwal | Radiation Therapy | 20 | 100% |
| Maria-Amelia M. Rodrigues | Radiation Therapy | 24 | 100% |
| Eva Marie Suarez | Radiation Therapy | 27 | 100% |
| Joseph Markowitz | Hematology/Oncology | 20 | 100% |
| Mihir Naik | Radiation Therapy | 25 | 100% |
| Bushra Fathima Shariff | Internal Medicine | 33 | 100% |
| Silke Natasha Hunter | Family Practice | 21 | 100% |

**Right — Inbound (SASRec, T30, excl Lab):**
"When we predict this provider as the destination, how often are we correct?"

| Provider | Specialty | Triggers | Precision |
|---|---|---|---|
| Belle G. Heneberger | Mental Health Professional | 21 | 100.0% |
| American Renal Associates LLC | Dialysis Center | 330 | 97.0% |
| Angelos Koutsonikolis | Allergy/Immunology | 23 | 95.7% |
| Gary W. Price | Plastic Surgery | 206 | 94.2% |
| DaVita, Inc. | Dialysis Center | 328 | 93.3% |
| Andrew S. Bagg | Allergy/Immunology | 28 | 92.9% |
| Joel Wayne Phillips | Allergy/Immunology | 479 | 91.7% |
| Brett E. Stanaland | Allergy/Immunology | 35 | 91.4% |
| Rodrigo Baltodano | Internal Medicine | 198 | 90.4% |
| Amanda Jean Mitchell | Chiropractor | 100 | 90.0% |

**Body:**
- Outbound: providers with structured referral patterns (oncology, radiation therapy) are predicted perfectly.
- Inbound: dialysis centers and allergy/immunology providers are the most reliably predicted destinations — high-volume, condition-specific routing.
- Laboratory Center excluded from both views — high volume but not clinically actionable.

---

## R4: Performance by Evidence and Diagnosis

**Title:** "Sequence models outperform Markov at every evidence level — and dominate low-evidence transitions"

**Visual:** Grouped bar chart — 3 buckets × 4 models at Hit@5 (PROV-VIS-04)

| Evidence Bucket | BERT4Rec | HSTU | SASRec | Markov |
|---|---|---|---|---|
| High | 43.4% | 44.8% | 46.1% | 20.6% |
| Medium | 44.4% | 45.8% | 47.4% | 5.2% |
| Low | 43.8% | 45.5% | 46.8% | 2.5% |

**Body:**
- Markov collapses on medium/low evidence (5.2% and 2.5%) — not enough training transitions to form reliable probabilities.
- Sequence models maintain 43–47% accuracy regardless of evidence level — they learn from visit patterns, not just frequency counts.
- Top 10 diagnoses by Hit@5 (outbound, SASRec): all oncology/chemotherapy-related — most structured referral pathways at provider level.

**Top 10 Outbound Diagnoses (SASRec, T30):**

| Diagnosis | Triggers | Hit@5 |
|---|---|---|
| Agranulocytosis secondary to cancer chemotherapy (D70.1) | 382 | 86.4% |
| Adverse effect of antineoplastic/immunosuppressive (T45.1X5A) | 143 | 84.6% |
| Diffuse large B-cell lymphoma (C83.30) | 211 | 84.4% |
| Anemia due to antineoplastic chemotherapy (D64.81) | 252 | 84.1% |
| Encounter for antineoplastic chemotherapy (Z51.11) | 3,159 | 84.0% |
| Secondary malignant neoplasm (C77.1) | 200 | 84.0% |
| Encounter for antineoplastic immunotherapy (Z51.12) | 1,329 | 84.0% |
| Antineoplastic chemo induced pancytopenia (D61.810) | 191 | 82.7% |
| Other complication of kidney transplant (T86.19) | 139 | 82.0% |
| Acute embolism and thrombosis (I82.C11) | 121 | 81.8% |

---

# FACTS SUMMARY

| FACT | Value |
|---|---|
| PROV-FACT-01 Best model Hit@5 T30 | SASRec 46.1% |
| PROV-FACT-02 Markov Hit@5 T30 | 20.3% |
| PROV-FACT-03 Best model Hit@5 T180 | SASRec 69.3% |
| PROV-FACT-04 Segment: Senior | 47.2% |
| PROV-FACT-04 Segment: Adult Female | 46.4% |
| PROV-FACT-04 Segment: Adult Male | 44.1% |
| PROV-FACT-04 Segment: Children | 28.8% |
| PROV-FACT-05 Evidence High (SASRec) | 46.1% |
| PROV-FACT-05 Evidence Medium (SASRec) | 47.4% |
| PROV-FACT-05 Evidence Low (SASRec) | 46.8% |
| PROV-FACT-09 Total providers | 747,762 |
| PROV-FACT-09 Top 80% | 31,634 |

---

# INBOUND vs OUTBOUND LABELS

| Slide | Chart | Direction | Meaning |
|---|---|---|---|
| R2 | Overall grouped bar | Overall | Hit@5 across all triggers |
| R3 left | Top 10 providers | Outbound | "Member leaves this provider — can we predict who they see next?" |
| R3 right | Top 10 providers | Inbound | "When we predict this provider, how often are we correct?" |
| R4 | Evidence buckets | Overall | Performance by training data density |
| R4 body | Top 10 dx | Outbound | "Given this diagnosis, how often correct at provider level?" |
