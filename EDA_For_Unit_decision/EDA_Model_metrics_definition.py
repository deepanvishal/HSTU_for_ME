# ============================================================
# NB_Reference_Metrics.py
# Purpose : Reference guide — metric definitions and calculations
#           for next visit specialty prediction
# ============================================================
from IPython.display import display, Markdown

display(Markdown("""
# Reference Guide — Model Evaluation Metrics

---

## Context

Each record in this evaluation has:
- **1 trigger diagnosis code** — the ICD-10 code that initiated the prediction
- **1 true label set** — the set of specialties the member actually visited in the time window (can be 0, 1, or many)
- **K predictions** — the model's top-K ranked specialty predictions

The model predicts specialties, not visits. One member may visit 3 different specialties
in the T0_30 window — all 3 are in the true label set. The model's job is to predict
as many of those as possible within its top-K predictions.

---

## What is K?

K is the cutoff — how many of the model's predictions we evaluate.

- **K=1** — did the model's single best prediction match any true label?
- **K=3** — did any of the model's top 3 predictions match any true label?
- **K=5** — did any of the model's top 5 predictions match any true label?

Higher K is easier to satisfy. K=3 is the primary metric in this project
because it balances precision (not predicting everything) with recall (catching real visits).

A model always produces 5 ranked predictions. Evaluating at K=1, K=3, K=5 shows
how quickly it surfaces the right answer in its ranking.

---

## The Four Metrics

### Hit@K (Binary Relevance)

> Did the model predict at least one correct specialty in its top K?

- **Range:** 0 or 1 per trigger
- **Interpretation:** 1 = success, 0 = complete miss
- **Averaged** across all triggers to get overall Hit@K

**Why it matters:** The most intuitive metric. For a care manager acting on the model's
top-3 recommendations — did at least one of those recommendations turn out to be right?

---

### Precision@K

> Of the K predictions made, what fraction were correct?

```
Precision@K = (number of correct predictions in top K) / K
```

- **Range:** 0 to 1
- **Note:** denominator is always K, not the number of true labels

**Why it matters:** Measures how much noise is in the predictions.
A model predicting 3 specialties where 2 are correct has Precision@3 = 0.67.

---

### Recall@K

> Of all the true labels, what fraction did the model find in top K?

```
Recall@K = (number of correct predictions in top K) / (number of true labels)
```

- **Range:** 0 to 1
- **Note:** denominator is the size of the true label set, not K

**Why it matters:** Measures completeness. If a member will visit 4 specialties
and the model finds 3 of them in top 5, Recall@5 = 0.75.
Precision and Recall trade off — a model predicting all 70 specialties would have
Recall = 1.0 but Precision ≈ 0.

---

### NDCG@K (Normalized Discounted Cumulative Gain)

> Did the model rank the correct predictions near the top?

This is the most nuanced metric. It rewards models that put correct predictions
at rank 1 more than at rank 3, and more than at rank 5.

**Step 1 — Discounted Cumulative Gain (DCG):**
```
DCG@K = sum of [ 1 / log2(rank + 1) ] for each correct prediction in top K
```
Discount weights by rank:
| Rank | Discount = 1/log2(rank+1) |
|------|--------------------------|
| 1    | 1.000                    |
| 2    | 0.631                    |
| 3    | 0.500                    |
| 4    | 0.431                    |
| 5    | 0.387                    |

**Step 2 — Ideal DCG (IDCG):**
The best possible DCG — what if all true labels were ranked at positions 1, 2, 3, ...?
```
IDCG@K = DCG of the ideal ranking, capped at min(|true labels|, K)
```

**Step 3 — NDCG:**
```
NDCG@K = DCG@K / IDCG@K
```
- **Range:** 0 to 1
- **NDCG = 1.0** means perfect ranking — all true labels at the top

**Why it matters:** Two models can both achieve Hit@3 = 1.0, but one always puts
the right answer at rank 1 and the other at rank 3. NDCG distinguishes them.

---

## Worked Examples

All examples use the same setup:
- **Trigger:** ICD-10 Z12.11 (encounter for screening for malignant neoplasm of colon)
- **Model produces 5 ranked predictions**

---

### Scenario 1 — Single True Label, Correct at Rank 1

**True label set:** { Gastroenterology }
**Predictions:** [ Gastroenterology, Internal Medicine, Cardiology, Oncology, Surgery ]

| Metric | Calculation | Value |
|--------|-------------|-------|
| Hit@1  | Gastroenterology in top 1? Yes | **1.0** |
| Hit@3  | Gastroenterology in top 3? Yes | **1.0** |
| Hit@5  | Gastroenterology in top 5? Yes | **1.0** |
| Precision@3 | 1 correct / 3 = | **0.333** |
| Recall@3 | 1 correct / 1 true label = | **1.000** |
| DCG@3 | 1/log2(2) = 1.000 | |
| IDCG@3 | 1/log2(2) = 1.000 (ideal = rank 1) | |
| NDCG@3 | 1.000 / 1.000 = | **1.000** |

---

### Scenario 2 — Single True Label, Correct at Rank 3

**True label set:** { Gastroenterology }
**Predictions:** [ Internal Medicine, Cardiology, Gastroenterology, Oncology, Surgery ]

| Metric | Calculation | Value |
|--------|-------------|-------|
| Hit@1  | Gastroenterology in top 1? No | **0.0** |
| Hit@3  | Gastroenterology in top 3? Yes | **1.0** |
| Hit@5  | Gastroenterology in top 5? Yes | **1.0** |
| Precision@3 | 1 correct / 3 = | **0.333** |
| Recall@3 | 1 correct / 1 true label = | **1.000** |
| DCG@3 | 1/log2(4) = 0.500 | |
| IDCG@3 | 1/log2(2) = 1.000 (ideal = rank 1) | |
| NDCG@3 | 0.500 / 1.000 = | **0.500** |

**Key insight:** Hit@3 is identical to Scenario 1. NDCG@3 is penalized because
the correct answer appeared at rank 3 instead of rank 1.

---

### Scenario 3 — Multiple True Labels, Partial Hit

**True label set:** { Gastroenterology, Oncology, Surgery }  ← 3 specialties visited
**Predictions:** [ Gastroenterology, Internal Medicine, Cardiology, Oncology, Primary Care ]

| Metric | Calculation | Value |
|--------|-------------|-------|
| Hit@1  | Any of {Gastro, Onco, Surgery} in top 1? Yes (Gastro) | **1.0** |
| Hit@3  | Any in top 3? Yes (Gastro at rank 1) | **1.0** |
| Hit@5  | Any in top 5? Yes (Gastro rank 1, Onco rank 4) | **1.0** |
| Precision@3 | 1 correct in top 3 / 3 = | **0.333** |
| Recall@3 | 1 correct in top 3 / 3 true labels = | **0.333** |
| DCG@3 | 1/log2(2) + 0 + 0 = 1.000 | |
| IDCG@3 | Ideal: top 3 all correct = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.000 + 0.631 + 0.500 = 2.131 | |
| NDCG@3 | 1.000 / 2.131 = | **0.469** |

**Key insight:** Even though Hit@3 = 1.0, NDCG@3 = 0.469 because the model
missed 2 of the 3 true labels entirely. The model found one right answer early
but failed to capture the full picture.

---

### Scenario 4 — Multiple True Labels, All Found

**True label set:** { Gastroenterology, Oncology, Surgery }
**Predictions:** [ Gastroenterology, Oncology, Surgery, Internal Medicine, Cardiology ]

| Metric | Calculation | Value |
|--------|-------------|-------|
| Hit@3  | Any in top 3? Yes | **1.0** |
| Precision@3 | 3 correct / 3 = | **1.000** |
| Recall@3 | 3 correct / 3 true labels = | **1.000** |
| DCG@3 | 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.000 + 0.631 + 0.500 = 2.131 | |
| IDCG@3 | Same as DCG (perfect ranking) | |
| NDCG@3 | 2.131 / 2.131 = | **1.000** |

---

### Scenario 5 — Multiple True Labels, Found Out of Order

**True label set:** { Gastroenterology, Oncology, Surgery }
**Predictions:** [ Surgery, Internal Medicine, Oncology, Gastroenterology, Cardiology ]

All 3 true labels appear in top 5 but not at top ranks.

| Metric | Calculation | Value |
|--------|-------------|-------|
| Hit@3  | Any in top 3? Yes (Surgery rank 1, Onco rank 3) | **1.0** |
| Precision@3 | 2 correct / 3 = | **0.667** |
| Recall@3 | 2 correct / 3 true labels = | **0.667** |
| DCG@3 | 1/log2(2) + 0 + 1/log2(4) = 1.000 + 0 + 0.500 = 1.500 | |
| IDCG@3 | 1.000 + 0.631 + 0.500 = 2.131 (ideal order) | |
| NDCG@3 | 1.500 / 2.131 = | **0.704** |

**Compare to Scenario 4:** Same 3 true labels, same 5 predictions, but different order.
NDCG drops from 1.0 to 0.704 purely due to ranking quality.

---

### Scenario 6 — No True Labels (Member Had No Downstream Visits)

**True label set:** { } ← member had no qualifying visits in the window
**Predictions:** [ Gastroenterology, Internal Medicine, Cardiology, Oncology, Surgery ]

This trigger is **excluded from evaluation** for that time window.

The `is_t30_qualified`, `is_t60_qualified`, `is_t180_qualified` flags
control this — a trigger is only evaluated for a window if it has at least
one true label in that window. A member with no T0_30 visits contributes
to T30_60 and T60_180 evaluation only if they had visits there.

**No metrics are computed. The trigger does not count toward n_triggers for that window.**

---

### Scenario 7 — Large True Label Set (Many Visits)

**True label set:** { Gastro, Oncology, Surgery, Radiology, Primary Care, Cardiology }  ← 6 labels
**Predictions:** [ Gastro, Primary Care, Oncology, Internal Medicine, Radiology ]
3 of 5 predictions are correct (Gastro rank 1, Primary Care rank 2, Oncology rank 3).

| Metric | Calculation | Value |
|--------|-------------|-------|
| Hit@3  | Any correct in top 3? Yes | **1.0** |
| Precision@3 | 3 correct / 3 = | **1.000** |
| Recall@3 | 3 correct / **6** true labels = | **0.500** |
| DCG@3 | 1.000 + 0.631 + 0.500 = 2.131 | |
| IDCG@3 | Capped at K=3: 1.000 + 0.631 + 0.500 = 2.131 | |
| NDCG@3 | 2.131 / 2.131 = | **1.000** |

**Key insight:** IDCG is capped at K — you cannot be expected to find more than K
labels when only predicting K items. NDCG@3 = 1.0 even though only 3 of 6 labels
were found, because the model perfectly used its 3 slots. Recall@3 = 0.5 captures
the incompleteness. This is why both NDCG and Recall are reported.

---

### Scenario 8 — No Transition Probability for This Diagnosis (Markov Only)

**True label set:** { Gastroenterology }
**Markov predictions:** None — this ICD-10 code never appeared in training data

The Markov model has no transition probability to draw from.
The trigger appears in predictions table with `predicted_specialty = NULL`.
It is **excluded from the Markov prediction rollup** (WHERE clause filters NULLs).
This trigger will not appear in `A870800_gen_rec_markov_trigger_scores`.

**Impact on metrics:** Markov n_triggers will be lower than SASRec/BERT4Rec
for rare diagnosis codes. This is a known limitation of frequency-based baselines.

---

## Summary Table

| Metric | Numerator | Denominator | Sensitive to rank order? | Multi-label friendly? |
|--------|-----------|-------------|--------------------------|----------------------|
| Hit@K | 1 if any hit | 1 (binary) | No | Yes |
| Precision@K | hits in top K | K | No | Partial |
| Recall@K | hits in top K | true label count | No | Yes |
| NDCG@K | discounted hits | ideal discounted hits | **Yes** | **Yes** |

**Primary metric in this project: NDCG@3**
Because it handles multi-label, rewards early correct predictions,
and normalizes for varying true label set sizes.

---

## Common Questions

**Q: Why does Hit@5 sometimes equal Hit@3?**
If the correct answer appeared in ranks 1-3, Hit@3 = Hit@5 = 1.0.
Hit@5 > Hit@3 only when the first correct prediction appears at rank 4 or 5.

**Q: Can NDCG@3 be higher than NDCG@5?**
No — adding more positions can only maintain or improve NDCG since
additional correct predictions at ranks 4-5 add to DCG while IDCG
is recalculated with the higher K cap.

**Q: Why is Recall@K often lower than Precision@K?**
When true label sets are large (many downstream visits), it's harder
to find all of them within K predictions. Recall penalizes for missed labels.
Precision only cares about what was predicted, not what was missed.

**Q: The model predicts the same top-5 for all windows (Markov). Is that fair?**
No — it is not fair, and intentional. Markov is a time-blind baseline.
Its flat performance across windows shows exactly what temporal modeling adds.
SASRec and BERT4Rec learn visit sequences over time, giving them an edge
at T0_30 that should degrade more gracefully at T60_180 than Markov does.
"""))

print("NB_Reference_Metrics loaded")
