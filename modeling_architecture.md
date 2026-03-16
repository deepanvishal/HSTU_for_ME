# Modeling Architecture
## Next Visit Prediction
### South Florida Claims Data — 2022 to 2025

---

## Objective

Predict the next provider specialty a member will visit, given their first encounter
with a new diagnosis. Outputs generated at three time windows — T30, T60, T180 days.

---

## Target Variable

- **Prediction target:** Provider specialty
- **Time windows:** T30, T60, T180 — cumulative days after trigger visit
- **Label type:** Multi-label — multiple specialties can be visited within a window
- **Top K:** Top 1, Top 3, Top 5 (extensible to Top 10)

---

## Trigger Definition

- First encounter of a new ICD-10 diagnosis code for a member
- CCSR used for EDA and scope selection only — not a model input

---

## Candidate Features

### Clinical
- Trigger ICD-10 diagnosis code
- Member's prior diagnosis history (sequence of ICD-10 codes)

### Member Demographics
- Age
- Gender

### Temporal
- Days since prior visit (delta-t)
- Log-scale bucketed delta-t for HSTU input
- Visit sequence length

---

## Sample Size

- **Initial training run:** 5% of total member population
- Expand to full population once model architecture is validated

---

## Models

Four models evaluated independently — not stages, parallel comparison.

### Model 1 — Logistic Regression / XGBoost

- No sequence modeling
- Input: trigger diagnosis, age, gender, member segment
- Output: top K specialty predictions
- Purpose: interpretable accuracy floor — establishes whether any signal exists

---

### Model 2 — SASRec

- Self-attention over visit sequence
- Input: prior diagnosis sequence + age + gender
- Output: top K specialty predictions across T30, T60, T180
- Captures visit order without temporal gap encoding

---

### Model 3 — BERT4Rec

- Bidirectional transformer with masked item prediction
- Input: prior diagnosis sequence + age + gender
- Output: top K specialty predictions
- Stronger on longer sequences — bidirectional context

---

### Model 4 — HSTU

- Hierarchical Sequential Transduction Units with delta-t modulation
- Input: prior diagnosis sequence + age + gender + delta-t buckets
- Output: top K specialty predictions across T30, T60, T180
- Native temporal gap encoding — clinically meaningful for healthcare sequences

#### Why HSTU

| Criterion | HSTU | SASRec |
|---|---|---|
| Irregular time gaps | Native delta-t encoding | Not handled natively |
| Feature interaction | Explicit gating via U(X) | No interaction layer |
| Attention normalization | Per-element SiLU | Row-wise softmax |

In healthcare, when a visit occurred relative to prior visits is as clinically meaningful
as what the visit was. A CAD diagnosis 6 months ago carries different weight than
the same diagnosis 2 weeks ago. HSTU captures this natively.

#### Embeddings

| Modality | Dimension |
|---|---|
| Diagnosis (ICD-10) | 32 |
| Age | 32 |
| Gender | 32 |
| delta-t bucket | 32 |

```
Diagnosis embedding   →  32-dim
Age embedding         →  32-dim
Gender embedding      →  32-dim
                      =  96-dim visit token

+ delta_t bucket      →  32-dim HSTU rating embedding
                      =  128-dim HSTU input
```

#### HSTU Configuration

| Parameter | Value |
|---|---|
| max_sequence_len | 20 |
| embedding_dim | 96 |
| hstu_dim | 128 |
| num_blocks | 2 |
| num_heads | 4 |
| linear_dim | 128 |
| attention_dim | 64 |
| dropout_rate | 0.2 |
| rating_embedding_dim | 32 |
| num_ratings | 16 |

#### Prediction Heads

```
Shared HSTU Backbone (128-dim context vector)
|-- Head T30   →  sigmoid  →  binary vector over specialty label space
|-- Head T60   →  sigmoid  →  binary vector over specialty label space
'-- Head T180  →  sigmoid  →  binary vector over specialty label space
```

Sigmoid not softmax — multiple labels can be true simultaneously.
Loss = BCE(T30) + BCE(T60) + BCE(T180)

---

### Model 5 — HSTU with RL Penalty (Extension)

- HSTU backbone with reinforcement learning layer on top
- Reward signal based on:
  - Correctness of specialty prediction
  - Provider quality within predicted specialty
  - Specialty appropriateness for diagnosis
- Penalty applied for predictions that route to low-quality or inappropriate providers
- Purpose: move from pure predictive accuracy to clinically optimized recommendations

---

## Training Strategy

| Decision | Choice |
|---|---|
| Data split | Time-based — pre-2024 train, 2024 onwards test |
| Sample size | 5% of members for initial run |
| Class imbalance | Weighted BCE — inverse frequency per label |
| Sequence cap | Last 20 visits |
| Precision | FP16 |
| Batch size | 512 |
| Early stopping | 5 epochs no validation improvement |

---

## Evaluation Metrics

All metrics computed for K = 1, 3, 5 across T30, T60, T180.

| Metric | Definition |
|---|---|
| Hit@K | 1 if actual specialty appears in top K predictions |
| Precision@K | Relevant items in top K / K |
| Recall@K | Relevant items in top K / total relevant items |
| NDCG@K | Hit weighted by rank — rank 1 hit scores higher than rank K |

Evaluated per cohort (Adult Female, Adult Male, Senior, Children) and overall.

**Baseline:** Markov transition probability model establishes the performance floor.
All four models compared against Markov baseline. A model must meaningfully exceed
Markov to justify its complexity.

---

## QA and Validation

### Data QA
- Row count checks at each pipeline stage
- Null rate checks on ICD-10, specialty, member_id
- Duplicate check on member + visit date + dx
- Train/test leakage check — no test triggers in training data

### Model QA
- Prediction distribution check — model should not predict same specialty for all members
- Label coverage — all test labels appear in training
- Per-cohort performance — no cohort significantly lower than overall
- Clinical spot check — top 20 to 30 predictions reviewed for coherence

### Clinical Validation
PCP visit with new diabetes diagnosis should predict Endocrinology within T30
and Lab within T30. Reviewed with network team or clinical SME.

---

## Infrastructure

| Component | Detail |
|---|---|
| GPU | 2x Nvidia T4 (16GB VRAM each, 32GB total) |
| BQ Project | anbc-hcb-dev |
| BQ Dataset | provider_ds_netconf_data_hcb_dev |
| HSTU Source | github.com/facebookresearch/generative-recommenders |
| PyTorch | 2.10.0 |
| Data Scale | 5% sample — approximately 1.5M to 2.5M member sequences |
