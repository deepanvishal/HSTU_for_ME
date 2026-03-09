# Provider Network Referral Prediction
**ML System Design — HSTU Architecture**
Healthcare Network Configuration | Data Science

---

## Business Objective

Insurance claims data is used to understand provider affiliations and predict what happens after a member visits a provider with a specific diagnosis. The network team needs to identify high-value referral chains — for example, PCP John referring to Specialist Jacob — so they can target upstream referrers who drive members to high-quality, efficient downstream providers.

One model trained at the member journey level. Outputs aggregated to four levels:

| Level | Input | Output |
|---|---|---|
| 1 | Provider Type + Diagnosis | Next visit specialty at T+30, T+60, T+180 |
| 2 | Provider John + Diagnosis | Next visit specialty at T+30, T+60, T+180 |
| 3 | Provider Type + Diagnosis + Member Profile | Risk-adjusted next visit specialty |
| 4 | Provider John + Diagnosis + Member Profile | Most specific prediction |

Levels 1 and 2 are statistical aggregations of the member-level model output. No separate model per level is required.

---

## Week 1 — Barriers

### Barrier 1 — Defining the First Visit

Identifying where a care episode begins is the most significant analytical challenge. Without a reliable first visit anchor, the sequence model learns from noise as much as signal.

A first visit should satisfy two conditions:
1. A qualifying provider contact — most reliably a PCP visit, though specialist-initiated episodes exist
2. A new or recurrent diagnosis — a code not seen in the member's claim history in the prior 12 months, or a chronic condition flagged as active

**Option A — PCP + New Diagnosis (Phase 1)**
A PCP encounter where a qualifying diagnosis code appears for the first time in a rolling 12-month lookback. Simple, defensible, fast.
Limitation: misses specialist-initiated or ER-initiated episodes.

**Option B — Episode of Care (Phase 2)**
Claims grouped into episodes using a 90-day gap rule. First claim in a new episode = first visit. Clinically accurate but requires condition-specific gap definitions.

Recommendation: Start with Option A. Tighten to Option B once model value is proven.

---

### Barrier 2 — Scope

Starting with a single specialty rather than the full provider network.

**Why:**
- Referral patterns within one specialty are more homogeneous and easier to validate
- Smaller label space produces measurable accuracy benchmarks early
- Clinical SME validation is tractable on a focused cohort
- Lessons transfer directly to the expansion framework

**Proposed starting specialty:** Cardiology or Oncology — high visit frequency, well-defined referral chains, interpretable pathways.

---

### Barrier 3 — Scale and Accuracy

Training on unfiltered full-population data across all conditions produces low accuracy. Initial benchmarks on 1% sample with unfiltered sequences: approximately 20% on next-visit prediction. Expected — not a model failure.

**Root cause:** Mixing all conditions forces the model to find a single representation for fundamentally different care journeys. Diabetic patient sequences look nothing like oncology sequences.

**How this is addressed — two-stage prediction:**

Stage 1 predicts the next likely specialty. Label space is small (~50 specialties), patterns are generalizable, accuracy is measurably higher.

Stage 2 maps specialty to provider. When provider-level data is sufficient, predict provider directly. When provider data is sparse, specialty prediction is the output — mapped to available in-network providers for that specialty and geography downstream.

The model degrades gracefully. Provider-level prediction is an enhancement, not a dependency. In thin markets, specialty prediction still produces actionable network output.

---

## Data Sources

| Table | Key Fields |
|---|---|
| Medical Claims | member_id, claim_id, service_date, provider_id, provider_type, place_of_service, diagnosis_code, procedure_code, gender |
| Provider Specialty | provider_id, specialty_code, specialty_name |

No pharmacy claims. No member eligibility table. No line of business. Everything derives from these two tables.

---

## Data Preparation

### Scope Filter
- Keep: service_date within 24-month lookback window
- Keep: last 20 visits per member (recency_rank <= 20) — HSTU only uses the last 20 visits; pulling more wastes compute
- Keep: provider_type in 20 specialties + PCP
- Remove: ER place_of_service (POS 23)
- Remove: duplicate/adjusted claims — keep latest version per claim
- Remove: members with fewer than 10 visits total

### Visit Construction

Same member + same date = one visit. Claims grouped by member_id + service_date.

| Field | Aggregation |
|---|---|
| provider_ids | All qualifying providers on that date |
| specialty_codes | All specialties on that date |
| dx_list | All diagnosis codes — one per claim, collected across claims |
| procedure_codes | All procedure codes on that date |
| delta_t_bucket | Log-scale bucket of days since prior visit |

### Co-visit Handling

Multiple qualifying providers on the same date for the same member: each provider embedded separately, vectors mean pooled into a single visit representation. HSTU sees one token per date — no duplicate timestamps.

Tradeoff: co-visit signal preserved but individual provider identity diluted. Accepted for Phase 1.

### Label Construction

Multi-label classification. Windows are cumulative.

| Label | Logic |
|---|---|
| T+30 | All specialties (or providers, or dx codes) visited within 30 days of anchor visit |
| T+60 | Cumulative — all within 60 days |
| T+180 | Cumulative — all within 180 days |

Three targets trained separately: specialty, provider, dx. Sequence cache is shared across targets. Only label files differ per target.

### Delta-T Bucketing

Log-scale bucketing per Meta's HSTU paper:

```
bucket(delta_t) = floor( log(max(1, |delta_t|)) / 0.301 )
```

Finer granularity for recent gaps, coarser for distant gaps. Bucket integer fed into HSTU's rating embedding layer as the time signal.

---

## EDA Plan

| Method | Purpose |
|---|---|
| Markov Chain Transition Matrix | First-order specialty-to-specialty and dx-to-dx transition probabilities. Visualized as heatmap or network graph. Fast, interpretable, good for stakeholder presentation. |
| Sequential Pattern Mining (PrefixSpan) | Identifies frequent multi-step referral chains. Example: Cardiology → Echo → Interventional Cardiology in X% of members with I25. Captures chains Markov misses. |
| Co-occurrence Analysis | Top dx code pairs and triplets appearing within the same 30/60/180 day window. Surfaces condition clusters and referral triggers. |
| Community Detection (Louvain) | Applied to provider co-occurrence graphs to surface provider clusters that co-refer. Maps informal referral networks not in formal agreements. |
| Time-Segmented Analysis | Compare transition frequencies at 30, 60, 180 day windows. Demonstrates shorter windows have stronger signal — justifies prediction task design. |

---

## Model Methodology

### Algorithm Progression

| Stage | Model | Purpose |
|---|---|---|
| Baseline | Logistic Regression / XGBoost | Visit-level features, no sequence modeling. Sets accuracy floor. Expected hit@5: 25–35%. |
| Stage 2 | SASRec | Self-attention sequential model. Captures visit order. Faster than transformer models. |
| Stage 3 | BERT4Rec | Bidirectional transformer. Stronger on longer sequences. Appropriate for episode-bounded sequences in Phase 2. |
| Stage 4 | HSTU | Explicit temporal gap encoding. Primary model. |

---

### HSTU — Model Design

#### Why HSTU

| Criterion | HSTU | SASRec |
|---|---|---|
| Irregular time gaps | Native delta_t encoding | Not handled natively |
| Feature interaction | Explicit gating via U(X) mechanism | No interaction layer |
| Attention normalization | Per-element SiLU | Row-wise softmax |

Standard recommendation models treat sequences as ordered but do not encode time between events. In healthcare referral prediction, *when* a visit occurred relative to the prior visit is as clinically meaningful as *what* the visit was — a CAD diagnosis 6 months ago carries different weight than the same diagnosis 2 weeks ago. HSTU captures this natively.

---

#### Embeddings

Four modalities embedded via SVD on co-occurrence graphs. Each modality produces a 32-dim vector. Mean pooled per visit across all codes in that modality.

| Modality | Dim | Graph | Notes |
|---|---|---|---|
| Provider | 32 | Provider co-occurrence — providers appearing in same member journey | Mean pool across co-visiting providers |
| Specialty | 32 | Specialty co-occurrence graph | Mean pool across specialties on visit date |
| Diagnosis | 32 | ICD code co-occurrence within member visits | Mean pool across dx_list |
| Procedure | 32 | Procedure co-occurrence within member visits | Mean pool across procedure_codes |

```
Provider embedding    →  32-dim
Specialty embedding   →  32-dim
Diagnosis embedding   →  32-dim
Procedure embedding   →  32-dim
                      = 128-dim visit token

+ delta_t bucket      →  scalar  →  HSTU rating embedding (32-dim, separate)
```

Total HSTU input dimension: 160 (128 visit token + 32 delta_t embedding).

**Phase 2 extension — Member Profile:**
A member snapshot embedding (128-dim) built from historical ICD code presence/absence, trained end-to-end with the model, will extend the visit token to 256-dim. Not in current implementation.

---

#### HSTU Configuration

| Parameter | Value | Notes |
|---|---|---|
| max_sequence_len | 20 | Last 20 visits per member |
| embedding_dim | 128 | Visit token dimension (4 × 32) |
| hstu_dim | 160 | 128 + 32 delta_t |
| num_blocks | 2 | HSTU layers |
| num_heads | 4 | Attention heads |
| linear_dim | 128 | Feedforward layer size |
| attention_dim | 64 | Attention projection |
| dropout_rate | 0.2 | Applied to linear and attention layers |
| rating_embedding_dim | 32 | delta_t bucket embedding size |
| num_ratings | 16 | Number of delta_t buckets |

---

#### HSTU Components

**Component 1 — Time Encoding**
delta_t bucket fed into a learned embedding. Produces a time vector that modulates attention weight. Recent visits weighted higher by default; model learns the decay function from data.

**Component 2 — Time-Modulated Self Attention**
Every token attends to every other token in the sequence. Attention scores multiplied by time decay weight. Clinically relevant — a diagnosis from 6 months ago may still receive high attention if it predicts a known delayed referral pattern.

**Component 3 — Feature Interaction**
Within each visit token, learned cross-interactions between provider, specialty, diagnosis, and procedure components via elementwise gating (U(X) mechanism). This is where the core business question lives: Provider John + Diagnosis XYZ produces a specific interaction signal different from any other provider seeing the same diagnosis.

**Component 4 — Prediction Heads**

```
Shared HSTU Backbone (128-dim context vector)
|-- Head T+30   →  sigmoid  →  binary vector over label space
|-- Head T+60   →  sigmoid  →  binary vector over label space
'-- Head T+180  →  sigmoid  →  binary vector over label space
```

Sigmoid not softmax — multiple labels can be true simultaneously.
Loss = BCE(T+30) + BCE(T+60) + BCE(T+180)

---

#### SequentialFeatures Input

Mapping healthcare concepts to HSTU's SequentialFeatures NamedTuple:

| HSTU Field | Our Concept |
|---|---|
| past_lengths | Number of visits per member |
| past_ids | provider_ids |
| past_embeddings | Precomputed 128-dim visit tokens |
| past_payloads['timestamps'] | Visit dates as unix timestamps |
| past_payloads['ratings'] | Log-scale bucketed delta_t |

---

### Training Strategy

| Decision | Choice | Reason |
|---|---|---|
| Data split | Time-based | Prevents future data leakage |
| Train | Months 1–18 of 24-month window | |
| Validation | Months 19–21 | |
| Test | Months 22–24 | |
| Class imbalance | Weighted BCE — inverse frequency per label | Prevents prediction of only common specialties |
| Sequence cap | Last 20 visits | VRAM constraint |
| Precision | FP16 | Cuts VRAM in half |
| Batch size | 512 | Tuned for 2× T4 |
| Early stopping | 5 epochs no val F1 improvement | |
| First run | 10% member sample | Validate architecture before full training |

---

### Validation

| Metric | What It Measures |
|---|---|
| Precision@K | Of top K predicted labels — how many actually occurred |
| Recall@K | Of actual labels — how many did the model predict |
| F1@K | Balance of precision and recall |
| Hit Rate@K | Did model predict at least one correct label in top K |

Evaluated separately for T+30, T+60, T+180. T+30 is the hardest window, T+180 the easiest.

**Clinical Validation:**
Network team or clinical SME reviews top 20–30 prediction examples for coherence. Example: PCP visit with hypertension + newly flagged diabetes should predict Endocrinology and Cardiology within 180 days.

---

## Infrastructure

| Component | Detail |
|---|---|
| GPU | 2× Nvidia T4 (16GB VRAM each, 32GB total) |
| HSTU source | github.com/facebookresearch/generative-recommenders |
| PyTorch | 2.10.0 — repo expects 1.13.1, monitor for runtime errors |
| CUDA | 12.9.4 — repo expects 11.8.x, monitor for runtime errors |
| Data scale | ~300M claims → ~30–50M visits → ~5–10M member sequences → ~33M rows after recency_rank <= 20 filter |
| BQ project | anbc-hcb-dev |
| BQ dataset | provider_ds_netconf_data_hcb_dev |

---

## Next Steps

1. Align on starting specialty with manager and clinical SME
2. Finalize first visit definition — Option A for Phase 1
3. Run EDA on scoped cohort — Markov transitions and co-occurrence as deliverables
4. Establish baseline accuracy with logistic regression before presenting HSTU results
5. Define accuracy targets for Phase 1 sign-off — recommended hit@5 > 40% on specialty prediction
6. Evaluate co-visit mean pooling impact on provider-level prediction accuracy
7. If provider accuracy degrades — evaluate primary provider wins as fallback
