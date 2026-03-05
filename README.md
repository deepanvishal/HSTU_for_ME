# Provider Affiliation & Next Visit Prediction
**ML System Design Document — HSTU Architecture**  
Healthcare Data Science

---

## 1. Business Problem

Insurance claims data is used to understand provider affiliations and predict what happens after a member visits a provider with a specific diagnosis. The network team needs to identify high-value referral chains (e.g. PCP John referring to Specialist Jacob) so they can target upstream referrers who drive members to high-quality, efficient downstream providers.

---

## 2. Required Output Levels

One model trained at member journey level. Outputs aggregated to four levels:

| Level | Input | Output |
|---|---|---|
| 1 | Provider Type + Diagnosis | Next visit specialty at T+30, T+60, T+180 |
| 2 | Provider John + Diagnosis | Next visit specialty at T+30, T+60, T+180 |
| 3 | Provider Type + Diagnosis + Member Profile | Risk-adjusted next visit specialty |
| 4 | Provider John + Diagnosis + Member Profile | Most specific prediction |

Levels 1 and 2 are statistical aggregations of the member-level model output. No separate model needed.

---

## 3. Data Sources

| Table | Key Fields |
|---|---|
| Medical Claims | member_id, claim_id, service_date, provider_id, provider_type, place_of_service, diagnosis_code (one per claim), procedure_code, gender |
| Provider Specialty | provider_id, specialty_code, specialty_name |

No pharmacy claims. No member eligibility table. No line of business. Everything derives from these two tables.

---

## 4. Data Preparation Pipeline

### 4.1 Scope Filter

- Keep: service_date within 24-month lookback window
- Keep: provider_type in 20 specialties + PCP (joined to specialty table)
- Remove: ER place_of_service (POS 23)
- Remove: duplicate/adjusted claims — keep latest version
- Remove: members with fewer than 10 visits total

### 4.2 Visit Construction

Same member + same date = 1 visit. Group claims by member_id + service_date.

| Field | Aggregation Logic |
|---|---|
| provider_id | All qualifying providers on that date (in specialty scope) |
| dx_list | All diagnosis codes — one dx per claim, collected across claims on that date |
| procedure_codes | All procedure codes on that date |
| gender | From claims — mode if conflicting across claims |
| POS | Highest acuity if multiple on same date |

### 4.3 Co-visit Handling

When multiple qualifying providers appear on the same date for the same member:

- Each provider is embedded separately into a 128-dim vector
- Vectors are mean pooled into a single 128-dim provider representation for that visit
- HSTU sees one token per date — no duplicate timestamps in the sequence

Tradeoff: co-visit signal preserved but individual provider identity is diluted. Accepted for Phase 1. Revisit if provider-level prediction accuracy is poor.

### 4.4 Member Profile Snapshot

Built as of each visit date using only prior claims history. No imputation. Profile fills progressively as visits accumulate. Zero vector for cold start (no prior history before Visit 1).

| Feature | Logic |
|---|---|
| Comorbidity flags | Rolling 12-month lookback. ICD codes grouped via CCSR or CCI grouper. Binary flag per condition category. |
| Visit history features | Total visits, distinct specialties, distinct providers seen in last 12 months |
| Member embedding input | 15,000-dim binary vector of all historical ICD codes as presence/absence flags |

### 4.5 Label Construction

Multi-label classification. Windows are cumulative — T+60 includes all visits within 60 days including those within T+30.

| Label | Logic |
|---|---|
| T+30 | Binary flag per specialty — did member visit this specialty within 30 days of visit date T |
| T+60 | Cumulative — all specialties visited within 60 days of T |
| T+180 | Cumulative — all specialties visited within 180 days of T |

### 4.6 Delta-T Bucketing

Days since last visit is log-scale bucketed using Meta's HSTU paper formula:

```
bucket(delta_t) = floor( log(max(1, |delta_t|)) / 0.301 )
```

Log scale provides finer granularity for recent gaps and coarser for distant gaps. The bucket integer is fed into HSTU's rating embedding layer as the time signal.

---

## 5. Embeddings

| Embedding | Dim | Method | Dynamic? |
|---|---|---|---|
| Provider | 128 | Random walk on provider co-occurrence graph (Word2Vec style). Providers in similar member journeys are close in embedding space. | Yes — rolled by visit date |
| Diagnosis (visit-level) | 64 | Random walk on ICD code co-occurrence graph. Codes co-occurring within member visits are close in space. Mean pooled across dx_list at visit time. | Per visit |
| Member Profile | 128 | 15,000 binary ICD flags compressed via embedding layer trained end-to-end with model. | Yes — snapshot prior to each visit date |

### 5.1 Final Visit Token

```
Provider embedding   ->  128-dim  (rolling by date)
Member embedding     ->  128-dim  (snapshot prior to visit)
Diagnosis embedding  ->   64-dim  (mean pool of dx_list codes)
                     = 320-dim token per visit

+ delta_t bucket     ->  scalar   -> HSTU rating embedding (separate)
```

---

## 6. HSTU Architecture

### 6.1 Why HSTU

| Criterion | HSTU | SASRec |
|---|---|---|
| Irregular time gaps | Native time encoding | Not handled natively |
| Feature interaction | Explicit gating — U(X) mechanism | No explicit interaction layer |
| Attention normalization | Per-element SiLU | Row-wise softmax |
| Availability | GitHub only — no PyPI | pip installable via RecBole |

### 6.2 Four HSTU Components

**Component 1: Time Encoding**  
delta_t bucket fed into learned embedding. Produces a time vector that modulates how much attention each past visit receives. Recent visits weighted higher.

**Component 2: Time-Modulated Self Attention**  
Every token attends to every other token in sequence. Attention scores multiplied by time decay weight. Model learns the decay function from data — a CAD diagnosis 6 months ago may still receive high attention if clinically relevant.

**Component 3: Feature Interaction**  
Within each 320-dim token, learned cross-interactions between provider, member, and diagnosis components via elementwise gating (U(X) mechanism). This is where the business question lives: Provider John + Diagnosis XYZ + Member Profile produces a specific interaction signal different from any other provider seeing the same diagnosis.

**Component 4: Prediction Heads**

```
Shared HSTU Backbone (320-dim context vector)
|-- Head T+30  -> sigmoid -> binary vector over specialties
|-- Head T+60  -> sigmoid -> binary vector over specialties
'-- Head T+180 -> sigmoid -> binary vector over specialties
```

Sigmoid not softmax — multiple specialties can be true simultaneously.  
Loss = BCE(T+30) + BCE(T+60) + BCE(T+180)

### 6.3 HSTU Configuration

| Parameter | Value | Notes |
|---|---|---|
| max_sequence_len | 20 | Cap member history at last 20 visits |
| embedding_dim | 320 | Token dimension |
| num_blocks | 2 | HSTU layers — start conservative |
| num_heads | 4 | Attention heads |
| linear_dim | 128 | Feedforward layer size |
| attention_dim | 64 | Attention projection |
| dropout_rate | 0.2 | Both linear and attention dropout |
| rating_embedding_dim | 32 | delta_t bucket embedding size |
| enable_relative_attention_bias | True | Critical for temporal encoding |

### 6.4 SequentialFeatures Input

Data structure mapping our healthcare concepts to Meta's HSTU SequentialFeatures NamedTuple:

| Their Field | Their Concept | Our Concept |
|---|---|---|
| past_lengths | Sequence length per user | Number of visits per member |
| past_ids | Item IDs | provider_ids |
| past_embeddings | Item embeddings (None in their code) | Our precomputed 320-dim visit tokens |
| past_payloads['timestamps'] | Unix timestamps | Visit dates as unix timestamps |
| past_payloads['ratings'] | Star ratings | Log-scale bucketed delta_t |

```python
features = SequentialFeatures(
    past_lengths    = member_visit_counts,   # [B] int64
    past_ids        = provider_ids,          # [B, N] int64
    past_embeddings = visit_tokens,          # [B, N, 320] float
    past_payloads   = {
        'timestamps': visit_timestamps,      # [B, N] unix timestamps
        'ratings':    delta_t_buckets        # [B, N] log-scale buckets
    }
)
```

---

## 7. Training Strategy

| Decision | Choice | Reason |
|---|---|---|
| Data split | Time-based (not random) | Prevents future data leakage into training |
| Train | Months 1-18 of 24-month window | |
| Validation | Months 19-21 | |
| Test | Months 22-24 | |
| Class imbalance | Weighted BCE — inverse frequency per specialty | Prevents model from predicting only common specialties |
| Sequence cap | Last 20 visits per member | VRAM constraint on 2x T4 |
| Precision | FP16 (half precision) | Cuts VRAM in half |
| Batch size | 64-128 sequences | T4 VRAM limit |
| Early stopping | 5 epochs no val F1 improvement | |
| First run | 10% member sample | Validate architecture before full training |

---

## 8. Validation

### 8.1 Metrics

| Metric | What It Measures |
|---|---|
| Precision@K | Of top K predicted specialties — how many actually occurred |
| Recall@K | Of actual specialties visited — how many did model predict |
| F1@K | Balance of precision and recall |
| Hit Rate | Did model predict at least one correct specialty |

Evaluate separately for T+30, T+60, T+180. T+30 hardest, T+180 easiest. Holdout = last N visits per member.

### 8.2 Clinical Validation

Have network team or clinician review top 20-30 prediction examples for clinical coherence. Example: PCP visit with hypertension + newly flagged diabetes should predict Endocrinology and Cardiology within 180 days.

---

## 9. Infrastructure

| Component | Detail |
|---|---|
| GPU | 2x Nvidia T4 (16GB VRAM each, 32GB total) |
| HSTU source | github.com/facebookresearch/generative-recommenders — cloned locally with submodules |
| Install | git clone --recurse-submodules + pip install -e ./generative-recommenders |
| Import path | from generative_recommenders.research.modeling.sequential.hstu import HSTU |
| sys.path fix | sys.path.insert(0, './generative-recommenders') required |
| PyTorch version | 2.10.0 — conflicts with repo expected 1.13.1. Monitor for runtime errors. |
| CUDA version | 12.9.4 — conflicts with repo expected 11.8.x. Monitor for runtime errors. |
| Data scale | ~300M claim rows -> ~30-50M visits -> ~5-10M member sequences |

---

## 10. Open Questions and Next Steps

- Write data preparation code: claims -> visit table -> member snapshots -> SequentialFeatures
- Train provider embeddings via random walk on provider co-occurrence graph
- Train diagnosis embeddings via random walk on ICD code co-occurrence graph
- Inspect output_postproc_module options from output_preprocessors.py
- Wire embedding_module, similarity_module, output_postproc_module for HSTU init
- Run small-scale training on 10% sample — validate predictions clinically
- Evaluate whether co-visit mean pooling degrades provider-level prediction accuracy
- If degraded — revisit primary provider wins approach as fallback
