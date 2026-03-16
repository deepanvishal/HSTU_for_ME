# Code Index
## Next Visit Prediction — HSTU
### GitHub: https://github.com/deepanvishal/HSTU_for_ME

---

## Repository Files

| Type | File | Purpose |
|---|---|---|
| SQL | `Graphs_counting.sql` | Provider co-occurrence graph edge counting for embedding generation |
| SQL | `Inference_data.sql` | Data preparation for model inference |
| SQL | `Visits.sql` | Visit construction from raw claims |
| SQL | `label.sql` | Label generation for T30, T60, T180 targets |
| SQL | `member_profile_snapshot.sql` | Member demographic and clinical profile snapshot |
| SQL | `sequence.sql` | Sequential visit data preparation for HSTU input |
| Python | `HSTU_model.py` | HSTU model architecture definition |
| Python | `Hstu_import.py` | HSTU library imports and configuration |
| Python | `data_prep.py` | Data preparation pipeline — visit construction, feature engineering |
| Python | `hstu_pytorch.py` | PyTorch HSTU implementation |
| Python | `notebook0_eda.py` | Initial EDA notebook |
| Python | `past_embeddings.py` | Pre-computed visit embedding generation via SVD |
| Python | `validation.py` | Model validation and metric computation |

---

## EDA Files (This Project)

| Type | File | Purpose |
|---|---|---|
| SQL | `scope_stats.sql` | South Florida scope statistics — member, provider, claim, visit counts |
| SQL | `data_quality.sql` | Data quality coverage — ICD-10, specialty, CCSR mapping rates |
| SQL | `entropy_comparison.sql` | Entropy comparison across 9 combinations and 3 analytical lenses |
| SQL | `order1_transitions.sql` | 9 Order 1 Markov transition tables |
| SQL | `order2_transitions.sql` | 9 Order 2 Markov transition tables |
| SQL | `entropy_summary.sql` | Weighted average entropy summary across all 18 tables |
| SQL | `track1_base.sql` | Track 1 base table — member + trigger + downstream visits as ARRAY STRUCT |
| SQL | `track1_summary.sql` | Track 1 summary — first specialist per window aggregated |
| SQL | `track1_penetration.sql` | Penetration rate and binary entropy per specialty per window |
| SQL | `markov_baseline.sql` | Markov baseline train/test split and prediction tables |
| Python | `EDA_07_plots.py` | Order 1 EDA — Parts 1, 2, 3 — Diagnosis to Specialty, CCSR, Diagnosis |
| Python | `EDA_08_order2_dx_to_specialty.py` | Order 2 EDA — 3-layer network, heatmap, entropy bar chart |
| Python | `EDA_T30.py` | Time window EDA — T30 — penetration rate, binary entropy, network |
| Python | `EDA_T60.py` | Time window EDA — T60 |
| Python | `EDA_T180.py` | Time window EDA — T180 |
| Python | `EDA_markov_baseline.py` | Markov baseline metrics — Hit@K, Precision@K, Recall@K, NDCG@K |
| Markdown | `eda_plan.md` | EDA plan — purpose, decisions, findings, limitations |
| Markdown | `data_dictionary.md` | This file — raw sources and derived table definitions |
| Markdown | `modeling_architecture.md` | Model objective, features, architecture, evaluation |
| Markdown | `code_index.md` | This file — all code files and purpose |
