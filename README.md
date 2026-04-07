# Provider Network Referral Prediction

Next visit specialty and provider prediction using sequential recommendation models (SASRec, BERT4Rec, HSTU) on South Florida medical claims data.

## Repository Structure

```
├── 01_data_foundation/          Claims → visits → triggers → sequences
│   ├── _01_data_setup_members.sql
│   ├── _02_data_setup_visits.sql
│   ├── _03_data_setup_qualified_visits.sql
│   └── _04_visit_sequences.sql
├── 02_specialty_preprocessing/   Sampling, train/test split, sequence tables
│   ├── Model_data_setup_01.sql
│   ├── Model_sampling.sql
│   ├── Model_train_test_split.sql
│   └── model_input_sequence_data.sql
├── 03_specialty_models/          Specialty-level training, scoring, baseline
│   ├── Model_build_train_dataset.py
│   ├── Model_build_test_dataset.py
│   ├── Model_Train_Test_split_QA.py
│   ├── Model_BERT4Rec.py
│   ├── Model_SASRec.py
│   ├── Model_SASRec_Score.py
│   ├── Model_Bert4Rec_score.py
│   ├── Model_Markov.sql
│   └── Model_Markov_score.sql
├── 04_provider_preprocessing/    Provider vocab, train/test, sequences
│   ├── 03_provider_transitions_and_cutoff.sql
│   ├── sql01b_hard_negative
│   ├── sql02_provider_train_test.sql
│   └── sql03_provider_sequence_input.sql
├── 05_provider_models/           Provider-level training, scoring, baseline
│   ├── nb01_build_provider_vocab.py
│   ├── nb02_build_provider_train_dataset.py
│   ├── nb03_build_provider_test_dataset.py
│   ├── nb04_Model_SASRec_provider.py
│   ├── nb05_Model_BERT4rec_provider.py
│   ├── nb06_Model_HSTU_provider.py
│   ├── nb07_scoring_notebook.py
│   └── sql04_markov_baseline.sql
├── 06_evaluation/                Model evaluation and post-model analysis
│   ├── specialty/
│   │   ├── Post_model_analysis_data_setup1.sql
│   │   └── Model_metrics_at_5_comp.sql
│   └── provider/
│       ├── sql05_provider_pred_eval.sql
│       ├── sql06_prov_visit_level.sql
│       ├── SQL_PMA_00_lookup.sql
│       ├── SQL_PMA_01 through PMA_04
│       ├── SQL_PMA_rollup.sql
│       ├── SQL_PMA_rollup2.sql
│       ├── NB_PMA01_post_model_analysis.py
│       └── NB_PMA02_post_model_analysis2.py
├── docs/
│   ├── data_dictionary.md
│   └── model_manual_v2.docx
├── requirements.txt
└── README.md
```

## Environment

- Python 3.10+
- PyTorch 2.0+
- 2x Nvidia T4 (32GB VRAM total)
- Google Cloud BigQuery access to `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev`

```bash
pip install -r requirements.txt
```

## Data Source

All models built on `A870800_claims_gen_rec_2022_2025_sfl` — CVS/Aetna medical claims, South Florida commercial population, 2022-2025.

## Run Order

Execute in this exact order. Each step depends on the prior step completing.

### Phase 1 — Data Foundation (BigQuery)

| Step | File | Output |
|------|------|--------|
| 1 | `01_data_foundation/_01_data_setup_members.sql` | member_qualified |
| 2 | `01_data_foundation/_02_data_setup_visits.sql` | visits |
| 3 | `01_data_foundation/_03_data_setup_qualified_visits.sql` | triggers_qualified, visits_qualified |
| 4 | `01_data_foundation/_04_visit_sequences.sql` | model_input_sequences |

### Phase 2 — Specialty Preprocessing (BigQuery)

| Step | File | Output |
|------|------|--------|
| 5 | `02_specialty_preprocessing/Model_data_setup_01.sql` | model_train, model_test, markov_train |
| 6 | `02_specialty_preprocessing/Model_sampling.sql` | sample_members_1/5/10pct |
| 7 | `02_specialty_preprocessing/Model_train_test_split.sql` | model_train/test_Xpct, markov_train_Xpct |
| 8 | `02_specialty_preprocessing/model_input_sequence_data.sql` | train/test_sequences_Xpct |

### Phase 3 — Specialty Models (Python + BigQuery)

| Step | File | Output |
|------|------|--------|
| 9 | `03_specialty_models/Model_build_train_dataset.py` | numpy caches + vocab.pkl |
| 10 | `03_specialty_models/Model_build_test_dataset.py` | test numpy caches |
| 11 | `03_specialty_models/Model_Train_Test_split_QA.py` | validation (gate) |
| 12 | `03_specialty_models/Model_BERT4Rec.py` | bert4rec checkpoint |
| 13 | `03_specialty_models/Model_SASRec.py` | sasrec checkpoint |
| 14 | `03_specialty_models/Model_Markov.sql` | markov_predictions (BQ) |
| 15 | `03_specialty_models/Model_Markov_score.sql` | markov_trigger_scores (BQ) |
| 16 | `03_specialty_models/Model_SASRec_Score.py` | trigger_scores (BQ) |
| 17 | `03_specialty_models/Model_Bert4Rec_score.py` | trigger_scores (BQ) |

### Phase 4 — Provider Models (BigQuery + Python)

| Step | File | Output |
|------|------|--------|
| 18 | `04_provider_preprocessing/03_provider_transitions_and_cutoff.sql` | provider_vocab, provider_transitions |
| 19 | `04_provider_preprocessing/sql01b_hard_negative` | primary_specialty, hardneg_lookup |
| 20 | `04_provider_preprocessing/sql02_provider_train_test.sql` | provider train/test/markov tables |
| 21 | `04_provider_preprocessing/sql03_provider_sequence_input.sql` | provider sequences |
| 22 | `05_provider_models/sql04_markov_baseline.sql` | provider markov predictions |
| 23 | `05_provider_models/nb01_build_provider_vocab.py` | vocab .pkl files |
| 24 | `05_provider_models/nb02_build_provider_train_dataset.py` | train numpy caches |
| 25 | `05_provider_models/nb03_build_provider_test_dataset.py` | test numpy caches |
| 26 | `05_provider_models/nb04_Model_SASRec_provider.py` | sasrec_provider checkpoint |
| 27 | `05_provider_models/nb05_Model_BERT4rec_provider.py` | bert4rec_provider checkpoint |
| 28 | `05_provider_models/nb06_Model_HSTU_provider.py` | hstu_provider checkpoint |
| 29 | `05_provider_models/nb07_scoring_notebook.py` | provider_trigger_scores (BQ) |

### Phase 5 — Evaluation (BigQuery + Python)

| Step | File | Output |
|------|------|--------|
| 30 | `06_evaluation/provider/sql05_provider_pred_eval.sql` | provider_eval_5pct |
| 31 | `06_evaluation/provider/sql06_prov_visit_level.sql` | provider_eval_visit_5pct |
| 32 | `06_evaluation/provider/SQL_PMA_00_lookup.sql` | provider_name_lookup |
| 33-36 | `06_evaluation/provider/SQL_PMA_01 through PMA_04, rollup, rollup2` | PMA tables |
| 37 | `06_evaluation/specialty/Post_model_analysis_data_setup1.sql` | specialty perf tables |
| 38 | `06_evaluation/specialty/Model_metrics_at_5_comp.sql` | analysis_perf_full |

## Results

| Level | Best Model | Hit@5 at T30 |
|-------|-----------|-------------|
| Specialty | BERT4Rec | ~85% |
| Provider | SASRec | ~46% |

## Documentation

See `docs/model_manual_v2.docx` for the full technical handoff including table lineage, architecture details, glossary, limitations, and improvement roadmap.

See `docs/data_dictionary.md` for table schemas.
