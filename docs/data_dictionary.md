# Data Dictionary

All tables in `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev`. Prefix: `A870800_gen_rec_`.

---

## External Sources

| Table | Description |
|---|---|
| `A870800_claims_gen_rec_2022_2025_sfl` | Medical claims, South Florida, 2022-2025. One row per claim line. |
| `A870800_claims_gen_rec_members` | Member enrollment records with eff_dt, age, gender, geography. |
| `edp-prod-hcbstorage...AHRQ_CCSR_DX_20260101` | AHRQ CCSR ICD-10 → clinical category mapping. |
| `edp-prod-hcbstorage...GLOBAL_LOOKUP` | Specialty code → description lookup. Filter: lookup_column_nm = 'specialty_ctg_cd'. |
| `edp-prod-hcbstorage...ICD9_DIAGNOSIS` | ICD-10 code → description lookup. |

---

## 01 — Data Foundation

### member_qualified

One row per member_id. Created by `_01`.

| Column | Type | Description |
|---|---|---|
| member_id | STRING | Unique member identifier |
| enrollment_start | DATE | MIN(eff_dt) |
| enrollment_end | DATE | MAX(eff_dt) |
| enrollment_window_months | INT64 | DATE_DIFF(end, start, MONTH) |
| enrolled_months | INT64 | COUNT(DISTINCT eff_dt) |
| zip_cd | STRING | MAX(zip_cd) |
| state_postal_cd | STRING | MAX |
| county_cd | STRING | MAX |
| county_nm | STRING | MAX |
| market | STRING | MAX |
| submarket | STRING | MAX |

### visits

One row per (member_id, visit_date, srv_prvdr_id, specialty_ctg_cd, dx_raw). Created by `_02`.

| Column | Type | Description |
|---|---|---|
| member_id | STRING | |
| visit_date | DATE | srv_start_dt renamed |
| visit_rank | INT64 | DENSE_RANK per member by visit_date |
| srv_prvdr_id | STRING | Provider ID |
| specialty_ctg_cd | STRING | Specialty code |
| specialty_desc | STRING | From GLOBAL_LOOKUP join |
| dx_raw | STRING | pri_icd9_dx_cd (contains ICD-10) |
| dx_clean | STRING | Periods removed |
| ccsr_category | STRING | From AHRQ CCSR join |
| ccsr_category_description | STRING | |
| plc_srv_cd | STRING | Place of service |
| med_cost_ctg_cd | STRING | Medical cost category |
| age_nbr | INT64 | |
| gender_cd | STRING | |
| member_segment | STRING | Children / Adult_Female / Adult_Male / Senior |
| allowed_amt | FLOAT64 | SUM from claims dedup |

### triggers_qualified

One row per (member_id, trigger_date, trigger_dx). Created by `_03`.

| Column | Type | Description |
|---|---|---|
| member_id | STRING | |
| trigger_date | DATE | First encounter date for this dx |
| trigger_rank | INT64 | Rank of trigger by date per member |
| trigger_dx | STRING | Raw ICD-10 |
| trigger_dx_clean | STRING | Periods removed |
| trigger_ccsr | STRING | CCSR category |
| trigger_ccsr_desc | STRING | |
| trigger_specialty | STRING | Specialty at trigger visit |
| trigger_specialty_desc | STRING | |
| age_nbr | INT64 | |
| gender_cd | STRING | |
| member_segment | STRING | |
| enrollment_start | DATE | From member_qualified |
| enrollment_end | DATE | |
| enrolled_months | INT64 | |
| enrollment_window_months | INT64 | |
| rule1_enrolled_12m | BOOL | DATE_DIFF(trigger, enrollment_start) >= 365 |
| rule2_dx_not_seen_12m | BOOL | NOT EXISTS same dx in prior 12m |
| has_claims_12m_before | BOOL | Any claims in 12m lookback (informational) |
| is_left_qualified | BOOL | rule1 AND rule2 |
| is_t30_qualified | BOOL | left + T30 right boundary |
| is_t60_qualified | BOOL | left + T60 right boundary |
| is_t180_qualified | BOOL | left + T180 right boundary |

### visits_qualified

One row per trigger + downstream visit. Created by `_03`.

| Column | Type | Description |
|---|---|---|
| member_id | STRING | |
| trigger_date | DATE | |
| trigger_dx | STRING | |
| trigger_specialty | STRING | |
| member_segment | STRING | |
| is_left_qualified | BOOL | |
| is_t30_qualified | BOOL | |
| is_t60_qualified | BOOL | |
| is_t180_qualified | BOOL | |
| visit_date | DATE | Downstream visit date |
| downstream_visit_rank | INT64 | |
| days_since_trigger | INT64 | DATE_DIFF(visit, trigger, DAY) |
| srv_prvdr_id | STRING | Downstream provider |
| specialty_ctg_cd | STRING | Downstream specialty |
| specialty_desc | STRING | |
| dx_raw | STRING | |
| allowed_amt | FLOAT64 | |

### model_input_sequences

One row per (member, trigger, label_specialty, time_bucket). Created by `_04`.

| Column | Type | Description |
|---|---|---|
| member_id | STRING | |
| trigger_date | DATE | |
| trigger_dx | STRING | |
| trigger_dx_clean | STRING | |
| trigger_ccsr | STRING | |
| trigger_specialty | STRING | |
| member_segment | STRING | |
| age_nbr | INT64 | |
| gender_cd | STRING | |
| is_t30/t60/t180_qualified | BOOL | |
| has_claims_12m_before | BOOL | |
| visit_sequence | ARRAY<STRUCT> | Pre-trigger visits (180d lookback), ordered by visit_date |
| label_specialty | STRING | Specialty visited post-trigger |
| label_specialty_desc | STRING | |
| time_bucket | STRING | T0_30 / T30_60 / T60_180 |
| days_to_specialty | INT64 | |

---

## 02 — Specialty Preprocessing

### model_train / model_test

Full population train/test split. Created by `Model_data_setup_01.sql`.

Same schema as model_input_sequences. Split: trigger_date < 2024 (train), >= 2024 (test). Gated by has_claims_12m_before = TRUE.

### markov_train

Transition counts for Markov baseline. Created by `Model_data_setup_01.sql`.

| Column | Type | Description |
|---|---|---|
| trigger_dx | STRING | |
| member_segment | STRING | |
| next_specialty | STRING | |
| transition_count | INT64 | COUNT(*) |
| unique_members | INT64 | COUNT(DISTINCT member_id) |

### sample_members_{1/5/10}pct

One row per member_id. Created by `Model_sampling.sql`.

### model_train_{pct} / model_test_{pct}

Per-sample train/test. Created by `Model_train_test_split.sql`. Schema: member_id, trigger_date, trigger_dx, member_segment, age_nbr, gender_cd, is_t30/t60/t180_qualified, label_specialty, time_bucket.

### train_sequences_{pct} / test_sequences_{pct}

Flat pre-trigger sequences. Created by `model_input_sequence_data.sql`.

| Column | Type | Description |
|---|---|---|
| member_id | STRING | |
| trigger_date | DATE | |
| trigger_dx | STRING | |
| member_segment | STRING | |
| is_t30/t60/t180_qualified | BOOL | |
| specialty_ctg_cd | STRING | Sequence token |
| recency_rank | INT64 | 1 = most recent visit before trigger, capped at 20 |

---

## 04 — Provider Preprocessing

### provider_vocab

One row per srv_prvdr_id. Created by `03_provider_transitions_and_cutoff.sql`.

| Column | Type | Description |
|---|---|---|
| srv_prvdr_id | STRING | |
| claim_count | INT64 | Total claims for this provider |
| cumulative_pct | FLOAT64 | Running share of total volume |
| is_top80 | BOOL | TRUE if cumulative volume <= 80% |

### provider_transitions

One row per (from_provider, to_provider). Created by `03_provider_transitions_and_cutoff.sql`.

### provider_primary_specialty

One row per srv_prvdr_id. Created by `sql01b_hard_negative`. Most frequent specialty.

### provider_hardneg_lookup

Hard negative candidates per (from_provider, specialty). Created by `sql01b_hard_negative`.

### provider_model_train_{pct} / provider_model_test_{pct}

Same structure as specialty train/test but label = srv_prvdr_id. Train filtered to is_top80 = TRUE. Test keeps all providers.

### provider_model_train_agg_{pct} / provider_model_test_agg_{pct}

One row per trigger. Labels pre-aggregated via ARRAY_AGG. Created by `sql02`.

### provider_train_sequences_{pct} / provider_test_sequences_{pct}

Flat rows per (trigger, recency_rank). Grain: (member, date, provider). Includes delta_t_bucket. Created by `sql03`.

---

## 06 — Evaluation

### trigger_scores / markov_trigger_scores

Specialty-level scores. One row per (member, trigger, window, model).

| Column | Type | Description |
|---|---|---|
| model | STRING | SASRec / BERT4Rec / Markov |
| member_id | STRING | |
| trigger_date | STRING | Cast from DATE |
| trigger_dx | STRING | |
| member_segment | STRING | |
| time_bucket | STRING | T0_30 / T30_60 / T60_180 |
| top5_predictions | STRING | Pipe-delimited specialty codes |
| true_labels | STRING | Pipe-delimited true specialties |
| hit_at_1 / hit_at_3 / hit_at_5 | FLOAT64 | 0 or 1 |
| ndcg_at_1 / ndcg_at_3 / ndcg_at_5 | FLOAT64 | |

### provider_trigger_scores

Same schema as trigger_scores but for provider-level predictions. Pipe-strings contain srv_prvdr_id values.

### provider_eval_5pct

One row per (member, trigger_date, trigger_dx, time_bucket, model). Created by `sql05`.

Includes: TP, FP, FN, hit_at_K, ndcg_at_K, from_provider, top5 predictions + scores.

### provider_eval_visit_5pct

Visit-level rollup of provider_eval. Created by `sql06`.

### PMA tables

| Table | Created By | Grain |
|---|---|---|
| pma_transition_bucket_5pct | SQL_PMA_01 | (evidence_bucket, model, window) |
| pma_dx_summary_5pct | SQL_PMA_02 | (trigger_dx, model, window) |
| pma_provider_dx_summary_5pct | SQL_PMA_03 | (from_provider, trigger_dx, predicted_provider, model, window) |
| pma_provider_summary_5pct | SQL_PMA_04 | (direction, provider, model, window) |
| pma_monthly_provider_dx_5pct | SQL_PMA_rollup | (from_provider, trigger_dx, predicted_provider, model, month) |
| pma_monthly_provider_5pct | SQL_PMA_rollup2 | (from_provider, predicted_provider, model, month) |
| provider_name_lookup | SQL_PMA_00 | (srv_prvdr_id) |
| analysis_perf_overall | Post_model_analysis_data_setup1 | (model, window, segment) |
| analysis_perf_by_diag | Post_model_analysis_data_setup1 | (model, window, trigger_dx) |
| analysis_perf_by_ending_specialty | Post_model_analysis_data_setup1 | (model, window, specialty) |
| analysis_perf_full | Model_metrics_at_5_comp | (model, window, segment) with precision/recall |
