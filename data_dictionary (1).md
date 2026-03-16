# Data Sources and Data Dictionary
## Next Visit Prediction — HSTU
### South Florida Claims Data — 2022 to 2025

---

## 1. Raw Data Sources

| Table | Full BQ Path | Description |
|---|---|---|
| Claims | `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_2022_2025_sfl` | Medical claims 2022-2025, South Florida |
| CCSR Mapping | `edp-prod-hcbstorage.edp_hcb_mwb_bh_analytics_cnsv.AHRQ_CCSR_DX_20260101` | AHRQ CCSR ICD-10 to clinical category mapping |
| ICD-10 Descriptions | `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` | ICD-10 diagnosis code descriptions |
| Specialty Descriptions | `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` | Specialty category code descriptions |

### Key Columns — Claims Table

| Column | Description |
|---|---|
| `member_id` | Unique member identifier |
| `srv_prvdr_id` | Servicing provider identifier |
| `srv_start_dt` | Service start date |
| `pri_icd9_dx_cd` | Primary ICD-10 diagnosis code |
| `specialty_ctg_cd` | Provider specialty category code |
| `age_nbr` | Member age |
| `gender_cd` | Member gender |

---

## 2. Derived Tables

### 2.1 Data Preparation Tables

| Table | Full BQ Path | Purpose |
|---|---|---|
| Claims Flagged | `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_claims_flagged` | Claims enriched with visit flags, CCSR mapping, first encounter flags |
| Visit Spine | `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_visit_spine` | One row per member + visit date with visit rank |
| Triggers | `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_triggers` | One row per member + trigger date + trigger diagnosis (first encounter) |

#### A870800_gen_rec_claims_flagged

| Column | Description |
|---|---|
| `member_id` | Member identifier |
| `srv_prvdr_id` | Provider identifier |
| `visit_date` | Service date |
| `visit_number` | Dense rank of visit date per member |
| `specialty_ctg_cd` | Provider specialty code |
| `dx_raw` | Raw ICD-10 code |
| `dx_clean` | ICD-10 code with periods removed |
| `is_first_dx_encounter` | TRUE if first time member presents with this diagnosis |
| `is_first_provider_visit` | TRUE if first time member sees this provider |
| `visit_flag` | first_member_visit / new_provider_new_dx / new_provider_known_dx / known_provider_new_dx / known_provider_known_dx |
| `ccsr_category` | CCSR clinical category code |
| `ccsr_category_description` | CCSR clinical category description |
| `age_nbr` | Member age |
| `gender_cd` | Member gender |
| `member_segment` | Adult_Female / Adult_Male / Senior / Children |

#### A870800_gen_rec_visit_spine

| Column | Description |
|---|---|
| `member_id` | Member identifier |
| `visit_date` | Service date |
| `visit_rank` | Dense rank of visit date per member (1 = first visit) |

#### A870800_gen_rec_triggers

| Column | Description |
|---|---|
| `member_id` | Member identifier |
| `trigger_date` | Date of first encounter with trigger diagnosis |
| `trigger_dx` | Trigger ICD-10 diagnosis code |
| `trigger_dx_clean` | Trigger ICD-10 with periods removed |
| `trigger_dx_desc` | Trigger diagnosis description |
| `trigger_ccsr` | CCSR category of trigger diagnosis |
| `trigger_ccsr_desc` | CCSR category description |
| `trigger_specialty` | Specialty code at trigger visit |
| `trigger_specialty_desc` | Specialty description at trigger visit |
| `member_segment` | Adult_Female / Adult_Male / Senior / Children |
| `trigger_rank` | Rank of trigger by date per member |

---

### 2.2 Markov Transition Tables

#### Order 1 — Representative Schema (dx_to_specialty_order1)

All 9 Order 1 tables follow the same schema. Table names listed below.

| Column | Description |
|---|---|
| `current_dx` | Current ICD-10 diagnosis code |
| `current_dx_desc` | Current diagnosis description |
| `current_ccsr` | CCSR category of current diagnosis |
| `current_ccsr_desc` | CCSR category description |
| `next_specialty` | Next visit provider specialty code |
| `next_specialty_desc` | Next specialty description |
| `member_segment` | Cohort |
| `transition_count` | Number of transitions |
| `unique_members` | Distinct members driving the transition |
| `dx_total` | Total transitions from current state |
| `conditional_probability` | transition_count / dx_total |
| `conditional_entropy` | -SUM(p * LOG(p)) per current state |

**All Order 1 table names:**

- `A870800_gen_rec_f_dx_to_specialty_order1`
- `A870800_gen_rec_f_dx_to_dx_order1`
- `A870800_gen_rec_f_dx_to_ccsr_order1`
- `A870800_gen_rec_f_specialty_to_specialty_order1`
- `A870800_gen_rec_f_specialty_to_dx_order1`
- `A870800_gen_rec_f_specialty_to_ccsr_order1`
- `A870800_gen_rec_f_ccsr_to_specialty_order1`
- `A870800_gen_rec_f_ccsr_to_dx_order1`
- `A870800_gen_rec_f_ccsr_to_ccsr_order1`

#### Order 2 — Representative Schema (dx_to_specialty_order2)

Same as Order 1 with two current state columns instead of one.

| Column | Description |
|---|---|
| `current_dx_v1` | Trigger ICD-10 diagnosis code (V1) |
| `current_dx_v1_desc` | Trigger diagnosis description |
| `current_ccsr_v1` | CCSR of trigger diagnosis |
| `current_ccsr_v1_desc` | CCSR description |
| `current_dx_v2` | Second visit ICD-10 diagnosis code (V2) |
| `current_dx_v2_desc` | Second visit diagnosis description |
| `next_specialty` | Next visit specialty code |
| `next_specialty_desc` | Next specialty description |
| `member_segment` | Cohort |
| `transition_count` | Number of transitions |
| `unique_members` | Distinct members |
| `pair_total` | Total transitions from (V1, V2) pair |
| `conditional_probability` | transition_count / pair_total |
| `conditional_entropy` | -SUM(p * LOG(p)) per (V1, V2) pair |

**All Order 2 table names:**

- `A870800_gen_rec_f_dx_to_specialty_order2`
- `A870800_gen_rec_f_dx_to_dx_order2`
- `A870800_gen_rec_f_dx_to_ccsr_order2`
- `A870800_gen_rec_f_specialty_to_specialty_order2`
- `A870800_gen_rec_f_specialty_to_dx_order2`
- `A870800_gen_rec_f_specialty_to_ccsr_order2`
- `A870800_gen_rec_f_ccsr_to_specialty_order2`
- `A870800_gen_rec_f_ccsr_to_dx_order2`
- `A870800_gen_rec_f_ccsr_to_ccsr_order2`

#### Entropy Summary

| Table | Full BQ Path | Purpose |
|---|---|---|
| Entropy Summary | `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_entropy_summary` | Weighted average entropy per combination, order, and cohort |

| Column | Description |
|---|---|
| `combination` | Transition combination (e.g. dx_to_specialty) |
| `markov_order` | 1 or 2 |
| `member_segment` | Cohort |
| `weighted_avg_entropy` | Volume-weighted average conditional entropy |
| `median_entropy` | Median conditional entropy |
| `total_transitions` | Total transition count |
| `unique_current_states` | Distinct current state values |

---

### 2.3 Time Window Analysis Tables

| Table | Full BQ Path | Purpose |
|---|---|---|
| Track 1 Base | `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_base` | Member + trigger + all downstream visits within T180 as ARRAY STRUCT |
| Track 1 Summary | `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_summary` | First specialist per window per member per trigger — aggregated |
| Track 1 Penetration | `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_track1_penetration` | Penetration rate and binary entropy per specialty per trigger per window |

#### A870800_gen_rec_f_track1_base

| Column | Description |
|---|---|
| `member_id` | Member identifier |
| `trigger_date` | Date of first diagnosis encounter |
| `trigger_dx` | Trigger ICD-10 code |
| `trigger_dx_desc` | Trigger diagnosis description |
| `trigger_ccsr` | CCSR of trigger |
| `trigger_ccsr_desc` | CCSR description |
| `trigger_specialty` | Specialty at trigger visit |
| `trigger_specialty_desc` | Specialty description |
| `member_segment` | Cohort |
| `downstream_visits` | ARRAY STRUCT of all visits within T180 — each struct contains visit_date, days_since_trigger, specialty, specialty_desc, dx, dx_clean, dx_desc, ccsr, ccsr_desc |

#### A870800_gen_rec_f_track1_summary

| Column | Description |
|---|---|
| `trigger_dx` | Trigger ICD-10 code |
| `trigger_dx_desc` | Trigger diagnosis description |
| `trigger_ccsr` | CCSR of trigger |
| `trigger_ccsr_desc` | CCSR description |
| `trigger_specialty` | Specialty at trigger visit |
| `v2_dx` | Second visit ICD-10 code |
| `v2_dx_desc` | Second visit diagnosis description |
| `next_specialty` | First specialist seen after V2 within window |
| `next_specialty_desc` | Specialty description |
| `member_segment` | Cohort |
| `time_window` | T30 / T60 / T180 |
| `transition_count` | Number of member+trigger pairs |
| `unique_members` | Distinct members |
| `pair_total` | Total transitions from (trigger_dx, v2_dx) pair |
| `avg_days_to_specialty` | Average days from trigger to first specialist |
| `median_days_to_specialty` | Median days from trigger to first specialist |
| `conditional_probability` | transition_count / pair_total |
| `conditional_entropy` | -SUM(p * LOG(p)) per pair |

#### A870800_gen_rec_f_track1_penetration

| Column | Description |
|---|---|
| `trigger_dx` | Trigger ICD-10 code |
| `trigger_dx_desc` | Trigger diagnosis description |
| `trigger_ccsr` | CCSR of trigger |
| `trigger_ccsr_desc` | CCSR description |
| `trigger_specialty` | Specialty at trigger visit |
| `member_segment` | Cohort |
| `visit_specialty` | Specialty visited within window |
| `visit_specialty_desc` | Specialty description |
| `time_window` | T30 / T60 / T180 |
| `members_visited` | Distinct members who visited this specialty |
| `visit_count` | Total visits to this specialty |
| `total_members` | Total members with this trigger diagnosis |
| `penetration_rate` | members_visited / total_members |
| `binary_entropy` | -(p * LOG(p)) - ((1-p) * LOG(1-p)) where p = penetration_rate |

---

### 2.4 Markov Baseline Tables

| Table | Full BQ Path | Purpose |
|---|---|---|
| Markov Eval | `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_eval` | Member level first specialist per window with train/test split flag |
| Markov Train | `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_train` | Transition probabilities from pre-2024 triggers |
| Markov Test | `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_gen_rec_f_markov_predictions` | Test members joined to top 5 predicted specialties |

#### A870800_gen_rec_f_markov_eval

| Column | Description |
|---|---|
| `member_id` | Member identifier |
| `trigger_dx` | Trigger ICD-10 code |
| `trigger_dx_desc` | Trigger diagnosis description |
| `trigger_ccsr` | CCSR of trigger |
| `trigger_date` | Trigger date |
| `v2_dx` | Second visit ICD-10 code |
| `actual_specialty` | First specialist seen after V2 within window |
| `actual_specialty_desc` | Specialty description |
| `member_segment` | Cohort |
| `time_window` | T30 / T60 / T180 |
| `days_since_trigger` | Days from trigger to actual specialty visit |
| `split` | train (pre-2024) / test (2024 onwards) |

#### A870800_gen_rec_f_markov_train

| Column | Description |
|---|---|
| `trigger_dx` | Trigger ICD-10 code |
| `v2_dx` | Second visit ICD-10 code |
| `next_specialty` | Predicted next specialty |
| `member_segment` | Cohort |
| `time_window` | T30 / T60 / T180 |
| `transition_count` | Distinct members in training data |
| `pair_total` | Total transitions from pair |
| `train_probability` | transition_count / pair_total |
| `specialty_rank` | Rank of specialty by probability for this pair (1 = highest) |

#### A870800_gen_rec_f_markov_predictions

| Column | Description |
|---|---|
| `member_id` | Member identifier |
| `trigger_date` | Trigger date |
| `trigger_dx` | Trigger ICD-10 code |
| `v2_dx` | Second visit ICD-10 code |
| `actual_specialty` | Actual specialty visited in test period |
| `member_segment` | Cohort |
| `time_window` | T30 / T60 / T180 |
| `predicted_specialty` | Predicted specialty from Markov model |
| `train_probability` | Probability assigned by Markov model |
| `specialty_rank` | Rank of prediction (1 = top prediction) |
