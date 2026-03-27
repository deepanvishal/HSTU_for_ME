# ============================================================
# Block 1 — Raw Claims Facts
# FACT-01: Total claims count
# FACT-02: Total unique members
# FACT-11: Provider count per specialty
# FACT-34: Dataset date range
# FACT-35: Total unique providers
# Source: A870800_claims_gen_rec_2022_2025_sfl
# ============================================================
import pandas as pd
from google.cloud import bigquery
from IPython.display import display, Markdown

DS = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
client = bigquery.Client(project="anbc-hcb-dev")

# ── Query 1: Scalar facts ────────────────────────────────────
scalars = client.query(f"""
    SELECT
        COUNT(*)                                         AS total_claims
        ,COUNT(DISTINCT member_id)                       AS total_members
        ,COUNT(DISTINCT srv_prvdr_id)                    AS total_providers
        ,MIN(srv_start_dt)                               AS date_min
        ,MAX(srv_start_dt)                               AS date_max
    FROM `{DS}.A870800_claims_gen_rec_2022_2025_sfl`
""").to_dataframe().iloc[0]

FACT_01 = f"{scalars['total_claims']:,.0f}"
FACT_02 = f"{scalars['total_members']:,.0f}"
FACT_35 = f"{scalars['total_providers']:,.0f}"
FACT_34_MIN = str(scalars['date_min'].date())
FACT_34_MAX = str(scalars['date_max'].date())

display(Markdown(f"""
| FACT | Value |
|---|---|
| FACT-01 Total claims | {FACT_01} |
| FACT-02 Total members | {FACT_02} |
| FACT-35 Total providers | {FACT_35} |
| FACT-34 Date range | {FACT_34_MIN} to {FACT_34_MAX} |
"""))

# ── Query 2: Provider count per specialty ─────────────────────
fact_11 = client.query(f"""
    SELECT
        specialty_ctg_cd                                 AS specialty
        ,COUNT(DISTINCT srv_prvdr_id)                    AS provider_count
    FROM `{DS}.A870800_claims_gen_rec_2022_2025_sfl`
    WHERE specialty_ctg_cd IS NOT NULL
      AND TRIM(specialty_ctg_cd) != ''
    GROUP BY 1
    ORDER BY 2 DESC
""").to_dataframe()

FACT_11_MAX = f"{fact_11.iloc[0]['provider_count']:,.0f}"
FACT_11_MAX_NAME = fact_11.iloc[0]['specialty']
FACT_11_MIN = f"{fact_11.iloc[-1]['provider_count']:,.0f}"
FACT_11_MIN_NAME = fact_11.iloc[-1]['specialty']

display(Markdown(f"""
### FACT-11: Provider Count per Specialty
| Rank | Specialty | Providers |
|---|---|---|
| Top | {FACT_11_MAX_NAME} | {FACT_11_MAX} |
| Bottom | {FACT_11_MIN_NAME} | {FACT_11_MIN} |
"""))
display(fact_11)

print("Block 1 done.")
