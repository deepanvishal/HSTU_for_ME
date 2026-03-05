from google.cloud import bigquery
import numpy as np
from collections import defaultdict

client = bigquery.Client(project='anbc-hcb-dev')

member_sequences = defaultdict(list)
member_labels    = defaultdict(list)

def process_sequence_chunk(chunk):
    for row in chunk.itertuples():
        member_sequences[row.member_id].append({
            'visit_seq_num'  : row.visit_seq_num
            ,'delta_t_bucket': row.delta_t_bucket
            ,'provider_ids'  : row.provider_ids
            ,'dx_list'       : row.dx_list
            ,'prior_dx_list' : row.prior_dx_list
        })

def process_label_chunk(chunk):
    for row in chunk.itertuples():
        member_labels[row.member_id].append({
            'visit_seq_num'   : row.visit_seq_num
            ,'specialties_30' : row.specialties_30
            ,'specialties_60' : row.specialties_60
            ,'specialties_180': row.specialties_180
            ,'providers_30'   : row.providers_30
            ,'providers_60'   : row.providers_60
            ,'providers_180'  : row.providers_180
            ,'dx_30'          : row.dx_30
            ,'dx_60'          : row.dx_60
            ,'dx_180'         : row.dx_180
        })

sequence_query = """
SELECT
    member_id
    ,visit_seq_num
    ,delta_t_bucket
    ,provider_ids
    ,dx_list
    ,prior_dx_list
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_visit_sequence`
ORDER BY member_id, visit_seq_num
"""

label_query = """
SELECT
    member_id
    ,visit_seq_num
    ,specialties_30, specialties_60, specialties_180
    ,providers_30,   providers_60,   providers_180
    ,dx_30,          dx_60,          dx_180
FROM `anbc-hcb-dev.provider_ds_netconf_data_hcb_dev.A870800_claims_gen_rec_label`
ORDER BY member_id, visit_seq_num
"""

CHUNK_SIZE = 100_000

# sequence
job = client.query(sequence_query)
df  = job.to_dataframe()
for start in range(0, len(df), CHUNK_SIZE):
    chunk = df.iloc[start:start + CHUNK_SIZE]
    process_sequence_chunk(chunk)
    del chunk
del df

# labels
job = client.query(label_query)
df  = job.to_dataframe()
for start in range(0, len(df), CHUNK_SIZE):
    chunk = df.iloc[start:start + CHUNK_SIZE]
    process_label_chunk(chunk)
    del chunk
del df

print(f"Members loaded: {len(member_sequences)}")
