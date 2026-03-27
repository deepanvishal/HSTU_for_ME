# ============================================================
# Block 8 — Provider-Level Prediction Facts + Visuals
# Requires: client, DS
# Sources: provider_eval_5pct, pma_transition_bucket_5pct,
#          pma_dx_summary_5pct, pma_provider_summary_5pct
# ============================================================

PROV_MODELS = ["SASRec", "BERT4Rec", "HSTU", "Markov"]
PROV_COLORS = {"SASRec": "#3B82F6", "BERT4Rec": "#059669",
               "HSTU": "#8172B2", "Markov": "#9CA3AF"}

# ── PROV-FACT-01/02/03/10: Overall Hit@5 all models all windows ──
prov_overall = client.query(f"""
    SELECT model, time_bucket
        ,COUNT(*) AS n_triggers
        ,ROUND(AVG(hit_at_5), 4) AS hit_at_5
    FROM `{DS}.A870800_gen_rec_provider_eval_5pct`
    WHERE (tp + fn) > 0
    GROUP BY model, time_bucket
    ORDER BY model, time_bucket
""").to_dataframe()

display(Markdown("### PROV-FACT-10: All Models Hit@5"))
display(prov_overall)

# ── PROV-FACT-04: Hit@5 by segment — best model ──
prov_best = prov_overall[prov_overall["time_bucket"] == "T0_30"].sort_values("hit_at_5", ascending=False).iloc[0]["model"]
print(f"Best provider model at T30: {prov_best}")

prov_seg = client.query(f"""
    SELECT member_segment
        ,COUNT(*) AS n_triggers
        ,ROUND(AVG(hit_at_5), 4) AS hit_at_5
    FROM `{DS}.A870800_gen_rec_provider_eval_5pct`
    WHERE time_bucket = 'T0_30' AND model = '{prov_best}'
      AND (tp + fn) > 0
    GROUP BY member_segment
    ORDER BY hit_at_5 DESC
""").to_dataframe()

display(Markdown("### PROV-FACT-04: Hit@5 by Segment"))
display(prov_seg)

# ── PROV-FACT-05: Hit@5 by evidence bucket ──
prov_buckets = client.query(f"""
    SELECT transition_bucket, model, hit_at_5, n_triggers
    FROM `{DS}.A870800_gen_rec_pma_transition_bucket_5pct`
    WHERE time_bucket = 'T0_30'
    ORDER BY model, transition_bucket
""").to_dataframe()

display(Markdown("### PROV-FACT-05: Hit@5 by Evidence Bucket at T30"))
display(prov_buckets)

# ── PROV-FACT-07: Top 10 outbound providers (excl Lab) ──
prov_outbound = client.query(f"""
    SELECT srv_prvdr_id, provider_name, specialty_desc
        ,n_triggers, hit_at_5
    FROM `{DS}.A870800_gen_rec_pma_provider_summary_5pct`
    WHERE provider_direction = 'Outbound'
      AND time_bucket = 'T0_30'
      AND model = '{prov_best}'
      AND specialty_desc NOT LIKE '%Lab%'
      AND specialty_desc NOT LIKE '%lab%'
      AND n_triggers >= 20
    ORDER BY hit_at_5 DESC
    LIMIT 10
""").to_dataframe()

display(Markdown("### PROV-FACT-07: Top 10 Outbound Providers (excl Lab)"))
display(prov_outbound)

# ── PROV-FACT-08: Top 10 inbound providers (excl Lab) ──
prov_inbound = client.query(f"""
    SELECT srv_prvdr_id, provider_name, specialty_desc
        ,n_triggers, overall_precision
    FROM `{DS}.A870800_gen_rec_pma_provider_summary_5pct`
    WHERE provider_direction = 'Inbound'
      AND time_bucket = 'T0_30'
      AND model = '{prov_best}'
      AND specialty_desc NOT LIKE '%Lab%'
      AND specialty_desc NOT LIKE '%lab%'
      AND n_triggers >= 20
    ORDER BY overall_precision DESC
    LIMIT 10
""").to_dataframe()

display(Markdown("### PROV-FACT-08: Top 10 Inbound Providers (excl Lab)"))
display(prov_inbound)

# ── PROV-FACT-06: Top 10 dx by Hit@5 (excl Lab ending specialty) ──
prov_dx = client.query(f"""
    SELECT trigger_dx, dx_desc, n_triggers, hit_at_5
    FROM `{DS}.A870800_gen_rec_pma_dx_summary_5pct`
    WHERE time_bucket = 'T0_30'
      AND model = '{prov_best}'
      AND n_triggers >= 100
    ORDER BY hit_at_5 DESC
    LIMIT 10
""").to_dataframe()

display(Markdown("### PROV-FACT-06: Top 10 DX by Hit@5 (Outbound)"))
display(prov_dx)

# ── PROV-FACT-09: Provider vocab ──
prov_vocab = client.query(f"""
    SELECT COUNT(*) AS total, COUNTIF(is_top80) AS top80
    FROM `{DS}.A870800_gen_rec_provider_vocab`
""").to_dataframe().iloc[0]

display(Markdown(f"""
### PROV-FACT-09: Provider Vocab
- Total providers: {int(prov_vocab['total']):,}
- Top 80%: {int(prov_vocab['top80']):,}
"""))

print("Block 8 facts done.")
