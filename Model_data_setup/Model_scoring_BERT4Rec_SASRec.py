# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7B — SCORE TRIGGERS AND WRITE TO BIGQUERY
# Purpose : For every test trigger + window — store top-5 predictions,
#           true labels, and per-K metrics for post-hoc analysis
# Table   : A870800_gen_rec_trigger_scores (APPEND)
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 7B — Score Triggers"))
print("Section 7B — scoring test triggers...")

# Rebuild test loader with shuffle=False to keep order aligned with test_records
score_loader = DataLoader(
    SpecialtyDataset(test_records),   # SASRec
    # BERT4Rec: BERT4RecDataset(test_records, mask_for_inference=True)
    batch_size=BATCH_SIZE * 2, shuffle=False, **_loader_kwargs
)

t_model.eval()
all_rows = []
record_idx = 0

with torch.no_grad():
    for batch in score_loader:
        seq   = batch["sequence"].to(DEVICE, non_blocking=True)
        lens  = batch["seq_len"].to(DEVICE, non_blocking=True)
        # For BERT4Rec replace lens with:
        # tm = batch["target_mask"].to(DEVICE, non_blocking=True)
        m30   = batch["is_t30"].to(DEVICE, non_blocking=True)
        m60   = batch["is_t60"].to(DEVICE, non_blocking=True)
        m180  = batch["is_t180"].to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            p30, p60, p180 = t_model(seq, lens)
            # For BERT4Rec: p30, p60, p180 = t_model(seq, tm)

        # Apply sigmoid to get probabilities from logits
        p30  = torch.sigmoid(p30.float())
        p60  = torch.sigmoid(p60.float())
        p180 = torch.sigmoid(p180.float())

        bs = seq.size(0)

        for i in range(bs):
            rec = test_records[record_idx]
            record_idx += 1

            for window, pred, mask in [
                ("T0_30",   p30[i],  m30[i]),
                ("T30_60",  p60[i],  m60[i]),
                ("T60_180", p180[i], m180[i]),
            ]:
                if not mask.item():
                    continue

                # Top 5 predictions
                top5_vals, top5_idx = torch.topk(pred, 5)
                top5_specs  = [idx_to_specialty.get(idx.item(), "UNK")
                               for idx in top5_idx]
                top5_scores = [round(v.item(), 4) for v in top5_vals]

                # True labels for this window
                bucket_map  = {"T0_30": "label_t30", "T30_60": "label_t60", "T60_180": "label_t180"}
                true_vec    = rec[bucket_map[window]]         # numpy float32 [NUM_SPECIALTIES]
                true_specs  = [idx_to_specialty[j]
                               for j in range(len(true_vec)) if true_vec[j] > 0]

                true_set    = set(true_specs)
                top5_set_1  = set(top5_specs[:1])
                top5_set_3  = set(top5_specs[:3])
                top5_set_5  = set(top5_specs[:5])

                # Per-K metrics
                def hit(pred_set):
                    return 1.0 if pred_set & true_set else 0.0

                def ndcg(preds, k):
                    disc = [1.0 / np.log2(r + 2) for r in range(k)]
                    dcg  = sum(disc[r] for r, sp in enumerate(preds[:k]) if sp in true_set)
                    n    = min(len(true_set), k)
                    idcg = sum(1.0 / np.log2(r + 2) for r in range(n))
                    return round(dcg / idcg, 4) if idcg > 0 else 0.0

                all_rows.append({
                    "member_id":        rec.get("member_id", ""),
                    "trigger_date":     rec["trigger_date"],
                    "trigger_dx":       rec.get("trigger_dx", ""),
                    "member_segment":   rec.get("member_segment", ""),
                    "time_bucket":      window,
                    "true_labels":      "|".join(sorted(true_specs)),
                    "top5_predictions": "|".join(top5_specs),
                    "top5_scores":      "|".join(str(s) for s in top5_scores),
                    "hit_at_1":         hit(top5_set_1),
                    "hit_at_3":         hit(top5_set_3),
                    "hit_at_5":         hit(top5_set_5),
                    "ndcg_at_1":        ndcg(top5_specs, 1),
                    "ndcg_at_3":        ndcg(top5_specs, 3),
                    "ndcg_at_5":        ndcg(top5_specs, 5),
                    "model":            "SASRec",   # change to "BERT4Rec" in NB_09
                    "sample":           SAMPLE,
                    "run_timestamp":    RUN_TIMESTAMP,
                })

print(f"Scored {len(all_rows):,} trigger-window pairs")

# Write to BQ in batches of 100K to avoid memory issues
BATCH_BQ = 100_000
schema = [
    bigquery.SchemaField("member_id",        "STRING"),
    bigquery.SchemaField("trigger_date",     "STRING"),
    bigquery.SchemaField("trigger_dx",       "STRING"),
    bigquery.SchemaField("member_segment",   "STRING"),
    bigquery.SchemaField("time_bucket",      "STRING"),
    bigquery.SchemaField("true_labels",      "STRING"),
    bigquery.SchemaField("top5_predictions", "STRING"),
    bigquery.SchemaField("top5_scores",      "STRING"),
    bigquery.SchemaField("hit_at_1",         "FLOAT64"),
    bigquery.SchemaField("hit_at_3",         "FLOAT64"),
    bigquery.SchemaField("hit_at_5",         "FLOAT64"),
    bigquery.SchemaField("ndcg_at_1",        "FLOAT64"),
    bigquery.SchemaField("ndcg_at_3",        "FLOAT64"),
    bigquery.SchemaField("ndcg_at_5",        "FLOAT64"),
    bigquery.SchemaField("model",            "STRING"),
    bigquery.SchemaField("sample",           "STRING"),
    bigquery.SchemaField("run_timestamp",    "STRING"),
]
job_cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND", schema=schema)

for start in range(0, len(all_rows), BATCH_BQ):
    chunk = pd.DataFrame(all_rows[start:start + BATCH_BQ])
    client.load_table_from_dataframe(
        chunk,
        f"{DS}.A870800_gen_rec_trigger_scores",
        job_config=job_cfg
    ).result()
    print(f"  Written rows {start:,} — {min(start+BATCH_BQ, len(all_rows)):,}")

print(f"Section 7B done — {len(all_rows):,} rows written, time={time.time()-t0:.1f}s")
display(Markdown(f"**Section 7B:** {len(all_rows):,} trigger scores written | **Time:** {time.time()-t0:.1f}s"))
