# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SCORE TRIGGERS AND WRITE TO BIGQUERY
# Fully vectorized: no Python loops for metric computation
# Python loop only assembles strings — unavoidable for BQ row format
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
display(Markdown("---\n## Section 5 — Score Triggers"))
print("Section 5 — scoring triggers (fully vectorized)...")

score_loader = DataLoader(
    SpecialtyDataset(test_data),       # BERT4Rec: BERT4RecDataset(test_data)
    batch_size=BATCH_SIZE, shuffle=False, **_loader_kwargs
)

# ── Precompute once outside all loops ─────────────────────────────────────────
SCORE_K = 5

# Discount vectors and cumsums on GPU — no Python IDCG loop
disc_score   = (1.0 / torch.log2(
    torch.arange(2, SCORE_K + 2, dtype=torch.float32)
)).to(DEVICE)
disc_cumsums = {k: disc_score[:k].cumsum(0) for k in [1, 3, 5]}

# spec_lookup numpy array — O(1) index vs dict.get()
max_idx     = max(idx_to_specialty.keys()) + 1
spec_lookup = np.array(["UNK"] * max_idx, dtype=object)
for idx, sp in idx_to_specialty.items():
    spec_lookup[idx] = sp

# Pre-extract test_data arrays — avoid repeated dict key lookup in loop
member_ids_arr    = test_data["member_ids"]
trigger_dates_arr = test_data["trigger_dates"]
trigger_dxs_arr   = test_data["trigger_dxs"]
segments_arr      = test_data["segments"]
BUCKET_ARRAYS     = {
    "T0_30":   test_data["lab_t30"],
    "T30_60":  test_data["lab_t60"],
    "T60_180": test_data["lab_t180"],
}

# Column-wise lists — pd.DataFrame(col_dict) is 3-5x faster than list of dicts
cols = {
    "member_id": [], "trigger_date": [], "trigger_dx": [], "member_segment": [],
    "time_bucket": [], "true_labels": [], "top5_predictions": [], "top5_scores": [],
    "hit_at_1": [], "hit_at_3": [], "hit_at_5": [],
    "ndcg_at_1": [], "ndcg_at_3": [], "ndcg_at_5": [],
    "model": [], "sample": [], "run_timestamp": [],
}


# vec_ndcg defined once outside batch loop
def vec_ndcg(k, hits, n_true):
    d    = disc_score[:k]
    dcg  = (hits[:, :k].float() * d).sum(1)
    ni   = n_true.clamp(min=1, max=k).long() - 1       # 0-based cumsum index
    idcg = disc_cumsums[k][ni]
    return (dcg / idcg.clamp(min=1e-8)).cpu().numpy()


record_idx = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(score_loader):
        seq   = batch["sequence"].to(DEVICE, non_blocking=True)
        lens  = batch["seq_len"].to(DEVICE, non_blocking=True)
        # BERT4Rec: replace lens with tm = batch["target_mask"].to(DEVICE, non_blocking=True)
        m30   = batch["is_t30"].to(DEVICE, non_blocking=True)
        m60   = batch["is_t60"].to(DEVICE, non_blocking=True)
        m180  = batch["is_t180"].to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            p30, p60, p180 = model(seq, lens)
            # BERT4Rec: model(seq, tm)

        p30  = torch.sigmoid(p30.float())
        p60  = torch.sigmoid(p60.float())
        p180 = torch.sigmoid(p180.float())
        bs   = seq.size(0)

        for window, pred, mask in [
            ("T0_30",   p30,  m30),
            ("T30_60",  p60,  m60),
            ("T60_180", p180, m180),
        ]:
            n_qual = mask.sum().item()
            if n_qual == 0:
                continue

            qual_pos = mask.nonzero(as_tuple=True)[0].cpu().numpy()
            qual_ri  = record_idx + qual_pos

            pred_m  = pred[mask].float()
            lbl_np  = BUCKET_ARRAYS[window][qual_ri]
            lbl_m   = torch.from_numpy(lbl_np).to(DEVICE)

            # One topk call for all n records
            top5_vals, top5_idx = torch.topk(pred_m, SCORE_K, dim=1)
            hits   = lbl_m.gather(1, top5_idx)

            # hit@k — tensor slices, no loop
            hit1   = (hits[:, :1].sum(1) > 0).float()
            hit3   = (hits[:, :3].sum(1) > 0).float()
            hit5   = (hits[:, :5].sum(1) > 0).float()

            # ndcg@k — fully vectorized via cumsum
            n_true = lbl_m.sum(1)
            ndcg1  = vec_ndcg(1, hits, n_true)
            ndcg3  = vec_ndcg(3, hits, n_true)
            ndcg5  = vec_ndcg(5, hits, n_true)

            # Single GPU→CPU transfer for all hit arrays
            hits_cpu      = torch.stack([hit1, hit3, hit5], dim=1).cpu().numpy()
            top5_idx_cpu  = top5_idx.cpu().numpy()
            top5_vals_cpu = top5_vals.cpu().numpy()

            # Python loop — string assembly only, no computation
            for j, ri in enumerate(qual_ri):
                top5_specs  = list(spec_lookup[top5_idx_cpu[j]])
                top5_scores = [round(float(v), 4) for v in top5_vals_cpu[j]]

                true_pos   = np.where(lbl_np[j] > 0)[0]
                true_specs = list(spec_lookup[true_pos[true_pos < max_idx]])

                cols["member_id"].append(str(member_ids_arr[ri]))
                cols["trigger_date"].append(str(trigger_dates_arr[ri]))
                cols["trigger_dx"].append(str(trigger_dxs_arr[ri]))
                cols["member_segment"].append(str(segments_arr[ri]))
                cols["time_bucket"].append(window)
                cols["true_labels"].append("|".join(sorted(true_specs)))
                cols["top5_predictions"].append("|".join(top5_specs))
                cols["top5_scores"].append("|".join(str(s) for s in top5_scores))
                cols["hit_at_1"].append(float(hits_cpu[j, 0]))
                cols["hit_at_3"].append(float(hits_cpu[j, 1]))
                cols["hit_at_5"].append(float(hits_cpu[j, 2]))
                cols["ndcg_at_1"].append(round(float(ndcg1[j]), 4))
                cols["ndcg_at_3"].append(round(float(ndcg3[j]), 4))
                cols["ndcg_at_5"].append(round(float(ndcg5[j]), 4))
                cols["model"].append("SASRec")       # BERT4Rec: "BERT4Rec"
                cols["sample"].append(SAMPLE)
                cols["run_timestamp"].append(RUN_TIMESTAMP)

        record_idx += bs

        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(score_loader)} | "
                  f"Rows so far: {len(cols['member_id']):,}")

n_scored  = len(cols["member_id"])
scores_df = pd.DataFrame(cols)
del cols
print(f"Scored {n_scored:,} trigger-window pairs")

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

for start in range(0, n_scored, BATCH_BQ):
    chunk = scores_df.iloc[start:start + BATCH_BQ]
    client.load_table_from_dataframe(
        chunk, f"{DS}.A870800_gen_rec_trigger_scores",
        job_config=job_cfg
    ).result()
    print(f"  Written {start:,} — {min(start+BATCH_BQ, n_scored):,}")

del scores_df
print(f"Section 5 done — {n_scored:,} rows written, time={time.time()-t0:.1f}s")
display(Markdown(f"**5:** {n_scored:,} trigger scores written | **Time:** {time.time()-t0:.1f}s"))
