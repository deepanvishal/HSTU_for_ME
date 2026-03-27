# ============================================================
# Block 8c — Provider Transition Analysis
# 1. Top transitions by volume (all)
# 2. Top transitions by volume (excl Lab)
# 3. Top transitions by accuracy (excl Lab, min 20 predictions)
# 4. Model performance across member segment
# Requires: client, DS, PROV_BEST from Block 8
# Source: provider_eval_5pct, provider_model_test_agg_5pct,
#         provider_markov_train_5pct, provider_name_lookup,
#         provider_primary_specialty
# ============================================================
from IPython.display import display, Markdown

PROV_BEST = "SASRec"

# ── 1/2/3: Transition-level analysis ─────────────────────────
transitions = client.query(f"""
    WITH
    eval_with_from AS (
        SELECT
            e.member_id, e.trigger_date, e.trigger_dx, e.member_segment
            ,e.time_bucket, e.model
            ,e.top5_predictions, e.top5_scores, e.true_labels
            ,CAST(t.from_provider AS STRING) AS from_provider
        FROM `{DS}.A870800_gen_rec_provider_eval_5pct` e
        JOIN `{DS}.A870800_gen_rec_provider_model_test_agg_5pct` t
            ON CAST(e.member_id AS STRING) = CAST(t.member_id AS STRING)
            AND CAST(e.trigger_date AS STRING) = CAST(t.trigger_date AS STRING)
            AND CAST(e.trigger_dx AS STRING) = CAST(t.trigger_dx AS STRING)
            AND CAST(e.member_segment AS STRING) = CAST(t.member_segment AS STRING)
        WHERE e.time_bucket = 'T0_30'
          AND e.model = '{PROV_BEST}'
          AND (e.tp + e.fn) > 0
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY CAST(e.member_id AS STRING), CAST(e.trigger_date AS STRING),
                         CAST(e.trigger_dx AS STRING), CAST(e.member_segment AS STRING),
                         e.time_bucket, e.model
            ORDER BY CAST(t.from_provider AS STRING)
        ) = 1
    ),
    exploded AS (
        SELECT
            ef.from_provider
            ,CAST(pred AS STRING) AS to_provider
            ,IF(CONCAT('|', ef.true_labels, '|')
                LIKE CONCAT('%|', CAST(pred AS STRING), '|%'), 1, 0) AS is_correct
        FROM eval_with_from ef
        CROSS JOIN UNNEST(SPLIT(ef.top5_predictions, '|')) AS pred WITH OFFSET pos
        WHERE pos = 0
          AND pred IS NOT NULL AND TRIM(pred) != ''
    ),
    agg AS (
        SELECT
            from_provider
            ,to_provider
            ,COUNT(*) AS times_predicted
            ,SUM(is_correct) AS times_correct
            ,ROUND(SAFE_DIVIDE(SUM(is_correct), COUNT(*)), 4) AS accuracy
        FROM exploded
        GROUP BY 1, 2
    ),
    train_evidence AS (
        SELECT
            CAST(from_provider AS STRING) AS from_provider
            ,CAST(to_provider AS STRING) AS to_provider
            ,SUM(transition_count) AS train_transitions
        FROM `{DS}.A870800_gen_rec_provider_markov_train_5pct`
        GROUP BY 1, 2
    ),
    enriched AS (
        SELECT
            a.from_provider
            ,a.to_provider
            ,a.times_predicted
            ,a.times_correct
            ,a.accuracy
            ,COALESCE(te.train_transitions, 0) AS train_transitions
            ,COALESCE(n1.provider_name, a.from_provider) AS from_name
            ,COALESCE(n2.provider_name, a.to_provider) AS to_name
            ,COALESCE(ps1.primary_specialty, 'UNK') AS from_specialty_cd
            ,COALESCE(ps2.primary_specialty, 'UNK') AS to_specialty_cd
            ,COALESCE(g1.global_lookup_desc, 'Unknown') AS from_specialty
            ,COALESCE(g2.global_lookup_desc, 'Unknown') AS to_specialty
        FROM agg a
        LEFT JOIN train_evidence te
            ON a.from_provider = te.from_provider AND a.to_provider = te.to_provider
        LEFT JOIN `{DS}.A870800_gen_rec_provider_name_lookup` n1
            ON a.from_provider = CAST(n1.srv_prvdr_id AS STRING)
        LEFT JOIN `{DS}.A870800_gen_rec_provider_name_lookup` n2
            ON a.to_provider = CAST(n2.srv_prvdr_id AS STRING)
        LEFT JOIN `{DS}.A870800_gen_rec_provider_primary_specialty` ps1
            ON a.from_provider = CAST(ps1.srv_prvdr_id AS STRING)
        LEFT JOIN `{DS}.A870800_gen_rec_provider_primary_specialty` ps2
            ON a.to_provider = CAST(ps2.srv_prvdr_id AS STRING)
        LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` g1
            ON g1.global_lookup_cd = ps1.primary_specialty
            AND LOWER(g1.lookup_column_nm) = 'specialty_ctg_cd'
        LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.GLOBAL_LOOKUP` g2
            ON g2.global_lookup_cd = ps2.primary_specialty
            AND LOWER(g2.lookup_column_nm) = 'specialty_ctg_cd'
    )
    SELECT * FROM enriched
    ORDER BY times_predicted DESC
""").to_dataframe()

# ── TABLE 1: Top 15 transitions by volume (all) ──
display(Markdown("### Table 1: Top 15 Transitions by Volume (All)"))
t1 = transitions.head(15)[["from_name", "from_specialty", "to_name", "to_specialty",
                            "times_predicted", "times_correct", "accuracy", "train_transitions"]]
display(t1)

# ── TABLE 2: Top 15 transitions by volume (excl Lab) ──
display(Markdown("### Table 2: Top 15 Transitions by Volume (Excl Lab)"))
no_lab = transitions[
    ~transitions["from_specialty"].str.contains("Lab", case=False, na=False) &
    ~transitions["to_specialty"].str.contains("Lab", case=False, na=False)
]
t2 = no_lab.head(15)[["from_name", "from_specialty", "to_name", "to_specialty",
                       "times_predicted", "times_correct", "accuracy", "train_transitions"]]
display(t2)

# ── TABLE 3: Top 15 transitions by accuracy (excl Lab, min 20 predictions) ──
display(Markdown("### Table 3: Highest Accuracy Transitions (Excl Lab, min 20)"))
t3 = (no_lab[no_lab["times_predicted"] >= 20]
      .sort_values("accuracy", ascending=False)
      .head(15)[["from_name", "from_specialty", "to_name", "to_specialty",
                  "times_predicted", "times_correct", "accuracy", "train_transitions"]])
display(t3)


# ── TABLE 4: Model performance by member segment ─────────────
display(Markdown("### Table 4: All Models — Hit@5 by Member Segment at T30"))
seg_all = client.query(f"""
    SELECT
        model
        ,member_segment
        ,COUNT(*) AS n_triggers
        ,ROUND(AVG(hit_at_5), 4) AS hit_at_5
    FROM `{DS}.A870800_gen_rec_provider_eval_5pct`
    WHERE time_bucket = 'T0_30' AND (tp + fn) > 0
    GROUP BY model, member_segment
    ORDER BY model, hit_at_5 DESC
""").to_dataframe()
seg_all["member_segment"] = seg_all["member_segment"].replace({
    "Adult_Female": "Adult Female", "Adult_Male": "Adult Male"})

seg_pivot = seg_all.pivot(index="model", columns="member_segment", values="hit_at_5")
seg_pivot = (seg_pivot * 100).round(1)
display(seg_pivot)

print("Block 8c done.")
