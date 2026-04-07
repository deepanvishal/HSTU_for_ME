# ============================================================
# EDA_trigger_sequence_and_followup.py
# Purpose : Two analyses for qualified triggers
#   1. Visit history distribution before trigger (0-6, 7-20, 20+)
#   2. Days between trigger visit and follow-up visit
#      - Any follow-up visit
#      - Broken down by T30 / T60 / T180 windows
# Source  : A870800_gen_rec_triggers_qualified
#           A870800_gen_rec_visits
# ============================================================

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from google.cloud import bigquery
from IPython.display import display, Markdown

DS     = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
client = bigquery.Client(project="anbc-hcb-dev")

# ── Color palette ────────────────────────────────────────────
C_BLUE   = "#2C6FAC"
C_TEAL   = "#2A9D8F"
C_ORANGE = "#E76F51"
C_YELLOW = "#E9C46A"
C_DARK   = "#1A1A2E"
C_LIGHT  = "#F4F6F9"

display(Markdown("""
# EDA — Trigger Sequence & Follow-Up Visit Analysis

**Two questions:**
1. How much visit history do members have before their qualifying trigger?
2. How many days pass before a member's next visit after the trigger?
"""))


# ════════════════════════════════════════════════════════════
# SECTION 1: Pre-trigger visit history distribution
# ════════════════════════════════════════════════════════════

display(Markdown("---\n## Section 1 — Pre-Trigger Visit History"))
t0 = time.time()

seq_df = client.query(f"""
    WITH pre_trigger_counts AS (
        SELECT
            t.member_id
            ,t.trigger_date
            ,t.trigger_dx_clean
            -- Count distinct visit dates BEFORE trigger within full history
            ,COUNT(DISTINCT v.visit_date)                  AS visits_before_trigger
        FROM `{DS}.A870800_gen_rec_triggers_qualified` t
        LEFT JOIN `{DS}.A870800_gen_rec_visits` v
            ON t.member_id = v.member_id
            AND v.visit_date < t.trigger_date
        WHERE t.is_left_qualified = TRUE
        GROUP BY t.member_id, t.trigger_date, t.trigger_dx_clean
    )
    SELECT
        CASE
            WHEN visits_before_trigger BETWEEN 0 AND 6  THEN '0–6 visits'
            WHEN visits_before_trigger BETWEEN 7 AND 20 THEN '7–20 visits'
            WHEN visits_before_trigger > 20             THEN '20+ visits'
        END                                              AS visit_bucket
        ,COUNT(*)                                        AS trigger_count
        ,ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS pct
        ,ROUND(AVG(visits_before_trigger), 1)            AS avg_visits
    FROM pre_trigger_counts
    GROUP BY visit_bucket
    ORDER BY visit_bucket
""").to_dataframe()

print(f"Loaded in {time.time()-t0:.1f}s")
display(seq_df)
total_triggers = seq_df["trigger_count"].sum()
print(f"\nTotal qualified triggers: {total_triggers:,}")


# ── Plot 1a: Bar chart — trigger count by visit bucket ───────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=C_LIGHT)
fig.suptitle(
    "Pre-Trigger Visit History Distribution\nAmong Left-Qualified Triggers",
    fontsize=14, fontweight="bold", color=C_DARK, y=1.01
)

colors = [C_BLUE, C_TEAL, C_ORANGE]
buckets = seq_df["visit_bucket"].tolist()
counts  = seq_df["trigger_count"].tolist()
pcts    = seq_df["pct"].tolist()

# Bar chart — absolute counts
ax1 = axes[0]
bars = ax1.bar(buckets, counts, color=colors, edgecolor="white",
               linewidth=1.5, width=0.55)
for bar, pct in zip(bars, pcts):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total_triggers * 0.005,
        f"{pct:.1f}%",
        ha="center", va="bottom", fontsize=11,
        fontweight="bold", color=C_DARK
    )
ax1.set_title("Trigger Count by Pre-Trigger Visit Bucket",
              fontsize=11, fontweight="bold", color=C_DARK, pad=10)
ax1.set_ylabel("Number of Qualified Triggers", fontsize=10)
ax1.set_xlabel("Visits Before Trigger", fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"
))
ax1.set_facecolor(C_LIGHT)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# Pie chart — % split
ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(
    counts,
    labels=buckets,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 11, "color": C_DARK}
)
for at in autotexts:
    at.set_fontweight("bold")
    at.set_color("white")
ax2.set_title(
    f"% of Total Qualified Triggers\n(N = {total_triggers:,})",
    fontsize=11, fontweight="bold", color=C_DARK, pad=10
)
ax2.set_facecolor(C_LIGHT)

plt.tight_layout()
plt.savefig("eda_trigger_visit_history.png", dpi=150, bbox_inches="tight",
            facecolor=C_LIGHT)
plt.show()
print(f"Section 1 done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION 2: Days between trigger and next visit
# ════════════════════════════════════════════════════════════

display(Markdown("---\n## Section 2 — Days Between Trigger and Follow-Up Visit"))
t0 = time.time()

days_df = client.query(f"""
    WITH next_visits AS (
        SELECT
            t.member_id
            ,t.trigger_date
            ,t.trigger_dx_clean
            ,MIN(v.visit_date)                             AS next_visit_date
            ,DATE_DIFF(MIN(v.visit_date), t.trigger_date, DAY)
                                                           AS days_to_next_visit
        FROM `{DS}.A870800_gen_rec_triggers_qualified` t
        JOIN `{DS}.A870800_gen_rec_visits` v
            ON t.member_id = v.member_id
            AND v.visit_date > t.trigger_date
        WHERE t.is_left_qualified = TRUE
        GROUP BY t.member_id, t.trigger_date, t.trigger_dx_clean
    )
    SELECT
        days_to_next_visit
        ,CASE
            WHEN days_to_next_visit BETWEEN 1  AND 30  THEN 'T30 (1–30d)'
            WHEN days_to_next_visit BETWEEN 31 AND 60  THEN 'T60 (31–60d)'
            WHEN days_to_next_visit BETWEEN 61 AND 180 THEN 'T180 (61–180d)'
            ELSE                                             'Beyond 180d'
        END                                                AS time_window
        ,COUNT(*)                                          AS trigger_count
    FROM next_visits
    WHERE days_to_next_visit IS NOT NULL
      AND days_to_next_visit > 0
    GROUP BY days_to_next_visit, time_window
    ORDER BY days_to_next_visit
""").to_dataframe()

print(f"Loaded {len(days_df):,} rows in {time.time()-t0:.1f}s")

# Window summary
window_summary = (
    days_df.groupby("time_window")["trigger_count"]
    .sum()
    .reset_index()
    .sort_values("time_window")
)
window_summary["pct"] = (
    100.0 * window_summary["trigger_count"] / window_summary["trigger_count"].sum()
).round(2)
display(window_summary)


# ── Plot 2a: Distribution of days — ALL follow-up visits ─────
fig = plt.figure(figsize=(16, 14), facecolor=C_LIGHT)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle(
    "Days Between Trigger Visit and Follow-Up Visit",
    fontsize=15, fontweight="bold", color=C_DARK, y=1.01
)

WINDOW_COLORS = {
    "T30 (1–30d)":    C_BLUE,
    "T60 (31–60d)":   C_TEAL,
    "T180 (61–180d)": C_ORANGE,
    "Beyond 180d":    C_YELLOW,
}

# -- Top row spanning: full distribution (days 1–180)
ax_full = fig.add_subplot(gs[0, :])
full_180 = days_df[days_df["days_to_next_visit"] <= 180].copy()
colors_bar = full_180["time_window"].map(WINDOW_COLORS)

ax_full.bar(
    full_180["days_to_next_visit"],
    full_180["trigger_count"],
    color=colors_bar,
    width=1.0,
    edgecolor="none"
)
ax_full.set_title(
    "Full Distribution — Days to Next Visit (1–180 days)\nColor = Time Window",
    fontsize=11, fontweight="bold", color=C_DARK
)
ax_full.set_xlabel("Days After Trigger", fontsize=10)
ax_full.set_ylabel("Number of Triggers", fontsize=10)
ax_full.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"
))
ax_full.set_facecolor(C_LIGHT)
ax_full.spines[["top", "right"]].set_visible(False)
ax_full.grid(axis="y", linestyle="--", alpha=0.35)

# Legend
from matplotlib.patches import Patch
legend_patches = [
    Patch(color=v, label=k) for k, v in WINDOW_COLORS.items()
]
ax_full.legend(handles=legend_patches, loc="upper right", fontsize=9,
               framealpha=0.8)

# Median line
median_days = np.average(
    days_df[days_df["days_to_next_visit"] <= 180]["days_to_next_visit"],
    weights=days_df[days_df["days_to_next_visit"] <= 180]["trigger_count"]
)
ax_full.axvline(median_days, color="crimson", linewidth=1.5,
                linestyle="--", alpha=0.8)
ax_full.text(median_days + 2, ax_full.get_ylim()[1] * 0.9,
             f"Weighted mean\n{median_days:.0f}d",
             color="crimson", fontsize=9, fontweight="bold")


# -- Bottom row: one plot per window
windows_ordered = ["T30 (1–30d)", "T60 (31–60d)", "T180 (61–180d)", "Beyond 180d"]
window_ranges   = {
    "T30 (1–30d)":    (1,   30),
    "T60 (31–60d)":   (31,  60),
    "T180 (61–180d)": (61, 180),
    "Beyond 180d":    (181, days_df["days_to_next_visit"].max()),
}
positions = [(1, 0), (1, 1), (2, 0), (2, 1)]

for (row, col), window in zip(positions, windows_ordered):
    ax = fig.add_subplot(gs[row, col])
    lo, hi = window_ranges[window]
    sub = days_df[
        (days_df["days_to_next_visit"] >= lo) &
        (days_df["days_to_next_visit"] <= hi)
    ]
    color = WINDOW_COLORS[window]
    total_w = sub["trigger_count"].sum()
    pct_w   = 100.0 * total_w / days_df["trigger_count"].sum()

    ax.bar(
        sub["days_to_next_visit"],
        sub["trigger_count"],
        color=color,
        width=1.0,
        edgecolor="none",
        alpha=0.85
    )

    if len(sub) > 0:
        wmean = np.average(
            sub["days_to_next_visit"],
            weights=sub["trigger_count"]
        )
        ax.axvline(wmean, color="black", linewidth=1.5,
                   linestyle="--", alpha=0.7)
        ax.text(
            wmean + (hi - lo) * 0.03,
            sub["trigger_count"].max() * 0.88,
            f"Mean: {wmean:.0f}d",
            color="black", fontsize=8, fontweight="bold"
        )

    ax.set_title(
        f"{window}\nN = {total_w:,}  ({pct_w:.1f}% of all triggers)",
        fontsize=10, fontweight="bold", color=C_DARK
    )
    ax.set_xlabel("Days After Trigger", fontsize=9)
    ax.set_ylabel("Trigger Count", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    ax.set_facecolor(C_LIGHT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.savefig("eda_trigger_followup_days.png", dpi=150, bbox_inches="tight",
            facecolor=C_LIGHT)
plt.show()
print(f"Section 2 done — {time.time()-t0:.1f}s")


# ════════════════════════════════════════════════════════════
# SECTION 3: Summary table
# ════════════════════════════════════════════════════════════

display(Markdown("---\n## Section 3 — Combined Summary"))

summary_rows = []
for _, row in seq_df.iterrows():
    summary_rows.append({
        "Analysis":    "Pre-Trigger History",
        "Bucket":      row["visit_bucket"],
        "Count":       f"{row['trigger_count']:,}",
        "Pct":         f"{row['pct']:.1f}%",
        "Avg Visits":  f"{row['avg_visits']:.1f}"
    })
for _, row in window_summary.iterrows():
    summary_rows.append({
        "Analysis":    "Days to Next Visit",
        "Bucket":      row["time_window"],
        "Count":       f"{row['trigger_count']:,}",
        "Pct":         f"{row['pct']:.1f}%",
        "Avg Visits":  "—"
    })

summary_out = pd.DataFrame(summary_rows)
display(summary_out)

print("\nAll sections complete.")
