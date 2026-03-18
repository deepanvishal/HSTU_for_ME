# ============================================================
# NB_04 — Qualified Population Profile
# Purpose : Characterize the population that remains after
#           boundary setting. Who are we actually analyzing?
#           Is the population representative and large enough?
# Sources : A870800_gen_rec_triggers_qualified
#           A870800_gen_rec_visits_qualified
# ============================================================
from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, Markdown

client = bigquery.Client(project="anbc-hcb-dev")
DATASET = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
TQ = f"`{DATASET}.A870800_gen_rec_triggers_qualified`"
VQ = f"`{DATASET}.A870800_gen_rec_visits_qualified`"

def fmt_millions(x):
    return f"${x/1_000_000:.2f}M"

def fmt_usd(x):
    return f"${x:,.0f}"

def fmt_count(x):
    return f"{int(x):,}"

def fmt_pct(x):
    return f"{x:.1f}%"


display(Markdown("""
---
# NB 04 — Qualified Population Profile
## Who Are We Actually Analyzing?

After applying left and right boundary rules, this notebook profiles
the population that remains — the members and triggers that form the
foundation of all downstream analysis and modeling.

The purpose is to answer:
- Who is in this population demographically?
- Is the population stable across years?
- Which clinical domains and specialties dominate?
- Is the qualified population representative of the broader enrolled population?

All figures in this notebook reflect the left-qualified population
unless otherwise stated.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MEMBER DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 1 — Member Demographics

Age and gender distribution of members with at least one left-qualified trigger.
"""))

demographics = client.query(f"""
SELECT
    member_segment
    ,gender_cd
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,ROUND(AVG(age_nbr), 1)                              AS avg_age
    ,MIN(age_nbr)                                        AS min_age
    ,MAX(age_nbr)                                        AS max_age
FROM {TQ}
WHERE is_left_qualified = TRUE
GROUP BY member_segment, gender_cd
ORDER BY member_segment, gender_cd
""").to_dataframe()

display(demographics.rename(columns={
    "member_segment": "Segment",
    "gender_cd": "Gender",
    "unique_members": "Unique Members",
    "avg_age": "Avg Age",
    "min_age": "Min Age",
    "max_age": "Max Age"
}).reset_index(drop=True))

# cohort distribution
cohort_dist = client.query(f"""
SELECT
    member_segment
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,COUNT(*)                                            AS total_triggers
FROM {TQ}
WHERE is_left_qualified = TRUE
GROUP BY member_segment
ORDER BY unique_members DESC
""").to_dataframe()

total_members = cohort_dist["unique_members"].sum()
cohort_dist["pct_members"] = (cohort_dist["unique_members"] / total_members * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

colors = ["#4C9BE8", "#F4845F", "#5DBE7E", "#F7C948"]
axes[0].pie(cohort_dist["unique_members"],
            labels=cohort_dist["member_segment"],
            autopct="%1.1f%%", colors=colors,
            startangle=90, textprops={"fontsize": 10})
axes[0].set_title("Member Distribution by Cohort", fontsize=12, fontweight="bold")

axes[1].bar(cohort_dist["member_segment"],
            cohort_dist["total_triggers"],
            color=colors, alpha=0.85)
axes[1].set_title("Total Qualified Triggers by Cohort", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Trigger Count", fontsize=10)
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Qualified Population — Cohort Distribution",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("cohort_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# age distribution
age_dist = client.query(f"""
SELECT
    age_nbr
    ,COUNT(DISTINCT member_id)                           AS unique_members
FROM {TQ}
WHERE is_left_qualified = TRUE
  AND age_nbr IS NOT NULL
GROUP BY age_nbr
ORDER BY age_nbr
""").to_dataframe()

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(age_dist["age_nbr"], age_dist["unique_members"],
       color="#4C9BE8", alpha=0.8, width=0.8)
ax.axvline(18, color="#F4845F", linestyle="--", alpha=0.7, label="Age 18")
ax.axvline(65, color="#5DBE7E", linestyle="--", alpha=0.7, label="Age 65")
ax.set_xlabel("Age", fontsize=10)
ax.set_ylabel("Unique Members", fontsize=10)
ax.set_title("Age Distribution — Qualified Population",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("age_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — QUALIFIED TRIGGERS BY YEAR
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 2 — Qualified Triggers by Year

How are qualified triggers distributed across the analysis period?
Stable trigger counts across years indicate consistent data.
2022 is expected to have lower counts — it is the runway year where
most members do not yet have 12 months of enrollment history.
"""))

triggers_year = client.query(f"""
SELECT
    EXTRACT(YEAR FROM trigger_date)                      AS trigger_year
    ,COUNT(*)                                            AS total_triggers
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,COUNT(DISTINCT trigger_dx_clean)                    AS unique_dx_codes
    ,COUNTIF(is_t30_qualified = TRUE)                    AS t30_qualified
    ,COUNTIF(is_t60_qualified = TRUE)                    AS t60_qualified
    ,COUNTIF(is_t180_qualified = TRUE)                   AS t180_qualified
FROM {TQ}
WHERE is_left_qualified = TRUE
GROUP BY trigger_year
ORDER BY trigger_year
""").to_dataframe()

display(triggers_year.rename(columns={
    "trigger_year": "Year",
    "total_triggers": "Total Triggers",
    "unique_members": "Unique Members",
    "unique_dx_codes": "Unique DX Codes",
    "t30_qualified": "T30 Qualified",
    "t60_qualified": "T60 Qualified",
    "t180_qualified": "T180 Qualified"
}).reset_index(drop=True))

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

x = triggers_year["trigger_year"].astype(str)
axes[0].bar(x, triggers_year["total_triggers"], color="#4C9BE8", alpha=0.85)
axes[0].set_title("Left Qualified Triggers by Year", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Trigger Count", fontsize=10)
axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

width = 0.25
xi = np.arange(len(triggers_year))
axes[1].bar(xi - width, triggers_year["t30_qualified"],  width,
            label="T30", color="#5DBE7E", alpha=0.85)
axes[1].bar(xi,         triggers_year["t60_qualified"],  width,
            label="T60", color="#F7C948", alpha=0.85)
axes[1].bar(xi + width, triggers_year["t180_qualified"], width,
            label="T180", color="#F4845F", alpha=0.85)
axes[1].set_xticks(xi)
axes[1].set_xticklabels(triggers_year["trigger_year"].astype(str))
axes[1].set_title("Window-Qualified Triggers by Year", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Trigger Count", fontsize=10)
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
axes[1].legend(fontsize=9)
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Qualified Triggers by Year", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("triggers_by_year.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRIGGERS BY SPECIALTY — FP/I VS OTHER
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 3 — Triggers by Trigger Specialty

Which specialty was the member visiting when the trigger diagnosis was made?

Family Practice (FP) and Internal Medicine (I) are the primary care specialties
expected to dominate first encounter triggers. This section validates that
assumption and shows the full specialty breakdown.

The FP/I lens used in EDA restricts analysis to triggers occurring at
primary care visits — where referral routing decisions are most clinically meaningful.
"""))

trigger_specialty = client.query(f"""
SELECT
    trigger_specialty
    ,trigger_specialty_desc
    ,COUNT(*)                                            AS trigger_count
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,COUNTIF(is_t180_qualified = TRUE)                   AS t180_qualified
FROM {TQ}
WHERE is_left_qualified = TRUE
  AND trigger_specialty IS NOT NULL
GROUP BY trigger_specialty, trigger_specialty_desc
ORDER BY trigger_count DESC
LIMIT 20
""").to_dataframe()

total_triggers = trigger_specialty["trigger_count"].sum()
trigger_specialty["pct"] = (trigger_specialty["trigger_count"] / total_triggers * 100).round(1)

fp_i = trigger_specialty[trigger_specialty["trigger_specialty"].isin(["FP", "I"])]["trigger_count"].sum()
other = total_triggers - fp_i

display(Markdown(f"""
**FP and Internal Medicine triggers:** {fmt_count(fp_i)} ({fmt_pct(fp_i/total_triggers*100)} of total)
**All other specialties:** {fmt_count(other)} ({fmt_pct(other/total_triggers*100)} of total)
"""))

display(trigger_specialty[[
    "trigger_specialty_desc", "trigger_count", "unique_members",
    "t180_qualified", "pct"
]].rename(columns={
    "trigger_specialty_desc": "Trigger Specialty",
    "trigger_count": "Trigger Count",
    "unique_members": "Unique Members",
    "t180_qualified": "T180 Qualified",
    "pct": "% of Total"
}).reset_index(drop=True))

fig, axes = plt.subplots(1, 2, figsize=(22, 8))

# FP/I vs other pie
axes[0].pie([fp_i, other],
            labels=["FP and Internal Medicine", "All Other Specialties"],
            autopct="%1.1f%%",
            colors=["#4C9BE8", "#CCCCCC"],
            startangle=90, textprops={"fontsize": 10})
axes[0].set_title("FP/I vs Other Specialty Triggers", fontsize=12, fontweight="bold")

# top 15 specialties bar
top15 = trigger_specialty.head(15).sort_values("trigger_count", ascending=True)
bar_colors = ["#4C9BE8" if s in ["FP", "I"] else "#CCCCCC"
              for s in top15["trigger_specialty"]]
axes[1].barh(top15["trigger_specialty_desc"].str[:35],
             top15["trigger_count"], color=bar_colors, alpha=0.85)
axes[1].set_xlabel("Trigger Count", fontsize=10)
axes[1].set_title("Top 15 Trigger Specialties\n(blue = FP/I)",
                  fontsize=11, fontweight="bold")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
plt.setp(axes[1].get_yticklabels(), fontsize=8)
axes[1].grid(axis="x", linestyle="--", alpha=0.4)

fig.suptitle("Qualified Triggers by Specialty", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("trigger_specialty.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRIGGERS BY CCSR DOMAIN
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 4 — Qualified Triggers by Clinical Domain

Which clinical domains generate the most qualified triggers?

CCSR groups ICD-10 codes into approximately 530 clinical categories.
The top domains here represent the conditions most frequently
presenting as first encounters in this population.

These domains drive the bulk of the transition analysis and model training.
"""))

trigger_ccsr = client.query(f"""
SELECT
    trigger_ccsr
    ,trigger_ccsr_desc
    ,COUNT(*)                                            AS trigger_count
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,COUNT(DISTINCT trigger_dx_clean)                    AS unique_dx_codes
    ,COUNTIF(is_t180_qualified = TRUE)                   AS t180_qualified
FROM {TQ}
WHERE is_left_qualified = TRUE
  AND trigger_ccsr IS NOT NULL
GROUP BY trigger_ccsr, trigger_ccsr_desc
ORDER BY trigger_count DESC
LIMIT 20
""").to_dataframe()

display(trigger_ccsr[[
    "trigger_ccsr_desc", "trigger_count", "unique_members",
    "unique_dx_codes", "t180_qualified"
]].rename(columns={
    "trigger_ccsr_desc": "Clinical Domain",
    "trigger_count": "Trigger Count",
    "unique_members": "Unique Members",
    "unique_dx_codes": "Unique DX Codes",
    "t180_qualified": "T180 Qualified"
}).reset_index(drop=True))

fig, axes = plt.subplots(1, 2, figsize=(24, 9))

top15_ccsr = trigger_ccsr.head(15).sort_values("trigger_count", ascending=True)
axes[0].barh(top15_ccsr["trigger_ccsr_desc"].str[:40],
             top15_ccsr["trigger_count"],
             color="#4C9BE8", alpha=0.85)
axes[0].set_xlabel("Trigger Count", fontsize=10)
axes[0].set_title("Top 15 Clinical Domains by Trigger Volume",
                  fontsize=11, fontweight="bold")
axes[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
plt.setp(axes[0].get_yticklabels(), fontsize=8)
axes[0].grid(axis="x", linestyle="--", alpha=0.4)

top15_ccsr_m = trigger_ccsr.sort_values("unique_members", ascending=False)\
               .head(15).sort_values("unique_members", ascending=True)
axes[1].barh(top15_ccsr_m["trigger_ccsr_desc"].str[:40],
             top15_ccsr_m["unique_members"],
             color="#F4845F", alpha=0.85)
axes[1].set_xlabel("Unique Members", fontsize=10)
axes[1].set_title("Top 15 Clinical Domains by Unique Members",
                  fontsize=11, fontweight="bold")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
plt.setp(axes[1].get_yticklabels(), fontsize=8)
axes[1].grid(axis="x", linestyle="--", alpha=0.4)

fig.suptitle("Qualified Triggers by Clinical Domain",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("trigger_ccsr.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DOWNSTREAM VISITS PER MEMBER
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 5 — Downstream Visits Per Member After Trigger

How many visits does a member make after a qualified trigger within T180?

This shows the richness of the downstream observation window.
Members with more downstream visits provide richer training signal.
Members with zero downstream visits contribute a trigger but no label.
"""))

downstream_dist = client.query(f"""
SELECT
    member_id
    ,trigger_date
    ,trigger_dx
    ,COUNT(DISTINCT visit_date)                          AS downstream_visit_count
    ,ROUND(SUM(allowed_amt), 2)                          AS downstream_spend
FROM {VQ}
WHERE is_left_qualified = TRUE
GROUP BY member_id, trigger_date, trigger_dx
""").to_dataframe()

downstream_dist["downstream_visit_count"] = downstream_dist["downstream_visit_count"].astype(float)
downstream_dist["downstream_spend"] = downstream_dist["downstream_spend"].astype(float)

pcts = downstream_dist["downstream_visit_count"].describe(
    percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

display(Markdown(f"""
| Percentile | Downstream Visits |
|---|---|
| 10th | {pcts['10%']:.0f} |
| 25th | {pcts['25%']:.0f} |
| Median | {pcts['50%']:.0f} |
| 75th | {pcts['75%']:.0f} |
| 90th | {pcts['90%']:.0f} |
| 95th | {pcts['95%']:.0f} |
| Mean | {pcts['mean']:.1f} |

**Triggers with zero downstream visits:** {fmt_count((downstream_dist['downstream_visit_count'] == 0).sum())}
({fmt_pct((downstream_dist['downstream_visit_count'] == 0).sum() / len(downstream_dist) * 100)})
"""))

fig, ax = plt.subplots(figsize=(14, 6))
cap = downstream_dist["downstream_visit_count"].quantile(0.95)
plot_data = downstream_dist[downstream_dist["downstream_visit_count"] <= cap]["downstream_visit_count"]
ax.hist(plot_data, bins=40, color="#4C9BE8", alpha=0.8, edgecolor="white")
for pct, val, color in [
    ("p25", pcts["25%"], "#5DBE7E"),
    ("p50", pcts["50%"], "orange"),
    ("p75", pcts["75%"], "#F4845F"),
]:
    ax.axvline(val, color=color, linestyle="--", alpha=0.8,
               label=f"p{int(pct[1:])} = {val:.0f}")
ax.set_xlabel("Downstream Visits Within T180 Per Trigger", fontsize=10)
ax.set_ylabel("Number of Triggers", fontsize=10)
ax.set_title("Distribution of Downstream Visits Per Trigger\n(capped at 95th percentile)",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("downstream_visit_dist.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SPEND DISTRIBUTION AFTER QUALIFICATION
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 6 — Spend Distribution After Qualification

How is allowed spend distributed across qualified triggers?

This shows the financial scope of the qualified population and
identifies whether spend is concentrated in a small number of triggers
or broadly distributed.

Total spend shown in millions USD. Per-trigger and per-member
figures shown in USD.
"""))

spend_summary = client.query(f"""
SELECT
    ROUND(SUM(allowed_amt), 2)                           AS total_allowed_amt
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,COUNT(DISTINCT CONCAT(member_id, '_',
        CAST(trigger_date AS STRING), '_',
        trigger_dx))                                     AS unique_triggers
    ,ROUND(SUM(allowed_amt) /
        NULLIF(COUNT(DISTINCT member_id), 0), 2)         AS avg_per_member
    ,ROUND(SUM(allowed_amt) /
        NULLIF(COUNT(DISTINCT CONCAT(member_id, '_',
            CAST(trigger_date AS STRING), '_',
            trigger_dx)), 0), 2)                         AS avg_per_trigger
FROM {VQ}
WHERE is_left_qualified = TRUE
""").to_dataframe()

s = spend_summary.iloc[0]

display(Markdown(f"""
| Metric | Value |
|---|---|
| Total downstream allowed spend | {fmt_millions(s['total_allowed_amt'])} |
| Unique members | {fmt_count(s['unique_members'])} |
| Unique triggers | {fmt_count(s['unique_triggers'])} |
| Avg spend per member | {fmt_usd(s['avg_per_member'])} |
| Avg spend per trigger | {fmt_usd(s['avg_per_trigger'])} |
"""))

# spend by window
spend_by_window = client.query(f"""
SELECT
    CASE
        WHEN days_since_trigger <= 30  THEN 'T0_30'
        WHEN days_since_trigger <= 60  THEN 'T30_60'
        WHEN days_since_trigger <= 180 THEN 'T60_180'
    END                                                  AS time_bucket
    ,ROUND(SUM(allowed_amt), 2)                          AS total_allowed_amt
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,COUNT(DISTINCT CONCAT(member_id, '_',
        CAST(trigger_date AS STRING), '_',
        trigger_dx))                                     AS unique_triggers
FROM {VQ}
WHERE is_left_qualified = TRUE
  AND days_since_trigger <= 180
GROUP BY time_bucket
ORDER BY time_bucket
""").to_dataframe()

display(Markdown("#### Downstream Spend by Time Window"))
spend_window_display = spend_by_window.copy()
spend_window_display["total_allowed_amt"] = spend_window_display["total_allowed_amt"].apply(fmt_millions)
spend_window_display["unique_members"] = spend_window_display["unique_members"].apply(fmt_count)
spend_window_display["unique_triggers"] = spend_window_display["unique_triggers"].apply(fmt_count)
display(spend_window_display.rename(columns={
    "time_bucket": "Time Window",
    "total_allowed_amt": "Total Spend (USD M)",
    "unique_members": "Unique Members",
    "unique_triggers": "Unique Triggers"
}).reset_index(drop=True))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

colors = ["#5DBE7E", "#F7C948", "#F4845F"]
axes[0].bar(spend_by_window["time_bucket"],
            spend_by_window["total_allowed_amt"] / 1_000_000,
            color=colors, alpha=0.85)
axes[0].set_ylabel("Total Spend (USD M)", fontsize=10)
axes[0].set_title("Downstream Spend by Time Window", fontsize=11, fontweight="bold")
axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

per_member_spend = client.query(f"""
SELECT
    member_id
    ,ROUND(SUM(allowed_amt), 2) AS total_spend
FROM {VQ}
WHERE is_left_qualified = TRUE
GROUP BY member_id
""").to_dataframe()

per_member_spend["total_spend"] = per_member_spend["total_spend"].astype(float)
cap = per_member_spend["total_spend"].quantile(0.95)
plot_data = per_member_spend[per_member_spend["total_spend"] <= cap]["total_spend"]
axes[1].hist(plot_data, bins=50, color="#4C9BE8", alpha=0.8, edgecolor="white")
axes[1].set_xlabel("Total Downstream Spend Per Member ($)", fontsize=10)
axes[1].set_ylabel("Number of Members", fontsize=10)
axes[1].set_title("Distribution of Downstream Spend Per Member\n(capped at 95th percentile)",
                  fontsize=11, fontweight="bold")
axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_usd(x)))
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Qualified Population — Spend Distribution",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("spend_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Summary — Qualified Population Profile

This section characterizes the population that forms the basis of
all downstream transition analysis and model training.

**Demographics:**
The qualified population spans four cohorts — Adult Female, Adult Male,
Senior, and Children. The cohort distribution shows whether the population
is broadly representative or skewed toward a specific age group.

**Temporal stability:**
Trigger volume by year shows whether the analysis population is consistent
across the 2022-2025 window. Significant drops in any year warrant investigation.

**Clinical scope:**
The top CCSR domains and trigger specialties define the clinical breadth
of the analysis. The FP/I share of triggers validates the primary care
first-encounter framing used in the FP/I analytical lens.

**Downstream richness:**
The distribution of downstream visits per trigger shows whether the
observation window produces enough follow-up data per trigger for
reliable transition analysis.

**Financial scope:**
Spend distribution confirms the financial relevance of the qualified
population and shows how downstream costs accumulate across T30, T60, and T180.

Transition analysis and model training begin in the next section.

---
"""))
