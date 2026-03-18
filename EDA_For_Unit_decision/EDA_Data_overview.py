# ============================================================
# NB_01 — Data Overview
# Purpose : Raw data characteristics before any filtering
#           Establishes baseline population, volume, quality
#           and visit behavior for South Florida market
# Sources : A870800_claims_gen_rec_2022_2025_sfl
#           A870800_gen_rec_member_qualified
#           A870800_gen_rec_visits
# ============================================================
from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, Markdown

client = bigquery.Client(project="anbc-hcb-dev")
DATASET = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"
RAW = f"`{DATASET}.A870800_claims_gen_rec_2022_2025_sfl`"
MEMBERS = f"`{DATASET}.A870800_gen_rec_member_qualified`"
VISITS = f"`{DATASET}.A870800_gen_rec_visits`"


def fmt_millions(x):
    return f"${x/1_000_000:.2f}M"

def fmt_usd(x):
    return f"${x:,.0f}"

def fmt_count(x):
    return f"{x:,.0f}"


display(Markdown("""
---
# NB 01 — Data Overview
## South Florida Claims Data — 2022 to 2025

This notebook establishes the baseline picture of the South Florida market
before any analytical filters are applied.

The purpose is to answer:
- How large is this population?
- What does the raw data look like?
- Is the data consistent and complete enough to support analysis?
- How do members use the healthcare system?

All findings here inform the boundary setting decisions documented in NB 02.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 2 — RAW DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Chapter 2 — Raw Data Overview

### Market and Time Frame

**Geography:** South Florida
**Time Frame:** January 2022 to December 2025
**Source:** CVS/Aetna medical claims

The analysis uses a 4-year window. 2022 serves as the runway year —
providing claims history for left boundary qualification of 2023 triggers.
The primary analysis period is 2023 through mid-2025.

---
"""))


# ── 2.1 POPULATION SCOPE ─────────────────────────────────────────────────────
display(Markdown("""
### 2.1 Population Scope

Total enrolled members vs members who actually generated claims.
The gap between enrolled and claimants represents members who were in the plan
but did not seek care — important context for understanding data coverage.
"""))

pop = client.query(f"""
SELECT
    COUNT(DISTINCT member_id)                            AS total_enrolled
    ,(SELECT COUNT(DISTINCT member_id) FROM {RAW})       AS total_claimants
FROM {MEMBERS}
""").to_dataframe()

display(Markdown(f"""
| Metric | Count |
|---|---|
| Total enrolled members | {fmt_count(pop['total_enrolled'].iloc[0])} |
| Total claimants (members with claims) | {fmt_count(pop['total_claimants'].iloc[0])} |
| Non-claimant enrolled members | {fmt_count(pop['total_enrolled'].iloc[0] - pop['total_claimants'].iloc[0])} |
| Claimant rate | {pop['total_claimants'].iloc[0] / pop['total_enrolled'].iloc[0] * 100:.1f}% |
"""))


# ── 2.2 MEMBERS BY YEAR ───────────────────────────────────────────────────────
display(Markdown("""
### 2.2 Members and Claimants by Year

Year-by-year breakdown of enrolled members and active claimants.
Stable counts across years indicate consistent data. Large drops may indicate
data gaps or plan changes worth flagging.
"""))

members_year = client.query(f"""
SELECT
    EXTRACT(YEAR FROM eff_dt)                           AS year
    ,COUNT(DISTINCT member_id)                           AS enrolled_members
FROM `{DATASET}.A870800_claims_gen_rec_members`
GROUP BY year
ORDER BY year
""").to_dataframe()

claimants_year = client.query(f"""
SELECT
    EXTRACT(YEAR FROM srv_start_dt)                     AS year
    ,COUNT(DISTINCT member_id)                           AS claimants
    ,COUNT(DISTINCT CONCAT(member_id, '_',
        CAST(srv_start_dt AS STRING)))                   AS total_visits
    ,COUNT(*)                                            AS total_claims
    ,ROUND(SUM(allowed_amt), 2)                          AS total_allowed_amt
FROM {RAW}
GROUP BY year
ORDER BY year
""").to_dataframe()

yearly = members_year.merge(claimants_year, on="year", how="outer").fillna(0)
yearly_display = yearly.copy()
yearly_display["enrolled_members"] = yearly_display["enrolled_members"].apply(fmt_count)
yearly_display["claimants"] = yearly_display["claimants"].apply(fmt_count)
yearly_display["total_visits"] = yearly_display["total_visits"].apply(fmt_count)
yearly_display["total_claims"] = yearly_display["total_claims"].apply(fmt_count)
yearly_display["total_allowed_amt"] = yearly_display["total_allowed_amt"].apply(fmt_millions)

display(yearly_display.rename(columns={
    "year": "Year",
    "enrolled_members": "Enrolled Members",
    "claimants": "Claimants",
    "total_visits": "Total Visits",
    "total_claims": "Total Claims",
    "total_allowed_amt": "Total Allowed Amt"
}).reset_index(drop=True))

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

axes[0].bar(yearly["year"].astype(str), yearly["enrolled_members"],
            color="#4C9BE8", alpha=0.85, label="Enrolled")
axes[0].bar(yearly["year"].astype(str), yearly["claimants"],
            color="#F4845F", alpha=0.85, label="Claimants")
axes[0].set_title("Enrolled vs Claimants by Year", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Members", fontsize=9)
axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

axes[1].bar(yearly["year"].astype(str), yearly["total_claims"],
            color="#5DBE7E", alpha=0.85)
axes[1].set_title("Total Claims by Year", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Claims", fontsize=9)
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

ax2 = axes[2].twinx()
axes[2].bar(yearly["year"].astype(str), yearly["total_allowed_amt"] / 1_000_000,
            color="#4C9BE8", alpha=0.7, label="Total Spend (USD M)")
ax2.plot(yearly["year"].astype(str),
         yearly["total_allowed_amt"] / yearly["claimants"].replace(0, np.nan),
         color="#F4845F", marker="o", linewidth=2, label="Avg Per Claimant ($)")
axes[2].set_title("Spend by Year", fontsize=11, fontweight="bold")
axes[2].set_ylabel("Total Spend (USD M)", fontsize=9)
ax2.set_ylabel("Avg Spend Per Claimant ($)", fontsize=9)
axes[2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_usd(x)))
lines1, labels1 = axes[2].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=8)
axes[2].grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Year-by-Year Volume and Spend — South Florida", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("yearly_overview.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 2.3 DATA QUALITY COVERAGE ─────────────────────────────────────────────────
display(Markdown("""
### 2.3 Data Quality Coverage

Percentage of claims with valid ICD-10 codes, specialty codes, and CCSR mappings.
These rates define the analytical ceiling — transitions can only be computed
for claims where all three are present.

Any rate below 90% warrants investigation before proceeding with analysis.
"""))

quality = client.query(f"""
WITH base AS (
    SELECT
        COUNT(*)                                         AS total_claims
        ,COUNTIF(pri_icd9_dx_cd IS NOT NULL
            AND TRIM(pri_icd9_dx_cd) != '')              AS claims_with_icd10
        ,COUNTIF(specialty_ctg_cd IS NOT NULL
            AND TRIM(specialty_ctg_cd) != '')            AS claims_with_specialty
    FROM {RAW}
),
ccsr AS (
    SELECT COUNTIF(ccsr_category IS NOT NULL
        AND ccsr_category != '')                         AS claims_with_ccsr
    FROM {VISITS}
)
SELECT b.*, c.claims_with_ccsr
FROM base b, ccsr c
""").to_dataframe()

tc = quality["total_claims"].iloc[0]
display(Markdown(f"""
| Metric | Count | Coverage |
|---|---|---|
| Total claims | {fmt_count(tc)} | 100% |
| Claims with valid ICD-10 | {fmt_count(quality['claims_with_icd10'].iloc[0])} | {quality['claims_with_icd10'].iloc[0]/tc*100:.1f}% |
| Claims with specialty code | {fmt_count(quality['claims_with_specialty'].iloc[0])} | {quality['claims_with_specialty'].iloc[0]/tc*100:.1f}% |
| Claims with CCSR mapping | {fmt_count(quality['claims_with_ccsr'].iloc[0])} | {quality['claims_with_ccsr'].iloc[0]/tc*100:.1f}% |
"""))

fig, ax = plt.subplots(figsize=(10, 4))
metrics = ["ICD-10", "Specialty", "CCSR"]
rates = [
    quality["claims_with_icd10"].iloc[0] / tc * 100,
    quality["claims_with_specialty"].iloc[0] / tc * 100,
    quality["claims_with_ccsr"].iloc[0] / tc * 100
]
colors = ["#5DBE7E" if r >= 90 else "#F4845F" for r in rates]
bars = ax.barh(metrics, rates, color=colors, alpha=0.85)
ax.axvline(90, color="red", linestyle="--", alpha=0.6, label="90% threshold")
for bar, rate in zip(bars, rates):
    ax.text(bar.get_width() - 1, bar.get_y() + bar.get_height() / 2,
            f"{rate:.1f}%", va="center", ha="right", fontsize=10,
            color="white", fontweight="bold")
ax.set_xlabel("Coverage %", fontsize=10)
ax.set_title("Data Quality Coverage", fontsize=12, fontweight="bold")
ax.set_xlim(0, 105)
ax.legend(fontsize=9)
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("data_quality.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 3 — VISIT CHARACTERISTICS
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Chapter 3 — Visit Characteristics

Understanding how members use the healthcare system — visit frequency,
time between visits, specialty mix, and spend distribution.

These patterns directly inform modeling decisions:
- Visit distribution informs sequence length for HSTU
- Time between visits informs delta-t bucketing
- Specialty mix shows what the data is dominated by
- Spend distribution identifies outlier pathways

---
"""))


# ── 3.1 VISIT DISTRIBUTION PER MEMBER ────────────────────────────────────────
display(Markdown("""
### 3.1 Visit Distribution Per Member

How many visits does the average member have across the full 2022-2025 period?

This determines whether the last 20 visits cap in the model is appropriate —
if most members have fewer than 20 visits, the cap is irrelevant.
If many members have far more than 20, the cap is actively truncating history.

Percentile markers show where most of the population sits.
"""))

visit_dist = client.query(f"""
SELECT
    member_id
    ,COUNT(DISTINCT visit_date)                          AS visit_count
FROM {VISITS}
GROUP BY member_id
""").to_dataframe()

visit_dist["visit_count"] = visit_dist["visit_count"].astype(float)
pcts = visit_dist["visit_count"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

display(Markdown(f"""
| Percentile | Visits Per Member |
|---|---|
| 10th | {pcts['10%']:.0f} |
| 25th | {pcts['25%']:.0f} |
| Median (50th) | {pcts['50%']:.0f} |
| 75th | {pcts['75%']:.0f} |
| 90th | {pcts['90%']:.0f} |
| 95th | {pcts['95%']:.0f} |
| 99th | {pcts['99%']:.0f} |
| Mean | {pcts['mean']:.1f} |
"""))

fig, ax = plt.subplots(figsize=(14, 6))
cap = visit_dist["visit_count"].quantile(0.99)
plot_data = visit_dist[visit_dist["visit_count"] <= cap]["visit_count"]
ax.hist(plot_data, bins=50, color="#4C9BE8", alpha=0.8, edgecolor="white")
for pct, val, color in [
    ("p25", pcts["25%"], "#5DBE7E"),
    ("p50", pcts["50%"], "orange"),
    ("p75", pcts["75%"], "#F4845F"),
    ("p95", pcts["95%"], "red")
]:
    ax.axvline(val, color=color, linestyle="--", alpha=0.8, label=f"{pct}: {val:.0f}")
ax.axvline(20, color="black", linestyle="-", linewidth=2, alpha=0.6, label="Model cap: 20")
ax.set_xlabel("Visits Per Member", fontsize=10)
ax.set_ylabel("Number of Members", fontsize=10)
ax.set_title("Distribution of Visits Per Member (capped at 99th percentile)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("visit_dist_per_member.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 3.2 TIME BETWEEN VISITS ───────────────────────────────────────────────────
display(Markdown("""
### 3.2 Time Between Consecutive Visits

How many days elapse between a member's consecutive visits?

This directly informs the delta-t bucketing strategy for HSTU.
A distribution skewed toward short gaps suggests members visit frequently.
A wide distribution with long tails suggests episodic care patterns.

Short gaps (same-day or next-day visits) may represent multi-specialty visits
or hospital stays rather than independent clinical encounters.
"""))

delta_t = client.query(f"""
WITH ranked AS (
    SELECT
        member_id
        ,visit_date
        ,LAG(visit_date) OVER (
            PARTITION BY member_id ORDER BY visit_date
        )                                                AS prior_visit_date
    FROM (SELECT DISTINCT member_id, visit_date FROM {VISITS})
)
SELECT
    DATE_DIFF(visit_date, prior_visit_date, DAY)         AS days_between
FROM ranked
WHERE prior_visit_date IS NOT NULL
  AND DATE_DIFF(visit_date, prior_visit_date, DAY) > 0
""").to_dataframe()

delta_t["days_between"] = delta_t["days_between"].astype(float)
dpcts = delta_t["days_between"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

display(Markdown(f"""
| Percentile | Days Between Visits |
|---|---|
| 10th | {dpcts['10%']:.0f} |
| 25th | {dpcts['25%']:.0f} |
| Median (50th) | {dpcts['50%']:.0f} |
| 75th | {dpcts['75%']:.0f} |
| 90th | {dpcts['90%']:.0f} |
| 95th | {dpcts['95%']:.0f} |
| Mean | {dpcts['mean']:.1f} |
"""))

fig, ax = plt.subplots(figsize=(14, 6))
cap = delta_t["days_between"].quantile(0.95)
plot_data = delta_t[delta_t["days_between"] <= cap]["days_between"]
ax.hist(plot_data, bins=60, color="#5DBE7E", alpha=0.8, edgecolor="white")
for pct, val, color in [
    ("p25", dpcts["25%"], "#4C9BE8"),
    ("p50", dpcts["50%"], "orange"),
    ("p75", dpcts["75%"], "#F4845F"),
    ("T30", 30, "red"),
]:
    ax.axvline(val, color=color, linestyle="--", alpha=0.8,
               label=f"{'p50' if pct == 'p50' else pct}: {val:.0f} days")
ax.set_xlabel("Days Between Consecutive Visits", fontsize=10)
ax.set_ylabel("Frequency", fontsize=10)
ax.set_title("Distribution of Time Between Visits (capped at 95th percentile)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("delta_t_dist.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 3.3 TOP SPECIALTIES ───────────────────────────────────────────────────────
display(Markdown("""
### 3.3 Top Specialties by Volume and Spend

Which specialties dominate the claims data?

This shows whether FP/I (Family Practice and Internal Medicine) represent
a meaningful share of visits — validating the FP-first analytical lens.
It also reveals which specialties carry the most spend weight.

Total spend shown in millions USD. Per-visit cost shown in USD.
"""))

spec_vol = client.query(f"""
SELECT
    specialty_ctg_cd
    ,specialty_desc
    ,COUNT(DISTINCT CONCAT(member_id, '_', CAST(visit_date AS STRING))) AS visit_count
    ,COUNT(DISTINCT member_id)                           AS unique_members
    ,ROUND(SUM(allowed_amt), 2)                          AS total_allowed_amt
    ,ROUND(SUM(allowed_amt) /
        NULLIF(COUNT(DISTINCT CONCAT(member_id, '_',
            CAST(visit_date AS STRING))), 0), 2)         AS avg_per_visit
FROM {VISITS}
WHERE specialty_ctg_cd IS NOT NULL
GROUP BY specialty_ctg_cd, specialty_desc
ORDER BY visit_count DESC
LIMIT 20
""").to_dataframe()

spec_display = spec_vol.copy()
spec_display["visit_count"] = spec_display["visit_count"].apply(fmt_count)
spec_display["unique_members"] = spec_display["unique_members"].apply(fmt_count)
spec_display["total_allowed_amt"] = spec_display["total_allowed_amt"].apply(fmt_millions)
spec_display["avg_per_visit"] = spec_display["avg_per_visit"].apply(fmt_usd)

display(spec_display[[
    "specialty_desc", "visit_count", "unique_members",
    "total_allowed_amt", "avg_per_visit"
]].rename(columns={
    "specialty_desc": "Specialty",
    "visit_count": "Total Visits",
    "unique_members": "Unique Members",
    "total_allowed_amt": "Total Spend (USD M)",
    "avg_per_visit": "Avg Per Visit ($)"
}).reset_index(drop=True))

fig, axes = plt.subplots(1, 2, figsize=(24, 10))
top15_vol = spec_vol.head(15).sort_values("visit_count", ascending=True)
axes[0].barh(top15_vol["specialty_desc"], top15_vol["visit_count"],
             color="#4C9BE8", alpha=0.85)
axes[0].set_xlabel("Total Visits", fontsize=9)
axes[0].set_title("Top 15 Specialties by
