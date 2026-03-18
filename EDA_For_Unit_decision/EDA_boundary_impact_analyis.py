# ============================================================
# NB_03 — Boundary Impact Analysis
# Purpose : Quantify the impact of each boundary rule on the
#           trigger population. Justify all data decisions
#           with evidence from the data itself.
# Sources : A870800_gen_rec_triggers_qualified
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
TQ = f"`{DATASET}.A870800_gen_rec_triggers_qualified`"
V  = f"`{DATASET}.A870800_gen_rec_visits`"

def fmt_millions(x):
    return f"${x/1_000_000:.2f}M" if x >= 1_000_000 else f"${x:,.0f}"

def fmt_count(x):
    return f"{int(x):,}"

def fmt_pct(x):
    return f"{x:.1f}%"

display(Markdown("""
---
# NB 03 — Boundary Impact Analysis
## How Data Decisions Shape the Analytical Population

This notebook quantifies the impact of every data decision made in this
analysis — from how a first encounter is defined, to how each boundary
rule filters the trigger population.

Every number here comes directly from the data. No thresholds are assumed.
The purpose is to give full transparency into what the analysis is built on
and what was excluded — and why.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FIRST ENCOUNTER DEFINITION AND JUSTIFICATION
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 1 — First Encounter Definition and Justification

### What is a First Encounter in This Analysis?

A first encounter is defined as the **earliest date a member presents with
a specific ICD-10 diagnosis code** in the claims data.

In code:
```sql
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY member_id, dx_clean
    ORDER BY visit_date
) = 1
```

This means:
- One trigger per member per diagnosis code — the earliest date only
- All subsequent appearances of the same diagnosis for the same member
  are excluded as triggers — regardless of how much time has passed
- This is an absolute first occurrence definition — not a recurrence definition

### Why Recurrence is Excluded

The analytical goal of this project is to predict **which provider specialty
a member visits next after a new diagnosis is made**.

A recurrence — the same diagnosis reappearing after a gap — does not represent
a new routing decision. It represents a continuation of an established care
pathway. The member already has a known specialist relationship for that
condition. The next visit after a recurrence is most likely a return to a
known provider — not a new clinical routing event.

Including recurrences would answer a different question:
*Given an ongoing condition, where does the member return?*

That is not the question this analysis is designed to answer.

### Data Evidence Supporting This Decision

The following counts were computed directly from the data to validate
this decision before finalizing the approach.
"""))

# recurrence validation query
recurrence = client.query(f"""
WITH first_occ AS (
    SELECT COUNT(*) AS first_occurrence_triggers
        ,COUNT(DISTINCT member_id) AS first_occ_members
        ,COUNT(DISTINCT trigger_dx_clean) AS first_occ_dx_codes
    FROM {TQ}
    WHERE is_left_qualified = TRUE
),
recurrence_occ AS (
    SELECT COUNT(*) AS recurrence_triggers
        ,COUNT(DISTINCT member_id) AS recurrence_members
        ,COUNT(DISTINCT dx_clean) AS recurrence_dx_codes
    FROM (
        SELECT
            member_id
            ,dx_clean
            ,visit_date
            ,LAG(visit_date) OVER (
                PARTITION BY member_id, dx_clean
                ORDER BY visit_date
            ) AS prior_dx_date
        FROM {V}
    )
    WHERE prior_dx_date IS NOT NULL
      AND DATE_DIFF(visit_date, prior_dx_date, DAY) > 365
)
SELECT * FROM first_occ, recurrence_occ
""").to_dataframe()

r = recurrence.iloc[0]
pct_triggers = r["recurrence_triggers"] / r["first_occurrence_triggers"] * 100
pct_members  = r["recurrence_members"]  / r["first_occ_members"]  * 100
pct_dx       = r["recurrence_dx_codes"] / r["first_occ_dx_codes"] * 100

display(Markdown(f"""
| Metric | First Occurrence | Recurrence (>12M gap) | Recurrence as % of First Occurrence |
|---|---|---|---|
| Triggers | {fmt_count(r['first_occurrence_triggers'])} | {fmt_count(r['recurrence_triggers'])} | {fmt_pct(pct_triggers)} |
| Unique Members | {fmt_count(r['first_occ_members'])} | {fmt_count(r['recurrence_members'])} | {fmt_pct(pct_members)} |
| Unique DX Codes | {fmt_count(r['first_occ_dx_codes'])} | {fmt_count(r['recurrence_dx_codes'])} | {fmt_pct(pct_dx)} |
"""))

display(Markdown(f"""
**Interpretation:**

Recurrence logic would add {fmt_pct(pct_triggers)} more triggers to the dataset.
The members who have recurrences represent {fmt_pct(pct_members)} of the first occurrence
member population — meaning the majority of recurrence members already have a
first occurrence trigger in the dataset.

The recurrence DX codes represent {fmt_pct(pct_dx)} of first occurrence DX codes —
recurrences are concentrated in a subset of diagnoses, predominantly chronic
conditions where established care relationships are strongest and new routing
signal is weakest.

**Decision: First occurrence only.**
"""))

# top recurring dx codes
top_recur_dx = client.query(f"""
SELECT
    dx_clean
    ,MAX(COALESCE(d.icd9_dx_dscrptn, dx_clean))         AS dx_desc
    ,COUNT(*) AS recurrence_count
    ,COUNT(DISTINCT member_id) AS unique_members
FROM (
    SELECT
        member_id
        ,dx_clean
        ,visit_date
        ,LAG(visit_date) OVER (
            PARTITION BY member_id, dx_clean
            ORDER BY visit_date
        ) AS prior_dx_date
    FROM {V}
)
LEFT JOIN `edp-prod-hcbstorage.edp_hcb_core_cnsv.ICD9_DIAGNOSIS` d
    ON dx_clean = d.icd9_dx_cd
WHERE prior_dx_date IS NOT NULL
  AND DATE_DIFF(visit_date, prior_dx_date, DAY) > 365
GROUP BY dx_clean
ORDER BY recurrence_count DESC
LIMIT 15
""").to_dataframe()

display(Markdown("""
#### Top 15 Diagnosis Codes by Recurrence Volume

These are the diagnoses most frequently seen as recurrences after a 12-month gap.
As expected, chronic conditions dominate — confirming that recurrences represent
continuation of established care rather than new routing events.
"""))

display(top_recur_dx[["dx_desc", "recurrence_count", "unique_members"]].rename(columns={
    "dx_desc": "Diagnosis",
    "recurrence_count": "Recurrence Count",
    "unique_members": "Unique Members"
}).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(14, 7))
plot = top_recur_dx.sort_values("recurrence_count", ascending=True)
ax.barh(plot["dx_desc"].str[:50], plot["recurrence_count"],
        color="#4C9BE8", alpha=0.85)
ax.set_xlabel("Recurrence Count", fontsize=10)
ax.set_title("Top 15 Diagnoses by Recurrence Volume\n(same DX reappearing after 12-month gap)",
             fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
plt.setp(ax.get_yticklabels(), fontsize=8)
plt.tight_layout()
plt.savefig("top_recurrence_dx.png", dpi=150, bbox_inches="tight")
plt.show()

display(Markdown("""
### Rule 2 — A Note on Implementation

Rule 2 checks whether the trigger diagnosis appeared in any claim in the
12 months before the trigger date. As implemented in the code:

```sql
CASE
    WHEN EXISTS (
        SELECT 1 FROM dx_history d
        WHERE d.member_id = f.member_id
          AND d.dx_clean = f.trigger_dx_clean
          AND d.visit_date >= DATE_SUB(f.trigger_date, INTERVAL 365 DAY)
          AND d.visit_date < f.trigger_date
    ) THEN FALSE ELSE TRUE
END AS rule2_dx_not_seen_12m
```

Because `first_encounters` already selects the absolute earliest occurrence
of each diagnosis per member, Rule 2 will always pass — there can be no
prior occurrence of the same DX within 12 months of the first occurrence.

Rule 2 is retained in the code as:
- Documentation of analytical intent
- A safeguard in case the trigger source changes
- An informational flag for audit purposes

It does not filter any triggers in the current implementation.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LEFT BOUNDARY IMPACT
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 2 — Left Boundary Impact

### What the Left Boundary Does

The left boundary rules determine whether a first encounter has enough
prior claims history to be considered a reliable trigger.

Two rules apply:
- **Rule 1:** Member enrolled at least 12 months before trigger date
- **Rule 2:** Trigger DX not seen in 12 months before trigger (see Section 1)

Both rules must pass for a trigger to be left-qualified.
"""))

left_impact = client.query(f"""
SELECT
    COUNT(*)                                             AS total_first_encounters
    ,COUNTIF(rule1_enrolled_12m = TRUE)                  AS after_rule1
    ,COUNTIF(rule2_dx_not_seen_12m = TRUE)               AS after_rule2
    ,COUNTIF(is_left_qualified = TRUE)                   AS left_qualified
    ,COUNT(DISTINCT member_id)                           AS total_members
    ,COUNT(DISTINCT CASE WHEN rule1_enrolled_12m = TRUE
        THEN member_id END)                              AS members_after_rule1
    ,COUNT(DISTINCT CASE WHEN is_left_qualified = TRUE
        THEN member_id END)                              AS members_left_qualified
FROM {TQ}
""").to_dataframe()

l = left_impact.iloc[0]

display(Markdown(f"""
| Stage | Triggers | Members | % Triggers Retained |
|---|---|---|---|
| All first encounters | {fmt_count(l['total_first_encounters'])} | {fmt_count(l['total_members'])} | 100% |
| After Rule 1 (enrolled 12M) | {fmt_count(l['after_rule1'])} | {fmt_count(l['members_after_rule1'])} | {fmt_pct(l['after_rule1']/l['total_first_encounters']*100)} |
| After Rule 2 (DX not seen 12M) | {fmt_count(l['after_rule2'])} | — | {fmt_pct(l['after_rule2']/l['total_first_encounters']*100)} |
| Left qualified (both rules) | {fmt_count(l['left_qualified'])} | {fmt_count(l['members_left_qualified'])} | {fmt_pct(l['left_qualified']/l['total_first_encounters']*100)} |

**Triggers excluded by Rule 1:** {fmt_count(l['total_first_encounters'] - l['after_rule1'])} ({fmt_pct((l['total_first_encounters'] - l['after_rule1'])/l['total_first_encounters']*100)})
**Triggers excluded by Rule 2:** {fmt_count(l['total_first_encounters'] - l['after_rule2'])} ({fmt_pct((l['total_first_encounters'] - l['after_rule2'])/l['total_first_encounters']*100)})
"""))

# left impact by year
left_by_year = client.query(f"""
SELECT
    EXTRACT(YEAR FROM trigger_date)                      AS trigger_year
    ,COUNT(*)                                            AS total_first_encounters
    ,COUNTIF(rule1_enrolled_12m = TRUE)                  AS after_rule1
    ,COUNTIF(is_left_qualified = TRUE)                   AS left_qualified
    ,COUNT(DISTINCT member_id)                           AS total_members
    ,COUNT(DISTINCT CASE WHEN is_left_qualified = TRUE
        THEN member_id END)                              AS qualified_members
FROM {TQ}
GROUP BY trigger_year
ORDER BY trigger_year
""").to_dataframe()

display(Markdown("""
#### Left Boundary Impact by Year

2022 is expected to have low qualification rates — triggers in 2022 require
enrollment starting before January 2021 which is before the dataset window.
"""))

left_by_year["pct_qualified"] = (
    left_by_year["left_qualified"] / left_by_year["total_first_encounters"] * 100
).round(1)

display(left_by_year.rename(columns={
    "trigger_year": "Year",
    "total_first_encounters": "Total Triggers",
    "after_rule1": "After Rule 1",
    "left_qualified": "Left Qualified",
    "total_members": "Total Members",
    "qualified_members": "Qualified Members",
    "pct_qualified": "% Qualified"
}).reset_index(drop=True))

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

x = left_by_year["trigger_year"].astype(str)
axes[0].bar(x, left_by_year["total_first_encounters"],
            color="#CCCCCC", alpha=0.9, label="Total First Encounters")
axes[0].bar(x, left_by_year["left_qualified"],
            color="#4C9BE8", alpha=0.85, label="Left Qualified")
axes[0].set_title("Trigger Volume by Year\nTotal vs Left Qualified",
                  fontsize=11, fontweight="bold")
axes[0].set_ylabel("Trigger Count", fontsize=9)
axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", linestyle="--", alpha=0.4)

axes[1].bar(x, left_by_year["pct_qualified"], color="#5DBE7E", alpha=0.85)
axes[1].set_title("Qualification Rate by Year\n(Left Qualified / Total First Encounters)",
                  fontsize=11, fontweight="bold")
axes[1].set_ylabel("% Qualified", fontsize=9)
axes[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

fig.suptitle("Left Boundary Impact by Year", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("left_boundary_by_year.png", dpi=150, bbox_inches="tight")
plt.show()

# left impact by member segment
left_by_segment = client.query(f"""
SELECT
    member_segment
    ,COUNT(*)                                            AS total_first_encounters
    ,COUNTIF(is_left_qualified = TRUE)                   AS left_qualified
    ,COUNT(DISTINCT member_id)                           AS total_members
    ,COUNT(DISTINCT CASE WHEN is_left_qualified = TRUE
        THEN member_id END)                              AS qualified_members
FROM {TQ}
GROUP BY member_segment
ORDER BY total_first_encounters DESC
""").to_dataframe()

left_by_segment["pct_qualified"] = (
    left_by_segment["left_qualified"] / left_by_segment["total_first_encounters"] * 100
).round(1)

display(Markdown("#### Left Boundary Impact by Member Segment"))
display(left_by_segment.rename(columns={
    "member_segment": "Segment",
    "total_first_encounters": "Total Triggers",
    "left_qualified": "Left Qualified",
    "total_members": "Total Members",
    "qualified_members": "Qualified Members",
    "pct_qualified": "% Qualified"
}).reset_index(drop=True))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RIGHT BOUNDARY IMPACT
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 3 — Right Boundary Impact

### What the Right Boundary Does

The right boundary rules ensure sufficient follow-up data exists after
a qualified trigger to observe where the member goes next.

Two conditions must hold per window:
- The trigger date must fall within the dataset cutoff for that window
- The member must remain enrolled through the full follow-up window

**Cutoff dates:**
- T30: trigger on or before November 30 2025, enrollment covers +30 days
- T60: trigger on or before October 31 2025, enrollment covers +60 days
- T180: trigger on or before June 30 2025, enrollment covers +180 days

**Partial = Invalid.** A trigger valid for T30 but not T60 is excluded
from T60 analysis entirely.
"""))

right_impact = client.query(f"""
SELECT
    COUNTIF(is_left_qualified = TRUE)                    AS left_qualified
    ,COUNTIF(is_t30_qualified = TRUE)                    AS t30_qualified
    ,COUNTIF(is_t60_qualified = TRUE)                    AS t60_qualified
    ,COUNTIF(is_t180_qualified = TRUE)                   AS t180_qualified
    ,COUNT(DISTINCT CASE WHEN is_t30_qualified = TRUE
        THEN member_id END)                              AS t30_members
    ,COUNT(DISTINCT CASE WHEN is_t60_qualified = TRUE
        THEN member_id END)                              AS t60_members
    ,COUNT(DISTINCT CASE WHEN is_t180_qualified = TRUE
        THEN member_id END)                              AS t180_members
    -- lost to dataset end only (enrollment is fine but dataset ends)
    ,COUNTIF(is_left_qualified = TRUE
        AND trigger_date > DATE '2025-06-30'
        AND enrollment_end >= DATE_ADD(trigger_date, INTERVAL 180 DAY))
                                                         AS lost_to_dataset_end_t180
    -- lost to enrollment end only (dataset is fine but member disenrolled)
    ,COUNTIF(is_left_qualified = TRUE
        AND trigger_date <= DATE '2025-06-30'
        AND enrollment_end < DATE_ADD(trigger_date, INTERVAL 180 DAY))
                                                         AS lost_to_enrollment_end_t180
FROM {TQ}
""").to_dataframe()

ri = right_impact.iloc[0]
lq = ri["left_qualified"]

display(Markdown(f"""
| Stage | Triggers | Members | % of Left Qualified |
|---|---|---|---|
| Left qualified | {fmt_count(lq)} | — | 100% |
| T30 qualified | {fmt_count(ri['t30_qualified'])} | {fmt_count(ri['t30_members'])} | {fmt_pct(ri['t30_qualified']/lq*100)} |
| T60 qualified | {fmt_count(ri['t60_qualified'])} | {fmt_count(ri['t60_members'])} | {fmt_pct(ri['t60_qualified']/lq*100)} |
| T180 qualified | {fmt_count(ri['t180_qualified'])} | {fmt_count(ri['t180_members'])} | {fmt_pct(ri['t180_qualified']/lq*100)} |

**T180 triggers lost to dataset end (enrollment OK, data ends):** {fmt_count(ri['lost_to_dataset_end_t180'])}
**T180 triggers lost to enrollment end (data OK, member disenrolled):** {fmt_count(ri['lost_to_enrollment_end_t180'])}
"""))

# right impact by year
right_by_year = client.query(f"""
SELECT
    EXTRACT(YEAR FROM trigger_date)                      AS trigger_year
    ,COUNTIF(is_left_qualified = TRUE)                   AS left_qualified
    ,COUNTIF(is_t30_qualified = TRUE)                    AS t30_qualified
    ,COUNTIF(is_t60_qualified = TRUE)                    AS t60_qualified
    ,COUNTIF(is_t180_qualified = TRUE)                   AS t180_qualified
FROM {TQ}
GROUP BY trigger_year
ORDER BY trigger_year
""").to_dataframe()

display(Markdown("""
#### Right Boundary Impact by Trigger Year

Triggers from 2025 are most affected by right boundary rules — 
especially T180 which requires the trigger to occur before June 2025.
"""))

display(right_by_year.rename(columns={
    "trigger_year": "Year",
    "left_qualified": "Left Qualified",
    "t30_qualified": "T30 Qualified",
    "t60_qualified": "T60 Qualified",
    "t180_qualified": "T180 Qualified"
}).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(right_by_year))
width = 0.2
ax.bar(x - width*1.5, right_by_year["left_qualified"],  width, label="Left Qualified",  color="#CCCCCC", alpha=0.9)
ax.bar(x - width*0.5, right_by_year["t30_qualified"],   width, label="T30 Qualified",   color="#5DBE7E", alpha=0.85)
ax.bar(x + width*0.5, right_by_year["t60_qualified"],   width, label="T60 Qualified",   color="#F7C948", alpha=0.85)
ax.bar(x + width*1.5, right_by_year["t180_qualified"],  width, label="T180 Qualified",  color="#F4845F", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(right_by_year["trigger_year"].astype(str))
ax.set_ylabel("Trigger Count", fontsize=10)
ax.set_title("Right Boundary Impact by Trigger Year",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("right_boundary_by_year.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MEMBER IMPACT
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 4 — Member Impact

How many members have at least one qualified trigger per window?
How many members have no qualified triggers at all?
How many qualified triggers does the average member have?
"""))

member_impact = client.query(f"""
WITH member_counts AS (
    SELECT
        member_id
        ,COUNT(*) AS total_triggers
        ,COUNTIF(is_left_qualified = TRUE) AS left_qualified
        ,COUNTIF(is_t30_qualified = TRUE) AS t30_qualified
        ,COUNTIF(is_t60_qualified = TRUE) AS t60_qualified
        ,COUNTIF(is_t180_qualified = TRUE) AS t180_qualified
    FROM {TQ}
    GROUP BY member_id
)
SELECT
    COUNT(*) AS total_members
    ,COUNTIF(left_qualified > 0) AS members_with_left_qualified
    ,COUNTIF(t30_qualified > 0) AS members_with_t30
    ,COUNTIF(t60_qualified > 0) AS members_with_t60
    ,COUNTIF(t180_qualified > 0) AS members_with_t180
    ,COUNTIF(left_qualified = 0) AS members_no_qualified_triggers
    ,ROUND(AVG(left_qualified), 2) AS avg_left_qualified_per_member
    ,ROUND(AVG(t180_qualified), 2) AS avg_t180_qualified_per_member
    ,MAX(left_qualified) AS max_left_qualified_per_member
FROM member_counts
""").to_dataframe()

mi = member_impact.iloc[0]

display(Markdown(f"""
| Metric | Count | % of Total Members |
|---|---|---|
| Total members with any trigger | {fmt_count(mi['total_members'])} | 100% |
| Members with left qualified trigger | {fmt_count(mi['members_with_left_qualified'])} | {fmt_pct(mi['members_with_left_qualified']/mi['total_members']*100)} |
| Members with T30 qualified trigger | {fmt_count(mi['members_with_t30'])} | {fmt_pct(mi['members_with_t30']/mi['total_members']*100)} |
| Members with T60 qualified trigger | {fmt_count(mi['members_with_t60'])} | {fmt_pct(mi['members_with_t60']/mi['total_members']*100)} |
| Members with T180 qualified trigger | {fmt_count(mi['members_with_t180'])} | {fmt_pct(mi['members_with_t180']/mi['total_members']*100)} |
| Members with no qualified triggers | {fmt_count(mi['members_no_qualified_triggers'])} | {fmt_pct(mi['members_no_qualified_triggers']/mi['total_members']*100)} |

**Average left qualified triggers per member:** {mi['avg_left_qualified_per_member']}
**Average T180 qualified triggers per member:** {mi['avg_t180_qualified_per_member']}
**Maximum left qualified triggers for one member:** {fmt_count(mi['max_left_qualified_per_member'])}
"""))

# distribution of qualified triggers per member
trigger_dist = client.query(f"""
SELECT
    left_qualified_count
    ,COUNT(*) AS member_count
FROM (
    SELECT
        member_id
        ,COUNTIF(is_left_qualified = TRUE) AS left_qualified_count
    FROM {TQ}
    GROUP BY member_id
)
GROUP BY left_qualified_count
ORDER BY left_qualified_count
""").to_dataframe()

display(Markdown("""
#### Distribution of Qualified Triggers Per Member

Shows how many qualified triggers each member has.
Members with many qualified triggers have a rich history of new diagnoses.
"""))

fig, ax = plt.subplots(figsize=(14, 6))
cap = trigger_dist["left_qualified_count"].quantile(0.95)
plot_data = trigger_dist[trigger_dist["left_qualified_count"] <= cap]
ax.bar(plot_data["left_qualified_count"], plot_data["member_count"],
       color="#4C9BE8", alpha=0.85, width=0.8)
ax.set_xlabel("Number of Left Qualified Triggers Per Member", fontsize=10)
ax.set_ylabel("Number of Members", fontsize=10)
ax.set_title("Distribution of Qualified Triggers Per Member\n(capped at 95th percentile)",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("trigger_dist_per_member.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HAS CLAIMS 12M BEFORE FLAG
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 5 — has_claims_12m_before Flag

This flag identifies qualified triggers where the member had no claims
in the 12 months before the trigger date.

These members are valid for EDA and Markov analysis — their trigger is
genuine and their right boundary is satisfied.

However they are excluded from sequence model training because the model
requires a non-empty visit sequence as input. A member with no prior claims
in the lookback window has no input sequence — the model has nothing to
learn from.

This section shows how many triggers are affected by this flag and the
practical impact on the model training population.
"""))

claims_flag = client.query(f"""
SELECT
    has_claims_12m_before
    ,COUNT(*) AS trigger_count
    ,COUNT(DISTINCT member_id) AS unique_members
    ,COUNTIF(is_t30_qualified = TRUE) AS t30_qualified
    ,COUNTIF(is_t60_qualified = TRUE) AS t60_qualified
    ,COUNTIF(is_t180_qualified = TRUE) AS t180_qualified
FROM {TQ}
WHERE is_left_qualified = TRUE
GROUP BY has_claims_12m_before
""").to_dataframe()

display(claims_flag.rename(columns={
    "has_claims_12m_before": "Has Claims 12M Before",
    "trigger_count": "Trigger Count",
    "unique_members": "Unique Members",
    "t30_qualified": "T30 Qualified",
    "t60_qualified": "T60 Qualified",
    "t180_qualified": "T180 Qualified"
}).reset_index(drop=True))

total_lq = claims_flag["trigger_count"].sum()
no_claims = claims_flag[claims_flag["has_claims_12m_before"] == False]["trigger_count"].sum() \
    if False in claims_flag["has_claims_12m_before"].values else 0

display(Markdown(f"""
**Triggers with no prior claims (excluded from sequence model):**
{fmt_count(no_claims)} ({fmt_pct(no_claims/total_lq*100)} of left qualified triggers)

**Triggers with prior claims (eligible for sequence model):**
{fmt_count(total_lq - no_claims)} ({fmt_pct((total_lq - no_claims)/total_lq*100)} of left qualified triggers)
"""))

fig, ax = plt.subplots(figsize=(10, 5))
labels = ["Has Claims 12M Before", "No Claims 12M Before"]
sizes  = [total_lq - no_claims, no_claims]
colors = ["#4C9BE8", "#F4845F"]
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                   autopct="%1.1f%%", startangle=90,
                                   textprops={"fontsize": 10})
ax.set_title("Left Qualified Triggers — Claims 12M Before Flag",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("claims_flag_pie.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FULL FUNNEL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Section 6 — Full Funnel Summary

The complete picture — from all first encounters to the final model-ready
population per time window.
"""))

# combine all counts into one funnel table
funnel = client.query(f"""
SELECT
    COUNT(*) AS all_first_encounters
    ,COUNTIF(rule1_enrolled_12m = TRUE) AS after_rule1
    ,COUNTIF(is_left_qualified = TRUE) AS left_qualified
    ,COUNTIF(is_t30_qualified = TRUE) AS t30_qualified
    ,COUNTIF(is_t60_qualified = TRUE) AS t60_qualified
    ,COUNTIF(is_t180_qualified = TRUE) AS t180_qualified
    ,COUNTIF(is_t180_qualified = TRUE
        AND has_claims_12m_before = TRUE) AS t180_model_ready
    ,COUNT(DISTINCT member_id) AS all_members
    ,COUNT(DISTINCT CASE WHEN is_left_qualified = TRUE
        THEN member_id END) AS left_qualified_members
    ,COUNT(DISTINCT CASE WHEN is_t180_qualified = TRUE
        THEN member_id END) AS t180_members
    ,COUNT(DISTINCT CASE WHEN is_t180_qualified = TRUE
        AND has_claims_12m_before = TRUE
        THEN member_id END) AS t180_model_ready_members
FROM {TQ}
""").to_dataframe()

f = funnel.iloc[0]
base = f["all_first_encounters"]

funnel_rows = [
    ["All first encounters",               fmt_count(f["all_first_encounters"]),  fmt_count(f["all_members"]),                "100%"],
    ["After Rule 1 (enrolled 12M)",        fmt_count(f["after_rule1"]),           "—",                                        fmt_pct(f["after_rule1"]/base*100)],
    ["Left qualified (Rule 1 + Rule 2)",   fmt_count(f["left_qualified"]),        fmt_count(f["left_qualified_members"]),     fmt_pct(f["left_qualified"]/base*100)],
    ["T30 qualified",                      fmt_count(f["t30_qualified"]),         "—",                                        fmt_pct(f["t30_qualified"]/base*100)],
    ["T60 qualified",                      fmt_count(f["t60_qualified"]),         "—",                                        fmt_pct(f["t60_qualified"]/base*100)],
    ["T180 qualified",                     fmt_count(f["t180_qualified"]),        fmt_count(f["t180_members"]),               fmt_pct(f["t180_qualified"]/base*100)],
    ["T180 + has claims 12M (model ready)",fmt_count(f["t180_model_ready"]),      fmt_count(f["t180_model_ready_members"]),   fmt_pct(f["t180_model_ready"]/base*100)],
]

funnel_df = pd.DataFrame(funnel_rows, columns=["Stage", "Triggers", "Members", "% of All First Encounters"])
display(funnel_df)

# funnel chart
fig, ax = plt.subplots(figsize=(14, 8))
stages = [r[0] for r in funnel_rows]
counts = [f["all_first_encounters"], f["after_rule1"], f["left_qualified"],
          f["t30_qualified"], f["t60_qualified"], f["t180_qualified"], f["t180_model_ready"]]
colors = ["#CCCCCC", "#A8D8EA", "#4C9BE8", "#5DBE7E", "#F7C948", "#F4845F", "#C0392B"]

bars = ax.barh(stages[::-1], counts[::-1], color=colors[::-1], alpha=0.85)
for bar, count in zip(bars, counts[::-1]):
    ax.text(bar.get_width() * 1.005, bar.get_y() + bar.get_height() / 2,
            fmt_count(count), va="center", fontsize=8, color="#333333")
ax.set_xlabel("Trigger Count", fontsize=10)
ax.set_title("Full Qualification Funnel — From First Encounters to Model Ready",
             fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_count(x)))
plt.setp(ax.get_yticklabels(), fontsize=8)
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("qualification_funnel.png", dpi=150, bbox_inches="tight")
plt.show()

display(Markdown("""
---
## Summary

This notebook documents every data decision that shapes the analytical population:

**First encounter definition:**
Absolute first occurrence of each diagnosis per member. Recurrences excluded
by design — they represent continuation of established care pathways, not
new routing events. Validated with data showing recurrences add limited
volume and concentrate in chronic conditions.

**Left boundary:**
Ensures triggers have sufficient claims history to be reliable first encounters.
Rule 1 (enrolled 12M) is the primary filter. Rule 2 is a no-op under current
logic but retained for documentation and auditability.

**Right boundary:**
Ensures follow-up data exists after each trigger. Three independent windows —
T30, T60, T180. Partial qualification is treated as invalid. Triggers are
lost to either dataset end or member disenrollment — both are quantified above.

**has_claims_12m_before flag:**
Valid triggers with no prior claims are included in EDA and Markov analysis
but excluded from sequence model training where an input sequence is required.

The qualification funnel chart above shows exactly how many triggers and
members survive each stage — from raw first encounters to the final
model-ready population.

---
"""))
