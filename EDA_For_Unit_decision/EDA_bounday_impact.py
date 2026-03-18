# ============================================================
# NB_02 — Boundary Rules
# Purpose : Define and justify left and right boundary rules
#           Visualize each test case as a member timeline
#           Real member examples to be plugged in later
# ============================================================
from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from IPython.display import display, Markdown
from datetime import date, timedelta

client = bigquery.Client(project="anbc-hcb-dev")
DATASET = "anbc-hcb-dev.provider_ds_netconf_data_hcb_dev"

DATASET_START = date(2022, 1, 1)
DATASET_END   = date(2025, 12, 31)


display(Markdown("""
---
# NB 02 — Boundary Rules
## Defining Valid Triggers for Analysis

Not all first encounters are analytically trustworthy.
This notebook defines the rules that determine which first encounters
qualify for the analysis — and shows exactly what each rule does
using member timeline visualizations.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 4 — DEFINING THE FIRST ENCOUNTER
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Chapter 4 — Defining the First Encounter

### Visit Definition

A visit is defined as a unique combination of:
- Member
- Service date
- Provider specialty
- Diagnosis code

Multiple claim lines for the same combination on the same date are collapsed
into one visit. A single date can produce multiple visits if the member was
seen for different diagnoses or by different specialties.

### First Encounter of a Diagnosis

A first encounter is defined as the **first date a member presents with a
specific ICD-10 diagnosis code** — regardless of provider, specialty, or
visit history.

This first encounter is used as the **starting point** to identify the
characteristics of the next visit.

### Known Limitation

A first encounter of a diagnosis code is not necessarily the first time
a member has experienced the underlying condition. A member may have been
treated for the condition at a different insurer, or before the dataset window.
This is a known trade-off — acceptable for prediction purposes but important
for clinical interpretation.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 5 — LEFT BOUNDARY RULES
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Chapter 5 — Left Boundary Rules

Left boundary rules determine whether a trigger has enough prior history
to be considered a reliable first encounter.

Without these rules, a member who joined the plan in 2024 and immediately
presents with Diabetes would be counted as a first encounter — even though
they may have been treated for Diabetes for years at a different insurer.

### Rule 1 — Enrolled at Least 12 Months Before Trigger

**Condition:** `enrollment_start <= trigger_date - 365 days`

**Rationale:**
The member must have been enrolled for at least 12 months before the trigger.
This ensures we have a full year of claims history to check whether the
diagnosis truly is new. A member enrolled for only 1 month before the trigger
gives us almost no history to validate against.

**What happens without it:**
Members who recently joined the plan with pre-existing conditions would be
incorrectly flagged as first encounters — introducing noise into the analysis.

### Rule 2 — Trigger Diagnosis Not Seen in Prior 12 Months

**Condition:** Trigger DX code does not appear in any claim in the 12 months
before the trigger date.

**Rationale:**
Even if a member is enrolled, they may have had the same diagnosis code in a
prior visit within the past year — meaning this is a follow-up or recurrence,
not a new encounter.

**What happens without it:**
Follow-up visits for ongoing conditions would be treated as first encounters —
significantly inflating trigger counts and polluting the analysis.

### Informational Flag — has_claims_12m_before

Not enforced as a filter. Records whether the member had any claims in the
12 months before the trigger.

Members with no prior claims can still qualify — they may be healthy members
who rarely seek care. However this flag is used to exclude these members from
sequence model training since there is no input sequence to learn from.

---
"""))


# ── TIMELINE DRAWING FUNCTION ─────────────────────────────────────────────────
def draw_timeline(ax, case_num, title,
                  enrollment_start, enrollment_end,
                  claims, trigger_date, trigger_dx,
                  dx_claims,
                  rule1_pass, rule2_pass,
                  verdict, verdict_reason,
                  show_lookback=True,
                  t30=None, t60=None, t180=None):

    ds = date(2022, 1, 1)
    de = date(2025, 12, 31)
    total_days = (de - ds).days

    def to_x(d):
        return (d - ds).days / total_days

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # enrollment bar
    x0 = to_x(max(enrollment_start, ds))
    x1 = to_x(min(enrollment_end, de))
    ax.barh(0.5, x1 - x0, left=x0, height=0.12,
            color="#CCCCCC", alpha=0.8, align="center")

    # lookback window
    if show_lookback and trigger_date:
        lb_start = trigger_date - timedelta(days=365)
        lx0 = to_x(max(lb_start, ds))
        lx1 = to_x(trigger_date)
        ax.barh(0.5, lx1 - lx0, left=lx0, height=0.12,
                color="#4C9BE8", alpha=0.25, align="center")
        ax.text((lx0 + lx1) / 2, 0.65, "12M Lookback",
                ha="center", va="bottom", fontsize=6, color="#4C9BE8")

    # T windows
    if trigger_date:
        if t30:
            wx1 = to_x(min(trigger_date + timedelta(days=30), de))
            ax.barh(0.5, wx1 - to_x(trigger_date), left=to_x(trigger_date),
                    height=0.08, color="#5DBE7E", alpha=0.3, align="center")
            ax.text(wx1, 0.42, "T30", ha="center", va="top", fontsize=5, color="#5DBE7E")
        if t60:
            wx1 = to_x(min(trigger_date + timedelta(days=60), de))
            ax.barh(0.5, wx1 - to_x(trigger_date), left=to_x(trigger_date),
                    height=0.06, color="#F7C948", alpha=0.3, align="center")
            ax.text(wx1, 0.38, "T60", ha="center", va="top", fontsize=5, color="#F7C948")
        if t180:
            wx1 = to_x(min(trigger_date + timedelta(days=180), de))
            ax.barh(0.5, wx1 - to_x(trigger_date), left=to_x(trigger_date),
                    height=0.04, color="#F4845F", alpha=0.3, align="center")
            ax.text(wx1, 0.35, "T180", ha="center", va="top", fontsize=5, color="#F4845F")

    # enrollment end marker
    ex = to_x(min(enrollment_end, de))
    ax.axvline(ex, color="#AAAAAA", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.text(ex, 0.78, "Enroll\nEnd", ha="center", va="bottom",
            fontsize=5, color="#888888")

    # claims dots
    for c in claims:
        if ds <= c <= de:
            ax.plot(to_x(c), 0.5, "o", color="#4C9BE8", markersize=5, alpha=0.8)

    # dx claims (rule 2 relevant)
    for c in dx_claims:
        if ds <= c <= de:
            ax.plot(to_x(c), 0.5, "o", color="#F4845F", markersize=6,
                    alpha=0.9, zorder=5)

    # trigger marker
    if trigger_date and ds <= trigger_date <= de:
        tx = to_x(trigger_date)
        ax.annotate("", xy=(tx, 0.58), xytext=(tx, 0.72),
                    arrowprops=dict(arrowstyle="->", color="darkorange", lw=2))
        ax.text(tx, 0.74, "Trigger", ha="center", va="bottom",
                fontsize=6, color="darkorange", fontweight="bold")

    # year markers
    for y in [2022, 2023, 2024, 2025]:
        yx = to_x(date(y, 1, 1))
        ax.axvline(yx, color="#DDDDDD", linewidth=0.8)
        ax.text(yx, 0.08, str(y), ha="center", va="bottom", fontsize=6, color="#999999")

    # rules
    r1 = "✓" if rule1_pass else "✗"
    r2 = "✓" if rule2_pass else "✗"
    r1c = "#5DBE7E" if rule1_pass else "#F4845F"
    r2c = "#5DBE7E" if rule2_pass else "#F4845F"
    ax.text(0.01, 0.22, f"R1: {r1}", transform=ax.transAxes,
            fontsize=7, color=r1c, fontweight="bold")
    ax.text(0.01, 0.14, f"R2: {r2}", transform=ax.transAxes,
            fontsize=7, color=r2c, fontweight="bold")

    # verdict box
    vc = "#5DBE7E" if verdict == "Valid" else "#F4845F" if verdict == "Invalid" else "#F7C948"
    ax.text(0.99, 0.18, verdict, transform=ax.transAxes,
            fontsize=8, color=vc, fontweight="bold", ha="right")
    ax.text(0.99, 0.10, verdict_reason, transform=ax.transAxes,
            fontsize=5.5, color="#555555", ha="right", wrap=True)

    ax.set_title(f"Case {case_num}: {title}", fontsize=8, fontweight="bold", pad=2)


# ── LEFT BOUNDARY TEST CASES ──────────────────────────────────────────────────
left_cases = [
    dict(case_num=1,  title="Enrolled 2022, trigger 2024, DX new",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,3,15), date(2022,8,10), date(2023,4,20), date(2023,10,5)],
         trigger_date=date(2024,2,15), trigger_dx="E11.9",
         dx_claims=[],
         rule1_pass=True, rule2_pass=True,
         verdict="Valid", verdict_reason="Full 12M lookback, DX confirmed new"),

    dict(case_num=2,  title="Enrolled 2022, trigger Mar 2022",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,2,10)],
         trigger_date=date(2022,3,15), trigger_dx="E11.9",
         dx_claims=[],
         rule1_pass=False, rule2_pass=False,
         verdict="Invalid", verdict_reason="Only 1M enrolled before trigger"),

    dict(case_num=3,  title="Enrolled 2024, trigger 2024",
         enrollment_start=date(2024,1,15), enrollment_end=date(2025,12,31),
         claims=[],
         trigger_date=date(2024,6,10), trigger_dx="I10",
         dx_claims=[],
         rule1_pass=False, rule2_pass=False,
         verdict="Invalid", verdict_reason="Less than 12M enrolled before trigger"),

    dict(case_num=4,  title="Enrolled 2022, trigger 2023, DX new",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,3,1), date(2022,9,15)],
         trigger_date=date(2023,5,10), trigger_dx="I10",
         dx_claims=[],
         rule1_pass=True, rule2_pass=True,
         verdict="Valid", verdict_reason="12M lookback satisfied, DX confirmed new"),

    dict(case_num=5,  title="Enrolled 2022, trigger Feb 2022",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,1,20)],
         trigger_date=date(2022,2,10), trigger_dx="J45",
         dx_claims=[],
         rule1_pass=False, rule2_pass=False,
         verdict="Invalid", verdict_reason="Only 1M history before trigger"),

    dict(case_num=6,  title="Enrolled 2022, gap in 2023, trigger 2024",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,3,15), date(2022,8,10)],
         trigger_date=date(2024,2,15), trigger_dx="E11.9",
         dx_claims=[],
         rule1_pass=True, rule2_pass=True,
         verdict="Valid", verdict_reason="Rule 1 passes, DX new — gap is informational only"),

    dict(case_num=7,  title="DX seen in lookback window",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,3,15), date(2022,8,10), date(2023,4,20)],
         trigger_date=date(2024,6,10), trigger_dx="E11.9",
         dx_claims=[date(2022,8,10)],
         rule1_pass=True, rule2_pass=False,
         verdict="Invalid", verdict_reason="DX E11.9 seen Aug 2022 — not a first encounter"),

    dict(case_num=8,  title="Earliest valid trigger Jan 2023",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,3,15), date(2022,9,20)],
         trigger_date=date(2023,1,15), trigger_dx="F32",
         dx_claims=[],
         rule1_pass=True, rule2_pass=True,
         verdict="Valid", verdict_reason="Jan 2023 earliest valid — full 2022 lookback"),

    dict(case_num=9,  title="First claim IS the trigger",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[],
         trigger_date=date(2022,1,15), trigger_dx="I10",
         dx_claims=[],
         rule1_pass=False, rule2_pass=False,
         verdict="Invalid", verdict_reason="No claims before trigger — cannot confirm DX new"),

    dict(case_num=10, title="Enrolled Jun 2023, trigger Jan 2024",
         enrollment_start=date(2023,6,1),  enrollment_end=date(2025,12,31),
         claims=[date(2023,8,10), date(2023,11,20)],
         trigger_date=date(2024,1,15), trigger_dx="E11.9",
         dx_claims=[],
         rule1_pass=False, rule2_pass=True,
         verdict="Invalid", verdict_reason="Enrolled Jun 2023 — less than 12M before Jan 2024"),

    dict(case_num=11, title="Enrolled 2022, trigger 2025, DX new",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,3,15), date(2023,9,10), date(2024,3,20), date(2024,9,15)],
         trigger_date=date(2025,10,10), trigger_dx="N18",
         dx_claims=[],
         rule1_pass=True, rule2_pass=True,
         verdict="Valid", verdict_reason="3 years history, DX confirmed new in 2025"),

    dict(case_num=12, title="Enrolled 2022, first claim 2023, trigger 2024",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2023,4,15), date(2023,10,20)],
         trigger_date=date(2024,11,10), trigger_dx="M54",
         dx_claims=[],
         rule1_pass=True, rule2_pass=True,
         verdict="Valid", verdict_reason="12M lookback 2023, DX confirmed new"),
]

display(Markdown("""
### Left Boundary — 12 Test Cases

Each timeline shows:
- **Grey bar** — enrollment period
- **Blue shaded** — 12-month lookback window before trigger
- **Blue dots** — regular claims
- **Red dots** — claims with trigger DX code (Rule 2 relevant)
- **Orange arrow** — trigger date
- **R1 / R2** — rule pass (green ✓) or fail (red ✗)
- **Verdict** — Valid / Invalid / Partial
"""))

fig, axes = plt.subplots(4, 3, figsize=(22, 20))
axes = axes.flatten()

for i, case in enumerate(left_cases):
    draw_timeline(axes[i], **case, show_lookback=True)

fig.suptitle("Left Boundary Test Cases — Member Timelines",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("left_boundary_cases.png", dpi=150, bbox_inches="tight")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 6 — RIGHT BOUNDARY RULES
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Chapter 6 — Right Boundary Rules

Right boundary rules ensure that after a valid trigger, we have sufficient
follow-up data to observe what happens next.

Without these rules, a trigger in November 2025 would be included in T180
analysis — but only 45 days of follow-up data exist in the dataset.
The T180 label would be based on incomplete information.

### Rule 1 — Trigger Within Dataset Window Per Time Window

**Conditions:**
- T30: `trigger_date <= 2025-11-30`
- T60: `trigger_date <= 2025-10-31`
- T180: `trigger_date <= 2025-06-30`

**Rationale:**
Ensures the full follow-up window exists within the dataset.
A trigger too close to the dataset end cannot have complete follow-up.

### Rule 2 — Member Enrolled Through Follow-up Window

**Conditions:**
- T30: `enrollment_end >= trigger_date + 30 days`
- T60: `enrollment_end >= trigger_date + 60 days`
- T180: `enrollment_end >= trigger_date + 180 days`

**Rationale:**
Even if the dataset extends far enough, a member who disenrolled shortly
after the trigger will have no claims in our data for the follow-up period.
Their follow-up is unobservable even though they may have sought care elsewhere.

### Partial = Invalid

A trigger that qualifies for T30 but not T60 is treated as invalid for T60
analysis. There is no partial credit. Each window is independently qualified.

---
"""))

right_cases = [
    dict(case_num=1,  title="Jan 2024 trigger, enrolled Dec 2025",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2023,6,10), date(2024,2,20), date(2024,8,15)],
         trigger_date=date(2024,1,15), trigger_dx="E11.9",
         dx_claims=[],
         rule1_pass=True, rule2_pass=True,
         verdict="Valid", verdict_reason="Full T180 available",
         show_lookback=False, t30=True, t60=True, t180=True),

    dict(case_num=2,  title="Jun 2025 trigger, exactly at T180",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,6,10)],
         trigger_date=date(2025,6,30), trigger_dx="I10",
         dx_claims=[],
         rule1_pass=True, rule2_pass=True,
         verdict="Valid", verdict_reason="Exactly at T180 boundary",
         show_lookback=False, t30=True, t60=True, t180=True),

    dict(case_num=3,  title="Aug 2025 trigger, T180 exceeds dataset",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,8,10)],
         trigger_date=date(2025,8,15), trigger_dx="J45",
         dx_claims=[],
         rule1_pass=True, rule2_pass=False,
         verdict="Partial", verdict_reason="T30 T60 valid, T180 exceeds Dec 2025",
         show_lookback=False, t30=True, t60=True, t180=False),

    dict(case_num=4,  title="Oct 2025 trigger, T30 T60 only",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,10,10)],
         trigger_date=date(2025,10,15), trigger_dx="F32",
         dx_claims=[],
         rule1_pass=True, rule2_pass=False,
         verdict="Partial", verdict_reason="T30 T60 only — T180 exceeds dataset",
         show_lookback=False, t30=True, t60=True, t180=False),

    dict(case_num=5,  title="Nov 2025 trigger, T30 only",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,11,10)],
         trigger_date=date(2025,11,15), trigger_dx="N18",
         dx_claims=[],
         rule1_pass=True, rule2_pass=False,
         verdict="Partial", verdict_reason="T30 only — T60 T180 exceed dataset",
         show_lookback=False, t30=True, t60=False, t180=False),

    dict(case_num=6,  title="Dec 2025 trigger, no window valid",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,12,1)],
         trigger_date=date(2025,12,15), trigger_dx="M54",
         dx_claims=[],
         rule1_pass=False, rule2_pass=False,
         verdict="Invalid", verdict_reason="At dataset end — no follow-up window",
         show_lookback=False, t30=False, t60=False, t180=False),

    dict(case_num=7,  title="Jan 2024 trigger, disenrolls Feb 2024",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2024,2,28),
         claims=[date(2023,6,10), date(2024,1,20)],
         trigger_date=date(2024,1,15), trigger_dx="E11.9",
         dx_claims=[],
         rule1_pass=True, rule2_pass=False,
         verdict="Partial", verdict_reason="Disenrolls Feb 2024 — T30 only",
         show_lookback=False, t30=True, t60=False, t180=False),

    dict(case_num=8,  title="Jan 2024 trigger, disenrolls Aug 2024",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2024,8,31),
         claims=[date(2023,6,10), date(2024,1,20)],
         trigger_date=date(2024,1,15), trigger_dx="I10",
         dx_claims=[],
         rule1_pass=True, rule2_pass=True,
         verdict="Valid", verdict_reason="Enrollment covers full T180",
         show_lookback=False, t30=True, t60=True, t180=True),

    dict(case_num=9,  title="Jan 2024 trigger, disenrolls Jan 2024",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2024,1,15),
         claims=[date(2023,6,10)],
         trigger_date=date(2024,1,15), trigger_dx="J45",
         dx_claims=[],
         rule1_pass=False, rule2_pass=False,
         verdict="Invalid", verdict_reason="Disenrolls at trigger date — no follow-up",
         show_lookback=False, t30=False, t60=False, t180=False),

    dict(case_num=10, title="Mar 2025 trigger, enrolled Dec 2025",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,3,10)],
         trigger_date=date(2025,3,15), trigger_dx="F32",
         dx_claims=[],
         rule1_pass=True, rule2_pass=True,
         verdict="Valid", verdict_reason="Full T180 within dataset and enrollment",
         show_lookback=False, t30=True, t60=True, t180=True),

    dict(case_num=11, title="Jul 2025 trigger, T180 exceeds dataset",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,7,10)],
         trigger_date=date(2025,7,15), trigger_dx="N18",
         dx_claims=[],
         rule1_pass=True, rule2_pass=False,
         verdict="Partial", verdict_reason="T30 T60 valid — T180 exceeds Dec 2025",
         show_lookback=False, t30=True, t60=True, t180=False),

    dict(case_num=12, title="Jun 2025 trigger, disenrolls Sep 2025",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,9,30),
         claims=[date(2024,6,10)],
         trigger_date=date(2025,6,30), trigger_dx="M54",
         dx_claims=[],
         rule1_pass=True, rule2_pass=False,
         verdict="Partial", verdict_reason="T30 T60 valid — enrollment ends Sep 2025",
         show_lookback=False, t30=True, t60=True, t180=False),
]

display(Markdown("""
### Right Boundary — 12 Test Cases

Each timeline shows:
- **Grey bar** — enrollment period
- **Grey dotted line** — enrollment end date
- **Green shaded** — T30 follow-up window
- **Yellow shaded** — T60 follow-up window
- **Orange shaded** — T180 follow-up window
- **Orange arrow** — trigger date
- **R1** — dataset window check pass or fail
- **R2** — enrollment coverage check pass or fail
- **Verdict** — Valid / Invalid / Partial (Partial = invalid for that window)
"""))

fig, axes = plt.subplots(4, 3, figsize=(22, 20))
axes = axes.flatten()

for i, case in enumerate(right_cases):
    draw_timeline(axes[i], **case)

fig.suptitle("Right Boundary Test Cases — Member Timelines",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("right_boundary_cases.png", dpi=150, bbox_inches="tight")
plt.show()

display(Markdown("""
---
## Chapter 6 Summary

Right boundary rules ensure follow-up data completeness.

Key takeaways:
- T180 is the most restrictive — triggers must occur before June 2025
- T30 is the most permissive — triggers can occur through November 2025
- Member disenrollment can further restrict window eligibility
- Partial qualification is treated as invalid — no half windows

The quantitative impact of these rules on the total trigger population
is documented in NB 03 — Boundary Impact.

---
"""))
