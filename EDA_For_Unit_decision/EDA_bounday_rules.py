# ============================================================
# NB_02 — Boundary Rules
# Purpose : Define and justify left and right boundary rules
#           Visualize each test case as a member timeline
#           Explain each case in plain language
# ============================================================
from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

Each timeline is followed by a plain language explanation of what
the member's situation was and why the verdict is Valid, Invalid, or Partial.

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

    # dx claims
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

    # verdict
    vc = "#5DBE7E" if verdict == "Valid" else "#F4845F" if verdict == "Invalid" else "#F7C948"
    ax.text(0.99, 0.18, verdict, transform=ax.transAxes,
            fontsize=8, color=vc, fontweight="bold", ha="right")
    ax.text(0.99, 0.10, verdict_reason, transform=ax.transAxes,
            fontsize=5.5, color="#555555", ha="right")

    ax.set_title(f"Case {case_num}: {title}", fontsize=8, fontweight="bold", pad=4)


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

**How to read the timelines:**
- **Grey bar** — full enrollment period from membership records
- **Blue shaded region** — 12-month lookback window before trigger date
- **Blue dots** — regular claims in the member's history
- **Red dots** — claims where the trigger diagnosis code appears (Rule 2 relevant)
- **Orange arrow** — trigger date (first encounter of the diagnosis)
- **Grey dotted line** — enrollment end date
- **R1 ✓/✗** — Rule 1 pass or fail (enrolled 12 months before trigger)
- **R2 ✓/✗** — Rule 2 pass or fail (DX not seen in lookback window)
- **Verdict** — green Valid, red Invalid
"""))

fig, axes = plt.subplots(4, 3, figsize=(22, 22))
axes = axes.flatten()

for i, case in enumerate(left_cases):
    draw_timeline(axes[i], **case, show_lookback=True)

fig.suptitle("Left Boundary Test Cases — Member Timelines",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("left_boundary_cases.png", dpi=150, bbox_inches="tight")
plt.show()


display(Markdown("""
### Left Boundary — Case by Case Explanation

---

**Case 1 — Valid**
Member enrolled January 2022. Has claims in 2022 and 2023.
Diabetes (E11.9) appears for the first time in February 2024.
Rule 1 passes — enrolled more than 12 months before February 2024.
Rule 2 passes — E11.9 not seen anywhere in the 12 months before trigger.
This is a clean, reliable first encounter.

---

**Case 2 — Invalid**
Member enrolled January 2022 but the trigger occurs in March 2022 —
only 2 months after enrollment. Rule 1 fails immediately.
We have almost no claims history to confirm the diagnosis is new.
Any trigger in 2022 will fail Rule 1 because the dataset starts in January 2022
and 12 months of lookback are not available until January 2023.

---

**Case 3 — Invalid**
Member enrolled January 2024 and the trigger occurs June 2024.
Only 5 months of enrollment before the trigger. Rule 1 fails.
This member may have had prior diagnoses at a different insurer that we cannot see.

---

**Case 4 — Valid**
Member enrolled January 2022. Claims in March and September 2022.
Hypertension (I10) first appears in May 2023.
Rule 1 passes — enrolled 16 months before the trigger.
Rule 2 passes — I10 not seen in the prior 12 months.
Clean first encounter.

---

**Case 5 — Invalid**
Member enrolled January 2022 but the trigger is February 2022 — just one month later.
Rule 1 fails. The lookback window would need to extend to February 2021 —
before the dataset starts. We have no history to validate against.

---

**Case 6 — Valid**
Member enrolled January 2022 with claims in 2022 but a gap in 2023.
Trigger occurs February 2024.
Rule 1 passes — enrolled more than 12 months before trigger.
Rule 2 passes — E11.9 not seen in 2023 (which is the lookback window).
The gap in claims is noted in the `has_claims_12m_before` flag but does not
disqualify the trigger. The member simply had no claims that year.

---

**Case 7 — Invalid**
Member enrolled January 2022 with claims throughout.
Diabetes (E11.9) appears in August 2022 — shown as a red dot.
The trigger occurs June 2024.
Rule 1 passes — enrolled long before the trigger.
Rule 2 fails — E11.9 was seen in August 2022 which is within the 12 months
before the June 2024 trigger. This is a recurrence, not a first encounter.

---

**Case 8 — Valid**
Member enrolled January 2022 with claims in March and September 2022.
Depression (F32) first appears in January 2023.
This is the earliest possible valid trigger — January 2023 is exactly
12 months after the dataset start of January 2022.
Both rules pass. This is the left edge of the valid trigger window.

---

**Case 9 — Invalid**
Member enrolled January 2022 but has no prior claims.
The very first claim is the trigger in January 2022.
Rule 1 fails — enrolled the same month as the trigger.
There is no history at all to confirm the diagnosis is new.

---

**Case 10 — Invalid**
Member enrolled June 2023 with claims in August and November 2023.
Diabetes (E11.9) first appears January 2024.
Rule 2 passes — E11.9 not seen in prior claims.
Rule 1 fails — enrolled June 2023 which is only 7 months before January 2024.
We need at least 12 months of enrollment. This member would qualify
if their trigger occurred after June 2024.

---

**Case 11 — Valid**
Member enrolled January 2022 with consistent claims across all years.
CKD (N18) first appears October 2025.
Rule 1 passes easily — over 3 years of enrollment before trigger.
Rule 2 passes — N18 not seen in the prior 12 months.
Strong, well-evidenced first encounter.

---

**Case 12 — Valid**
Member enrolled January 2022 but first claim appears April 2023.
Back Pain (M54) first appears November 2024.
Rule 1 passes — enrolled January 2022 which is well before November 2023
(12 months before trigger).
Rule 2 passes — M54 not seen in the April to November 2023 lookback window.
The gap between enrollment and first claim is acceptable — enrollment date
is what matters for Rule 1, not first claim date.

---
"""))


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

**How to read the timelines:**
- **Grey bar** — full enrollment period from membership records
- **Grey dotted line** — enrollment end date
- **Green shaded region** — T30 follow-up window (30 days after trigger)
- **Yellow shaded region** — T60 follow-up window (60 days after trigger)
- **Orange shaded region** — T180 follow-up window (180 days after trigger)
- **Blue dots** — claims in member history
- **Orange arrow** — trigger date
- **R1 ✓/✗** — dataset window check pass or fail
- **R2 ✓/✗** — enrollment coverage check pass or fail
- **Verdict** — green Valid, red Invalid, yellow Partial (Partial = invalid for that window)
"""))

fig, axes = plt.subplots(4, 3, figsize=(22, 22))
axes = axes.flatten()

for i, case in enumerate(right_cases):
    draw_timeline(axes[i], **case)

fig.suptitle("Right Boundary Test Cases — Member Timelines",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("right_boundary_cases.png", dpi=150, bbox_inches="tight")
plt.show()


display(Markdown("""
### Right Boundary — Case by Case Explanation

---

**Case 1 — Valid (all windows)**
Member enrolled January 2022 with trigger in January 2024.
Dataset end is December 2025 — 24 months of follow-up available.
Member remains enrolled through December 2025.
T30, T60, and T180 all fit comfortably within both the dataset and enrollment period.

---

**Case 2 — Valid (all windows)**
Trigger occurs June 30 2025 — exactly the last date that qualifies for T180.
Adding 180 days to June 30 lands on December 27 2025 — within the dataset.
Member enrolled through December 2025. All three windows valid.
This is the right edge of the T180 eligible window.

---

**Case 3 — Partial (T30 and T60 only)**
Trigger occurs August 15 2025.
T30 window ends September 14 2025 — within dataset. Valid.
T60 window ends October 14 2025 — within dataset. Valid.
T180 window would end February 11 2026 — beyond December 2025 dataset end. Invalid.
Member is enrolled but the data does not exist to observe T180 follow-up.

---

**Case 4 — Partial (T30 and T60 only)**
Trigger October 15 2025.
T30 valid — ends November 14 2025.
T60 valid — ends December 14 2025.
T180 would end April 13 2026 — beyond dataset. Invalid.
Same pattern as Case 3 but the trigger is later so T60 is also closer to the boundary.

---

**Case 5 — Partial (T30 only)**
Trigger November 15 2025.
T30 valid — ends December 15 2025 — just within dataset.
T60 would end January 14 2026 — beyond dataset. Invalid.
T180 would end May 14 2026 — beyond dataset. Invalid.
Only the shortest window is observable.

---

**Case 6 — Invalid (no windows)**
Trigger December 15 2025 — at the dataset end.
T30 would end January 14 2026 — beyond dataset.
No follow-up window is observable at all.
This trigger is excluded from all analysis.

---

**Case 7 — Partial (T30 only)**
Trigger January 15 2024. Member disenrolls February 28 2024 — only 45 days later.
T30 valid — enrollment covers 30 days after trigger.
T60 invalid — member disenrolls before 60 days are up.
T180 invalid — member disenrolls well before 180 days.
After disenrollment there are no observable claims even though the dataset continues.

---

**Case 8 — Valid (all windows)**
Trigger January 15 2024. Member disenrolls August 31 2024 — 229 days later.
T30 valid — enrollment covers 30 days.
T60 valid — enrollment covers 60 days.
T180 valid — enrollment covers 180 days (July 14 2024 is within August 31 enrollment).
All windows observable within enrollment period.

---

**Case 9 — Invalid (no windows)**
Trigger January 15 2024. Member disenrolls January 15 2024 — the same day.
No follow-up window is observable.
The enrollment end coincides with the trigger date.
Excluded from all analysis.

---

**Case 10 — Valid (all windows)**
Trigger March 15 2025. Member enrolled through December 2025.
T180 window ends September 11 2025 — well within both dataset and enrollment.
All three windows valid. Clean trigger with full observable follow-up.

---

**Case 11 — Partial (T30 and T60 only)**
Trigger July 15 2025. Member enrolled through December 2025.
T30 valid — ends August 14 2025.
T60 valid — ends September 13 2025.
T180 would end January 11 2026 — beyond dataset end. Invalid.
Same as Case 3 and 4 — late 2025 triggers lose T180 eligibility.

---

**Case 12 — Partial (T30 and T60 only)**
Trigger June 30 2025. Member disenrolls September 30 2025 — 92 days later.
T30 valid — enrollment covers 30 days.
T60 valid — enrollment covers 60 days (August 29 is within September 30 enrollment).
T180 invalid — member disenrolls September 30 before the 180-day window closes December 27.
Enrollment ends before T180 is complete even though the dataset continues.

---
"""))

display(Markdown("""
---
## Summary — Boundary Rules

**Left boundary ensures:**
- Member was enrolled long enough to have meaningful claims history
- The trigger diagnosis is genuinely new — not a recurrence or ongoing condition

**Right boundary ensures:**
- Enough follow-up data exists in the dataset after the trigger
- The member remained enrolled long enough to generate observable follow-up claims

**Partial qualification is treated as fully invalid per window.**
A trigger valid for T30 but not T60 is excluded from T60 analysis entirely.
This maintains clean, unambiguous population definitions per time window.

The quantitative impact of these rules on the total trigger population
is documented in NB 03 — Boundary Impact.

---
"""))
