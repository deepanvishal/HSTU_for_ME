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

Each timeline is followed by a plain language explanation of what
the member's situation was and why the verdict is Valid or Invalid.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 4 — DEFINING THE FIRST ENCOUNTER
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Chapter 4 — Defining the First Encounter

### Visit Definition

A visit is defined as a unique combination of member, service date,
provider specialty, and diagnosis code. Multiple claim lines for the same
combination on the same date are collapsed into one visit.

### First Encounter of a Diagnosis

A first encounter is the **earliest date a member presents with a specific
ICD-10 diagnosis code** — regardless of provider or specialty.
One trigger per member per diagnosis code across the full dataset.

### Known Limitation

A first encounter in this dataset is not necessarily the first time a member
has experienced the condition. Prior treatment at a different insurer or
before the dataset window is not observable.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 5 — LEFT BOUNDARY RULES
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Chapter 5 — Left Boundary Rules

Left boundary rules determine whether a first encounter has enough prior
history to be considered a reliable trigger.

### Rule 1 — Enrolled at Least 12 Months Before Trigger

**Condition:** `enrollment_start <= trigger_date - 365 days`

**Rationale:**
The member must have been enrolled for at least 12 months before the trigger.
This ensures we have a full year of claims history to confirm the diagnosis
is genuinely new to our data. A member enrolled for only a few months before
the trigger gives us insufficient history to validate against.

**What happens without it:**
Members who recently joined the plan with pre-existing conditions would be
incorrectly flagged as first encounters — introducing noise into the analysis.

### Rule 2 — Trigger Diagnosis Not Seen in Prior 12 Months

Rule 2 — trigger diagnosis not seen in prior 12 months — is inherently
satisfied by the first encounter definition and requires no additional filtering.

---
"""))


# ── TIMELINE DRAWING FUNCTION ─────────────────────────────────────────────────
def draw_timeline(ax, case_num, title,
                  enrollment_start, enrollment_end,
                  claims, trigger_date,
                  rule1_pass,
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

    # rule 1 only
    r1 = "✓" if rule1_pass else "✗"
    r1c = "#5DBE7E" if rule1_pass else "#F4845F"
    ax.text(0.01, 0.18, f"R1: {r1}", transform=ax.transAxes,
            fontsize=7, color=r1c, fontweight="bold")

    # verdict
    vc = "#5DBE7E" if verdict == "Valid" else "#F4845F" if verdict == "Invalid" else "#F7C948"
    ax.text(0.99, 0.18, verdict, transform=ax.transAxes,
            fontsize=8, color=vc, fontweight="bold", ha="right")
    ax.text(0.99, 0.10, verdict_reason, transform=ax.transAxes,
            fontsize=5.5, color="#555555", ha="right")

    ax.set_title(f"Case {case_num}: {title}", fontsize=8, fontweight="bold", pad=4)


def add_legend(fig, mode="left"):
    handles = [
        mpatches.Patch(color="#CCCCCC", alpha=0.8, label="Enrollment period"),
        mlines.Line2D([], [], color="#4C9BE8", marker="o", linestyle="None",
                      markersize=6, label="Claim"),
        mlines.Line2D([], [], color="darkorange", marker=r"$\downarrow$",
                      linestyle="None", markersize=8, label="Trigger date"),
        mlines.Line2D([], [], color="#AAAAAA", linestyle=":",
                      linewidth=1.5, label="Enrollment end"),
    ]
    if mode == "left":
        handles.append(
            mpatches.Patch(color="#4C9BE8", alpha=0.25, label="12M lookback window")
        )
    else:
        handles += [
            mpatches.Patch(color="#5DBE7E", alpha=0.3, label="T30 follow-up window"),
            mpatches.Patch(color="#F7C948", alpha=0.3, label="T60 follow-up window"),
            mpatches.Patch(color="#F4845F", alpha=0.3, label="T180 follow-up window"),
        ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=8,
               bbox_to_anchor=(0.5, -0.02),
               frameon=True, framealpha=0.9)


# ── LEFT BOUNDARY TEST CASES ──────────────────────────────────────────────────
left_cases = [
    dict(case_num=1,  title="Enrolled 2022, trigger 2024, DX new",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,3,15), date(2022,8,10), date(2023,4,20), date(2023,10,5)],
         trigger_date=date(2024,2,15),
         rule1_pass=True,
         verdict="Valid", verdict_reason="Full 12M lookback, DX confirmed new"),

    dict(case_num=2,  title="Enrolled 2022, trigger Mar 2022",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,2,10)],
         trigger_date=date(2022,3,15),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="Only 2M enrolled before trigger"),

    dict(case_num=3,  title="Enrolled Jun 2023, trigger Jan 2024",
         enrollment_start=date(2023,6,1),  enrollment_end=date(2025,12,31),
         claims=[date(2023,8,10), date(2023,11,20)],
         trigger_date=date(2024,1,15),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="Enrolled Jun 2023 — 7M before trigger"),

    dict(case_num=4,  title="Enrolled 2022, trigger 2023, DX new",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,3,1), date(2022,9,15)],
         trigger_date=date(2023,5,10),
         rule1_pass=True,
         verdict="Valid", verdict_reason="12M lookback satisfied, DX confirmed new"),

    dict(case_num=5,  title="First claim IS the trigger",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[],
         trigger_date=date(2022,1,15),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="Enrolled same month as trigger"),

    dict(case_num=6,  title="Enrolled 2022, gap in 2023, trigger 2024",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,3,15), date(2022,8,10)],
         trigger_date=date(2024,2,15),
         rule1_pass=True,
         verdict="Valid", verdict_reason="Rule 1 passes — gap in claims is informational only"),

    dict(case_num=7,  title="Enrolled 2024, trigger 2024",
         enrollment_start=date(2024,1,15), enrollment_end=date(2025,12,31),
         claims=[],
         trigger_date=date(2024,6,10),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="Enrolled Jan 2024 — only 5M before trigger"),

    dict(case_num=8,  title="Earliest valid trigger Jan 2023",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,3,15), date(2022,9,20)],
         trigger_date=date(2023,1,15),
         rule1_pass=True,
         verdict="Valid", verdict_reason="Jan 2023 earliest valid — full 2022 lookback"),

    dict(case_num=9,  title="Enrolled 2022, trigger 2025, DX new",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2022,3,15), date(2023,9,10), date(2024,3,20), date(2024,9,15)],
         trigger_date=date(2025,10,10),
         rule1_pass=True,
         verdict="Valid", verdict_reason="3 years history, DX confirmed new in 2025"),
]

display(Markdown("""
### Left Boundary — 9 Test Cases

**How to read the timelines:**
- **Grey bar** — full enrollment period
- **Blue shaded region** — 12-month lookback window before trigger
- **Blue dots** — claims in member history
- **Orange arrow** — trigger date
- **Grey dotted line** — enrollment end date
- **R1 ✓/✗** — Rule 1 pass or fail
- **Verdict** — green Valid, red Invalid
"""))

fig, axes = plt.subplots(3, 3, figsize=(22, 18))
axes = axes.flatten()

for i, case in enumerate(left_cases):
    draw_timeline(axes[i], **case, show_lookback=True)

add_legend(fig, mode="left")
fig.suptitle("Left Boundary Test Cases — Member Timelines",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 0.99])
plt.savefig("left_boundary_cases.png", dpi=150, bbox_inches="tight")
plt.show()


display(Markdown("""
### Left Boundary — Case by Case Explanation

---

**Case 1 — Valid**
Member enrolled January 2022 with claims in 2022 and 2023.
Diabetes (E11.9) appears for the first time in February 2024.
Rule 1 passes — enrolled more than 12 months before trigger.
Clean, reliable first encounter.

---

**Case 2 — Invalid**
Member enrolled January 2022 but trigger occurs March 2022 — only 2 months
after enrollment. Rule 1 fails. Insufficient history to confirm the diagnosis
is new to the plan.

---

**Case 3 — Invalid**
Member enrolled June 2023 with claims in August and November 2023.
Trigger occurs January 2024 — only 7 months after enrollment.
Rule 1 fails. Member needs to reach June 2024 before any trigger qualifies.

---

**Case 4 — Valid**
Member enrolled January 2022 with claims in March and September 2022.
Hypertension (I10) first appears May 2023.
Rule 1 passes — 16 months enrolled before trigger. Clean first encounter.

---

**Case 5 — Invalid**
Member enrolled January 2022 but the very first claim is the trigger
in January 2022 — the same month as enrollment.
Rule 1 fails. No history exists before the trigger at all.

---

**Case 6 — Valid**
Member enrolled January 2022 with claims in 2022 but a gap through all of 2023.
Trigger occurs February 2024.
Rule 1 passes — enrolled more than 12 months before trigger.
The gap in claims is noted in the has_claims_12m_before flag but does not
disqualify the trigger. Absence of claims is not absence of enrollment.

---

**Case 7 — Invalid**
Member enrolled January 2024 with no prior claims.
Trigger occurs June 2024 — only 5 months after enrollment.
Rule 1 fails. This member would need to reach January 2025 before
any trigger can qualify.

---

**Case 8 — Valid**
Member enrolled January 2022 with claims in March and September 2022.
Depression (F32) first appears January 2023.
This is the earliest possible valid trigger for a member enrolled
January 2022 — exactly 12 months of history available.
Rule 1 passes. Clean first encounter at the left edge of the valid window.

---

**Case 9 — Valid**
Member enrolled January 2022 with consistent claims across all years.
CKD (N18) first appears October 2025.
Rule 1 passes easily — over 3 years of enrollment before trigger.
Strong, well-evidenced first encounter.

---
"""))


# ══════════════════════════════════════════════════════════════════════════════
# CHAPTER 6 — RIGHT BOUNDARY RULES
# ══════════════════════════════════════════════════════════════════════════════
display(Markdown("""
---
## Chapter 6 — Right Boundary Rules

Right boundary rules ensure that after a valid trigger, sufficient follow-up
data exists to observe where the member goes next.

### Rule 1 — Trigger Within Dataset Window Per Time Window

**Conditions:**
- T30: `trigger_date <= 2025-11-30`
- T60: `trigger_date <= 2025-10-31`
- T180: `trigger_date <= 2025-06-30`

**Rationale:**
Ensures the full follow-up window exists within the dataset end of December 2025.

### Rule 2 — Member Enrolled Through Follow-up Window

**Conditions:**
- T30: `enrollment_end >= trigger_date + 30 days`
- T60: `enrollment_end >= trigger_date + 60 days`
- T180: `enrollment_end >= trigger_date + 180 days`

**Rationale:**
A member who disenrolled shortly after the trigger has no observable claims
in our data for the follow-up period — even if the dataset continues.

### Partial = Invalid

A trigger valid for T30 but not T60 is excluded from T60 analysis entirely.
No partial credit. Each window is independently qualified.

---
"""))

right_cases = [
    dict(case_num=1,  title="Jan 2024 trigger, enrolled Dec 2025",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2023,6,10), date(2024,2,20)],
         trigger_date=date(2024,1,15),
         rule1_pass=True,
         verdict="Valid", verdict_reason="Full T180 available",
         show_lookback=False, t30=True, t60=True, t180=True),

    dict(case_num=2,  title="Jun 2025 trigger, exactly at T180",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,6,10)],
         trigger_date=date(2025,6,30),
         rule1_pass=True,
         verdict="Valid", verdict_reason="Exactly at T180 boundary",
         show_lookback=False, t30=True, t60=True, t180=True),

    dict(case_num=3,  title="Aug 2025 trigger, T180 exceeds dataset",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,8,10)],
         trigger_date=date(2025,8,15),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="T30 T60 only — T180 exceeds Dec 2025",
         show_lookback=False, t30=True, t60=True, t180=False),

    dict(case_num=4,  title="Nov 2025 trigger, T30 only",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,11,10)],
         trigger_date=date(2025,11,15),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="T30 only — T60 T180 exceed dataset",
         show_lookback=False, t30=True, t60=False, t180=False),

    dict(case_num=5,  title="Dec 2025 trigger, no window valid",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,12,1)],
         trigger_date=date(2025,12,15),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="At dataset end — no follow-up window",
         show_lookback=False, t30=False, t60=False, t180=False),

    dict(case_num=6,  title="Jan 2024 trigger, disenrolls Feb 2024",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2024,2,28),
         claims=[date(2023,6,10), date(2024,1,20)],
         trigger_date=date(2024,1,15),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="Disenrolls Feb 2024 — T30 only",
         show_lookback=False, t30=True, t60=False, t180=False),

    dict(case_num=7,  title="Jan 2024 trigger, disenrolls Aug 2024",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2024,8,31),
         claims=[date(2023,6,10), date(2024,1,20)],
         trigger_date=date(2024,1,15),
         rule1_pass=True,
         verdict="Valid", verdict_reason="Enrollment covers full T180",
         show_lookback=False, t30=True, t60=True, t180=True),

    dict(case_num=8,  title="Jan 2024 trigger, disenrolls Jan 2024",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2024,1,15),
         claims=[date(2023,6,10)],
         trigger_date=date(2024,1,15),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="Disenrolls at trigger date — no follow-up",
         show_lookback=False, t30=False, t60=False, t180=False),

    dict(case_num=9,  title="Mar 2025 trigger, enrolled Dec 2025",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,3,10)],
         trigger_date=date(2025,3,15),
         rule1_pass=True,
         verdict="Valid", verdict_reason="Full T180 within dataset and enrollment",
         show_lookback=False, t30=True, t60=True, t180=True),

    dict(case_num=10, title="Jul 2025 trigger, T180 exceeds dataset",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,7,10)],
         trigger_date=date(2025,7,15),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="T30 T60 valid — T180 exceeds Dec 2025",
         show_lookback=False, t30=True, t60=True, t180=False),

    dict(case_num=11, title="Jun 2025 trigger, disenrolls Sep 2025",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,9,30),
         claims=[date(2024,6,10)],
         trigger_date=date(2025,6,30),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="T30 T60 valid — enrollment ends Sep 2025",
         show_lookback=False, t30=True, t60=True, t180=False),

    dict(case_num=12, title="Oct 2025 trigger, T30 T60 only",
         enrollment_start=date(2022,1,1),  enrollment_end=date(2025,12,31),
         claims=[date(2024,10,10)],
         trigger_date=date(2025,10,15),
         rule1_pass=False,
         verdict="Invalid", verdict_reason="T30 T60 only — T180 exceeds dataset",
         show_lookback=False, t30=True, t60=True, t180=False),
]

display(Markdown("""
### Right Boundary — 12 Test Cases

**How to read the timelines:**
- **Grey bar** — full enrollment period
- **Grey dotted line** — enrollment end date
- **Green shaded** — T30 follow-up window
- **Yellow shaded** — T60 follow-up window
- **Orange shaded** — T180 follow-up window
- **Blue dots** — claims in member history
- **Orange arrow** — trigger date
- **R1 ✓/✗** — right boundary pass or fail
- **Verdict** — green Valid, red Invalid
"""))

fig, axes = plt.subplots(4, 3, figsize=(22, 22))
axes = axes.flatten()

for i, case in enumerate(right_cases):
    draw_timeline(axes[i], **case)

add_legend(fig, mode="right")
fig.suptitle("Right Boundary Test Cases — Member Timelines",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 0.99])
plt.savefig("right_boundary_cases.png", dpi=150, bbox_inches="tight")
plt.show()


display(Markdown("""
### Right Boundary — Case by Case Explanation

---

**Case 1 — Valid (all windows)**
Trigger January 2024. Member enrolled through December 2025.
24 months of follow-up available. All three windows fit within both
dataset and enrollment. Clean full qualification.

---

**Case 2 — Valid (all windows)**
Trigger June 30 2025 — the last date that qualifies for T180.
T180 window ends December 27 2025 — just within dataset.
Member enrolled through December 2025. All windows valid.
This is the right edge of the T180 eligible window.

---

**Case 3 — Invalid**
Trigger August 15 2025.
T30 and T60 windows fall within dataset but T180 would end February 2026
— beyond the December 2025 dataset end.
Partial qualification is treated as invalid. Excluded from all windows.

---

**Case 4 — Invalid**
Trigger November 15 2025.
Only T30 fits within dataset — T60 and T180 both exceed December 2025.
Partial qualification is treated as invalid. Excluded from all windows.

---

**Case 5 — Invalid**
Trigger December 15 2025 — at the dataset end.
No follow-up window fits within the dataset at all.
Excluded entirely.

---

**Case 6 — Invalid**
Trigger January 15 2024. Member disenrolls February 28 2024 — 45 days later.
Only T30 is covered by enrollment. T60 and T180 are not.
Partial qualification is treated as invalid. Excluded from all windows.

---

**Case 7 — Valid (all windows)**
Trigger January 15 2024. Member disenrolls August 31 2024 — 229 days later.
T180 window ends July 14 2024 — within the August 31 enrollment end.
All three windows covered by enrollment. Valid.

---

**Case 8 — Invalid**
Trigger January 15 2024. Member disenrolls January 15 2024 — the same day.
Enrollment ends at the trigger date. No follow-up observable at all.
Excluded entirely.

---

**Case 9 — Valid (all windows)**
Trigger March 15 2025. Member enrolled through December 2025.
T180 ends September 11 2025 — well within dataset and enrollment.
All windows valid.

---

**Case 10 — Invalid**
Trigger July 15 2025.
T30 and T60 fit within dataset but T180 would end January 2026.
Partial qualification treated as invalid. Excluded from all windows.

---

**Case 11 — Invalid**
Trigger June 30 2025. Member disenrolls September 30 2025.
T30 and T60 are covered by enrollment but T180 would end December 27 2025
— after the September 30 disenrollment date.
Partial qualification treated as invalid. Excluded from all windows.

---

**Case 12 — Invalid**
Trigger October 15 2025.
T30 and T60 fit within dataset but T180 would end April 2026.
Partial qualification treated as invalid. Excluded from all windows.

---
"""))

display(Markdown("""
---
## Summary — Boundary Rules

**Left boundary ensures:**
- Member was enrolled long enough to have meaningful claims history
- The trigger diagnosis is the absolute first occurrence — not a recurrence

**Right boundary ensures:**
- Enough follow-up data exists in the dataset after the trigger
- The member remained enrolled through the full follow-up window

**Partial qualification is treated as fully invalid per window.**
A trigger valid for T30 but not T60 is excluded from T60 analysis entirely.
This maintains clean, unambiguous population definitions per time window.

The quantitative impact of these rules on the total trigger population
is documented in NB 03 — Boundary Impact.

---
"""))
