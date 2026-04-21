# Medicare Network Adequacy & Capacity Modeling
## Project Approach — Stakeholder Summary

---

## What Are We Trying to Answer?

> **Does Aetna's Medicare Advantage provider network have enough doctors and facilities, in the right places, to serve our members?**

For every county in Florida, we want to know:
- Are there enough providers of each specialty type?
- Can members actually reach those providers within a reasonable distance?
- Which counties are at risk of failing federal compliance standards?

---

## Why Does This Matter?

The Centers for Medicare & Medicaid Services (CMS) sets strict rules about how close providers must be to Medicare Advantage members. If Aetna's network does not meet these standards, the plan faces regulatory penalties, contract sanctions, and reputational risk.

This project builds an analytic model to identify gaps **before** CMS audits them.

---

## How We Did It — In Plain Steps

| Step | What We Did | Why |
|------|-------------|-----|
| 1 | Collected the CMS rulebook (42 CFR 422.116) | To know exactly what the compliance thresholds are for every specialty and every county type |
| 2 | Classified every Florida county into one of 5 types | CMS has different rules for dense cities vs rural areas. A cardiologist must be within 10 miles in Miami but 60 miles in a rural county |
| 3 | Pulled Medicare enrollment data by county | To know how many Medicare members live in each area — this determines how many providers are required |
| 4 | Mapped Aetna's provider specialties to CMS specialty categories | Aetna and CMS use different specialty names. We matched them up so we're comparing apples to apples |
| 5 | Located every contracted provider on a map using their zip code | So we can measure actual distances between members and providers |
| 6 | Measured distances between members and providers | For every member zip code, we found all contracted providers within the CMS-allowed distance for each specialty |
| 7 | Rolled up results to county level | CMS evaluates compliance at the county level |
| 8 | Flagged counties as Compliant or Non-Compliant | Based on two CMS tests: enough providers, and enough members within reach |

---

## The Two Compliance Tests CMS Requires

| Test | What It Checks | Threshold |
|------|----------------|-----------|
| Access | % of members who have at least 1 provider within the maximum allowed distance | 90% in cities, 85% in rural areas |
| Count | Actual number of contracted providers vs minimum required per 1,000 members | Varies by specialty and county type |

Both tests must pass for a county to be compliant.

---

## What the Output Looks Like

For every combination of **County × Specialty × Plan Type**, we produce:

| Output | Description |
|--------|-------------|
| % Members with Access | What share of members can reach at least 1 provider |
| Required Provider Count | How many providers CMS requires |
| Actual Provider Count | How many Aetna has contracted |
| Gap | How many providers are missing |
| Compliance Status | COMPLIANT or NON-COMPLIANT |

---

## Scope of This Analysis

| Item | Scope |
|------|-------|
| Geography | Florida (67 counties) |
| Plan Types | MA-HMO, MA-PPO |
| Specialties | 43 CMS-defined specialty and facility types |
| Data Snapshot | Most recent available month |

---

## Known Limitations

| Limitation | Impact |
|------------|--------|
| Some Aetna specialties do not map exactly to CMS categories | Compliance scores for those specialties may be slightly inflated |
| Population data is from 2018 Census | May not reflect current member distribution in fast-growing Florida counties |
| 26 Florida counties have no Aetna providers at all | These counties are automatically non-compliant |
| Distance measured zip code to zip code, not exact address | Introduces small margin of error, especially in large rural counties |
