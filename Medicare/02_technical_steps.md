# Medicare Network Adequacy & Capacity Modeling
## Technical Steps

---

## Architecture Overview

```
CMS Regulation (422.116)
        ↓
Reference Tables (thresholds, ratios, county types)
        ↓
Staging Tables (beneficiaries, providers)
        ↓
fact_zip_access (distance matrix + access flag)
        ↓
fact_gap_analysis (county rollup + compliance)
        ↓
Output Tables (compliance summary, risk counties)
```

---

## Table Build Order

| Order | Table | Type | Purpose | Key Data Sources |
|-------|-------|------|---------|-----------------|
| 1 | `ref_specialty_crosswalk` | Reference | Maps CMS 422.116 specialties to Aetna specialty codes | Manually coded from 422.116 + Aetna Global Lookup |
| 2 | `ref_time_distance` | Reference | Max time + distance thresholds per specialty × county type | 42 CFR 422.116 Table 1 |
| 3 | `ref_min_ratio` | Reference | Min providers per 1,000 beneficiaries per specialty × county type | 42 CFR 422.116 Table 2 |
| 4 | `ref_county_classification` | Reference | Florida county type classification (Large Metro/Metro/Micro/Rural/CEAC) | `bigquery-public-data.geo_us_boundaries.counties` + `census_bureau_acs.county_2020_5yr` |
| 5 | `ref_zip_reference` | Reference | Florida zip code centroids, area, radius, county mapping | `geo_us_boundaries.zip_codes` + `census_bureau_acs.zip_codes_2018_5yr` + `geo_us_boundaries.counties` |
| 6 | `stg_beneficiaries` | Staging | Demand side — zip level population + county context | `ref_zip_reference` + `cms_medicare_penetration` |
| 7 | `ref_county_name_crosswalk` | Reference | Maps Aetna county names to Census county names + FIPS | Manually coded from comparison of Aetna vs Census county lists |
| 8 | `stg_providers` | Staging | Supply side — Aetna contracted providers with specialty + location | `A870800_medicare_supply_demand_mbr_with_zip` + `ref_specialty_crosswalk` + `ref_county_name_crosswalk` + `ref_zip_reference` |
| 9 | `fact_zip_access` | Fact | For each bene zip × specialty × plan type: count providers within threshold, flag has_access | `stg_beneficiaries` + `stg_providers` + `ref_zip_reference` + `ref_time_distance` |
| 10 | `fact_gap_analysis` | Fact | County level compliance rollup: pct_covered, required vs actual, gap, compliance status | `fact_zip_access` + `ref_min_ratio` + `stg_beneficiaries` |

---

## Key SQL Logic

### County Type Classification
```sql
CASE
  WHEN (population >= 1000000 AND pop_density >= 1000)
    OR (population >= 500000  AND pop_density >= 1500)
    OR (pop_density >= 5000)                          THEN 'Large Metro'
  WHEN (population >= 200000  AND pop_density >= 10)  THEN 'Metro'
  WHEN (population >= 10000   AND pop_density >= 50)  THEN 'Micro'
  WHEN pop_density < 10                               THEN 'CEAC'
  ELSE                                                     'Rural'
END
```

### Distance Calculation (meters to miles)
```sql
ST_DISTANCE(
  ST_GEOGPOINT(bene_zip_long, bene_zip_lat),
  ST_GEOGPOINT(provider_zip_long, provider_zip_lat)
) / 1609.34 AS distance_miles
```

### Core Distance Filter
```sql
WHERE ST_DISTANCE(bene_zip_centroid, provider_zip_centroid) / 1609.34
      <= ref_time_distance.max_distance_miles
```

### Zip Radius Confidence Interval
```sql
zip_radius_miles = SQRT(area_sq_miles / PI())

distance_lower_bound = distance - bene_zip_radius - provider_zip_radius
distance_upper_bound = distance + bene_zip_radius + provider_zip_radius

confidence_flag =
  CASE
    WHEN upper_bound <= max_distance THEN 'CLEARLY COMPLIANT'
    WHEN lower_bound >= max_distance THEN 'CLEARLY NON-COMPLIANT'
    ELSE                                  'BORDERLINE'
  END
```

### Required Provider Count
```sql
required_count = CEIL(min_ratio_per_1000 * county_eligibles / 1000)
```

### Compliance Tests
```sql
-- Test 1: Access
pct_covered = SUM(zip_population WHERE has_access) / SUM(zip_population)
access_compliant = pct_covered >= compliance_threshold  -- 0.90 or 0.85

-- Test 2: Count
count_compliant = actual_provider_count >= required_provider_count

-- Overall
compliance_status = CASE WHEN access_compliant AND count_compliant
                    THEN 'COMPLIANT' ELSE 'NON-COMPLIANT' END
```

---

## Key Joins

| Join | Left Table | Right Table | Join Key | Purpose |
|------|-----------|-------------|----------|---------|
| Specialty mapping | `stg_providers` | `ref_specialty_crosswalk` | `specialty_ctg_cd = aetna_cd` | Fan out Aetna codes to CMS specialties |
| County name fix | `stg_providers` | `ref_county_name_crosswalk` | `county_nm = aetna_county_nm` | Resolve name mismatches |
| Zip geo | `stg_providers` / `stg_beneficiaries` | `ref_zip_reference` | `zip_cd = zip_code` | Get lat/long for distance calc |
| Distance filter | `stg_beneficiaries` | `stg_providers` | `ST_DISTANCE <= max_distance_miles` | Core compliance filter |
| Threshold lookup | distance matrix | `ref_time_distance` | `cms_specialty + county_type` | Get max allowed distance |
| Ratio lookup | gap analysis | `ref_min_ratio` | `cms_specialty + county_type` | Get required provider count |

---

## BigQuery Dataset Details

| Environment | Project | Dataset |
|-------------|---------|---------|
| Development | `anbc-hcb-dev` | `provider_ds_netconf_data_hcb_dev` |
| Production source | `anbc-hcb-prod` | `provider_ds_netconf_data_hcb_prod` |
| Public geo | `bigquery-public-data` | `geo_us_boundaries` |
| Public census | `bigquery-public-data` | `census_bureau_acs` |

All tables prefixed: `A870800_medicare_supply_demand_`
