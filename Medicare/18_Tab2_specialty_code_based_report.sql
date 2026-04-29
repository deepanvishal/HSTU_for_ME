from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─── COLORS ──────────────────────────────────────────────────
DARK_BLUE    = "1F3864"
MID_BLUE     = "2E75B6"
LIGHT_BLUE   = "D6E4F0"
GREY         = "F2F2F2"
DARK_GREY    = "595959"
WHITE        = "FFFFFF"
YELLOW       = "FFF2CC"
LIGHT_YELLOW = "FFFDE7"

def fill(hex_color):
    return PatternFill("solid", fgColor=hex_color)

def thin_border():
    s = Side(style="thin", color="CCCCCC")
    return Border(left=s, right=s, top=s, bottom=s)

def cell(ws, ref, value, bold=False, color="000000", bg=None,
         size=10, h_align="left", wrap=True, bdr=False, italic=False):
    c = ws[ref]
    c.value = value
    c.font = Font(name="Arial", bold=bold, color=color, size=size, italic=italic)
    if bg:
        c.fill = fill(bg)
    c.alignment = Alignment(horizontal=h_align, vertical="center", wrap_text=wrap)
    if bdr:
        c.border = thin_border()
    return c

def section_header(ws, row, col_start, col_end, text):
    start = get_column_letter(col_start)
    end   = get_column_letter(col_end)
    ws.merge_cells(f"{start}{row}:{end}{row}")
    cell(ws, f"{start}{row}", text,
         bold=True, color=WHITE, bg=MID_BLUE, size=11, h_align="left")
    ws.row_dimensions[row].height = 20
    return row + 1

def kv(ws, row, label, value, ncols=4, label_bg=GREY, value_bg=WHITE, h=18):
    ws.merge_cells(f"B{row}:C{row}")
    cell(ws, f"B{row}", label, bold=True, size=10, bg=label_bg, bdr=True)
    ws.merge_cells(f"D{row}:G{row}")
    cell(ws, f"D{row}", value, size=10, bg=value_bg, bdr=True, wrap=True)
    ws.row_dimensions[row].height = h
    return row + 1

def blank(ws, row, h=6):
    ws.row_dimensions[row].height = h
    return row + 1


def build_tab1(wb):
    ws = wb.create_sheet("1. Project Overview")
    ws.sheet_view.showGridLines = False

    # col widths
    ws.column_dimensions["A"].width = 3
    ws.column_dimensions["B"].width = 28
    ws.column_dimensions["C"].width = 28
    ws.column_dimensions["D"].width = 20
    ws.column_dimensions["E"].width = 20
    ws.column_dimensions["F"].width = 20
    ws.column_dimensions["G"].width = 20

    # ── TITLE ──────────────────────────────────────────────
    ws.merge_cells("B1:G1")
    cell(ws, "B1", "Medicare Supply Demand",
         bold=True, color=WHITE, bg=DARK_BLUE, size=18, h_align="center")
    ws.row_dimensions[1].height = 45

    ws.merge_cells("B2:G2")
    cell(ws, "B2", "Network Adequacy & Capacity Modeling  |  Florida Medicare Advantage  |  Plan Year 2026",
         color="AAAAAA", bg=DARK_BLUE, size=10, h_align="center", italic=True)
    ws.row_dimensions[2].height = 20

    row = 4

    # ── OBJECTIVE ──────────────────────────────────────────
    row = section_header(ws, row, 2, 7, "  PROJECT OBJECTIVE")
    row = blank(ws, row)
    ws.merge_cells(f"B{row}:G{row}")
    cell(ws, f"B{row}",
         "Build analytic models to determine whether the Aetna Medicare Advantage provider network "
         "has the right capacity, specialties, and geographic distribution — and to identify where "
         "to add, remove, or reconfigure providers under CMS regulatory requirements.",
         size=10, bg=WHITE, wrap=True)
    ws.row_dimensions[row].height = 40
    row += 2

    # ── DELIVERABLES ───────────────────────────────────────
    row = section_header(ws, row, 2, 7, "  DELIVERABLES")
    row = blank(ws, row)
    for label, value in [
        ("Compliance Table",
         "County × Specialty × Plan Type → COMPLIANT / NON-COMPLIANT per 42 CFR 422.116"),
        ("Access Coverage",
         "% of Medicare beneficiaries per county with at least 1 contracted provider within CMS distance threshold"),
        ("Provider Gap",
         "Contracted provider count vs CMS-required minimum per county per specialty"),
        ("Counties at Risk",
         "Counties failing access % or provider count standard — prioritized by gap size"),
        ("Bed Count Compliance",
         "Acute Inpatient Hospital contracted beds vs CMS required beds per county"),
    ]:
        row = kv(ws, row, f"  • {label}", value)
    row = blank(ws, row)

    # ── SCOPE ──────────────────────────────────────────────
    row = section_header(ws, row, 2, 7, "  SCOPE")
    row = blank(ws, row)
    for label, value in [
        ("Geography",       "Florida — 67 member counties evaluated for compliance"),
        ("Plan Types",      "MA-HMO, MA-PPO"),
        ("CMS Specialties", "43 provider and facility specialty types per 42 CFR 422.116"),
        ("Regulatory Year", "CMS 2026 HSD Reference File (published December 17, 2025)"),
        ("Data Snapshot",   "Most recent available month — CMS MA penetration file"),
    ]:
        row = kv(ws, row, f"  {label}", value)
    row = blank(ws, row)

    # ── GEOGRAPHY & DISTANCE ───────────────────────────────
    row = section_header(ws, row, 2, 7, "  HOW GEOGRAPHY & DISTANCE WORKS")
    row = blank(ws, row)
    for label, value in [
        ("Member Side",
         "All 67 Florida counties. Population sourced at zip code level from ACS 2018 Census. "
         "Compliance is evaluated at the MEMBER county level — not provider county."),
        ("Provider Side",
         "41 of 67 Florida counties have contracted Aetna providers. "
         "26 counties have zero contracted providers and are automatically non-compliant. "
         "Provider location is determined using the zip code centroid (geographic center of the zip)."),
        ("Distance Method",
         "Straight-line distance measured from member zip centroid to provider zip centroid "
         "using BigQuery ST_DISTANCE function, converted from meters to miles. "
         "CMS uses drive time — straight-line is an approximation."),
        ("Threshold",
         "CMS specifies a maximum distance per specialty per county type. "
         "A provider counts toward a member zip ONLY if the distance is within this threshold. "
         "Threshold uses the MEMBER county type — not the provider county type."),
        ("Cross-County Access",
         "A provider in one county can count toward compliance in a neighboring member county "
         "if their zip centroid is within the CMS distance threshold. "
         "Compliance is always measured from the member's perspective."),
        ("Rollup to County",
         "After identifying which member zips have access, population is rolled up to the "
         "member county. % Members With Access = population in zips with access / total county population."),
        ("Zip Uncertainty",
         "A confidence band is calculated using zip radius = SQRT(area_sq_miles / PI()). "
         "Distance lower/upper bound = measured distance ± (member zip radius + provider zip radius). "
         "Borderline cases near the threshold are flagged separately."),
    ]:
        row = kv(ws, row, f"  {label}", value, h=40)
    row = blank(ws, row)

    # ── V2 APPROACH ────────────────────────────────────────
    row = section_header(ws, row, 2, 7, "  V2 APPROACH — SPECIALTY MAPPING")
    row = blank(ws, row)
    for label, value in [
        ("Method",
         "Uses specialty_cd (raw specialty code from RPDB_RPNPRAC network table) "
         "mapped to CMS specialties via Global Lookup Table. "
         "One provider can map to multiple CMS specialties."),
        ("Multi-Specialty",
         "Provider network IDs are exploded from the network_id field in the provider file. "
         "Each provider's full specialty list is retrieved — not just the primary specialty category."),
        ("Difference from V1",
         "V1 used specialty_ctg_cd (primary specialty category code only — single code per provider). "
         "V2 uses specialty_cd (all specialty codes per provider via network join — broader coverage)."),
    ]:
        row = kv(ws, row, f"  {label}", value, h=35)
    row = blank(ws, row)

    # ── SPECIALTY MAPPING TABLE ────────────────────────────
    row = section_header(ws, row, 2, 7, "  CMS SPECIALTY → AETNA CODE MAPPING (43 Specialties)")
    row = blank(ws, row)

    import pandas as pd
    df_map = pd.read_csv('/home/claude/cms_to_aetna_full_mapping.csv',
                         names=['cms_specialty','aetna_code','aetna_description'],
                         skiprows=1, on_bad_lines='skip')
    grouped = df_map.groupby('cms_specialty').apply(
        lambda x: ', '.join(f"{row['aetna_code']} - {row['aetna_description']}"
                            for _, row in x.iterrows())
    ).reset_index()
    grouped.columns = ['cms_specialty', 'codes_with_desc']

    # preserve 422.116 order
    cms_order = [
        "Primary Care","Allergy and Immunology","Cardiology","Chiropractor",
        "Clinical Psychology","Clinical Social Work","Dermatology","Endocrinology",
        "ENT/Otolaryngology","Gastroenterology","General Surgery","Gynecology OB/GYN",
        "Infectious Diseases","Nephrology","Neurology","Neurosurgery",
        "Oncology Medical/Surgical","Oncology Radiation","Ophthalmology",
        "Orthopedic Surgery","Physiatry Rehabilitative Med","Plastic Surgery",
        "Podiatry","Psychiatry","Pulmonology","Rheumatology","Urology",
        "Vascular Surgery","Cardiothoracic Surgery","Acute Inpatient Hospitals",
        "Cardiac Surgery Program","Cardiac Catheterization","Critical Care ICU",
        "Surgical Services ASC","Skilled Nursing Facility","Diagnostic Radiology",
        "Mammography","Physical Therapy","Occupational Therapy","Speech Therapy",
        "Inpatient Psychiatric","Outpatient Infusion/Chemo","Outpatient Behavioral Health",
    ]
    lookup = dict(zip(grouped['cms_specialty'], grouped['codes_with_desc']))

    # column headers
    ws.merge_cells(f"B{row}:C{row}")
    cell(ws, f"B{row}", "CMS Specialty",
         bold=True, color=WHITE, bg=DARK_GREY, size=9, h_align="center", bdr=True)
    ws.merge_cells(f"D{row}:G{row}")
    cell(ws, f"D{row}", "Specialty Code - Description (comma separated)",
         bold=True, color=WHITE, bg=DARK_GREY, size=9, h_align="center", bdr=True)
    ws.row_dimensions[row].height = 16
    row += 1

    for i, cms in enumerate(cms_order):
        bg = LIGHT_BLUE if i % 2 == 0 else WHITE
        codes = lookup.get(cms, "No mapping found")

        ws.merge_cells(f"B{row}:C{row}")
        cell(ws, f"B{row}", cms, bold=True, size=9, bg=bg, bdr=True, wrap=False)

        ws.merge_cells(f"D{row}:G{row}")
        cell(ws, f"D{row}", codes, size=9, bg=bg, bdr=True, wrap=True)

        # auto height based on content length
        estimated_lines = max(2, len(codes) // 120 + 1)
        ws.row_dimensions[row].height = estimated_lines * 14
        row += 1

    row = blank(ws, row)

    # ── ASSUMPTIONS ────────────────────────────────────────
    row = section_header(ws, row, 2, 7, "  KEY ASSUMPTIONS & DATA DECISIONS")
    row = blank(ws, row)
    for label, value in [
        ("Required Provider Count",
         "Sourced directly from CMS 2026 HSD Reference File. "
         "Uses 95th percentile MA plan enrollment — not total Medicare eligibles."),
        ("Compliance Threshold",
         "90% for Large Metro and Metro counties. 85% for Micro, Rural, CEAC. "
         "Per 42 CFR 422.116(d)(4)."),
        ("Facility Minimum Count",
         "13 facility specialty types require minimum 1 per county (flat). "
         "Per 42 CFR 422.116(e)(2)(iii)."),
        ("Acute Inpatient Beds",
         "Required = CEIL(12.2 × beneficiaries_required_to_cover / 1,000). "
         "Measured in contracted BEDS not hospital count. Source: hosp_list_cmi."),
        ("Population Data",
         "ACS 2018 5-year estimates at zip code level. "
         "2020 zip-level data not available in BigQuery public data at time of analysis."),
        ("Distance Limitation",
         "Straight-line distance used. CMS uses drive time. "
         "Rural counties most affected — actual drive distances will be longer."),
        ("Telehealth Credit",
         "NOT applied. 42 CFR 422.116(d)(5) allows 10% credit for 14 specialties. "
         "No telehealth flag available in provider data."),
        ("Plan Type Independence",
         "MA-HMO and MA-PPO evaluated separately. "
         "A provider in MA-HMO does not count toward MA-PPO compliance."),
    ]:
        row = kv(ws, row, f"  {label}", value, label_bg=LIGHT_YELLOW, h=30)

    return ws


# ── TAB 2: COMPLIANCE REPORT ──────────────────────────────────

def build_tab2(wb, df):
    ws = wb.create_sheet("2. Compliance Report")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A5"

    # col widths
    col_widths = {
        "A": 20,  # county
        "B": 14,  # county type
        "C": 28,  # cms specialty
        "D": 10,  # plan type
        "E": 16,  # total bene
        "F": 22,  # bene required
        "G": 14,  # 95th ratio
        "H": 16,  # required count
        "I": 14,  # threshold
        "J": 18,  # county population
        "K": 20,  # pop with access
        "L": 16,  # pct covered
        "M": 18,  # actual count
        "N": 16,  # contracted beds
        "O": 14,  # gap
        "P": 14,  # access compliant
        "Q": 14,  # count compliant
        "R": 16,  # compliance status
    }
    for col, w in col_widths.items():
        ws.column_dimensions[col].width = w

    # ── ROW 1: TITLE ─────────────────────────────────────────
    ws.merge_cells("A1:R1")
    cell(ws, "A1", "Medicare Supply Demand — Compliance Report (V2)",
         bold=True, color=WHITE, bg=DARK_BLUE, size=14, h_align="center")
    ws.row_dimensions[1].height = 35

    # ── ROW 2: COLOR BAND LABELS ──────────────────────────────
    for rng, text, bg in [
        ("A2:D2",  "  IDENTIFIERS",                              DARK_GREY),
        ("E2:I2",  "  CMS RULES  (42 CFR 422.116 + HSD File)",   MID_BLUE),
        ("J2:N2",  "  AETNA NETWORK DATA",                       "C55A11"),
        ("O2:R2",  "  COMPLIANCE RESULTS",                       DARK_BLUE),
    ]:
        ws.merge_cells(rng)
        cell(ws, rng.split(":")[0], text,
             bold=True, color=WHITE, bg=bg, size=9, h_align="left")
    ws.row_dimensions[2].height = 16

    # ── ROW 3: CALLOUTS ───────────────────────────────────────
    callouts = {
        "A3": "",
        "B3": "",
        "C3": "",
        "D3": "",
        "E3": "Source: CMS 2026 HSD Reference File",
        "F3": "95th pct ratio × total Medicare beneficiaries",
        "G3": "CMS published 95th percentile base ratio",
        "H3": "From HSD file directly — not estimated",
        "I3": "90% Large Metro/Metro | 85% Micro/Rural/CEAC",
        "J3": "ACS 2018 zip population rolled to county",
        "K3": "SUM(zip_population WHERE has_access = TRUE)",
        "L3": "population_with_access / total_county_population",
        "M3": "COUNT(DISTINCT provider_id) within max_distance_miles",
        "N3": "SUM(Beds) from hosp_list_cmi — Acute Inpatient only",
        "O3": "required_provider_count − actual_count",
        "P3": "pct_covered >= compliance_threshold",
        "Q3": "actual_count >= required_provider_count",
        "R3": "BOTH access AND count standards met",
    }
    for ref, txt in callouts.items():
        cell(ws, ref, txt, size=8, color="666666", bg="F9F9F9",
             italic=True, wrap=True)
    ws.row_dimensions[3].height = 28

    # ── ROW 4: COLUMN HEADERS ────────────────────────────────
    headers = [
        ("A4", "County",                        DARK_GREY),
        ("B4", "County Type",                   DARK_GREY),
        ("C4", "CMS Specialty",                 DARK_GREY),
        ("D4", "Plan Type",                     DARK_GREY),
        ("E4", "Total Medicare\nBeneficiaries",  MID_BLUE),
        ("F4", "Beneficiaries\nRequired to Cover", MID_BLUE),
        ("G4", "95th Pct\nBase Ratio",           MID_BLUE),
        ("H4", "CMS Required\nCount",            MID_BLUE),
        ("I4", "Access\nThreshold",              MID_BLUE),
        ("J4", "County Population\n(ACS 2018)",  "C55A11"),
        ("K4", "Population\nWith Access",        "C55A11"),
        ("L4", "% Members\nWith Access",         "C55A11"),
        ("M4", "Contracted\nProviders / Beds",   "C55A11"),
        ("N4", "Contracted Beds\n(Inpatient Only)", "C55A11"),
        ("O4", "Gap\n(Required - Actual)",       DARK_BLUE),
        ("P4", "Access\nStandard Met",           DARK_BLUE),
        ("Q4", "Count\nStandard Met",            DARK_BLUE),
        ("R4", "Compliance\nStatus",             DARK_BLUE),
    ]
    ws.row_dimensions[4].height = 35
    for ref, label, bg in headers:
        cell(ws, ref, label, bold=True, color=WHITE,
             bg=bg, size=9, h_align="center", bdr=True)

    # ── DATA ROWS ────────────────────────────────────────────
    LIGHT_GREEN  = "E2EFDA"
    LIGHT_RED    = "FFE0E0"
    LIGHT_BLUE_D = "D6E4F0"
    LIGHT_ORANGE = "FCE4D6"

    for i, (_, row) in enumerate(df.iterrows()):
        r = i + 5
        is_compliant = str(row.get("compliance_status", "")).strip() == "COMPLIANT"
        row_bg = LIGHT_GREEN if is_compliant else LIGHT_RED

        def v(col):
            val = row.get(col, 0)
            if val is None or (isinstance(val, float) and str(val) == 'nan'):
                return 0
            return val

        # bool → Yes/No
        access_c = "Yes" if bool(v("access_compliant")) else "No"
        count_c  = "Yes" if bool(v("count_compliant"))  else "No"

        # beds: 0 for non-hospital
        beds = v("total_contracted_beds")
        if beds is None or beds == 0:
            beds = 0

        data = [
            ("A", v("county_name"),                      DARK_GREY,  row_bg),
            ("B", v("county_type"),                      DARK_GREY,  row_bg),
            ("C", v("cms_specialty"),                    DARK_GREY,  row_bg),
            ("D", v("plan_type"),                        DARK_GREY,  row_bg),
            ("E", v("county_total_beneficiaries"),       MID_BLUE,   LIGHT_BLUE_D),
            ("F", v("beneficiaries_required_to_cover"),  MID_BLUE,   LIGHT_BLUE_D),
            ("G", v("ratio_95th_percentile"),            MID_BLUE,   LIGHT_BLUE_D),
            ("H", v("required_provider_count"),          MID_BLUE,   LIGHT_BLUE_D),
            ("I", v("compliance_threshold"),             MID_BLUE,   LIGHT_BLUE_D),
            ("J", v("total_county_population"),          "C55A11",   LIGHT_ORANGE),
            ("K", v("population_with_access"),           "C55A11",   LIGHT_ORANGE),
            ("L", v("pct_covered"),                      "C55A11",   LIGHT_ORANGE),
            ("M", v("actual_count"),                     "C55A11",   LIGHT_ORANGE),
            ("N", beds,                                  "C55A11",   LIGHT_ORANGE),
            ("O", v("provider_gap"),                     DARK_BLUE,  row_bg),
            ("P", access_c,  DARK_BLUE,
             LIGHT_GREEN if access_c == "Yes" else LIGHT_RED),
            ("Q", count_c,   DARK_BLUE,
             LIGHT_GREEN if count_c  == "Yes" else LIGHT_RED),
            ("R", v("compliance_status"), DARK_BLUE,
             LIGHT_GREEN if is_compliant else LIGHT_RED),
        ]

        for col, val, txt_color, bg_color in data:
            c = ws[f"{col}{r}"]
            c.value = val
            c.font = Font(name="Arial", color=txt_color, size=9,
                          bold=(col == "R"))
            c.fill = fill(bg_color)
            c.alignment = Alignment(horizontal="center", vertical="center",
                                    wrap_text=False)
            c.border = thin_border()
            if col == "L":
                c.number_format = "0.0%"
            elif col == "G":
                c.number_format = "0.0000"
            elif col == "I":
                c.number_format = "0%"

        ws.row_dimensions[r].height = 15

    # note
    note_r = len(df) + 5 + 1
    ws.merge_cells(f"A{note_r}:R{note_r}")
    cell(ws, f"A{note_r}",
         "NOTE: Contracted Beds (col N) populated only for Acute Inpatient Hospitals — 0 for all other specialties. "
         "Gap is negative when actual count exceeds required (surplus). "
         "Compliance Status = COMPLIANT only when BOTH Access Standard AND Count Standard are met.",
         size=8, color="666666", bg="F9F9F9", italic=True, wrap=True)
    ws.row_dimensions[note_r].height = 30

    return ws


# ── MAIN ─────────────────────────────────────────────────────
import pandas as pd
from google.cloud import bigquery

PROJECT = "anbc-hcb-dev"
DATASET = "provider_ds_netconf_data_hcb_dev"
PREFIX  = "A870800_medicare_supply_demand"

COMPLIANCE_QUERY = f"""
SELECT
  county_name,
  county_type,
  cms_specialty,
  plan_type,
  COALESCE(county_total_beneficiaries, 0)       AS county_total_beneficiaries,
  COALESCE(beneficiaries_required_to_cover, 0)  AS beneficiaries_required_to_cover,
  COALESCE(ratio_95th_percentile, 0)            AS ratio_95th_percentile,
  COALESCE(required_provider_count, 0)          AS required_provider_count,
  COALESCE(compliance_threshold, 0)             AS compliance_threshold,
  COALESCE(total_county_population, 0)          AS total_county_population,
  COALESCE(population_with_access, 0)           AS population_with_access,
  COALESCE(pct_covered, 0)                      AS pct_covered,
  COALESCE(actual_count, 0)                     AS actual_count,
  COALESCE(total_contracted_beds, 0)            AS total_contracted_beds,
  COALESCE(provider_gap, 0)                     AS provider_gap,
  access_compliant,
  count_compliant,
  compliance_status
FROM `{PROJECT}.{DATASET}.{PREFIX}_fact_gap_analysis_v2`
ORDER BY county_name, cms_specialty, plan_type
"""

if __name__ == "__main__":
    client = bigquery.Client(project=PROJECT)

    print("Querying fact_gap_analysis_v2...")
    df = client.query(COMPLIANCE_QUERY).to_dataframe()
    print(f"  {len(df):,} rows")

    wb = Workbook()
    wb.remove(wb.active)

    print("Building Tab 1...")
    build_tab1(wb)

    print("Building Tab 2...")
    build_tab2(wb, df)

    output = "medicare_supply_demand_v2.xlsx"
    wb.save(output)
    print(f"Saved: {output}")
