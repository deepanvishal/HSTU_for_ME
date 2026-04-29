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

    # embedded specialty mapping — code - description per CMS specialty
    SPECIALTY_MAPPING = {
    'Acute Inpatient Hospitals': '2OT - Hospital Outpatient, CH - Children\'s Hospital, HO - Acute Short Term Hospital, HSLT - Hospitalist, LHO - Long Term Acute Care Hospital, 2HSLT - Hospitalist, 2SH - Specialty Hospital, 91002 - Hospitalist',
    'Allergy and Immunology': '10326 - Allergy/Immunology, 10501 - Allergy & Immunology, 10603 - Dermatological Immunology/Diag, 90003 - Allergy, 90004 - Allergy (Pediatric), 90335 - Immunology, 90386 - Otolaryngology/Allergy, 91124 - Transplant & Immunology, 2A - Allergy, 2AIM - Immunology, 2AIMP - Immunology (Pediatric), 2AP - Allergy (Pediatric), 2ENA - Otolaryngology/Allergy, 2ENA - Otolaryngology/Allergy, 10411 - Pediatric Allergy & Immunology',
    'Cardiac Catheterization': '10302 - Cardiac Electrophysiology, 91205 - Cardiac Monitoring Service, 2CEP - Cardiac Electrophysiology, 10332 - Interventional Cardiology',
    'Cardiac Surgery Program': '91046 - Cardiac Valve Replacement, 91205 - Cardiac Monitoring Service, 91206 - Cardiac Surgery Program, 2CS - Cardiothoracic/Cardiovascular',
    'Cardiology': '10302 - Cardiac Electrophysiology, 10303 - Cardiovascular Disease, 10322 - Cardiology, 40312 - Nuclear Cardiology, 90313 - Cardiology (Invasive), 91046 - Cardiac Valve Replacement, 91205 - Cardiac Monitoring Service, 2C - Cardiology, 2CC - Cardiology (Pediatric), 2CEP - Cardiac Electrophysiology, 2CI - Cardiology (Invasive), 2CS - Cardiothoracic/Cardiovascular, 10332 - Interventional Cardiology, 10339 - Cardiology Adv Heart Failure/, 10403 - Pediatric Cardiology',
    'Cardiothoracic Surgery': '30805 - Surgery Thoracic Cardiovascul, 30812 - Surgery Congenital Cardiac/Th, 30901 - Surgery Thoracic, 91206 - Cardiac Surgery Program, 91215 - Heart Transplant Program, 2CS - Cardiothoracic/Cardiovascular, 2TS - Thoracic Surgery, 10426 - Pediatric Thoracic Surgery',
    'Chiropractor': '91146 - Chiropractics, 2CH - Chiropractics, DC - Chiropractor',
    'Clinical Psychology': '90305 - Adolescent Psychology, 90314 - Child Psychology, 91018 - Psychological Testing, 91029 - Neuropsych Testing, CP - Clinical Psychologist, NPS - Neuropsychologist, 2NPH - Neuropsychology, 2PHA - Adolescent Psychology, 2PHGR - Geriatric Psychology, 2PHP - Child Psychology',
    'Clinical Social Work': '90371 - Psychiatric Social Worker, 91207 - Certified Social Work, 2MLS - Social Worker Masters Licensed, 2MUS - Social Worker(Masters w/o Lic), SW - Clinical Social Worker, 2PYSW - Psychiatric Social Worker',
    'Critical Care ICU': '10304 - Critical Care Medicine, 10432 - Pediatric Intensive Care, 20102 - Critical Care Medicine/Obstetr, 30102 - Critical Care Medicine/Anesthe, 30302 - Critical Care Medicine Neurolo, 30803 - Surgery Critical care, 91125 - Trauma Surgical Critical Care, 91165 - Intensive Care Coordination, 2CCM - Critical Care Medicine, 2CCMP - Critical Care Medicine (Pediat, 10404 - Pediatric Critical Care, 11016 - Neurocritical Care',
    'Dermatology': '10430 - Pediatric Dermatology, 10601 - Dermatology, 10602 - Dermatopathology/Dermatology, 40207 - Dermatopathology/Pathology, 2D - Dermatology, 2DP - Dermatopathology, 2DPD - Dermatology (Pediatric)',
    'Diagnostic Radiology': '40306 - Diagnostic Roentgenology, 40311 - Neuroradiology, 40313 - Nuclear Imaging and Therapy, 40315 - Diagnostic Ultrasound, 40320 - Body Imaging, 91224 - Medical Imaging, RFA - Radiology Center',
    'ENT/Otolaryngology': '30601 - Otolaryngology, 30603 - Otorhinolaryngology & Oro-Faci, 30604 - Otorhinolaryngology/Plastic Su, 30605 - Otology, 30607 - Otorhinolaryngology, 30608 - Otology/Neurotology, 30609 - Otolaryngology (Pediatrics), 30806 - Surgery Head & Neck, 90386 - Otolaryngology/Allergy, 91117 - Sleep Medicine (Otolaryngology), 2EN - Otolaryngology, 2ENA - Otolaryngology/Allergy, 2PEN - Otolaryngology (Pediatric), 2ENA - Otolaryngology/Allergy, 2ENHN - Otolaryngology(Head&Neck) Su, 2ENN - Neuro-Otology, 10417 - Pediatric Otolaryngology, 14601 - Neurotology, 91061 - ENT Trauma, 91078 - Otolaryngology (ENT) Cancer Su',
    'Endocrinology': '10306 - Endocrinology Diabetes & Meta, 10319 - Endocrinology, 20105 - Endocrinology Reproductive, 2E - Endocrinology, 2PE - Endocrinology (Pediatric), 10405 - Pediatric Endocrinology, 91059 - Endocrine Surgery',
    'Gastroenterology': '10307 - Gastroenterology, 91045 - Capsule Endoscopy, 2PG - Gastroenterology (Pediatric), 2G - Gastroenterology, 2PG - Gastroenterology (Pediatric), 10406 - Pediatric Gastroenterology, 91060 - Endoscopic Ultrasound, 91063 - Endoscopic Retrograde Cholangi, 91065 - Esophageal Motility Disorders',
    'General Surgery': '30502 - Surgery Hand/Orthopedic, 30702 - Surgery Hand/Plastic, 30803 - Surgery Critical care, 30804 - Surgery General Vascular, 30809 - Surgery Hand, 30810 - Surgery Oncology, 30811 - Surgery Hospice and Palliativ, 2S - Surgery (General)',
    'Gynecology OB/GYN': '20104 - Maternal & Fetal Medicine, 20106 - Gynecology, 20107 - Perinatology, 20108 - Obstetrics & Gynecology - CA P, 20109 - Obstetrics/Gynecology Hospice, 20110 - Female Pelvic Medicine & Recon, 30807 - Surgery Obstetrics & Gynecolo, 90069 - Perinatology/PF, 90304 - Adolescent Gynecology, 90355 - Obstetrics, 90398 - Uro-Gynecology, 91097 - Pediatric Gynecology, 91102 - Pediatric Uro-Gynecology, 2PAOG - Physicians Assistant Ob/Gyn, 2NPOG - Nurse practitioner (ob/gyn), 2OG - Ob/Gyn, 2OGA - Adolescent Gynecology, 2OGOB - Obstetrics, 2OH - Perinatology, 2UGY - Uro-gynecology, 20191 - Obstetrics & Gynecology',
    'Infectious Diseases': '10310 - Infectious Disease, 91158 - Infectious Disease Focus, 2III - Infectious Disease, 2IIP - Infectious Diseases(Pediatric), 10412 - Pediatric Infectious Disease',
    'Inpatient Psychiatric': 'RTF - Residential Treatment Facility, 2PLMD - Palliative Medicine, 91001 - Palliative Medicine, 91003 - Psychotic Disorders',
    'Mammography': '91223 - Mammography',
    'Nephrology': '10312 - Nephrology, 91217 - Hemodialysis, 91220 - Kidney Transplant Program, 2N - Nephrology, DI - Dialysis Center, 2NP - Nephrology (Pediatric), 2HD - Hemodialysis, 10408 - Pediatric Nephrology',
    'Neurology': '10806 - Neuromuscular Medicine Physica, 91044 - Botox injections Neurology, 91149 - Sleep Medicine-Neurology, 2NE - Neurology, 2PN - Neurology (Pediatric), 10334 - Vascular Neurology, 10422 - Pediatric Neurology, 11002 - Neurology, 11003 - Neurology Child, 11006 - Neurology & Psychiatry, 11008 - Neurology Chemical, 11009 - Child Neurology, 11014 - Neurology/Psychiatry Hospice, 11015 - Sleep Medicine - Neurology, 11016 - Neurocritical Care, 11102 - Neuromuscular Medicine Psychia, 11103 - Epilepsy, 91062 - Epilepsy Surgery, 91081 - Movement Disorders, 91082 - Multiple Sclerosis, 91084 - Neuromuscular Medicine, 91086 - Neurovascular Surgery',
    'Neurosurgery': '10803 - Spinal Cord Injury Medicine, 90347 - Neurosurgery (Pediatric), 90348 - Neurosurgery (Spine), 91118 - Spinal Cord Stimulation, 91119 - Stereotactic & Functional Neur, 2NS - Neurosurgery, 2NSP - Neurosurgery (Pediatric), 2NSS - Neurosurgery (Spine)',
    'Occupational Therapy': '90374 - Occupational Therapy (Pediatri, 91142 - Occupational Therapy, 2TO - Occupational Therapy',
    'Oncology Medical/Surgical': '10311 - Oncology Medical, 10315 - Hematology/Oncology, 20103 - Oncology Gynecologic, 30810 - Surgery Oncology, 90372 - Radiation Oncology (Pediatric), 91126 - Urologic Oncology, 91129 - Surgery Carcinoid, 2ROP - Radiation Oncology (Pediatric), 91085 - Neuro-Oncology',
    'Oncology Radiation': '40303 - Radiation Oncology, 40304 - Radiological Physics, 40310 - Therapeutic Radiology, 40316 - Radiation Therapy, 40318 - Radium Therapy, 90372 - Radiation Oncology (Pediatric), 2RO - Radiation Therapy, 2ROP - Radiation Oncology (Pediatric)',
    'Ophthalmology': '30401 - Opthalmology, 30402 - Retinal Opthalmology, 30403 - Sleep Medicine-Ophthalmology/O, 90089 - Retinal Specialist, 90311 - Anterior Segment (Glaucoma), 90315 - Corneal Specialist, 90343 - Neuro-Ophthalmology, 90356 - Oculoplastic Surgery, 91155 - Pediatric Ophthalmology, 2PEO - Ophthalmology (Pediatric), 2O - Ophthalmology, 2OAG - Anterior Segment (Glaucoma), 2OC - Corneal Specialist, 2PSOC - Oculoplastic Surgery, 2RS - Retinal Specialist, 10414 - Pediatric Opthalmology, 91070 - Glaucoma Service, 91087 - Ophthamologic Cancer, 91088 - Orbital Surgery',
    'Orthopedic Surgery': '10317 - Oncology Orthopedic, 30501 - Surgery Orthopedic, 30502 - Surgery Hand/Orthopedic, 30503 - Surgery Knee, 90361 - Orthopedics (Foot & Ankle), 90362 - Orthopedics (Joint Replacement, 90365 - Orthopedics Surgery (Spine), 91101 - Pediatric Orthopedic Oncology, 2OR - Orthopedics, 2ORFA - Orthopedics (Foot & Ankle), 2ORON - Orthopedics (Oncology), 2ORR - Orthopedics (Joint Replacement, 2ORS - Orthopedics Surgery (Spine), 2ORSM - Orthopedics (Sports Medicine), 2POR - Orthopedics (Pediatric), 10418 - Pediatric Orthopedic, 91092 - Orthopedic Elbow Replacement, 91093 - Orthopedic Trauma, 91094 - Orthopedic Shoulder',
    'Outpatient Behavioral Health': '10204 - Addiction Medicine, 90001 - Addictions Counselor, 91032 - Applied Behavioral Analysis, 91134 - Behavioral Health Rehabilitati, 91174 - Mobile Crisis Intervention (MC, 91175 - Behavioral Health Services Tel, 91278 - Applied Behavioral Analysis (A, 2AC - Addictions Counselor, 2MH - Mental Health-Substance Abuse, ABA - Applied Behavioral Analysis, BHR - Behavioral Health Rehabilitati, CAC - Certified Addictions Counselor, CMC - Community Mental Health Center, MH - Mental Health - Substance Abus, SA - Substance Abuse Facility, 11007 - Addictionology, 11011 - Addiction Psychiatry, 90428 - Mental Health, 91005 - Dialectic Behavioral Therapy, 91006 - Cognitive Behavioral Therapy, 91011 - Substance Abuse Professional, 91012 - Crisis Intervention',
    'Outpatient Infusion/Chemo': '91180 - Antibiotic Infusion, 91218 - Home Infusion Therapy for HIV, 91234 - Outpatient Infusion/Chemothera, HI - Home Infusion, IC - Infusion Center, 2IC - Infusion Center',
    'Physiatry Rehabilitative Med': '10801 - Physical Medicine & Rehabilita, 10802 - Rehabilitation Medicine, 10805 - Physical Medicine Hospice and, 10807 - Pediatric Physical Medicine an, 2PM - Physical Medicine, 2RM - Rehab Medicine',
    'Physical Therapy': '90331 - Hand Rehabilitation, 90375 - Physical Therapy (Pediatric), 91141 - Physical Therapy, 2HR - Hand Rehabilitation, 2PT - Physical Therapy',
    'Plastic Surgery': '10428 - Pediatric Plastic Surgery, 30602 - Surgery Oro-Facial Plastic, 90308 - Facial Plastic and Reconstruct, 90316 - Craniofacial Surgery, 90317 - Craniofacial Surgery (Pediatri, 90356 - Oculoplastic Surgery, 91111 - Reconstructive Breast Surgery, 91112 - Reconstructive Breast Surgery, 91113 - Reconstructive Breast Surgery, 2PS - Plastic Surgery, 2PSCF - Craniofacial Surgery, 2PSCP - Craniofacial Surgery (Pediatri, 2PSOC - Oculoplastic Surgery, 2PSP - Plastic Surgery (Pediatric), 91054 - Craniofacial Plastics',
    'Podiatry': '91213 - Foot and Ankle Surgery, 91214 - Foot Surgery, DP - Podiatrist, 2PO - Podiatry',
    'Primary Care': '10101 - General Practice, 10201 - Family Practice, 10202 - Geriatric Medicine/Family Prac, 10301 - Internal Medicine, 10308 - Geriatric Medicine/Internal Me, 10433 - Sports Medicine/Pediatrics, 10438 - Pediatrics Hospice and Pallia, 30609 - Otolaryngology (Pediatrics), 50101 - General Practice - Dental, 90360 - Oral Surgery (Pediatrics), 91151 - Sleep Medicine-Family Practice, 91154 - Obesity Medicine-Pediatrics, 91209 - Config-Primary Care, 91210 - Config-Primary Care Attestatio, 2IM - Internal Medicine, 2P - Pediatrics, 2FP - Family Practice, 2GP - General Practice, 2I - Internal Medicine, 10336 - Internal Medicine Hospice, 10401 - Pediatrics, 10421 - Pediatric Internal Medicine',
    'Psychiatry': '91244 - Psychiatry Autism Spectrum, 91245 - Psychiatry Child & Adolescent, 91246 - Psychiatry Child & Adolescent, 91247 - Psychiatry Child & Adolestcent, 91249 - Psychiatry Child & Adolescent, 91250 - Psychiatry Home Based Services, 91252 - Psychiatry Trauma/Crisis, 2PPY - Psychiatry (Pediatric), 2PY - Psychiatry, 2PYGR - Geriatric Psychiatry, 11001 - Psychiatry, 11004 - Psychiatry Child & Adolescent, 11005 - Psychiatry Geriatric, 11006 - Neurology & Psychiatry, 11007 - Addictionology, 11010 - Child Psychiatry, 11011 - Addiction Psychiatry, 11013 - Forensic Medicine, 11014 - Neurology/Psychiatry Hospice, 11015 - Sleep Medicine - Neurology, 11101 - Psychomatic Medicine',
    'Pulmonology': '10304 - Critical Care Medicine, 10313 - Pulmonary Disease, 10318 - Medical Diseases of Chest, 20102 - Critical Care Medicine/Obstetr, 30102 - Critical Care Medicine/Anesthe, 30302 - Critical Care Medicine Neurolo, 91139 - Sleep Medicine - Pulmonology, 2CCM - Critical Care Medicine, 2CCMP - Critical Care Medicine (Pediat, 2PD - Pulmonary Disease, 2PPD - Pulmonary Diseases (Pediatric), 10409 - Pediatric Pulmonology',
    'Rheumatology': '10314 - Rheumatology, 91041 - Arthritis Reconstruction, 2RH - Rheumatology, 2RHP - Rheumatology (Pediatric), 10420 - Pediatric Rheumatology',
    'Skilled Nursing Facility': '91287 - Assisted Living Center, 91294 - Skilled Nursing Facilities, 91301 - Nursing Facility Transition Di, 91302 - Recuperative Care, ALC - Assisted Living Center, LSS - Long-Term Services and Support, SK - Skilled Nursing Facility, 2SNF - Skilled Nursing Facility',
    'Speech Therapy': '90373 - Speech Therapy (Pediatric), 91143 - Speech Therapy, 91257 - Speech/Hearing, 91258 - Speech/Hearing Therapy, 91259 - Speech/Language/Hearing Therap, SH - Speech Pathologist, ST - Speech Therapist, 2TT - Speech Therapy',
    'Surgical Services ASC': '91235 - Outpatient Surgery, AC - Ambulatory Surgicenter, FEC - Freestanding Emergency Center, 2FS - Free Standing Surgical Unit',
    'Urology': '30301 - Surgery Neurological, 30808 - Surgery Urological, 31001 - Urology, 90379 - Urology (Male Infertility), 90398 - Uro-Gynecology, 91044 - Botox injections Neurology, 91102 - Pediatric Uro-Gynecology, 91126 - Urologic Oncology, 91127 - UROLOGICTR, 91149 - Sleep Medicine-Neurology, 91220 - Kidney Transplant Program, 2NE - Neurology, 2PN - Neurology (Pediatric), 2PU - Urology (Pediatric), 2U - Urology, 2UGY - Uro-gynecology, 2UMI - Urology (Male Infertility), 10334 - Vascular Neurology, 10415 - Pediatric Urology, 10422 - Pediatric Neurology, 11002 - Neurology, 11003 - Neurology Child, 11006 - Neurology & Psychiatry, 11008 - Neurology Chemical, 11009 - Child Neurology, 11014 - Neurology/Psychiatry Hospice, 11015 - Sleep Medicine - Neurology',
    'Vascular Surgery': '40317 - Vascular & Interventional Radi, 40319 - Angiography and Interventional, 90071 - Peripheral Vascular Disease, 2IY - Peripheral Vascular Disease, 2VS - Vascular Surgery, 10334 - Vascular Neurology, 91086 - Neurovascular Surgery',
}
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
    lookup = SPECIALTY_MAPPING

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


from openpyxl.formatting.rule import ColorScaleRule

# ── TAB 3 defined here before MAIN ───────────────────────────

PROJECT        = "anbc-hcb-dev"           # table project
CLIENT_PROJECT = "anbc-dev-prv-nc-ds"     # billing/auth project
DATASET        = "provider_ds_netconf_data_hcb_dev"
PREFIX         = "A870800_medicare_supply_demand"

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

SUMMARY_SPECIALTY_QUERY = f"""
SELECT
  cms_specialty,
  plan_type,
  COUNTIF(compliance_status = 'COMPLIANT')     AS compliant_counties,
  COUNTIF(compliance_status = 'NON-COMPLIANT') AS non_compliant_counties,
  COUNT(*)                                      AS total_counties,
  ROUND(
    COUNTIF(compliance_status = 'COMPLIANT') / COUNT(*), 4
  )                                             AS pct_compliant,
  COUNTIF(access_compliant = FALSE)             AS access_failures,
  COUNTIF(count_compliant = FALSE)              AS count_failures
FROM `{PROJECT}.{DATASET}.{PREFIX}_fact_gap_analysis_v2`
GROUP BY cms_specialty, plan_type
ORDER BY pct_compliant ASC, cms_specialty, plan_type
"""


# ── TAB 3: SUMMARY BY PLAN × SPECIALTY ───────────────────────

def build_tab3(wb, df_summary):
    ws = wb.create_sheet("3. Summary by Specialty")
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "A5"

    col_widths = {
        "A": 30,  # cms specialty
        "B": 12,  # plan type
        "C": 16,  # compliant counties
        "D": 20,  # non-compliant counties
        "E": 14,  # total counties
        "F": 14,  # pct compliant
        "G": 16,  # access failures
        "H": 16,  # count failures
    }
    for col, w in col_widths.items():
        ws.column_dimensions[col].width = w

    # ── ROW 1: TITLE ─────────────────────────────────────────
    ws.merge_cells("A1:H1")
    cell(ws, "A1", "Medicare Supply Demand — Specialty Compliance Summary (V2)",
         bold=True, color=WHITE, bg=DARK_BLUE, size=14, h_align="center")
    ws.row_dimensions[1].height = 35

    # ── ROW 2: SUBTITLE ──────────────────────────────────────
    ws.merge_cells("A2:H2")
    cell(ws, "A2",
         "Grain: CMS Specialty × Plan Type  |  Each row = county-level pass/fail counts  |  "
         "Sorted by % Compliant ascending (worst performing specialties first)",
         size=9, color="666666", bg="F9F9F9", italic=True, h_align="left")
    ws.row_dimensions[2].height = 18

    # ── ROW 3: CALLOUTS ──────────────────────────────────────
    callouts = {
        "A3": "",
        "B3": "",
        "C3": "Counties where BOTH access % AND count standard are met",
        "D3": "Counties where EITHER access % OR count standard fails",
        "E3": "Total Florida counties evaluated",
        "F3": "compliant_counties / total_counties",
        "G3": "Counties where pct_covered < compliance_threshold",
        "H3": "Counties where actual_count < required_provider_count",
    }
    for ref, txt in callouts.items():
        cell(ws, ref, txt, size=8, color="666666",
             bg="F9F9F9", italic=True, wrap=True)
    ws.row_dimensions[3].height = 28

    # ── ROW 4: HEADERS ───────────────────────────────────────
    headers = [
        ("A4", "CMS Specialty",         DARK_GREY),
        ("B4", "Plan Type",             DARK_GREY),
        ("C4", "Compliant\nCounties",   "375623"),
        ("D4", "Non-Compliant\nCounties","C00000"),
        ("E4", "Total\nCounties",        DARK_BLUE),
        ("F4", "% Compliant",            DARK_BLUE),
        ("G4", "Access\nFailures",       MID_BLUE),
        ("H4", "Count\nFailures",        MID_BLUE),
    ]
    ws.row_dimensions[4].height = 35
    for ref, label, bg in headers:
        cell(ws, ref, label, bold=True, color=WHITE,
             bg=bg, size=10, h_align="center", bdr=True)

    # ── DATA ROWS ────────────────────────────────────────────
    prev_specialty = None
    alt = True

    for i, (_, row) in enumerate(df_summary.iterrows()):
        r = i + 5

        # alternate shade per specialty group
        if row['cms_specialty'] != prev_specialty:
            alt = not alt
            prev_specialty = row['cms_specialty']
        row_bg = GREY if alt else WHITE

        pct = float(row.get('pct_compliant', 0) or 0)

        data = [
            ("A", row.get('cms_specialty', ''),           DARK_GREY,  row_bg),
            ("B", row.get('plan_type', ''),                DARK_GREY,  row_bg),
            ("C", int(row.get('compliant_counties', 0) or 0),    "375623", "E2EFDA"),
            ("D", int(row.get('non_compliant_counties', 0) or 0),"C00000", "FFE0E0"),
            ("E", int(row.get('total_counties', 0) or 0),         DARK_BLUE, row_bg),
            ("F", pct,                                     DARK_BLUE,  row_bg),
            ("G", int(row.get('access_failures', 0) or 0),        MID_BLUE,  LIGHT_BLUE),
            ("H", int(row.get('count_failures', 0) or 0),         MID_BLUE,  LIGHT_BLUE),
        ]

        for col, val, txt_color, bg_color in data:
            c = ws[f"{col}{r}"]
            c.value = val
            c.font = Font(name="Arial", color=txt_color, size=10,
                          bold=(col == "F"))
            c.fill = fill(bg_color)
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = thin_border()
            if col == "F":
                c.number_format = "0.0%"

        ws.row_dimensions[r].height = 16

    # ── GRADIENT COLOR SCALE ON % COMPLIANT (col F) ──────────
    last_row = len(df_summary) + 4
    ws.conditional_formatting.add(
        f"F5:F{last_row}",
        ColorScaleRule(
            start_type="num", start_value=0,   start_color="C00000",
            mid_type="num",   mid_value=0.5,   mid_color="FFEB84",
            end_type="num",   end_value=1,     end_color="375623"
        )
    )

    return ws


# ── MAIN ─────────────────────────────────────────────────────
import pandas as pd
from google.cloud import bigquery
    client = bigquery.Client(project=CLIENT_PROJECT)

    print("Querying compliance data...")
    df = client.query(COMPLIANCE_QUERY).to_dataframe()
    print(f"  {len(df):,} rows")

    print("Querying specialty summary...")
    df_summary = client.query(SUMMARY_SPECIALTY_QUERY).to_dataframe()
    print(f"  {len(df_summary):,} rows")

    wb = Workbook()
    wb.remove(wb.active)

    print("Building Tab 1...")
    build_tab1(wb)

    print("Building Tab 2...")
    build_tab2(wb, df)

    print("Building Tab 3...")
    build_tab3(wb, df_summary)

    output = "medicare_supply_demand_v2.xlsx"
    wb.save(output)
    print(f"Saved: {output}")
