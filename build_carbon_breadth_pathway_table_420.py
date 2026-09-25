import os
import pandas as pd


# =============================================================================
# INPUT / OUTPUT PATHS
# =============================================================================

INPUT_FILE = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\BUSCO20_annotation\UniProt\Carbon_Breadth_top20_UniProt_annotation_inspection_420.csv"

OUTPUT_DIR = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\BUSCO20_annotation\pathway_analysis_420"

OUTPUT_TABLE = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_top20_pathway_ready_420.csv"
)

OUTPUT_REPORT = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_top20_pathway_summary_420.txt"
)


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("BUILDING CARBON BREADTH TOP-20 PATHWAY TABLE")
print("=" * 80)

print("\nInput:")
print(INPUT_FILE)


# =============================================================================
# CHECK INPUT
# =============================================================================

if not os.path.exists(INPUT_FILE):

    print("\nERROR: Input file not found:")
    print(INPUT_FILE)

    raise SystemExit(1)


os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# LOAD
# =============================================================================

df = pd.read_csv(INPUT_FILE)

print("\nInput table loaded.")
print("Shape:", df.shape)

print("\nColumns:")
for column in df.columns:
    print(" ", column)


# =============================================================================
# HELPER
# =============================================================================

def clean(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in {
        "nan",
        "none",
        "na",
        "n/a",
        "-"
    }:
        return ""

    return value


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

required = [
    "BUSCO_ID",
    "Functional_Description",
    "UniProt",
    "GO",
    "KEGG",
    "EC"
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:

    print("\nERROR: Required columns missing:")
    for column in missing:
        print(" ", column)

    raise SystemExit(1)


# =============================================================================
# CLEAN ANNOTATIONS
# =============================================================================

for column in required:

    df[column] = df[column].apply(clean)


# =============================================================================
# ANNOTATION COUNTS
# =============================================================================

def count_annotations(value):

    value = clean(value)

    if not value:
        return 0

    # Most annotation fields are semicolon-separated.
    # This counts unique non-empty entries.
    parts = [
        x.strip()
        for x in value.split(";")
        if x.strip()
    ]

    return len(list(dict.fromkeys(parts)))


df["UniProt_Count"] = df["UniProt"].apply(
    count_annotations
)

df["GO_Count"] = df["GO"].apply(
    count_annotations
)

df["KEGG_Count"] = df["KEGG"].apply(
    count_annotations
)

df["EC_Count"] = df["EC"].apply(
    count_annotations
)


# =============================================================================
# ANNOTATION STATUS
# =============================================================================

def annotation_status(row):

    if (
        row["GO_Count"] > 0
        and row["KEGG_Count"] > 0
        and row["EC_Count"] > 0
    ):
        return "GO + KEGG + EC"

    if (
        row["GO_Count"] > 0
        and row["KEGG_Count"] > 0
    ):
        return "GO + KEGG"

    if row["GO_Count"] > 0:

        if row["EC_Count"] > 0:
            return "GO + EC"

        return "GO only"

    if row["KEGG_Count"] > 0:

        if row["EC_Count"] > 0:
            return "KEGG + EC"

        return "KEGG only"

    if row["EC_Count"] > 0:
        return "EC only"

    return "Functional description only"


df["Annotation_Status"] = df.apply(
    annotation_status,
    axis=1
)


# =============================================================================
# PATHWAY READINESS
# =============================================================================

def pathway_status(row):

    if (
        row["KEGG_Count"] > 0
        or row["EC_Count"] > 0
    ):
        return "Pathway-ready"

    if row["GO_Count"] > 0:
        return "GO-functional annotation"

    return "Functional annotation only"


df["Pathway_Readiness"] = df.apply(
    pathway_status,
    axis=1
)


# =============================================================================
# ANNOTATION COMPLETENESS
# =============================================================================

def completeness(row):

    fields = [
        row["UniProt_Count"] > 0,
        row["GO_Count"] > 0,
        row["KEGG_Count"] > 0,
        row["EC_Count"] > 0
    ]

    return round(
        100 * sum(fields) / len(fields),
        1
    )


df["Annotation_Completeness_Percent"] = df.apply(
    completeness,
    axis=1
)


# =============================================================================
# ORDER COLUMNS
# =============================================================================

ordered_columns = [
    "BUSCO_ID",
    "Functional_Description",
    "UniProt",
    "EC",
    "GO",
    "KEGG",
    "UniProt_Count",
    "GO_Count",
    "KEGG_Count",
    "EC_Count",
    "Annotation_Status",
    "Pathway_Readiness",
    "Annotation_Completeness_Percent"
]

# Keep only columns that actually exist
ordered_columns = [
    column
    for column in ordered_columns
    if column in df.columns
]

pathway_df = df[ordered_columns].copy()


# =============================================================================
# SAVE TABLE
# =============================================================================

pathway_df.to_csv(
    OUTPUT_TABLE,
    index=False
)


# =============================================================================
# SUMMARY
# =============================================================================

total = len(pathway_df)

uniprot_n = sum(
    pathway_df["UniProt_Count"] > 0
)

go_n = sum(
    pathway_df["GO_Count"] > 0
)

kegg_n = sum(
    pathway_df["KEGG_Count"] > 0
)

ec_n = sum(
    pathway_df["EC_Count"] > 0
)

pathway_ready_n = sum(
    pathway_df["Pathway_Readiness"]
    == "Pathway-ready"
)


# =============================================================================
# PRINT BUSCO SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("TOP-20 PATHWAY-READY SUMMARY")
print("=" * 80)

print(
    f"\nTotal candidate BUSCOs : {total}"
)

print(
    f"UniProt annotated       : {uniprot_n}"
)

print(
    f"GO annotated            : {go_n}"
)

print(
    f"KEGG annotated          : {kegg_n}"
)

print(
    f"EC annotated            : {ec_n}"
)

print(
    f"Pathway-ready candidates: {pathway_ready_n}"
)


# =============================================================================
# PRINT INDIVIDUAL CANDIDATES
# =============================================================================

print("\n" + "=" * 80)
print("BUSCO → FUNCTION → PATHWAY ANNOTATION")
print("=" * 80)

for _, row in pathway_df.iterrows():

    print("\n" + "-" * 75)

    print(
        "BUSCO:",
        row["BUSCO_ID"]
    )

    print(
        "Function:",
        row["Functional_Description"]
        or "None"
    )

    print(
        "UniProt:",
        row["UniProt"]
        or "None"
    )

    print(
        "EC:",
        row["EC"]
        or "None"
    )

    print(
        "GO:",
        row["GO"]
        or "None"
    )

    print(
        "KEGG:",
        row["KEGG"]
        or "None"
    )

    print(
        "Status:",
        row["Annotation_Status"]
    )

    print(
        "Pathway readiness:",
        row["Pathway_Readiness"]
    )


# =============================================================================
# WRITE REPORT
# =============================================================================

report = []

report.append(
    "CARBON BREADTH TOP-20 PATHWAY ANALYSIS SUMMARY"
)

report.append("=" * 80)

report.append(
    f"Total candidate BUSCOs: {total}"
)

report.append(
    f"UniProt annotated: {uniprot_n}"
)

report.append(
    f"GO annotated: {go_n}"
)

report.append(
    f"KEGG annotated: {kegg_n}"
)

report.append(
    f"EC annotated: {ec_n}"
)

report.append(
    f"Pathway-ready candidates: {pathway_ready_n}"
)

report.append("")

report.append(
    "BUSCO-level annotation summary"
)

report.append("=" * 80)


for _, row in pathway_df.iterrows():

    report.append("")

    report.append(
        f"BUSCO: {row['BUSCO_ID']}"
    )

    report.append(
        f"Function: "
        f"{row['Functional_Description'] or 'None'}"
    )

    report.append(
        f"UniProt: "
        f"{row['UniProt'] or 'None'}"
    )

    report.append(
        f"EC: "
        f"{row['EC'] or 'None'}"
    )

    report.append(
        f"GO: "
        f"{row['GO'] or 'None'}"
    )

    report.append(
        f"KEGG: "
        f"{row['KEGG'] or 'None'}"
    )

    report.append(
        f"Annotation status: "
        f"{row['Annotation_Status']}"
    )

    report.append(
        f"Pathway readiness: "
        f"{row['Pathway_Readiness']}"
    )

    report.append(
        f"Annotation completeness: "
        f"{row['Annotation_Completeness_Percent']}%"
    )


with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "\n".join(report)
    )


# =============================================================================
# OUTPUTS
# =============================================================================

print("\n" + "=" * 80)
print("PATHWAY TABLE COMPLETE")
print("=" * 80)

print("\nPathway-ready table:")
print(OUTPUT_TABLE)

print("\nSummary report:")
print(OUTPUT_REPORT)

print("\n" + "=" * 80)
print("NEXT STEP")
print("=" * 80)

print(
    "\nThe next step is to map the available KEGG/EC annotations"
)

print(
    "to biological pathways and summarize which pathways contain"
)

print(
    "the 20 Carbon Breadth candidate BUSCOs."
)

print("\nDone.")