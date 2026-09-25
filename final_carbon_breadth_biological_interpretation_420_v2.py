# final_carbon_breadth_biological_interpretation_420_v2.py

import os
import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\BUSCO20_annotation"

# Corrected pathway-analysis directory
CORRECTED_DIR = os.path.join(
    BASE,
    "FINAL_CARBON_BREADTH_INTERPRETATION_NAMED_420",
    "analysis_corrected"
)

# Final output directory
OUTPUT_DIR = os.path.join(
    BASE,
    "FINAL_CARBON_BREADTH_BIOLOGICAL_INTERPRETATION_420_V2"
)

TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Corrected input
INPUT_FILE = os.path.join(
    CORRECTED_DIR,
    "Carbon_Breadth_BUSCO_named_pathway_summary_CORRECTED_420.csv"
)

FREQUENCY_FILE = os.path.join(
    CORRECTED_DIR,
    "Carbon_Breadth_named_pathway_frequency_CORRECTED_420.csv"
)

RECORDS_FILE = os.path.join(
    CORRECTED_DIR,
    "Carbon_Breadth_named_pathway_records_CORRECTED_420.csv"
)

# =============================================================================
# OUTPUT FILES
# =============================================================================

BUSCO_SUMMARY_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_BUSCO_pathway_summary_420_v2.csv"
)

PATHWAY_FREQUENCY_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_pathway_frequency_420_v2.csv"
)

PATHWAY_RECORDS_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_pathway_records_420_v2.csv"
)

COVERAGE_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_annotation_coverage_420_v2.csv"
)

FUNCTIONAL_GROUP_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_functional_pathway_groups_420_v2.csv"
)

REPORT_OUT = os.path.join(
    REPORT_DIR,
    "final_carbon_breadth_biological_interpretation_420_v2.txt"
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_pathways(value):
    """
    Split KEGG pathway names/IDs while preserving the mapping.
    Handles common separators used in previous outputs.
    """
    value = clean_text(value)

    if not value:
        return []

    # Primary separator used in these outputs
    if ";" in value:
        parts = value.split(";")
    elif "|" in value:
        parts = value.split("|")
    else:
        parts = [value]

    return [
        x.strip()
        for x in parts
        if x.strip() and x.strip().lower() not in ["nan", "none"]
    ]


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("FINAL CARBON BREADTH BIOLOGICAL INTERPRETATION V2")
print("=" * 80)

print("\nCorrected input:")
print(INPUT_FILE)

# =============================================================================
# CHECK INPUT
# =============================================================================

if not os.path.exists(INPUT_FILE):
    print("\nERROR: Corrected pathway summary was not found.")
    print(INPUT_FILE)
    raise SystemExit(1)

# =============================================================================
# LOAD BUSCO SUMMARY
# =============================================================================

df = pd.read_csv(INPUT_FILE)

print("\nInput loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for col in df.columns:
    print(f"  {col}")

# =============================================================================
# IDENTIFY COLUMNS
# =============================================================================

busco_col = None
pathway_count_col = None
pathway_id_col = None
pathway_name_col = None
gene_count_col = None

for col in df.columns:

    c = col.lower()

    if "busco" in c:
        busco_col = col

    if "kegg_gene_count" in c:
        gene_count_col = col

    if "kegg_pathway_count" in c:
        pathway_count_col = col

    if "kegg_pathway_ids" in c:
        pathway_id_col = col

    if "kegg_pathway_names" in c:
        pathway_name_col = col

print("\nDetected columns:")
print("BUSCO column        :", busco_col)
print("KEGG gene count     :", gene_count_col)
print("KEGG pathway count  :", pathway_count_col)
print("KEGG pathway IDs    :", pathway_id_col)
print("KEGG pathway names  :", pathway_name_col)

if busco_col is None:
    raise SystemExit("ERROR: BUSCO_ID column not found.")

# =============================================================================
# NORMALIZE COUNTS
# =============================================================================

if gene_count_col:
    df[gene_count_col] = pd.to_numeric(
        df[gene_count_col],
        errors="coerce"
    ).fillna(0).astype(int)

if pathway_count_col:
    df[pathway_count_col] = pd.to_numeric(
        df[pathway_count_col],
        errors="coerce"
    ).fillna(0).astype(int)

# =============================================================================
# VALIDATE PATHWAY MAPPING
# =============================================================================

print("\n" + "=" * 80)
print("VALIDATING PATHWAY COVERAGE")
print("=" * 80)

total_candidates = len(df)

if pathway_count_col:
    mapped_mask = df[pathway_count_col] > 0
else:
    mapped_mask = pd.Series(False, index=df.index)

mapped_candidates = int(mapped_mask.sum())
unmapped_candidates = total_candidates - mapped_candidates

print(f"Total candidate BUSCOs : {total_candidates}")
print(f"Pathway-mapped         : {mapped_candidates}")
print(f"Not pathway-mapped     : {unmapped_candidates}")

# =============================================================================
# BUILD BUSCO SUMMARY
# =============================================================================

summary_columns = [
    busco_col
]

for col in [
    "Functional_Description",
    gene_count_col,
    pathway_count_col,
    pathway_id_col,
    pathway_name_col,
    "Pathway_Mapped"
]:
    if col and col in df.columns and col not in summary_columns:
        summary_columns.append(col)

busco_summary = df[summary_columns].copy()

# Rename BUSCO column consistently
if busco_col != "BUSCO_ID":
    busco_summary.rename(
        columns={busco_col: "BUSCO_ID"},
        inplace=True
    )

busco_summary["Pathway_Mapped"] = (
    busco_summary[pathway_count_col] > 0
    if pathway_count_col in busco_summary.columns
    else False
)

# =============================================================================
# SAVE BUSCO SUMMARY
# =============================================================================

busco_summary.to_csv(
    BUSCO_SUMMARY_OUT,
    index=False
)

# =============================================================================
# EXTRACT PATHWAY RECORDS
# =============================================================================

print("\n" + "=" * 80)
print("EXTRACTING NAMED PATHWAY RECORDS")
print("=" * 80)

records = []

for _, row in df.iterrows():

    busco = clean_text(row[busco_col])

    pathway_ids = split_pathways(
        row[pathway_id_col]
    ) if pathway_id_col else []

    pathway_names = split_pathways(
        row[pathway_name_col]
    ) if pathway_name_col else []

    # Pair IDs and names where possible
    max_len = max(
        len(pathway_ids),
        len(pathway_names)
    )

    for i in range(max_len):

        pathway_id = (
            pathway_ids[i]
            if i < len(pathway_ids)
            else ""
        )

        pathway_name = (
            pathway_names[i]
            if i < len(pathway_names)
            else ""
        )

        if pathway_id or pathway_name:

            records.append({
                "BUSCO_ID": busco,
                "KEGG_Pathway_ID": pathway_id,
                "KEGG_Pathway_Name": pathway_name
            })

pathway_records = pd.DataFrame(records)

# =============================================================================
# REMOVE DUPLICATES
# =============================================================================

if not pathway_records.empty:

    pathway_records = pathway_records.drop_duplicates(
        subset=[
            "BUSCO_ID",
            "KEGG_Pathway_ID",
            "KEGG_Pathway_Name"
        ]
    ).reset_index(drop=True)

# =============================================================================
# SAVE PATHWAY RECORDS
# =============================================================================

pathway_records.to_csv(
    PATHWAY_RECORDS_OUT,
    index=False
)

# =============================================================================
# PATHWAY FREQUENCY
# =============================================================================

print("\n" + "=" * 80)
print("CALCULATING PATHWAY FREQUENCY")
print("=" * 80)

if not pathway_records.empty:

    pathway_frequency = (
        pathway_records
        .groupby(
            ["KEGG_Pathway_ID", "KEGG_Pathway_Name"],
            dropna=False
        )
        .agg(
            BUSCO_Count=("BUSCO_ID", "nunique")
        )
        .reset_index()
        .sort_values(
            ["BUSCO_Count", "KEGG_Pathway_ID"],
            ascending=[False, True]
        )
    )

else:

    pathway_frequency = pd.DataFrame(
        columns=[
            "KEGG_Pathway_ID",
            "KEGG_Pathway_Name",
            "BUSCO_Count"
        ]
    )

pathway_frequency.to_csv(
    PATHWAY_FREQUENCY_OUT,
    index=False
)

# =============================================================================
# COVERAGE TABLE
# =============================================================================

coverage = pd.DataFrame({
    "Metric": [
        "Total candidate BUSCOs",
        "BUSCOs with KEGG annotations",
        "BUSCOs mapped to pathways",
        "BUSCOs not mapped to pathways",
        "Unique KEGG pathways",
        "Pathway mapping records"
    ],
    "Count": [
        total_candidates,

        int(
            (df[gene_count_col] > 0).sum()
        ) if gene_count_col else 0,

        mapped_candidates,

        unmapped_candidates,

        pathway_frequency.shape[0],

        pathway_records.shape[0]
    ]
})

coverage.to_csv(
    COVERAGE_OUT,
    index=False
)

# =============================================================================
# FUNCTIONAL / BIOLOGICAL GROUPING
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING BIOLOGICAL PATHWAY GROUPS")
print("=" * 80)

groups = []

# Broad categories based ONLY on pathway names returned in the analysis.
# These are descriptive groupings, not rankings.

for _, row in pathway_records.iterrows():

    name = clean_text(
        row["KEGG_Pathway_Name"]
    )

    name_lower = name.lower()

    if any(
        x in name_lower
        for x in [
            "carbon",
            "carbon fixation",
            "citrate cycle",
            "glycolysis",
            "gluconeogenesis",
            "metabolism"
        ]
    ):
        category = "Central metabolism"

    elif any(
        x in name_lower
        for x in [
            "amino acid",
            "alanine",
            "glutamate",
            "glycine",
            "serine",
            "valine",
            "leucine",
            "isoleucine"
        ]
    ):
        category = "Amino acid metabolism"

    elif any(
        x in name_lower
        for x in [
            "lipid",
            "fatty acid",
            "glycerolipid",
            "glycerophospholipid",
            "sphingolipid"
        ]
    ):
        category = "Lipid metabolism"

    elif any(
        x in name_lower
        for x in [
            "nucleotide",
            "purine",
            "pyrimidine"
        ]
    ):
        category = "Nucleotide metabolism"

    elif any(
        x in name_lower
        for x in [
            "rna",
            "transcription",
            "translation",
            "ribosome",
            "spliceosome"
        ]
    ):
        category = "Information processing"

    elif any(
        x in name_lower
        for x in [
            "protein",
            "proteasome",
            "protein processing"
        ]
    ):
        category = "Protein processing"

    elif any(
        x in name_lower
        for x in [
            "transport",
            "vesicle",
            "endocytosis",
            "membrane"
        ]
    ):
        category = "Transport and membrane processes"

    else:
        category = "Other biological processes"

    groups.append({
        "BUSCO_ID": row["BUSCO_ID"],
        "KEGG_Pathway_ID": row["KEGG_Pathway_ID"],
        "KEGG_Pathway_Name": row["KEGG_Pathway_Name"],
        "Functional_Category": category
    })

functional_groups = pd.DataFrame(groups)

functional_groups.to_csv(
    FUNCTIONAL_GROUP_OUT,
    index=False
)

# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "=" * 80)
print("FINAL CARBON BREADTH BIOLOGICAL INTERPRETATION")
print("=" * 80)

print(f"Total candidate BUSCOs       : {total_candidates}")
print(f"Pathway-mapped BUSCOs        : {mapped_candidates}")
print(f"Not pathway-mapped           : {unmapped_candidates}")
print(f"Unique KEGG pathways         : {pathway_frequency.shape[0]}")
print(f"Pathway mapping records      : {pathway_records.shape[0]}")

# =============================================================================
# REPORT CONTENT
# =============================================================================

report_lines = []

report_lines.append("=" * 80)
report_lines.append(
    "FINAL CARBON BREADTH BIOLOGICAL INTERPRETATION V2"
)
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append(
    "This report summarizes the functional and KEGG pathway "
    "annotations associated with the 20 Carbon Breadth candidate BUSCOs."
)

report_lines.append("")

report_lines.append("PATHWAY COVERAGE")
report_lines.append("-" * 80)

report_lines.append(
    f"Total candidate BUSCOs       : {total_candidates}"
)

report_lines.append(
    f"BUSCOs with KEGG annotations : "
    f"{int((df[gene_count_col] > 0).sum()) if gene_count_col else 0}"
)

report_lines.append(
    f"BUSCOs mapped to pathways    : {mapped_candidates}"
)

report_lines.append(
    f"BUSCOs not pathway-mapped    : {unmapped_candidates}"
)

report_lines.append(
    f"Unique KEGG pathways        : {pathway_frequency.shape[0]}"
)

report_lines.append(
    f"Pathway mapping records      : {pathway_records.shape[0]}"
)

report_lines.append("")

report_lines.append("BUSCO-LEVEL PATHWAY SUMMARY")
report_lines.append("-" * 80)

for _, row in busco_summary.iterrows():

    busco = row["BUSCO_ID"]

    description = clean_text(
        row["Functional_Description"]
    ) if "Functional_Description" in row else ""

    gene_count = (
        row[gene_count_col]
        if gene_count_col in row
        else 0
    )

    pathway_count = (
        row[pathway_count_col]
        if pathway_count_col in row
        else 0
    )

    mapped = (
        "Yes"
        if pathway_count > 0
        else "No"
    )

    report_lines.append(
        f"{busco} | {description} | "
        f"KEGG genes={gene_count} | "
        f"Pathways={pathway_count} | "
        f"Mapped={mapped}"
    )

report_lines.append("")

report_lines.append("MOST REPRESENTED PATHWAYS")
report_lines.append("-" * 80)

for _, row in pathway_frequency.head(20).iterrows():

    report_lines.append(
        f"{row['KEGG_Pathway_ID']} | "
        f"{row['KEGG_Pathway_Name']} | "
        f"BUSCO count={row['BUSCO_Count']}"
    )

report_lines.append("")

report_lines.append("IMPORTANT INTERPRETATION")
report_lines.append("-" * 80)

report_lines.append(
    "The pathway analysis is based on the corrected KEGG gene-to-pathway "
    "mapping generated for the 20 Carbon Breadth candidate BUSCOs."
)

report_lines.append(
    "Only candidates with successfully retrieved KEGG pathway mappings "
    "are counted as pathway-mapped."
)

report_lines.append(
    "The 11 candidates without pathway mappings should not be interpreted "
    "as biologically irrelevant; they represent candidates for which "
    "a KEGG pathway association was not recovered in the current "
    "annotation workflow."
)

report_lines.append(
    "Pathway counts represent annotation coverage and should not be "
    "interpreted as enrichment or statistical significance."
)

report_lines.append("")

report_lines.append("OUTPUT FILES")
report_lines.append("-" * 80)

report_lines.append(BUSCO_SUMMARY_OUT)
report_lines.append(PATHWAY_FREQUENCY_OUT)
report_lines.append(PATHWAY_RECORDS_OUT)
report_lines.append(COVERAGE_OUT)
report_lines.append(FUNCTIONAL_GROUP_OUT)
report_lines.append(REPORT_OUT)

with open(
    REPORT_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(report_lines))

# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("FINAL BIOLOGICAL INTERPRETATION COMPLETE")
print("=" * 80)

print(f"\nTotal candidates       : {total_candidates}")
print(f"Pathway-mapped         : {mapped_candidates}")
print(f"Not pathway-mapped     : {unmapped_candidates}")
print(
    f"Unique KEGG pathways   : "
    f"{pathway_frequency.shape[0]}"
)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nTABLES:")
print(" ", BUSCO_SUMMARY_OUT)
print(" ", PATHWAY_FREQUENCY_OUT)
print(" ", PATHWAY_RECORDS_OUT)
print(" ", COVERAGE_OUT)
print(" ", FUNCTIONAL_GROUP_OUT)

print("\nREPORT:")
print(" ", REPORT_OUT)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)