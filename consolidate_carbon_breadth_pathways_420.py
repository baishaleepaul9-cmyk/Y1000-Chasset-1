import os
import re
import pandas as pd
from collections import defaultdict

# =============================================================================
# CARBON BREADTH — ORGANISM-INDEPENDENT KEGG PATHWAY CONSOLIDATION
# =============================================================================

BASE = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\BUSCO20_annotation"

INPUT_DIR = os.path.join(
    BASE,
    "FINAL_CARBON_BREADTH_BIOLOGICAL_INTERPRETATION_420_V2",
    "tables"
)

INPUT_FILE = os.path.join(
    INPUT_DIR,
    "Carbon_Breadth_final_pathway_records_420_v2.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "FINAL_CARBON_BREADTH_BIOLOGICAL_INTERPRETATION_420_V2_CONSOLIDATED"
)

TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# =============================================================================
# LOAD
# =============================================================================

print("=" * 80)
print("CARBON BREADTH KEGG PATHWAY CONSOLIDATION")
print("=" * 80)

print("\nInput:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    print("\nERROR: Input file not found.")
    print("Expected:")
    print(INPUT_FILE)
    raise SystemExit(1)

df = pd.read_csv(INPUT_FILE)

print("\nInput loaded successfully.")
print("Rows:", len(df))
print("Columns:")
for c in df.columns:
    print(" ", c)

# =============================================================================
# FIND IMPORTANT COLUMNS
# =============================================================================

def find_column(columns, candidates):
    for candidate in candidates:
        for col in columns:
            if col.lower() == candidate.lower():
                return col
    return None

busco_col = find_column(
    df.columns,
    ["BUSCO_ID", "Busco_ID", "BUSCO"]
)

pathway_id_col = find_column(
    df.columns,
    [
        "KEGG_Pathway_ID",
        "KEGG_Pathway",
        "Pathway_ID",
        "Pathway"
    ]
)

pathway_name_col = find_column(
    df.columns,
    [
        "KEGG_Pathway_Name",
        "KEGG_Pathway_Names",
        "Pathway_Name",
        "Pathway_Names"
    ]
)

print("\nDetected columns:")
print("BUSCO column       :", busco_col)
print("Pathway ID column  :", pathway_id_col)
print("Pathway name column:", pathway_name_col)

if busco_col is None or pathway_name_col is None:
    print("\nERROR: Required BUSCO/pathway columns were not found.")
    raise SystemExit(1)

# =============================================================================
# NORMALIZE PATHWAY NAME
# =============================================================================

def normalize_pathway_name(name):
    """
    Remove organism-specific suffixes while retaining the actual
    biological pathway name.
    """

    if pd.isna(name):
        return ""

    name = str(name).strip()

    if not name:
        return ""

    # ---------------------------------------------------------
    # KEGG commonly uses:
    #
    # "Pathway name - Organism"
    #
    # Remove the organism-specific suffix.
    # ---------------------------------------------------------

    if " - " in name:
        name = name.split(" - ")[0].strip()

    # ---------------------------------------------------------
    # Remove duplicate whitespace
    # ---------------------------------------------------------

    name = re.sub(r"\s+", " ", name)

    return name


df["Normalized_Pathway_Name"] = df[pathway_name_col].apply(
    normalize_pathway_name
)

# =============================================================================
# NORMALIZE KEGG PATHWAY IDs
# =============================================================================

if pathway_id_col is not None:

    def normalize_pathway_id(value):
        if pd.isna(value):
            return ""
        return str(value).strip()

    df["Normalized_Pathway_ID"] = df[pathway_id_col].apply(
        normalize_pathway_id
    )

else:
    df["Normalized_Pathway_ID"] = ""

# =============================================================================
# REMOVE EMPTY PATHWAYS
# =============================================================================

df = df[
    df["Normalized_Pathway_Name"].astype(str).str.strip() != ""
].copy()

print("\nValid pathway records:", len(df))

# =============================================================================
# UNIQUE ORGANISM-SPECIFIC → NORMALIZED MAPPING
# =============================================================================

mapping = (
    df[
        [
            busco_col,
            pathway_id_col if pathway_id_col else "Normalized_Pathway_ID",
            pathway_name_col,
            "Normalized_Pathway_Name"
        ]
    ]
    .drop_duplicates()
    .copy()
)

mapping.columns = [
    "BUSCO_ID",
    "KEGG_Pathway_ID",
    "Original_Pathway_Name",
    "Normalized_Pathway_Name"
]

mapping_file = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_pathway_normalization_mapping_420.csv"
)

mapping.to_csv(mapping_file, index=False)

# =============================================================================
# BUSCO × NORMALIZED PATHWAY MATRIX
# =============================================================================

busco_pathway = (
    mapping[
        ["BUSCO_ID", "Normalized_Pathway_Name"]
    ]
    .drop_duplicates()
)

matrix = pd.crosstab(
    busco_pathway["BUSCO_ID"],
    busco_pathway["Normalized_Pathway_Name"]
)

matrix_file = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_normalized_pathway_matrix_420.csv"
)

matrix.to_csv(matrix_file)

# =============================================================================
# PATHWAY FREQUENCY
# =============================================================================

pathway_frequency = (
    busco_pathway
    .groupby("Normalized_Pathway_Name")["BUSCO_ID"]
    .nunique()
    .reset_index()
)

pathway_frequency.columns = [
    "Normalized_Pathway_Name",
    "BUSCO_Count"
]

pathway_frequency = pathway_frequency.sort_values(
    ["BUSCO_Count", "Normalized_Pathway_Name"],
    ascending=[False, True]
)

frequency_file = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_normalized_pathway_frequency_420.csv"
)

pathway_frequency.to_csv(
    frequency_file,
    index=False
)

# =============================================================================
# PATHWAY → BUSCO RECORDS
# =============================================================================

records = (
    busco_pathway
    .groupby("Normalized_Pathway_Name")["BUSCO_ID"]
    .apply(lambda x: "; ".join(sorted(set(x))))
    .reset_index()
)

records["BUSCO_Count"] = records["BUSCO_ID"].apply(
    lambda x: len([i for i in x.split("; ") if i])
)

records = records[
    [
        "Normalized_Pathway_Name",
        "BUSCO_Count",
        "BUSCO_ID"
    ]
].sort_values(
    ["BUSCO_Count", "Normalized_Pathway_Name"],
    ascending=[False, True]
)

records_file = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_normalized_pathway_records_420.csv"
)

records.to_csv(records_file, index=False)

# =============================================================================
# ORGANISM-SPECIFIC PATHWAY COLLAPSE STATISTICS
# =============================================================================

original_unique = (
    mapping["Original_Pathway_Name"]
    .nunique()
)

normalized_unique = (
    mapping["Normalized_Pathway_Name"]
    .nunique()
)

# =============================================================================
# COVERAGE
# =============================================================================

total_buscos = df[busco_col].nunique()

mapped_buscos = (
    mapping["BUSCO_ID"]
    .nunique()
)

unmapped_buscos = total_buscos - mapped_buscos

# =============================================================================
# REPORT
# =============================================================================

report_file = os.path.join(
    REPORT_DIR,
    "Carbon_Breadth_normalized_pathway_consolidation_report_420.txt"
)

with open(report_file, "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("CARBON BREADTH ORGANISM-INDEPENDENT KEGG PATHWAY CONSOLIDATION\n")
    f.write("=" * 80 + "\n\n")

    f.write("INPUT\n")
    f.write("-" * 80 + "\n")
    f.write(INPUT_FILE + "\n\n")

    f.write("PATHWAY COVERAGE\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total candidate BUSCOs       : {total_buscos}\n")
    f.write(f"BUSCOs mapped to pathways    : {mapped_buscos}\n")
    f.write(f"BUSCOs without pathways      : {unmapped_buscos}\n\n")

    f.write("PATHWAY CONSOLIDATION\n")
    f.write("-" * 80 + "\n")
    f.write(
        f"Original organism-specific pathway names : "
        f"{original_unique}\n"
    )
    f.write(
        f"Normalized biological pathway concepts   : "
        f"{normalized_unique}\n"
    )

    f.write("\nMOST REPRESENTED NORMALIZED PATHWAYS\n")
    f.write("-" * 80 + "\n")

    for _, row in pathway_frequency.iterrows():

        f.write(
            f"{row['Normalized_Pathway_Name']} | "
            f"BUSCO count={row['BUSCO_Count']}\n"
        )

    f.write("\n\nINTERPRETATION NOTE\n")
    f.write("-" * 80 + "\n")
    f.write(
        "Organism-specific KEGG pathway labels were consolidated by "
        "removing the organism suffix from pathway names. This prevents "
        "the same biological pathway from being counted as separate "
        "pathways solely because annotations were retrieved from "
        "different KEGG organisms.\n\n"
    )

    f.write(
        "Pathway frequency represents annotation coverage among the "
        "20 Carbon Breadth candidate BUSCOs. It is not a pathway "
        "enrichment test and should not be interpreted as statistical "
        "significance.\n"
    )

# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("PATHWAY CONSOLIDATION COMPLETE")
print("=" * 80)

print("\nPATHWAY COVERAGE")
print("-" * 80)
print("Total candidate BUSCOs       :", total_buscos)
print("BUSCOs mapped to pathways    :", mapped_buscos)
print("BUSCOs without pathways      :", unmapped_buscos)

print("\nPATHWAY CONSOLIDATION")
print("-" * 80)
print(
    "Original organism-specific pathways:",
    original_unique
)
print(
    "Normalized biological pathways    :",
    normalized_unique
)

print("\nMOST REPRESENTED NORMALIZED PATHWAYS")
print("-" * 80)

for _, row in pathway_frequency.head(20).iterrows():

    print(
        f"{row['Normalized_Pathway_Name']} | "
        f"BUSCO count={row['BUSCO_Count']}"
    )

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nTABLES:")
print(mapping_file)
print(frequency_file)
print(records_file)
print(matrix_file)

print("\nREPORT:")
print(report_file)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)