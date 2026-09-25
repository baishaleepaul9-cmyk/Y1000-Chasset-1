import os
import pandas as pd
from collections import defaultdict

# =============================================================================
# FINAL CARBON BREADTH BIOLOGICAL INTERPRETATION V3
# =============================================================================

BASE = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\BUSCO20_annotation"

INPUT_DIR = os.path.join(
    BASE,
    "FINAL_CARBON_BREADTH_BIOLOGICAL_INTERPRETATION_420_V2_CONSOLIDATED"
)

INPUT_TABLE_DIR = os.path.join(INPUT_DIR, "tables")

OUTPUT_DIR = os.path.join(
    BASE,
    "FINAL_CARBON_BREADTH_BIOLOGICAL_INTERPRETATION_420_V3"
)

TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# =============================================================================
# INPUT FILES
# =============================================================================

PATHWAY_MAPPING = os.path.join(
    INPUT_TABLE_DIR,
    "Carbon_Breadth_pathway_normalization_mapping_420.csv"
)

PATHWAY_FREQUENCY = os.path.join(
    INPUT_TABLE_DIR,
    "Carbon_Breadth_normalized_pathway_frequency_420.csv"
)

PATHWAY_RECORDS = os.path.join(
    INPUT_TABLE_DIR,
    "Carbon_Breadth_normalized_pathway_records_420.csv"
)

BUSCO_MATRIX = os.path.join(
    INPUT_TABLE_DIR,
    "Carbon_Breadth_BUSCO_normalized_pathway_matrix_420.csv"
)

# =============================================================================
# OUTPUT FILES
# =============================================================================

FINAL_BUSCO_SUMMARY = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_BUSCO_biological_summary_420_v3.csv"
)

FINAL_PATHWAY_SUMMARY = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_normalized_pathway_summary_420_v3.csv"
)

FINAL_THEME_SUMMARY = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_biological_theme_summary_420_v3.csv"
)

FINAL_BUSCO_MATRIX = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_BUSCO_pathway_matrix_420_v3.csv"
)

FINAL_COVERAGE = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_annotation_coverage_420_v3.csv"
)

FINAL_REPORT = os.path.join(
    REPORT_DIR,
    "final_carbon_breadth_biological_interpretation_420_v3.txt"
)

# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("FINAL CARBON BREADTH BIOLOGICAL INTERPRETATION V3")
print("=" * 80)

print(
    "\nThis report summarizes the consolidated biological pathway "
    "annotations associated with the 20 Carbon Breadth candidate BUSCOs."
)

print(
    "\nThe analysis is restricted to the original 20 Carbon Breadth candidates."
)

print(
    "No new KEGG retrieval or BUSCO selection is performed."
)

# =============================================================================
# CHECK INPUTS
# =============================================================================

print("\n" + "=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)

input_files = {
    "Pathway normalization mapping": PATHWAY_MAPPING,
    "Pathway frequency": PATHWAY_FREQUENCY,
    "Pathway records": PATHWAY_RECORDS,
    "BUSCO pathway matrix": BUSCO_MATRIX
}

for name, path in input_files.items():

    print(f"\n{name}:")
    print(path)

    if not os.path.exists(path):
        print("ERROR: File not found.")
        raise FileNotFoundError(path)

    print("OK")

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "=" * 80)
print("LOADING CONSOLIDATED DATA")
print("=" * 80)

mapping_df = pd.read_csv(PATHWAY_MAPPING)
frequency_df = pd.read_csv(PATHWAY_FREQUENCY)
records_df = pd.read_csv(PATHWAY_RECORDS)
matrix_df = pd.read_csv(BUSCO_MATRIX)

print("\nPathway normalization mapping shape:", mapping_df.shape)
print("Pathway frequency shape:", frequency_df.shape)
print("Pathway records shape:", records_df.shape)
print("BUSCO pathway matrix shape:", matrix_df.shape)

print("\nRecords columns:")
for c in records_df.columns:
    print(" ", c)

# =============================================================================
# COLUMN DETECTION
# =============================================================================

def find_column(df, candidates):

    lower_columns = {
        str(c).lower(): c
        for c in df.columns
    }

    # Exact match
    for candidate in candidates:

        if candidate.lower() in lower_columns:
            return lower_columns[candidate.lower()]

    # Partial match
    for column in df.columns:

        column_lower = str(column).lower()

        for candidate in candidates:

            if candidate.lower() in column_lower:
                return column

    return None


busco_col = find_column(
    records_df,
    [
        "BUSCO_ID",
        "BUSCO"
    ]
)

pathway_col = find_column(
    records_df,
    [
        "Normalized_Pathway",
        "Normalized pathway",
        "Pathway",
        "KEGG_Pathway_Name"
    ]
)

print("\n" + "=" * 80)
print("COLUMN DETECTION")
print("=" * 80)

print("BUSCO column   :", busco_col)
print("Pathway column :", pathway_col)

if busco_col is None:
    raise ValueError(
        "Could not identify the BUSCO column in the pathway records file."
    )

if pathway_col is None:
    raise ValueError(
        "Could not identify the normalized pathway column "
        "in the pathway records file."
    )

# =============================================================================
# CLEAN DATA
# =============================================================================

records_df[busco_col] = (
    records_df[busco_col]
    .astype(str)
    .str.strip()
)

records_df[pathway_col] = (
    records_df[pathway_col]
    .astype(str)
    .str.strip()
)

records_df = records_df[
    (records_df[busco_col] != "") &
    (records_df[busco_col].str.lower() != "nan") &
    (records_df[pathway_col] != "") &
    (records_df[pathway_col].str.lower() != "nan")
].copy()

# =============================================================================
# ORIGINAL 20 CARBON BREADTH BUSCO CANDIDATES
# =============================================================================

original_buscos = [
    "1679at4891",
    "2307at4891",
    "2471at4891",
    "3574at4891",
    "4322at4891",
    "5746at4891",
    "6711at4891",
    "9782at4891",
    "12468at4891",
    "12523at4891",
    "18913at4891",
    "22532at4891",
    "23379at4891",
    "24318at4891",
    "27915at4891",
    "28058at4891",
    "29457at4891",
    "33531at4891",
    "34052at4891",
    "36839at4891"
]

print("\n" + "=" * 80)
print("CARBON BREADTH CANDIDATES")
print("=" * 80)

print("Total candidates:", len(original_buscos))

# =============================================================================
# BUSCO → NORMALIZED PATHWAY SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("BUILDING BUSCO-LEVEL BIOLOGICAL SUMMARY")
print("=" * 80)

busco_rows = []

for busco in original_buscos:

    subset = records_df[
        records_df[busco_col] == busco
    ]

    pathways = sorted(
        set(
            subset[pathway_col]
            .dropna()
            .astype(str)
            .str.strip()
        )
    )

    pathways = [
        p
        for p in pathways
        if p and p.lower() != "nan"
    ]

    busco_rows.append(
        {
            "BUSCO_ID": busco,
            "Normalized_Pathway_Count": len(pathways),
            "Normalized_Pathways": "; ".join(pathways),
            "Pathway_Mapped": (
                "Yes"
                if pathways
                else "No"
            )
        }
    )

busco_summary = pd.DataFrame(busco_rows)

# =============================================================================
# NORMALIZED PATHWAY SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("BUILDING NORMALIZED PATHWAY SUMMARY")
print("=" * 80)

pathway_to_buscos = defaultdict(set)

for _, row in records_df.iterrows():

    pathway = str(row[pathway_col]).strip()
    busco = str(row[busco_col]).strip()

    if pathway and busco:
        pathway_to_buscos[pathway].add(busco)

pathway_rows = []

for pathway, buscos in sorted(pathway_to_buscos.items()):

    pathway_rows.append(
        {
            "Normalized_Pathway": pathway,
            "BUSCO_Count": len(buscos),
            "BUSCOs": "; ".join(sorted(buscos))
        }
    )

pathway_summary = pd.DataFrame(pathway_rows)

# =============================================================================
# BROAD BIOLOGICAL THEME CLASSIFICATION
# =============================================================================

print("\n" + "=" * 80)
print("ASSIGNING BROAD BIOLOGICAL THEMES")
print("=" * 80)


def assign_theme(pathway):

    p = str(pathway).lower()

    # -------------------------------------------------------------------------
    # METABOLISM
    # -------------------------------------------------------------------------

    if any(
        term in p
        for term in [
            "metabolism",
            "biosynthesis",
            "carbon metabolism",
            "lysine biosynthesis",
            "citrate cycle",
            "tca cycle",
            "glyoxylate",
            "2-oxocarboxylic"
        ]
    ):
        return "Metabolism"

    # -------------------------------------------------------------------------
    # RNA / GENE EXPRESSION
    # -------------------------------------------------------------------------

    if any(
        term in p
        for term in [
            "rna degradation",
            "mrna surveillance",
            "rna transport",
            "ribosome",
            "translation",
            "transcription"
        ]
    ):
        return "RNA and Gene Expression"

    # -------------------------------------------------------------------------
    # GENOME / CELL REGULATION
    # -------------------------------------------------------------------------

    if any(
        term in p
        for term in [
            "cell cycle",
            "chromatin",
            "proteasome"
        ]
    ):
        return "Genome Maintenance and Cell Regulation"

    # -------------------------------------------------------------------------
    # MEMBRANE / TRAFFICKING
    # -------------------------------------------------------------------------

    if any(
        term in p
        for term in [
            "endocytosis",
            "gpi-anchor",
            "vesicle",
            "membrane"
        ]
    ):
        return "Membrane Trafficking and Organization"

    # -------------------------------------------------------------------------
    # CYTOSKELETON
    # -------------------------------------------------------------------------

    if any(
        term in p
        for term in [
            "motor proteins",
            "cytoskeleton",
            "muscle cells"
        ]
    ):
        return "Cytoskeleton and Cellular Transport"

    # -------------------------------------------------------------------------
    # PEROXISOME
    # -------------------------------------------------------------------------

    if "peroxisome" in p:
        return "Organelle and Peroxisomal Functions"

    # -------------------------------------------------------------------------
    # SIGNALING
    # -------------------------------------------------------------------------

    if any(
        term in p
        for term in [
            "mapk",
            "signaling",
            "igsf cam"
        ]
    ):
        return "Cell Signaling"

    # -------------------------------------------------------------------------
    # HOST-PATHOGEN
    # -------------------------------------------------------------------------

    if "viral" in p:
        return "Host–Pathogen Interaction"

    # -------------------------------------------------------------------------
    # OTHER
    # -------------------------------------------------------------------------

    return "Other Cellular Processes"


pathway_summary["Biological_Theme"] = (
    pathway_summary["Normalized_Pathway"]
    .apply(assign_theme)
)

# =============================================================================
# BIOLOGICAL THEME SUMMARY
# =============================================================================

theme_rows = []

for theme, group in pathway_summary.groupby(
    "Biological_Theme"
):

    buscos = set()

    for value in group["BUSCOs"]:

        if pd.isna(value):
            continue

        for busco in str(value).split(";"):

            busco = busco.strip()

            if busco:
                buscos.add(busco)

    theme_rows.append(
        {
            "Biological_Theme": theme,
            "Normalized_Pathway_Count": len(group),
            "BUSCO_Count": len(buscos),
            "Pathways": "; ".join(
                sorted(
                    group["Normalized_Pathway"].tolist()
                )
            ),
            "BUSCOs": "; ".join(
                sorted(buscos)
            )
        }
    )

theme_summary = pd.DataFrame(theme_rows)

if not theme_summary.empty:

    theme_summary = theme_summary.sort_values(
        [
            "BUSCO_Count",
            "Normalized_Pathway_Count"
        ],
        ascending=False
    )

# =============================================================================
# BUSCO × NORMALIZED PATHWAY MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("BUILDING BUSCO × PATHWAY MATRIX")
print("=" * 80)

all_pathways = sorted(
    pathway_summary["Normalized_Pathway"]
    .dropna()
    .unique()
)

matrix_rows = []

for busco in original_buscos:

    row = {
        "BUSCO_ID": busco
    }

    busco_pathways = set(
        records_df.loc[
            records_df[busco_col] == busco,
            pathway_col
        ]
    )

    for pathway in all_pathways:

        row[pathway] = (
            1
            if pathway in busco_pathways
            else 0
        )

    matrix_rows.append(row)

final_matrix = pd.DataFrame(matrix_rows)

# =============================================================================
# COVERAGE
# =============================================================================

total_candidates = len(original_buscos)

mapped_candidates = int(
    (
        busco_summary["Pathway_Mapped"]
        == "Yes"
    ).sum()
)

unmapped_candidates = (
    total_candidates
    - mapped_candidates
)

unique_pathways = len(pathway_summary)

coverage_df = pd.DataFrame(
    [
        {
            "Metric": "Total Carbon Breadth candidates",
            "Value": total_candidates
        },
        {
            "Metric": "Pathway-mapped candidates",
            "Value": mapped_candidates
        },
        {
            "Metric": "Not pathway-mapped candidates",
            "Value": unmapped_candidates
        },
        {
            "Metric": "Normalized biological pathways",
            "Value": unique_pathways
        },
        {
            "Metric": "Pathway mapping records",
            "Value": len(records_df)
        }
    ]
)

# =============================================================================
# SAVE TABLES
# =============================================================================

print("\n" + "=" * 80)
print("SAVING FINAL TABLES")
print("=" * 80)

busco_summary.to_csv(
    FINAL_BUSCO_SUMMARY,
    index=False
)

pathway_summary.to_csv(
    FINAL_PATHWAY_SUMMARY,
    index=False
)

theme_summary.to_csv(
    FINAL_THEME_SUMMARY,
    index=False
)

final_matrix.to_csv(
    FINAL_BUSCO_MATRIX,
    index=False
)

coverage_df.to_csv(
    FINAL_COVERAGE,
    index=False
)

print("\nSaved:")
print(FINAL_BUSCO_SUMMARY)
print(FINAL_PATHWAY_SUMMARY)
print(FINAL_THEME_SUMMARY)
print(FINAL_BUSCO_MATRIX)
print(FINAL_COVERAGE)

# =============================================================================
# REPORT
# =============================================================================

report_lines = []

report_lines.append(
    "=" * 80
)

report_lines.append(
    "FINAL CARBON BREADTH BIOLOGICAL INTERPRETATION V3"
)

report_lines.append(
    "=" * 80
)

report_lines.append("")

report_lines.append(
    "This report summarizes the consolidated biological "
    "pathway annotations associated with the 20 Carbon "
    "Breadth candidate BUSCOs."
)

report_lines.append("")

report_lines.append("ANALYSIS SCOPE")

report_lines.append("-" * 80)

report_lines.append(
    "The analysis is restricted to the original 20 "
    "Carbon Breadth candidate BUSCOs."
)

report_lines.append(
    "The pathway annotations originate from the corrected "
    "KEGG gene-to-pathway mapping followed by pathway "
    "normalization."
)

report_lines.append(
    "No new KEGG retrieval was performed during this V3 analysis."
)

report_lines.append("")

report_lines.append("PATHWAY COVERAGE")

report_lines.append("-" * 80)

report_lines.append(
    f"Total candidate BUSCOs        : {total_candidates}"
)

report_lines.append(
    f"Pathway-mapped candidates     : {mapped_candidates}"
)

report_lines.append(
    f"Not pathway-mapped            : {unmapped_candidates}"
)

report_lines.append(
    f"Normalized biological pathways: {unique_pathways}"
)

report_lines.append(
    f"Pathway mapping records       : {len(records_df)}"
)

report_lines.append("")

report_lines.append("BUSCO-LEVEL BIOLOGICAL SUMMARY")

report_lines.append("-" * 80)

for _, row in busco_summary.iterrows():

    pathways = row["Normalized_Pathways"]

    if not pathways:
        pathways = (
            "No normalized pathway recovered"
        )

    report_lines.append(
        f"{row['BUSCO_ID']} | "
        f"Pathways={row['Normalized_Pathway_Count']} | "
        f"{pathways}"
    )

report_lines.append("")

report_lines.append("NORMALIZED PATHWAY SUMMARY")

report_lines.append("-" * 80)

for _, row in pathway_summary.iterrows():

    report_lines.append(
        f"{row['Normalized_Pathway']} | "
        f"BUSCO count={row['BUSCO_Count']} | "
        f"BUSCOs={row['BUSCOs']} | "
        f"Theme={row['Biological_Theme']}"
    )

report_lines.append("")

report_lines.append("BROAD BIOLOGICAL THEMES")

report_lines.append("-" * 80)

for _, row in theme_summary.iterrows():

    report_lines.append(
        f"{row['Biological_Theme']} | "
        f"Pathways={row['Normalized_Pathway_Count']} | "
        f"BUSCO count={row['BUSCO_Count']}"
    )

report_lines.append("")

report_lines.append("INTERPRETATION")

report_lines.append("-" * 80)

report_lines.append(
    "The Carbon Breadth candidate set contains functions "
    "distributed across metabolic, cellular, regulatory, "
    "membrane-trafficking, cytoskeletal, organellar, and "
    "other cellular processes represented by the recovered "
    "normalized KEGG pathways."
)

report_lines.append(
    "The pathway normalization step collapses organism-specific "
    "KEGG records into shared biological pathway concepts."
)

report_lines.append(
    "This allows the 20 candidate BUSCOs to be interpreted "
    "at the level of broader biological functions rather than "
    "individual organism-specific KEGG pathway records."
)

report_lines.append(
    "Pathway frequency represents annotation coverage and "
    "should not be interpreted as statistical enrichment."
)

report_lines.append(
    "Candidates without recovered pathway annotations should "
    "not be interpreted as biologically irrelevant. Their "
    "absence reflects the current annotation and pathway "
    "retrieval coverage."
)

report_lines.append(
    "The pathway assignments do not by themselves establish "
    "causal involvement in carbon breadth."
)

report_lines.append("")

report_lines.append("OUTPUT FILES")

report_lines.append("-" * 80)

report_lines.append(FINAL_BUSCO_SUMMARY)
report_lines.append(FINAL_PATHWAY_SUMMARY)
report_lines.append(FINAL_THEME_SUMMARY)
report_lines.append(FINAL_BUSCO_MATRIX)
report_lines.append(FINAL_COVERAGE)

with open(
    FINAL_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report_lines)
    )

# =============================================================================
# FINAL CONSOLE OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("FINAL CARBON BREADTH BIOLOGICAL INTERPRETATION V3 COMPLETE")
print("=" * 80)

print(
    f"\nTotal candidates       : {total_candidates}"
)

print(
    f"Pathway-mapped         : {mapped_candidates}"
)

print(
    f"Not pathway-mapped     : {unmapped_candidates}"
)

print(
    f"Normalized pathways    : {unique_pathways}"
)

print(
    f"Pathway records        : {len(records_df)}"
)

print("\n" + "=" * 80)
print("BIOLOGICAL THEMES")
print("=" * 80)

for _, row in theme_summary.iterrows():

    print(
        f"{row['Biological_Theme']} | "
        f"Pathways={row['Normalized_Pathway_Count']} | "
        f"BUSCOs={row['BUSCO_Count']}"
    )

print("\n" + "=" * 80)
print("OUTPUT DIRECTORY")
print("=" * 80)

print(OUTPUT_DIR)

print("\nTABLES:")
print(FINAL_BUSCO_SUMMARY)
print(FINAL_PATHWAY_SUMMARY)
print(FINAL_THEME_SUMMARY)
print(FINAL_BUSCO_MATRIX)
print(FINAL_COVERAGE)

print("\nREPORT:")
print(FINAL_REPORT)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)