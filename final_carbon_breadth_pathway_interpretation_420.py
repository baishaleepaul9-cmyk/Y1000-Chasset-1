import os
import pandas as pd
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = (
    r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset"
    r"\phylogeny_aware_ml\phylogeny_cv\model_training"
    r"\feature_interpretation_420\functional_annotation_420"
    r"\BUSCO20_annotation"
)

# -------------------------------------------------------------------------
# ACTUAL RESOLVED PATHWAY FILES FROM YOUR PREVIOUS STEP
# -------------------------------------------------------------------------

RESOLVED_DIR = os.path.join(
    BASE_DIR,
    "FINAL_CARBON_BREADTH_ANNOTATION_420",
    "pathway_name_resolution_420"
)

DETAILED_MAPPING_FILE = os.path.join(
    RESOLVED_DIR,
    "Carbon_Breadth_top20_KEGG_gene_pathway_mapping_named_420.csv"
)

BUSCO_SUMMARY_FILE = os.path.join(
    RESOLVED_DIR,
    "Carbon_Breadth_top20_BUSCO_pathway_summary_named_420.csv"
)

PATHWAY_SUMMARY_FILE = os.path.join(
    RESOLVED_DIR,
    "Carbon_Breadth_top20_unique_pathways_named_420.csv"
)

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "FINAL_CARBON_BREADTH_INTERPRETATION_NAMED_420"
)

TABLE_DIR = os.path.join(
    OUTPUT_DIR,
    "tables"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("FINAL CARBON BREADTH PATHWAY INTERPRETATION")
print("=" * 80)


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

print("\nChecking resolved pathway files...")

required_files = [
    DETAILED_MAPPING_FILE,
    BUSCO_SUMMARY_FILE,
    PATHWAY_SUMMARY_FILE
]

for f in required_files:

    print("\n", f)

    if not os.path.exists(f):
        print("ERROR: FILE NOT FOUND")
        raise SystemExit(1)

    print("OK")


# =============================================================================
# LOAD DETAILED PATHWAY MAPPING
# =============================================================================

print("\n" + "=" * 80)
print("LOADING RESOLVED KEGG PATHWAY MAPPING")
print("=" * 80)

df = pd.read_csv(
    DETAILED_MAPPING_FILE
)

print("\nInput shape:")
print(df.shape)

print("\nColumns:")

for col in df.columns:
    print(" ", col)


# =============================================================================
# IDENTIFY COLUMNS
# =============================================================================

def locate_column(columns, possible_names):

    for name in possible_names:

        for col in columns:

            if col.lower().strip() == name.lower().strip():
                return col

    return None


BUSCO_COL = locate_column(
    df.columns,
    [
        "BUSCO_ID",
        "BUSCO"
    ]
)

PATHWAY_ID_COL = locate_column(
    df.columns,
    [
        "KEGG_Pathway",
        "KEGG_Pathway_ID",
        "Pathway_ID",
        "Pathway"
    ]
)

PATHWAY_NAME_COL = locate_column(
    df.columns,
    [
        "KEGG_Pathway_Name",
        "Pathway_Name",
        "PathwayName"
    ]
)

KEGG_GENE_COL = locate_column(
    df.columns,
    [
        "KEGG_Gene",
        "KEGG_Gene_ID",
        "KEGG_ID"
    ]
)


print("\nDetected columns:")

print("BUSCO column       :", BUSCO_COL)
print("Pathway ID column  :", PATHWAY_ID_COL)
print("Pathway name column:", PATHWAY_NAME_COL)
print("KEGG gene column   :", KEGG_GENE_COL)


if BUSCO_COL is None:
    print("\nERROR: BUSCO column could not be identified.")
    raise SystemExit(1)

if PATHWAY_NAME_COL is None:
    print("\nERROR: Pathway name column could not be identified.")
    print("\nAvailable columns:")
    print(list(df.columns))
    raise SystemExit(1)


# =============================================================================
# CLEAN DATA
# =============================================================================

df[BUSCO_COL] = (
    df[BUSCO_COL]
    .fillna("")
    .astype(str)
    .str.strip()
)

df[PATHWAY_NAME_COL] = (
    df[PATHWAY_NAME_COL]
    .fillna("")
    .astype(str)
    .str.strip()
)

if PATHWAY_ID_COL is not None:

    df[PATHWAY_ID_COL] = (
        df[PATHWAY_ID_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )

if KEGG_GENE_COL is not None:

    df[KEGG_GENE_COL] = (
        df[KEGG_GENE_COL]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# Remove records without pathway names
named_df = df[
    df[PATHWAY_NAME_COL] != ""
].copy()


print("\nRecords containing resolved pathway names:")
print(len(named_df))


# =============================================================================
# BUSCO → PATHWAY SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("CREATING BUSCO → NAMED PATHWAY SUMMARY")
print("=" * 80)

busco_records = []

for busco, group in named_df.groupby(BUSCO_COL):

    pathway_names = sorted(
        set(
            group[PATHWAY_NAME_COL]
            .dropna()
            .astype(str)
            .str.strip()
        )
    )

    pathway_names = [
        x for x in pathway_names
        if x
    ]

    pathway_ids = []

    if PATHWAY_ID_COL is not None:

        pathway_ids = sorted(
            set(
                group[PATHWAY_ID_COL]
                .dropna()
                .astype(str)
                .str.strip()
            )
        )

        pathway_ids = [
            x for x in pathway_ids
            if x
        ]

    genes = []

    if KEGG_GENE_COL is not None:

        genes = sorted(
            set(
                group[KEGG_GENE_COL]
                .dropna()
                .astype(str)
                .str.strip()
            )
        )

        genes = [
            x for x in genes
            if x
        ]

    busco_records.append({

        "BUSCO_ID": busco,

        "KEGG_Gene_Count": len(genes),

        "KEGG_Pathway_Count": len(pathway_names),

        "KEGG_Pathway_IDs":
            "; ".join(pathway_ids),

        "KEGG_Pathway_Names":
            "; ".join(pathway_names),

        "Pathway_Mapped":
            len(pathway_names) > 0
    })


busco_summary = pd.DataFrame(
    busco_records
)


# =============================================================================
# INCLUDE ALL 20 BUSCO CANDIDATES
# =============================================================================

# Your previous result contains 20 candidates.
# The resolved pathway file contains only the 9 mapped candidates.
#
# Therefore we explicitly retain the 20-candidate universe here.

ALL_BUSCOS = [
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


all_busco_df = pd.DataFrame({
    "BUSCO_ID": ALL_BUSCOS
})


busco_summary = all_busco_df.merge(
    busco_summary,
    on="BUSCO_ID",
    how="left"
)


# Fill unmapped candidates
busco_summary["KEGG_Gene_Count"] = (
    busco_summary["KEGG_Gene_Count"]
    .fillna(0)
    .astype(int)
)

busco_summary["KEGG_Pathway_Count"] = (
    busco_summary["KEGG_Pathway_Count"]
    .fillna(0)
    .astype(int)
)

busco_summary["KEGG_Pathway_IDs"] = (
    busco_summary["KEGG_Pathway_IDs"]
    .fillna("")
)

busco_summary["KEGG_Pathway_Names"] = (
    busco_summary["KEGG_Pathway_Names"]
    .fillna("")
)

busco_summary["Pathway_Mapped"] = (
    busco_summary["Pathway_Mapped"]
    .fillna(False)
)


# =============================================================================
# SAVE BUSCO SUMMARY
# =============================================================================

BUSCO_OUTPUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_named_pathway_summary_420.csv"
)

busco_summary.to_csv(
    BUSCO_OUTPUT,
    index=False
)


# =============================================================================
# PATHWAY FREQUENCY
# =============================================================================

print("\n" + "=" * 80)
print("CALCULATING NAMED PATHWAY FREQUENCY")
print("=" * 80)

pathway_buscos = {}

for _, row in named_df.iterrows():

    pathway = row[PATHWAY_NAME_COL]
    busco = row[BUSCO_COL]

    if not pathway:
        continue

    if pathway not in pathway_buscos:
        pathway_buscos[pathway] = set()

    pathway_buscos[pathway].add(busco)


pathway_records = []

for pathway, buscos in pathway_buscos.items():

    pathway_records.append({

        "KEGG_Pathway_Name":
            pathway,

        "BUSCO_Count":
            len(buscos),

        "BUSCOs":
            "; ".join(sorted(buscos))
    })


pathway_frequency = pd.DataFrame(
    pathway_records
)


pathway_frequency = pathway_frequency.sort_values(
    by="BUSCO_Count",
    ascending=False
)


PATHWAY_OUTPUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_named_pathway_frequency_420.csv"
)

pathway_frequency.to_csv(
    PATHWAY_OUTPUT,
    index=False
)


# =============================================================================
# BUSCO × PATHWAY MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("CREATING BUSCO × PATHWAY MATRIX")
print("=" * 80)

matrix_source = named_df[
    [
        BUSCO_COL,
        PATHWAY_NAME_COL
    ]
].drop_duplicates()

matrix_source["Present"] = 1


matrix = matrix_source.pivot_table(
    index=BUSCO_COL,
    columns=PATHWAY_NAME_COL,
    values="Present",
    aggfunc="max",
    fill_value=0
)


# Make sure all 20 BUSCOs appear
matrix = matrix.reindex(
    ALL_BUSCOS,
    fill_value=0
)


MATRIX_OUTPUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_named_pathway_matrix_420.csv"
)

matrix.to_csv(
    MATRIX_OUTPUT
)


# =============================================================================
# FUNCTIONAL CATEGORY CLASSIFICATION
# =============================================================================

print("\n" + "=" * 80)
print("CLASSIFYING NAMED PATHWAYS")
print("=" * 80)


def classify_pathway(pathway):

    p = pathway.lower()

    metabolism = [
        "carbon metabolism",
        "glycolysis",
        "gluconeogenesis",
        "citrate cycle",
        "tca cycle",
        "amino acid metabolism",
        "amino acid biosynthesis",
        "fatty acid",
        "purine metabolism",
        "pyrimidine metabolism",
        "oxidative phosphorylation",
        "metabolic pathways",
        "biosynthesis"
    ]

    rna = [
        "rna transport",
        "rna degradation",
        "spliceosome",
        "ribosome",
        "translation",
        "transcription",
        "mrna"
    ]

    protein = [
        "protein processing",
        "proteasome",
        "protein export",
        "protein folding",
        "ubiquitin"
    ]

    membrane = [
        "endocytosis",
        "vesicle",
        "membrane transport",
        "transport",
        "secretion"
    ]

    cellular = [
        "cell cycle",
        "cytoskeleton",
        "mitosis",
        "meiosis",
        "cellular"
    ]

    signaling = [
        "signaling",
        "signal",
        "mapk",
        "pi3k",
        "calcium",
        "kinase"
    ]

    if any(term in p for term in metabolism):
        return "Metabolism / enzymatic processes"

    if any(term in p for term in rna):
        return "RNA processing / information processing"

    if any(term in p for term in protein):
        return "Protein processing / regulation"

    if any(term in p for term in membrane):
        return "Transport / membrane processes"

    if any(term in p for term in cellular):
        return "Cellular organization / cell cycle"

    if any(term in p for term in signaling):
        return "Cell signaling / regulation"

    return "Other / pathway annotation"


pathway_frequency["Functional_Category"] = (
    pathway_frequency["KEGG_Pathway_Name"]
    .apply(classify_pathway)
)


CATEGORY_OUTPUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_functional_category_summary_420.csv"
)


category_summary = (
    pathway_frequency
    .groupby("Functional_Category")
    .agg(
        Unique_Pathways=(
            "KEGG_Pathway_Name",
            "count"
        ),
        BUSCO_Representation=(
            "BUSCO_Count",
            "sum"
        )
    )
    .reset_index()
    .sort_values(
        "Unique_Pathways",
        ascending=False
    )
)


category_summary.to_csv(
    CATEGORY_OUTPUT,
    index=False
)


# =============================================================================
# COVERAGE
# =============================================================================

total_candidates = 20

mapped_candidates = int(
    busco_summary["Pathway_Mapped"].sum()
)

unmapped_candidates = (
    total_candidates -
    mapped_candidates
)

unique_pathways = (
    len(pathway_frequency)
)


coverage_df = pd.DataFrame({

    "Metric": [
        "Total Carbon Breadth candidates",
        "Candidates mapped to KEGG pathways",
        "Candidates without pathway mapping",
        "Unique named KEGG pathways"
    ],

    "Count": [
        total_candidates,
        mapped_candidates,
        unmapped_candidates,
        unique_pathways
    ]
})


COVERAGE_OUTPUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_named_pathway_coverage_420.csv"
)


coverage_df.to_csv(
    COVERAGE_OUTPUT,
    index=False
)


# =============================================================================
# FINAL REPORT
# =============================================================================

REPORT_OUTPUT = os.path.join(
    REPORT_DIR,
    "Carbon_Breadth_final_named_pathway_interpretation_420.txt"
)


with open(
    REPORT_OUTPUT,
    "w",
    encoding="utf-8"
) as report:

    report.write("=" * 80 + "\n")
    report.write(
        "FINAL CARBON BREADTH PATHWAY BIOLOGICAL INTERPRETATION\n"
    )
    report.write("=" * 80 + "\n\n")

    report.write("CANDIDATE OVERVIEW\n")
    report.write("-" * 80 + "\n")

    report.write(
        f"Total Carbon Breadth candidates : "
        f"{total_candidates}\n"
    )

    report.write(
        f"Candidates mapped to pathways   : "
        f"{mapped_candidates}\n"
    )

    report.write(
        f"Candidates without pathways     : "
        f"{unmapped_candidates}\n"
    )

    report.write(
        f"Unique named KEGG pathways      : "
        f"{unique_pathways}\n\n"
    )

    report.write("=" * 80 + "\n")
    report.write("BUSCO → NAMED PATHWAYS\n")
    report.write("=" * 80 + "\n\n")

    for _, row in busco_summary.iterrows():

        report.write(
            f"BUSCO: {row['BUSCO_ID']}\n"
        )

        report.write(
            f"KEGG genes: "
            f"{row['KEGG_Gene_Count']}\n"
        )

        report.write(
            f"Pathway count: "
            f"{row['KEGG_Pathway_Count']}\n"
        )

        if row["KEGG_Pathway_Names"]:

            report.write("Pathways:\n")

            for pathway in row[
                "KEGG_Pathway_Names"
            ].split("; "):

                report.write(
                    f"  - {pathway}\n"
                )

        else:

            report.write(
                "Pathways: None retrieved\n"
            )

        report.write("\n")

    report.write("=" * 80 + "\n")
    report.write("MOST REPRESENTED PATHWAYS\n")
    report.write("=" * 80 + "\n\n")

    for _, row in pathway_frequency.head(20).iterrows():

        report.write(
            f"{row['KEGG_Pathway_Name']} | "
            f"BUSCO count = "
            f"{row['BUSCO_Count']}\n"
        )

    report.write("\n")

    report.write("=" * 80 + "\n")
    report.write("FUNCTIONAL CATEGORY SUMMARY\n")
    report.write("=" * 80 + "\n\n")

    for _, row in category_summary.iterrows():

        report.write(
            f"{row['Functional_Category']}\n"
            f"  Unique pathways: "
            f"{row['Unique_Pathways']}\n"
            f"  BUSCO representation: "
            f"{row['BUSCO_Representation']}\n\n"
        )

    report.write("=" * 80 + "\n")
    report.write("INTERPRETATION NOTE\n")
    report.write("=" * 80 + "\n\n")

    report.write(
        "The pathway analysis describes KEGG pathway annotations "
        "associated with the 20 selected Carbon Breadth candidate "
        "BUSCOs. It is a functional annotation analysis and should "
        "not be interpreted as formal pathway enrichment unless an "
        "appropriate background set is supplied.\n"
    )


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n")
print("=" * 80)
print("FINAL NAMED PATHWAY INTERPRETATION COMPLETE")
print("=" * 80)

print("\nTotal candidates       :", total_candidates)
print("Pathway-mapped         :", mapped_candidates)
print("Not pathway-mapped     :", unmapped_candidates)
print("Unique named pathways  :", unique_pathways)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print("\nTABLES:")

print(
    " ",
    BUSCO_OUTPUT
)

print(
    " ",
    PATHWAY_OUTPUT
)

print(
    " ",
    MATRIX_OUTPUT
)

print(
    " ",
    CATEGORY_OUTPUT
)

print(
    " ",
    COVERAGE_OUTPUT
)

print("\nREPORT:")

print(
    " ",
    REPORT_OUTPUT
)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)