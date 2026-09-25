# =============================================================================
# FINAL CARBON BREADTH BIOLOGICAL INTERPRETATION
# Project: Y1000 Chassis / Carbon Breadth
# Run ID: 420
#
# PURPOSE
# -------
# This script performs the FINAL interpretation of the already annotated
# 20 Carbon Breadth BUSCO candidates.
#
# IMPORTANT:
#   - No new annotation is performed.
#   - No UniProt queries are performed.
#   - No KEGG queries are performed.
#   - No protein extraction is performed.
#
# It integrates the existing:
#   1. Final integrated annotation
#   2. Detailed KEGG gene -> pathway mapping
#   3. Candidate BUSCO information
#
# OUTPUTS:
#   - candidate functional summary
#   - BUSCO x pathway matrix
#   - pathway frequency table
#   - functional category table
#   - KEGG pathway network
#   - annotation coverage figure
#   - pathway frequency figure
#   - candidate pathway heatmap
#   - final biological interpretation report
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import re
import sys

# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

BASE = Path(
    r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset"
    r"\phylogeny_aware_ml\phylogeny_cv\model_training"
    r"\feature_interpretation_420\functional_annotation_420"
    r"\BUSCO20_annotation"
)

FINAL_DIR = BASE / "FINAL_CARBON_BREADTH_ANNOTATION_420"

PATHWAY_DIR = BASE / "pathway_analysis_420"

OUTPUT_DIR = BASE / "FINAL_CARBON_BREADTH_INTERPRETATION_420"

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. INPUT FILES
# =============================================================================

FINAL_INTEGRATED = (
    FINAL_DIR /
    "Carbon_Breadth_top20_FINAL_integrated_annotation_420.csv"
)

FINAL_PATHWAY = (
    FINAL_DIR /
    "Carbon_Breadth_top20_FINAL_pathway_annotation_420.csv"
)

CANDIDATE_SUMMARY = (
    FINAL_DIR /
    "Carbon_Breadth_top20_FINAL_candidate_summary_420.csv"
)

PATHWAY_SUMMARY = (
    FINAL_DIR /
    "Carbon_Breadth_FINAL_pathway_summary_420.csv"
)

DETAILED_PATHWAY = (
    PATHWAY_DIR /
    "Carbon_Breadth_top20_KEGG_gene_pathway_mapping_420.csv"
)


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def clean_text(value):
    """Convert values to clean strings while preserving missing values."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_annotation(value):
    """
    Split semicolon-delimited annotation fields.
    Handles NaN and empty strings.
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "na"}:
        return []

    parts = []

    for item in value.split(";"):
        item = item.strip()

        if item:
            parts.append(item)

    return parts


def find_column(df, candidates, required=True):
    """
    Find a column using case-insensitive matching.
    """
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            f"Could not find required column.\n"
            f"Looking for: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def safe_filename(text):
    """Create a filesystem-safe filename."""
    text = re.sub(r"[^\w\-\.]+", "_", str(text))
    return text[:150]


# =============================================================================
# 4. LOAD INPUTS
# =============================================================================

print("=" * 80)
print("FINAL CARBON BREADTH BIOLOGICAL INTERPRETATION")
print("=" * 80)

print("\nINPUT FILES")
print("-" * 80)

for path in [
    FINAL_INTEGRATED,
    FINAL_PATHWAY,
    CANDIDATE_SUMMARY,
    PATHWAY_SUMMARY,
    DETAILED_PATHWAY
]:
    print(path)

print("\n")


def load_required(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired input file was not found:\n{path}\n"
        )

    return pd.read_csv(path)


integrated = load_required(FINAL_INTEGRATED)
pathway_annotation = load_required(FINAL_PATHWAY)
candidate_summary = load_required(CANDIDATE_SUMMARY)
pathway_summary = load_required(PATHWAY_SUMMARY)
detailed_pathway = load_required(DETAILED_PATHWAY)


# =============================================================================
# 5. DISPLAY INPUT INFORMATION
# =============================================================================

print("=" * 80)
print("INPUT SUMMARY")
print("=" * 80)

print(f"Final integrated annotation : {integrated.shape}")
print(f"Pathway annotation          : {pathway_annotation.shape}")
print(f"Candidate summary            : {candidate_summary.shape}")
print(f"Pathway summary              : {pathway_summary.shape}")
print(f"Detailed pathway mapping     : {detailed_pathway.shape}")

print("\n")


# =============================================================================
# 6. IDENTIFY IMPORTANT COLUMNS
# =============================================================================

BUSCO_COL = find_column(
    integrated,
    ["BUSCO_ID", "BUSCO", "busco_id"]
)

FUNCTION_COL = find_column(
    integrated,
    [
        "Functional_Description",
        "Function",
        "Functional Description"
    ],
    required=False
)

GO_COL = find_column(
    integrated,
    ["GO", "GO_Terms", "GO_Term"],
    required=False
)

KEGG_COL = find_column(
    integrated,
    ["KEGG", "KEGG_ID", "KEGG_Gene"],
    required=False
)

EC_COL = find_column(
    integrated,
    ["EC", "EC_Number", "EC_Numbers"],
    required=False
)


# =============================================================================
# 7. CREATE MASTER CANDIDATE TABLE
# =============================================================================

print("=" * 80)
print("BUILDING MASTER CANDIDATE TABLE")
print("=" * 80)

master = integrated.copy()

master[BUSCO_COL] = master[BUSCO_COL].astype(str).str.strip()

# Make sure we have exactly the candidate BUSCOs represented in the
# integrated annotation.
candidate_buscos = master[BUSCO_COL].dropna().unique().tolist()

print(f"\nCandidate BUSCOs found: {len(candidate_buscos)}")

if len(candidate_buscos) != 20:
    print(
        f"WARNING: Expected 20 candidates but found "
        f"{len(candidate_buscos)}."
    )


# =============================================================================
# 8. BUILD KEGG GENE -> PATHWAY TABLE
# =============================================================================

print("\n")
print("=" * 80)
print("PROCESSING KEGG GENE -> PATHWAY MAPPING")
print("=" * 80)

print("\nDetailed pathway mapping columns:")
for col in detailed_pathway.columns:
    print(f"  {col}")


# Identify BUSCO column
DETAIL_BUSCO_COL = find_column(
    detailed_pathway,
    ["BUSCO_ID", "BUSCO", "busco_id"]
)

# Identify KEGG gene column
DETAIL_KEGG_COL = find_column(
    detailed_pathway,
    [
        "KEGG_Gene",
        "KEGG_Gene_ID",
        "KEGG_ID",
        "KEGG"
    ],
    required=False
)

# Identify pathway ID column
PATHWAY_ID_COL = find_column(
    detailed_pathway,
    [
        "Pathway_ID",
        "KEGG_Pathway",
        "Pathway",
        "KEGG_Pathway_ID"
    ],
    required=False
)

# Identify pathway name
PATHWAY_NAME_COL = find_column(
    detailed_pathway,
    [
        "Pathway_Name",
        "KEGG_Pathway_Name",
        "Pathway_Name_KEGG"
    ],
    required=False
)


print("\nDetected columns:")
print(f"BUSCO column       : {DETAIL_BUSCO_COL}")
print(f"KEGG gene column   : {DETAIL_KEGG_COL}")
print(f"Pathway ID column  : {PATHWAY_ID_COL}")
print(f"Pathway name column: {PATHWAY_NAME_COL}")


# =============================================================================
# 9. NORMALIZE DETAILED PATHWAY DATA
# =============================================================================

records = []

for _, row in detailed_pathway.iterrows():

    busco = clean_text(row[DETAIL_BUSCO_COL])

    if not busco:
        continue

    # Get KEGG gene
    if DETAIL_KEGG_COL is not None:
        kegg_gene = clean_text(row[DETAIL_KEGG_COL])
    else:
        kegg_gene = ""

    # Get pathway ID
    if PATHWAY_ID_COL is not None:
        pathway_id = clean_text(row[PATHWAY_ID_COL])
    else:
        pathway_id = ""

    # Get pathway name
    if PATHWAY_NAME_COL is not None:
        pathway_name = clean_text(row[PATHWAY_NAME_COL])
    else:
        pathway_name = ""

    # Some previous pathway tables may store multiple pathway IDs
    # in a single cell.
    pathway_ids = split_annotation(pathway_id)

    # Same for names
    pathway_names = split_annotation(pathway_name)

    if pathway_ids:

        for i, pid in enumerate(pathway_ids):

            pname = ""

            if i < len(pathway_names):
                pname = pathway_names[i]

            records.append({
                "BUSCO_ID": busco,
                "KEGG_Gene": kegg_gene,
                "Pathway_ID": pid,
                "Pathway_Name": pname
            })

    else:

        # Preserve rows even if pathway ID is absent.
        records.append({
            "BUSCO_ID": busco,
            "KEGG_Gene": kegg_gene,
            "Pathway_ID": "",
            "Pathway_Name": ""
        })


mapping = pd.DataFrame(records)

if mapping.empty:
    print("\nWARNING: No pathway records were detected.")

else:

    # Remove exact duplicates
    mapping = mapping.drop_duplicates()

    print(f"\nNormalized pathway records: {len(mapping)}")

    print(
        f"Unique BUSCOs: "
        f"{mapping['BUSCO_ID'].nunique()}"
    )

    print(
        f"Unique KEGG genes: "
        f"{mapping['KEGG_Gene'].replace('', np.nan).nunique()}"
    )

    print(
        f"Unique pathways: "
        f"{mapping['Pathway_ID'].replace('', np.nan).nunique()}"
    )


# =============================================================================
# 10. SAVE CLEAN DETAILED MAPPING
# =============================================================================

CLEAN_MAPPING = (
    TABLE_DIR /
    "Carbon_Breadth_clean_BUSCO_KEGG_pathway_mapping_420.csv"
)

mapping.to_csv(CLEAN_MAPPING, index=False)


# =============================================================================
# 11. BUILD BUSCO x PATHWAY MATRIX
# =============================================================================

print("\n")
print("=" * 80)
print("BUILDING BUSCO × PATHWAY MATRIX")
print("=" * 80)

valid_mapping = mapping[
    mapping["Pathway_ID"].astype(str).str.strip() != ""
].copy()

if valid_mapping.empty:

    pathway_matrix = pd.DataFrame(
        {"BUSCO_ID": candidate_buscos}
    )

else:

    pathway_matrix = pd.crosstab(
        valid_mapping["BUSCO_ID"],
        valid_mapping["Pathway_ID"]
    )

    pathway_matrix = (
        pathway_matrix
        .reindex(candidate_buscos, fill_value=0)
        .reset_index()
    )

    pathway_matrix.columns.name = None


PATHWAY_MATRIX_FILE = (
    TABLE_DIR /
    "Carbon_Breadth_BUSCO_x_KEGG_pathway_matrix_420.csv"
)

pathway_matrix.to_csv(
    PATHWAY_MATRIX_FILE,
    index=False
)

print(
    f"BUSCO × pathway matrix: "
    f"{pathway_matrix.shape}"
)


# =============================================================================
# 12. PATHWAY FREQUENCY
# =============================================================================

print("\n")
print("=" * 80)
print("CALCULATING PATHWAY FREQUENCY")
print("=" * 80)

if valid_mapping.empty:

    pathway_frequency = pd.DataFrame(
        columns=[
            "Pathway_ID",
            "Pathway_Name",
            "BUSCO_Count",
            "KEGG_Gene_Count"
        ]
    )

else:

    pathway_frequency = (
        valid_mapping
        .groupby(
            ["Pathway_ID", "Pathway_Name"],
            dropna=False
        )
        .agg(
            BUSCO_Count=("BUSCO_ID", "nunique"),
            KEGG_Gene_Count=("KEGG_Gene", "nunique")
        )
        .reset_index()
        .sort_values(
            ["BUSCO_Count", "KEGG_Gene_Count"],
            ascending=False
        )
    )


PATHWAY_FREQ_FILE = (
    TABLE_DIR /
    "Carbon_Breadth_pathway_frequency_final_420.csv"
)

pathway_frequency.to_csv(
    PATHWAY_FREQ_FILE,
    index=False
)

print(
    f"Unique pathways: "
    f"{len(pathway_frequency)}"
)


# =============================================================================
# 13. CANDIDATE FUNCTIONAL SUMMARY
# =============================================================================

print("\n")
print("=" * 80)
print("BUILDING CANDIDATE FUNCTIONAL SUMMARY")
print("=" * 80)

candidate_rows = []

for _, row in master.iterrows():

    busco = clean_text(row[BUSCO_COL])

    function = ""

    if FUNCTION_COL is not None:
        function = clean_text(row[FUNCTION_COL])

    go_terms = []

    if GO_COL is not None:
        go_terms = split_annotation(row[GO_COL])

    kegg_genes = []

    if KEGG_COL is not None:
        kegg_genes = split_annotation(row[KEGG_COL])

    ec_terms = []

    if EC_COL is not None:
        ec_terms = split_annotation(row[EC_COL])

    candidate_mapping = valid_mapping[
        valid_mapping["BUSCO_ID"] == busco
    ]

    pathway_ids = (
        candidate_mapping["Pathway_ID"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    pathway_names = (
        candidate_mapping["Pathway_Name"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    pathway_names = [
        x for x in pathway_names
        if x.strip() and x.lower() != "nan"
    ]

    candidate_rows.append({
        "BUSCO_ID": busco,
        "Functional_Description": function,
        "GO_Count": len(go_terms),
        "KEGG_Gene_Count": len(kegg_genes),
        "EC_Count": len(ec_terms),
        "KEGG_Pathway_Count": len(pathway_ids),
        "Pathway_Mapped": len(pathway_ids) > 0,
        "KEGG_Pathways": "; ".join(pathway_ids),
        "KEGG_Pathway_Names": "; ".join(pathway_names)
    })


candidate_functional = pd.DataFrame(candidate_rows)


CANDIDATE_FUNCTIONAL_FILE = (
    TABLE_DIR /
    "Carbon_Breadth_final_functional_candidate_table_420.csv"
)

candidate_functional.to_csv(
    CANDIDATE_FUNCTIONAL_FILE,
    index=False
)


# =============================================================================
# 14. FUNCTIONAL CATEGORY CLASSIFICATION
# =============================================================================
#
# This is deliberately a transparent rule-based grouping.
# It is NOT claiming a new annotation.
#
# Categories are based only on the existing functional descriptions.
# =============================================================================

def assign_functional_category(function):

    f = str(function).lower()

    if any(
        x in f for x in [
            "helicase",
            "rna helicase",
            "methyltransferase",
            "methyltransfer",
            "rna",
            "mrna cap",
            "trna"
        ]
    ):
        return "RNA processing / RNA metabolism"

    if any(
        x in f for x in [
            "aconitase",
            "dehydratase",
            "transferase",
            "phosphatase",
            "gpi biosynthesis"
        ]
    ):
        return "Metabolism / enzymatic processes"

    if any(
        x in f for x in [
            "kinesin",
            "snf7",
            "pig-f",
            "gpi",
            "yip",
            "armadillo"
        ]
    ):
        return "Cellular organization / trafficking"

    if any(
        x in f for x in [
            "sant",
            "myb",
            "wd40",
            "pdz",
            "smr"
        ]
    ):
        return "Protein interaction / regulation"

    if any(
        x in f for x in [
            "ribosomal",
            "ribosome"
        ]
    ):
        return "Translation / mitochondrial translation"

    if any(
        x in f for x in [
            "atpase assembly",
            "atpase"
        ]
    ):
        return "ATP-dependent cellular processes"

    return "Other / domain-level annotation"


candidate_functional["Functional_Category"] = (
    candidate_functional["Functional_Description"]
    .apply(assign_functional_category)
)


# Reorder columns
candidate_functional = candidate_functional[
    [
        "BUSCO_ID",
        "Functional_Description",
        "Functional_Category",
        "GO_Count",
        "KEGG_Gene_Count",
        "EC_Count",
        "KEGG_Pathway_Count",
        "Pathway_Mapped",
        "KEGG_Pathways",
        "KEGG_Pathway_Names"
    ]
]


candidate_functional.to_csv(
    CANDIDATE_FUNCTIONAL_FILE,
    index=False
)


# =============================================================================
# 15. FUNCTIONAL CATEGORY FREQUENCY
# =============================================================================

category_frequency = (
    candidate_functional
    .groupby("Functional_Category")
    .agg(
        Candidate_Count=("BUSCO_ID", "count"),
        Pathway_Mapped_Count=("Pathway_Mapped", "sum")
    )
    .reset_index()
    .sort_values(
        "Candidate_Count",
        ascending=False
    )
)

CATEGORY_FILE = (
    TABLE_DIR /
    "Carbon_Breadth_functional_category_summary_420.csv"
)

category_frequency.to_csv(
    CATEGORY_FILE,
    index=False
)


# =============================================================================
# 16. GO TERM FREQUENCY
# =============================================================================

go_records = []

for _, row in master.iterrows():

    busco = clean_text(row[BUSCO_COL])

    if GO_COL is None:
        continue

    for go in split_annotation(row[GO_COL]):

        go_records.append({
            "BUSCO_ID": busco,
            "GO": go
        })


go_df = pd.DataFrame(go_records)

if go_df.empty:

    go_frequency = pd.DataFrame(
        columns=[
            "GO",
            "BUSCO_Count"
        ]
    )

else:

    go_frequency = (
        go_df
        .groupby("GO")
        .agg(
            BUSCO_Count=("BUSCO_ID", "nunique")
        )
        .reset_index()
        .sort_values(
            "BUSCO_Count",
            ascending=False
        )
    )


GO_FILE = (
    TABLE_DIR /
    "Carbon_Breadth_GO_term_summary_420.csv"
)

go_frequency.to_csv(
    GO_FILE,
    index=False
)


# =============================================================================
# 17. EC SUMMARY
# =============================================================================

ec_records = []

for _, row in master.iterrows():

    busco = clean_text(row[BUSCO_COL])

    if EC_COL is None:
        continue

    for ec in split_annotation(row[EC_COL]):

        ec_records.append({
            "BUSCO_ID": busco,
            "EC": ec
        })


ec_df = pd.DataFrame(ec_records)

if ec_df.empty:

    ec_summary = pd.DataFrame(
        columns=[
            "EC",
            "BUSCO_Count"
        ]
    )

else:

    ec_summary = (
        ec_df
        .groupby("EC")
        .agg(
            BUSCO_Count=("BUSCO_ID", "nunique")
        )
        .reset_index()
        .sort_values(
            "BUSCO_Count",
            ascending=False
        )
    )


EC_FILE = (
    TABLE_DIR /
    "Carbon_Breadth_EC_summary_420.csv"
)

ec_summary.to_csv(
    EC_FILE,
    index=False
)


# =============================================================================
# 18. ANNOTATION COVERAGE
# =============================================================================

total_candidates = len(candidate_functional)

with_go = int(
    (candidate_functional["GO_Count"] > 0).sum()
)

with_kegg = int(
    (candidate_functional["KEGG_Gene_Count"] > 0).sum()
)

with_ec = int(
    (candidate_functional["EC_Count"] > 0).sum()
)

with_pathway = int(
    candidate_functional["Pathway_Mapped"].sum()
)

coverage = pd.DataFrame({
    "Annotation_Type": [
        "Total candidates",
        "GO annotated",
        "KEGG annotated",
        "EC annotated",
        "Pathway mapped"
    ],
    "Count": [
        total_candidates,
        with_go,
        with_kegg,
        with_ec,
        with_pathway
    ]
})

COVERAGE_FILE = (
    TABLE_DIR /
    "Carbon_Breadth_annotation_coverage_420.csv"
)

coverage.to_csv(
    COVERAGE_FILE,
    index=False
)


# =============================================================================
# 19. FIGURE 1 — ANNOTATION COVERAGE
# =============================================================================

print("\n")
print("=" * 80)
print("GENERATING FIGURES")
print("=" * 80)

plt.figure(figsize=(9, 6))

labels = [
    "Total candidates",
    "GO",
    "KEGG",
    "EC",
    "KEGG pathway"
]

values = [
    total_candidates,
    with_go,
    with_kegg,
    with_ec,
    with_pathway
]

plt.bar(labels, values)

plt.ylabel("Number of BUSCO candidates")
plt.title(
    "Carbon Breadth Candidate Annotation Coverage"
)

plt.xticks(rotation=25, ha="right")
plt.tight_layout()

coverage_fig = (
    FIGURE_DIR /
    "Carbon_Breadth_annotation_coverage_420.png"
)

plt.savefig(
    coverage_fig,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 20. FIGURE 2 — TOP PATHWAY FREQUENCIES
# =============================================================================

top_n = 20

top_pathways = pathway_frequency.head(top_n).copy()

if not top_pathways.empty:

    # Prefer pathway names where available
    plot_labels = []

    for _, row in top_pathways.iterrows():

        pid = clean_text(row["Pathway_ID"])
        pname = clean_text(row["Pathway_Name"])

        if pname:
            label = f"{pid}: {pname}"
        else:
            label = pid

        plot_labels.append(
            textwrap.fill(label, width=45)
        )

    plt.figure(figsize=(12, 9))

    plt.barh(
        range(len(top_pathways)),
        top_pathways["BUSCO_Count"]
    )

    plt.yticks(
        range(len(top_pathways)),
        plot_labels
    )

    plt.xlabel(
        "Number of Carbon Breadth BUSCO candidates"
    )

    plt.title(
        "Most Frequently Represented KEGG Pathways"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    pathway_fig = (
        FIGURE_DIR /
        "Carbon_Breadth_KEGG_pathway_frequency_420.png"
    )

    plt.savefig(
        pathway_fig,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# 21. FIGURE 3 — CANDIDATE PATHWAY HEATMAP
# =============================================================================

if not valid_mapping.empty:

    # Select pathways represented by at least one candidate.
    # To keep the figure readable, use the 30 most represented pathways.
    selected_pathways = (
        pathway_frequency
        .head(30)["Pathway_ID"]
        .tolist()
    )

    heatmap = pathway_matrix.set_index("BUSCO_ID")

    available = [
        p for p in selected_pathways
        if p in heatmap.columns
    ]

    if available:

        heatmap = heatmap[available]

        plt.figure(
            figsize=(
                max(12, len(available) * 0.35),
                max(8, len(heatmap) * 0.35)
            )
        )

        plt.imshow(
            heatmap.values,
            aspect="auto",
            interpolation="nearest"
        )

        plt.xticks(
            range(len(available)),
            [
                textwrap.fill(x, 18)
                for x in available
            ],
            rotation=90
        )

        plt.yticks(
            range(len(heatmap.index)),
            heatmap.index
        )

        plt.xlabel("KEGG pathway")
        plt.ylabel("Carbon Breadth BUSCO")

        plt.title(
            "Carbon Breadth BUSCO × KEGG Pathway Representation"
        )

        plt.colorbar(
            label="KEGG mapping count"
        )

        plt.tight_layout()

        heatmap_fig = (
            FIGURE_DIR /
            "Carbon_Breadth_BUSCO_pathway_heatmap_420.png"
        )

        plt.savefig(
            heatmap_fig,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# =============================================================================
# 22. FIGURE 4 — FUNCTIONAL CATEGORY DISTRIBUTION
# =============================================================================

category_plot = (
    category_frequency
    .sort_values("Candidate_Count", ascending=True)
)

plt.figure(figsize=(10, 6))

plt.barh(
    category_plot["Functional_Category"],
    category_plot["Candidate_Count"]
)

plt.xlabel("Number of candidate BUSCOs")
plt.ylabel("Functional category")

plt.title(
    "Functional Categories of Carbon Breadth Candidates"
)

plt.tight_layout()

category_fig = (
    FIGURE_DIR /
    "Carbon_Breadth_functional_category_distribution_420.png"
)

plt.savefig(
    category_fig,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 23. BUSCO–PATHWAY NETWORK TABLE
# =============================================================================
#
# Instead of requiring NetworkX, create a clean edge table that can be used
# directly in Cytoscape or Gephi.
# =============================================================================

network_edges = valid_mapping[
    [
        "BUSCO_ID",
        "Pathway_ID",
        "Pathway_Name"
    ]
].drop_duplicates()


NETWORK_FILE = (
    TABLE_DIR /
    "Carbon_Breadth_BUSCO_KEGG_network_edges_420.csv"
)

network_edges.to_csv(
    NETWORK_FILE,
    index=False
)


# =============================================================================
# 24. PATHWAY GROUPING
# =============================================================================
#
# Broad grouping based on pathway-name text only.
# This is an interpretive visualization aid, not a replacement for KEGG
# annotation.
# =============================================================================

def pathway_category(pathway_name):

    p = str(pathway_name).lower()

    if any(
        x in p for x in [
            "carbon",
            "glycolysis",
            "citrate",
            "amino acid",
            "biosynthesis",
            "metabolism",
            "fatty acid",
            "pyruvate",
            "starch",
            "sugar"
        ]
    ):
        return "Metabolism"

    if any(
        x in p for x in [
            "rna",
            "translation",
            "ribosome",
            "spliceosome",
            "transcription"
        ]
    ):
        return "Gene expression / RNA"

    if any(
        x in p for x in [
            "protein processing",
            "protein",
            "proteasome",
            "ubiquitin"
        ]
    ):
        return "Protein processing"

    if any(
        x in p for x in [
            "transport",
            "vesicle",
            "endocytosis",
            "membrane"
        ]
    ):
        return "Transport / membrane"

    if any(
        x in p for x in [
            "cell cycle",
            "cytoskeleton",
            "motor",
            "mitosis"
        ]
    ):
        return "Cellular organization"

    if any(
        x in p for x in [
            "signaling",
            "signal",
            "mapk",
            "calcium"
        ]
    ):
        return "Signaling"

    return "Other / pathway-specific"


pathway_frequency["Broad_Category"] = (
    pathway_frequency["Pathway_Name"]
    .apply(pathway_category)
)


PATHWAY_CATEGORY_FILE = (
    TABLE_DIR /
    "Carbon_Breadth_pathway_category_summary_420.csv"
)

pathway_frequency.to_csv(
    PATHWAY_CATEGORY_FILE,
    index=False
)


# =============================================================================
# 25. FINAL INTERPRETATION REPORT
# =============================================================================

REPORT_FILE = (
    REPORT_DIR /
    "Carbon_Breadth_final_biological_interpretation_420.txt"
)

mapped_buscos = candidate_functional[
    candidate_functional["Pathway_Mapped"]
]["BUSCO_ID"].tolist()

unmapped_buscos = candidate_functional[
    ~candidate_functional["Pathway_Mapped"]
]["BUSCO_ID"].tolist()


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as report:

    report.write("=" * 80 + "\n")
    report.write(
        "FINAL CARBON BREADTH BIOLOGICAL INTERPRETATION\n"
    )
    report.write("=" * 80 + "\n\n")

    report.write(
        "This report summarizes functional and pathway annotations "
        "for the 20 Carbon Breadth candidate BUSCOs.\n\n"
    )

    report.write(
        "No new functional annotation was performed in this stage. "
        "The interpretation is based on the existing UniProt, GO, "
        "EC and KEGG annotations and the previously generated "
        "KEGG gene-to-pathway mappings.\n\n"
    )

    # -------------------------------------------------------------------------
    # Overall coverage
    # -------------------------------------------------------------------------

    report.write("=" * 80 + "\n")
    report.write("1. ANNOTATION COVERAGE\n")
    report.write("=" * 80 + "\n\n")

    report.write(
        f"Total Carbon Breadth candidates : {total_candidates}\n"
    )

    report.write(
        f"Candidates with GO annotations   : {with_go}\n"
    )

    report.write(
        f"Candidates with KEGG annotations : {with_kegg}\n"
    )

    report.write(
        f"Candidates with EC annotations   : {with_ec}\n"
    )

    report.write(
        f"Candidates mapped to pathways    : {with_pathway}\n"
    )

    report.write(
        f"Unique KEGG pathways represented : "
        f"{len(pathway_frequency)}\n\n"
    )

    # -------------------------------------------------------------------------
    # Pathway-mapped candidates
    # -------------------------------------------------------------------------

    report.write("=" * 80 + "\n")
    report.write("2. PATHWAY-MAPPED CANDIDATES\n")
    report.write("=" * 80 + "\n\n")

    for busco in mapped_buscos:

        row = candidate_functional[
            candidate_functional["BUSCO_ID"] == busco
        ].iloc[0]

        report.write(
            f"{busco}\n"
        )

        report.write(
            f"  Function: "
            f"{row['Functional_Description']}\n"
        )

        report.write(
            f"  KEGG genes: "
            f"{row['KEGG_Gene_Count']}\n"
        )

        report.write(
            f"  Pathways: "
            f"{row['KEGG_Pathway_Count']}\n"
        )

        if row["KEGG_Pathway_Names"]:

            names = row["KEGG_Pathway_Names"].split(";")

            report.write(
                "  Pathway names:\n"
            )

            for name in names[:20]:

                report.write(
                    f"    - {name.strip()}\n"
                )

            if len(names) > 20:

                report.write(
                    f"    ... and "
                    f"{len(names) - 20} additional pathway "
                    f"annotations\n"
                )

        report.write("\n")

    # -------------------------------------------------------------------------
    # Unmapped candidates
    # -------------------------------------------------------------------------

    report.write("=" * 80 + "\n")
    report.write("3. CANDIDATES WITHOUT KEGG PATHWAY MAPPING\n")
    report.write("=" * 80 + "\n\n")

    for busco in unmapped_buscos:

        row = candidate_functional[
            candidate_functional["BUSCO_ID"] == busco
        ].iloc[0]

        report.write(
            f"{busco}\n"
        )

        report.write(
            f"  Function: "
            f"{row['Functional_Description']}\n"
        )

        report.write(
            f"  GO terms: "
            f"{row['GO_Count']}\n"
        )

        report.write(
            f"  EC annotations: "
            f"{row['EC_Count']}\n\n"
        )

    # -------------------------------------------------------------------------
    # Pathway frequency
    # -------------------------------------------------------------------------

    report.write("=" * 80 + "\n")
    report.write("4. MOST FREQUENTLY REPRESENTED PATHWAYS\n")
    report.write("=" * 80 + "\n\n")

    if pathway_frequency.empty:

        report.write(
            "No KEGG pathway records were available.\n"
        )

    else:

        for _, row in pathway_frequency.head(20).iterrows():

            pid = clean_text(row["Pathway_ID"])
            pname = clean_text(row["Pathway_Name"])

            report.write(
                f"{pid}"
            )

            if pname:
                report.write(
                    f" | {pname}"
                )

            report.write(
                f" | BUSCO candidates: "
                f"{int(row['BUSCO_Count'])}"
            )

            report.write(
                f" | KEGG genes: "
                f"{int(row['KEGG_Gene_Count'])}\n"
            )

    report.write("\n")

    # -------------------------------------------------------------------------
    # Functional categories
    # -------------------------------------------------------------------------

    report.write("=" * 80 + "\n")
    report.write("5. FUNCTIONAL CATEGORY SUMMARY\n")
    report.write("=" * 80 + "\n\n")

    for _, row in category_frequency.iterrows():

        report.write(
            f"{row['Functional_Category']}: "
            f"{int(row['Candidate_Count'])} candidates; "
            f"{int(row['Pathway_Mapped_Count'])} pathway-mapped\n"
        )

    report.write("\n")

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------

    report.write("=" * 80 + "\n")
    report.write("6. BIOLOGICAL INTERPRETATION\n")
    report.write("=" * 80 + "\n\n")

    report.write(
        "The Carbon Breadth candidate set contains proteins representing "
        "multiple functional classes rather than a single molecular "
        "function. The annotated candidates include enzymatic functions, "
        "RNA-related functions, protein-interaction domains, intracellular "
        "organization and trafficking-related functions, and other "
        "conserved protein domains.\n\n"
    )

    report.write(
        f"KEGG-based pathway mapping identified "
        f"{with_pathway} of the {total_candidates} candidates with at "
        f"least one pathway association. Across these candidates, "
        f"{len(pathway_frequency)} unique KEGG pathways were represented. "
        f"This indicates that the Carbon Breadth candidates span multiple "
        f"functional contexts rather than being restricted to one pathway.\n\n"
    )

    report.write(
        "The candidate-level pathway counts should be interpreted as "
        "annotation breadth rather than direct evidence that every "
        "candidate independently drives each associated pathway. "
        "Multiple KEGG orthologous or species-specific gene mappings can "
        "associate a BUSCO with the same or multiple pathways.\n\n"
    )

    report.write(
        "Candidates without KEGG pathway mappings should not be treated "
        "as biologically unimportant. Their current annotations may be "
        "domain-level, structural, regulatory, or poorly represented in "
        "the available KEGG mapping. GO and EC annotations should therefore "
        "be considered alongside KEGG pathway coverage.\n\n"
    )

    report.write(
        "The resulting interpretation supports a multi-functional view of "
        "the Carbon Breadth candidate set. Further biological conclusions "
        "should be based on the original Carbon Breadth scores, comparative "
        "phylogenetic evidence, sequence conservation and candidate-specific "
        "functional evidence rather than pathway count alone.\n\n"
    )

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------

    report.write("=" * 80 + "\n")
    report.write("7. REPRODUCIBILITY / INPUTS\n")
    report.write("=" * 80 + "\n\n")

    report.write(
        f"Final integrated annotation:\n"
        f"{FINAL_INTEGRATED}\n\n"
    )

    report.write(
        f"Final pathway annotation:\n"
        f"{FINAL_PATHWAY}\n\n"
    )

    report.write(
        f"Detailed KEGG pathway mapping:\n"
        f"{DETAILED_PATHWAY}\n\n"
    )

    report.write(
        f"Analysis script:\n"
        f"final_carbon_breadth_biological_interpretation_420.py\n"
    )


# =============================================================================
# 26. FINAL CONSOLE SUMMARY
# =============================================================================

print("\n")
print("=" * 80)
print("FINAL CARBON BREADTH INTERPRETATION COMPLETE")
print("=" * 80)

print(
    f"\nTotal candidates              : {total_candidates}"
)

print(
    f"Candidates with GO           : {with_go}"
)

print(
    f"Candidates with KEGG         : {with_kegg}"
)

print(
    f"Candidates with EC           : {with_ec}"
)

print(
    f"Candidates mapped to pathway : {with_pathway}"
)

print(
    f"Unique KEGG pathways         : "
    f"{len(pathway_frequency)}"
)

print("\n" + "=" * 80)
print("OUTPUT DIRECTORY")
print("=" * 80)

print(OUTPUT_DIR)

print("\nTABLES")
print("-" * 80)

for path in [
    CLEAN_MAPPING,
    PATHWAY_MATRIX_FILE,
    PATHWAY_FREQ_FILE,
    CANDIDATE_FUNCTIONAL_FILE,
    CATEGORY_FILE,
    GO_FILE,
    EC_FILE,
    COVERAGE_FILE,
    NETWORK_FILE,
    PATHWAY_CATEGORY_FILE
]:
    print(path)


print("\nFIGURES")
print("-" * 80)

for path in FIGURE_DIR.glob("*.png"):
    print(path)


print("\nREPORT")
print("-" * 80)

print(REPORT_FILE)

print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)