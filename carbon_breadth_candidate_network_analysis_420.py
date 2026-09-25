# ================================================================
# CARBON BREADTH CANDIDATE NETWORK ANALYSIS
# ================================================================
#
# Purpose:
# Build the downstream biological network from the ALREADY COMPLETED
# Carbon Breadth annotation pipeline.
#
# Pipeline already completed:
#   ML -> top 20 BUSCOs -> proteins -> genes -> GO/KEGG/EC
#      -> KEGG pathways -> normalized pathways -> biological themes
#
# This script DOES NOT rerun annotation.
#
# It generates:
#   1. BUSCO-pathway network
#   2. BUSCO-theme network
#   3. Pathway-theme network
#   4. Candidate connectivity ranking
#   5. Pathway connectivity ranking
#   6. Cytoscape node/edge files
#   7. BUSCO x pathway matrix
#   8. Network figure
#   9. Final report
#
# ================================================================

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ================================================================
# CONFIGURATION
# ================================================================

BASE = (
    r"C:\Y1000_chassis_project\results"
    r"\stage5_phylogeny_ml_dataset"
    r"\phylogeny_aware_ml"
    r"\phylogeny_cv"
    r"\model_training"
    r"\feature_interpretation_420"
    r"\functional_annotation_420"
    r"\BUSCO20_annotation"
)

V3_DIR = os.path.join(
    BASE,
    "FINAL_CARBON_BREADTH_BIOLOGICAL_INTERPRETATION_420_V3"
)

TABLE_DIR = os.path.join(V3_DIR, "tables")

# Main V3 biological summary
BUSCO_SUMMARY = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_BUSCO_biological_summary_420_v3.csv"
)

# Normalized pathway summary
PATHWAY_SUMMARY = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_normalized_pathway_summary_420_v3.csv"
)

# BUSCO x pathway matrix
PATHWAY_MATRIX = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_final_BUSCO_pathway_matrix_420_v3.csv"
)

# Final output directory
OUT_DIR = os.path.join(
    BASE,
    "FINAL_CARBON_BREADTH_NETWORK_ANALYSIS_420"
)

TABLE_OUT = os.path.join(OUT_DIR, "tables")
FIG_OUT = os.path.join(OUT_DIR, "figures")
REPORT_OUT = os.path.join(OUT_DIR, "reports")
CYTO_OUT = os.path.join(OUT_DIR, "cytoscape")

for directory in [OUT_DIR, TABLE_OUT, FIG_OUT, REPORT_OUT, CYTO_OUT]:
    os.makedirs(directory, exist_ok=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def clean_text(value):
    """Convert NaN/None to empty string and normalize whitespace."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def split_values(value):
    """
    Split pathway/theme fields safely.

    Handles:
      ;
      |
      //
      comma
    """
    value = clean_text(value)

    if not value:
        return []

    parts = re.split(r"\s*;\s*|\s*\|\s*|\s*//\s*", value)

    result = []

    for p in parts:
        p = p.strip()
        if p and p.lower() not in {"nan", "none"}:
            result.append(p)

    return list(dict.fromkeys(result))


def find_column(df, candidates):
    """Find a column using exact or case-insensitive matching."""

    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    # fallback partial matching
    for c in df.columns:
        cl = str(c).lower()

        for candidate in candidates:
            if candidate.lower() in cl:
                return c

    return None


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("CARBON BREADTH CANDIDATE NETWORK ANALYSIS")
print("=" * 80)

print()
print("This script uses the EXISTING V3 biological interpretation.")
print("No protein, UniProt, GO or KEGG annotation will be rerun.")
print()


# ================================================================
# CHECK INPUT FILES
# ================================================================

print("=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)

required_files = {
    "BUSCO biological summary": BUSCO_SUMMARY,
    "Normalized pathway summary": PATHWAY_SUMMARY,
    "BUSCO pathway matrix": PATHWAY_MATRIX
}

for name, path in required_files.items():

    print(f"{name}:")
    print(path)

    if not os.path.exists(path):
        print()
        print("ERROR: Required file not found.")
        print(path)
        raise SystemExit(1)

    print("  FOUND")
    print()


# ================================================================
# LOAD DATA
# ================================================================

print("=" * 80)
print("LOADING EXISTING V3 RESULTS")
print("=" * 80)

busco_df = pd.read_csv(BUSCO_SUMMARY)
pathway_df = pd.read_csv(PATHWAY_SUMMARY)
matrix_df = pd.read_csv(PATHWAY_MATRIX)

print()
print("BUSCO summary shape:", busco_df.shape)
print("Pathway summary shape:", pathway_df.shape)
print("Pathway matrix shape:", matrix_df.shape)

print()
print("BUSCO summary columns:")
for c in busco_df.columns:
    print(" ", c)

print()
print("Pathway summary columns:")
for c in pathway_df.columns:
    print(" ", c)

print()


# ================================================================
# DETECT IMPORTANT COLUMNS
# ================================================================

busco_col = find_column(
    busco_df,
    ["BUSCO_ID", "BUSCO"]
)

theme_col = find_column(
    busco_df,
    [
        "Biological_Theme",
        "Broad_Biological_Theme",
        "Biological_Theme"
    ]
)

function_col = find_column(
    busco_df,
    [
        "Functional_Description",
        "Function",
        "Functional Description"
    ]
)

mapped_col = find_column(
    busco_df,
    [
        "Pathway_Mapped",
        "Pathway Mapped"
    ]
)

print("=" * 80)
print("DETECTED BUSCO COLUMNS")
print("=" * 80)

print("BUSCO column     :", busco_col)
print("Function column  :", function_col)
print("Theme column     :", theme_col)
print("Mapped column    :", mapped_col)

if busco_col is None:
    raise SystemExit("ERROR: BUSCO_ID column could not be detected.")


# ================================================================
# DETECT PATHWAY INFORMATION
# ================================================================

pathway_name_col = find_column(
    pathway_df,
    [
        "Normalized_Pathway",
        "Normalized_Pathway_Name",
        "Pathway",
        "Pathway_Name"
    ]
)

pathway_count_col = find_column(
    pathway_df,
    [
        "BUSCO_Count",
        "Pathway_BUSCO_Count",
        "BUSCO count"
    ]
)

print()
print("=" * 80)
print("DETECTED PATHWAY COLUMNS")
print("=" * 80)

print("Pathway name column :", pathway_name_col)
print("Pathway count column:", pathway_count_col)

if pathway_name_col is None:
    raise SystemExit(
        "ERROR: Normalized pathway column could not be detected."
    )


# ================================================================
# IDENTIFY PATHWAY COLUMNS IN MATRIX
# ================================================================

matrix_busco_col = find_column(
    matrix_df,
    ["BUSCO_ID", "BUSCO"]
)

if matrix_busco_col is None:
    raise SystemExit(
        "ERROR: BUSCO column not found in pathway matrix."
    )

matrix_pathway_cols = [
    c for c in matrix_df.columns
    if c != matrix_busco_col
]

print()
print("=" * 80)
print("PATHWAY MATRIX")
print("=" * 80)

print("BUSCO column:", matrix_busco_col)
print("Number of pathway columns:", len(matrix_pathway_cols))


# ================================================================
# CREATE BUSCO MASTER TABLE
# ================================================================

print()
print("=" * 80)
print("BUILDING BUSCO MASTER TABLE")
print("=" * 80)

master = busco_df.copy()

master[busco_col] = master[busco_col].astype(str).str.strip()

master["Function"] = (
    master[function_col].apply(clean_text)
    if function_col
    else ""
)

master["Biological_Theme"] = (
    master[theme_col].apply(clean_text)
    if theme_col
    else ""
)

if mapped_col:
    master["Pathway_Mapped"] = (
        master[mapped_col]
        .astype(str)
        .str.lower()
        .isin(["true", "yes", "1"])
    )
else:
    master["Pathway_Mapped"] = False


# ================================================================
# BUILD BUSCO -> PATHWAY EDGES
# ================================================================

print()
print("=" * 80)
print("BUILDING BUSCO × PATHWAY NETWORK")
print("=" * 80)

busco_pathway_records = []

for _, row in matrix_df.iterrows():

    busco = clean_text(row[matrix_busco_col])

    if not busco:
        continue

    for pathway_col in matrix_pathway_cols:

        value = row[pathway_col]

        try:
            numeric_value = float(value)
        except:
            numeric_value = 0

        if numeric_value > 0:

            pathway = clean_text(pathway_col)

            if pathway:

                busco_pathway_records.append({
                    "BUSCO_ID": busco,
                    "Normalized_Pathway": pathway,
                    "Association": 1
                })

busco_pathway_edges = pd.DataFrame(
    busco_pathway_records
).drop_duplicates()

print()
print("BUSCO-pathway edges:", len(busco_pathway_edges))
print(
    "Unique BUSCOs:",
    busco_pathway_edges["BUSCO_ID"].nunique()
    if not busco_pathway_edges.empty else 0
)
print(
    "Unique pathways:",
    busco_pathway_edges["Normalized_Pathway"].nunique()
    if not busco_pathway_edges.empty else 0
)


# ================================================================
# IF MATRIX IS EMPTY, TRY PATHWAY SUMMARY
# ================================================================

if busco_pathway_edges.empty:

    print()
    print("WARNING:")
    print("Pathway matrix produced no edges.")
    print("Trying BUSCO summary pathway fields.")

    fallback_pathway_col = find_column(
        busco_df,
        [
            "Normalized_Pathways",
            "Normalized_Pathway",
            "Pathways",
            "KEGG_Pathway_Names"
        ]
    )

    if fallback_pathway_col:

        records = []

        for _, row in busco_df.iterrows():

            busco = clean_text(row[busco_col])

            pathways = split_values(
                row[fallback_pathway_col]
            )

            for pathway in pathways:

                records.append({
                    "BUSCO_ID": busco,
                    "Normalized_Pathway": pathway,
                    "Association": 1
                })

        busco_pathway_edges = pd.DataFrame(
            records
        ).drop_duplicates()


# ================================================================
# ADD FUNCTION + THEME
# ================================================================

print()
print("=" * 80)
print("ADDING BIOLOGICAL INFORMATION")
print("=" * 80)

busco_metadata = master[
    [
        busco_col,
        "Function",
        "Biological_Theme",
        "Pathway_Mapped"
    ]
].copy()

busco_metadata.columns = [
    "BUSCO_ID",
    "Function",
    "Biological_Theme",
    "Pathway_Mapped"
]

busco_pathway_edges = busco_pathway_edges.merge(
    busco_metadata,
    on="BUSCO_ID",
    how="left"
)


# ================================================================
# BUSCO CONNECTIVITY
# ================================================================

print()
print("=" * 80)
print("CALCULATING CANDIDATE CONNECTIVITY")
print("=" * 80)

if not busco_pathway_edges.empty:

    busco_degree = (
        busco_pathway_edges
        .groupby("BUSCO_ID")
        .agg(
            Pathway_Degree=(
                "Normalized_Pathway",
                "nunique"
            )
        )
        .reset_index()
    )

else:

    busco_degree = pd.DataFrame(
        columns=[
            "BUSCO_ID",
            "Pathway_Degree"
        ]
    )


# Include ALL 20 BUSCOs, including unmapped ones
all_buscos = master[busco_col].astype(str).str.strip().unique()

all_busco_df = pd.DataFrame({
    "BUSCO_ID": all_buscos
})

busco_degree = all_busco_df.merge(
    busco_degree,
    on="BUSCO_ID",
    how="left"
)

busco_degree["Pathway_Degree"] = (
    busco_degree["Pathway_Degree"]
    .fillna(0)
    .astype(int)
)

busco_degree = busco_degree.merge(
    busco_metadata,
    on="BUSCO_ID",
    how="left"
)

busco_degree = busco_degree.sort_values(
    ["Pathway_Degree", "BUSCO_ID"],
    ascending=[False, True]
)

busco_degree["Connectivity_Rank"] = (
    busco_degree["Pathway_Degree"]
    .rank(
        method="dense",
        ascending=False
    )
    .astype(int)
)

print()
print("BUSCO connectivity calculated.")
print()


# ================================================================
# PATHWAY CONNECTIVITY
# ================================================================

print("=" * 80)
print("CALCULATING PATHWAY CONNECTIVITY")
print("=" * 80)

if not busco_pathway_edges.empty:

    pathway_degree = (
        busco_pathway_edges
        .groupby("Normalized_Pathway")
        .agg(
            BUSCO_Degree=(
                "BUSCO_ID",
                "nunique"
            )
        )
        .reset_index()
    )

else:

    pathway_degree = pd.DataFrame(
        columns=[
            "Normalized_Pathway",
            "BUSCO_Degree"
        ]
    )

pathway_degree = pathway_degree.sort_values(
    ["BUSCO_Degree", "Normalized_Pathway"],
    ascending=[False, True]
)

if not pathway_degree.empty:

    pathway_degree["Connectivity_Rank"] = (
        pathway_degree["BUSCO_Degree"]
        .rank(
            method="dense",
            ascending=False
        )
        .astype(int)
    )


print()
print("Pathway connectivity calculated.")
print()


# ================================================================
# BUSCO × THEME NETWORK
# ================================================================

print("=" * 80)
print("BUILDING BUSCO × BIOLOGICAL THEME NETWORK")
print("=" * 80)

busco_theme_edges = (
    master[
        [
            busco_col,
            "Biological_Theme"
        ]
    ]
    .rename(
        columns={
            busco_col: "BUSCO_ID"
        }
    )
)

busco_theme_edges = busco_theme_edges[
    busco_theme_edges["Biological_Theme"].notna()
]

busco_theme_edges = busco_theme_edges[
    busco_theme_edges["Biological_Theme"].astype(str).str.strip() != ""
]

busco_theme_edges = busco_theme_edges.drop_duplicates()

print()
print("BUSCO-theme edges:", len(busco_theme_edges))


# ================================================================
# PATHWAY × THEME NETWORK
# ================================================================

print()
print("=" * 80)
print("BUILDING PATHWAY × BIOLOGICAL THEME NETWORK")
print("=" * 80)

pathway_theme_records = []

for _, edge in busco_pathway_edges.iterrows():

    theme = clean_text(edge["Biological_Theme"])
    pathway = clean_text(edge["Normalized_Pathway"])

    if theme and pathway:

        pathway_theme_records.append({
            "Normalized_Pathway": pathway,
            "Biological_Theme": theme,
            "Association": 1
        })

pathway_theme_edges = pd.DataFrame(
    pathway_theme_records
).drop_duplicates()

print()
print(
    "Pathway-theme edges:",
    len(pathway_theme_edges)
)


# ================================================================
# BUSCO × PATHWAY MATRIX
# ================================================================

print()
print("=" * 80)
print("BUILDING FINAL BUSCO × PATHWAY MATRIX")
print("=" * 80)

if not busco_pathway_edges.empty:

    network_matrix = pd.crosstab(
        busco_pathway_edges["BUSCO_ID"],
        busco_pathway_edges["Normalized_Pathway"]
    )

    network_matrix = network_matrix.reindex(
        all_buscos,
        fill_value=0
    )

else:

    network_matrix = pd.DataFrame(
        index=all_buscos
    )


# ================================================================
# CANDIDATE PRIORITIZATION
# ================================================================

print()
print("=" * 80)
print("BUILDING CANDIDATE PRIORITIZATION")
print("=" * 80)

candidate_priority = master[
    [
        busco_col,
        "Function",
        "Biological_Theme",
        "Pathway_Mapped"
    ]
].copy()

candidate_priority = candidate_priority.rename(
    columns={
        busco_col: "BUSCO_ID"
    }
)

candidate_priority = candidate_priority.merge(
    busco_degree[
        [
            "BUSCO_ID",
            "Pathway_Degree",
            "Connectivity_Rank"
        ]
    ],
    on="BUSCO_ID",
    how="left"
)

candidate_priority["Pathway_Degree"] = (
    candidate_priority["Pathway_Degree"]
    .fillna(0)
    .astype(int)
)

candidate_priority["Pathway_Mapped"] = (
    candidate_priority["Pathway_Mapped"]
    .fillna(False)
)

candidate_priority = candidate_priority.sort_values(
    [
        "Pathway_Mapped",
        "Pathway_Degree",
        "BUSCO_ID"
    ],
    ascending=[
        False,
        False,
        True
    ]
)

candidate_priority["Network_Priority_Rank"] = range(
    1,
    len(candidate_priority) + 1
)


# ================================================================
# SAVE TABLES
# ================================================================

print()
print("=" * 80)
print("SAVING NETWORK TABLES")
print("=" * 80)

outputs = {}

outputs["BUSCO pathway edges"] = os.path.join(
    TABLE_OUT,
    "Carbon_Breadth_BUSCO_pathway_network_edges_420.csv"
)

outputs["BUSCO theme edges"] = os.path.join(
    TABLE_OUT,
    "Carbon_Breadth_BUSCO_theme_network_edges_420.csv"
)

outputs["Pathway theme edges"] = os.path.join(
    TABLE_OUT,
    "Carbon_Breadth_pathway_theme_network_edges_420.csv"
)

outputs["BUSCO connectivity"] = os.path.join(
    TABLE_OUT,
    "Carbon_Breadth_BUSCO_connectivity_420.csv"
)

outputs["Pathway connectivity"] = os.path.join(
    TABLE_OUT,
    "Carbon_Breadth_pathway_connectivity_420.csv"
)

outputs["Candidate prioritization"] = os.path.join(
    TABLE_OUT,
    "Carbon_Breadth_candidate_network_prioritization_420.csv"
)

outputs["BUSCO pathway matrix"] = os.path.join(
    TABLE_OUT,
    "Carbon_Breadth_BUSCO_pathway_network_matrix_420.csv"
)

outputs["BUSCO metadata"] = os.path.join(
    TABLE_OUT,
    "Carbon_Breadth_BUSCO_network_metadata_420.csv"
)


busco_pathway_edges.to_csv(
    outputs["BUSCO pathway edges"],
    index=False
)

busco_theme_edges.to_csv(
    outputs["BUSCO theme edges"],
    index=False
)

pathway_theme_edges.to_csv(
    outputs["Pathway theme edges"],
    index=False
)

busco_degree.to_csv(
    outputs["BUSCO connectivity"],
    index=False
)

pathway_degree.to_csv(
    outputs["Pathway connectivity"],
    index=False
)

candidate_priority.to_csv(
    outputs["Candidate prioritization"],
    index=False
)

network_matrix.to_csv(
    outputs["BUSCO pathway matrix"]
)

busco_metadata.to_csv(
    outputs["BUSCO metadata"],
    index=False
)

for label, path in outputs.items():
    print(f"{label}:")
    print(path)
    print()


# ================================================================
# CYTOSCAPE NODE TABLE
# ================================================================

print("=" * 80)
print("CREATING CYTOSCAPE NODE TABLE")
print("=" * 80)

nodes = []

# BUSCO nodes
for _, row in busco_metadata.iterrows():

    nodes.append({
        "Node_ID": row["BUSCO_ID"],
        "Node_Type": "BUSCO",
        "Label": row["BUSCO_ID"],
        "Function": row["Function"],
        "Biological_Theme": row["Biological_Theme"],
        "Pathway_Mapped": row["Pathway_Mapped"]
    })


# Pathway nodes
for pathway in sorted(
    busco_pathway_edges["Normalized_Pathway"].dropna().unique()
    if not busco_pathway_edges.empty
    else []
):

    nodes.append({
        "Node_ID": "PATHWAY::" + pathway,
        "Node_Type": "Pathway",
        "Label": pathway,
        "Function": "",
        "Biological_Theme": "",
        "Pathway_Mapped": True
    })


# Theme nodes
for theme in sorted(
    busco_theme_edges["Biological_Theme"].dropna().unique()
    if not busco_theme_edges.empty
    else []
):

    nodes.append({
        "Node_ID": "THEME::" + theme,
        "Node_Type": "Biological_Theme",
        "Label": theme,
        "Function": "",
        "Biological_Theme": theme,
        "Pathway_Mapped": True
    })


cytoscape_nodes = pd.DataFrame(nodes)

node_file = os.path.join(
    CYTO_OUT,
    "Carbon_Breadth_Cytoscape_nodes_420.csv"
)

cytoscape_nodes.to_csv(
    node_file,
    index=False
)


# ================================================================
# CYTOSCAPE EDGE TABLE
# ================================================================

print("Creating Cytoscape edge table...")

cyto_edges = []

# BUSCO -> pathway
for _, row in busco_pathway_edges.iterrows():

    cyto_edges.append({
        "Source": row["BUSCO_ID"],
        "Target": "PATHWAY::" + row["Normalized_Pathway"],
        "Interaction": "associated_with",
        "Edge_Type": "BUSCO_Pathway"
    })


# BUSCO -> theme
for _, row in busco_theme_edges.iterrows():

    cyto_edges.append({
        "Source": row["BUSCO_ID"],
        "Target": "THEME::" + row["Biological_Theme"],
        "Interaction": "belongs_to_theme",
        "Edge_Type": "BUSCO_Theme"
    })


# Pathway -> theme
for _, row in pathway_theme_edges.iterrows():

    cyto_edges.append({
        "Source": "PATHWAY::" + row["Normalized_Pathway"],
        "Target": "THEME::" + row["Biological_Theme"],
        "Interaction": "belongs_to_theme",
        "Edge_Type": "Pathway_Theme"
    })


cytoscape_edges = pd.DataFrame(cyto_edges)

edge_file = os.path.join(
    CYTO_OUT,
    "Carbon_Breadth_Cytoscape_edges_420.csv"
)

cytoscape_edges.to_csv(
    edge_file,
    index=False
)

print()
print("Cytoscape nodes:")
print(node_file)

print()
print("Cytoscape edges:")
print(edge_file)


# ================================================================
# NETWORK FIGURE
# ================================================================

print()
print("=" * 80)
print("CREATING NETWORK FIGURE")
print("=" * 80)

if not busco_pathway_edges.empty:

    pathway_counts = (
        busco_pathway_edges
        .groupby("Normalized_Pathway")
        ["BUSCO_ID"]
        .nunique()
        .sort_values(
            ascending=True
        )
    )

    plt.figure(
        figsize=(12, 8)
    )

    pathway_counts.plot(
        kind="barh"
    )

    plt.xlabel(
        "Number of Carbon Breadth candidate BUSCOs"
    )

    plt.ylabel(
        "Normalized KEGG pathway"
    )

    plt.title(
        "Carbon Breadth Candidate–Pathway Associations"
    )

    plt.tight_layout()

    figure_path = os.path.join(
        FIG_OUT,
        "Carbon_Breadth_candidate_pathway_network_420.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Network figure:")
    print(figure_path)

else:

    figure_path = None

    print(
        "No BUSCO-pathway edges available; "
        "figure not generated."
    )


# ================================================================
# NETWORK STATISTICS
# ================================================================

total_candidates = len(master)

mapped_candidates = int(
    busco_pathway_edges["BUSCO_ID"].nunique()
    if not busco_pathway_edges.empty
    else 0
)

unmapped_candidates = (
    total_candidates -
    mapped_candidates
)

unique_pathways = int(
    busco_pathway_edges["Normalized_Pathway"]
    .nunique()
    if not busco_pathway_edges.empty
    else 0
)

unique_themes = int(
    busco_theme_edges["Biological_Theme"]
    .nunique()
    if not busco_theme_edges.empty
    else 0
)

busco_pathway_edge_count = len(
    busco_pathway_edges
)

busco_theme_edge_count = len(
    busco_theme_edges
)

pathway_theme_edge_count = len(
    pathway_theme_edges
)


# ================================================================
# FINAL REPORT
# ================================================================

print()
print("=" * 80)
print("WRITING FINAL NETWORK REPORT")
print("=" * 80)

report_path = os.path.join(
    REPORT_OUT,
    "Carbon_Breadth_candidate_network_analysis_report_420.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "CARBON BREADTH CANDIDATE NETWORK ANALYSIS\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        "This analysis uses the completed Carbon Breadth "
        "biological interpretation pipeline.\n"
    )

    f.write(
        "No new functional annotation was performed.\n\n"
    )

    f.write(
        "NETWORK COVERAGE\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        f"Total candidate BUSCOs       : "
        f"{total_candidates}\n"
    )

    f.write(
        f"Pathway-mapped BUSCOs        : "
        f"{mapped_candidates}\n"
    )

    f.write(
        f"Not pathway-mapped BUSCOs    : "
        f"{unmapped_candidates}\n"
    )

    f.write(
        f"Unique normalized pathways   : "
        f"{unique_pathways}\n"
    )

    f.write(
        f"Unique biological themes     : "
        f"{unique_themes}\n"
    )

    f.write(
        f"BUSCO-pathway edges          : "
        f"{busco_pathway_edge_count}\n"
    )

    f.write(
        f"BUSCO-theme edges            : "
        f"{busco_theme_edge_count}\n"
    )

    f.write(
        f"Pathway-theme edges          : "
        f"{pathway_theme_edge_count}\n\n"
    )

    f.write(
        "INTERPRETATION\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "The network represents associations between "
        "the 20 Carbon Breadth candidate BUSCOs, "
        "their normalized KEGG pathways, and the "
        "broad biological themes assigned during "
        "the V3 biological interpretation.\n\n"
    )

    f.write(
        "Pathway degree represents the number of "
        "normalized pathways associated with a BUSCO. "
        "It is a network connectivity measure and "
        "should not be interpreted as a statistical "
        "significance score.\n\n"
    )

    f.write(
        "The 11 candidates without recovered KEGG "
        "pathways are retained in the candidate table "
        "and are not considered biologically irrelevant. "
        "Their lack of pathway connectivity reflects "
        "the current annotation coverage.\n\n"
    )

    f.write(
        "CANDIDATE CONNECTIVITY\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for _, row in busco_degree.iterrows():

        f.write(
            f"{row['BUSCO_ID']} | "
            f"Pathway degree={row['Pathway_Degree']} | "
            f"Mapped={row['Pathway_Mapped']} | "
            f"Theme={row['Biological_Theme']}\n"
        )

    f.write("\n")

    f.write(
        "MOST CONNECTED PATHWAYS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    for _, row in pathway_degree.head(20).iterrows():

        f.write(
            f"{row['Normalized_Pathway']} | "
            f"BUSCO degree={row['BUSCO_Degree']}\n"
        )

    f.write("\n")

    f.write(
        "BIOLOGICAL THEMES\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    theme_summary = (
        busco_theme_edges
        .groupby("Biological_Theme")
        ["BUSCO_ID"]
        .nunique()
        .sort_values(
            ascending=False
        )
    )

    for theme, count in theme_summary.items():

        f.write(
            f"{theme} | "
            f"BUSCO count={count}\n"
        )

    f.write("\n")

    f.write(
        "IMPORTANT LIMITATIONS\n"
    )

    f.write(
        "-" * 80 + "\n"
    )

    f.write(
        "1. Network degree is descriptive and does not "
        "represent causal importance.\n"
    )

    f.write(
        "2. Pathway association does not establish "
        "functional causality for the Carbon Breadth trait.\n"
    )

    f.write(
        "3. KEGG annotation coverage varies among candidates.\n"
    )

    f.write(
        "4. Candidates without recovered pathway mappings "
        "should not be discarded solely on that basis.\n"
    )

    f.write(
        "5. Independent biological validation is still "
        "required before experimental chassis selection.\n"
    )


# ================================================================
# FINAL OUTPUT
# ================================================================

print()
print("=" * 80)
print("CARBON BREADTH CANDIDATE NETWORK ANALYSIS COMPLETE")
print("=" * 80)

print()
print(f"Total candidate BUSCOs    : {total_candidates}")
print(f"Pathway-mapped BUSCOs     : {mapped_candidates}")
print(f"Not pathway-mapped        : {unmapped_candidates}")
print(f"Unique normalized pathways: {unique_pathways}")
print(f"Biological themes         : {unique_themes}")
print(f"BUSCO-pathway edges       : {busco_pathway_edge_count}")

print()
print("=" * 80)
print("OUTPUT DIRECTORY")
print("=" * 80)

print(OUT_DIR)

print()
print("TABLES:")
for label, path in outputs.items():
    print(path)

print()
print("CYTOSCAPE:")
print(node_file)
print(edge_file)

print()
print("FIGURE:")
if figure_path:
    print(figure_path)

print()
print("REPORT:")
print(report_path)

print()
print("=" * 80)
print("NEXT STAGE: CANDIDATE CHASSIS PRIORITIZATION")
print("=" * 80)