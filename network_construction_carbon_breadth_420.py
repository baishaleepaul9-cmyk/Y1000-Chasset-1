# ================================================================
# CARBON BREADTH — EVIDENCE-AWARE NETWORK CONSTRUCTION
# Corrected version
#
# Fixes:
#   1. Top-20 Feature -> BUSCO_ID handling
#   2. Explicit evidence-aware pathway counting
#   3. Windows long-path figure saving
#   4. Prevents functional descriptions/categories from
#      being incorrectly interpreted as pathway evidence
# ================================================================

from pathlib import Path
import json
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")


# ================================================================
# INPUT FILES
# ================================================================

TOP20_FEATURES = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
    / "tables"
    / "Carbon_Breadth_top20_BUSCO_features_420.csv"
)

BUSCO_ANNOTATION = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
    / "functional_annotation_420"
    / "tables"
    / "Carbon_Breadth_BUSCO_functional_annotation_420.csv"
)

PATHWAY_INTEGRATION = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
    / "functional_annotation_420"
    / "pathway_mapping_420"
    / "pathway_level_integration_420"
    / "tables"
    / "Carbon_Breadth_pathway_level_integrated_candidates_420.csv"
)

EXTERNAL_ANNOTATION = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
    / "functional_annotation_420"
    / "pathway_mapping_420"
    / "external_annotation_420"
    / "tables"
    / "Carbon_Breadth_external_functional_annotation_420.csv"
)


# ================================================================
# OUTPUT DIRECTORY
# ================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
    / "network_analysis_420"
)

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# SHORT FIGURE FALLBACK
# ================================================================

SHORT_FIGURE_DIR = PROJECT_ROOT / "network_figures_420"
SHORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# OUTPUT FILES
# ================================================================

NODES_OUT = (
    TABLE_DIR / "Carbon_Breadth_network_nodes_420.csv"
)

EDGES_OUT = (
    TABLE_DIR / "Carbon_Breadth_network_edges_420.csv"
)

EVIDENCE_OUT = (
    TABLE_DIR / "Carbon_Breadth_candidate_evidence_420.csv"
)

FUNCTIONAL_GROUPS_OUT = (
    TABLE_DIR / "Carbon_Breadth_functional_groups_420.csv"
)

INTERPRETATION_OUT = (
    TABLE_DIR / "Carbon_Breadth_candidate_chassis_interpretation_420.csv"
)

SUMMARY_OUT = (
    TABLE_DIR / "Carbon_Breadth_network_summary_420.json"
)

REPORT_OUT = (
    REPORT_DIR / "Carbon_Breadth_network_analysis_report_420.txt"
)

FIGURE_OUT = (
    FIGURE_DIR / "Carbon_Breadth_candidate_function_network_420.png"
)

SHORT_FIGURE_OUT = (
    SHORT_FIGURE_DIR / "CB_network_420.png"
)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def banner(text):
    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


def read_csv(path, name):
    require_file(path)

    print(f"\n{name}:")
    print(path)

    df = pd.read_csv(path)

    print(f"Shape: {df.shape}")

    return df


def normalize_bool(value):
    """
    Convert common TRUE/FALSE representations to bool.

    Importantly:
    missing/unknown values are False.
    """

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    value = str(value).strip().lower()

    return value in {
        "true",
        "1",
        "yes",
        "y",
        "supported",
        "present"
    }


def first_existing_column(df, candidates):

    for col in candidates:
        if col in df.columns:
            return col

    return None


# ================================================================
# START
# ================================================================

banner(
    "CARBON BREADTH — EVIDENCE-AWARE NETWORK CONSTRUCTION"
)

print("Project root:")
print(PROJECT_ROOT)


# ================================================================
# 1. LOAD TOP-20 FEATURES
# ================================================================

banner("LOADING TOP-20 BUSCO FEATURE IMPORTANCE")

top20 = read_csv(
    TOP20_FEATURES,
    "Top-20 feature file"
)

print("\nColumns:")
print(top20.columns.tolist())


# ------------------------------------------------
# Handle Feature vs BUSCO_ID
# ------------------------------------------------

if "BUSCO_ID" in top20.columns:

    top20["BUSCO_ID"] = (
        top20["BUSCO_ID"]
        .astype(str)
        .str.strip()
    )

elif "Feature" in top20.columns:

    print(
        "\nTop-20 file contains 'Feature' instead of "
        "'BUSCO_ID'."
    )

    top20["BUSCO_ID"] = (
        top20["Feature"]
        .astype(str)
        .str.strip()
    )

else:

    raise KeyError(
        "Top-20 feature file must contain either "
        "'BUSCO_ID' or 'Feature'."
    )


# Remove invalid IDs

top20 = top20[
    top20["BUSCO_ID"].notna()
].copy()

top20 = top20[
    top20["BUSCO_ID"].astype(str).str.len() > 0
].copy()


# Unique candidates

candidate_ids = (
    top20["BUSCO_ID"]
    .drop_duplicates()
    .tolist()
)


print("\n" + "=" * 80)
print("TOP-20 BUSCO CANDIDATES")
print("=" * 80)

print(candidate_ids)

print(
    f"\nUnique predictive candidates: "
    f"{len(candidate_ids)}"
)


# ================================================================
# 2. LOAD BUSCO FUNCTIONAL ANNOTATION
# ================================================================

banner("LOADING BUSCO FUNCTIONAL ANNOTATION")

busco = read_csv(
    BUSCO_ANNOTATION,
    "BUSCO annotation"
)

print("\nColumns:")
print(busco.columns.tolist())


if "BUSCO_ID" not in busco.columns:
    raise KeyError(
        "BUSCO annotation table must contain BUSCO_ID."
    )


busco["BUSCO_ID"] = (
    busco["BUSCO_ID"]
    .astype(str)
    .str.strip()
)


# Keep only predictive candidates

busco = busco[
    busco["BUSCO_ID"].isin(candidate_ids)
].copy()


print(
    f"\nRows belonging to candidate BUSCOs: "
    f"{len(busco)}"
)

print(
    f"Unique candidate BUSCOs represented: "
    f"{busco['BUSCO_ID'].nunique()}"
)


# ================================================================
# 3. LOAD PATHWAY-LEVEL INTEGRATION
# ================================================================

banner("LOADING PATHWAY-LEVEL INTEGRATION")

pathway = read_csv(
    PATHWAY_INTEGRATION,
    "Pathway-level integration"
)

print("\nColumns:")
print(pathway.columns.tolist())


if "BUSCO_ID" not in pathway.columns:
    raise KeyError(
        "Pathway integration table must contain BUSCO_ID."
    )


pathway["BUSCO_ID"] = (
    pathway["BUSCO_ID"]
    .astype(str)
    .str.strip()
)


pathway = pathway[
    pathway["BUSCO_ID"].isin(candidate_ids)
].copy()


print(
    f"\nUnique integrated BUSCOs: "
    f"{pathway['BUSCO_ID'].nunique()}"
)


# ================================================================
# 4. LOAD EXTERNAL ANNOTATION
# ================================================================

banner("LOADING EXTERNAL FUNCTIONAL ANNOTATION")

external = read_csv(
    EXTERNAL_ANNOTATION,
    "External annotation"
)

print("\nColumns:")
print(external.columns.tolist())


if "BUSCO_ID" not in external.columns:
    raise KeyError(
        "External annotation table must contain BUSCO_ID."
    )


external["BUSCO_ID"] = (
    external["BUSCO_ID"]
    .astype(str)
    .str.strip()
)


external = external[
    external["BUSCO_ID"].isin(candidate_ids)
].copy()


print(
    f"\nUnique externally annotated BUSCOs: "
    f"{external['BUSCO_ID'].nunique()}"
)


# ================================================================
# 5. REDUCE BUSCO ANNOTATION TO ONE ROW / BUSCO
# ================================================================

banner("REDUCING BUSCO ANNOTATION")

# Prefer the row with highest importance if available.

if "Mean_Importance" in busco.columns:

    busco["Mean_Importance_numeric"] = pd.to_numeric(
        busco["Mean_Importance"],
        errors="coerce"
    )

    busco_reduced = (
        busco
        .sort_values(
            ["BUSCO_ID", "Mean_Importance_numeric"],
            ascending=[True, False]
        )
        .drop_duplicates(
            "BUSCO_ID"
        )
        .copy()
    )

else:

    busco_reduced = (
        busco
        .drop_duplicates("BUSCO_ID")
        .copy()
    )


print(
    f"Unique annotated BUSCOs: "
    f"{busco_reduced['BUSCO_ID'].nunique()}"
)


# ================================================================
# 6. REDUCE EXTERNAL ANNOTATION
# ================================================================

banner("REDUCING EXTERNAL ANNOTATION")

external_reduced = (
    external
    .drop_duplicates("BUSCO_ID")
    .copy()
)


print(
    f"Unique externally annotated BUSCOs: "
    f"{external_reduced['BUSCO_ID'].nunique()}"
)


# ================================================================
# 7. REDUCE PATHWAY INTEGRATION
# ================================================================

banner("REDUCING PATHWAY INTEGRATION")

pathway_reduced = (
    pathway
    .drop_duplicates("BUSCO_ID")
    .copy()
)


print(
    f"Unique integrated BUSCOs: "
    f"{pathway_reduced['BUSCO_ID'].nunique()}"
)


# ================================================================
# 8. BUILD MASTER TABLE
# ================================================================

banner(
    "MERGING PREDICTIVE AND FUNCTIONAL EVIDENCE"
)


# ------------------------------------------------
# Start with Top-20 predictive features
# ------------------------------------------------

master = top20.copy()


# ------------------------------------------------
# Merge BUSCO annotation
# ------------------------------------------------

busco_merge_cols = [
    c for c in [
        "BUSCO_ID",
        "Functional_Description",
        "Species",
        "Assembly_Accession",
        "Status",
        "Sequence",
        "Start",
        "End",
        "Strand",
        "Score",
        "Length",
        "OrthoDB_URL"
    ]
    if c in busco_reduced.columns
]


busco_merge = busco_reduced[
    busco_merge_cols
].copy()


master = master.merge(
    busco_merge,
    on="BUSCO_ID",
    how="left",
    suffixes=("", "_BUSCO")
)


# ================================================================
# 9. MERGE PATHWAY INTEGRATION
# ================================================================

pathway_merge_cols = [
    c for c in [
        "BUSCO_ID",
        "Stable_Candidate",
        "OrthoDB_Query_Success",
        "HTTP_Status",
        "OrthoDB_Name",
        "Functional_Category",
        "GO_Molecular_Function",
        "GO_Biological_Process",
        "GO_Cellular_Component",
        "EC_Number",
        "KEGG_Pathway",
        "InterPro_Domains",
        "Phyletic_Profile",
        "Evolutionary_Rate",
        "External_Annotation_Status",
        "Pathway_Evidence_Status",
        "InterPro_Evidence_Status",
        "Publication_Interpretation_Level",
        "Query_Error",
        "Pathway_Supported",
        "GO_Supported",
        "InterPro_Supported",
        "External_Evidence_Text",
        "Evidence_Class",
        "Evidence_Strength",
        "Biological_Interpretation"
    ]
    if c in pathway_reduced.columns
]


pathway_merge = pathway_reduced[
    pathway_merge_cols
].copy()


master = master.merge(
    pathway_merge,
    on="BUSCO_ID",
    how="left",
    suffixes=("", "_PATHWAY")
)


# ================================================================
# 10. MERGE EXTERNAL ANNOTATION
# ================================================================

external_merge_cols = [
    c for c in [
        "BUSCO_ID",
        "OrthoDB_Query_Success",
        "HTTP_Status",
        "OrthoDB_Name",
        "Functional_Category",
        "GO_Molecular_Function",
        "GO_Biological_Process",
        "GO_Cellular_Component",
        "EC_Number",
        "KEGG_Pathway",
        "InterPro_Domains",
        "Phyletic_Profile",
        "Evolutionary_Rate",
        "External_Annotation_Status",
        "Pathway_Evidence_Status",
        "InterPro_Evidence_Status",
        "Publication_Interpretation_Level",
        "Query_Error"
    ]
    if c in external_reduced.columns
]


external_merge = external_reduced[
    external_merge_cols
].copy()


# Avoid duplicate column problems where possible

external_merge = external_merge.rename(
    columns={
        c: f"{c}_EXTERNAL"
        for c in external_merge.columns
        if c != "BUSCO_ID"
        and c in master.columns
    }
)


master = master.merge(
    external_merge,
    on="BUSCO_ID",
    how="left"
)


# ================================================================
# 11. CLEAN DUPLICATE / SUFFIXED COLUMNS
# ================================================================

# ------------------------------------------------
# Resolve evidence columns
# ------------------------------------------------

def resolve_column(master_df, base_name):

    if base_name in master_df.columns:
        return base_name

    possible = [
        f"{base_name}_PATHWAY",
        f"{base_name}_EXTERNAL",
        f"{base_name}_BUSCO"
    ]

    for c in possible:
        if c in master_df.columns:
            return c

    return None


go_col = resolve_column(
    master,
    "GO_Supported"
)

interpro_col = resolve_column(
    master,
    "InterPro_Supported"
)

pathway_col = resolve_column(
    master,
    "Pathway_Supported"
)


print("\nResolved evidence columns:")

print(
    "GO_Supported       :",
    go_col
)

print(
    "InterPro_Supported :",
    interpro_col
)

print(
    "Pathway_Supported  :",
    pathway_col
)


# ================================================================
# 12. CREATE EXPLICIT EVIDENCE FLAGS
# ================================================================

banner(
    "CALCULATING FUNCTIONAL EVIDENCE FLAGS"
)


if go_col is not None:

    master["GO_Supported_Final"] = (
        master[go_col]
        .apply(normalize_bool)
    )

else:

    master["GO_Supported_Final"] = False


if interpro_col is not None:

    master["InterPro_Supported_Final"] = (
        master[interpro_col]
        .apply(normalize_bool)
    )

else:

    master["InterPro_Supported_Final"] = False


if pathway_col is not None:

    master["Pathway_Supported_Final"] = (
        master[pathway_col]
        .apply(normalize_bool)
    )

else:

    master["Pathway_Supported_Final"] = False


# ================================================================
# IMPORTANT:
#
# We DO NOT infer pathway evidence from:
#
#   Functional_Description
#   Functional_Category
#   OrthoDB_Name
#   GO terms merely existing
#   InterPro domains merely existing
#
# Only the explicit Pathway_Supported field counts.
# ================================================================


master["External_Evidence_Count"] = (
    master["GO_Supported_Final"].astype(int)
    +
    master["InterPro_Supported_Final"].astype(int)
    +
    master["Pathway_Supported_Final"].astype(int)
)


master["Any_External_Functional_Evidence"] = (
    master["External_Evidence_Count"] > 0
)


# ================================================================
# 13. STABILITY
# ================================================================

if "Fold_Stability" in master.columns:

    master["Fold_Stability_numeric"] = pd.to_numeric(
        master["Fold_Stability"],
        errors="coerce"
    )

else:

    master["Fold_Stability_numeric"] = np.nan


master["Stable_Predictive"] = (
    master["Fold_Stability_numeric"] >= 0.8
)


# ================================================================
# 14. FUNCTIONAL CATEGORY
# ================================================================

banner(
    "ASSIGNING FUNCTIONAL GROUPS"
)


def classify_function(row):

    category = str(
        row.get(
            "Functional_Category",
            ""
        )
    ).lower()

    description = str(
        row.get(
            "Functional_Description",
            ""
        )
    ).lower()

    combined = (
        category
        + " "
        + description
    )

    if any(
        x in combined
        for x in [
            "rna helicase",
            "rna processing",
            "rna",
            "mrna",
            "trna",
            "ribosomal",
            "methyltransferase"
        ]
    ):
        return "RNA processing / gene expression"

    if any(
        x in combined
        for x in [
            "kinesin",
            "cytoskeleton",
            "actin",
            "tubulin",
            "armadillo",
            "wd40"
        ]
    ):
        return "Cytoskeleton / cellular organization"

    if any(
        x in combined
        for x in [
            "metabolic",
            "metabolism",
            "aconitase",
            "dehydratase",
            "enzyme",
            "biosynthesis"
        ]
    ):
        return "Metabolism"

    if any(
        x in combined
        for x in [
            "membrane",
            "gpi",
            "vesicle",
            "trafficking",
            "snf7"
        ]
    ):
        return "Membrane trafficking / cellular organization"

    if any(
        x in combined
        for x in [
            "domain",
            "pdz",
            "sant",
            "myb",
            "regulatory"
        ]
    ):
        return "Protein domain / regulatory function"

    if any(
        x in combined
        for x in [
            "atpase",
            "helicase",
            "phosphatase",
            "enzymatic"
        ]
    ):
        return "Enzymatic / metabolic activity"

    return "Other / unclear functional assignment"


master["Functional_Group"] = (
    master.apply(
        classify_function,
        axis=1
    )
)


# ================================================================
# 15. CANDIDATE CLASSIFICATION
# ================================================================

banner(
    "CLASSIFYING CANDIDATES"
)


def classify_candidate(row):

    stable = bool(
        row["Stable_Predictive"]
    )

    evidence = int(
        row["External_Evidence_Count"]
    )

    if stable and evidence > 0:

        return (
            "Stable predictive candidate "
            "- externally functionally supported"
        )

    if stable and evidence == 0:

        return (
            "Stable predictive candidate "
            "- functional validation required"
        )

    if (not stable) and evidence > 0:

        return (
            "Functional candidate "
            "- predictive stability requires validation"
        )

    return (
        "Exploratory candidate "
        "- requires functional validation"
    )


master["Candidate_Classification"] = (
    master.apply(
        classify_candidate,
        axis=1
    )
)


# ================================================================
# 16. EVIDENCE SUMMARY
# ================================================================

total_candidates = len(master)

stable_candidates = int(
    master["Stable_Predictive"].sum()
)

go_supported = int(
    master["GO_Supported_Final"].sum()
)

interpro_supported = int(
    master["InterPro_Supported_Final"].sum()
)

pathway_supported = int(
    master["Pathway_Supported_Final"].sum()
)

external_supported = int(
    master["Any_External_Functional_Evidence"].sum()
)


banner(
    "CORRECTED NETWORK EVIDENCE SUMMARY"
)

print(
    f"Total predictive candidates       : "
    f"{total_candidates}"
)

print(
    f"Stable predictive candidates     : "
    f"{stable_candidates}"
)

print(
    f"GO-supported                     : "
    f"{go_supported}"
)

print(
    f"InterPro-supported               : "
    f"{interpro_supported}"
)

print(
    f"Pathway-supported                : "
    f"{pathway_supported}"
)

print(
    f"Any external functional evidence : "
    f"{external_supported}"
)


# ================================================================
# 17. CANDIDATE EVIDENCE TABLE
# ================================================================

evidence_columns = [
    "BUSCO_ID",
    "Mean_Importance",
    "Fold_Stability",
    "Stable_Predictive",
    "GO_Supported_Final",
    "InterPro_Supported_Final",
    "Pathway_Supported_Final",
    "External_Evidence_Count",
    "Any_External_Functional_Evidence",
    "Functional_Group",
    "Candidate_Classification"
]


evidence_columns = [
    c
    for c in evidence_columns
    if c in master.columns
]


candidate_evidence = master[
    evidence_columns
].copy()


candidate_evidence.to_csv(
    EVIDENCE_OUT,
    index=False
)


# ================================================================
# 18. NETWORK NODES
# ================================================================

banner(
    "CREATING NETWORK NODES"
)


nodes = []


# Candidate nodes

for _, row in master.iterrows():

    busco_id = row["BUSCO_ID"]

    nodes.append({
        "Node_ID": busco_id,
        "Node_Type": "BUSCO",
        "Label": busco_id,
        "Functional_Group":
            row.get(
                "Functional_Group",
                ""
            ),
        "Mean_Importance":
            row.get(
                "Mean_Importance",
                np.nan
            ),
        "Fold_Stability":
            row.get(
                "Fold_Stability",
                np.nan
            ),
        "Stable_Predictive":
            row.get(
                "Stable_Predictive",
                False
            ),
        "External_Evidence_Count":
            row.get(
                "External_Evidence_Count",
                0
            )
    })


# Evidence nodes only when actual evidence exists

if go_supported > 0:

    nodes.append({
        "Node_ID": "GO_FUNCTIONAL_EVIDENCE",
        "Node_Type": "Evidence",
        "Label": "GO functional evidence",
        "Functional_Group": "GO",
        "Mean_Importance": np.nan,
        "Fold_Stability": np.nan,
        "Stable_Predictive": False,
        "External_Evidence_Count": go_supported
    })


if interpro_supported > 0:

    nodes.append({
        "Node_ID": "INTERPRO_DOMAIN_EVIDENCE",
        "Node_Type": "Evidence",
        "Label": "InterPro domain evidence",
        "Functional_Group": "InterPro",
        "Mean_Importance": np.nan,
        "Fold_Stability": np.nan,
        "Stable_Predictive": False,
        "External_Evidence_Count": interpro_supported
    })


if pathway_supported > 0:

    nodes.append({
        "Node_ID": "PATHWAY_EVIDENCE",
        "Node_Type": "Evidence",
        "Label": "Pathway evidence",
        "Functional_Group": "Pathway",
        "Mean_Importance": np.nan,
        "Fold_Stability": np.nan,
        "Stable_Predictive": False,
        "External_Evidence_Count": pathway_supported
    })


nodes_df = pd.DataFrame(nodes)

nodes_df.to_csv(
    NODES_OUT,
    index=False
)


# ================================================================
# 19. NETWORK EDGES
# ================================================================

banner(
    "CREATING NETWORK EDGES"
)


edges = []


for _, row in master.iterrows():

    busco_id = row["BUSCO_ID"]


    # -------------------------
    # GO edge
    # -------------------------

    if row["GO_Supported_Final"]:

        edges.append({
            "Source": busco_id,
            "Target": "GO_FUNCTIONAL_EVIDENCE",
            "Edge_Type": "GO_support",
            "Evidence": "GO"
        })


    # -------------------------
    # InterPro edge
    # -------------------------

    if row["InterPro_Supported_Final"]:

        edges.append({
            "Source": busco_id,
            "Target": "INTERPRO_DOMAIN_EVIDENCE",
            "Edge_Type": "InterPro_support",
            "Evidence": "InterPro"
        })


    # -------------------------
    # Pathway edge
    # -------------------------

    if row["Pathway_Supported_Final"]:

        edges.append({
            "Source": busco_id,
            "Target": "PATHWAY_EVIDENCE",
            "Edge_Type": "Pathway_support",
            "Evidence": "Pathway"
        })


edges_df = pd.DataFrame(
    edges,
    columns=[
        "Source",
        "Target",
        "Edge_Type",
        "Evidence"
    ]
)


edges_df.to_csv(
    EDGES_OUT,
    index=False
)


# ================================================================
# 20. FUNCTIONAL GROUP SUMMARY
# ================================================================

banner(
    "FUNCTIONAL GROUP SUMMARY"
)


group_summary = (
    master
    .groupby(
        "Functional_Group",
        dropna=False
    )
    .agg(
        Candidate_Count=(
            "BUSCO_ID",
            "nunique"
        ),
        Mean_Importance=(
            "Mean_Importance",
            "mean"
        ),
        Mean_Fold_Stability=(
            "Fold_Stability_numeric",
            "mean"
        ),
        GO_Supported=(
            "GO_Supported_Final",
            "sum"
        ),
        InterPro_Supported=(
            "InterPro_Supported_Final",
            "sum"
        ),
        Pathway_Supported=(
            "Pathway_Supported_Final",
            "sum"
        )
    )
    .reset_index()
)


group_summary.to_csv(
    FUNCTIONAL_GROUPS_OUT,
    index=False
)


print(
    group_summary.to_string(
        index=False
    )
)


# ================================================================
# 21. CANDIDATE CHASSIS INTERPRETATION
# ================================================================

banner(
    "CANDIDATE CHASSIS INTERPRETATION"
)


interpretation_rows = []


for _, row in master.iterrows():

    evidence_count = int(
        row["External_Evidence_Count"]
    )

    if evidence_count >= 2:

        interpretation = (
            "Predictive candidate with "
            "multiple external functional "
            "evidence types; suitable for "
            "priority biological validation."
        )

    elif evidence_count == 1:

        interpretation = (
            "Predictive candidate with one "
            "external functional evidence type; "
            "requires targeted validation."
        )

    elif row["Stable_Predictive"]:

        interpretation = (
            "Stable predictive candidate "
            "without current external functional "
            "support; requires gene/protein-level "
            "functional annotation."
        )

    else:

        interpretation = (
            "Exploratory predictive candidate; "
            "requires additional predictive and "
            "functional validation."
        )


    interpretation_rows.append({

        "BUSCO_ID":
            row["BUSCO_ID"],

        "Mean_Importance":
            row.get(
                "Mean_Importance",
                np.nan
            ),

        "Fold_Stability":
            row.get(
                "Fold_Stability",
                np.nan
            ),

        "Stable_Predictive":
            row["Stable_Predictive"],

        "Functional_Group":
            row["Functional_Group"],

        "GO_Supported":
            row["GO_Supported_Final"],

        "InterPro_Supported":
            row["InterPro_Supported_Final"],

        "Pathway_Supported":
            row["Pathway_Supported_Final"],

        "External_Evidence_Count":
            evidence_count,

        "Candidate_Classification":
            row["Candidate_Classification"],

        "Chassis_Interpretation":
            interpretation

    })


interpretation_df = pd.DataFrame(
    interpretation_rows
)


interpretation_df.to_csv(
    INTERPRETATION_OUT,
    index=False
)


# ================================================================
# 22. GENERATE NETWORK FIGURE
# ================================================================

banner(
    "GENERATING NETWORK FIGURE"
)


# We deliberately create a simple evidence network rather than
# relying on NetworkX, so the script has fewer dependencies.

fig, ax = plt.subplots(
    figsize=(14, 10)
)


# ------------------------------------------------
# Candidate positions
# ------------------------------------------------

candidate_master = master.copy()

n = len(candidate_master)

if n == 0:

    ax.text(
        0.5,
        0.5,
        "No candidate nodes available",
        ha="center",
        va="center"
    )

else:

    angles = np.linspace(
        0,
        2 * np.pi,
        n,
        endpoint=False
    )

    radius = 1.0

    positions = {}

    for i, (_, row) in enumerate(
        candidate_master.iterrows()
    ):

        x = radius * np.cos(
            angles[i]
        )

        y = radius * np.sin(
            angles[i]
        )

        positions[
            row["BUSCO_ID"]
        ] = (x, y)


    # ------------------------------------------------
    # Evidence node positions
    # ------------------------------------------------

    evidence_positions = {}

    evidence_nodes = []

    if go_supported > 0:

        evidence_nodes.append(
            (
                "GO_FUNCTIONAL_EVIDENCE",
                "GO functional evidence"
            )
        )

    if interpro_supported > 0:

        evidence_nodes.append(
            (
                "INTERPRO_DOMAIN_EVIDENCE",
                "InterPro domain evidence"
            )
        )

    if pathway_supported > 0:

        evidence_nodes.append(
            (
                "PATHWAY_EVIDENCE",
                "Pathway evidence"
            )
        )


    for i, (
        node_id,
        label
    ) in enumerate(evidence_nodes):

        y = (
            0.75
            - i * 0.75
        )

        x = 2.0

        evidence_positions[
            node_id
        ] = (x, y)


    # ------------------------------------------------
    # Draw edges
    # ------------------------------------------------

    for _, edge in edges_df.iterrows():

        source = edge["Source"]

        target = edge["Target"]

        if (
            source not in positions
            or target not in evidence_positions
        ):

            continue

        x1, y1 = positions[source]

        x2, y2 = evidence_positions[target]

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=1
        )


    # ------------------------------------------------
    # Draw BUSCO nodes
    # ------------------------------------------------

    for _, row in candidate_master.iterrows():

        busco_id = row["BUSCO_ID"]

        x, y = positions[
            busco_id
        ]

        if row["External_Evidence_Count"] > 0:

            size = 160

        elif row["Stable_Predictive"]:

            size = 120

        else:

            size = 80


        ax.scatter(
            x,
            y,
            s=size,
            zorder=3
        )

        ax.text(
            x,
            y,
            busco_id,
            fontsize=7,
            ha="center",
            va="center"
        )


    # ------------------------------------------------
    # Draw evidence nodes
    # ------------------------------------------------

    for node_id, label in evidence_nodes:

        x, y = evidence_positions[
            node_id
        ]

        ax.scatter(
            x,
            y,
            s=400,
            marker="s",
            zorder=4
        )

        ax.text(
            x + 0.08,
            y,
            label,
            fontsize=9,
            ha="left",
            va="center"
        )


ax.set_title(
    "Carbon Breadth — Evidence-Aware Candidate Function Network"
)

ax.axis("off")

plt.tight_layout()


# ================================================================
# 23. SAFE FIGURE SAVE
# ================================================================

figure_saved = False

try:

    fig.savefig(
        FIGURE_OUT,
        dpi=300,
        bbox_inches="tight"
    )

    figure_saved = True

    print("\nFigure saved:")
    print(FIGURE_OUT)

except OSError as e:

    print(
        "\nWARNING: Could not save figure to the "
        "long project path."
    )

    print(
        f"Original error: {e}"
    )

    print(
        "\nTrying short Windows-safe path..."
    )

    try:

        fig.savefig(
            SHORT_FIGURE_OUT,
            dpi=300,
            bbox_inches="tight"
        )

        figure_saved = True

        print(
            "\nFigure saved to short path:"
        )

        print(
            SHORT_FIGURE_OUT
        )

    except OSError as e2:

        print(
            "\nERROR: Figure could not be saved."
        )

        print(
            f"Fallback error: {e2}"
        )


plt.close(fig)


# ================================================================
# 24. SUMMARY JSON
# ================================================================

summary = {

    "total_predictive_candidates":
        total_candidates,

    "stable_predictive_candidates":
        stable_candidates,

    "GO_supported":
        go_supported,

    "InterPro_supported":
        interpro_supported,

    "Pathway_supported":
        pathway_supported,

    "Any_external_functional_evidence":
        external_supported,

    "network_nodes":
        len(nodes_df),

    "network_edges":
        len(edges_df),

    "figure_saved":
        figure_saved,

    "figure_path":
        str(
            FIGURE_OUT
            if FIGURE_OUT.exists()
            else SHORT_FIGURE_OUT
        ),

    "important_note":
        (
            "Pathway support is counted only from the "
            "explicit Pathway_Supported field. "
            "Functional descriptions, categories, "
            "OrthoDB names, GO terms, and InterPro "
            "domains are not automatically treated as "
            "pathway evidence."
        )
}


with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ================================================================
# 25. REPORT
# ================================================================

report_lines = [

    "CARBON BREADTH — EVIDENCE-AWARE NETWORK ANALYSIS",
    "",
    "=" * 80,
    "",
    f"Total predictive candidates       : {total_candidates}",
    f"Stable predictive candidates     : {stable_candidates}",
    f"GO-supported                     : {go_supported}",
    f"InterPro-supported               : {interpro_supported}",
    f"Pathway-supported                : {pathway_supported}",
    f"Any external functional evidence : {external_supported}",
    f"Network nodes                    : {len(nodes_df)}",
    f"Network edges                    : {len(edges_df)}",
    "",
    "=" * 80,
    "",
    "IMPORTANT INTERPRETATION",
    "",
    "The network represents available evidence relationships.",
    "",
    "A predictive BUSCO is not automatically a functionally",
    "validated candidate.",
    "",
    "Pathway support is counted only when the explicit",
    "Pathway_Supported field is TRUE.",
    "",
    "Functional descriptions and functional categories are",
    "not treated as pathway evidence.",
    "",
    "GO and InterPro evidence are reported separately.",
    "",
    "The network does not establish causality between a BUSCO",
    "and Carbon_Breadth.",
    "",
    "The next biological step is gene/protein-level mapping",
    "followed by functional and pathway interpretation.",
    ""
]


with open(
    REPORT_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            report_lines
        )
    )


# ================================================================
# 26. FINAL OUTPUT
# ================================================================

banner(
    "NETWORK CONSTRUCTION COMPLETE"
)

print(
    f"\nTotal predictive candidates: "
    f"{total_candidates}"
)

print(
    f"Stable predictive candidates: "
    f"{stable_candidates}"
)

print(
    f"GO-supported: "
    f"{go_supported}"
)

print(
    f"InterPro-supported: "
    f"{interpro_supported}"
)

print(
    f"Pathway-supported: "
    f"{pathway_supported}"
)

print(
    f"Any external functional evidence: "
    f"{external_supported}"
)

print("\nOutputs:")

print(
    "\nNodes:"
)

print(
    NODES_OUT
)

print(
    "\nEdges:"
)

print(
    EDGES_OUT
)

print(
    "\nCandidate evidence:"
)

print(
    EVIDENCE_OUT
)

print(
    "\nFunctional groups:"
)

print(
    FUNCTIONAL_GROUPS_OUT
)

print(
    "\nCandidate chassis interpretation:"
)

print(
    INTERPRETATION_OUT
)

print(
    "\nSummary:"
)

print(
    SUMMARY_OUT
)

print(
    "\nReport:"
)

print(
    REPORT_OUT
)

if figure_saved:

    print(
        "\nFigure:"
    )

    if FIGURE_OUT.exists():

        print(
            FIGURE_OUT
        )

    else:

        print(
            SHORT_FIGURE_OUT
        )


print()
print("=" * 80)
print("NEXT SCIENTIFIC STAGE")
print("=" * 80)

print(
    """
BUSCO → GENE/PROTEIN MAPPING
→ FUNCTIONAL ANNOTATION
→ GO / INTERPRO / EC / KEGG
→ PATHWAY-LEVEL INTERPRETATION
→ GENE–PATHWAY NETWORK
→ CANDIDATE CHASSIS INTERPRETATION
→ INDEPENDENT VALIDATION
"""
)

print(
    "IMPORTANT:"
)

print(
    "The network represents available evidence relationships."
)

print(
    "It does not establish that a BUSCO causes Carbon_Breadth."
)

print("=" * 80)