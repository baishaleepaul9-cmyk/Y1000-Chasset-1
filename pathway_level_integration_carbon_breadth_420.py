# ================================================================
# PATHWAY-LEVEL INTEGRATION — CARBON BREADTH
# Y1000+ Yeast Chassis Project
#
# Purpose:
#   Integrate:
#       1. Top-20 BUSCO feature importance
#       2. Stable BUSCO candidates
#       3. BUSCO functional annotations
#       4. External GO / InterPro / pathway evidence
#
# Outputs:
#   - Integrated candidate table
#   - Evidence classification
#   - Functional grouping
#   - Pathway-level summary
#   - Network-ready edge table
#   - Candidate-chassis interpretation
#
# IMPORTANT:
#   Existing models, OOF predictions and previous annotations
#   are NOT overwritten.
# ================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
import warnings

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")

MODEL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
)

FEATURE_INTERPRETATION_ROOT = (
    MODEL_ROOT
    / "feature_interpretation_420"
)

FUNCTIONAL_ANNOTATION_ROOT = (
    FEATURE_INTERPRETATION_ROOT
    / "functional_annotation_420"
)

PATHWAY_ROOT = (
    FUNCTIONAL_ANNOTATION_ROOT
    / "pathway_mapping_420"
)

EXTERNAL_ROOT = (
    PATHWAY_ROOT
    / "external_annotation_420"
)

# ================================================================
# CORRECT INPUT PATHS
# ================================================================

# ------------------------------------------------
# IMPORTANT FIX:
# Top-20 BUSCO feature table is NOT inside
# functional_annotation_420.
#
# It was generated earlier under:
#
# feature_interpretation_420\tables\
# ------------------------------------------------

TOP20_FEATURES = (
    FEATURE_INTERPRETATION_ROOT
    / "tables"
    / "Carbon_Breadth_top20_BUSCO_features_420.csv"
)

STABLE_FEATURES = (
    FEATURE_INTERPRETATION_ROOT
    / "tables"
    / "Carbon_Breadth_stable_BUSCO_candidates_420.csv"
)

INTERPRETATION_TABLE = (
    FEATURE_INTERPRETATION_ROOT
    / "tables"
    / "Carbon_Breadth_BUSCO_interpretation_420.csv"
)

# ------------------------------------------------
# BUSCO functional annotation
# ------------------------------------------------

BUSCO_ANNOTATION = (
    FUNCTIONAL_ANNOTATION_ROOT
    / "tables"
    / "Carbon_Breadth_BUSCO_functional_annotation_420.csv"
)

BUSCO_CANDIDATE_SUMMARY = (
    FUNCTIONAL_ANNOTATION_ROOT
    / "tables"
    / "Carbon_Breadth_BUSCO_candidate_summary_420.csv"
)

# ------------------------------------------------
# External annotation
# ------------------------------------------------

EXTERNAL_MASTER = (
    EXTERNAL_ROOT
    / "tables"
    / "Carbon_Breadth_external_functional_annotation_420.csv"
)

EXTERNAL_PATHWAY = (
    EXTERNAL_ROOT
    / "tables"
    / "Carbon_Breadth_pathway_supported_candidates_420.csv"
)

EXTERNAL_GO = (
    EXTERNAL_ROOT
    / "tables"
    / "Carbon_Breadth_GO_annotated_candidates_420.csv"
)

EXTERNAL_INTERPRO = (
    EXTERNAL_ROOT
    / "tables"
    / "Carbon_Breadth_InterPro_annotated_candidates_420.csv"
)

# ================================================================
# OUTPUT DIRECTORIES
# ================================================================

OUTPUT_ROOT = PATHWAY_ROOT / "pathway_level_integration_420"

TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TEXT_DIR = OUTPUT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def safe_read_csv(path, required=True):
    """
    Read CSV safely and report useful diagnostics.
    """

    path = Path(path)

    if not path.exists():

        if required:
            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}"
            )

        print(
            f"\nOptional file not found:\n{path}"
        )

        return pd.DataFrame()

    try:

        df = pd.read_csv(path)

    except Exception as e:

        if required:
            raise RuntimeError(
                f"\nCould not read CSV:\n{path}\n\n"
                f"Error: {e}"
            )

        print(
            f"\nCould not read optional file:\n{path}\n"
            f"Error: {e}"
        )

        return pd.DataFrame()

    return df


def find_column(df, candidates, required=False):

    if df.empty:
        if required:
            raise ValueError(
                "Cannot identify column from empty dataframe."
            )
        return None

    # Exact matching first
    for col in candidates:

        if col in df.columns:
            return col

    # Case-insensitive matching
    normalized = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = str(candidate).strip().lower()

        if key in normalized:
            return normalized[key]

    # Partial matching
    for col in df.columns:

        col_lower = str(col).lower()

        for candidate in candidates:

            if str(candidate).lower() in col_lower:
                return col

    if required:

        raise ValueError(
            "\nRequired column not found.\n"
            f"Expected one of: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return None


def clean_busco_id(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Remove accidental whitespace
    value = re.sub(r"\s+", "", value)

    return value


def flatten_value(value):

    if pd.isna(value):
        return ""

    if isinstance(value, (list, tuple, set)):

        return "; ".join(
            str(x) for x in value
        )

    return str(value)


def text_present(value):

    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    if value in {
        "",
        "nan",
        "none",
        "na",
        "n/a",
        "not found",
        "not_found",
        "no",
        "false",
        "0",
    }:
        return False

    return True


def contains_any(text, keywords):

    if not text_present(text):
        return False

    text = str(text).lower()

    return any(
        keyword.lower() in text
        for keyword in keywords
    )


# ================================================================
# START
# ================================================================

print("=" * 80)
print("PATHWAY-LEVEL INTEGRATION — CARBON BREADTH")
print("=" * 80)

print("\nProject:")
print(PROJECT_ROOT)

print("\nOutput:")
print(OUTPUT_ROOT)


# ================================================================
# CHECK INPUT FILES
# ================================================================

print("\n" + "=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)

input_files = {
    "Top-20 feature importance": TOP20_FEATURES,
    "Stable candidates": STABLE_FEATURES,
    "BUSCO annotation": BUSCO_ANNOTATION,
    "BUSCO candidate summary": BUSCO_CANDIDATE_SUMMARY,
    "External annotation": EXTERNAL_MASTER,
    "External pathway": EXTERNAL_PATHWAY,
    "External GO": EXTERNAL_GO,
    "External InterPro": EXTERNAL_INTERPRO,
}

for name, path in input_files.items():

    status = "FOUND" if path.exists() else "NOT FOUND"

    print(f"{name:<30} : {status}")
    print(f"  {path}")


# ================================================================
# LOAD TOP-20 FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 80)
print("LOADING TOP-20 FEATURE IMPORTANCE")
print("=" * 80)

top20 = safe_read_csv(
    TOP20_FEATURES,
    required=True
)

print("Shape:", top20.shape)
print("Columns:")
print(list(top20.columns))


top20_busco_col = find_column(
    top20,
    [
        "Feature",
        "BUSCO_ID",
        "Busco_ID",
        "BUSCO",
    ],
    required=True
)

top20_busco = (
    top20[top20_busco_col]
    .map(clean_busco_id)
)

top20["BUSCO_ID"] = top20_busco

print("\nTop-20 BUSCO candidates:")
print(
    top20["BUSCO_ID"]
    .dropna()
    .unique()
)


# ================================================================
# LOAD STABLE CANDIDATES
# ================================================================

print("\n" + "=" * 80)
print("LOADING STABLE BUSCO CANDIDATES")
print("=" * 80)

stable = safe_read_csv(
    STABLE_FEATURES,
    required=False
)

if not stable.empty:

    print("Shape:", stable.shape)
    print("Columns:")
    print(list(stable.columns))

    stable_busco_col = find_column(
        stable,
        [
            "Feature",
            "BUSCO_ID",
            "Busco_ID",
            "BUSCO",
        ],
        required=False
    )

    if stable_busco_col:

        stable["BUSCO_ID"] = (
            stable[stable_busco_col]
            .map(clean_busco_id)
        )

else:

    stable = pd.DataFrame(
        columns=["BUSCO_ID"]
    )


# ================================================================
# LOAD BUSCO FUNCTIONAL ANNOTATION
# ================================================================

print("\n" + "=" * 80)
print("LOADING BUSCO FUNCTIONAL ANNOTATION")
print("=" * 80)

annotation = safe_read_csv(
    BUSCO_ANNOTATION,
    required=True
)

print("Shape:", annotation.shape)

print("\nPrimary annotation columns:")
print(list(annotation.columns))


annotation_busco_col = find_column(
    annotation,
    [
        "BUSCO_ID",
        "Feature",
        "Busco_ID",
        "BUSCO",
    ],
    required=True
)

annotation["BUSCO_ID"] = (
    annotation[annotation_busco_col]
    .map(clean_busco_id)
)


# ================================================================
# FILTER TO TOP-20
# ================================================================

selected_buscos = set(
    top20["BUSCO_ID"]
    .dropna()
    .unique()
)

print("\nSelected BUSCO count:")
print(len(selected_buscos))

annotation_selected = annotation[
    annotation["BUSCO_ID"].isin(selected_buscos)
].copy()

print(
    "\nAnnotation rows corresponding to "
    "Top-20 candidates:",
    len(annotation_selected)
)


# ================================================================
# COLLAPSE ANNOTATION TO BUSCO LEVEL
# ================================================================

print("\n" + "=" * 80)
print("COLLAPSING BUSCO ANNOTATIONS")
print("=" * 80)

description_col = find_column(
    annotation_selected,
    [
        "Functional_Description",
        "Description",
        "Function",
        "Product",
    ],
    required=False
)

status_col = find_column(
    annotation_selected,
    [
        "Status",
        "BUSCO_Status",
    ],
    required=False
)

species_col = find_column(
    annotation_selected,
    [
        "Species",
    ],
    required=False
)

assembly_col = find_column(
    annotation_selected,
    [
        "Assembly_Accession",
        "Assembly",
        "Accession",
    ],
    required=False
)


def unique_join(series):

    values = []

    for x in series:

        if not text_present(x):
            continue

        x = str(x).strip()

        if x not in values:
            values.append(x)

    return "; ".join(values)


annotation_records = []

for busco_id, group in annotation_selected.groupby(
    "BUSCO_ID",
    sort=False
):

    record = {
        "BUSCO_ID": busco_id,
        "Annotation_Genome_Count": (
            group[assembly_col].nunique()
            if assembly_col
            else np.nan
        ),
    }

    if description_col:

        record["Functional_Description"] = unique_join(
            group[description_col]
        )

    else:

        record["Functional_Description"] = ""

    if status_col:

        status_values = group[status_col].astype(str)

        record["Complete_Count"] = (
            status_values
            .str.lower()
            .eq("complete")
            .sum()
        )

        record["Missing_Count"] = (
            status_values
            .str.lower()
            .eq("missing")
            .sum()
        )

    else:

        record["Complete_Count"] = np.nan
        record["Missing_Count"] = np.nan

    if species_col:

        record["Example_Species"] = unique_join(
            group[species_col].head(10)
        )

    else:

        record["Example_Species"] = ""

    annotation_records.append(record)


annotation_summary = pd.DataFrame(
    annotation_records
)

print(
    "Unique annotated BUSCOs:",
    annotation_summary["BUSCO_ID"].nunique()
)


# ================================================================
# MERGE FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 80)
print("MERGING FEATURE IMPORTANCE")
print("=" * 80)

importance_cols = [
    "Mean_Importance",
    "SD_Importance",
    "Median_Importance",
    "Min_Importance",
    "Max_Importance",
    "Mean_Baseline_R2",
    "Positive_Importance_Folds",
    "Fold_Stability",
    "Importance_Rank",
]

available_importance_cols = [
    c for c in importance_cols
    if c in top20.columns
]

top20_importance = top20[
    ["BUSCO_ID"] + available_importance_cols
].copy()

# Remove duplicate BUSCO entries
top20_importance = (
    top20_importance
    .drop_duplicates(subset="BUSCO_ID")
)

master = top20_importance.merge(
    annotation_summary,
    on="BUSCO_ID",
    how="left"
)


# ================================================================
# MERGE STABLE STATUS
# ================================================================

if not stable.empty and "BUSCO_ID" in stable.columns:

    stable_info_cols = [
        c for c in [
            "Mean_Importance",
            "SD_Importance",
            "Fold_Stability",
            "Positive_Importance_Folds",
        ]
        if c in stable.columns
    ]

    stable_info = stable[
        ["BUSCO_ID"] + stable_info_cols
    ].copy()

    stable_info = (
        stable_info
        .drop_duplicates("BUSCO_ID")
    )

    stable_info = stable_info.rename(
        columns={
            c: f"Stable_{c}"
            for c in stable_info_cols
        }
    )

    master = master.merge(
        stable_info,
        on="BUSCO_ID",
        how="left"
    )

    master["Stable_Candidate"] = True

else:

    master["Stable_Candidate"] = False


master["Stable_Candidate"] = (
    master["Stable_Candidate"]
    .fillna(False)
)


# ================================================================
# LOAD EXTERNAL ANNOTATION
# ================================================================

print("\n" + "=" * 80)
print("LOADING EXTERNAL FUNCTIONAL ANNOTATION")
print("=" * 80)

external = safe_read_csv(
    EXTERNAL_MASTER,
    required=False
)

print("External annotation shape:")
print(external.shape)

if not external.empty:

    print("\nExternal annotation columns:")
    print(list(external.columns))


# ================================================================
# PREPARE EXTERNAL ANNOTATION
# ================================================================

if not external.empty:

    external_busco_col = find_column(
        external,
        [
            "BUSCO_ID",
            "Feature",
            "Busco_ID",
            "BUSCO",
        ],
        required=False
    )

    if external_busco_col:

        external["BUSCO_ID"] = (
            external[external_busco_col]
            .map(clean_busco_id)
        )

        # Keep one record per BUSCO
        external_records = []

        for busco_id, group in external.groupby(
            "BUSCO_ID",
            sort=False
        ):

            record = {
                "BUSCO_ID": busco_id
            }

            for col in group.columns:

                if col == "BUSCO_ID":
                    continue

                record[col] = unique_join(
                    group[col]
                )

            external_records.append(record)

        external_summary = pd.DataFrame(
            external_records
        )

        master = master.merge(
            external_summary,
            on="BUSCO_ID",
            how="left",
            suffixes=("", "_External")
        )

    else:

        external_summary = pd.DataFrame()

else:

    external_summary = pd.DataFrame()


# ================================================================
# LOAD PATHWAY / GO / INTERPRO SUPPORT
# ================================================================

print("\n" + "=" * 80)
print("LOADING EXTERNAL EVIDENCE TABLES")
print("=" * 80)


def load_evidence_table(path, label):

    df = safe_read_csv(
        path,
        required=False
    )

    print(
        f"{label:<15}: {len(df)} rows"
    )

    if df.empty:
        return df

    busco_col = find_column(
        df,
        [
            "BUSCO_ID",
            "Feature",
            "Busco_ID",
            "BUSCO",
        ],
        required=False
    )

    if busco_col:

        df["BUSCO_ID"] = (
            df[busco_col]
            .map(clean_busco_id)
        )

    return df


pathway_df = load_evidence_table(
    EXTERNAL_PATHWAY,
    "Pathway"
)

go_df = load_evidence_table(
    EXTERNAL_GO,
    "GO"
)

interpro_df = load_evidence_table(
    EXTERNAL_INTERPRO,
    "InterPro"
)


# ================================================================
# CREATE EVIDENCE FLAGS
# ================================================================

print("\n" + "=" * 80)
print("CREATING FUNCTIONAL EVIDENCE FLAGS")
print("=" * 80)


pathway_buscos = set()

if not pathway_df.empty and "BUSCO_ID" in pathway_df.columns:

    pathway_buscos = set(
        pathway_df["BUSCO_ID"]
        .dropna()
        .unique()
    )


go_buscos = set()

if not go_df.empty and "BUSCO_ID" in go_df.columns:

    go_buscos = set(
        go_df["BUSCO_ID"]
        .dropna()
        .unique()
    )


interpro_buscos = set()

if not interpro_df.empty and "BUSCO_ID" in interpro_df.columns:

    interpro_buscos = set(
        interpro_df["BUSCO_ID"]
        .dropna()
        .unique()
    )


master["Pathway_Supported"] = (
    master["BUSCO_ID"]
    .isin(pathway_buscos)
)

master["GO_Supported"] = (
    master["BUSCO_ID"]
    .isin(go_buscos)
)

master["InterPro_Supported"] = (
    master["BUSCO_ID"]
    .isin(interpro_buscos)
)


# ================================================================
# EXTERNAL MASTER EVIDENCE DETECTION
# ================================================================

external_text_columns = []

if not external.empty:

    for col in external.columns:

        if col == "BUSCO_ID":
            continue

        if (
            external[col].dtype == object
            or str(external[col].dtype).startswith("string")
        ):

            external_text_columns.append(col)


# Build a lookup of all external text
external_text_lookup = {}

if not external.empty and "BUSCO_ID" in external.columns:

    for busco_id, group in external.groupby(
        "BUSCO_ID"
    ):

        pieces = []

        for col in external_text_columns:

            pieces.extend(
                group[col]
                .dropna()
                .astype(str)
                .tolist()
            )

        external_text_lookup[busco_id] = " ".join(
            pieces
        ).lower()


master["External_Evidence_Text"] = (
    master["BUSCO_ID"]
    .map(external_text_lookup)
    .fillna("")
)


# ================================================================
# FUNCTIONAL CATEGORY ASSIGNMENT
# ================================================================

print("\n" + "=" * 80)
print("ASSIGNING FUNCTIONAL CATEGORIES")
print("=" * 80)


def classify_function(description):

    text = str(description).lower()

    if not text_present(description):
        return "Uncharacterized / insufficient annotation"

    if any(
        x in text
        for x in [
            "transporter",
            "transport",
            "membrane",
            "abc ",
            "permease",
        ]
    ):
        return "Transport / membrane"

    if any(
        x in text
        for x in [
            "helicase",
            "rna helicase",
            "rna processing",
            "methyltransferase",
            "ribosomal",
            "translation",
            "rna",
        ]
    ):
        return "RNA processing / gene expression"

    if any(
        x in text
        for x in [
            "kinase",
            "phosphatase",
            "atpase",
            "atp-dependent",
            "enzyme",
        ]
    ):
        return "Enzymatic / metabolic activity"

    if any(
        x in text
        for x in [
            "glycos",
            "sugar",
            "carbohydrate",
            "hydrolase",
            "transferase",
            "metabolic",
        ]
    ):
        return "Metabolism"

    if any(
        x in text
        for x in [
            "kinesin",
            "cytoskeleton",
            "motor domain",
        ]
    ):
        return "Cytoskeleton / cellular organization"

    if any(
        x in text
        for x in [
            "gpi",
            "vesicle",
            "snf7",
            "endosome",
            "trafficking",
        ]
    ):
        return "Membrane trafficking / cellular organization"

    if any(
        x in text
        for x in [
            "domain",
            "repeat",
            "armadillo",
            "wd40",
            "pdz",
            "sant",
            "myb",
        ]
    ):
        return "Protein domain / regulatory function"

    return "Other / unclear functional assignment"


master["Functional_Category"] = (
    master["Functional_Description"]
    .fillna("")
    .map(classify_function)
)


# ================================================================
# EVIDENCE CLASSIFICATION
# ================================================================

print("\n" + "=" * 80)
print("EVIDENCE CLASSIFICATION")
print("=" * 80)


def classify_evidence(row):

    pathway = bool(row["Pathway_Supported"])
    go = bool(row["GO_Supported"])
    interpro = bool(row["InterPro_Supported"])

    external_text = str(
        row.get(
            "External_Evidence_Text",
            ""
        )
    )

    if pathway:

        return "Pathway-supported"

    if go and interpro:

        return "GO + InterPro supported"

    if go:

        return "GO-supported"

    if interpro:

        return "InterPro-supported"

    if text_present(external_text):

        return "External annotation — manual interpretation"

    return "No external pathway evidence"


master["Evidence_Class"] = (
    master.apply(
        classify_evidence,
        axis=1
    )
)


# ================================================================
# EVIDENCE STRENGTH
# ================================================================

def evidence_strength(row):

    evidence = row["Evidence_Class"]

    if evidence == "Pathway-supported":
        return "Strong"

    if evidence in {
        "GO + InterPro supported",
        "GO-supported",
        "InterPro-supported",
    }:
        return "Moderate"

    if evidence == "External annotation — manual interpretation":
        return "Limited"

    return "None"


master["Evidence_Strength"] = (
    master.apply(
        evidence_strength,
        axis=1
    )
)


# ================================================================
# BIOLOGICAL INTERPRETATION
# ================================================================

def create_interpretation(row):

    function = (
        row.get(
            "Functional_Description",
            ""
        )
    )

    category = (
        row.get(
            "Functional_Category",
            "Other / unclear functional assignment"
        )
    )

    evidence = (
        row.get(
            "Evidence_Class",
            "No external pathway evidence"
        )
    )

    importance = row.get(
        "Mean_Importance",
        np.nan
    )

    stability = row.get(
        "Fold_Stability",
        np.nan
    )

    return (
        f"BUSCO {row['BUSCO_ID']} is associated with "
        f"the functional category '{category}'. "
        f"Annotation: {function}. "
        f"Evidence class: {evidence}. "
        f"Mean predictive importance: {importance}. "
        f"Fold stability: {stability}. "
        f"This represents predictive/associational evidence "
        f"and should not be interpreted as proof of causality."
    )


master["Biological_Interpretation"] = (
    master.apply(
        create_interpretation,
        axis=1
    )
)


# ================================================================
# RANK FINAL CANDIDATES
# ================================================================

if "Importance_Rank" in master.columns:

    master = master.sort_values(
        "Importance_Rank",
        ascending=True
    )

else:

    master = master.sort_values(
        "Mean_Importance",
        ascending=False
    )


# ================================================================
# SAVE MASTER INTEGRATION TABLE
# ================================================================

MASTER_OUTPUT = (
    TABLE_DIR
    / "Carbon_Breadth_pathway_level_integrated_candidates_420.csv"
)

master.to_csv(
    MASTER_OUTPUT,
    index=False
)

print("\nMaster integration table saved:")
print(MASTER_OUTPUT)


# ================================================================
# PATHWAY-LEVEL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("PATHWAY / FUNCTIONAL GROUP SUMMARY")
print("=" * 80)


group_summary = (
    master
    .groupby(
        "Functional_Category",
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
            "Fold_Stability",
            "mean"
        ),
        GO_Supported=(
            "GO_Supported",
            "sum"
        ),
        InterPro_Supported=(
            "InterPro_Supported",
            "sum"
        ),
        Pathway_Supported=(
            "Pathway_Supported",
            "sum"
        ),
    )
    .reset_index()
    .sort_values(
        "Mean_Importance",
        ascending=False
    )
)

GROUP_OUTPUT = (
    TABLE_DIR
    / "Carbon_Breadth_functional_group_summary_420.csv"
)

group_summary.to_csv(
    GROUP_OUTPUT,
    index=False
)

print(group_summary.to_string(index=False))


# ================================================================
# EVIDENCE SUMMARY
# ================================================================

evidence_summary = (
    master
    .groupby(
        [
            "Evidence_Class",
            "Evidence_Strength"
        ],
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
            "Fold_Stability",
            "mean"
        ),
    )
    .reset_index()
)

EVIDENCE_OUTPUT = (
    TABLE_DIR
    / "Carbon_Breadth_evidence_summary_420.csv"
)

evidence_summary.to_csv(
    EVIDENCE_OUTPUT,
    index=False
)


# ================================================================
# NETWORK-READY NODE TABLE
# ================================================================

node_columns = [
    "BUSCO_ID",
    "Functional_Description",
    "Functional_Category",
    "Mean_Importance",
    "SD_Importance",
    "Fold_Stability",
    "Importance_Rank",
    "GO_Supported",
    "InterPro_Supported",
    "Pathway_Supported",
    "Evidence_Class",
    "Evidence_Strength",
]

node_columns = [
    c for c in node_columns
    if c in master.columns
]

network_nodes = master[
    node_columns
].copy()

NODE_OUTPUT = (
    TABLE_DIR
    / "Carbon_Breadth_network_nodes_420.csv"
)

network_nodes.to_csv(
    NODE_OUTPUT,
    index=False
)


# ================================================================
# NETWORK-READY EDGE TABLE
# ================================================================

edges = []

for _, row in master.iterrows():

    busco = row["BUSCO_ID"]
    category = row["Functional_Category"]

    if text_present(category):

        edges.append(
            {
                "Source": busco,
                "Target": category,
                "Relationship": "functional_category",
                "Evidence": row["Evidence_Class"],
                "Importance": row.get(
                    "Mean_Importance",
                    np.nan
                ),
            }
        )


edge_df = pd.DataFrame(edges)

EDGE_OUTPUT = (
    TABLE_DIR
    / "Carbon_Breadth_network_edges_420.csv"
)

edge_df.to_csv(
    EDGE_OUTPUT,
    index=False
)


# ================================================================
# CANDIDATE CHASSIS INTERPRETATION TABLE
# ================================================================

candidate_columns = [
    "BUSCO_ID",
    "Importance_Rank",
    "Mean_Importance",
    "SD_Importance",
    "Fold_Stability",
    "Functional_Description",
    "Functional_Category",
    "Evidence_Class",
    "Evidence_Strength",
    "Biological_Interpretation",
]

candidate_columns = [
    c for c in candidate_columns
    if c in master.columns
]

candidate_table = master[
    candidate_columns
].copy()

CANDIDATE_OUTPUT = (
    TABLE_DIR
    / "Carbon_Breadth_candidate_chassis_interpretation_420.csv"
)

candidate_table.to_csv(
    CANDIDATE_OUTPUT,
    index=False
)


# ================================================================
# SUMMARY COUNTS
# ================================================================

total_candidates = len(master)

go_supported = int(
    master["GO_Supported"].sum()
)

interpro_supported = int(
    master["InterPro_Supported"].sum()
)

pathway_supported = int(
    master["Pathway_Supported"].sum()
)

strong = int(
    (master["Evidence_Strength"] == "Strong").sum()
)

moderate = int(
    (master["Evidence_Strength"] == "Moderate").sum()
)

limited = int(
    (master["Evidence_Strength"] == "Limited").sum()
)

none = int(
    (master["Evidence_Strength"] == "None").sum()
)


# ================================================================
# JSON SUMMARY
# ================================================================

summary = {

    "Phenotype": "Carbon_Breadth",

    "Total_candidates": total_candidates,

    "GO_supported": go_supported,

    "InterPro_supported": interpro_supported,

    "Pathway_supported": pathway_supported,

    "Strong_external_support": strong,

    "Moderate_external_support": moderate,

    "Limited_external_support": limited,

    "No_external_evidence": none,

    "Top20_feature_file": str(
        TOP20_FEATURES
    ),

    "Stable_candidate_file": str(
        STABLE_FEATURES
    ),

    "BUSCO_annotation_file": str(
        BUSCO_ANNOTATION
    ),

    "External_annotation_file": str(
        EXTERNAL_MASTER
    ),

    "Interpretation": (
        "Feature importance provides predictive evidence. "
        "Functional annotations and external databases provide "
        "supporting biological context. These results do not "
        "establish causality or experimentally validate the "
        "candidate BUSCOs."
    ),
}


SUMMARY_OUTPUT = (
    TABLE_DIR
    / "Carbon_Breadth_pathway_integration_summary_420.json"
)

with open(
    SUMMARY_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ================================================================
# TEXT REPORT
# ================================================================

REPORT_OUTPUT = (
    TEXT_DIR
    / "Carbon_Breadth_pathway_integration_report_420.txt"
)

with open(
    REPORT_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CARBON BREADTH — PATHWAY LEVEL INTEGRATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total candidates: {total_candidates}\n"
    )

    f.write(
        f"GO-supported: {go_supported}\n"
    )

    f.write(
        f"InterPro-supported: {interpro_supported}\n"
    )

    f.write(
        f"Pathway-supported: {pathway_supported}\n"
    )

    f.write(
        f"Strong evidence: {strong}\n"
    )

    f.write(
        f"Moderate evidence: {moderate}\n"
    )

    f.write(
        f"Limited evidence: {limited}\n"
    )

    f.write(
        f"No external evidence: {none}\n\n"
    )

    f.write(
        "IMPORTANT INTERPRETATION NOTE\n"
    )

    f.write(
        "Feature importance is predictive evidence and "
        "not proof of causality. Functional annotations "
        "provide biological context and should be "
        "experimentally validated before being interpreted "
        "as causal determinants of carbon utilization.\n\n"
    )

    f.write(
        "FUNCTIONAL GROUPS\n"
    )

    f.write(
        group_summary.to_string(index=False)
    )


# ================================================================
# CONSOLE SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("FUNCTIONAL EVIDENCE SUMMARY")
print("=" * 80)

print(
    f"Total candidates              : {total_candidates}"
)

print(
    f"GO-supported                  : {go_supported}"
)

print(
    f"InterPro-supported            : {interpro_supported}"
)

print(
    f"Pathway-supported             : {pathway_supported}"
)

print(
    f"Strong evidence               : {strong}"
)

print(
    f"Moderate evidence             : {moderate}"
)

print(
    f"Limited evidence              : {limited}"
)

print(
    f"No external evidence         : {none}"
)


print("\n" + "=" * 80)
print("PATHWAY-LEVEL INTEGRATION COMPLETE")
print("=" * 80)

print("\nMaster integrated table:")
print(MASTER_OUTPUT)

print("\nFunctional group summary:")
print(GROUP_OUTPUT)

print("\nEvidence summary:")
print(EVIDENCE_OUTPUT)

print("\nNetwork nodes:")
print(NODE_OUTPUT)

print("\nNetwork edges:")
print(EDGE_OUTPUT)

print("\nCandidate chassis interpretation:")
print(CANDIDATE_OUTPUT)

print("\nJSON summary:")
print(SUMMARY_OUTPUT)

print("\nReport:")
print(REPORT_OUTPUT)

print("\nExisting models and OOF predictions were NOT overwritten.")

print("\nNext scientific stage:")
print(
    "Network construction → candidate chassis interpretation "
    "→ independent validation"
)