# ================================================================
# CANDIDATE PRIORITIZATION — CARBON BREADTH — 420 TAXA
# Direct-input version
#
# Uses only previously generated project outputs.
# Does NOT recursively search BUSCO directories.
#
# Purpose:
#   Integrate predictive importance + stability + BUSCO annotation
#   + external functional evidence and classify candidates for
#   downstream biological validation.
#
# IMPORTANT:
#   "Candidate" does NOT mean experimentally validated.
# ================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")

BASE = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
)


# ================================================================
# KNOWN EXISTING INPUT FILES
# ================================================================

TOP20_FILE = (
    BASE
    / "tables"
    / "Carbon_Breadth_top20_BUSCO_features_420.csv"
)

FUNCTIONAL_ANNOTATION = (
    BASE
    / "functional_annotation_420"
    / "tables"
    / "Carbon_Breadth_BUSCO_functional_annotation_420.csv"
)

INTEGRATED_FILE = (
    BASE
    / "functional_annotation_420"
    / "pathway_mapping_420"
    / "pathway_level_integration_420"
    / "tables"
    / "Carbon_Breadth_pathway_level_integrated_candidates_420.csv"
)

EXTERNAL_FILE = (
    BASE
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
    BASE
    / "candidate_prioritization_420"
)

TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def load_required(path, name):

    print("\n" + "=" * 80)
    print(f"LOADING {name}")
    print("=" * 80)

    print(f"File:\n{path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n\n"
            "Check that the previous pipeline stage completed."
        )

    df = pd.read_csv(path)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    print(list(df.columns))

    return df


def first_existing_column(df, candidates):

    for col in candidates:

        if col in df.columns:
            return col

    return None


def normalize_busco(x):

    if pd.isna(x):
        return np.nan

    return str(x).strip()


def bool_from_value(x):

    if pd.isna(x):
        return False

    s = str(x).strip().lower()

    return s in {
        "1",
        "true",
        "yes",
        "y",
        "supported",
        "complete",
        "moderate",
        "strong"
    }


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("CARBON BREADTH — CANDIDATE PRIORITIZATION")
print("=" * 80)

print(
    "\nDirect-input mode."
)

print(
    "No recursive BUSCO-directory search will be performed."
)


# ================================================================
# CHECK KNOWN INPUTS
# ================================================================

print("\n" + "=" * 80)
print("CHECKING KNOWN INPUT FILES")
print("=" * 80)

input_files = [
    ("Top-20 feature table", TOP20_FILE),
    ("BUSCO functional annotation", FUNCTIONAL_ANNOTATION),
    ("Pathway-level integration", INTEGRATED_FILE),
    ("External functional annotation", EXTERNAL_FILE),
]

for label, path in input_files:

    if path.exists():

        print(f"\n[FOUND] {label}")
        print(path)

    else:

        print(f"\n[MISSING] {label}")
        print(path)


# ================================================================
# LOAD TOP-20 FEATURE IMPORTANCE
# ================================================================

top20 = load_required(
    TOP20_FILE,
    "TOP-20 BUSCO FEATURE IMPORTANCE"
)


# ================================================================
# IDENTIFY BUSCO COLUMN
# ================================================================

top20_busco_col = first_existing_column(
    top20,
    [
        "BUSCO_ID",
        "Feature",
        "busco_id",
        "feature"
    ]
)

if top20_busco_col is None:

    raise ValueError(
        "Could not identify BUSCO column in top-20 feature table."
    )


top20["BUSCO_ID"] = (
    top20[top20_busco_col]
    .map(normalize_busco)
)

top20 = top20.dropna(
    subset=["BUSCO_ID"]
).copy()

top20 = top20.drop_duplicates(
    subset=["BUSCO_ID"]
)


print("\n" + "=" * 80)
print("TOP-20 BUSCO CANDIDATES")
print("=" * 80)

print(
    top20["BUSCO_ID"].tolist()
)

print(
    f"\nUnique predictive candidates: "
    f"{len(top20)}"
)


# ================================================================
# LOAD BUSCO FUNCTIONAL ANNOTATION
# ================================================================

annotation = load_required(
    FUNCTIONAL_ANNOTATION,
    "BUSCO FUNCTIONAL ANNOTATION"
)

if "BUSCO_ID" not in annotation.columns:

    raise ValueError(
        "BUSCO_ID column missing from functional annotation table."
    )

annotation["BUSCO_ID"] = (
    annotation["BUSCO_ID"]
    .map(normalize_busco)
)


# ================================================================
# LOAD PATHWAY INTEGRATION
# ================================================================

integrated = load_required(
    INTEGRATED_FILE,
    "PATHWAY-LEVEL INTEGRATION"
)

if "BUSCO_ID" not in integrated.columns:

    raise ValueError(
        "BUSCO_ID column missing from pathway integration table."
    )

integrated["BUSCO_ID"] = (
    integrated["BUSCO_ID"]
    .map(normalize_busco)
)


# ================================================================
# LOAD EXTERNAL ANNOTATION
# ================================================================

external = load_required(
    EXTERNAL_FILE,
    "EXTERNAL FUNCTIONAL ANNOTATION"
)

if "BUSCO_ID" not in external.columns:

    raise ValueError(
        "BUSCO_ID column missing from external annotation table."
    )

external["BUSCO_ID"] = (
    external["BUSCO_ID"]
    .map(normalize_busco)
)


# ================================================================
# REDUCE BUSCO ANNOTATION
# ================================================================

print("\n" + "=" * 80)
print("REDUCING BUSCO ANNOTATION")
print("=" * 80)

annotation_columns = [
    "BUSCO_ID",
    "Functional_Description",
    "Status",
    "Mean_Importance",
    "Fold_Stability",
    "Importance_Rank"
]

annotation_columns = [
    c
    for c in annotation_columns
    if c in annotation.columns
]

annotation_reduced = (
    annotation[annotation_columns]
    .drop_duplicates(
        subset=["BUSCO_ID"]
    )
    .copy()
)

print(
    "Unique annotated BUSCOs:",
    annotation_reduced["BUSCO_ID"].nunique()
)


# ================================================================
# REDUCE EXTERNAL ANNOTATION
# ================================================================

print("\n" + "=" * 80)
print("REDUCING EXTERNAL ANNOTATION")
print("=" * 80)

external_columns = [
    "BUSCO_ID",
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
    "Publication_Interpretation_Level"
]

external_columns = [
    c
    for c in external_columns
    if c in external.columns
]

external_reduced = (
    external[external_columns]
    .drop_duplicates(
        subset=["BUSCO_ID"]
    )
    .copy()
)

print(
    "Unique externally annotated BUSCOs:",
    external_reduced["BUSCO_ID"].nunique()
)


# ================================================================
# REDUCE PATHWAY INTEGRATION
# ================================================================

print("\n" + "=" * 80)
print("REDUCING PATHWAY INTEGRATION")
print("=" * 80)

integration_columns = [
    "BUSCO_ID",
    "Functional_Category",
    "GO_Supported",
    "InterPro_Supported",
    "Pathway_Supported",
    "Evidence_Class"
]

integration_columns = [
    c
    for c in integration_columns
    if c in integrated.columns
]

integration_reduced = (
    integrated[integration_columns]
    .drop_duplicates(
        subset=["BUSCO_ID"]
    )
    .copy()
)

print(
    "Unique integrated BUSCOs:",
    integration_reduced["BUSCO_ID"].nunique()
)


# ================================================================
# MERGE ALL EVIDENCE
# ================================================================

print("\n" + "=" * 80)
print("MERGING PREDICTIVE AND FUNCTIONAL EVIDENCE")
print("=" * 80)

master = top20.copy()


master = master.merge(
    annotation_reduced,
    on="BUSCO_ID",
    how="left",
    suffixes=(
        "",
        "_annotation"
    )
)


master = master.merge(
    external_reduced,
    on="BUSCO_ID",
    how="left",
    suffixes=(
        "",
        "_external"
    )
)


master = master.merge(
    integration_reduced,
    on="BUSCO_ID",
    how="left",
    suffixes=(
        "",
        "_integration"
    )
)


# ================================================================
# RESOLVE FUNCTIONAL CATEGORY
# ================================================================

if "Functional_Category" not in master.columns:

    if "Functional_Category_external" in master.columns:

        master["Functional_Category"] = (
            master["Functional_Category_external"]
        )

    elif "Functional_Category_integration" in master.columns:

        master["Functional_Category"] = (
            master["Functional_Category_integration"]
        )

    else:

        master["Functional_Category"] = (
            "Other / unclear functional assignment"
        )


# ================================================================
# EVIDENCE FLAGS
# ================================================================

print("\n" + "=" * 80)
print("CALCULATING FUNCTIONAL EVIDENCE FLAGS")
print("=" * 80)


# ------------------------------------------------
# GO evidence
# ------------------------------------------------

if "GO_Supported" in master.columns:

    go_flag = master["GO_Supported"].map(
        bool_from_value
    )

else:

    go_flag = pd.Series(
        False,
        index=master.index
    )


for col in [
    "GO_Molecular_Function",
    "GO_Biological_Process",
    "GO_Cellular_Component"
]:

    if col in master.columns:

        go_flag = (
            go_flag
            |
            master[col].fillna("").astype(str).str.strip().ne("")
            &
            master[col].fillna("").astype(str).str.lower().ne("nan")
        )


master["GO_Supported_Flag"] = go_flag


# ------------------------------------------------
# InterPro evidence
# ------------------------------------------------

if "InterPro_Supported" in master.columns:

    interpro_flag = master[
        "InterPro_Supported"
    ].map(
        bool_from_value
    )

else:

    interpro_flag = pd.Series(
        False,
        index=master.index
    )


if "InterPro_Domains" in master.columns:

    interpro_flag = (
        interpro_flag
        |
        (
            master["InterPro_Domains"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )
        &
        (
            master["InterPro_Domains"]
            .fillna("")
            .astype(str)
            .str.lower()
            .ne("nan")
        )
    )


master["InterPro_Supported_Flag"] = (
    interpro_flag
)


# ------------------------------------------------
# Pathway evidence
# ------------------------------------------------

if "Pathway_Supported" in master.columns:

    pathway_flag = master[
        "Pathway_Supported"
    ].map(
        bool_from_value
    )

else:

    pathway_flag = pd.Series(
        False,
        index=master.index
    )


if "KEGG_Pathway" in master.columns:

    pathway_flag = (
        pathway_flag
        |
        (
            master["KEGG_Pathway"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )
        &
        (
            master["KEGG_Pathway"]
            .fillna("")
            .astype(str)
            .str.lower()
            .ne("nan")
        )
    )


master["Pathway_Supported_Flag"] = (
    pathway_flag
)


# ================================================================
# ANY EXTERNAL EVIDENCE
# ================================================================

master["External_Evidence_Count"] = (
    master[
        [
            "GO_Supported_Flag",
            "InterPro_Supported_Flag",
            "Pathway_Supported_Flag"
        ]
    ]
    .sum(axis=1)
)


master["Any_External_Functional_Evidence"] = (
    master["External_Evidence_Count"] > 0
)


# ================================================================
# PREDICTIVE STABILITY
# ================================================================

if "Fold_Stability" in master.columns:

    master["Fold_Stability"] = pd.to_numeric(
        master["Fold_Stability"],
        errors="coerce"
    )

    master["Stable_Predictive_Signal"] = (
        master["Fold_Stability"] >= 0.8
    )

else:

    master["Stable_Predictive_Signal"] = False


if "Mean_Importance" in master.columns:

    master["Mean_Importance"] = pd.to_numeric(
        master["Mean_Importance"],
        errors="coerce"
    )

else:

    master["Mean_Importance"] = np.nan


# ================================================================
# CANDIDATE CLASSIFICATION
# ================================================================

print("\n" + "=" * 80)
print("CLASSIFYING CANDIDATES")
print("=" * 80)


def classify_candidate(row):

    stable = bool(
        row["Stable_Predictive_Signal"]
    )

    go = bool(
        row["GO_Supported_Flag"]
    )

    interpro = bool(
        row["InterPro_Supported_Flag"]
    )

    pathway = bool(
        row["Pathway_Supported_Flag"]
    )

    external_count = sum(
        [
            go,
            interpro,
            pathway
        ]
    )

    if stable and pathway:

        return (
            "High-priority functional candidate"
        )

    elif stable and external_count >= 1:

        return (
            "Strong predictive + functional candidate"
        )

    elif stable and external_count == 0:

        return (
            "Stable predictive candidate - "
            "functional validation required"
        )

    elif (not stable) and external_count >= 1:

        return (
            "Functional candidate - "
            "predictive stability requires validation"
        )

    else:

        return (
            "Exploratory candidate - "
            "requires functional validation"
        )


master["Candidate_Classification"] = (
    master.apply(
        classify_candidate,
        axis=1
    )
)


# ================================================================
# PRIORITIZATION GROUP
# ================================================================

classification_order = {

    "High-priority functional candidate": 1,

    "Strong predictive + functional candidate": 2,

    "Stable predictive candidate - "
    "functional validation required": 3,

    "Functional candidate - "
    "predictive stability requires validation": 4,

    "Exploratory candidate - "
    "requires functional validation": 5
}


master["Prioritization_Group"] = (
    master[
        "Candidate_Classification"
    ]
    .map(
        classification_order
    )
    .fillna(99)
    .astype(int)
)


# ================================================================
# SORT CANDIDATES
# ================================================================

master = master.sort_values(
    [
        "Prioritization_Group",
        "External_Evidence_Count",
        "Stable_Predictive_Signal",
        "Mean_Importance"
    ],
    ascending=[
        True,
        False,
        False,
        False
    ]
).reset_index(
    drop=True
)


master["Prioritization_Rank"] = (
    np.arange(
        1,
        len(master) + 1
    )
)


# ================================================================
# FINAL TABLE COLUMNS
# ================================================================

preferred_columns = [

    "Prioritization_Rank",

    "BUSCO_ID",

    "Functional_Description",

    "Functional_Category",

    "Mean_Importance",

    "Fold_Stability",

    "Importance_Rank",

    "Stable_Predictive_Signal",

    "GO_Supported_Flag",

    "InterPro_Supported_Flag",

    "Pathway_Supported_Flag",

    "External_Evidence_Count",

    "Any_External_Functional_Evidence",

    "Candidate_Classification",

    "Prioritization_Group",

    "GO_Molecular_Function",

    "GO_Biological_Process",

    "GO_Cellular_Component",

    "InterPro_Domains",

    "KEGG_Pathway",

    "Phyletic_Profile",

    "Evolutionary_Rate",

    "Publication_Interpretation_Level"
]


final_columns = [
    c
    for c in preferred_columns
    if c in master.columns
]


final_table = master[
    final_columns
].copy()


# ================================================================
# SAVE MASTER PRIORITIZATION TABLE
# ================================================================

MASTER_OUTPUT = (
    TABLE_DIR
    / "Carbon_Breadth_candidate_prioritization_420.csv"
)

final_table.to_csv(
    MASTER_OUTPUT,
    index=False
)


# ================================================================
# SAVE FUNCTIONALLY SUPPORTED CANDIDATES
# ================================================================

# IMPORTANT:
# This selects candidates with at least one type of
# external functional evidence.
#
# It does NOT mean these candidates are experimentally validated.

functional_candidates = final_table[
    final_table["External_Evidence_Count"] > 0
].copy()


FUNCTIONAL_OUTPUT = (
    TABLE_DIR
    / "Carbon_Breadth_functionally_supported_candidates_420.csv"
)

functional_candidates.to_csv(
    FUNCTIONAL_OUTPUT,
    index=False
)


# ================================================================
# SAVE STABLE PREDICTIVE CANDIDATES
# ================================================================

stable_candidates = final_table[
    final_table["Stable_Predictive_Signal"] == True
].copy()


STABLE_OUTPUT = (
    TABLE_DIR
    / "Carbon_Breadth_stable_predictive_candidates_420.csv"
)

stable_candidates.to_csv(
    STABLE_OUTPUT,
    index=False
)


# ================================================================
# SUMMARY COUNTS
# ================================================================

total = len(
    final_table
)

stable_n = int(
    final_table[
        "Stable_Predictive_Signal"
    ].sum()
)

go_n = int(
    final_table[
        "GO_Supported_Flag"
    ].sum()
)

interpro_n = int(
    final_table[
        "InterPro_Supported_Flag"
    ].sum()
)

pathway_n = int(
    final_table[
        "Pathway_Supported_Flag"
    ].sum()
)

external_n = int(
    (
        final_table[
            "External_Evidence_Count"
        ] > 0
    ).sum()
)


# ================================================================
# JSON SUMMARY
# ================================================================

summary = {

    "Phenotype":
        "Carbon_Breadth",

    "Total_predictive_candidates":
        total,

    "Stable_predictive_candidates":
        stable_n,

    "GO_supported_candidates":
        go_n,

    "InterPro_supported_candidates":
        interpro_n,

    "Pathway_supported_candidates":
        pathway_n,

    "Candidates_with_any_external_functional_evidence":
        external_n,

    "Interpretation":
        (
            "The 20 candidates represent predictive BUSCO "
            "features identified from the phylogeny-aware "
            "machine-learning analysis. External functional "
            "evidence is used for prioritization only. "
            "Functional annotation does not establish "
            "causality or experimental validation."
        )
}


SUMMARY_JSON = (
    TABLE_DIR
    / "Carbon_Breadth_candidate_prioritization_summary_420.json"
)


with open(
    SUMMARY_JSON,
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

REPORT = (
    REPORT_DIR
    / "Carbon_Breadth_candidate_prioritization_report_420.txt"
)


with open(
    REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CARBON BREADTH — CANDIDATE PRIORITIZATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Total predictive candidates: {total}\n"
    )

    f.write(
        f"Stable predictive candidates: {stable_n}\n"
    )

    f.write(
        f"GO-supported candidates: {go_n}\n"
    )

    f.write(
        f"InterPro-supported candidates: {interpro_n}\n"
    )

    f.write(
        f"Pathway-supported candidates: {pathway_n}\n"
    )

    f.write(
        "Candidates with any external functional evidence: "
        f"{external_n}\n\n"
    )

    f.write(
        "INTERPRETATION\n"
    )

    f.write(
        "The prioritization integrates predictive importance, "
        "cross-fold stability, BUSCO annotation and available "
        "external functional evidence. A candidate without "
        "external annotation is not considered biologically "
        "irrelevant; it is classified as requiring additional "
        "functional validation.\n\n"
    )

    f.write(
        "CANDIDATE CLASSIFICATIONS\n"
    )

    counts = (
        final_table[
            "Candidate_Classification"
        ]
        .value_counts()
    )

    for category, count in counts.items():

        f.write(
            f"{category}: {count}\n"
        )


# ================================================================
# CONSOLE SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("CANDIDATE PRIORITIZATION SUMMARY")
print("=" * 80)

print(
    f"Total predictive candidates       : {total}"
)

print(
    f"Stable predictive candidates     : {stable_n}"
)

print(
    f"GO-supported                     : {go_n}"
)

print(
    f"InterPro-supported               : {interpro_n}"
)

print(
    f"Pathway-supported                : {pathway_n}"
)

print(
    f"Any external functional evidence : {external_n}"
)


# ================================================================
# CLASSIFICATION TABLE
# ================================================================

print("\n" + "=" * 80)
print("CANDIDATE CLASSIFICATION")
print("=" * 80)

classification_columns = [
    "Prioritization_Rank",
    "BUSCO_ID",
    "Mean_Importance",
    "Fold_Stability",
    "External_Evidence_Count",
    "Candidate_Classification"
]

classification_columns = [
    c
    for c in classification_columns
    if c in final_table.columns
]


print(
    final_table[
        classification_columns
    ].to_string(
        index=False
    )
)


# ================================================================
# OUTPUT LOCATIONS
# ================================================================

print("\n" + "=" * 80)
print("OUTPUTS")
print("=" * 80)

print("\nMaster prioritization:")
print(MASTER_OUTPUT)

print("\nFunctionally supported candidates:")
print(FUNCTIONAL_OUTPUT)

print("\nStable predictive candidates:")
print(STABLE_OUTPUT)

print("\nJSON summary:")
print(SUMMARY_JSON)

print("\nReport:")
print(REPORT)


# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 80)
print("CANDIDATE PRIORITIZATION COMPLETE")
print("=" * 80)

print(
    "\nNo BUSCO directories were recursively searched."
)

print(
    "Existing models, OOF predictions, annotations, "
    "and pathway results were NOT overwritten."
)

print(
    "\nThe 20 predictive candidates have been retained."
)

print(
    "External functional evidence is used for prioritization, "
    "not as proof of causality."
)