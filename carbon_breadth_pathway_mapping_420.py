# =============================================================================
# CARBON BREADTH — BUSCO FUNCTIONAL / PATHWAY MAPPING
# Project: Y1000+ Yeast Chassis
# Stage: Functional interpretation → pathway mapping
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import re
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")

BASE_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
)

ANNOTATION_DIR = (
    BASE_DIR
    / "feature_interpretation_420"
    / "functional_annotation_420"
)

TABLE_DIR = ANNOTATION_DIR / "tables"

ANNOTATION_FILE = (
    TABLE_DIR
    / "Carbon_Breadth_BUSCO_functional_annotation_420.csv"
)

CANDIDATE_FILE = (
    TABLE_DIR
    / "Carbon_Breadth_BUSCO_candidate_summary_420.csv"
)

OUTPUT_DIR = (
    ANNOTATION_DIR
    / "pathway_mapping_420"
)

OUTPUT_TABLE_DIR = OUTPUT_DIR / "tables"
OUTPUT_FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def find_column(df, candidates):
    """
    Find a column using case-insensitive matching.
    """
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    # partial matching
    for c in df.columns:
        lc = str(c).lower()
        for candidate in candidates:
            if candidate.lower() in lc:
                return c

    return None


# =============================================================================
# LOAD FILES
# =============================================================================

print_header("CARBON BREADTH — FUNCTIONAL / PATHWAY MAPPING")

print(f"Annotation file:")
print(ANNOTATION_FILE)

print("\nCandidate summary:")
print(CANDIDATE_FILE)

if not ANNOTATION_FILE.exists():
    raise FileNotFoundError(
        f"Functional annotation file not found:\n{ANNOTATION_FILE}"
    )

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(
        f"Candidate summary file not found:\n{CANDIDATE_FILE}"
    )

# =============================================================================
# LOAD ANNOTATION TABLE
# =============================================================================

print_header("LOADING FUNCTIONAL ANNOTATION TABLE")

annotation = pd.read_csv(ANNOTATION_FILE)

print(f"Shape: {annotation.shape}")

print("\nColumns:")
print(list(annotation.columns))

# =============================================================================
# LOAD CANDIDATE TABLE
# =============================================================================

print_header("LOADING CANDIDATE SUMMARY")

candidates = pd.read_csv(CANDIDATE_FILE)

print(f"Shape: {candidates.shape}")

print("\nColumns:")
print(list(candidates.columns))

# =============================================================================
# IDENTIFY IMPORTANT COLUMNS
# =============================================================================

busco_annotation_col = find_column(
    annotation,
    [
        "BUSCO_ID",
        "Busco_ID",
        "Feature",
        "busco",
        "busco_id"
    ]
)

busco_candidate_col = find_column(
    candidates,
    [
        "BUSCO_ID",
        "Busco_ID",
        "Feature",
        "busco",
        "busco_id"
    ]
)

description_col = find_column(
    annotation,
    [
        "Description",
        "Functional_Description",
        "Function",
        "Product",
        "Annotation"
    ]
)

status_col = find_column(
    annotation,
    [
        "Status",
        "BUSCO_Status",
        "Busco_Status"
    ]
)

importance_col = find_column(
    annotation,
    [
        "Mean_Importance",
        "Importance"
    ]
)

stability_col = find_column(
    annotation,
    [
        "Fold_Stability",
        "Stability"
    ]
)

print_header("IDENTIFIED COLUMNS")

print(f"BUSCO column       : {busco_annotation_col}")
print(f"Candidate BUSCO    : {busco_candidate_col}")
print(f"Description column : {description_col}")
print(f"Status column      : {status_col}")
print(f"Importance column  : {importance_col}")
print(f"Stability column   : {stability_col}")

if busco_annotation_col is None:
    raise ValueError(
        "Could not identify BUSCO ID column in functional annotation table."
    )

if busco_candidate_col is None:
    raise ValueError(
        "Could not identify BUSCO ID column in candidate summary."
    )

if description_col is None:
    raise ValueError(
        "Could not identify functional description column."
    )

# =============================================================================
# NORMALIZE BUSCO IDs
# =============================================================================

annotation["_BUSCO_NORMALIZED"] = (
    annotation[busco_annotation_col]
    .astype(str)
    .str.strip()
)

candidates["_BUSCO_NORMALIZED"] = (
    candidates[busco_candidate_col]
    .astype(str)
    .str.strip()
)

candidate_buscos = sorted(
    candidates["_BUSCO_NORMALIZED"]
    .dropna()
    .unique()
)

print_header("SELECTED BUSCO CANDIDATES")

print(f"Number of candidates: {len(candidate_buscos)}")

for x in candidate_buscos:
    print(x)

# =============================================================================
# FILTER TO SELECTED BUSCOs
# =============================================================================

selected = annotation[
    annotation["_BUSCO_NORMALIZED"].isin(candidate_buscos)
].copy()

print_header("FILTERED FUNCTIONAL ANNOTATIONS")

print(f"Rows retained: {len(selected)}")
print(f"Unique BUSCOs: {selected['_BUSCO_NORMALIZED'].nunique()}")

# =============================================================================
# FUNCTIONAL DESCRIPTION CONSOLIDATION
# =============================================================================

print_header("CONSOLIDATING FUNCTIONAL DESCRIPTIONS")

summary_rows = []

for busco in candidate_buscos:

    subset = selected[
        selected["_BUSCO_NORMALIZED"] == busco
    ].copy()

    descriptions = (
        subset[description_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    descriptions = descriptions[
        descriptions != ""
    ]

    description_counts = descriptions.value_counts()

    if len(description_counts) > 0:
        primary_description = description_counts.index[0]
        description_frequency = int(description_counts.iloc[0])
    else:
        primary_description = "No functional description recovered"
        description_frequency = 0

    # Number of genomes where this BUSCO is Complete
    if status_col is not None:

        status_values = (
            subset[status_col]
            .astype(str)
            .str.lower()
        )

        complete_count = int(
            status_values.str.contains("complete", na=False).sum()
        )

    else:
        complete_count = np.nan

    row = {
        "BUSCO_ID": busco,
        "Primary_Functional_Description": primary_description,
        "Description_Supporting_Rows": description_frequency,
        "Annotation_Rows": len(subset),
        "Complete_Rows": complete_count,
        "Unique_Descriptions": descriptions.nunique()
    }

    # Add candidate ML statistics if available
    candidate_match = candidates[
        candidates["_BUSCO_NORMALIZED"] == busco
    ]

    if len(candidate_match) > 0:

        c = candidate_match.iloc[0]

        for col in [
            "Mean_Importance",
            "SD_Importance",
            "Median_Importance",
            "Min_Importance",
            "Max_Importance",
            "Mean_Baseline_R2",
            "Positive_Importance_Folds",
            "Fold_Stability",
            "Importance_Rank"
        ]:
            if col in candidate_match.columns:
                row[col] = c[col]

    summary_rows.append(row)

functional_summary = pd.DataFrame(summary_rows)

# =============================================================================
# CONSERVATIVE FUNCTION CLASSIFICATION
# =============================================================================

print_header("CLASSIFYING FUNCTIONAL EVIDENCE")

def classify_function(description):

    d = description.lower()

    categories = []

    keyword_groups = {
        "RNA_processing": [
            "rna helicase",
            "rna processing",
            "rna binding",
            "rna polymerase",
            "splice",
            "mrna",
            "mRNA".lower(),
            "ribosomal"
        ],

        "Protein_processing": [
            "protein",
            "protease",
            "peptidase",
            "chaperone",
            "folding"
        ],

        "Metabolism": [
            "metabolic",
            "metabolism",
            "enzyme",
            "dehydrogenase",
            "transferase",
            "synthase",
            "isomerase",
            "methyltransferase",
            "phosphatase",
            "atpase",
            "helicase"
        ],

        "Cellular_transport": [
            "transport",
            "transporter",
            "membrane",
            "vesicle",
            "endosome",
            "gpi biosynthesis"
        ],

        "Cell_structure": [
            "cytoskeleton",
            "kinesin",
            "actin",
            "tubulin",
            "pdz domain",
            "wd40",
            "armadillo"
        ],

        "Transcription_regulation": [
            "transcription",
            "transcription factor",
            "myb",
            "sant"
        ],

        "Translation": [
            "ribosomal",
            "translation",
            "ribosome"
        ],

        "DNA_RNA_modification": [
            "methyltransferase",
            "dna repair",
            "rna helicase",
            "cap guanine"
        ],

        "Protein_interaction": [
            "pdz domain",
            "snf7",
            "protein interaction",
            "protein yip",
            "ran-interacting"
        ]
    }

    for category, keywords in keyword_groups.items():

        for keyword in keywords:

            if keyword in d:
                categories.append(category)
                break

    if not categories:
        return "Unresolved_function"

    return "; ".join(sorted(set(categories)))


functional_summary[
    "Functional_Category"
] = functional_summary[
    "Primary_Functional_Description"
].apply(classify_function)

# =============================================================================
# PATHWAY EVIDENCE LEVEL
# =============================================================================

print_header("ASSIGNING PATHWAY EVIDENCE LEVEL")

def pathway_evidence(description):

    d = description.lower()

    # Direct pathway-like biochemical annotations
    direct_keywords = [
        "gpi biosynthesis",
        "aconitase",
        "isopropylmalate",
        "porphobilinogen",
        "methyltransferase"
    ]

    if any(k in d for k in direct_keywords):
        return "Potential_pathway_relevant"

    # Domain/family descriptions require additional mapping
    domain_keywords = [
        "domain",
        "family",
        "fold",
        "repeat"
    ]

    if any(k in d for k in domain_keywords):
        return "Requires_external_annotation"

    return "Functional_annotation_only"


functional_summary[
    "Pathway_Evidence"
] = functional_summary[
    "Primary_Functional_Description"
].apply(pathway_evidence)

# =============================================================================
# CANDIDATE PRIORITY
# =============================================================================

def candidate_priority(row):

    stability = row.get("Fold_Stability", np.nan)
    importance = row.get("Mean_Importance", np.nan)

    if pd.notna(stability) and pd.notna(importance):

        if stability >= 0.8 and importance >= 0.003:
            return "High_priority"

        if stability >= 0.6:
            return "Moderate_priority"

    return "Exploratory"

functional_summary[
    "Interpretation_Priority"
] = functional_summary.apply(
    candidate_priority,
    axis=1
)

# =============================================================================
# SAVE FUNCTIONAL SUMMARY
# =============================================================================

summary_file = (
    OUTPUT_TABLE_DIR
    / "Carbon_Breadth_BUSCO_functional_summary_420.csv"
)

functional_summary.to_csv(
    summary_file,
    index=False
)

print("\nSaved:")
print(summary_file)

# =============================================================================
# CREATE PATHWAY-READY TABLE
# =============================================================================

pathway_ready = functional_summary[
    [
        "BUSCO_ID",
        "Primary_Functional_Description",
        "Functional_Category",
        "Pathway_Evidence",
        "Interpretation_Priority"
    ]
].copy()

# Add ML fields if present
for col in [
    "Mean_Importance",
    "SD_Importance",
    "Fold_Stability",
    "Positive_Importance_Folds",
    "Importance_Rank"
]:

    if col in functional_summary.columns:
        pathway_ready[col] = functional_summary[col]

pathway_file = (
    OUTPUT_TABLE_DIR
    / "Carbon_Breadth_pathway_ready_candidates_420.csv"
)

pathway_ready.to_csv(
    pathway_file,
    index=False
)

print("\nPathway-ready table saved:")
print(pathway_file)

# =============================================================================
# FULL ANNOTATION SUBSET
# =============================================================================

full_subset_file = (
    OUTPUT_TABLE_DIR
    / "Carbon_Breadth_selected_BUSCO_full_annotations_420.csv"
)

selected.drop(
    columns=["_BUSCO_NORMALIZED"],
    errors="ignore"
).to_csv(
    full_subset_file,
    index=False
)

print("\nFull selected annotation table saved:")
print(full_subset_file)

# =============================================================================
# FUNCTIONAL CATEGORY SUMMARY
# =============================================================================

category_summary = (
    functional_summary
    .groupby("Functional_Category", dropna=False)
    .agg(
        Number_of_BUSCOs=("BUSCO_ID", "nunique")
    )
    .reset_index()
    .sort_values(
        "Number_of_BUSCOs",
        ascending=False
    )
)

category_file = (
    OUTPUT_TABLE_DIR
    / "Carbon_Breadth_functional_category_summary_420.csv"
)

category_summary.to_csv(
    category_file,
    index=False
)

print("\nFunctional category summary saved:")
print(category_file)

# =============================================================================
# HIGH PRIORITY CANDIDATES
# =============================================================================

high_priority = functional_summary[
    functional_summary["Interpretation_Priority"] == "High_priority"
].copy()

high_priority_file = (
    OUTPUT_TABLE_DIR
    / "Carbon_Breadth_high_priority_functional_candidates_420.csv"
)

high_priority.to_csv(
    high_priority_file,
    index=False
)

print("\nHigh-priority candidate table saved:")
print(high_priority_file)

# =============================================================================
# PRINT SUMMARY
# =============================================================================

print_header("FUNCTIONAL INTERPRETATION SUMMARY")

print(
    functional_summary[
        [
            "BUSCO_ID",
            "Primary_Functional_Description",
            "Functional_Category",
            "Pathway_Evidence",
            "Interpretation_Priority"
        ]
    ].to_string(index=False)
)

print_header("FUNCTIONAL CATEGORY COUNTS")

print(category_summary.to_string(index=False))

# =============================================================================
# JSON SUMMARY
# =============================================================================

json_summary = {
    "phenotype": "Carbon_Breadth",
    "n_candidate_BUSCOs": int(len(candidate_buscos)),
    "n_annotated_BUSCOs": int(
        (
            functional_summary[
                "Primary_Functional_Description"
            ]
            != "No functional description recovered"
        ).sum()
    ),
    "n_high_priority": int(len(high_priority)),
    "functional_categories": category_summary.to_dict(
        orient="records"
    ),
    "next_stage": [
        "External ortholog annotation",
        "GO mapping",
        "KEGG mapping",
        "InterPro/Pfam mapping",
        "Pathway interpretation",
        "Network analysis"
    ],
    "causality_warning": (
        "BUSCO feature importance represents predictive association "
        "and does not establish causal biological effects."
    )
}

json_file = (
    OUTPUT_TABLE_DIR
    / "Carbon_Breadth_functional_interpretation_summary_420.json"
)

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(
        json_summary,
        f,
        indent=2
    )

print("\nJSON summary saved:")
print(json_file)

# =============================================================================
# FINAL
# =============================================================================

print_header("FUNCTIONAL INTERPRETATION PREPARATION COMPLETE")

print(f"""
Phenotype:
Carbon_Breadth

Candidate BUSCOs:
{len(candidate_buscos)}

Functional summary:
{summary_file}

Pathway-ready candidates:
{pathway_file}

Full selected annotations:
{full_subset_file}

Functional category summary:
{category_file}

High-priority candidates:
{high_priority_file}

JSON summary:
{json_file}

NEXT STAGE:
External annotation → GO/KEGG/InterPro mapping
→ pathway interpretation → network analysis
→ candidate chassis interpretation

IMPORTANT:
This script deliberately does NOT assign unsupported KEGG/GO pathways.
Domain/family-level BUSCO descriptions are flagged for external annotation.
""")