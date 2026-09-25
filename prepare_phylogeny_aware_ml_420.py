# -*- coding: utf-8 -*-

"""
Prepare the final 420-taxon dataset for phylogeny-aware ML.

INPUTS
------
ML matrix:
C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\feature_qc\y1000_420_variable_busco_ml_matrix.csv

ASTRAL tree:
C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_qc\ASTRAL_420_taxon_pruned.nwk

OUTPUT
------
C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml

The script:
1. Loads the final 420-taxon variable-BUSCO matrix.
2. Explicitly identifies BUSCO columns.
3. Validates exactly 420 taxa.
4. Validates exactly 1988 variable BUSCO features.
5. Validates binary BUSCO values.
6. Validates phenotype columns.
7. Loads the 420-taxon ASTRAL tree.
8. Checks exact accession-level correspondence.
9. Creates a taxon mapping between ML matrix and tree.
10. Creates a phylogeny-aware ML-ready matrix.
11. Does NOT modify any input file.
"""

import os
import re
import sys
import pandas as pd
import numpy as np


# =============================================================================
# PATHS
# =============================================================================

PROJECT = r"C:\Y1000_chassis_project"

ML_MATRIX = (
    PROJECT
    + r"\results\stage5_phylogeny_ml_dataset\feature_qc"
    + r"\y1000_420_variable_busco_ml_matrix.csv"
)

ASTRAL_TREE = (
    PROJECT
    + r"\results\stage5_phylogeny_ml_dataset\phylogeny_qc"
    + r"\ASTRAL_420_taxon_pruned.nwk"
)

OUTPUT_DIR = (
    PROJECT
    + r"\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# EXPECTED STRUCTURE
# =============================================================================

EXPECTED_TAXA = 420
EXPECTED_VARIABLE_BUSCOS = 1988

IDENTIFIER_COLUMNS = [
    "Species",
    "Assembly_Accession",
]

PHENOTYPE_COLUMNS = [
    "N_Strains",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Carbon_Breadth_SD",
    "Nitrogen_Breadth_SD",
]

OPTIONAL_METADATA_COLUMNS = [
    "Phenotype_Source_Species",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_species(x):
    """Normalize species names for comparison."""
    if pd.isna(x):
        return ""

    x = str(x).strip().lower()

    # Collapse repeated whitespace
    x = re.sub(r"\s+", " ", x)

    return x


def normalize_accession(x):
    """Normalize assembly accession."""
    if pd.isna(x):
        return ""

    return str(x).strip().upper()


def is_busco_column(column):
    """
    BUSCO IDs in this project have the form:

        1234at4891

    or similar numeric ID + 'at' + numeric lineage ID.

    This avoids accidentally counting:
        Species
        Assembly_Accession
        phenotype columns
        Phenotype_Source_Species
    """

    column = str(column).strip()

    return bool(
        re.fullmatch(r"\d+at\d+", column)
    )


def parse_newick_tips(newick_file):
    """
    Lightweight Newick tip parser.

    Avoids requiring ete3 merely to inspect tip labels.
    """

    with open(newick_file, "r", encoding="utf-8") as handle:
        text = handle.read().strip()

    if not text:
        raise RuntimeError("ASTRAL tree file is empty.")

    # Remove final semicolon
    text = text.rstrip(";")

    # Remove branch lengths.
    # Example:
    # Taxon__GCA_123.1:0.004
    text = re.sub(r":[-+0-9.eE]+", "", text)

    # Remove internal node labels where applicable
    # We mainly need leaf labels.
    tokens = re.split(r"[(),;]", text)

    tips = []

    for token in tokens:

        token = token.strip()

        if not token:
            continue

        # Remove support/internal labels if present.
        # A real taxon should contain an accession.
        if "__GCA_" in token or "__GCF_" in token:

            tips.append(token)

    return tips


def extract_accession_from_tip(tip):
    """
    Extract accession from labels such as:

    Species_name__GCA_030558845.1
    """

    match = re.search(
        r"((?:GCA|GCF)_\d+\.\d+)",
        str(tip)
    )

    if match:
        return match.group(1).upper()

    return ""


def extract_species_from_tip(tip):
    """
    Convert:

        Species_name__GCA_030558845.1

    to:

        Species name
    """

    accession = extract_accession_from_tip(tip)

    if accession:

        prefix = str(tip).split("__" + accession)[0]

        return prefix.replace("_", " ").strip()

    return str(tip).replace("_", " ").strip()


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("PHYLOGENY-AWARE ML PREPARATION — FINAL 420 TAXA")
print("=" * 80)

print()
print("Project:")
print(PROJECT)

print()
print("ML matrix:")
print(ML_MATRIX)

print()
print("ASTRAL tree:")
print(ASTRAL_TREE)

print()
print("Output:")
print(OUTPUT_DIR)


# =============================================================================
# CHECK INPUTS
# =============================================================================

print()
print("=" * 80)
print("CHECKING INPUTS")
print("=" * 80)

if not os.path.isfile(ML_MATRIX):
    raise FileNotFoundError(
        f"Final ML matrix not found:\n{ML_MATRIX}"
    )

print("✓ Final 420 ML matrix found")

if not os.path.isfile(ASTRAL_TREE):
    raise FileNotFoundError(
        f"420-taxon ASTRAL tree not found:\n{ASTRAL_TREE}"
    )

print("✓ 420-taxon ASTRAL tree found")


# =============================================================================
# LOAD ML MATRIX
# =============================================================================

print()
print("=" * 80)
print("LOADING FINAL ML MATRIX")
print("=" * 80)

df = pd.read_csv(ML_MATRIX)

print()
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Required columns
for col in IDENTIFIER_COLUMNS:
    if col not in df.columns:
        raise RuntimeError(
            f"Required identifier column missing: {col}"
        )

for col in PHENOTYPE_COLUMNS:
    if col not in df.columns:
        raise RuntimeError(
            f"Required phenotype column missing: {col}"
        )

print("✓ Species column present")
print("✓ Assembly_Accession column present")
print("✓ Required phenotype columns present")


# =============================================================================
# TAXON QC
# =============================================================================

print()
print("=" * 80)
print("TAXON QC")
print("=" * 80)

if len(df) != EXPECTED_TAXA:
    raise RuntimeError(
        f"Expected {EXPECTED_TAXA} taxa but found {len(df)}."
    )

species_keys = df["Species"].map(normalize_species)
accession_keys = df["Assembly_Accession"].map(normalize_accession)

duplicate_species = species_keys.duplicated().sum()
duplicate_accessions = accession_keys.duplicated().sum()

print()
print(f"Taxa:                  {len(df)}")
print(f"Expected taxa:         {EXPECTED_TAXA}")
print(f"Duplicate species:     {duplicate_species}")
print(f"Duplicate accessions:  {duplicate_accessions}")

if duplicate_species != 0:
    raise RuntimeError(
        "Duplicate species detected in final 420 ML matrix."
    )

if duplicate_accessions != 0:
    raise RuntimeError(
        "Duplicate assembly accessions detected."
    )

print()
print("✓ Exactly 420 unique taxa")
print("✓ Species are unique")
print("✓ Assembly accessions are unique")


# =============================================================================
# BUSCO FEATURE DETECTION
# =============================================================================

print()
print("=" * 80)
print("IDENTIFYING BUSCO FEATURES")
print("=" * 80)

# IMPORTANT:
# Do NOT infer BUSCO columns merely by excluding metadata.
# Explicitly detect BUSCO IDs using the project's BUSCO naming pattern.

busco_columns = [
    col
    for col in df.columns
    if is_busco_column(col)
]

print()
print(f"BUSCO features detected: {len(busco_columns)}")
print(f"Expected BUSCO features: {EXPECTED_VARIABLE_BUSCOS}")

# Print diagnostic information
print()
print("Last 20 detected BUSCOs:")

for col in busco_columns[-20:]:
    print(f"  {col}")


# =============================================================================
# BUSCO COUNT VALIDATION
# =============================================================================

if len(busco_columns) != EXPECTED_VARIABLE_BUSCOS:

    print()
    print("ERROR: BUSCO feature count does not match expected count.")

    # Diagnose columns that look like BUSCOs
    print()
    print("Detected BUSCO count:")
    print(len(busco_columns))

    print()
    print("Expected:")
    print(EXPECTED_VARIABLE_BUSCOS)

    # Check the known extra BUSCO
    if "9995at4891" in busco_columns:
        print()
        print("✓ 9995at4891 is present and correctly recognized as a BUSCO.")

    # Show non-BUSCO columns
    non_busco = [
        c for c in df.columns
        if c not in busco_columns
    ]

    print()
    print("Non-BUSCO columns:")

    for c in non_busco:
        print(f"  {c}")

    raise RuntimeError(
        "Unexpected BUSCO feature count. "
        "The input matrix was NOT modified."
    )

print()
print("✓ Exactly 1988 variable BUSCO features detected")


# =============================================================================
# BUSCO ID UNIQUENESS
# =============================================================================

print()
print("=" * 80)
print("BUSCO ID QC")
print("=" * 80)

unique_buscos = len(set(busco_columns))

print()
print(f"BUSCO count:       {len(busco_columns)}")
print(f"Unique BUSCO IDs:  {unique_buscos}")

if unique_buscos != len(busco_columns):
    duplicates = pd.Series(busco_columns).value_counts()
    duplicates = duplicates[duplicates > 1]

    print()
    print("Duplicate BUSCO IDs:")
    print(duplicates)

    raise RuntimeError(
        "Duplicate BUSCO IDs detected."
    )

print()
print("✓ BUSCO IDs are unique")


# =============================================================================
# BUSCO VALUE QC
# =============================================================================

print()
print("=" * 80)
print("BUSCO VALUE QC")
print("=" * 80)

busco = df[busco_columns]

# Convert values carefully.
# This handles values such as:
# 0
# 1
# "0"
# "1"
# whitespace around values.

busco_clean = busco.apply(
    lambda col: pd.to_numeric(
        col.astype(str).str.strip(),
        errors="coerce"
    )
)

missing_busco = int(
    busco_clean.isna().sum().sum()
)

print()
print(f"Missing BUSCO cells: {missing_busco}")

if missing_busco != 0:

    bad_locations = np.argwhere(
        busco_clean.isna().to_numpy()
    )

    print()
    print("Example invalid/missing entries:")

    for row_idx, col_idx in bad_locations[:20]:

        species = df.iloc[row_idx]["Species"]
        feature = busco_columns[col_idx]

        print(
            f"  Species={species} | BUSCO={feature}"
        )

    raise RuntimeError(
        "BUSCO matrix contains missing or non-numeric values."
    )


# Unique numeric values
unique_values = sorted(
    pd.unique(
        busco_clean.to_numpy().ravel()
    ).tolist()
)

print()
print("Unique BUSCO values:")
print(unique_values)

if not set(unique_values).issubset({0, 1}):

    invalid_values = [
        x for x in unique_values
        if x not in {0, 1}
    ]

    print()
    print("Invalid BUSCO values:")
    print(invalid_values)

    raise RuntimeError(
        "BUSCO matrix contains values other than 0/1."
    )

print()
print("✓ BUSCO matrix is binary")
print("✓ No missing BUSCO values")


# =============================================================================
# PHENOTYPE QC
# =============================================================================

print()
print("=" * 80)
print("PHENOTYPE QC")
print("=" * 80)

phenotype_missing = df[PHENOTYPE_COLUMNS].isna().sum()

print()
print("Missing phenotype values:")

print(phenotype_missing.to_string())

if phenotype_missing.sum() != 0:
    raise RuntimeError(
        "Missing phenotype values detected."
    )

print()
print("✓ No missing phenotype values")


# =============================================================================
# PHENOTYPE VARIANCE
# =============================================================================

print()
print("Phenotype variance:")

for col in PHENOTYPE_COLUMNS:

    variance = df[col].var()

    print(
        f"  {col}: variance={variance}"
    )

    if variance == 0:
        raise RuntimeError(
            f"Zero variance phenotype variable: {col}"
        )

print()
print("✓ Phenotype variables have non-zero variance")


# =============================================================================
# LOAD ASTRAL TREE
# =============================================================================

print()
print("=" * 80)
print("LOADING 420-TAXON ASTRAL TREE")
print("=" * 80)

tree_tips = parse_newick_tips(ASTRAL_TREE)

print()
print(f"Tree tips detected: {len(tree_tips)}")

if len(tree_tips) != EXPECTED_TAXA:

    raise RuntimeError(
        f"Expected {EXPECTED_TAXA} tree tips "
        f"but detected {len(tree_tips)}."
    )

print()
print("✓ ASTRAL tree contains exactly 420 tips")


# =============================================================================
# TREE ACCESSION EXTRACTION
# =============================================================================

tree_accessions = [
    extract_accession_from_tip(tip)
    for tip in tree_tips
]

missing_tree_accessions = [
    tip
    for tip, acc in zip(tree_tips, tree_accessions)
    if not acc
]

if missing_tree_accessions:

    print()
    print("Tree tips without recognizable assembly accession:")

    for tip in missing_tree_accessions[:20]:
        print(f"  {tip}")

    raise RuntimeError(
        "Could not extract assembly accession from one or more tree tips."
    )

tree_accession_set = set(tree_accessions)

if len(tree_accession_set) != EXPECTED_TAXA:

    raise RuntimeError(
        "Duplicate tree assembly accessions detected."
    )

print()
print("✓ Tree accession labels are unique")


# =============================================================================
# MATRIX ↔ TREE ACCESSION RECONCILIATION
# =============================================================================

print()
print("=" * 80)
print("MATRIX ↔ TREE ACCESSION RECONCILIATION")
print("=" * 80)

matrix_accessions = set(
    accession_keys
)

shared_accessions = (
    matrix_accessions &
    tree_accession_set
)

tree_only = (
    tree_accession_set -
    matrix_accessions
)

matrix_only = (
    matrix_accessions -
    tree_accession_set
)

print()
print(f"Matrix accessions:       {len(matrix_accessions)}")
print(f"Tree accessions:         {len(tree_accession_set)}")
print(f"Shared accessions:       {len(shared_accessions)}")
print(f"Tree-only accessions:    {len(tree_only)}")
print(f"Matrix-only accessions:  {len(matrix_only)}")

if tree_only:

    print()
    print("TREE-ONLY ACCESSIONS:")

    for acc in sorted(tree_only):
        print(f"  {acc}")

if matrix_only:

    print()
    print("MATRIX-ONLY ACCESSIONS:")

    for acc in sorted(matrix_only):
        print(f"  {acc}")

if len(shared_accessions) != EXPECTED_TAXA:

    raise RuntimeError(
        "Matrix and ASTRAL tree do not have exact accession-level correspondence."
    )

print()
print("✓ Exact accession-level correspondence confirmed")


# =============================================================================
# CREATE TREE ↔ MATRIX MAPPING
# =============================================================================

print()
print("=" * 80)
print("CREATING TREE ↔ MATRIX TAXON MAPPING")
print("=" * 80)

matrix_lookup = {}

for idx, row in df.iterrows():

    acc = normalize_accession(
        row["Assembly_Accession"]
    )

    matrix_lookup[acc] = {
        "Species": row["Species"],
        "Assembly_Accession": row["Assembly_Accession"],
        "Matrix_Row": idx,
    }


mapping_rows = []

for tree_tip in tree_tips:

    acc = extract_accession_from_tip(
        tree_tip
    )

    info = matrix_lookup[acc]

    mapping_rows.append(
        {
            "Tree_Tip": tree_tip,
            "Tree_Assembly_Accession": acc,
            "Tree_Species": extract_species_from_tip(tree_tip),
            "Matrix_Species": info["Species"],
            "Matrix_Assembly_Accession": info["Assembly_Accession"],
            "Matrix_Row": info["Matrix_Row"],
            "Match": True,
        }
    )

mapping_df = pd.DataFrame(mapping_rows)

mapping_path = os.path.join(
    OUTPUT_DIR,
    "tree_matrix_taxon_mapping_420.csv"
)

mapping_df.to_csv(
    mapping_path,
    index=False
)

print()
print("✓ Mapping written:")
print(mapping_path)


# =============================================================================
# CREATE PHYLOGENY-AWARE ML MATRIX
# =============================================================================

print()
print("=" * 80)
print("CREATING PHYLOGENY-AWARE ML MATRIX")
print("=" * 80)

# The final ML matrix should contain:
#
# Species
# Assembly_Accession
# phenotype variables
# BUSCO features
#
# Phenotype_Source_Species is retained as metadata.
#
# No feature transformation is performed here.
# This step only prepares the validated matrix.

output_columns = [
    "Species",
    "Assembly_Accession",
]

for col in PHENOTYPE_COLUMNS:
    output_columns.append(col)

if "Phenotype_Source_Species" in df.columns:
    output_columns.append(
        "Phenotype_Source_Species"
    )

output_columns.extend(busco_columns)

phylo_ml = df[
    output_columns
].copy()


# =============================================================================
# REORDER MATRIX TO TREE ORDER
# =============================================================================

print()
print("Ordering ML matrix according to ASTRAL tree...")

row_lookup = {}

for idx, row in phylo_ml.iterrows():

    acc = normalize_accession(
        row["Assembly_Accession"]
    )

    row_lookup[acc] = idx


ordered_indices = [
    row_lookup[
        extract_accession_from_tip(tip)
    ]
    for tip in tree_tips
]

phylo_ml_tree_order = (
    phylo_ml
    .loc[ordered_indices]
    .reset_index(drop=True)
)


# =============================================================================
# FINAL MATRIX VALIDATION
# =============================================================================

print()
print("=" * 80)
print("FINAL PHYLOGENY-AWARE ML MATRIX VALIDATION")
print("=" * 80)

print()
print(
    f"Final rows:            {len(phylo_ml_tree_order)}"
)

print(
    f"Final columns:         {phylo_ml_tree_order.shape[1]}"
)

print(
    f"BUSCO features:        {len(busco_columns)}"
)

print(
    f"Phenotype variables:   {len(PHENOTYPE_COLUMNS)}"
)

expected_columns = (
    2
    + len(PHENOTYPE_COLUMNS)
    + (1 if "Phenotype_Source_Species" in df.columns else 0)
    + EXPECTED_VARIABLE_BUSCOS
)

print(
    f"Expected columns:      {expected_columns}"
)

if len(phylo_ml_tree_order) != EXPECTED_TAXA:
    raise RuntimeError(
        "Final phylogeny-aware matrix does not contain 420 taxa."
    )

if len(busco_columns) != EXPECTED_VARIABLE_BUSCOS:
    raise RuntimeError(
        "Final phylogeny-aware matrix does not contain 1988 BUSCO features."
    )

if phylo_ml_tree_order.shape[1] != expected_columns:
    raise RuntimeError(
        "Unexpected final matrix column count."
    )

print()
print("✓ 420 taxa")
print("✓ 1988 variable BUSCO features")
print("✓ Required phenotype variables retained")
print("✓ Exact accession correspondence retained")


# =============================================================================
# SAVE FINAL MATRIX
# =============================================================================

final_matrix_path = os.path.join(
    OUTPUT_DIR,
    "y1000_420_phylogeny_aware_ml_matrix.csv"
)

phylo_ml_tree_order.to_csv(
    final_matrix_path,
    index=False
)

print()
print("✓ Final phylogeny-aware ML matrix written:")
print(final_matrix_path)


# =============================================================================
# SAVE BUSCO FEATURE LIST
# =============================================================================

busco_list_path = os.path.join(
    OUTPUT_DIR,
    "y1000_420_variable_busco_feature_list.csv"
)

pd.DataFrame(
    {
        "BUSCO_ID": busco_columns
    }
).to_csv(
    busco_list_path,
    index=False
)

print()
print("✓ BUSCO feature list written:")
print(busco_list_path)


# =============================================================================
# SAVE TAXON ORDER
# =============================================================================

taxon_order_path = os.path.join(
    OUTPUT_DIR,
    "y1000_420_astral_taxon_order.csv"
)

taxon_order_df = pd.DataFrame(
    {
        "Tree_Order": range(
            1,
            len(tree_tips) + 1
        ),
        "Tree_Tip": tree_tips,
        "Assembly_Accession": [
            extract_accession_from_tip(t)
            for t in tree_tips
        ],
        "Species": [
            extract_species_from_tip(t)
            for t in tree_tips
        ],
    }
)

taxon_order_df.to_csv(
    taxon_order_path,
    index=False
)

print()
print("✓ Tree taxon order written:")
print(taxon_order_path)


# =============================================================================
# SAVE PREPARATION SUMMARY
# =============================================================================

summary = pd.DataFrame(
    [
        {
            "ML_Taxa": EXPECTED_TAXA,
            "BUSCO_Features": EXPECTED_VARIABLE_BUSCOS,
            "Phenotype_Variables": len(PHENOTYPE_COLUMNS),
            "Tree_Tips": len(tree_tips),
            "Shared_Accessions": len(shared_accessions),
            "Tree_Only_Accessions": len(tree_only),
            "Matrix_Only_Accessions": len(matrix_only),
            "Missing_BUSCO_Cells": missing_busco,
            "Negative_Branch_Lengths_Checked": "Not evaluated in lightweight parser",
            "Status": "PASSED",
        }
    ]
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "phylogeny_aware_ml_preparation_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# =============================================================================
# COMPLETE
# =============================================================================

print()
print("=" * 80)
print("PHYLOGENY-AWARE ML PREPARATION COMPLETE")
print("=" * 80)

print()
print("Final taxa:                  420")
print("Variable BUSCO features:     1988")
print("ASTRAL tree tips:             420")
print("Shared accessions:             420")
print("Tree-only accessions:            0")
print("Matrix-only accessions:          0")
print("Missing BUSCO cells:             0")

print()
print("OUTPUTS:")

print()
print("Final ML matrix:")
print(final_matrix_path)

print()
print("BUSCO feature list:")
print(busco_list_path)

print()
print("ASTRAL taxon order:")
print(taxon_order_path)

print()
print("Taxon mapping:")
print(mapping_path)

print()
print("Summary:")
print(summary_path)

print()
print("✓ Original 420 ML matrix was NOT modified.")
print("✓ Original ASTRAL tree was NOT modified.")
print("✓ 1988 variable BUSCO features retained.")
print("✓ 420 taxa retained.")
print("✓ Exact accession-level correspondence confirmed.")
print("✓ Dataset prepared for phylogeny-aware ML.")
print("=" * 80)