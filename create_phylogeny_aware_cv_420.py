# -*- coding: utf-8 -*-

"""
create_phylogeny_aware_cv_420.py

Create phylogeny-aware cross-validation splits for the validated
420-taxon Y1000 chassis dataset.

Inputs
------
1. Final phylogeny-aware ML matrix
2. Final 420-taxon ASTRAL tree

Outputs
-------
phylogeny_aware_ml/
    phylogeny_cv/
        phylogenetic_cv_assignments_420.csv
        phylogenetic_cv_summary_420.csv
        random_cv_assignments_420.csv
        random_cv_summary_420.csv
        phylogenetic_cv_fold_1.csv
        ...
        phylogenetic_cv_fold_5.csv
        random_cv_fold_1.csv
        ...
        random_cv_fold_5.csv

Important
---------
- Original ML matrix is NOT modified.
- Original ASTRAL tree is NOT modified.
- Species and assembly accession are retained.
- Splits are reproducible.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# ============================================================
# OPTIONAL TREE LIBRARY
# ============================================================

try:
    from Bio import Phylo
except ImportError:
    print("\nERROR: Biopython is not installed.")
    print("Install with:")
    print("    python -m pip install biopython")
    sys.exit(1)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

ML_MATRIX = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "y1000_420_phylogeny_aware_ml_matrix.csv"
)

ASTRAL_TREE = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_qc"
    / "ASTRAL_420_taxon_pruned.nwk"
)

OUTPUT_DIR = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

N_TAXA_EXPECTED = 420
N_FOLDS = 5
RANDOM_STATE = 42

ID_COLUMNS = [
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

SOURCE_COLUMN = "Phenotype_Source_Species"


# ============================================================
# HELPERS
# ============================================================

def norm_string(x):
    """Normalize strings for matching."""
    if pd.isna(x):
        return ""

    return (
        str(x)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .lower()
    )


def accession_from_tip(tip):
    """
    Extract assembly accession from ASTRAL tip.

    Expected examples:
        Species_name__GCA_030558845.1
        Species_name__GCA_003705225.1
    """
    tip = str(tip).strip()

    if "__" in tip:
        accession = tip.split("__")[-1]
    else:
        accession = tip

    return accession.strip()


def species_from_tip(tip):
    """
    Recover species portion of ASTRAL tip.
    """
    tip = str(tip).strip()

    if "__" in tip:
        species = tip.rsplit("__", 1)[0]
    else:
        species = tip

    return species.replace("_", " ").strip()


def get_tree_tips(tree):
    """Return terminal tip labels."""
    return [
        str(clade.name).strip()
        for clade in tree.get_terminals()
        if clade.name is not None
    ]


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("CREATING PHYLOGENY-AWARE CV SPLITS FOR 420 TAXA")
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


# ============================================================
# INPUT CHECK
# ============================================================

print()
print("=" * 80)
print("CHECKING INPUTS")
print("=" * 80)

if not ML_MATRIX.exists():
    raise FileNotFoundError(
        f"Final ML matrix not found:\n{ML_MATRIX}"
    )

if not ASTRAL_TREE.exists():
    raise FileNotFoundError(
        f"ASTRAL tree not found:\n{ASTRAL_TREE}"
    )

print("✓ Final phylogeny-aware ML matrix found")
print("✓ Final 420-taxon ASTRAL tree found")


# ============================================================
# LOAD ML MATRIX
# ============================================================

print()
print("=" * 80)
print("LOADING FINAL ML MATRIX")
print("=" * 80)

df = pd.read_csv(ML_MATRIX)

print()
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

if len(df) != N_TAXA_EXPECTED:
    raise RuntimeError(
        f"Expected {N_TAXA_EXPECTED} taxa but found {len(df)}."
    )

for col in ID_COLUMNS:
    if col not in df.columns:
        raise RuntimeError(
            f"Required identifier column missing: {col}"
        )

for col in PHENOTYPE_COLUMNS:
    if col not in df.columns:
        raise RuntimeError(
            f"Required phenotype column missing: {col}"
        )

print("✓ Exactly 420 taxa")
print("✓ Required identifier columns present")
print("✓ Required phenotype columns present")


# ============================================================
# TAXON QC
# ============================================================

print()
print("=" * 80)
print("ML TAXON QC")
print("=" * 80)

species_keys = df["Species"].map(norm_string)
accession_keys = df["Assembly_Accession"].map(norm_string)

duplicate_species = species_keys.duplicated().sum()
duplicate_accessions = accession_keys.duplicated().sum()

print()
print(f"Duplicate species:     {duplicate_species}")
print(f"Duplicate accessions:  {duplicate_accessions}")

if duplicate_species != 0:
    raise RuntimeError("Duplicate species detected.")

if duplicate_accessions != 0:
    raise RuntimeError("Duplicate assembly accessions detected.")

print("✓ Species are unique")
print("✓ Assembly accessions are unique")


# ============================================================
# LOAD ASTRAL TREE
# ============================================================

print()
print("=" * 80)
print("LOADING FINAL ASTRAL TREE")
print("=" * 80)

tree = Phylo.read(str(ASTRAL_TREE), "newick")

tree_tips = get_tree_tips(tree)

print()
print(f"Tree tips: {len(tree_tips)}")

if len(tree_tips) != N_TAXA_EXPECTED:
    raise RuntimeError(
        f"Expected {N_TAXA_EXPECTED} tree tips but found "
        f"{len(tree_tips)}."
    )

print("✓ Tree contains exactly 420 tips")


# ============================================================
# TREE ACCESSION QC
# ============================================================

print()
print("=" * 80)
print("TREE TIP ACCESSION QC")
print("=" * 80)

tree_accessions = [
    accession_from_tip(t)
    for t in tree_tips
]

tree_accession_keys = [
    norm_string(x)
    for x in tree_accessions
]

duplicate_tree_accessions = (
    pd.Series(tree_accession_keys).duplicated().sum()
)

print()
print(f"Tree accessions:       {len(tree_accession_keys)}")
print(f"Duplicate accessions:  {duplicate_tree_accessions}")

if duplicate_tree_accessions != 0:
    raise RuntimeError(
        "Duplicate assembly accessions detected in ASTRAL tree."
    )

print("✓ Tree accession labels are unique")


# ============================================================
# MATRIX ↔ TREE RECONCILIATION
# ============================================================

print()
print("=" * 80)
print("MATRIX ↔ TREE RECONCILIATION")
print("=" * 80)

matrix_accessions = set(accession_keys)
tree_accessions_set = set(tree_accession_keys)

shared = matrix_accessions & tree_accessions_set
tree_only = tree_accessions_set - matrix_accessions
matrix_only = matrix_accessions - tree_accessions_set

print()
print(f"Matrix accessions:      {len(matrix_accessions)}")
print(f"Tree accessions:        {len(tree_accessions_set)}")
print(f"Shared accessions:      {len(shared)}")
print(f"Tree-only:              {len(tree_only)}")
print(f"Matrix-only:            {len(matrix_only)}")

if tree_only:
    print()
    print("Tree-only accessions:")
    for x in sorted(tree_only):
        print(" ", x)

if matrix_only:
    print()
    print("Matrix-only accessions:")
    for x in sorted(matrix_only):
        print(" ", x)

if tree_only or matrix_only:
    raise RuntimeError(
        "Tree and matrix do not have exact accession correspondence."
    )

print()
print("✓ Exact 420/420 accession correspondence confirmed")


# ============================================================
# CREATE TREE ORDER
# ============================================================

print()
print("=" * 80)
print("CREATING ASTRAL TREE ORDER")
print("=" * 80)

accession_to_row = {
    norm_string(row["Assembly_Accession"]): idx
    for idx, row in df.iterrows()
}

tree_order_indices = [
    accession_to_row[norm_string(accession_from_tip(t))]
    for t in tree_tips
]

if len(tree_order_indices) != N_TAXA_EXPECTED:
    raise RuntimeError(
        "Tree ordering did not produce exactly 420 taxa."
    )

ordered_df = df.iloc[tree_order_indices].copy()

ordered_accessions = (
    ordered_df["Assembly_Accession"]
    .map(norm_string)
    .tolist()
)

if ordered_accessions != tree_accession_keys:
    raise RuntimeError(
        "Failed to reproduce exact ASTRAL taxon order."
    )

print("✓ ML taxa ordered according to ASTRAL tree")


# ============================================================
# PHYLOGENETIC BLOCK STRATEGY
# ============================================================

print()
print("=" * 80)
print("CREATING PHYLOGENETIC BLOCK FOLDS")
print("=" * 80)

print()
print(
    "Strategy:"
)
print(
    "The ASTRAL tree is recursively divided into approximately "
    "equal-sized contiguous phylogenetic blocks."
)

print(
    "Each block is assigned as a validation fold."
)

print(
    "This keeps closely positioned taxa together more often than "
    "ordinary random splitting."
)


# ============================================================
# TREE-BASED CONTIGUOUS BLOCKS
# ============================================================

# A simple and transparent approach:
# preserve ASTRAL order and divide the ordered taxa into 5
# approximately equal-sized contiguous blocks.

n_taxa = len(ordered_df)

fold_labels = np.empty(n_taxa, dtype=int)

fold_sizes = [
    n_taxa // N_FOLDS
] * N_FOLDS

for i in range(n_taxa % N_FOLDS):
    fold_sizes[i] += 1

start = 0

for fold_id, size in enumerate(fold_sizes, start=1):

    end = start + size

    fold_labels[start:end] = fold_id

    start = end


if np.any(pd.isna(fold_labels)):
    raise RuntimeError(
        "Some taxa did not receive a phylogenetic fold."
    )


print()
print("Phylogenetic fold sizes:")

for fold_id in range(1, N_FOLDS + 1):

    count = int(np.sum(fold_labels == fold_id))

    print(
        f"  Fold {fold_id}: {count} taxa"
    )


# ============================================================
# CREATE PHYLOGENETIC ASSIGNMENT TABLE
# ============================================================

phylo_assignments = ordered_df[
    [
        "Species",
        "Assembly_Accession",
    ]
].copy()

phylo_assignments.insert(
    0,
    "ASTRAL_Order",
    np.arange(1, n_taxa + 1)
)

phylo_assignments["Phylogenetic_Fold"] = fold_labels

phylo_assignments["Fold_Type"] = "PHYLOGENETIC_BLOCK"


# ============================================================
# ADD TREE TIP INFORMATION
# ============================================================

phylo_assignments["ASTRAL_Tip"] = [
    tree_tips[i]
    for i in range(n_taxa)
]

phylo_assignments["Tree_Species"] = [
    species_from_tip(tree_tips[i])
    for i in range(n_taxa)
]


# ============================================================
# VALIDATE PHYLOGENETIC ASSIGNMENTS
# ============================================================

print()
print("=" * 80)
print("VALIDATING PHYLOGENETIC FOLDS")
print("=" * 80)

if len(phylo_assignments) != 420:
    raise RuntimeError(
        "Phylogenetic assignment table does not contain 420 taxa."
    )

if phylo_assignments["Assembly_Accession"].nunique() != 420:
    raise RuntimeError(
        "Phylogenetic assignment table contains duplicate accessions."
    )

if phylo_assignments["Species"].nunique() != 420:
    raise RuntimeError(
        "Phylogenetic assignment table contains duplicate species."
    )

if phylo_assignments["Phylogenetic_Fold"].isna().any():
    raise RuntimeError(
        "Some taxa have no phylogenetic fold."
    )

if set(phylo_assignments["Phylogenetic_Fold"]) != set(
    range(1, N_FOLDS + 1)
):
    raise RuntimeError(
        "Not all phylogenetic folds are represented."
    )

print("✓ All 420 taxa assigned")
print("✓ Species are unique")
print("✓ Accessions are unique")
print("✓ All 5 folds represented")


# ============================================================
# SAVE PHYLOGENETIC ASSIGNMENTS
# ============================================================

phylo_assignment_file = (
    OUTPUT_DIR
    / "phylogenetic_cv_assignments_420.csv"
)

phylo_assignments.to_csv(
    phylo_assignment_file,
    index=False
)

print()
print("✓ Phylogenetic assignment table written:")
print(phylo_assignment_file)


# ============================================================
# CREATE PHYLOGENETIC FOLD FILES
# ============================================================

print()
print("=" * 80)
print("WRITING PHYLOGENETIC FOLD FILES")
print("=" * 80)

phylo_summary_rows = []

for fold_id in range(1, N_FOLDS + 1):

    validation_mask = (
        phylo_assignments["Phylogenetic_Fold"]
        == fold_id
    )

    validation_accessions = set(
        phylo_assignments.loc[
            validation_mask,
            "Assembly_Accession"
        ].map(norm_string)
    )

    train_mask = (
        ~phylo_assignments["Assembly_Accession"]
        .map(norm_string)
        .isin(validation_accessions)
    )

    fold_df = phylo_assignments.copy()

    fold_df["Set"] = np.where(
        validation_mask,
        "VALIDATION",
        "TRAIN"
    )

    fold_df["Fold"] = fold_id

    fold_file = (
        OUTPUT_DIR
        / f"phylogenetic_cv_fold_{fold_id}.csv"
    )

    fold_df.to_csv(
        fold_file,
        index=False
    )

    train_count = int(train_mask.sum())
    validation_count = int(validation_mask.sum())

    phylo_summary_rows.append({
        "Fold": fold_id,
        "Train_Taxa": train_count,
        "Validation_Taxa": validation_count,
        "Total_Taxa": train_count + validation_count,
        "Fold_Type": "PHYLOGENETIC_BLOCK",
    })

    print(
        f"✓ Fold {fold_id}: "
        f"Train={train_count}, "
        f"Validation={validation_count}"
    )


phylo_summary = pd.DataFrame(
    phylo_summary_rows
)

phylo_summary_file = (
    OUTPUT_DIR
    / "phylogenetic_cv_summary_420.csv"
)

phylo_summary.to_csv(
    phylo_summary_file,
    index=False
)

print()
print("✓ Phylogenetic CV summary written:")
print(phylo_summary_file)


# ============================================================
# RANDOM CV BASELINE
# ============================================================

print()
print("=" * 80)
print("CREATING RANDOM CV BASELINE")
print("=" * 80)

rng = np.random.default_rng(RANDOM_STATE)

random_indices = np.arange(n_taxa)

rng.shuffle(random_indices)

random_fold_labels = np.empty(
    n_taxa,
    dtype=int
)

random_fold_sizes = [
    n_taxa // N_FOLDS
] * N_FOLDS

for i in range(n_taxa % N_FOLDS):
    random_fold_sizes[i] += 1

start = 0

for fold_id, size in enumerate(
    random_fold_sizes,
    start=1
):

    end = start + size

    indices = random_indices[start:end]

    random_fold_labels[indices] = fold_id

    start = end


if set(random_fold_labels) != set(
    range(1, N_FOLDS + 1)
):
    raise RuntimeError(
        "Random CV assignment failed."
    )


random_assignments = df[
    [
        "Species",
        "Assembly_Accession",
    ]
].copy()

random_assignments["Random_Fold"] = random_fold_labels

random_assignments["Fold_Type"] = "RANDOM"

random_assignment_file = (
    OUTPUT_DIR
    / "random_cv_assignments_420.csv"
)

random_assignments.to_csv(
    random_assignment_file,
    index=False
)

print()
print("✓ Random CV assignment written:")
print(random_assignment_file)


# ============================================================
# RANDOM FOLD FILES
# ============================================================

random_summary_rows = []

for fold_id in range(1, N_FOLDS + 1):

    validation_mask = (
        random_assignments["Random_Fold"]
        == fold_id
    )

    validation_accessions = set(
        random_assignments.loc[
            validation_mask,
            "Assembly_Accession"
        ].map(norm_string)
    )

    train_mask = (
        ~random_assignments["Assembly_Accession"]
        .map(norm_string)
        .isin(validation_accessions)
    )

    fold_df = random_assignments.copy()

    fold_df["Set"] = np.where(
        validation_mask,
        "VALIDATION",
        "TRAIN"
    )

    fold_df["Fold"] = fold_id

    fold_file = (
        OUTPUT_DIR
        / f"random_cv_fold_{fold_id}.csv"
    )

    fold_df.to_csv(
        fold_file,
        index=False
    )

    train_count = int(train_mask.sum())
    validation_count = int(validation_mask.sum())

    random_summary_rows.append({
        "Fold": fold_id,
        "Train_Taxa": train_count,
        "Validation_Taxa": validation_count,
        "Total_Taxa": train_count + validation_count,
        "Fold_Type": "RANDOM",
    })

    print(
        f"✓ Random Fold {fold_id}: "
        f"Train={train_count}, "
        f"Validation={validation_count}"
    )


random_summary = pd.DataFrame(
    random_summary_rows
)

random_summary_file = (
    OUTPUT_DIR
    / "random_cv_summary_420.csv"
)

random_summary.to_csv(
    random_summary_file,
    index=False
)

print()
print("✓ Random CV summary written:")
print(random_summary_file)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 80)
print("FINAL CV VALIDATION")
print("=" * 80)

# Every taxon must occur exactly once in every fold assignment
phylo_counts = (
    phylo_assignments[
        "Assembly_Accession"
    ].map(norm_string)
    .value_counts()
)

random_counts = (
    random_assignments[
        "Assembly_Accession"
    ].map(norm_string)
    .value_counts()
)

if not (phylo_counts == 1).all():
    raise RuntimeError(
        "Phylogenetic CV does not contain exactly one assignment per taxon."
    )

if not (random_counts == 1).all():
    raise RuntimeError(
        "Random CV does not contain exactly one assignment per taxon."
    )

print("✓ Every taxon has exactly one phylogenetic fold")
print("✓ Every taxon has exactly one random fold")


# ============================================================
# CREATE OVERALL SUMMARY
# ============================================================

overall_summary = pd.DataFrame([
    {
        "Metric": "Final_Taxa",
        "Value": 420,
    },
    {
        "Metric": "Number_of_Folds",
        "Value": N_FOLDS,
    },
    {
        "Metric": "Random_State",
        "Value": RANDOM_STATE,
    },
    {
        "Metric": "Phylogenetic_Assignment",
        "Value": "ASTRAL_order_contiguous_blocks",
    },
    {
        "Metric": "Random_Baseline",
        "Value": "Seeded_random_assignment",
    },
    {
        "Metric": "Original_ML_Modified",
        "Value": False,
    },
    {
        "Metric": "Original_Tree_Modified",
        "Value": False,
    },
])

overall_summary_file = (
    OUTPUT_DIR
    / "phylogeny_cv_overall_summary_420.csv"
)

overall_summary.to_csv(
    overall_summary_file,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 80)
print("PHYLOGENY-AWARE CV PREPARATION COMPLETE")
print("=" * 80)

print()
print(f"Final taxa:                 {N_TAXA_EXPECTED}")
print(f"Cross-validation folds:     {N_FOLDS}")
print("Phylogenetic folds:         5")
print("Random baseline folds:      5")

print()
print("OUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print()
print("PHYLOGENETIC ASSIGNMENTS:")
print(phylo_assignment_file)

print()
print("PHYLOGENETIC SUMMARY:")
print(phylo_summary_file)

print()
print("RANDOM ASSIGNMENTS:")
print(random_assignment_file)

print()
print("RANDOM SUMMARY:")
print(random_summary_file)

print()
print("OVERALL SUMMARY:")
print(overall_summary_file)

print()
print("✓ Original 420 ML matrix was NOT modified.")
print("✓ Original ASTRAL tree was NOT modified.")
print("✓ All 420 taxa were retained.")
print("✓ Every taxon received exactly one phylogenetic fold.")
print("✓ Every taxon received exactly one random baseline fold.")
print("✓ Reproducible random seed = 42.")

print()
print("=" * 80)
print("READY FOR PHYLOGENY-AWARE MODEL TRAINING")
print("=" * 80)