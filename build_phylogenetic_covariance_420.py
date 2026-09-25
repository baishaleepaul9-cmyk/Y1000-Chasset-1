# -*- coding: utf-8 -*-

"""
BUILD PHYLOGENETIC DISTANCE AND COVARIANCE MATRICES
FOR THE FINAL 420-TAXON ML DATASET

INPUTS
------
Final phylogeny-aware ML matrix:
C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\y1000_420_phylogeny_aware_ml_matrix.csv

Final 420-taxon ASTRAL tree:
C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_qc\ASTRAL_420_taxon_pruned.nwk

OUTPUT
------
C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogenetic_structure

This script:
1. Loads the final 420-taxon ML matrix.
2. Loads the final 420-taxon ASTRAL tree.
3. Verifies exact accession-level correspondence.
4. Calculates the patristic phylogenetic distance matrix.
5. Calculates a Brownian-motion-style covariance matrix from
   the rooted tree.
6. Checks symmetry and numerical properties.
7. Saves taxon order and QC summaries.
8. Does NOT modify the ML matrix or ASTRAL tree.

IMPORTANT
---------
The covariance matrix is a tree-derived relationship matrix.
It is intended as the phylogenetic structure for subsequent
phylogeny-aware ML/statistical modelling.

No ML model is trained by this script.
"""

import os
import re
import sys
import math
import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT = r"C:\Y1000_chassis_project"

ML_MATRIX = (
    PROJECT
    + r"\results\stage5_phylogeny_ml_dataset"
    + r"\phylogeny_aware_ml"
    + r"\y1000_420_phylogeny_aware_ml_matrix.csv"
)

ASTRAL_TREE = (
    PROJECT
    + r"\results\stage5_phylogeny_ml_dataset"
    + r"\phylogeny_qc"
    + r"\ASTRAL_420_taxon_pruned.nwk"
)

OUTPUT_DIR = (
    PROJECT
    + r"\results\stage5_phylogeny_ml_dataset"
    + r"\phylogeny_aware_ml"
    + r"\phylogenetic_structure"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# EXPECTED VALUES
# =============================================================================

EXPECTED_TAXA = 420

PHENOTYPE_COLUMNS = [
    "N_Strains",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Carbon_Breadth_SD",
    "Nitrogen_Breadth_SD",
]

IDENTIFIER_COLUMNS = [
    "Species",
    "Assembly_Accession",
]


# =============================================================================
# HELPERS
# =============================================================================

def normalize_accession(value):
    """Normalize assembly accession."""

    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def normalize_species(value):
    """Normalize species name."""

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


def extract_accession(tip):
    """
    Extract GCA/GCF accession from ASTRAL tip.

    Example:
    Ambrosiozyma_vanderkliftii__GCA_003705225.1

    -> GCA_003705225.1
    """

    match = re.search(
        r"((?:GCA|GCF)_\d+\.\d+)",
        str(tip)
    )

    if match:
        return match.group(1).upper()

    return ""


def extract_species(tip):
    """
    Extract species portion from ASTRAL tip.
    """

    accession = extract_accession(tip)

    if accession:

        prefix = str(tip).split(
            "__" + accession
        )[0]

        return prefix.replace(
            "_",
            " "
        ).strip()

    return str(tip).replace(
        "_",
        " "
    ).strip()


# =============================================================================
# NEWICK PARSER
# =============================================================================

class Node:
    def __init__(self):
        self.name = ""
        self.length = 0.0
        self.children = []
        self.parent = None

    @property
    def is_leaf(self):
        return len(self.children) == 0


def parse_newick(text):
    """
    Parse a standard rooted Newick tree.

    Supports:
        taxon:branch_length
        (A:0.1,B:0.2):0.3

    Internal node labels are ignored for covariance calculation.
    """

    text = text.strip()

    if text.endswith(";"):
        text = text[:-1]

    root = Node()

    stack = [root]
    current = root

    token = ""
    reading_length = False
    length_token = ""

    def finish_token():
        nonlocal token
        nonlocal length_token
        nonlocal reading_length

        token_clean = token.strip()

        if token_clean:
            current.name = token_clean

        if length_token.strip():

            try:
                current.length = float(
                    length_token.strip()
                )

            except ValueError:
                current.length = 0.0

        token = ""
        length_token = ""
        reading_length = False

    i = 0

    while i < len(text):

        char = text[i]

        if char == "(":

            # Current node becomes an internal node
            new_node = Node()

            new_node.parent = current

            current.children.append(
                new_node
            )

            stack.append(
                new_node
            )

            current = new_node

            token = ""
            length_token = ""
            reading_length = False

        elif char == ",":

            finish_token()

            parent = current.parent

            if parent is None:
                raise RuntimeError(
                    "Malformed Newick tree."
                )

            new_node = Node()

            new_node.parent = parent

            parent.children.append(
                new_node
            )

            current = new_node

        elif char == ")":

            finish_token()

            if len(stack) <= 1:
                raise RuntimeError(
                    "Malformed Newick tree."
                )

            completed = stack.pop()

            current = completed.parent

        elif char == ":":

            reading_length = True

        else:

            if reading_length:
                length_token += char
            else:
                token += char

        i += 1

    finish_token()

    return root


# =============================================================================
# LOAD TREE
# =============================================================================

print("=" * 80)
print("BUILDING PHYLOGENETIC STRUCTURE — FINAL 420 TAXA")
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
# INPUT CHECK
# =============================================================================

print()
print("=" * 80)
print("CHECKING INPUTS")
print("=" * 80)

if not os.path.isfile(ML_MATRIX):

    raise FileNotFoundError(
        f"ML matrix not found:\n{ML_MATRIX}"
    )

print("✓ Final ML matrix found")


if not os.path.isfile(ASTRAL_TREE):

    raise FileNotFoundError(
        f"ASTRAL tree not found:\n{ASTRAL_TREE}"
    )

print("✓ Final ASTRAL tree found")


# =============================================================================
# LOAD ML MATRIX
# =============================================================================

print()
print("=" * 80)
print("LOADING FINAL 420-TAXON ML MATRIX")
print("=" * 80)

df = pd.read_csv(
    ML_MATRIX
)

print()
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


if len(df) != EXPECTED_TAXA:

    raise RuntimeError(
        f"Expected {EXPECTED_TAXA} taxa "
        f"but found {len(df)}."
    )


for col in IDENTIFIER_COLUMNS:

    if col not in df.columns:

        raise RuntimeError(
            f"Missing required column: {col}"
        )


for col in PHENOTYPE_COLUMNS:

    if col not in df.columns:

        raise RuntimeError(
            f"Missing phenotype column: {col}"
        )


print()
print("✓ Exactly 420 taxa")
print("✓ Identifier columns present")
print("✓ Phenotype columns present")


# =============================================================================
# MATRIX ACCESSION QC
# =============================================================================

matrix_accessions = (
    df["Assembly_Accession"]
    .map(normalize_accession)
)

matrix_species = (
    df["Species"]
    .map(normalize_species)
)

if matrix_accessions.duplicated().any():

    duplicates = (
        matrix_accessions[
            matrix_accessions.duplicated(
                keep=False
            )
        ]
        .tolist()
    )

    print()
    print("Duplicate accessions:")
    print(duplicates)

    raise RuntimeError(
        "Duplicate assembly accessions in ML matrix."
    )


if matrix_species.duplicated().any():

    duplicates = (
        matrix_species[
            matrix_species.duplicated(
                keep=False
            )
        ]
        .tolist()
    )

    print()
    print("Duplicate species:")
    print(duplicates)

    raise RuntimeError(
        "Duplicate species in ML matrix."
    )


print()
print("✓ Matrix species are unique")
print("✓ Matrix accessions are unique")


# =============================================================================
# READ NEWICK
# =============================================================================

print()
print("=" * 80)
print("LOADING ASTRAL TREE")
print("=" * 80)

with open(
    ASTRAL_TREE,
    "r",
    encoding="utf-8"
) as handle:

    newick = handle.read().strip()


if not newick:

    raise RuntimeError(
        "ASTRAL tree file is empty."
    )


root = parse_newick(
    newick
)


# =============================================================================
# COLLECT TREE TIPS
# =============================================================================

tree_nodes = []
tree_tips = []


def traverse(node):

    tree_nodes.append(
        node
    )

    if node.is_leaf:

        tree_tips.append(
            node
        )

    else:

        for child in node.children:

            traverse(
                child
            )


traverse(root)


print()
print(f"Tree tips: {len(tree_tips)}")


if len(tree_tips) != EXPECTED_TAXA:

    raise RuntimeError(
        f"Expected {EXPECTED_TAXA} "
        f"tree tips but found "
        f"{len(tree_tips)}."
    )


print()
print("✓ ASTRAL tree contains exactly 420 tips")


# =============================================================================
# TREE ACCESSION QC
# =============================================================================

tree_accessions = []

for node in tree_tips:

    accession = extract_accession(
        node.name
    )

    if not accession:

        raise RuntimeError(
            "Could not extract accession from "
            f"tree tip: {node.name}"
        )

    tree_accessions.append(
        accession
    )


tree_accession_set = set(
    tree_accessions
)


if len(tree_accession_set) != EXPECTED_TAXA:

    raise RuntimeError(
        "Duplicate tree accessions detected."
    )


print()
print("✓ Tree accessions are unique")


# =============================================================================
# MATRIX ↔ TREE RECONCILIATION
# =============================================================================

print()
print("=" * 80)
print("MATRIX ↔ TREE RECONCILIATION")
print("=" * 80)

matrix_accession_set = set(
    matrix_accessions
)

shared = (
    matrix_accession_set
    & tree_accession_set
)

tree_only = (
    tree_accession_set
    - matrix_accession_set
)

matrix_only = (
    matrix_accession_set
    - tree_accession_set
)

print()
print(f"Matrix taxa:             {len(matrix_accession_set)}")
print(f"Tree taxa:               {len(tree_accession_set)}")
print(f"Shared accessions:       {len(shared)}")
print(f"Tree-only accessions:    {len(tree_only)}")
print(f"Matrix-only accessions:  {len(matrix_only)}")


if tree_only:

    print()
    print("Tree-only:")

    for x in sorted(tree_only):
        print(
            f"  {x}"
        )


if matrix_only:

    print()
    print("Matrix-only:")

    for x in sorted(matrix_only):
        print(
            f"  {x}"
        )


if len(shared) != EXPECTED_TAXA:

    raise RuntimeError(
        "Tree and ML matrix do not have exact "
        "accession-level correspondence."
    )


print()
print("✓ Exact accession-level correspondence confirmed")


# =============================================================================
# TREE HEIGHT
# =============================================================================

print()
print("=" * 80)
print("TREE BRANCH QC")
print("=" * 80)

all_branch_lengths = []

for node in tree_nodes:

    if node.parent is not None:

        all_branch_lengths.append(
            node.length
        )


missing_lengths = sum(
    not np.isfinite(x)
    for x in all_branch_lengths
)

negative_lengths = sum(
    x < 0
    for x in all_branch_lengths
)

zero_lengths = sum(
    x == 0
    for x in all_branch_lengths
)

print()
print(
    f"Branches:             {len(all_branch_lengths)}"
)

print(
    f"Missing/invalid:      {missing_lengths}"
)

print(
    f"Negative:             {negative_lengths}"
)

print(
    f"Zero-length:          {zero_lengths}"
)

if missing_lengths:

    raise RuntimeError(
        "Tree contains missing/invalid branch lengths."
    )

if negative_lengths:

    raise RuntimeError(
        "Tree contains negative branch lengths."
    )


print()
print("✓ Branch lengths are valid")


# =============================================================================
# PHYLOGENETIC DISTANCE
# =============================================================================

print()
print("=" * 80)
print("CALCULATING PATRISTIC DISTANCE MATRIX")
print("=" * 80)


def root_to_tip_distance(node):

    distance = 0.0

    current = node

    while current.parent is not None:

        distance += current.length

        current = current.parent

    return distance


tip_depths = {
    extract_accession(node.name): root_to_tip_distance(node)
    for node in tree_tips
}


# Calculate ancestor relationships
def ancestor_path(node):

    path = []

    current = node

    while current is not None:

        path.append(current)

        current = current.parent

    return path


tip_paths = {
    extract_accession(node.name): ancestor_path(node)
    for node in tree_tips
}


tip_node_by_accession = {
    extract_accession(node.name): node
    for node in tree_tips
}


ordered_accessions = [
    extract_accession(node.name)
    for node in tree_tips
]


n = len(
    ordered_accessions
)

distance_matrix = np.zeros(
    (n, n),
    dtype=float
)


for i in range(n):

    acc_i = ordered_accessions[i]

    path_i = tip_paths[acc_i]

    path_i_set = set(
        path_i
    )

    for j in range(i + 1, n):

        acc_j = ordered_accessions[j]

        path_j = tip_paths[acc_j]

        # Find MRCA
        mrca = None

        for ancestor in path_i:

            if ancestor in set(path_j):

                mrca = ancestor

                break

        if mrca is None:

            raise RuntimeError(
                f"Could not find MRCA for "
                f"{acc_i} and {acc_j}"
            )

        depth_i = tip_depths[acc_i]
        depth_j = tip_depths[acc_j]

        depth_mrca = 0.0

        current = mrca

        while current.parent is not None:

            depth_mrca += current.length

            current = current.parent

        distance = (
            depth_i
            + depth_j
            - 2.0 * depth_mrca
        )

        # Guard against tiny floating-point errors
        if abs(distance) < 1e-12:

            distance = 0.0

        distance_matrix[i, j] = distance
        distance_matrix[j, i] = distance


distance_df = pd.DataFrame(
    distance_matrix,
    index=ordered_accessions,
    columns=ordered_accessions
)

distance_path = os.path.join(
    OUTPUT_DIR,
    "phylogenetic_patristic_distance_420.csv"
)

distance_df.to_csv(
    distance_path
)


print()
print("✓ Patristic distance matrix calculated")

print(
    f"Dimensions: {distance_matrix.shape}"
)

print(
    f"Minimum distance: {distance_matrix.min():.10f}"
)

print(
    f"Maximum distance: {distance_matrix.max():.10f}"
)


# =============================================================================
# DISTANCE MATRIX QC
# =============================================================================

print()
print("=" * 80)
print("DISTANCE MATRIX QC")
print("=" * 80)

distance_symmetry_error = np.max(
    np.abs(
        distance_matrix
        - distance_matrix.T
    )
)

distance_diagonal_max = np.max(
    np.abs(
        np.diag(distance_matrix)
    )
)

negative_distance_count = np.sum(
    distance_matrix < -1e-10
)

print()
print(
    f"Maximum symmetry error: "
    f"{distance_symmetry_error:.12e}"
)

print(
    f"Maximum diagonal value: "
    f"{distance_diagonal_max:.12e}"
)

print(
    f"Negative distances:     "
    f"{negative_distance_count}"
)


if distance_symmetry_error > 1e-8:

    raise RuntimeError(
        "Patristic distance matrix is not symmetric."
    )


if distance_diagonal_max > 1e-8:

    raise RuntimeError(
        "Patristic distance matrix diagonal is not zero."
    )


if negative_distance_count != 0:

    raise RuntimeError(
        "Negative phylogenetic distances detected."
    )


print()
print("✓ Distance matrix is symmetric")
print("✓ Diagonal is zero")
print("✓ No negative distances")


# =============================================================================
# BROWNIAN-MOTION COVARIANCE MATRIX
# =============================================================================

print()
print("=" * 80)
print("CALCULATING PHYLOGENETIC COVARIANCE MATRIX")
print("=" * 80)

"""
For a rooted tree under a Brownian-motion interpretation:

Cov(i,j) = shared path length from root to MRCA(i,j)

The diagonal is the root-to-tip distance.

This matrix represents shared evolutionary history.
"""


covariance_matrix = np.zeros(
    (n, n),
    dtype=float
)


for i in range(n):

    acc_i = ordered_accessions[i]

    path_i = tip_paths[acc_i]

    for j in range(i, n):

        acc_j = ordered_accessions[j]

        path_j = tip_paths[acc_j]

        path_j_set = set(
            path_j
        )

        mrca = None

        for ancestor in path_i:

            if ancestor in path_j_set:

                mrca = ancestor

                break

        if mrca is None:

            raise RuntimeError(
                f"Could not find MRCA for "
                f"{acc_i} and {acc_j}"
            )

        # Root-to-MRCA distance
        shared_length = 0.0

        current = mrca

        while current.parent is not None:

            shared_length += current.length

            current = current.parent

        covariance_matrix[i, j] = (
            shared_length
        )

        covariance_matrix[j, i] = (
            shared_length
        )


covariance_df = pd.DataFrame(
    covariance_matrix,
    index=ordered_accessions,
    columns=ordered_accessions
)

covariance_path = os.path.join(
    OUTPUT_DIR,
    "phylogenetic_covariance_brownian_420.csv"
)

covariance_df.to_csv(
    covariance_path
)


print()
print("✓ Phylogenetic covariance matrix calculated")

print(
    f"Dimensions: {covariance_matrix.shape}"
)


# =============================================================================
# COVARIANCE QC
# =============================================================================

print()
print("=" * 80)
print("COVARIANCE MATRIX QC")
print("=" * 80)

cov_symmetry_error = np.max(
    np.abs(
        covariance_matrix
        - covariance_matrix.T
    )
)

cov_min = covariance_matrix.min()
cov_max = covariance_matrix.max()

print()
print(
    f"Minimum covariance:     {cov_min:.10f}"
)

print(
    f"Maximum covariance:     {cov_max:.10f}"
)

print(
    f"Maximum symmetry error: "
    f"{cov_symmetry_error:.12e}"
)


if cov_symmetry_error > 1e-8:

    raise RuntimeError(
        "Phylogenetic covariance matrix "
        "is not symmetric."
    )


if cov_min < -1e-10:

    raise RuntimeError(
        "Negative covariance values detected."
    )


print()
print("✓ Covariance matrix is symmetric")
print("✓ No negative covariance values")


# =============================================================================
# EIGENVALUE QC
# =============================================================================

print()
print("=" * 80)
print("COVARIANCE MATRIX POSITIVE-SEMIDEFINITE QC")
print("=" * 80)

eigenvalues = np.linalg.eigvalsh(
    covariance_matrix
)

minimum_eigenvalue = eigenvalues.min()
maximum_eigenvalue = eigenvalues.max()

tolerance = max(
    1e-10,
    abs(maximum_eigenvalue) * 1e-10
)

negative_eigenvalues = np.sum(
    eigenvalues < -tolerance
)

near_zero_eigenvalues = np.sum(
    np.abs(eigenvalues) <= tolerance
)

print()
print(
    f"Minimum eigenvalue:       "
    f"{minimum_eigenvalue:.12e}"
)

print(
    f"Maximum eigenvalue:       "
    f"{maximum_eigenvalue:.12e}"
)

print(
    f"Negative eigenvalues:     "
    f"{negative_eigenvalues}"
)

print(
    f"Near-zero eigenvalues:    "
    f"{near_zero_eigenvalues}"
)

if negative_eigenvalues != 0:

    raise RuntimeError(
        "Phylogenetic covariance matrix "
        "is not positive semi-definite."
    )

print()
print(
    "✓ Covariance matrix is numerically "
    "positive semi-definite"
)


# =============================================================================
# CONDITIONING INFORMATION
# =============================================================================

positive_eigenvalues = eigenvalues[
    eigenvalues > tolerance
]

if len(positive_eigenvalues) > 0:

    condition_number = (
        positive_eigenvalues.max()
        /
        positive_eigenvalues.min()
    )

else:

    condition_number = np.inf


print()
print(
    f"Effective condition number: "
    f"{condition_number:.6e}"
)


# =============================================================================
# TAXON ORDER
# =============================================================================

print()
print("=" * 80)
print("SAVING PHYLOGENETIC TAXON ORDER")
print("=" * 80)

taxon_order_df = pd.DataFrame(
    {
        "Phylogenetic_Order": range(
            1,
            n + 1
        ),
        "Assembly_Accession":
            ordered_accessions,
        "Species": [
            extract_species(
                node.name
            )
            for node in tree_tips
        ],
    }
)

taxon_order_path = os.path.join(
    OUTPUT_DIR,
    "phylogenetic_taxon_order_420.csv"
)

taxon_order_df.to_csv(
    taxon_order_path,
    index=False
)

print()
print("✓ Taxon order written:")
print(taxon_order_path)


# =============================================================================
# SAVE QC SUMMARY
# =============================================================================

summary = pd.DataFrame(
    [
        {
            "Taxa": EXPECTED_TAXA,
            "Tree_Tips": len(tree_tips),
            "Shared_Accessions": len(shared),
            "Tree_Only": len(tree_only),
            "Matrix_Only": len(matrix_only),
            "Branches": len(all_branch_lengths),
            "Missing_Branch_Lengths": missing_lengths,
            "Negative_Branch_Lengths": negative_lengths,
            "Distance_Min": distance_matrix.min(),
            "Distance_Max": distance_matrix.max(),
            "Distance_Symmetry_Error": distance_symmetry_error,
            "Covariance_Min": covariance_matrix.min(),
            "Covariance_Max": covariance_matrix.max(),
            "Covariance_Symmetry_Error": cov_symmetry_error,
            "Minimum_Covariance_Eigenvalue": minimum_eigenvalue,
            "Negative_Covariance_Eigenvalues": negative_eigenvalues,
            "Status": "PASSED",
        }
    ]
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "phylogenetic_structure_qc_summary_420.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# =============================================================================
# FINAL REPORT
# =============================================================================

print()
print("=" * 80)
print("PHYLOGENETIC STRUCTURE CONSTRUCTION COMPLETE")
print("=" * 80)

print()
print("Final taxa:                  420")
print("ASTRAL tree tips:             420")
print("Shared accessions:             420")
print("Tree-only taxa:                  0")
print("Matrix-only taxa:                0")

print()
print("Patristic distance matrix:")
print(distance_path)

print()
print("Brownian covariance matrix:")
print(covariance_path)

print()
print("Taxon order:")
print(taxon_order_path)

print()
print("QC summary:")
print(summary_path)

print()
print("✓ Exact tree ↔ matrix correspondence")
print("✓ Patristic distance matrix generated")
print("✓ Phylogenetic covariance matrix generated")
print("✓ Covariance matrix passed symmetry QC")
print("✓ Covariance matrix passed PSD QC")
print("✓ Original ML matrix was NOT modified")
print("✓ Original ASTRAL tree was NOT modified")

print()
print("=" * 80)
print("READY FOR PHYLOGENY-AWARE ML MODELING")
print("=" * 80)