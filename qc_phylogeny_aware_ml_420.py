# ============================================================
# qc_phylogeny_aware_ml_420.py
#
# FINAL QC OF PHYLOGENY-AWARE 420-TAXON ML DATASET
#
# Python 3.14 compatible
# NO ete3 dependency
# ============================================================

from pathlib import Path
import sys
import re
import pandas as pd
import numpy as np


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

BUSCO_LIST = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "y1000_420_variable_busco_feature_list.csv"
)

TREE_ORDER = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "y1000_420_astral_taxon_order.csv"
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
)


# ============================================================
# EXPECTED STRUCTURE
# ============================================================

EXPECTED_TAXA = 420
EXPECTED_BUSCO = 1988
EXPECTED_PHENOTYPES = 6
EXPECTED_COLUMNS = 1997

META_COLUMNS = [
    "Species",
    "Assembly_Accession",
    "N_Strains",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Carbon_Breadth_SD",
    "Nitrogen_Breadth_SD",
    "Phenotype_Source_Species",
]

PHENOTYPE_COLUMNS = [
    "N_Strains",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Carbon_Breadth_SD",
    "Nitrogen_Breadth_SD",
]


# ============================================================
# NORMALIZATION FUNCTIONS
# ============================================================

def normalize_species(x):

    if pd.isna(x):
        return ""

    return (
        str(x)
        .strip()
        .lower()
        .replace("_", " ")
    )


def normalize_accession(x):

    if pd.isna(x):
        return ""

    return str(x).strip().upper()


# ============================================================
# NEWICK PARSER
# ============================================================

def remove_newick_comments(text):
    """
    Remove Newick square-bracket comments such as:

        [comment]

    while preserving tree structure.
    """

    return re.sub(r"\[[^\]]*\]", "", text)


def parse_newick_tips_and_branches(newick_text):
    """
    Lightweight Newick parser.

    Returns:
        tips
        terminal_branch_lengths
        internal_branch_lengths

    This parser is intentionally independent of ete3.
    """

    text = remove_newick_comments(newick_text)

    # Remove whitespace/newlines
    text = re.sub(r"\s+", "", text)

    tips = []
    terminal_branch_lengths = []
    internal_branch_lengths = []

    # --------------------------------------------------------
    # Parse terminal labels
    # --------------------------------------------------------

    # A terminal label occurs after:
    # (
    # ,
    #
    # and continues until:
    # ,
    # )
    # :
    # ;

    terminal_pattern = re.compile(
        r"(?:^|[,(])"
        r"([^(),:;]+)"
        r"(?:[:]([^(),;]+))?"
    )

    for match in terminal_pattern.finditer(text):

        label = match.group(1)

        if not label:
            continue

        label = label.strip()

        if not label:
            continue

        # Avoid accidentally treating internal structures as tips
        if label in {"(", ")", ",", ";"}:
            continue

        tips.append(label)

        branch = match.group(2)

        if branch is None or branch == "":
            terminal_branch_lengths.append(None)
        else:
            try:
                terminal_branch_lengths.append(float(branch))
            except ValueError:
                terminal_branch_lengths.append(None)

    # --------------------------------------------------------
    # Internal branch lengths
    # --------------------------------------------------------

    internal_pattern = re.compile(
        r"\)(?:[^(),:;]+)?(?::([^(),;]+))?"
    )

    for match in internal_pattern.finditer(text):

        branch = match.group(1)

        if branch is None or branch == "":
            internal_branch_lengths.append(None)
        else:
            try:
                internal_branch_lengths.append(float(branch))
            except ValueError:
                internal_branch_lengths.append(None)

    return (
        tips,
        terminal_branch_lengths,
        internal_branch_lengths,
    )


def accession_from_tree_tip(name):
    """
    Expected ASTRAL labels:

        Species_name__GCA_XXXXXXXXX.X

    Returns assembly accession.
    """

    name = str(name).strip()

    if "__" in name:
        accession = name.rsplit("__", 1)[-1]
    else:
        accession = name

    return normalize_accession(accession)


def species_from_tree_tip(name):
    """
    Extract species portion from:

        Species_name__GCA_XXXXXXXXX.X
    """

    name = str(name).strip()

    if "__" in name:
        species = name.rsplit("__", 1)[0]
    else:
        species = name

    return normalize_species(species)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("FINAL PHYLOGENY-AWARE ML DATASET QC")
print("=" * 80)

print("\nProject:")
print(PROJECT)

print("\nFinal ML matrix:")
print(ML_MATRIX)

print("\nASTRAL tree:")
print(ASTRAL_TREE)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# INPUT CHECK
# ============================================================

print("\n" + "=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)

required_files = [
    ML_MATRIX,
    BUSCO_LIST,
    TREE_ORDER,
    ASTRAL_TREE,
]

for f in required_files:

    if not f.exists():
        raise FileNotFoundError(
            f"\nMissing required file:\n{f}"
        )

    print(f"✓ Found: {f.name}")


# ============================================================
# LOAD ML MATRIX
# ============================================================

print("\n" + "=" * 80)
print("LOADING FINAL PHYLOGENY-AWARE ML MATRIX")
print("=" * 80)

df = pd.read_csv(ML_MATRIX)

print(f"\nRows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# TOTAL COLUMN QC
# ============================================================

print("\n" + "=" * 80)
print("TOTAL COLUMN QC")
print("=" * 80)

print(f"\nObserved columns: {len(df.columns)}")
print(f"Expected columns: {EXPECTED_COLUMNS}")

if len(df.columns) != EXPECTED_COLUMNS:
    raise RuntimeError(
        f"Expected {EXPECTED_COLUMNS} total columns "
        f"but found {len(df.columns)}."
    )

print("✓ Correct total column count")


# ============================================================
# COLUMN STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("COLUMN STRUCTURE QC")
print("=" * 80)

missing_meta = [
    c for c in META_COLUMNS
    if c not in df.columns
]

if missing_meta:

    print("\nMissing metadata columns:")

    for c in missing_meta:
        print(" ", c)

    raise RuntimeError(
        "Required metadata columns are missing."
    )

print("✓ Required metadata columns present")


missing_pheno = [
    c for c in PHENOTYPE_COLUMNS
    if c not in df.columns
]

if missing_pheno:

    raise RuntimeError(
        "Missing phenotype variables:\n"
        + "\n".join(missing_pheno)
    )

print("✓ Six phenotype variables present")


# ============================================================
# IDENTIFY BUSCO FEATURES
# ============================================================

print("\n" + "=" * 80)
print("IDENTIFYING BUSCO FEATURES")
print("=" * 80)

busco_features = [
    c
    for c in df.columns
    if c not in META_COLUMNS
]

print(f"\nBUSCO features detected: {len(busco_features)}")
print(f"Expected BUSCO features: {EXPECTED_BUSCO}")

if len(busco_features) != EXPECTED_BUSCO:

    print("\nLast 20 detected BUSCO columns:")

    for x in busco_features[-20:]:
        print(" ", repr(x))

    raise RuntimeError(
        f"Expected {EXPECTED_BUSCO} BUSCO features "
        f"but detected {len(busco_features)}."
    )

print("✓ Correct number of variable BUSCO features")


# ============================================================
# BUSCO ID UNIQUENESS
# ============================================================

print("\n" + "=" * 80)
print("BUSCO ID QC")
print("=" * 80)

duplicate_buscos = (
    pd.Series(busco_features)
    .duplicated()
)

duplicate_busco_count = int(
    duplicate_buscos.sum()
)

print(f"\nBUSCO features: {len(busco_features)}")
print(f"Duplicate BUSCO IDs: {duplicate_busco_count}")

if duplicate_busco_count:

    duplicated_ids = (
        pd.Series(busco_features)
        [duplicate_buscos]
        .tolist()
    )

    print("\nDuplicated IDs:")

    for x in duplicated_ids:
        print(" ", x)

    raise RuntimeError(
        "Duplicate BUSCO IDs detected."
    )

print("✓ BUSCO IDs are unique")


# ============================================================
# TAXON QC
# ============================================================

print("\n" + "=" * 80)
print("TAXON QC")
print("=" * 80)

if len(df) != EXPECTED_TAXA:

    raise RuntimeError(
        f"Expected {EXPECTED_TAXA} taxa "
        f"but found {len(df)}."
    )

species_keys = df["Species"].map(
    normalize_species
)

accession_keys = df["Assembly_Accession"].map(
    normalize_accession
)

duplicate_species = int(
    species_keys.duplicated().sum()
)

duplicate_accessions = int(
    accession_keys.duplicated().sum()
)

print(f"\nTaxa:                 {len(df)}")
print(f"Duplicate species:    {duplicate_species}")
print(f"Duplicate accessions: {duplicate_accessions}")

if duplicate_species:

    raise RuntimeError(
        "Duplicate species detected."
    )

if duplicate_accessions:

    raise RuntimeError(
        "Duplicate assembly accessions detected."
    )

print("✓ Exactly 420 unique taxa")
print("✓ Species are unique")
print("✓ Assembly accessions are unique")


# ============================================================
# PHENOTYPE QC
# ============================================================

print("\n" + "=" * 80)
print("PHENOTYPE QC")
print("=" * 80)

missing_pheno_values = (
    df[PHENOTYPE_COLUMNS]
    .isna()
    .sum()
)

print("\nMissing phenotype cells:")

for col, count in missing_pheno_values.items():

    print(
        f"  {col}: {int(count)}"
    )

if int(missing_pheno_values.sum()) != 0:

    raise RuntimeError(
        "Missing phenotype values detected."
    )

print("\n✓ No missing phenotype values")


# ============================================================
# BUSCO VALUE QC
# ============================================================

print("\n" + "=" * 80)
print("BUSCO VALUE QC")
print("=" * 80)

busco = df[busco_features]

missing_busco = int(
    busco.isna()
    .sum()
    .sum()
)

print(f"\nMissing BUSCO cells: {missing_busco}")

if missing_busco:

    raise RuntimeError(
        "BUSCO matrix contains missing values."
    )


# Convert to strings so mixed int/string
# values do not cause NumPy sorting errors.

unique_values = set(
    str(v).strip()
    for v in pd.unique(
        busco.to_numpy().ravel()
    )
)

print("\nUnique BUSCO values:")

for value in sorted(unique_values):
    print(" ", repr(value))


unexpected_values = (
    unique_values - {"0", "1"}
)

if unexpected_values:

    print("\nUnexpected BUSCO values:")

    for value in sorted(unexpected_values):
        print(" ", repr(value))

    raise RuntimeError(
        "BUSCO matrix contains values other than 0/1."
    )

print("\n✓ BUSCO matrix is binary")
print("✓ No missing BUSCO values")


# ============================================================
# BUSCO PREVALENCE
# ============================================================

print("\n" + "=" * 80)
print("BUSCO PREVALENCE QC")
print("=" * 80)

busco_numeric = busco.astype(int)

prevalence = busco_numeric.mean(
    axis=0
)

invariant = prevalence[
    (prevalence == 0)
    |
    (prevalence == 1)
]

variable = prevalence[
    (prevalence > 0)
    &
    (prevalence < 1)
]

rare_5 = prevalence[
    prevalence <= 0.05
]

rare_10 = prevalence[
    prevalence <= 0.10
]

high_95 = prevalence[
    prevalence > 0.95
]

high_99 = prevalence[
    prevalence > 0.99
]

print(
    f"\nTotal BUSCO features: {len(busco_features)}"
)

print(
    f"Invariant:            {len(invariant)}"
)

print(
    f"Variable:             {len(variable)}"
)

print(
    f"Rare <=5%:            {len(rare_5)}"
)

print(
    f"Rare <=10%:           {len(rare_10)}"
)

print(
    f"High >95%:            {len(high_95)}"
)

print(
    f"High >99%:            {len(high_99)}"
)

if len(variable) != EXPECTED_BUSCO:

    raise RuntimeError(
        "The final matrix does not contain "
        "exactly 1988 variable BUSCO features."
    )

print(
    "\n✓ All 1988 BUSCO features are variable"
)


# ============================================================
# BUSCO FEATURE LIST
# ============================================================

print("\n" + "=" * 80)
print("CHECKING BUSCO FEATURE LIST")
print("=" * 80)

feature_df = pd.read_csv(BUSCO_LIST)

print(
    f"\nFeature-list rows: {len(feature_df)}"
)


if "BUSCO_ID" in feature_df.columns:

    listed_buscos = (
        feature_df["BUSCO_ID"]
        .astype(str)
        .str.strip()
        .tolist()
    )

elif "Feature" in feature_df.columns:

    listed_buscos = (
        feature_df["Feature"]
        .astype(str)
        .str.strip()
        .tolist()
    )

else:

    listed_buscos = (
        feature_df.iloc[:, 0]
        .astype(str)
        .str.strip()
        .tolist()
    )


if len(listed_buscos) != EXPECTED_BUSCO:

    raise RuntimeError(
        f"BUSCO feature list contains "
        f"{len(listed_buscos)} IDs; "
        f"expected {EXPECTED_BUSCO}."
    )


matrix_buscos = set(
    busco_features
)

listed_busco_set = set(
    listed_buscos
)

missing_from_matrix = (
    listed_busco_set - matrix_buscos
)

missing_from_list = (
    matrix_buscos - listed_busco_set
)


if missing_from_matrix:

    print(
        "\nBUSCOs listed but absent from matrix:"
    )

    for x in sorted(missing_from_matrix):
        print(" ", x)

    raise RuntimeError(
        "BUSCO feature list does not match matrix."
    )


if missing_from_list:

    print(
        "\nBUSCOs present in matrix but absent "
        "from feature list:"
    )

    for x in sorted(missing_from_list):
        print(" ", x)

    raise RuntimeError(
        "BUSCO feature list does not match matrix."
    )


print(
    "✓ BUSCO feature list matches matrix exactly"
)


# ============================================================
# LOAD ASTRAL TREE
# ============================================================

print("\n" + "=" * 80)
print("LOADING FINAL 420-TAXON ASTRAL TREE")
print("=" * 80)

with open(
    ASTRAL_TREE,
    "r",
    encoding="utf-8"
) as handle:

    newick_text = handle.read()


if not newick_text.strip():

    raise RuntimeError(
        "ASTRAL Newick file is empty."
    )


(
    tree_tips,
    terminal_branch_lengths,
    internal_branch_lengths,
) = parse_newick_tips_and_branches(
    newick_text
)


print(
    f"\nTree tips detected: {len(tree_tips)}"
)


if len(tree_tips) != EXPECTED_TAXA:

    print("\nFirst 20 detected tips:")

    for tip in tree_tips[:20]:
        print(" ", tip)

    raise RuntimeError(
        f"Expected {EXPECTED_TAXA} tree tips "
        f"but found {len(tree_tips)}."
    )

print(
    "✓ Tree contains exactly 420 tips"
)


# ============================================================
# TREE TIP DUPLICATE QC
# ============================================================

print("\n" + "=" * 80)
print("TREE TIP DUPLICATE QC")
print("=" * 80)

tree_accessions = [
    accession_from_tree_tip(tip)
    for tip in tree_tips
]

duplicate_tree_accessions = (
    pd.Series(tree_accessions)
    .duplicated()
)

duplicate_tree_accession_count = int(
    duplicate_tree_accessions.sum()
)

print(
    f"\nTree accessions: {len(tree_accessions)}"
)

print(
    f"Duplicate tree accessions: "
    f"{duplicate_tree_accession_count}"
)

if duplicate_tree_accession_count:

    print("\nDuplicated tree accessions:")

    duplicated = (
        pd.Series(tree_accessions)
        [duplicate_tree_accessions]
        .tolist()
    )

    for x in duplicated:
        print(" ", x)

    raise RuntimeError(
        "Duplicate tree accession labels detected."
    )

print(
    "✓ Tree accession labels are unique"
)


# ============================================================
# ACCESSION RECONCILIATION
# ============================================================

print("\n" + "=" * 80)
print("MATRIX ↔ TREE ACCESSION RECONCILIATION")
print("=" * 80)

matrix_accessions = set(
    accession_keys
)

tree_accession_set = set(
    tree_accessions
)

shared_accessions = (
    matrix_accessions
    &
    tree_accession_set
)

matrix_only = (
    matrix_accessions
    -
    tree_accession_set
)

tree_only = (
    tree_accession_set
    -
    matrix_accessions
)

print(
    f"\nMatrix accessions:       "
    f"{len(matrix_accessions)}"
)

print(
    f"Tree accessions:         "
    f"{len(tree_accession_set)}"
)

print(
    f"Shared accessions:       "
    f"{len(shared_accessions)}"
)

print(
    f"Tree-only accessions:    "
    f"{len(tree_only)}"
)

print(
    f"Matrix-only accessions:  "
    f"{len(matrix_only)}"
)


if tree_only:

    print("\nTree-only:")

    for x in sorted(tree_only):
        print(" ", x)


if matrix_only:

    print("\nMatrix-only:")

    for x in sorted(matrix_only):
        print(" ", x)


if tree_only or matrix_only:

    raise RuntimeError(
        "Tree and ML matrix do not have "
        "exact accession correspondence."
    )


print(
    "\n✓ Exact accession-level correspondence confirmed"
)


# ============================================================
# SPECIES RECONCILIATION
# ============================================================

print("\n" + "=" * 80)
print("SPECIES RECONCILIATION")
print("=" * 80)

matrix_species = set(
    species_keys
)

tree_species = set(
    species_from_tree_tip(tip)
    for tip in tree_tips
)

shared_species = (
    matrix_species
    &
    tree_species
)

matrix_only_species = (
    matrix_species
    -
    tree_species
)

tree_only_species = (
    tree_species
    -
    matrix_species
)

print(
    f"\nMatrix species:       "
    f"{len(matrix_species)}"
)

print(
    f"Tree species:        "
    f"{len(tree_species)}"
)

print(
    f"Shared species:      "
    f"{len(shared_species)}"
)

print(
    f"Matrix-only species: "
    f"{len(matrix_only_species)}"
)

print(
    f"Tree-only species:   "
    f"{len(tree_only_species)}"
)


if matrix_only_species:

    print("\nMatrix-only species:")

    for x in sorted(matrix_only_species):
        print(" ", x)


if tree_only_species:

    print("\nTree-only species:")

    for x in sorted(tree_only_species):
        print(" ", x)


if matrix_only_species or tree_only_species:

    raise RuntimeError(
        "Tree and ML matrix do not have "
        "exact species correspondence."
    )


print(
    "\n✓ Exact species-level correspondence confirmed"
)


# ============================================================
# TREE ORDER FILE
# ============================================================

print("\n" + "=" * 80)
print("TREE ORDER ↔ ML MATRIX QC")
print("=" * 80)

tree_order_df = pd.read_csv(
    TREE_ORDER
)

print(
    f"\nTree-order rows: "
    f"{len(tree_order_df)}"
)

if len(tree_order_df) != EXPECTED_TAXA:

    raise RuntimeError(
        "ASTRAL taxon-order file does not "
        "contain 420 taxa."
    )


possible_accession_columns = [
    "Assembly_Accession",
    "Accession",
    "Tree_Accession",
    "accession",
]

order_accession_col = None

for col in possible_accession_columns:

    if col in tree_order_df.columns:

        order_accession_col = col
        break


if order_accession_col is None:

    raise RuntimeError(
        "Could not identify accession column "
        "in tree-order file."
    )


order_accessions = (
    tree_order_df[
        order_accession_col
    ]
    .map(normalize_accession)
    .tolist()
)


if set(order_accessions) != matrix_accessions:

    raise RuntimeError(
        "Tree-order file does not contain "
        "exactly the same 420 accessions "
        "as the ML matrix."
    )


print(
    "✓ Tree-order file contains the same 420 taxa"
)


# ============================================================
# TREE ORDER CHECK
# ============================================================

matrix_order = (
    accession_keys
    .tolist()
)

if matrix_order == order_accessions:

    print(
        "✓ ML matrix is already in ASTRAL tree order"
    )

else:

    print(
        "⚠ ML matrix row order differs from "
        "the stored ASTRAL taxon order."
    )

    print(
        "  This is acceptable if the final "
        "matrix was intentionally reordered."
    )


# ============================================================
# BRANCH LENGTH QC
# ============================================================

print("\n" + "=" * 80)
print("BRANCH LENGTH QC")
print("=" * 80)

missing_branch_lengths = sum(
    x is None
    for x in terminal_branch_lengths
)

negative_branch_lengths = sum(
    x is not None and x < 0
    for x in terminal_branch_lengths
)

print(
    f"\nTerminal branches:       "
    f"{len(terminal_branch_lengths)}"
)

print(
    f"Missing branch lengths:  "
    f"{missing_branch_lengths}"
)

print(
    f"Negative branch lengths: "
    f"{negative_branch_lengths}"
)


if missing_branch_lengths:

    raise RuntimeError(
        "Missing terminal branch lengths detected."
    )


if negative_branch_lengths:

    raise RuntimeError(
        "Negative terminal branch lengths detected."
    )


print(
    "\n✓ All terminal branches have valid lengths"
)


# ============================================================
# TREE STRUCTURE QC
# ============================================================

print("\n" + "=" * 80)
print("TREE STRUCTURE QC")
print("=" * 80)

# For a rooted, fully bifurcating 420-tip tree:
# internal nodes = 420 - 1 = 419
#
# We count closing parentheses as internal nodes.

internal_nodes = newick_text.count(")")

print(
    f"\nTerminal nodes: {len(tree_tips)}"
)

print(
    f"Internal nodes detected: {internal_nodes}"
)


if internal_nodes != EXPECTED_TAXA - 1:

    print(
        "\n⚠ Internal-node count is not "
        "419."
    )

    print(
        "  This may indicate a multifurcating "
        "or non-standard Newick structure."
    )

else:

    print(
        "✓ 420-tip rooted bifurcating structure detected"
    )


# ============================================================
# PARETO CHECK
# ============================================================

print("\n" + "=" * 80)
print("PARETO CANDIDATE CHECK")
print("=" * 80)

pareto_file = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "pareto_qc"
    / "pareto_420_species_level_reconciliation.csv"
)

pareto_present_count = None
pareto_absent_count = None

if pareto_file.exists():

    pareto_df = pd.read_csv(
        pareto_file
    )

    if "Pareto_Assembly_Accession" in pareto_df.columns:

        pareto_accessions = (
            pareto_df[
                "Pareto_Assembly_Accession"
            ]
            .map(normalize_accession)
            .dropna()
        )

        pareto_present = (
            pareto_accessions
            .isin(matrix_accessions)
        )

        pareto_present_count = int(
            pareto_present.sum()
        )

        pareto_absent_count = int(
            (~pareto_present).sum()
        )

        print(
            f"\nPareto candidates: "
            f"{len(pareto_accessions)}"
        )

        print(
            f"Present in final 420 ML dataset: "
            f"{pareto_present_count}"
        )

        print(
            f"Absent from final 420 ML dataset: "
            f"{pareto_absent_count}"
        )

        if pareto_absent_count == 0:

            print(
                "✓ All Pareto assembly accessions "
                "are represented"
            )

        else:

            print(
                "⚠ Some Pareto accessions are not "
                "direct assembly-level members "
                "of the final matrix."
            )

else:

    print(
        "\n⚠ Pareto reconciliation file not found."
    )

    print(
        "Skipping Pareto QC."
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "Metric": [
            "ML_taxa",
            "ML_columns",
            "BUSCO_features",
            "Phenotype_variables",
            "Missing_BUSCO_cells",
            "Missing_phenotype_cells",
            "Unique_species",
            "Unique_accessions",
            "ASTRAL_tree_tips",
            "Shared_accessions",
            "Tree_only_accessions",
            "Matrix_only_accessions",
            "Shared_species",
            "Tree_only_species",
            "Matrix_only_species",
            "Missing_terminal_branch_lengths",
            "Negative_terminal_branch_lengths",
            "Pareto_candidates",
            "Pareto_present",
            "Pareto_absent",
        ],

        "Value": [
            len(df),
            len(df.columns),
            len(busco_features),
            len(PHENOTYPE_COLUMNS),
            missing_busco,
            int(missing_pheno_values.sum()),
            len(matrix_species),
            len(matrix_accessions),
            len(tree_tips),
            len(shared_accessions),
            len(tree_only),
            len(matrix_only),
            len(shared_species),
            len(tree_only_species),
            len(matrix_only_species),
            missing_branch_lengths,
            negative_branch_lengths,
            (
                len(pareto_accessions)
                if pareto_present_count is not None
                else np.nan
            ),
            (
                pareto_present_count
                if pareto_present_count is not None
                else np.nan
            ),
            (
                pareto_absent_count
                if pareto_absent_count is not None
                else np.nan
            ),
        ],
    }
)


summary_file = (
    OUTPUT_DIR
    / "final_phylogeny_aware_ml_qc_summary_420.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL PHYLOGENY-AWARE ML VALIDATION")
print("=" * 80)

checks = {

    "ML_rows_420":
        len(df) == 420,

    "ML_columns_1997":
        len(df.columns) == 1997,

    "BUSCO_features_1988":
        len(busco_features) == 1988,

    "Phenotype_variables_6":
        len(PHENOTYPE_COLUMNS) == 6,

    "No_missing_BUSCO":
        missing_busco == 0,

    "No_missing_phenotype":
        int(missing_pheno_values.sum()) == 0,

    "Unique_species":
        duplicate_species == 0,

    "Unique_accessions":
        duplicate_accessions == 0,

    "Tree_tips_420":
        len(tree_tips) == 420,

    "Shared_accessions_420":
        len(shared_accessions) == 420,

    "Tree_only_0":
        len(tree_only) == 0,

    "Matrix_only_0":
        len(matrix_only) == 0,

    "Shared_species_420":
        len(shared_species) == 420,

    "Valid_terminal_branches":
        (
            missing_branch_lengths == 0
            and
            negative_branch_lengths == 0
        ),
}


print()

failed = []

for name, passed in checks.items():

    if passed:

        print(
            f"✓ {name}"
        )

    else:

        print(
            f"✗ {name}"
        )

        failed.append(name)


# ============================================================
# FAILURE
# ============================================================

if failed:

    print("\n" + "=" * 80)
    print("FINAL QC FAILED")
    print("=" * 80)

    print("\nFailed checks:")

    for item in failed:
        print(
            " -",
            item
        )

    raise RuntimeError(
        "Final phylogeny-aware ML QC failed."
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("✓ FINAL PHYLOGENY-AWARE ML QC PASSED")
print("=" * 80)

print("\nFinal dataset:")

print(
    f"  Taxa:                  {len(df)}"
)

print(
    f"  Total columns:         {len(df.columns)}"
)

print(
    f"  Variable BUSCOs:       {len(busco_features)}"
)

print(
    f"  Phenotype variables:   {len(PHENOTYPE_COLUMNS)}"
)

print("\nPhylogeny:")

print(
    f"  ASTRAL tips:            {len(tree_tips)}"
)

print(
    f"  Shared accessions:      "
    f"{len(shared_accessions)}"
)

print(
    f"  Tree-only:              {len(tree_only)}"
)

print(
    f"  Matrix-only:            {len(matrix_only)}"
)

print("\nQC summary:")

print(
    summary_file
)

print(
    "\n✓ Original phylogeny-aware ML matrix "
    "was NOT modified."
)

print(
    "✓ Original ASTRAL tree was NOT modified."
)

print(
    "✓ BUSCO feature list was NOT modified."
)

print(
    "✓ Final 420-taxon phylogeny-aware "
    "ML dataset passed QC."
)

print("\n" + "=" * 80)
print("READY FOR PHYLOGENY-AWARE ML MODELING")
print("=" * 80)