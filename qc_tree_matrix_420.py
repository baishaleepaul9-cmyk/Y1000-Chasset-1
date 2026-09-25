# ============================================================
# FINAL TREE ↔ 420-TAXON ML MATRIX CONSISTENCY QC
# ============================================================
#
# Purpose:
#   Verify that the final 420-taxon ML matrix and the final
#   420-taxon ASTRAL tree contain exactly the same taxa.
#
# Inputs:
#   - y1000_420_variable_busco_ml_matrix.csv
#   - ASTRAL_420_taxon_pruned.nwk
#
# Outputs:
#   - tree_matrix_taxon_mapping_420.csv
#   - tree_only_taxa_420.csv
#   - matrix_only_taxa_420.csv
#   - tree_matrix_qc_summary_420.csv
#
# Original files are NOT modified.
# ============================================================

from pathlib import Path
import pandas as pd
import re
import sys

try:
    from Bio import Phylo
except ImportError:
    print("ERROR: Biopython is required.")
    print("Install using:")
    print("    pip install biopython")
    sys.exit(1)


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    r"C:\Y1000_chassis_project"
)

ML_FILE = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "feature_qc"
    / "y1000_420_variable_busco_ml_matrix.csv"
)

TREE_FILE = (
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
    / "phylogeny_qc"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUTS
# ============================================================

MAPPING_FILE = (
    OUTPUT_DIR
    / "tree_matrix_taxon_mapping_420.csv"
)

TREE_ONLY_FILE = (
    OUTPUT_DIR
    / "tree_only_taxa_420.csv"
)

MATRIX_ONLY_FILE = (
    OUTPUT_DIR
    / "matrix_only_taxa_420.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "tree_matrix_qc_summary_420.csv"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_accession(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    match = re.search(
        r"(GC[AF]_\d+\.\d+)",
        value,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return value.upper()


def normalize_species(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.lower()


def clean_tree_label(value):

    if value is None:
        return ""

    value = str(value).strip()

    value = value.strip(
        "'\""
    )

    return value


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("FINAL TREE ↔ 420-TAXON ML MATRIX CONSISTENCY QC")
print("=" * 80)

print()
print("ML matrix:")
print(ML_FILE)

print()
print("ASTRAL tree:")
print(TREE_FILE)

print()
print("QC output:")
print(OUTPUT_DIR)


# ============================================================
# INPUT CHECK
# ============================================================

print()
print("=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)

if not ML_FILE.exists():

    raise FileNotFoundError(
        f"ML matrix not found:\n{ML_FILE}"
    )

if not TREE_FILE.exists():

    raise FileNotFoundError(
        f"420-taxon ASTRAL tree not found:\n{TREE_FILE}"
    )

print()
print("✓ ML matrix found")
print("✓ 420-taxon ASTRAL tree found")


# ============================================================
# LOAD ML MATRIX
# ============================================================

print()
print("=" * 80)
print("LOADING FINAL ML MATRIX")
print("=" * 80)

df = pd.read_csv(
    ML_FILE
)

print()
print(
    f"Rows:    {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)

if len(df) != 420:

    raise RuntimeError(
        f"Expected 420 ML taxa, "
        f"found {len(df)}."
    )

required = [
    "Species",
    "Assembly_Accession"
]

missing = [
    x
    for x in required
    if x not in df.columns
]

if missing:

    raise RuntimeError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print()
print("✓ Exactly 420 ML rows")
print("✓ Species column present")
print("✓ Assembly_Accession column present")


# ============================================================
# MATRIX IDENTIFIERS
# ============================================================

df["_species_key"] = (
    df["Species"]
    .apply(normalize_species)
)

df["_accession_key"] = (
    df["Assembly_Accession"]
    .apply(normalize_accession)
)


# ============================================================
# MATRIX DUPLICATE QC
# ============================================================

print()
print("=" * 80)
print("ML MATRIX DUPLICATE QC")
print("=" * 80)

duplicate_species = (
    df["_species_key"]
    .duplicated(
        keep=False
    )
)

duplicate_accessions = (
    df["_accession_key"]
    .duplicated(
        keep=False
    )

)

print()
print(
    "Duplicate species:",
    int(duplicate_species.sum())
)

print(
    "Duplicate accessions:",
    int(duplicate_accessions.sum())
)

if duplicate_species.any():

    print()
    print(
        df.loc[
            duplicate_species,
            [
                "Species",
                "Assembly_Accession"
            ]
        ].to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Duplicate species detected."
    )

if duplicate_accessions.any():

    print()
    print(
        df.loc[
            duplicate_accessions,
            [
                "Species",
                "Assembly_Accession"
            ]
        ].to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Duplicate assembly accessions detected."
    )

print()
print(
    "✓ Species are unique"
)

print(
    "✓ Assembly accessions are unique"
)


# ============================================================
# LOAD TREE
# ============================================================

print()
print("=" * 80)
print("LOADING 420-TAXON ASTRAL TREE")
print("=" * 80)

tree = Phylo.read(
    str(TREE_FILE),
    "newick"
)

terminals = tree.get_terminals()

tree_labels = [

    clean_tree_label(
        terminal.name
    )

    for terminal in terminals

    if terminal.name
]

print()
print(
    "Tree tips:",
    len(tree_labels)
)

if len(tree_labels) != 420:

    raise RuntimeError(
        f"Expected 420 tree tips, "
        f"found {len(tree_labels)}."
    )

print()
print("✓ Tree contains exactly 420 tips")


# ============================================================
# TREE TIP DUPLICATE QC
# ============================================================

print()
print("=" * 80)
print("TREE TIP DUPLICATE QC")
print("=" * 80)

tree_accessions = []

for label in tree_labels:

    accession = normalize_accession(
        label
    )

    tree_accessions.append(
        accession
    )

duplicate_tree_accessions = (
    pd.Series(
        tree_accessions
    ).duplicated(
        keep=False
    )
)

print()
print(
    "Duplicate tree accessions:",
    int(
        duplicate_tree_accessions.sum()
    )
)

if duplicate_tree_accessions.any():

    duplicated = sorted(
        set(
            accession
            for accession, duplicate
            in zip(
                tree_accessions,
                duplicate_tree_accessions
            )
            if duplicate
        )
    )

    print()
    print(
        "Duplicated tree accessions:"
    )

    for accession in duplicated:

        print(
            " ",
            accession
        )

    raise RuntimeError(
        "Duplicate accessions found in tree."
    )

print()
print(
    "✓ Tree accession labels are unique"
)


# ============================================================
# ACCESSION SET COMPARISON
# ============================================================

print()
print("=" * 80)
print("ACCESSION-LEVEL RECONCILIATION")
print("=" * 80)

matrix_accessions = set(
    df["_accession_key"]
)

tree_accession_set = set(
    tree_accessions
)

tree_only_accessions = (
    tree_accession_set
    - matrix_accessions
)

matrix_only_accessions = (
    matrix_accessions
    - tree_accession_set
)

shared_accessions = (
    matrix_accessions
    & tree_accession_set
)

print()
print(
    "Matrix accessions:",
    len(matrix_accessions)
)

print(
    "Tree accessions:",
    len(tree_accession_set)
)

print(
    "Shared accessions:",
    len(shared_accessions)
)

print(
    "Tree-only accessions:",
    len(tree_only_accessions)
)

print(
    "Matrix-only accessions:",
    len(matrix_only_accessions)
)


# ============================================================
# SPECIES-LEVEL COMPARISON
# ============================================================

print()
print("=" * 80)
print("SPECIES-LEVEL RECONCILIATION")
print("=" * 80)

matrix_species = set(
    df["_species_key"]
)

tree_species = set()

for label in tree_labels:

    # Remove accession suffix when possible.
    accession_match = re.search(
        r"__GC[AF]_\d+\.\d+$",
        label,
        flags=re.IGNORECASE
    )

    if accession_match:

        species_part = (
            label[
                :accession_match.start()
            ]
        )

    else:

        species_part = label

    species_part = (
        species_part
        .replace("_", " ")
        .strip()
    )

    tree_species.add(
        normalize_species(
            species_part
        )
    )

shared_species = (
    matrix_species
    & tree_species
)

tree_only_species = (
    tree_species
    - matrix_species
)

matrix_only_species = (
    matrix_species
    - tree_species
)

print()
print(
    "Matrix species:",
    len(matrix_species)
)

print(
    "Tree species:",
    len(tree_species)
)

print(
    "Shared species:",
    len(shared_species)
)

print(
    "Tree-only species:",
    len(tree_only_species)
)

print(
    "Matrix-only species:",
    len(matrix_only_species)
)


# ============================================================
# BUILD TAXON MAPPING TABLE
# ============================================================

print()
print("=" * 80)
print("CREATING TREE ↔ MATRIX TAXON MAPPING")
print("=" * 80)

matrix_by_accession = {}

for _, row in df.iterrows():

    matrix_by_accession[
        row["_accession_key"]
    ] = row

mapping_records = []

for label in tree_labels:

    accession = normalize_accession(
        label
    )

    if accession in matrix_by_accession:

        row = matrix_by_accession[
            accession
        ]

        mapping_records.append({

            "Tree_Tip":
                label,

            "Tree_Assembly_Accession":
                accession,

            "Matrix_Assembly_Accession":
                row[
                    "Assembly_Accession"
                ],

            "Matrix_Species":
                row["Species"],

            "Match_Status":
                "EXACT_ACCESSION_MATCH"

        })

    else:

        mapping_records.append({

            "Tree_Tip":
                label,

            "Tree_Assembly_Accession":
                accession,

            "Matrix_Assembly_Accession":
                "",

            "Matrix_Species":
                "",

            "Match_Status":
                "TREE_ONLY"

        })


mapping_df = pd.DataFrame(
    mapping_records
)

mapping_df.to_csv(
    MAPPING_FILE,
    index=False
)

print()
print(
    "✓ Taxon mapping written:"
)

print(
    MAPPING_FILE
)


# ============================================================
# TREE-ONLY TABLE
# ============================================================

tree_only_records = []

for accession in sorted(
    tree_only_accessions
):

    labels = [

        label

        for label in tree_labels

        if normalize_accession(
            label
        ) == accession
    ]

    for label in labels:

        tree_only_records.append({

            "Tree_Tip":
                label,

            "Assembly_Accession":
                accession
        })


tree_only_df = pd.DataFrame(
    tree_only_records
)

tree_only_df.to_csv(
    TREE_ONLY_FILE,
    index=False
)


# ============================================================
# MATRIX-ONLY TABLE
# ============================================================

matrix_only_df = df[
    df["_accession_key"].isin(
        matrix_only_accessions
    )
][
    [
        "Species",
        "Assembly_Accession"
    ]
].copy()

matrix_only_df.to_csv(
    MATRIX_ONLY_FILE,
    index=False
)


# ============================================================
# BRANCH LENGTH QC
# ============================================================

print()
print("=" * 80)
print("BRANCH LENGTH QC")
print("=" * 80)

terminal_branch_lengths = []

for terminal in terminals:

    branch_length = (
        terminal.branch_length
    )

    terminal_branch_lengths.append(
        branch_length
    )

missing_terminal_lengths = sum(
    x is None
    for x in terminal_branch_lengths
)

negative_terminal_lengths = sum(
    (
        x is not None
        and x < 0
    )
    for x in terminal_branch_lengths
)

print()
print(
    "Terminal branches:",
    len(terminal_branch_lengths)
)

print(
    "Missing branch lengths:",
    missing_terminal_lengths
)

print(
    "Negative branch lengths:",
    negative_terminal_lengths
)

if negative_terminal_lengths > 0:

    raise RuntimeError(
        "Negative branch lengths detected."
    )

if missing_terminal_lengths == 0:

    print()
    print(
        "✓ All terminal branches have lengths"
    )

else:

    print()
    print(
        "⚠ Some terminal branches have "
        "missing lengths."
    )


# ============================================================
# PHYLOGENETIC TREE STRUCTURE QC
# ============================================================

print()
print("=" * 80)
print("TREE STRUCTURE QC")
print("=" * 80)

total_clades = len(
    tree.get_nonterminals()
)

total_terminals = len(
    tree.get_terminals()
)

print()
print(
    "Terminal nodes:",
    total_terminals
)

print(
    "Internal nodes:",
    total_clades
)

if total_terminals != 420:

    raise RuntimeError(
        "Tree terminal count is not 420."
    )

print()
print(
    "✓ Tree structure loaded successfully"
)


# ============================================================
# FINAL CONSISTENCY TESTS
# ============================================================

print()
print("=" * 80)
print("FINAL CONSISTENCY TESTS")
print("=" * 80)

tests = {

    "ML_rows_420":
        len(df) == 420,

    "Tree_tips_420":
        len(tree_labels) == 420,

    "Unique_ML_species":
        len(matrix_species) == 420,

    "Unique_ML_accessions":
        len(matrix_accessions) == 420,

    "Unique_tree_accessions":
        len(tree_accession_set) == 420,

    "Shared_accessions_420":
        len(shared_accessions) == 420,

    "Tree_only_0":
        len(tree_only_accessions) == 0,

    "Matrix_only_0":
        len(matrix_only_accessions) == 0,

    "Shared_species_420":
        len(shared_species) == 420,

    "Duplicate_tree_accessions_0":
        int(
            duplicate_tree_accessions.sum()
        ) == 0,

    "Negative_branch_lengths_0":
        negative_terminal_lengths == 0
}


for name, passed in tests.items():

    if passed:

        print(
            f"✓ {name}"
        )

    else:

        print(
            f"✗ {name}"
        )


all_passed = all(
    tests.values()
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "ML_Taxa":
        len(df),

    "Tree_Tips":
        len(tree_labels),

    "Shared_Accessions":
        len(shared_accessions),

    "Tree_Only_Accessions":
        len(tree_only_accessions),

    "Matrix_Only_Accessions":
        len(matrix_only_accessions),

    "Shared_Species":
        len(shared_species),

    "Duplicate_ML_Species":
        int(
            duplicate_species.sum()
        ),

    "Duplicate_ML_Accessions":
        int(
            duplicate_accessions.sum()
        ),

    "Duplicate_Tree_Accessions":
        int(
            duplicate_tree_accessions.sum()
        ),

    "Missing_Terminal_Branch_Lengths":
        missing_terminal_lengths,

    "Negative_Terminal_Branch_Lengths":
        negative_terminal_lengths,

    "Final_QC_Passed":
        all_passed

}])


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("TREE ↔ 420 ML MATRIX QC COMPLETE")
print("=" * 80)

print()
print(
    f"ML taxa:                    {len(df)}"
)

print(
    f"Tree tips:                  {len(tree_labels)}"
)

print(
    f"Shared accessions:          {len(shared_accessions)}"
)

print(
    f"Tree-only accessions:       {len(tree_only_accessions)}"
)

print(
    f"Matrix-only accessions:     {len(matrix_only_accessions)}"
)

print(
    f"Shared species:             {len(shared_species)}"
)

print(
    f"Missing branch lengths:     {missing_terminal_lengths}"
)

print(
    f"Negative branch lengths:    {negative_terminal_lengths}"
)

print()
print("Mapping:")
print(MAPPING_FILE)

print()
print("Tree-only:")
print(TREE_ONLY_FILE)

print()
print("Matrix-only:")
print(MATRIX_ONLY_FILE)

print()
print("Summary:")
print(SUMMARY_FILE)


if all_passed:

    print()
    print("=" * 80)
    print("✓ FINAL PHYLOGENY ↔ ML QC PASSED")
    print("=" * 80)

    print()
    print(
        "The 420 ML taxa and 420-taxon ASTRAL tree "
        "have exact accession-level correspondence."
    )

    print()
    print(
        "The dataset is ready for the next "
        "phylogeny-aware ML preparation step."
    )

else:

    print()
    print("=" * 80)
    print("⚠ FINAL PHYLOGENY ↔ ML QC FAILED")
    print("=" * 80)

    print()
    print(
        "Do NOT proceed to phylogeny-aware ML "
        "until the failed checks are investigated."
    )

    raise RuntimeError(
        "Final tree ↔ ML consistency QC failed."
    )