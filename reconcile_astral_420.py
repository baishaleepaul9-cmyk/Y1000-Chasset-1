# ============================================================
# ASTRAL ↔ FINAL 420-TAXON RECONCILIATION
# ============================================================
#
# Purpose:
#   Reconcile the final 420-taxon ML dataset with the existing
#   ASTRAL species tree before phylogeny-aware ML.
#
# Inputs:
#   1. Final 420 variable-BUSCO ML matrix
#   2. Existing ASTRAL species tree
#
# Outputs:
#   - ASTRAL/tree membership table
#   - matched taxa table
#   - tree-only taxa
#   - ML-only taxa
#   - 420-taxon pruned ASTRAL tree
#   - reconciliation summary
#
# IMPORTANT:
#   Original ASTRAL tree is NEVER modified.
# ============================================================

from pathlib import Path
import pandas as pd
import re
import sys
import copy

try:
    from Bio import Phylo
except ImportError:
    print("ERROR: Biopython is required.")
    print("Install with:")
    print("    pip install biopython")
    sys.exit(1)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

ML_FILE = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "feature_qc"
    / "y1000_420_variable_busco_ml_matrix.csv"
)

ASTRAL_DIR = (
    PROJECT
    / "results"
    / "stage4B_phylogeny"
    / "postbusco_phylogeny_v2"
    / "astral"
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
# OUTPUT FILES
# ============================================================

MEMBERSHIP_FILE = (
    OUTPUT_DIR
    / "astral_420_membership.csv"
)

MATCHED_FILE = (
    OUTPUT_DIR
    / "astral_420_matched_taxa.csv"
)

TREE_ONLY_FILE = (
    OUTPUT_DIR
    / "astral_tree_only_taxa.csv"
)

ML_ONLY_FILE = (
    OUTPUT_DIR
    / "ml_420_only_taxa.csv"
)

PRUNED_TREE_FILE = (
    OUTPUT_DIR
    / "ASTRAL_420_taxon_pruned.nwk"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "astral_420_reconciliation_summary.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def norm(value):
    """
    Normalize species names for comparison.

    Handles:
      - leading/trailing spaces
      - repeated spaces
      - case differences
    """

    if pd.isna(value):
        return ""

    value = str(value).strip()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.lower()


def normalize_accession(value):
    """
    Normalize assembly accession.
    """

    if pd.isna(value):
        return ""

    value = str(value).strip()

    value = re.sub(
        r"\s+",
        "",
        value
    )

    return value.upper()


def clean_tree_label(label):
    """
    Clean common Newick tree-tip formatting.
    """

    if label is None:
        return ""

    label = str(label).strip()

    label = label.strip(
        "'\""
    )

    return label


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("ASTRAL ↔ FINAL 420-TAXON RECONCILIATION")
print("=" * 80)

print()
print("Project:")
print(PROJECT)

print()
print("ML dataset:")
print(ML_FILE)

print()
print("ASTRAL directory:")
print(ASTRAL_DIR)

print()
print("Output:")
print(OUTPUT_DIR)


# ============================================================
# CHECK INPUTS
# ============================================================

print()
print("=" * 80)
print("CHECKING INPUTS")
print("=" * 80)

if not ML_FILE.exists():

    raise FileNotFoundError(
        f"Final 420 ML dataset not found:\n{ML_FILE}"
    )

if not ASTRAL_DIR.exists():

    raise FileNotFoundError(
        f"ASTRAL directory not found:\n{ASTRAL_DIR}"
    )

print()
print("✓ Final ML matrix found")
print("✓ ASTRAL directory found")


# ============================================================
# LOAD FINAL 420 ML DATASET
# ============================================================

print()
print("=" * 80)
print("LOADING FINAL 420 ML DATASET")
print("=" * 80)

df = pd.read_csv(
    ML_FILE
)

print()
print(f"Rows:    {len(df)}")
print(f"Columns: {len(df.columns)}")

if len(df) != 420:

    raise RuntimeError(
        f"Expected exactly 420 taxa, "
        f"but found {len(df)}."
    )

required_columns = [
    "Species",
    "Assembly_Accession"
]

missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing_columns:

    raise RuntimeError(
        "Required columns missing from ML dataset:\n"
        + "\n".join(missing_columns)
    )

print()
print("✓ Species column present")
print("✓ Assembly_Accession column present")


# ============================================================
# CREATE NORMALIZED IDENTIFIERS
# ============================================================

df["_species_key"] = (
    df["Species"].apply(norm)
)

df["_accession_key"] = (
    df["Assembly_Accession"]
    .apply(normalize_accession)
)


# ============================================================
# ML DATASET UNIQUENESS QC
# ============================================================

print()
print("=" * 80)
print("ML DATASET TAXON QC")
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
    f"Taxa:                   {len(df)}"
)

print(
    f"Duplicate species:      "
    f"{int(duplicate_species.sum())}"
)

print(
    f"Duplicate accessions:   "
    f"{int(duplicate_accessions.sum())}"
)


if duplicate_species.any():

    print()
    print("Duplicated species:")

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
        "Species are not unique in the final "
        "420-taxon ML dataset."
    )


if duplicate_accessions.any():

    print()
    print("Duplicated accessions:")

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
        "Assembly accessions are not unique "
        "in the final 420-taxon ML dataset."
    )


print()
print(
    "✓ Exactly 420 unique ML taxa"
)

print(
    "✓ Species are unique"
)

print(
    "✓ Assembly accessions are unique"
)


# ============================================================
# LOCATE ASTRAL TREE
# ============================================================

print()
print("=" * 80)
print("LOCATING ASTRAL SPECIES TREE")
print("=" * 80)

tree_candidates = sorted(
    [
        p
        for p in ASTRAL_DIR.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in [
                ".nwk",
                ".newick",
                ".tree"
            ]
        )
    ]
)

if not tree_candidates:

    raise FileNotFoundError(
        "No Newick/tree file found inside:\n"
        f"{ASTRAL_DIR}"
    )

print()
print("Candidate tree files:")

for i, path in enumerate(
    tree_candidates,
    start=1
):

    print(
        f"  {i}. {path}"
    )


# ============================================================
# SELECT ASTRAL TREE
# ============================================================

preferred = []

for path in tree_candidates:

    name = path.name.lower()

    score = 0

    if "astral" in name:
        score += 10

    if "species" in name:
        score += 10

    if "tree" in name:
        score += 5

    if "midpoint" in name:
        score += 2

    if "gene" in name:
        score -= 20

    preferred.append(
        (
            score,
            path
        )
    )


preferred.sort(
    key=lambda x: (
        -x[0],
        str(x[1])
    )
)

ASTRAL_TREE = preferred[0][1]

print()
print("Selected ASTRAL tree:")
print(ASTRAL_TREE)


# ============================================================
# LOAD ASTRAL TREE
# ============================================================

print()
print("=" * 80)
print("LOADING ASTRAL TREE")
print("=" * 80)

try:

    tree = Phylo.read(
        str(ASTRAL_TREE),
        "newick"
    )

except Exception as e:

    raise RuntimeError(
        f"Could not read ASTRAL tree:\n"
        f"{ASTRAL_TREE}\n\n"
        f"Error: {e}"
    )


tree_tips = [
    clean_tree_label(
        clade.name
    )
    for clade in tree.get_terminals()
]

tree_tips = [
    tip
    for tip in tree_tips
    if tip
]

print()
print(
    f"ASTRAL tree tips: {len(tree_tips)}"
)

if len(tree_tips) == 0:

    raise RuntimeError(
        "ASTRAL tree contains no recognizable "
        "terminal labels."
    )

print()
print("✓ ASTRAL tree loaded")


# ============================================================
# TREE TIP INSPECTION
# ============================================================

print()
print("=" * 80)
print("TREE TIP LABEL INSPECTION")
print("=" * 80)

print()
print("First 20 ASTRAL tips:")

for tip in tree_tips[:20]:

    print(
        f"  {tip}"
    )


# ============================================================
# BUILD ML LOOKUPS
# ============================================================

species_lookup = {}

for _, row in df.iterrows():

    species_lookup[
        norm(row["Species"])
    ] = row


accession_lookup = {}

for _, row in df.iterrows():

    accession_lookup[
        normalize_accession(
            row["Assembly_Accession"]
        )
    ] = row


# ============================================================
# RECONCILE TREE TIPS
# ============================================================

print()
print("=" * 80)
print("RECONCILING ASTRAL TIPS AGAINST FINAL 420")
print("=" * 80)

records = []

matched_ml_accessions = set()

for tip in tree_tips:

    tip_clean = clean_tree_label(
        tip
    )

    tip_species_key = norm(
        tip_clean
    )

    tip_accession_key = normalize_accession(
        tip_clean
    )

    matched_row = None
    match_basis = None

    # --------------------------------------------------------
    # EXACT ASSEMBLY ACCESSION
    # --------------------------------------------------------

    if tip_accession_key in accession_lookup:

        matched_row = accession_lookup[
            tip_accession_key
        ]

        match_basis = (
            "EXACT_ASSEMBLY_ACCESSION"
        )

    # --------------------------------------------------------
    # EXACT SPECIES NAME
    # --------------------------------------------------------

    elif tip_species_key in species_lookup:

        matched_row = species_lookup[
            tip_species_key
        ]

        match_basis = (
            "EXACT_SPECIES"
        )

    # --------------------------------------------------------
    # ACCESSION EMBEDDED INSIDE TREE LABEL
    # --------------------------------------------------------

    else:

        accession_match = re.search(
            r"GCA_\d+\.\d+|GCF_\d+\.\d+",
            tip_clean,
            flags=re.IGNORECASE
        )

        if accession_match:

            extracted_accession = (
                accession_match
                .group(0)
                .upper()
            )

            if (
                extracted_accession
                in accession_lookup
            ):

                matched_row = (
                    accession_lookup[
                        extracted_accession
                    ]
                )

                match_basis = (
                    "EMBEDDED_ASSEMBLY_ACCESSION"
                )

    # --------------------------------------------------------
    # RECORD RESULT
    # --------------------------------------------------------

    if matched_row is not None:

        matched_species = (
            matched_row["Species"]
        )

        matched_accession = (
            matched_row[
                "Assembly_Accession"
            ]
        )

        matched_ml_accessions.add(
            normalize_accession(
                matched_accession
            )
        )

        records.append({

            "Tree_Tip":
                tip_clean,

            "Status":
                "MATCHED",

            "Matched_Species":
                matched_species,

            "Matched_Assembly_Accession":
                matched_accession,

            "Match_Basis":
                match_basis
        })

    else:

        records.append({

            "Tree_Tip":
                tip_clean,

            "Status":
                "TREE_ONLY",

            "Matched_Species":
                "",

            "Matched_Assembly_Accession":
                "",

            "Match_Basis":
                "NO_MATCH"
        })


membership_df = pd.DataFrame(
    records
)


# ============================================================
# MATCHED TAXA
# ============================================================

matched_df = membership_df[
    membership_df["Status"]
    == "MATCHED"
].copy()


# ============================================================
# TREE-ONLY TAXA
# ============================================================

tree_only_df = membership_df[
    membership_df["Status"]
    == "TREE_ONLY"
].copy()


# ============================================================
# ML-ONLY TAXA
# ============================================================

ml_only_rows = []

for _, row in df.iterrows():

    accession_key = normalize_accession(
        row["Assembly_Accession"]
    )

    if accession_key not in matched_ml_accessions:

        ml_only_rows.append({

            "Species":
                row["Species"],

            "Assembly_Accession":
                row["Assembly_Accession"]
        })


ml_only_df = pd.DataFrame(
    ml_only_rows
)


# ============================================================
# RECONCILIATION SUMMARY
# ============================================================

print()
print("=" * 80)
print("ASTRAL ↔ 420 RECONCILIATION SUMMARY")
print("=" * 80)

print()
print(
    f"Final ML taxa:        {len(df)}"
)

print(
    f"ASTRAL tree tips:     {len(tree_tips)}"
)

print(
    f"Matched tree tips:    {len(matched_df)}"
)

print(
    f"Tree-only taxa:       {len(tree_only_df)}"
)

print(
    f"ML-only taxa:         {len(ml_only_df)}"
)


# ============================================================
# SAVE RECONCILIATION TABLES
# ============================================================

membership_df.to_csv(
    MEMBERSHIP_FILE,
    index=False
)

matched_df.to_csv(
    MATCHED_FILE,
    index=False
)

tree_only_df.to_csv(
    TREE_ONLY_FILE,
    index=False
)

ml_only_df.to_csv(
    ML_ONLY_FILE,
    index=False
)


# ============================================================
# PRINT TREE-ONLY TAXA
# ============================================================

if len(tree_only_df) > 0:

    print()
    print("=" * 80)
    print("TREE-ONLY TAXA")
    print("=" * 80)

    print(
        tree_only_df[
            ["Tree_Tip"]
        ].to_string(
            index=False
        )
    )


# ============================================================
# PRINT ML-ONLY TAXA
# ============================================================

if len(ml_only_df) > 0:

    print()
    print("=" * 80)
    print("ML-ONLY TAXA")
    print("=" * 80)

    print(
        ml_only_df.to_string(
            index=False
        )
    )


# ============================================================
# CREATE 420-TAXON PRUNED ASTRAL TREE
# ============================================================

print()
print("=" * 80)
print("CREATING 420-TAXON ASTRAL TREE")
print("=" * 80)


# ------------------------------------------------------------
# A 420-taxon tree can only be produced when every ML taxon
# has a corresponding tree tip.
# ------------------------------------------------------------

if len(ml_only_df) > 0:

    print()
    print(
        "⚠ The ASTRAL tree does not contain "
        "all 420 ML taxa."
    )

    print()
    print(
        "A 420-taxon pruned tree will NOT "
        "be created."
    )

    print()
    print(
        "Resolve ML-only taxa before proceeding."
    )

else:

    # --------------------------------------------------------
    # Tree tips that correspond to the 420 ML taxa
    # --------------------------------------------------------

    keep_tips = set(
        matched_df[
            "Tree_Tip"
        ]
    )

    tips_to_remove = [
        tip
        for tip in tree_tips
        if tip not in keep_tips
    ]

    print()
    print(
        f"Tree tips to retain: "
        f"{len(keep_tips)}"
    )

    print(
        f"Tree tips to remove: "
        f"{len(tips_to_remove)}"
    )

    # --------------------------------------------------------
    # CRITICAL FIX:
    #
    # Bio.Phylo Tree has no .deepcopy() method.
    # Use Python's copy.deepcopy().
    # --------------------------------------------------------

    pruned_tree = copy.deepcopy(
        tree
    )

    # --------------------------------------------------------
    # Remove unwanted tips
    # --------------------------------------------------------

    for terminal in list(
        pruned_tree.get_terminals()
    ):

        terminal_name = (
            clean_tree_label(
                terminal.name
            )
        )

        if terminal_name not in keep_tips:

            pruned_tree.prune(
                terminal
            )

    # --------------------------------------------------------
    # Validate resulting tree
    # --------------------------------------------------------

    remaining_tips = [

        clean_tree_label(
            clade.name
        )

        for clade
        in pruned_tree.get_terminals()

        if clade.name
    ]

    print()
    print(
        f"Pruned tree tips: "
        f"{len(remaining_tips)}"
    )

    # --------------------------------------------------------
    # HARD VALIDATION
    # --------------------------------------------------------

    if len(remaining_tips) != 420:

        raise RuntimeError(
            "\n"
            "Pruned tree does not contain exactly "
            "420 tips.\n"
            f"Expected: 420\n"
            f"Found: {len(remaining_tips)}\n"
        )

    # --------------------------------------------------------
    # Check uniqueness
    # --------------------------------------------------------

    unique_remaining = set(
        remaining_tips
    )

    if len(unique_remaining) != 420:

        raise RuntimeError(
            "The pruned ASTRAL tree contains "
            "duplicate tip labels."
        )

    # --------------------------------------------------------
    # Write new tree
    # --------------------------------------------------------

    Phylo.write(
        pruned_tree,
        str(PRUNED_TREE_FILE),
        "newick"
    )

    print()
    print(
        "✓ 420-taxon ASTRAL tree written:"
    )

    print(
        PRUNED_TREE_FILE
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

all_ml_matched = (
    len(ml_only_df) == 0
)

all_expected_matched = (
    len(matched_df) == 420
)

tree_contains_420 = (
    len(tree_tips) >= 420
)

if (
    all_ml_matched
    and all_expected_matched
):

    print()
    print(
        "✓ All 420 ML taxa are represented "
        "in the ASTRAL tree."
    )

else:

    print()
    print(
        "⚠ ASTRAL ↔ ML reconciliation "
        "is incomplete."
    )


# ============================================================
# SUMMARY FILE
# ============================================================

summary = pd.DataFrame([{

    "Final_ML_Taxa":
        len(df),

    "ASTRAL_Tree_Tips":
        len(tree_tips),

    "Matched_Taxa":
        len(matched_df),

    "Tree_Only_Taxa":
        len(tree_only_df),

    "ML_Only_Taxa":
        len(ml_only_df),

    "All_420_ML_Taxa_Represented":
        all_ml_matched,

    "420_Taxon_Pruned_Tree_Created":
        PRUNED_TREE_FILE.exists(),

    "Original_ASTRAL_Modified":
        False

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
print("ASTRAL ↔ 420 RECONCILIATION COMPLETE")
print("=" * 80)

print()
print(
    f"Final ML taxa:       {len(df)}"
)

print(
    f"ASTRAL tree tips:    {len(tree_tips)}"
)

print(
    f"Matched:             {len(matched_df)}"
)

print(
    f"Tree-only:           {len(tree_only_df)}"
)

print(
    f"ML-only:             {len(ml_only_df)}"
)

print()
print("Membership table:")
print(MEMBERSHIP_FILE)

print()
print("Matched taxa:")
print(MATCHED_FILE)

print()
print("Tree-only taxa:")
print(TREE_ONLY_FILE)

print()
print("ML-only taxa:")
print(ML_ONLY_FILE)

print()
print("Summary:")
print(SUMMARY_FILE)

if PRUNED_TREE_FILE.exists():

    print()
    print(
        "420-taxon pruned ASTRAL tree:"
    )

    print(
        PRUNED_TREE_FILE
    )

print()
print(
    "✓ Final 420 ML dataset was NOT modified."
)

print(
    "✓ Original ASTRAL tree was NOT modified."
)

print(
    "✓ Reconciliation tables were written."
)

if PRUNED_TREE_FILE.exists():

    print(
        "✓ 420-taxon ASTRAL tree was created."
    )

print()
print("=" * 80)