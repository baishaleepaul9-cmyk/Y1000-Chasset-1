# ============================================================
# 420-TAXON ML DATASET ↔ ASTRAL PHYLOGENY QC
# ============================================================

from pathlib import Path
import pandas as pd
import re
import sys

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

ML_MATRIX = (
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# OUTPUT FILES
# ============================================================

MEMBERSHIP_FILE = OUTPUT_DIR / "astral_420_taxon_membership.csv"
SUMMARY_FILE = OUTPUT_DIR / "astral_420_taxon_qc_summary.csv"
TREE_TIPS_FILE = OUTPUT_DIR / "astral_tree_tip_inventory.csv"
MAPPING_FILE = OUTPUT_DIR / "astral_420_taxon_mapping.csv"


# ============================================================
# HELPERS
# ============================================================

def norm_species(value):
    """
    Normalize species names for comparison.

    Handles:
      - leading/trailing whitespace
      - repeated whitespace
      - underscores
      - case differences
    """
    if pd.isna(value):
        return ""

    s = str(value).strip()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def clean_tree_label(label):
    """
    Clean an ASTRAL tree tip label.
    """
    if label is None:
        return ""

    s = str(label).strip()

    # Remove surrounding quotes if present
    s = s.strip("'\"")

    # Newick labels sometimes contain underscores
    s = s.replace("_", " ")

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def is_accession(value):
    """
    Detect NCBI-style assembly accession.
    """
    if not value:
        return False

    return bool(
        re.fullmatch(
            r"(GC[AF]_\d+\.\d+)",
            str(value).strip(),
            flags=re.IGNORECASE
        )
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("420-TAXON ML DATASET ↔ ASTRAL PHYLOGENY QC")
print("=" * 80)

print()
print("Project:")
print(PROJECT)

print()
print("ML matrix:")
print(ML_MATRIX)

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
print("CHECKING INPUT FILES")
print("=" * 80)

if not ML_MATRIX.exists():
    raise FileNotFoundError(
        f"ML matrix not found:\n{ML_MATRIX}"
    )

if not ASTRAL_DIR.exists():
    raise FileNotFoundError(
        f"ASTRAL directory not found:\n{ASTRAL_DIR}"
    )

print("✓ ML matrix found")
print("✓ ASTRAL directory found")


# ============================================================
# LOAD ML MATRIX
# ============================================================

print()
print("=" * 80)
print("LOADING FINAL 420-TAXON ML MATRIX")
print("=" * 80)

ml = pd.read_csv(ML_MATRIX)

print()
print(f"Rows:    {ml.shape[0]}")
print(f"Columns: {ml.shape[1]}")

required_columns = [
    "Species",
    "Assembly_Accession"
]

missing_required = [
    c for c in required_columns
    if c not in ml.columns
]

if missing_required:
    raise RuntimeError(
        "Required columns missing from ML matrix:\n"
        + "\n".join(missing_required)
    )

print("✓ Species column present")
print("✓ Assembly_Accession column present")


# ============================================================
# ML TAXON QC
# ============================================================

print()
print("=" * 80)
print("ML TAXON QC")
print("=" * 80)

ml["_species_key"] = ml["Species"].apply(norm_species)
ml["_accession_key"] = (
    ml["Assembly_Accession"]
    .astype(str)
    .str.strip()
    .str.upper()
)

ml_species = set(
    x for x in ml["_species_key"]
    if x
)

ml_accessions = set(
    x for x in ml["_accession_key"]
    if x and x != "NAN"
)

print()
print(f"ML taxa:                  {len(ml)}")
print(f"Unique ML species:        {len(ml_species)}")
print(f"Unique ML accessions:     {len(ml_accessions)}")

duplicate_species = (
    ml["_species_key"]
    .value_counts()
)

duplicate_species = duplicate_species[
    duplicate_species > 1
]

duplicate_accessions = (
    ml["_accession_key"]
    .value_counts()
)

duplicate_accessions = duplicate_accessions[
    duplicate_accessions > 1
]

print()
print(f"Duplicate ML species:     {len(duplicate_species)}")
print(f"Duplicate ML accessions:  {len(duplicate_accessions)}")

if len(ml) != 420:
    raise RuntimeError(
        f"Expected exactly 420 ML taxa but found {len(ml)}."
    )

if len(duplicate_species) > 0:
    raise RuntimeError(
        "Duplicate species detected in final ML matrix."
    )

if len(duplicate_accessions) > 0:
    raise RuntimeError(
        "Duplicate assembly accessions detected in final ML matrix."
    )

print()
print("✓ Exactly 420 unique ML taxa")
print("✓ Assembly accessions are unique")


# ============================================================
# LOCATE ASTRAL TREE
# ============================================================

print()
print("=" * 80)
print("LOCATING ASTRAL SPECIES TREE")
print("=" * 80)

nwk_files = sorted(ASTRAL_DIR.glob("*.nwk"))

if not nwk_files:
    nwk_files = sorted(ASTRAL_DIR.glob("*.newick"))

if not nwk_files:
    raise FileNotFoundError(
        "No .nwk or .newick files found in:\n"
        f"{ASTRAL_DIR}"
    )

print()
print("Candidate Newick files:")

for f in nwk_files:
    print(f"  {f.name}")


# Prefer the species tree and midpoint tree if available.
preferred = []

for f in nwk_files:
    name = f.name.lower()

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

    preferred.append((score, f))

preferred.sort(
    key=lambda x: (-x[0], x[1].name)
)

ASTRAL_TREE = preferred[0][1]

print()
print("Selected ASTRAL tree:")
print(ASTRAL_TREE)

print()
print("✓ ASTRAL tree selected")


# ============================================================
# LOAD TREE
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
        f"Could not read ASTRAL tree:\n{ASTRAL_TREE}\n\n"
        f"Error: {e}"
    )

tips = [
    clean_tree_label(
        terminal.name
    )
    for terminal in tree.get_terminals()
]

tips = [
    x for x in tips
    if x
]

print()
print(f"Tree tips: {len(tips)}")

if len(tips) == 0:
    raise RuntimeError(
        "ASTRAL tree contains no readable terminal labels."
    )

print("✓ Tree loaded")


# ============================================================
# TREE TIP INVENTORY
# ============================================================

print()
print("=" * 80)
print("TREE TIP INVENTORY")
print("=" * 80)

tree_tip_df = pd.DataFrame({
    "Tree_Tip": tips,
    "Tree_Tip_Normalized": [
        norm_species(x)
        for x in tips
    ]
})

tree_tip_df["Is_Assembly_Accession"] = (
    tree_tip_df["Tree_Tip"]
    .apply(is_accession)
)

tree_tip_df.to_csv(
    TREE_TIPS_FILE,
    index=False
)

print()
print(f"Tree tips recorded: {len(tree_tip_df)}")
print()
print("Inventory:")
print(TREE_TIPS_FILE)


# ============================================================
# DUPLICATE TREE TIPS
# ============================================================

print()
print("=" * 80)
print("ASTRAL TREE TIP DUPLICATE QC")
print("=" * 80)

tree_species_keys = tree_tip_df[
    ~tree_tip_df["Is_Assembly_Accession"]
]["Tree_Tip_Normalized"]

duplicate_tree_species = (
    tree_species_keys
    .value_counts()
)

duplicate_tree_species = duplicate_tree_species[
    duplicate_tree_species > 1
]

tree_accession_keys = tree_tip_df[
    tree_tip_df["Is_Assembly_Accession"]
]["Tree_Tip"].str.upper()

duplicate_tree_accessions = (
    tree_accession_keys
    .value_counts()
)

duplicate_tree_accessions = (
    duplicate_tree_accessions[
        duplicate_tree_accessions > 1
    ]
)

print()
print(
    f"Duplicate tree species tips: "
    f"{len(duplicate_tree_species)}"
)

print(
    f"Duplicate tree accession tips: "
    f"{len(duplicate_tree_accessions)}"
)

if len(duplicate_tree_species) > 0:
    print()
    print("WARNING: duplicate species tips detected:")
    print(duplicate_tree_species)

if len(duplicate_tree_accessions) > 0:
    print()
    print("WARNING: duplicate accession tips detected:")
    print(duplicate_tree_accessions)


# ============================================================
# DETERMINE TREE REPRESENTATION
# ============================================================

print()
print("=" * 80)
print("DETERMINING TREE TAXON LABEL TYPE")
print("=" * 80)

accession_tip_count = int(
    tree_tip_df["Is_Assembly_Accession"].sum()
)

species_tip_count = len(tree_tip_df) - accession_tip_count

print()
print(f"Species-name tips:     {species_tip_count}")
print(f"Accession tips:        {accession_tip_count}")

if accession_tip_count > species_tip_count:
    TREE_LABEL_MODE = "ACCESSION"
else:
    TREE_LABEL_MODE = "SPECIES"

print()
print(f"Detected tree label mode: {TREE_LABEL_MODE}")


# ============================================================
# CREATE ML ↔ TREE MEMBERSHIP
# ============================================================

print()
print("=" * 80)
print("ML ↔ ASTRAL TAXON RECONCILIATION")
print("=" * 80)

tree_species_set = set(
    tree_tip_df[
        ~tree_tip_df["Is_Assembly_Accession"]
    ]["Tree_Tip_Normalized"]
)

tree_accession_set = set(
    tree_tip_df[
        tree_tip_df["Is_Assembly_Accession"]
    ]["Tree_Tip"]
    .astype(str)
    .str.strip()
    .str.upper()
)


records = []

for _, row in ml.iterrows():

    species = str(
        row["Species"]
    ).strip()

    accession = str(
        row["Assembly_Accession"]
    ).strip().upper()

    species_key = norm_species(species)

    species_match = (
        species_key in tree_species_set
    )

    accession_match = (
        accession in tree_accession_set
    )

    if accession_match:
        status = "MATCHED"
        match_basis = "EXACT_ASSEMBLY_ACCESSION"

        matched_tip = next(
            (
                x for x in tree_tip_df[
                    tree_tip_df[
                        "Is_Assembly_Accession"
                    ]
                ]["Tree_Tip"]
                if x.strip().upper() == accession
            ),
            None
        )

    elif species_match:
        status = "MATCHED"
        match_basis = "SPECIES_NAME"

        matched_tip = next(
            (
                x for x in tree_tip_df[
                    ~tree_tip_df[
                        "Is_Assembly_Accession"
                    ]
                ]["Tree_Tip"]
                if norm_species(x) == species_key
            ),
            None
        )

    else:
        status = "NOT_FOUND"
        match_basis = "NO_MATCH"
        matched_tip = None

    records.append({
        "Species": species,
        "Assembly_Accession": accession,
        "Tree_Matched": status,
        "Match_Basis": match_basis,
        "Matched_Tree_Tip": matched_tip
    })


membership = pd.DataFrame(records)


# ============================================================
# MEMBERSHIP SUMMARY
# ============================================================

print()
print("=" * 80)
print("MEMBERSHIP SUMMARY")
print("=" * 80)

matched = membership[
    membership["Tree_Matched"] == "MATCHED"
]

unmatched = membership[
    membership["Tree_Matched"] == "NOT_FOUND"
]

print()
print(f"Final ML taxa:       {len(membership)}")
print(f"Matched to ASTRAL:   {len(matched)}")
print(f"Not found:           {len(unmatched)}")

if len(matched) == 420:
    print()
    print("✓ ALL 420 ML TAXA ARE REPRESENTED IN ASTRAL")
else:
    print()
    print("⚠ NOT ALL 420 ML TAXA ARE REPRESENTED")

    print()
    print("Unmatched ML taxa:")

    if len(unmatched) > 0:
        print(
            unmatched[
                [
                    "Species",
                    "Assembly_Accession"
                ]
            ].to_string(index=False)
        )


# ============================================================
# TREE-ONLY TAXA
# ============================================================

print()
print("=" * 80)
print("ASTRAL-ONLY TAXON CHECK")
print("=" * 80)

ml_species_set = set(
    ml["_species_key"]
)

if TREE_LABEL_MODE == "SPECIES":

    astral_only = sorted(
        tree_species_set - ml_species_set
    )

    ml_only = sorted(
        ml_species_set - tree_species_set
    )

else:

    astral_only = sorted(
        tree_accession_set - ml_accessions
    )

    ml_only = sorted(
        ml_accessions - tree_accession_set
    )

print()
print(f"ASTRAL-only taxa: {len(astral_only)}")
print(f"ML-only taxa:     {len(ml_only)}")

if len(astral_only) > 0:

    print()
    print("First ASTRAL-only taxa:")

    for x in astral_only[:30]:
        print(f"  {x}")

if len(ml_only) > 0:

    print()
    print("First ML-only taxa:")

    for x in ml_only[:30]:
        print(f"  {x}")


# ============================================================
# PARETO CROSS-CHECK
# ============================================================

print()
print("=" * 80)
print("PARETO CROSS-CHECK")
print("=" * 80)

pareto_file = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "pareto_qc"
    / "pareto_420_species_level_reconciliation.csv"
)

if pareto_file.exists():

    pareto = pd.read_csv(
        pareto_file
    )

    print()
    print(
        f"Pareto reconciliation rows: "
        f"{len(pareto)}"
    )

    pareto_records = []

    for _, row in pareto.iterrows():

        pareto_species = str(
            row["Pareto_Species"]
        ).strip()

        matched_species = str(
            row.get(
                "Matched_Species",
                ""
            )
        ).strip()

        matched_accession = str(
            row.get(
                "Matched_Assembly_Accession",
                ""
            )
        ).strip().upper()

        # Direct accession check
        accession_present = (
            matched_accession
            in ml_accessions
        )

        # Species check
        species_present = (
            norm_species(matched_species)
            in ml_species_set
        )

        present = (
            accession_present
            or species_present
        )

        pareto_records.append({
            "Pareto_Assembly_Accession":
                row.get(
                    "Pareto_Assembly_Accession",
                    ""
                ),
            "Pareto_Species":
                pareto_species,
            "Matched_Assembly_Accession":
                matched_accession,
            "Matched_Species":
                matched_species,
            "Present_in_420_ML":
                present,
            "Present_in_ASTRAL":
                (
                    matched_accession
                    in set(
                        membership[
                            membership[
                                "Tree_Matched"
                            ] == "MATCHED"
                        ][
                            "Assembly_Accession"
                        ]
                    )
                )
        })

    pareto_check = pd.DataFrame(
        pareto_records
    )

    print()
    print(
        "Pareto candidates represented in "
        f"420 ML dataset: "
        f"{pareto_check['Present_in_420_ML'].sum()}"
        f"/{len(pareto_check)}"
    )

    print(
        "Pareto candidates represented in "
        f"ASTRAL: "
        f"{pareto_check['Present_in_ASTRAL'].sum()}"
        f"/{len(pareto_check)}"
    )

else:

    print()
    print(
        "⚠ Pareto reconciliation file not found."
    )

    pareto_check = None


# ============================================================
# SAVE MEMBERSHIP
# ============================================================

membership.to_csv(
    MEMBERSHIP_FILE,
    index=False
)

print()
print("Membership table:")
print(MEMBERSHIP_FILE)


# ============================================================
# SAVE PARETO CROSS-CHECK
# ============================================================

if pareto_check is not None:

    pareto_output = (
        OUTPUT_DIR
        / "astral_pareto_420_crosscheck.csv"
    )

    pareto_check.to_csv(
        pareto_output,
        index=False
    )

    print()
    print("Pareto cross-check:")
    print(pareto_output)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary = pd.DataFrame([
    {
        "Metric":
            "Final ML taxa",
        "Value":
            len(ml)
    },
    {
        "Metric":
            "Unique ML species",
        "Value":
            len(ml_species)
    },
    {
        "Metric":
            "ASTRAL tree tips",
        "Value":
            len(tips)
    },
    {
        "Metric":
            "ML taxa matched to ASTRAL",
        "Value":
            len(matched)
    },
    {
        "Metric":
            "ML taxa missing from ASTRAL",
        "Value":
            len(unmatched)
    },
    {
        "Metric":
            "ASTRAL-only taxa",
        "Value":
            len(astral_only)
    },
    {
        "Metric":
            "ML-only taxa",
        "Value":
            len(ml_only)
    },
    {
        "Metric":
            "Duplicate ML species",
        "Value":
            len(duplicate_species)
    },
    {
        "Metric":
            "Duplicate ML accessions",
        "Value":
            len(duplicate_accessions)
    },
    {
        "Metric":
            "Duplicate ASTRAL species tips",
        "Value":
            len(duplicate_tree_species)
    },
    {
        "Metric":
            "Duplicate ASTRAL accession tips",
        "Value":
            len(duplicate_tree_accessions)
    },
])


# ============================================================
# FINAL STATUS
# ============================================================

all_ml_matched = (
    len(membership) == 420
    and len(unmatched) == 0
)

no_duplicate_ml = (
    len(duplicate_species) == 0
    and len(duplicate_accessions) == 0
)

print()
print("=" * 80)
print("FINAL PHYLOGENY QC")
print("=" * 80)

print()
print(f"ML taxa:                    {len(ml)}")
print(f"ASTRAL tree tips:           {len(tips)}")
print(f"ML taxa matched:            {len(matched)}")
print(f"ML taxa unmatched:          {len(unmatched)}")
print(f"ASTRAL-only taxa:           {len(astral_only)}")
print(f"ML-only taxa:               {len(ml_only)}")

print()

if all_ml_matched and no_duplicate_ml:

    print("✓ ALL 420 ML TAXA RECONCILE WITH ASTRAL")
    print("✓ ML species are unique")
    print("✓ ML assembly accessions are unique")

    if len(astral_only) == 0:

        print("✓ ASTRAL contains exactly the ML taxon set")

        final_status = "PASS_EXACT_420_TAXON_RECONCILIATION"

    else:

        print(
            "✓ All ML taxa are represented in ASTRAL"
        )
        print(
            "⚠ ASTRAL contains additional taxa"
        )

        final_status = "PASS_ML_SUBSET_OF_ASTRAL"

else:

    print(
        "⚠ PHYLOGENY QC REQUIRES INVESTIGATION"
    )

    final_status = "FAIL_TAXON_RECONCILIATION"


summary.loc[
    len(summary)
] = {
    "Metric": "FINAL_STATUS",
    "Value": final_status
}

summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print()
print("=" * 80)
print("ASTRAL ↔ 420 QC COMPLETE")
print("=" * 80)

print()
print("Membership:")
print(MEMBERSHIP_FILE)

print()
print("Summary:")
print(SUMMARY_FILE)

print()
print("Tree inventory:")
print(TREE_TIPS_FILE)

print()
print("✓ Original 420 ML dataset was NOT modified.")
print("✓ ASTRAL tree was NOT modified.")
print("✓ Pareto candidate list was NOT modified.")