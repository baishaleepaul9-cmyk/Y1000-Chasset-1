from pathlib import Path
import pandas as pd
import numpy as np

# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

STAGE5_DIR = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
)

ML_DIR = (
    STAGE5_DIR
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
)

FEATURE_DIR = (
    ML_DIR
    / "feature_importance_analysis"
)

ROBUST_DIR = (
    FEATURE_DIR
    / "robust_feature_analysis"
)

OUTPUT_DIR = (
    ROBUST_DIR
    / "phylogenetic_candidate_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ================================================================
# INPUT FILES
# ================================================================

CANDIDATE_FILE = (
    ROBUST_DIR
    / "strong_multi_trait_busco_candidates_420.csv"
)

ML_MATRIX = (
    ML_DIR
    / "phylogeny_aware_ml_matrix_420.csv"
)

if not ML_MATRIX.exists():

    alternatives = [
        ML_DIR / "final_phylogeny_aware_ml_matrix_420.csv",
        STAGE5_DIR / "phylogeny_aware_ml_matrix_420.csv"
    ]

    for alternative in alternatives:

        if alternative.exists():

            ML_MATRIX = alternative
            break

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(
        f"Candidate file not found:\n{CANDIDATE_FILE}"
    )

if not ML_MATRIX.exists():
    raise FileNotFoundError(
        "Could not find the 420-taxon ML matrix.\n"
        f"Last checked:\n{ML_MATRIX}"
    )

# ================================================================
# LOAD CANDIDATES
# ================================================================

print("=" * 80)
print("PHYLOGENETIC ANALYSIS OF MULTI-TRAIT BUSCO CANDIDATES")
print("=" * 80)

candidates = pd.read_csv(
    CANDIDATE_FILE
)

print("\nCandidate file:")
print(CANDIDATE_FILE)

print("\nCandidate shape:")
print(candidates.shape)

print("\nCandidate columns:")
print(candidates.columns.tolist())

# ================================================================
# LOAD ML MATRIX
# ================================================================

df = pd.read_csv(
    ML_MATRIX
)

print("\nML matrix:")
print(ML_MATRIX)

print("\nML matrix shape:")
print(df.shape)

# ================================================================
# BASIC QC
# ================================================================

required_candidate_column = "BUSCO"

if required_candidate_column not in candidates.columns:
    raise RuntimeError(
        "BUSCO column missing from candidate table."
    )

if "Species" not in df.columns:
    raise RuntimeError(
        "Species column missing from ML matrix."
    )

# ================================================================
# GET CANDIDATE BUSCOs
# ================================================================

candidate_buscos = (
    candidates["BUSCO"]
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .tolist()
)

print(
    f"\nUnique candidate BUSCOs: "
    f"{len(candidate_buscos)}"
)

# ================================================================
# CHECK BUSCO AVAILABILITY
# ================================================================

available_buscos = [
    b
    for b in candidate_buscos
    if b in df.columns
]

missing_buscos = [
    b
    for b in candidate_buscos
    if b not in df.columns
]

print(
    f"BUSCOs available in ML matrix: "
    f"{len(available_buscos)}"
)

print(
    f"BUSCOs missing from ML matrix: "
    f"{len(missing_buscos)}"
)

if missing_buscos:

    print("\nMissing BUSCOs:")

    for b in missing_buscos:
        print("  ", b)

if len(available_buscos) == 0:
    raise RuntimeError(
        "None of the candidate BUSCOs were found "
        "in the ML matrix."
    )

# ================================================================
# BUILD 420-TAXON CANDIDATE MATRIX
# ================================================================

metadata_columns = [
    "Species",
    "Assembly_Accession",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Phenotype_Source_Species"
]

metadata_columns = [
    c
    for c in metadata_columns
    if c in df.columns
]

candidate_matrix = df[
    ["Species"] + available_buscos
].copy()

# ================================================================
# FORCE BINARY BUSCO VALUES
# ================================================================

for busco in available_buscos:

    candidate_matrix[busco] = pd.to_numeric(
        candidate_matrix[busco],
        errors="coerce"
    )

# Check missing values
missing_cells = int(
    candidate_matrix[
        available_buscos
    ].isna()
    .sum()
    .sum()
)

print(
    f"\nMissing candidate BUSCO cells: "
    f"{missing_cells}"
)

if missing_cells != 0:

    raise RuntimeError(
        "Candidate BUSCO matrix contains missing values."
    )

# Check binary values
unique_values = sorted(
    set(
        candidate_matrix[
            available_buscos
        ]
        .to_numpy()
        .ravel()
        .tolist()
    )
)

print(
    "\nUnique candidate BUSCO values:"
)

print(unique_values)

if not set(unique_values).issubset({0, 1}):

    raise RuntimeError(
        "Candidate BUSCO matrix contains "
        "values other than 0 and 1."
    )

candidate_matrix[
    available_buscos
] = candidate_matrix[
    available_buscos
].astype(int)

# ================================================================
# SAVE CANDIDATE MATRIX
# ================================================================

matrix_file = (
    OUTPUT_DIR
    / "79_multitrait_busco_matrix_420.csv"
)

candidate_matrix.to_csv(
    matrix_file,
    index=False
)

print(
    "\n✓ Candidate BUSCO matrix written:"
)

print(matrix_file)

# ================================================================
# PREVALENCE ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("BUSCO PREVALENCE ANALYSIS")
print("=" * 80)

prevalence_rows = []

n_taxa = len(candidate_matrix)

for busco in available_buscos:

    values = candidate_matrix[busco]

    present = int(values.sum())

    prevalence = present / n_taxa

    prevalence_rows.append({

        "BUSCO": busco,

        "Taxa_Total": n_taxa,

        "Taxa_Present": present,

        "Taxa_Absent": n_taxa - present,

        "Prevalence": prevalence

    })

prevalence_df = pd.DataFrame(
    prevalence_rows
)

prevalence_df = prevalence_df.sort_values(
    "Prevalence",
    ascending=False
)

prevalence_file = (
    OUTPUT_DIR
    / "79_multitrait_busco_prevalence_420.csv"
)

prevalence_df.to_csv(
    prevalence_file,
    index=False
)

print(
    "\n✓ Prevalence table written:"
)

print(prevalence_file)

# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("PHYLOGENETIC CANDIDATE MATRIX COMPLETE")
print("=" * 80)

print(
    f"\nTaxa: {n_taxa}"
)

print(
    f"Candidate BUSCOs: "
    f"{len(available_buscos)}"
)

print(
    "\nOutput:"
)

print(matrix_file)
print(prevalence_file)

print(
    "\nNext step:"
)

print(
    "Use the 420-taxon matrix together with "
    "the validated ASTRAL tree to test "
    "phylogenetic clustering."
)