from pathlib import Path
import pandas as pd

# ============================================================
# CHECK PARETO CANDIDATES IN FINAL 419 ML DATASET
# ============================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

ML_FILE = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "feature_qc"
    / "y1000_419_variable_busco_ml_matrix.csv"
)

OUTPUT_DIR = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "pareto_qc"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Pareto candidates confirmed from previous QC
# ------------------------------------------------------------

PARETO = [
    ("GCA_003705225.1", "Ambrosiozyma vanderkliftii"),
    ("GCA_003123585.1", "Barnettozyma californica"),
    ("GCA_003709245.3", "Cyberlindnera saturnus"),
    ("GCA_030558845.1", "Kodamaea laetipori"),
    ("GCA_030563145.1", "Schwanniomyces polymorphus var. africanus"),
    ("GCA_030463025.1", "Schwanniomyces polymorphus"),
    ("GCA_030583345.1", "Schwanniomyces pseudopolymorphus"),
    ("GCA_030583405.1", "Sugiyamaella americana"),
    ("GCA_030579815.1", "Sugiyamaella smithiae"),
    ("GCA_030558095.1", "Teunomyces funiuensis"),
    ("GCA_030564625.1", "Zygoascus hellenicus"),
]

# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("PARETO CANDIDATES ↔ FINAL 419 ML DATASET")
print("=" * 80)

print()
print("ML dataset:")
print(ML_FILE)

if not ML_FILE.exists():
    raise FileNotFoundError(
        f"ML dataset not found:\n{ML_FILE}"
    )

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(ML_FILE)

print()
print("ML dataset dimensions:")
print(f"  Rows:    {len(df)}")
print(f"  Columns: {len(df)}")

if len(df) != 419:
    raise RuntimeError(
        f"Expected 419 rows but found {len(df)}"
    )

# ============================================================
# NORMALIZATION
# ============================================================

def norm(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
    )


df["_species_key"] = df["Species"].apply(norm)

df["_accession_key"] = (
    df["Assembly_Accession"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# ============================================================
# CHECK EACH PARETO CANDIDATE
# ============================================================

print()
print("=" * 80)
print("PARETO MEMBERSHIP")
print("=" * 80)

results = []

for accession, species in PARETO:

    accession_key = accession.lower().strip()
    species_key = norm(species)

    accession_match = df[
        df["_accession_key"] == accession_key
    ]

    species_match = df[
        df["_species_key"] == species_key
    ]

    if len(accession_match) > 0:

        row = accession_match.iloc[0]

        status = "IN_419"

        print(
            f"✓ {accession:<16} "
            f"{species}"
        )

        print(
            f"    Phenotype: "
            f"Carbon={row['Carbon_Breadth']}, "
            f"Nitrogen={row['Nitrogen_Breadth']}, "
            f"Growth={row['Utilized_Median_Growth']}"
        )

        results.append({
            "Assembly_Accession": accession,
            "Species": species,
            "In_419": True,
            "Match_Type": "Assembly_Accession",
            "Carbon_Breadth": row["Carbon_Breadth"],
            "Nitrogen_Breadth": row["Nitrogen_Breadth"],
            "Utilized_Median_Growth": row["Utilized_Median_Growth"],
            "Carbon_Breadth_SD": row["Carbon_Breadth_SD"],
            "Nitrogen_Breadth_SD": row["Nitrogen_Breadth_SD"],
        })

    elif len(species_match) > 0:

        row = species_match.iloc[0]

        status = "SPECIES_ONLY"

        print(
            f"! {accession:<16} "
            f"{species}"
        )

        print(
            "    Species found, but accession does not match."
        )

        results.append({
            "Assembly_Accession": accession,
            "Species": species,
            "In_419": True,
            "Match_Type": "Species_only",
            "Carbon_Breadth": row["Carbon_Breadth"],
            "Nitrogen_Breadth": row["Nitrogen_Breadth"],
            "Utilized_Median_Growth": row["Utilized_Median_Growth"],
            "Carbon_Breadth_SD": row["Carbon_Breadth_SD"],
            "Nitrogen_Breadth_SD": row["Nitrogen_Breadth_SD"],
        })

    else:

        print(
            f"✗ {accession:<16} "
            f"{species}"
        )

        print(
            "    NOT PRESENT in final 419 ML dataset."
        )

        results.append({
            "Assembly_Accession": accession,
            "Species": species,
            "In_419": False,
            "Match_Type": "Not_found",
            "Carbon_Breadth": None,
            "Nitrogen_Breadth": None,
            "Utilized_Median_Growth": None,
            "Carbon_Breadth_SD": None,
            "Nitrogen_Breadth_SD": None,
        })

# ============================================================
# RESULT TABLE
# ============================================================

result_df = pd.DataFrame(results)

result_file = (
    OUTPUT_DIR
    / "pareto_membership_419_ml_dataset.csv"
)

result_df.to_csv(
    result_file,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

n_total = len(result_df)
n_in = int(result_df["In_419"].sum())
n_out = n_total - n_in

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print()
print(f"Total Pareto candidates:       {n_total}")
print(f"Present in 419 ML dataset:     {n_in}")
print(f"Absent from 419 ML dataset:    {n_out}")

print()

if n_out > 0:

    print("Pareto candidates NOT in 419:")

    for _, row in result_df[
        ~result_df["In_419"]
    ].iterrows():

        print(
            f"  {row['Assembly_Accession']} "
            f"{row['Species']}"
        )

else:

    print(
        "✓ ALL 11 Pareto candidates are represented "
        "in the final 419 ML dataset."
    )

# ============================================================
# SPECIFIC ASTRAL-REPRESENTED PARETO SET
# ============================================================

astral_pareto = result_df[
    result_df["Assembly_Accession"] != "GCA_030563145.1"
]

astral_in_419 = astral_pareto[
    astral_pareto["In_419"]
]

print()
print("=" * 80)
print("ASTRAL-REPRESENTED PARETO SET")
print("=" * 80)

print()
print(
    f"Pareto candidates represented in ASTRAL: "
    f"{len(astral_pareto)}"
)

print(
    f"Those present in 419 ML dataset: "
    f"{len(astral_in_419)}"
)

if len(astral_in_419) == len(astral_pareto):

    print()
    print(
        "✓ ALL ASTRAL-REPRESENTED PARETO CANDIDATES "
        "ARE IN THE 419 ML DATASET."
    )

else:

    print()
    print(
        "⚠ Some ASTRAL-represented Pareto candidates "
        "are absent from the 419 ML dataset."
    )

# ============================================================
# CLEANUP
# ============================================================

# Do not retain helper columns in memory/output.
df.drop(
    columns=[
        "_species_key",
        "_accession_key"
    ],
    inplace=True,
    errors="ignore"
)

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("PARETO ↔ 419 QC COMPLETE")
print("=" * 80)

print()
print("Output:")
print(result_file)

print()
print("✓ Final 419 ML dataset was NOT modified.")