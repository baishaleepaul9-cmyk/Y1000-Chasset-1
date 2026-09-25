# ================================================================
# PARETO ↔ FINAL 420 DATASET
# SPECIES-LEVEL REPRESENTATIVE RECONCILIATION
# ================================================================

import os
import pandas as pd

# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT = r"C:\Y1000_chassis_project"

ML_FILE = (
    PROJECT
    + r"\results\stage5_phylogeny_ml_dataset\feature_qc"
    + r"\y1000_420_variable_busco_ml_matrix.csv"
)

PARETO_FILE = (
    PROJECT
    + r"\results\stage5_phylogeny_ml_dataset\pareto_qc"
    + r"\pareto_membership_420_ml_dataset.csv"
)

MANIFEST_FILE = (
    PROJECT
    + r"\results\stage4B_phylogeny"
    + r"\stage4B_genome_fasta_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT
    + r"\results\stage5_phylogeny_ml_dataset"
    + r"\pareto_qc"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "pareto_420_species_level_reconciliation.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "pareto_420_species_level_reconciliation_summary.csv"
)


# ================================================================
# NORMALIZATION
# ================================================================

def norm_species(x):
    if pd.isna(x):
        return ""

    x = str(x).strip().lower()

    # Explicit normalization relevant to this project
    if x == "schwanniomyces polymorphus var. africanus":
        return "schwanniomyces polymorphus"

    if x == "schwanniomyces polymorphus var. polymorphus":
        return "schwanniomyces polymorphus"

    return x


# ================================================================
# LOAD FINAL 420 ML DATASET
# ================================================================

print("=" * 80)
print("PARETO ↔ FINAL 420 SPECIES-LEVEL RECONCILIATION")
print("=" * 80)

print("\nLoading final 420 ML dataset...")

ml = pd.read_csv(ML_FILE)

print(f"ML rows:    {len(ml)}")
print(f"ML columns: {len(ml.columns)}")

if len(ml) != 420:
    raise RuntimeError(
        f"Expected 420 ML taxa but found {len(ml)}."
    )

if "Species" not in ml.columns:
    raise RuntimeError("Species column missing from ML dataset.")

if "Assembly_Accession" not in ml.columns:
    raise RuntimeError(
        "Assembly_Accession column missing from ML dataset."
    )

ml["_species_key"] = ml["Species"].apply(norm_species)

print("✓ Final 420 dataset loaded.")


# ================================================================
# LOAD PARETO TABLE
# ================================================================

print("\nLoading Pareto membership table...")

pareto = pd.read_csv(PARETO_FILE)

print(f"Pareto rows: {len(pareto)}")

required_pareto = [
    "Pareto_Assembly_Accession",
    "Pareto_Species",
    "Present_in_420"
]

for col in required_pareto:
    if col not in pareto.columns:
        raise RuntimeError(
            f"Required Pareto column missing: {col}"
        )

pareto["_pareto_species_key"] = pareto[
    "Pareto_Species"
].apply(norm_species)

print("✓ Pareto table loaded.")


# ================================================================
# RECONCILE
# ================================================================

print("\n" + "=" * 80)
print("SPECIES-LEVEL RECONCILIATION")
print("=" * 80)

records = []

for _, row in pareto.iterrows():

    pareto_acc = str(
        row["Pareto_Assembly_Accession"]
    ).strip()

    pareto_species = str(
        row["Pareto_Species"]
    ).strip()

    species_key = row["_pareto_species_key"]

    # ------------------------------------------------------------
    # Exact accession match
    # ------------------------------------------------------------

    exact = ml[
        ml["Assembly_Accession"]
        .astype(str)
        .str.strip()
        == pareto_acc
    ]

    if len(exact) == 1:

        m = exact.iloc[0]

        records.append({
            "Pareto_Assembly_Accession": pareto_acc,
            "Pareto_Species": pareto_species,
            "Status": "EXACT_ASSEMBLY_MATCH",
            "Matched_Assembly_Accession":
                m["Assembly_Accession"],
            "Matched_Species":
                m["Species"],
            "Match_Basis":
                "Exact assembly accession",
            "Phenotype_N_Strains":
                m.get("N_Strains", None),
            "Carbon_Breadth":
                m.get("Carbon_Breadth", None),
            "Nitrogen_Breadth":
                m.get("Nitrogen_Breadth", None),
            "Utilized_Median_Growth":
                m.get("Utilized_Median_Growth", None)
        })

        continue

    # ------------------------------------------------------------
    # Species-level representative match
    # ------------------------------------------------------------

    species_match = ml[
        ml["_species_key"] == species_key
    ]

    if len(species_match) == 1:

        m = species_match.iloc[0]

        records.append({
            "Pareto_Assembly_Accession": pareto_acc,
            "Pareto_Species": pareto_species,
            "Status": "SPECIES_LEVEL_REPRESENTATIVE",
            "Matched_Assembly_Accession":
                m["Assembly_Accession"],
            "Matched_Species":
                m["Species"],
            "Match_Basis":
                "Pareto species mapped to final representative species",
            "Phenotype_N_Strains":
                m.get("N_Strains", None),
            "Carbon_Breadth":
                m.get("Carbon_Breadth", None),
            "Nitrogen_Breadth":
                m.get("Nitrogen_Breadth", None),
            "Utilized_Median_Growth":
                m.get("Utilized_Median_Growth", None)
        })

        continue

    # ------------------------------------------------------------
    # Multiple species-level matches
    # ------------------------------------------------------------

    if len(species_match) > 1:

        records.append({
            "Pareto_Assembly_Accession": pareto_acc,
            "Pareto_Species": pareto_species,
            "Status": "AMBIGUOUS_SPECIES_MATCH",
            "Matched_Assembly_Accession":
                "; ".join(
                    species_match[
                        "Assembly_Accession"
                    ].astype(str)
                ),
            "Matched_Species":
                "; ".join(
                    species_match[
                        "Species"
                    ].astype(str)
                ),
            "Match_Basis":
                "Multiple final representatives share normalized species",
            "Phenotype_N_Strains": None,
            "Carbon_Breadth": None,
            "Nitrogen_Breadth": None,
            "Utilized_Median_Growth": None
        })

        continue

    # ------------------------------------------------------------
    # Not represented
    # ------------------------------------------------------------

    records.append({
        "Pareto_Assembly_Accession": pareto_acc,
        "Pareto_Species": pareto_species,
        "Status": "NOT_REPRESENTED",
        "Matched_Assembly_Accession": None,
        "Matched_Species": None,
        "Match_Basis":
            "No exact accession or normalized species representative",
        "Phenotype_N_Strains": None,
        "Carbon_Breadth": None,
        "Nitrogen_Breadth": None,
        "Utilized_Median_Growth": None
    })


# ================================================================
# CREATE OUTPUT
# ================================================================

result = pd.DataFrame(records)

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("FINAL RECONCILIATION")
print("=" * 80)

print(
    f"\nTotal Pareto candidates: "
    f"{len(result)}"
)

print(
    "\nExact assembly matches: "
    f"{(result['Status'] == 'EXACT_ASSEMBLY_MATCH').sum()}"
)

print(
    "Species-level representatives: "
    f"{(result['Status'] == 'SPECIES_LEVEL_REPRESENTATIVE').sum()}"
)

print(
    "Ambiguous species matches: "
    f"{(result['Status'] == 'AMBIGUOUS_SPECIES_MATCH').sum()}"
)

print(
    "Not represented: "
    f"{(result['Status'] == 'NOT_REPRESENTED').sum()}"
)


# ================================================================
# SPECIAL CHECK
# ================================================================

special = result[
    result["Pareto_Species"]
    .astype(str)
    .str.contains(
        "Schwanniomyces polymorphus",
        case=False,
        na=False
    )
]

if len(special) > 0:

    print("\n" + "=" * 80)
    print("SCHWANNIOMYCES POLYMORPHUS CHECK")
    print("=" * 80)

    print(
        special[
            [
                "Pareto_Assembly_Accession",
                "Pareto_Species",
                "Status",
                "Matched_Assembly_Accession",
                "Matched_Species",
                "Match_Basis"
            ]
        ].to_string(index=False)
    )


# ================================================================
# SUMMARY TABLE
# ================================================================

summary = (
    result["Status"]
    .value_counts()
    .rename_axis("Status")
    .reset_index(name="Count")
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ================================================================
# CLEANUP
# ================================================================

ml.drop(
    columns=["_species_key"],
    inplace=True
)

pareto.drop(
    columns=["_pareto_species_key"],
    inplace=True
)


# ================================================================
# FINAL
# ================================================================

print("\n" + "=" * 80)
print("PARETO ↔ 420 RECONCILIATION COMPLETE")
print("=" * 80)

print(f"\nDetailed table:")
print(OUTPUT_FILE)

print("\nSummary:")
print(SUMMARY_FILE)

print("\n✓ Final 420 ML dataset was NOT modified.")
print("✓ Pareto candidate list was NOT modified.")
print("✓ Exact and species-level matches are explicitly distinguished.")