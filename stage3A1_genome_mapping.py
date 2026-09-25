import os
import pandas as pd
import requests
import time

# ============================================================
# STAGE 3A.1
# Map Y1000+ phenotype species to NCBI genome records
# ============================================================

INPUT_FILE = (
    r"C:\Y1000_chassis_project\results"
    r"\stage2A_multitrait_profile"
    r"\species_multitrait_profile.csv"
)

OUTPUT_DIR = (
    r"C:\Y1000_chassis_project\results"
    r"\stage3A_genome_mapping"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("STAGE 3A.1 — Y1000+ GENOME SPECIES MAPPING")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load phenotype species
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

species = (
    df["Species"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

print(
    f"\nPhenotype species: {len(species)}"
)


# ------------------------------------------------------------
# 2. NCBI E-utilities search
# ------------------------------------------------------------

BASE_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
)


def ncbi_search_species(species_name):

    params = {

        "db": "assembly",

        "term": (
            f'"{species_name}"[Organism] '
            f'AND PRJNA736342[BioProject]'
        ),

        "retmode": "json",

        "retmax": 100

    }

    try:

        response = requests.get(
            BASE_URL + "esearch.fcgi",
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        ids = (
            data
            .get("esearchresult", {})
            .get("idlist", [])
        )

        return ids

    except Exception as e:

        print(
            f"NCBI search failed for "
            f"{species_name}: {e}"
        )

        return []


# ------------------------------------------------------------
# 3. Search every phenotype species
# ------------------------------------------------------------

results = []

print(
    "\nSearching NCBI Y1000+ BioProject..."
)

for i, species_name in enumerate(
    species,
    start=1
):

    print(
        f"[{i}/{len(species)}] "
        f"{species_name}"
    )

    ids = ncbi_search_species(
        species_name
    )

    results.append({

        "Species": species_name,

        "NCBI_Assembly_Record_Count":
            len(ids),

        "NCBI_Assembly_IDs":
            ";".join(ids)

    })

    # Be polite to NCBI
    time.sleep(0.12)


# ------------------------------------------------------------
# 4. Convert to dataframe
# ------------------------------------------------------------

mapping = pd.DataFrame(results)


# ------------------------------------------------------------
# 5. Mapping status
# ------------------------------------------------------------

mapping["Genome_Matched"] = (

    mapping[
        "NCBI_Assembly_Record_Count"
    ] > 0

)


# ------------------------------------------------------------
# 6. Save mapping
# ------------------------------------------------------------

mapping_file = os.path.join(
    OUTPUT_DIR,
    "species_genome_mapping.csv"
)

mapping.to_csv(
    mapping_file,
    index=False
)


# ------------------------------------------------------------
# 7. Summary
# ------------------------------------------------------------

matched = mapping[
    "Genome_Matched"
].sum()

unmatched = (
    ~mapping[
        "Genome_Matched"
    ]
).sum()

print("\n" + "=" * 70)
print("MAPPING SUMMARY")
print("=" * 70)

print(
    f"\nPhenotype species: "
    f"{len(mapping)}"
)

print(
    f"Genome-matched species: "
    f"{matched}"
)

print(
    f"Unmatched species: "
    f"{unmatched}"
)

print(
    f"Genome coverage: "
    f"{100 * matched / len(mapping):.2f}%"
)


# ------------------------------------------------------------
# 8. Save unmatched species
# ------------------------------------------------------------

unmatched_df = mapping[
    ~mapping["Genome_Matched"]
].copy()

unmatched_file = os.path.join(
    OUTPUT_DIR,
    "unmatched_species.csv"
)

unmatched_df.to_csv(
    unmatched_file,
    index=False
)


# ------------------------------------------------------------
# 9. Save matched species
# ------------------------------------------------------------

matched_df = mapping[
    mapping["Genome_Matched"]
].copy()

matched_file = os.path.join(
    OUTPUT_DIR,
    "matched_species.csv"
)

matched_df.to_csv(
    matched_file,
    index=False
)


# ------------------------------------------------------------
# 10. Report
# ------------------------------------------------------------

report_file = os.path.join(
    OUTPUT_DIR,
    "stage3A1_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 3A.1 — Y1000+ GENOME MAPPING\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Phenotype species: "
        f"{len(mapping)}\n"
    )

    f.write(
        f"Genome-matched species: "
        f"{matched}\n"
    )

    f.write(
        f"Unmatched species: "
        f"{unmatched}\n"
    )

    f.write(
        f"Genome coverage: "
        f"{100 * matched / len(mapping):.2f}%\n"
    )


# ------------------------------------------------------------
# 11. Completion
# ------------------------------------------------------------

print("\nFiles generated:")

print(
    mapping_file
)

print(
    matched_file
)

print(
    unmatched_file
)

print(
    report_file
)

print("\nStage 3A.1 completed.")