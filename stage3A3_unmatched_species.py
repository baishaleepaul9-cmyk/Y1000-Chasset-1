import os
import time
import requests
import pandas as pd


# ============================================================
# STAGE 3A.3
# INVESTIGATION OF UNMATCHED PARETO SPECIES
# ============================================================

OUTPUT_DIR = (
    r"C:\Y1000_chassis_project\results"
    r"\stage3A3_unmatched_species"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


UNMATCHED_SPECIES = [
    "Ambrosiozyma vanderkliftii",
    "Barnettozyma californica"
]


NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


# ============================================================
# NCBI SEARCH FUNCTION
# ============================================================

def ncbi_search(term):

    url = NCBI_BASE + "esearch.fcgi"

    params = {
        "db": "assembly",
        "term": term,
        "retmode": "json",
        "retmax": 100
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    return response.json()["esearchresult"]


# ============================================================
# SEARCH STRATEGIES
# ============================================================

searches = []

for species in UNMATCHED_SPECIES:

    genus = species.split()[0]
    epithet = species.split()[1]

    queries = [
        (
            species,
            f'"{species}"[Organism]'
        ),
        (
            species + " + BioProject",
            f'"{species}"[Organism] AND PRJNA736342[BioProject]'
        ),
        (
            genus,
            f'"{genus}"[Organism]'
        ),
        (
            genus + " " + epithet,
            f'{genus}[Organism] AND {epithet}'
        )
    ]

    for label, query in queries:

        print("\n" + "-" * 70)
        print("Species:", species)
        print("Search:", label)
        print("Query:", query)

        try:

            result = ncbi_search(query)

            ids = result.get(
                "idlist",
                []
            )

            count = int(
                result.get(
                    "count",
                    0
                )
            )

            print(
                "Records found:",
                count
            )

            searches.append(
                {
                    "Target_Species": species,
                    "Search_Label": label,
                    "Query": query,
                    "Record_Count": count,
                    "Assembly_IDs": ";".join(ids)
                }
            )

        except Exception as e:

            print(
                "ERROR:",
                str(e)
            )

            searches.append(
                {
                    "Target_Species": species,
                    "Search_Label": label,
                    "Query": query,
                    "Record_Count": -1,
                    "Assembly_IDs": ""
                }
            )

        time.sleep(0.4)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame(searches)

results_file = os.path.join(
    OUTPUT_DIR,
    "unmatched_species_ncbi_search_results.csv"
)

results.to_csv(
    results_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STAGE 3A.3 SUMMARY")
print("=" * 70)

for species in UNMATCHED_SPECIES:

    subset = results[
        results["Target_Species"] == species
    ]

    print("\n" + species)

    for _, row in subset.iterrows():

        print(
            f"  {row['Search_Label']}: "
            f"{row['Record_Count']} record(s)"
        )


# ============================================================
# REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "stage3A3_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 3A.3 — UNMATCHED PARETO SPECIES INVESTIGATION\n"
    )

    f.write("=" * 70 + "\n\n")

    for species in UNMATCHED_SPECIES:

        f.write(
            f"\n{species}\n"
        )

        subset = results[
            results["Target_Species"] == species
        ]

        for _, row in subset.iterrows():

            f.write(
                f"{row['Search_Label']}: "
                f"{row['Record_Count']} record(s)\n"
            )

            if row["Assembly_IDs"]:

                f.write(
                    f"Assembly IDs: "
                    f"{row['Assembly_IDs']}\n"
                )


print("\n" + "=" * 70)
print("STAGE 3A.3 COMPLETED")
print("=" * 70)

print("\nResults saved to:")

print(results_file)

print(report_file)

print(
    "\nDo NOT download genomes yet."
)

print(
    "Use the search results to determine whether "
    "the two unmatched Pareto species have alternative "
    "NCBI assembly records."
)