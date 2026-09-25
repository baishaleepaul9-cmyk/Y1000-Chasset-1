import os
import pandas as pd


# ============================================================
# STAGE 3A.2
# Pareto Species — Genome Coverage Check
# ============================================================

PARETO_FILE = (
    r"C:\Y1000_chassis_project\results"
    r"\stage2B_pareto_analysis"
    r"\pareto_frontier.csv"
)

MAPPING_FILE = (
    r"C:\Y1000_chassis_project\results"
    r"\stage3A_genome_mapping"
    r"\species_genome_mapping.csv"
)

OUTPUT_DIR = (
    r"C:\Y1000_chassis_project\results"
    r"\stage3A2_pareto_genome_coverage"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD FILES
# ============================================================

print("=" * 70)
print("STAGE 3A.2 — PARETO SPECIES GENOME COVERAGE")
print("=" * 70)

print("\nLoading Pareto frontier...")

pareto = pd.read_csv(PARETO_FILE)

print(
    f"Pareto species: {len(pareto)}"
)

print("\nLoading genome mapping...")

mapping = pd.read_csv(MAPPING_FILE)

print(
    f"Mapped phenotype species: {len(mapping)}"
)


# ============================================================
# 2. SELECT RELEVANT MAPPING COLUMNS
# ============================================================

mapping_subset = mapping[
    [
        "Species",
        "NCBI_Assembly_Record_Count",
        "NCBI_Assembly_IDs",
        "Genome_Matched"
    ]
].copy()


# ============================================================
# 3. MERGE PARETO + GENOME INFORMATION
# ============================================================

coverage = pareto.merge(
    mapping_subset,
    on="Species",
    how="left"
)


# ============================================================
# 4. IDENTIFY MISSING MAPPINGS
# ============================================================

coverage["Genome_Matched"] = (
    coverage["Genome_Matched"]
    .fillna(False)
    .astype(bool)
)


# ============================================================
# 5. PRINT PARETO COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("PARETO SPECIES GENOME COVERAGE")
print("=" * 70)

display_columns = [
    "Species",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "NCBI_Assembly_Record_Count",
    "Genome_Matched"
]

print(
    coverage[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# 6. SUMMARY
# ============================================================

matched_count = coverage[
    "Genome_Matched"
].sum()

unmatched_count = (
    len(coverage)
    - matched_count
)

coverage_percentage = (
    100 * matched_count / len(coverage)
)


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"\nPareto species: "
    f"{len(coverage)}"
)

print(
    f"Pareto species with genomes: "
    f"{matched_count}"
)

print(
    f"Pareto species without genomes: "
    f"{unmatched_count}"
)

print(
    f"Pareto genome coverage: "
    f"{coverage_percentage:.2f}%"
)


# ============================================================
# 7. SAVE PARETO GENOME COVERAGE
# ============================================================

coverage_file = os.path.join(
    OUTPUT_DIR,
    "pareto_species_genome_coverage.csv"
)

coverage.to_csv(
    coverage_file,
    index=False
)


# ============================================================
# 8. SAVE MATCHED PARETO SPECIES
# ============================================================

matched = coverage[
    coverage["Genome_Matched"]
].copy()

matched_file = os.path.join(
    OUTPUT_DIR,
    "pareto_species_with_genomes.csv"
)

matched.to_csv(
    matched_file,
    index=False
)


# ============================================================
# 9. SAVE UNMATCHED PARETO SPECIES
# ============================================================

unmatched = coverage[
    ~coverage["Genome_Matched"]
].copy()

unmatched_file = os.path.join(
    OUTPUT_DIR,
    "pareto_species_without_genomes.csv"
)

unmatched.to_csv(
    unmatched_file,
    index=False
)


# ============================================================
# 10. SAVE ASSEMBLY ACCESSION TABLE
# ============================================================

assembly_table = coverage[
    coverage["Genome_Matched"]
][
    [
        "Species",
        "NCBI_Assembly_Record_Count",
        "NCBI_Assembly_IDs"
    ]
].copy()

assembly_file = os.path.join(
    OUTPUT_DIR,
    "pareto_species_assembly_records.csv"
)

assembly_table.to_csv(
    assembly_file,
    index=False
)


# ============================================================
# 11. CHECK FOR MULTIPLE ASSEMBLY RECORDS
# ============================================================

multiple_records = coverage[
    coverage[
        "NCBI_Assembly_Record_Count"
    ] > 1
].copy()

print("\n" + "=" * 70)
print("MULTIPLE ASSEMBLY RECORDS")
print("=" * 70)

print(
    f"\nPareto species with >1 "
    f"assembly record: "
    f"{len(multiple_records)}"
)

if len(multiple_records) > 0:

    print(
        multiple_records[
            [
                "Species",
                "NCBI_Assembly_Record_Count",
                "NCBI_Assembly_IDs"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 12. REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "stage3A2_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 3A.2 — PARETO SPECIES GENOME COVERAGE\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Pareto species: {len(coverage)}\n"
    )

    f.write(
        f"Genome-matched Pareto species: "
        f"{matched_count}\n"
    )

    f.write(
        f"Unmatched Pareto species: "
        f"{unmatched_count}\n"
    )

    f.write(
        f"Pareto genome coverage: "
        f"{coverage_percentage:.2f}%\n\n"
    )

    f.write(
        "Pareto species:\n\n"
    )

    f.write(
        coverage[
            display_columns
        ].to_string(index=False)
    )

    f.write("\n")


# ============================================================
# 13. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 3A.2 COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(
    coverage_file
)

print(
    matched_file
)

print(
    unmatched_file
)

print(
    assembly_file
)

print(
    report_file
)

print("\nNext step:")
print(
    "Stage 3A.3 — investigate unmatched species "
    "and finalize genomic feature source."
)
