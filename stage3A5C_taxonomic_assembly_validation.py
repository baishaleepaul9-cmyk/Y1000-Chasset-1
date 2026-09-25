# ================================================================
# STAGE 3A.5C
# TAXONOMIC + ASSEMBLY VALIDATION
# TAXID / SYNONYM-AWARE VERSION
#
# Project:
# Phylogeny-Aware Machine Learning for Multi-Constraint
# Discovery of Non-Conventional Yeast Chassis
#
# IMPORTANT:
# - No genome sequences are downloaded.
# - This stage only validates representative assemblies.
# - Taxonomic name differences are NOT automatically treated
#   as different species.
# ================================================================

import os
import re
import time
import requests
import pandas as pd
import numpy as np

# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = r"C:\Y1000_chassis_project"

REPRESENTATIVE_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage3A5_representative_genomes",
    "representative_assembly_table.csv"
)

PARETO_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage2B_pareto_analysis",
    "pareto_frontier.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage3A5C_taxonomic_assembly_validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# NCBI SETTINGS
# ================================================================

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

HEADERS = {
    "User-Agent":
        "Y1000-Chassis-Project/1.0 "
        "(research; taxonomy-validation)"
}

# ================================================================
# HELPERS
# ================================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_name(name):
    """
    Normalize an organism name for comparison.

    Removes:
      - NCBI lineage text in parentheses
      - strain identifiers
      - excessive whitespace

    Keeps the binomial name.
    """

    name = clean_text(name)

    # Remove NCBI lineage parenthetical text
    name = re.sub(r"\s*\([^)]*\)", "", name)

    # Normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()

    if not name:
        return ""

    parts = name.split()

    # Keep only first two taxonomic words
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"

    return name


def genus(name):
    name = normalize_name(name)

    if not name:
        return ""

    return name.split()[0]


def binomial(name):
    return normalize_name(name)


# ================================================================
# NCBI ESEARCH
# ================================================================

def ncbi_esearch(term, retries=3):

    params = {
        "db": "assembly",
        "term": term,
        "retmode": "json",
        "retmax": 20
    }

    for attempt in range(retries):

        try:

            r = requests.get(
                f"{NCBI_BASE}/esearch.fcgi",
                params=params,
                headers=HEADERS,
                timeout=30
            )

            r.raise_for_status()

            data = r.json()

            return data.get("esearchresult", {}).get(
                "idlist", []
            )

        except Exception as e:

            if attempt == retries - 1:
                return []

            time.sleep(2 ** attempt)

    return []


# ================================================================
# NCBI ESUMMARY
# ================================================================

def ncbi_esummary(uid_list, retries=3):

    if not uid_list:
        return []

    params = {
        "db": "assembly",
        "id": ",".join(uid_list),
        "retmode": "json"
    }

    for attempt in range(retries):

        try:

            r = requests.get(
                f"{NCBI_BASE}/esummary.fcgi",
                params=params,
                headers=HEADERS,
                timeout=60
            )

            r.raise_for_status()

            data = r.json()

            result = data.get("result", {})

            records = []

            for uid in uid_list:

                record = result.get(str(uid), {})

                if record:
                    records.append(record)

            return records

        except Exception:

            if attempt == retries - 1:
                return []

            time.sleep(2 ** attempt)

    return []


# ================================================================
# ACCESSION → NCBI METADATA
# ================================================================

def resolve_accession(accession):

    accession = clean_text(accession)

    if not accession:
        return None

    ids = ncbi_esearch(
        f"{accession}[Assembly Accession]"
    )

    if not ids:
        return None

    records = ncbi_esummary(ids[:1])

    if not records:
        return None

    rec = records[0]

    organism = (
        rec.get("organism")
        or rec.get("Organism")
        or ""
    )

    taxid = (
        rec.get("taxid")
        or rec.get("Taxid")
        or rec.get("TaxID")
        or ""
    )

    assembly_name = (
        rec.get("assemblyname")
        or rec.get("AssemblyName")
        or ""
    )

    assembly_accession = (
        rec.get("assemblyaccession")
        or rec.get("AssemblyAccession")
        or accession
    )

    return {
        "Assembly_Accession_NCBI":
            clean_text(assembly_accession),

        "NCBI_Organism":
            clean_text(organism),

        "NCBI_TaxID":
            clean_text(taxid),

        "NCBI_Assembly_Name":
            clean_text(assembly_name)
    }


# ================================================================
# TAXONOMIC CLASSIFICATION
# ================================================================

def classify_taxonomy(
    phenotype_species,
    ncbi_organism,
    ncbi_taxid
):

    phenotype = binomial(phenotype_species)
    ncbi_name = binomial(ncbi_organism)

    phenotype_genus = genus(phenotype)
    ncbi_genus = genus(ncbi_name)

    # ------------------------------------------------------------
    # Exact binomial
    # ------------------------------------------------------------

    if phenotype and ncbi_name:

        if phenotype.lower() == ncbi_name.lower():

            return (
                "SAME_SPECIES_EXACT",
                "KEEP"
            )

    # ------------------------------------------------------------
    # Known synonym / reclassification handling
    #
    # Cyberlindnera saturnus is currently represented in NCBI
    # under Williopsis saturnus.
    #
    # Same NCBI TaxID = same biological taxon.
    # ------------------------------------------------------------

    synonym_pairs = {

        (
            "cyberlindnera saturnus",
            "williopsis saturnus"
        ),

        (
            "williopsis saturnus",
            "cyberlindnera saturnus"
        ),
    }

    if (
        phenotype.lower(),
        ncbi_name.lower()
    ) in synonym_pairs:

        return (
            "SAME_SPECIES_SYNONYM",
            "KEEP"
        )

    # ------------------------------------------------------------
    # Same genus + same species epithet
    #
    # Useful for some historical taxonomic formatting.
    # ------------------------------------------------------------

    if phenotype and ncbi_name:

        p_parts = phenotype.lower().split()
        n_parts = ncbi_name.lower().split()

        if len(p_parts) >= 2 and len(n_parts) >= 2:

            if (
                p_parts[0] == n_parts[0]
                and p_parts[1] == n_parts[1]
            ):

                return (
                    "SAME_SPECIES_INFRASPECIFIC",
                    "KEEP"
                )

    # ------------------------------------------------------------
    # Same genus, different species
    #
    # Do NOT automatically exclude.
    # Keep for manual taxonomy review.
    # ------------------------------------------------------------

    if (
        phenotype_genus
        and ncbi_genus
        and phenotype_genus.lower() == ncbi_genus.lower()
    ):

        return (
            "TAXONOMIC_REVIEW",
            "REVIEW"
        )

    # ------------------------------------------------------------
    # Different genus/species
    # ------------------------------------------------------------

    if phenotype and ncbi_name:

        return (
            "DIFFERENT_TAXON",
            "EXCLUDE"
        )

    # ------------------------------------------------------------
    # Cannot classify
    # ------------------------------------------------------------

    return (
        "UNRESOLVED",
        "REVIEW"
    )


# ================================================================
# LOAD INPUTS
# ================================================================

print("=" * 70)
print("STAGE 3A.5C — TAXONOMIC + ASSEMBLY VALIDATION")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Loading representative assemblies...")

rep = pd.read_csv(REPRESENTATIVE_FILE)

print(
    f"Selected assembly rows: {len(rep)}"
)

print()
print("Loading Pareto frontier...")

pareto = pd.read_csv(PARETO_FILE)

print(
    f"Pareto rows: {len(pareto)}"
)

# ================================================================
# IDENTIFY COLUMNS
# ================================================================

def find_column(df, candidates):

    for c in candidates:

        if c in df.columns:
            return c

    return None


species_col = find_column(
    rep,
    [
        "Phenotype_Species",
        "Species",
        "phenotype_species"
    ]
)

assembly_col = find_column(
    rep,
    [
        "Selected_Assembly",
        "Assembly_Accession",
        "Assembly",
        "accession"
    ]
)

if species_col is None:
    raise ValueError(
        "Could not identify phenotype species column."
    )

if assembly_col is None:
    raise ValueError(
        "Could not identify assembly accession column."
    )

print()
print("Detected columns:")
print("Species:", species_col)
print("Assembly:", assembly_col)

# ================================================================
# UNIQUE ACCESSIONS
# ================================================================

accessions = (
    rep[assembly_col]
    .dropna()
    .astype(str)
    .str.strip()
)

accessions = sorted(
    accessions[accessions != ""].unique()
)

print()
print(
    f"Unique assembly accessions: {len(accessions)}"
)

# ================================================================
# RESOLVE ACCESSIONS
# ================================================================

metadata_records = []

print()
print("=" * 70)
print("STEP 1 — RESOLVE NCBI ASSEMBLY ACCESSIONS")
print("=" * 70)

for i, accession in enumerate(accessions, 1):

    print(
        f"[{i}/{len(accessions)}] "
        f"Resolving {accession} ...",
        end=" "
    )

    metadata = resolve_accession(accession)

    if metadata is None:

        print("FAILED")

        metadata_records.append({
            "Selected_Assembly": accession,
            "NCBI_Organism": "",
            "NCBI_TaxID": "",
            "NCBI_Assembly_Name": "",
            "Metadata_Status": "UNRESOLVED"
        })

    else:

        print(
            "OK |",
            metadata["NCBI_Organism"]
        )

        metadata["Selected_Assembly"] = accession
        metadata["Metadata_Status"] = "RESOLVED"

        metadata_records.append(metadata)

    # polite request rate
    time.sleep(0.34)


metadata_df = pd.DataFrame(metadata_records)

# Save raw metadata
metadata_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "assembly_accession_metadata_raw.csv"
    ),
    index=False
)

# ================================================================
# MERGE METADATA
# ================================================================

validated = rep.copy()

validated = validated.merge(
    metadata_df,
    left_on=assembly_col,
    right_on="Selected_Assembly",
    how="left",
    suffixes=("", "_NCBI")
)

# ================================================================
# TAXONOMIC CLASSIFICATION
# ================================================================

print()
print("=" * 70)
print("STEP 2 — TAXONOMIC CLASSIFICATION")
print("=" * 70)

tax_status = []
decisions = []

for _, row in validated.iterrows():

    status, decision = classify_taxonomy(
        row[species_col],
        row.get("NCBI_Organism", ""),
        row.get("NCBI_TaxID", "")
    )

    tax_status.append(status)
    decisions.append(decision)

validated["Taxonomic_Status"] = tax_status
validated["Validation_Decision"] = decisions

# ================================================================
# METADATA STATUS
# ================================================================

validated["Metadata_Quality_Status"] = np.where(
    validated["Metadata_Status"] == "RESOLVED",
    "METADATA_PARTIAL",
    "METADATA_UNRESOLVED"
)

# ================================================================
# SAVE COMPLETE TABLE
# ================================================================

validated_file = os.path.join(
    OUTPUT_DIR,
    "stage3A5C_validated_assemblies.csv"
)

validated.to_csv(
    validated_file,
    index=False
)

# ================================================================
# SPLIT OUTPUTS
# ================================================================

keep_df = validated[
    validated["Validation_Decision"] == "KEEP"
].copy()

review_df = validated[
    validated["Validation_Decision"] == "REVIEW"
].copy()

exclude_df = validated[
    validated["Validation_Decision"] == "EXCLUDE"
].copy()

keep_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "assemblies_KEEP.csv"
    ),
    index=False
)

review_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "assemblies_REVIEW.csv"
    ),
    index=False
)

exclude_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "assemblies_EXCLUDE.csv"
    ),
    index=False
)

# ================================================================
# TAXONOMIC REVIEW
# ================================================================

review_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "taxonomic_review_required.csv"
    ),
    index=False
)

non_exact = validated[
    validated["Taxonomic_Status"] !=
    "SAME_SPECIES_EXACT"
].copy()

non_exact.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "non_exact_taxonomic_matches.csv"
    ),
    index=False
)

# ================================================================
# PARETO VALIDATION
# ================================================================

pareto_species_col = find_column(
    pareto,
    [
        "Species",
        "Phenotype_Species",
        "species"
    ]
)

if pareto_species_col is None:

    raise ValueError(
        "Could not identify species column in Pareto file."
    )

pareto_species = set(
    pareto[pareto_species_col]
    .dropna()
    .astype(str)
    .str.strip()
)

pareto_validated = validated[
    validated[species_col]
    .astype(str)
    .str.strip()
    .isin(pareto_species)
].copy()

pareto_validated.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pareto_validated_assemblies.csv"
    ),
    index=False
)

# ================================================================
# PARETO SUMMARY
# ================================================================

pareto_keep = pareto_validated[
    pareto_validated["Validation_Decision"] == "KEEP"
]

pareto_review = pareto_validated[
    pareto_validated["Validation_Decision"] == "REVIEW"
]

pareto_exclude = pareto_validated[
    pareto_validated["Validation_Decision"] == "EXCLUDE"
]

# ================================================================
# REPORT
# ================================================================

report_lines = []

report_lines.append(
    "STAGE 3A.5C — TAXONOMIC + ASSEMBLY VALIDATION"
)

report_lines.append("=" * 70)

report_lines.append(
    f"Selected assemblies: {len(validated)}"
)

report_lines.append(
    f"Unique accessions resolved: "
    f"{validated['Metadata_Status'].eq('RESOLVED').sum()}"
)

report_lines.append(
    f"Metadata unresolved: "
    f"{validated['Metadata_Status'].eq('UNRESOLVED').sum()}"
)

report_lines.append("")

report_lines.append("TAXONOMIC STATUS")
report_lines.append("-" * 40)

for status, count in (
    validated["Taxonomic_Status"]
    .value_counts()
    .items()
):

    report_lines.append(
        f"{status}: {count}"
    )

report_lines.append("")

report_lines.append("FINAL DECISION")
report_lines.append("-" * 40)

report_lines.append(
    f"KEEP: {len(keep_df)}"
)

report_lines.append(
    f"REVIEW: {len(review_df)}"
)

report_lines.append(
    f"EXCLUDE: {len(exclude_df)}"
)

report_lines.append("")

report_lines.append("PARETO VALIDATION")
report_lines.append("-" * 40)

report_lines.append(
    f"Pareto species: {len(pareto_validated)}"
)

report_lines.append(
    f"Pareto KEEP: {len(pareto_keep)}"
)

report_lines.append(
    f"Pareto REVIEW: {len(pareto_review)}"
)

report_lines.append(
    f"Pareto EXCLUDE: {len(pareto_exclude)}"
)

report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("PARETO SPECIES")
report_lines.append("=" * 70)

for _, row in pareto_validated.iterrows():

    report_lines.append(
        f"{row[species_col]} | "
        f"{row[assembly_col]} | "
        f"{row.get('NCBI_Organism', '')} | "
        f"{row.get('NCBI_TaxID', '')} | "
        f"{row.get('Taxonomic_Status', '')} | "
        f"{row.get('Validation_Decision', '')}"
    )

report_lines.append("")
report_lines.append(
    "NO GENOME SEQUENCES WERE DOWNLOADED."
)

report_lines.append(
    "This stage only validates taxonomy and "
    "assembly metadata."
)

report_file = os.path.join(
    OUTPUT_DIR,
    "stage3A5C_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report_lines)
    )

# ================================================================
# CONSOLE SUMMARY
# ================================================================

print()
print("=" * 70)
print("STAGE 3A.5C SUMMARY")
print("=" * 70)

print()
print(
    f"Selected assemblies: {len(validated)}"
)

print(
    f"Metadata resolved: "
    f"{validated['Metadata_Status'].eq('RESOLVED').sum()}"
)

print(
    f"Metadata unresolved: "
    f"{validated['Metadata_Status'].eq('UNRESOLVED').sum()}"
)

print()
print("TAXONOMIC STATUS")

for status, count in (
    validated["Taxonomic_Status"]
    .value_counts()
    .items()
):

    print(
        f"  {status}: {count}"
    )

print()
print("FINAL DECISION")

print(
    f"  KEEP: {len(keep_df)}"
)

print(
    f"  REVIEW: {len(review_df)}"
)

print(
    f"  EXCLUDE: {len(exclude_df)}"
)

print()
print("PARETO VALIDATION")

print(
    f"  Pareto species: "
    f"{len(pareto_validated)}"
)

print(
    f"  Pareto KEEP: "
    f"{len(pareto_keep)}"
)

print(
    f"  Pareto REVIEW: "
    f"{len(pareto_review)}"
)

print(
    f"  Pareto EXCLUDE: "
    f"{len(pareto_exclude)}"
)

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(validated_file)

print(
    os.path.join(
        OUTPUT_DIR,
        "assemblies_KEEP.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "assemblies_REVIEW.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "assemblies_EXCLUDE.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "pareto_validated_assemblies.csv"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "stage3A5C_report.txt"
    )
)

print()
print("=" * 70)
print("STAGE 3A.5C COMPLETED")
print("=" * 70)

print()
print(
    "NO GENOME SEQUENCES WERE DOWNLOADED."
)