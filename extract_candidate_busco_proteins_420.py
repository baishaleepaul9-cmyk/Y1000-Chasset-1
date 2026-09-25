# ================================================================
# extract_candidate_busco_proteins_420.py
#
# PURPOSE
# -------
# Extract protein sequences for the 20 candidate BUSCOs using
# existing BUSCO results.
#
# IMPORTANT
# ---------
# This script DOES NOT:
#   - search 878,000 GFF/GTF files
#   - query NCBI
#   - require an email
#   - overwrite predictive models
#
# It uses the already recovered BUSCO mapping and searches only
# the 436 relevant BUSCO assembly directories.
# ================================================================

import os
import re
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")

BUSCO_ROOT = (
    PROJECT_ROOT
    / "results"
    / "stage4B_phylogeny"
    / "busco"
)

MAPPING_FILE = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
    / "functional_annotation_420"
    / "busco_gene_mapping_420"
    / "tables"
    / "Carbon_Breadth_BUSCO_gene_protein_mapping_420.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
    / "functional_annotation_420"
    / "busco_gene_mapping_420"
    / "candidate_proteins_420"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
FASTA_DIR = OUTPUT_ROOT / "fasta"
REPORT_DIR = OUTPUT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FASTA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def clean_string(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_filename(text):
    text = clean_string(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text


def read_fasta(path):
    """
    Read FASTA into:
        {header_without_>: sequence}
    """
    records = {}
    current = None
    seq_parts = []

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if line.startswith(">"):
                    if current is not None:
                        records[current] = "".join(seq_parts)

                    current = line[1:].strip()
                    seq_parts = []
                else:
                    seq_parts.append(line)

        if current is not None:
            records[current] = "".join(seq_parts)

    except Exception:
        return {}

    return records


def write_fasta(records, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for header, sequence in records:
            f.write(f">{header}\n")

            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i + 80] + "\n")


def fasta_header_matches_busco(header, busco_id):
    """
    BUSCO FASTA headers can differ slightly between BUSCO versions.

    Accept examples such as:
        36839at4891
        >36839at4891
        >36839at4891|...
        >...36839at4891...
    """

    header_clean = header.lstrip(">")

    if header_clean == busco_id:
        return True

    if header_clean.startswith(busco_id + "|"):
        return True

    if header_clean.startswith(busco_id + " "):
        return True

    if re.search(
        rf"(^|[|:_\s]){re.escape(busco_id)}($|[|:_\s])",
        header_clean
    ):
        return True

    return False


def find_busco_sequence_files(assembly_dir):
    """
    Search ONLY inside one relevant BUSCO assembly directory.

    No global GFF/GTF search.
    """

    candidates = []

    preferred_dirs = [
        assembly_dir / "run_saccharomycetes_odb10" / "busco_sequences",
        assembly_dir / "run_saccharomycetes_odb10" / "busco_sequences" / "single_copy_busco_sequences",
        assembly_dir / "run_saccharomycetes_odb10" / "busco_sequences" / "multi_copy_busco_sequences",
    ]

    # First search preferred BUSCO sequence directories
    for d in preferred_dirs:
        if d.exists():
            try:
                for p in d.rglob("*"):
                    if p.is_file() and p.suffix.lower() in {
                        ".faa",
                        ".fa",
                        ".fasta",
                        ".fas"
                    }:
                        candidates.append(p)
            except Exception:
                pass

    # If nothing found, search the assembly directory itself.
    if not candidates:
        try:
            for p in assembly_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in {
                    ".faa",
                    ".fa",
                    ".fasta",
                    ".fas"
                }:
                    candidates.append(p)
        except Exception:
            pass

    # Remove duplicates
    unique = []
    seen = set()

    for p in candidates:
        key = str(p).lower()

        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


# ================================================================
# START
# ================================================================

print("=" * 80)
print("CANDIDATE BUSCO PROTEIN EXTRACTION")
print("=" * 80)

print()
print("Mapping file:")
print(MAPPING_FILE)

print()
print("BUSCO root:")
print(BUSCO_ROOT)

# ================================================================
# CHECK INPUTS
# ================================================================

if not MAPPING_FILE.exists():
    raise FileNotFoundError(
        f"\nMapping file not found:\n{MAPPING_FILE}"
    )

if not BUSCO_ROOT.exists():
    raise FileNotFoundError(
        f"\nBUSCO root not found:\n{BUSCO_ROOT}"
    )

# ================================================================
# LOAD MAPPING
# ================================================================

print()
print("=" * 80)
print("LOADING EXISTING BUSCO MAPPING")
print("=" * 80)

mapping = pd.read_csv(
    MAPPING_FILE,
    low_memory=False
)

print(f"Mapping shape: {mapping.shape}")

required_columns = [
    "BUSCO_ID",
    "Species",
    "Assembly_Accession"
]

missing = [
    c for c in required_columns
    if c not in mapping.columns
]

if missing:
    raise KeyError(
        f"Missing required columns: {missing}"
    )

# ================================================================
# IDENTIFY TOP CANDIDATE BUSCOs
# ================================================================

candidate_buscos = (
    mapping["BUSCO_ID"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

print()
print("Candidate BUSCOs:")
for x in candidate_buscos:
    print(" ", x)

print()
print(f"Total candidates: {len(candidate_buscos)}")

# ================================================================
# COORDINATE-VALID RECORDS
# ================================================================

coord = mapping.copy()

if "Sequence" in coord.columns:
    coord["Sequence"] = coord["Sequence"].fillna("").astype(str).str.strip()

if "Start" in coord.columns:
    coord["Start"] = pd.to_numeric(
        coord["Start"],
        errors="coerce"
    )

if "End" in coord.columns:
    coord["End"] = pd.to_numeric(
        coord["End"],
        errors="coerce"
    )

coord_valid = coord[
    coord["BUSCO_ID"].isin(candidate_buscos)
].copy()

coord_valid = coord_valid[
    coord_valid["Sequence"].ne("")
    &
    coord_valid["Start"].notna()
    &
    coord_valid["End"].notna()
].copy()

print()
print("=" * 80)
print("VALID BUSCO RECORDS")
print("=" * 80)

print(
    f"Coordinate-valid records: {len(coord_valid)}"
)

print(
    f"Assemblies represented: "
    f"{coord_valid['Assembly_Accession'].nunique()}"
)

# ================================================================
# CREATE ASSEMBLY DIRECTORY INDEX
# ================================================================

print()
print("=" * 80)
print("INDEXING RELEVANT BUSCO ASSEMBLY DIRECTORIES")
print("=" * 80)

target_assemblies = sorted(
    coord_valid["Assembly_Accession"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)

assembly_dirs = {}

for i, assembly in enumerate(target_assemblies, 1):

    if i % 25 == 0 or i == 1 or i == len(target_assemblies):
        print(
            f"Checking assembly directories: "
            f"{i}/{len(target_assemblies)}"
        )

    # Expected pattern:
    # Species__GCA_xxxxxxxxx.x
    matches = list(
        BUSCO_ROOT.glob(f"*__{assembly}")
    )

    if matches:
        assembly_dirs[assembly] = matches[0]

print()
print(
    f"Relevant assembly directories found: "
    f"{len(assembly_dirs)}"
)

# ================================================================
# GROUP BUSCO RECORDS BY ASSEMBLY
# ================================================================

records_by_assembly = defaultdict(list)

for _, row in coord_valid.iterrows():

    assembly = clean_string(
        row["Assembly_Accession"]
    )

    if assembly:
        records_by_assembly[assembly].append(row)

# ================================================================
# EXTRACT PROTEINS
# ================================================================

print()
print("=" * 80)
print("EXTRACTING CANDIDATE BUSCO PROTEINS")
print("=" * 80)

protein_records = []
failed_records = []

processed_assemblies = 0

# Cache FASTA records so the same file isn't parsed repeatedly
fasta_cache = {}

for assembly, rows in records_by_assembly.items():

    processed_assemblies += 1

    if (
        processed_assemblies % 10 == 0
        or processed_assemblies == 1
        or processed_assemblies == len(records_by_assembly)
    ):
        print(
            f"[{processed_assemblies}/"
            f"{len(records_by_assembly)}] "
            f"{assembly}"
        )

    assembly_dir = assembly_dirs.get(assembly)

    if assembly_dir is None:

        for row in rows:
            failed_records.append({
                "BUSCO_ID": clean_string(row["BUSCO_ID"]),
                "Species": clean_string(row["Species"]),
                "Assembly_Accession": assembly,
                "Reason": "BUSCO assembly directory not found"
            })

        continue

    sequence_files = find_busco_sequence_files(
        assembly_dir
    )

    if not sequence_files:

        for row in rows:
            failed_records.append({
                "BUSCO_ID": clean_string(row["BUSCO_ID"]),
                "Species": clean_string(row["Species"]),
                "Assembly_Accession": assembly,
                "Reason": "No BUSCO protein FASTA found"
            })

        continue

    for row in rows:

        busco_id = clean_string(
            row["BUSCO_ID"]
        )

        found = False

        for fasta_file in sequence_files:

            cache_key = str(fasta_file)

            if cache_key not in fasta_cache:
                fasta_cache[cache_key] = read_fasta(
                    fasta_file
                )

            records = fasta_cache[cache_key]

            for header, sequence in records.items():

                if fasta_header_matches_busco(
                    header,
                    busco_id
                ):

                    protein_records.append({
                        "BUSCO_ID": busco_id,
                        "Species": clean_string(
                            row["Species"]
                        ),
                        "Assembly_Accession": assembly,
                        "Sequence": clean_string(
                            row["Sequence"]
                        ),
                        "Start": row["Start"],
                        "End": row["End"],
                        "Strand": clean_string(
                            row.get("Strand", "")
                        ),
                        "Protein_FASTA_Header": header,
                        "Protein_Sequence": sequence,
                        "Protein_Length": len(sequence),
                        "BUSCO_Description": clean_string(
                            row.get(
                                "Functional_Description",
                                ""
                            )
                        ),
                        "BUSCO_Run_Directory": clean_string(
                            row.get(
                                "BUSCO_Run_Directory",
                                ""
                            )
                        ),
                        "Protein_Source": str(
                            fasta_file
                        )
                    })

                    found = True
                    break

            if found:
                break

        if not found:

            failed_records.append({
                "BUSCO_ID": busco_id,
                "Species": clean_string(
                    row["Species"]
                ),
                "Assembly_Accession": assembly,
                "Reason": "BUSCO protein sequence not found"
            })

# ================================================================
# SAVE RAW PROTEIN RECORDS
# ================================================================

print()
print("=" * 80)
print("PROTEIN EXTRACTION SUMMARY")
print("=" * 80)

protein_df = pd.DataFrame(
    protein_records
)

failed_df = pd.DataFrame(
    failed_records
)

print(
    f"Protein records recovered: "
    f"{len(protein_df)}"
)

print(
    f"Failed records: "
    f"{len(failed_df)}"
)

if len(protein_df) > 0:

    protein_table = (
        TABLE_DIR
        / "Carbon_Breadth_candidate_BUSCO_proteins_420.csv"
    )

    protein_df.to_csv(
        protein_table,
        index=False
    )

else:

    protein_table = None

# ================================================================
# UNIQUE BUSCO × ASSEMBLY PROTEINS
# ================================================================

if len(protein_df) > 0:

    unique_proteins = (
        protein_df
        .drop_duplicates(
            subset=[
                "BUSCO_ID",
                "Assembly_Accession",
                "Protein_Sequence"
            ]
        )
        .copy()
    )

else:

    unique_proteins = pd.DataFrame()

print()
print(
    f"Unique BUSCO/assembly proteins: "
    f"{len(unique_proteins)}"
)

# ================================================================
# WRITE ONE MULTI-FASTA
# ================================================================

multi_fasta = (
    FASTA_DIR
    / "Carbon_Breadth_candidate_BUSCO_proteins_420.faa"
)

if len(unique_proteins) > 0:

    fasta_output = []

    for _, row in unique_proteins.iterrows():

        header = (
            f"{row['BUSCO_ID']}"
            f"|{row['Species']}"
            f"|{row['Assembly_Accession']}"
            f"|protein_length={row['Protein_Length']}"
        )

        fasta_output.append(
            (
                header,
                row["Protein_Sequence"]
            )
        )

    write_fasta(
        fasta_output,
        multi_fasta
    )

# ================================================================
# WRITE BUSCO-LEVEL SUMMARY
# ================================================================

summary_rows = []

for busco in candidate_buscos:

    sub = protein_df[
        protein_df["BUSCO_ID"] == busco
    ] if len(protein_df) > 0 else pd.DataFrame()

    summary_rows.append({
        "BUSCO_ID": busco,
        "Assemblies_With_Protein": (
            sub["Assembly_Accession"].nunique()
            if len(sub) > 0
            else 0
        ),
        "Protein_Records": len(sub),
        "Unique_Protein_Sequences": (
            sub["Protein_Sequence"]
            .nunique()
            if len(sub) > 0
            else 0
        ),
        "Mean_Protein_Length": (
            sub["Protein_Length"].mean()
            if len(sub) > 0
            else None
        ),
        "BUSCO_Description": (
            sub["BUSCO_Description"].iloc[0]
            if len(sub) > 0
            else ""
        )
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_file = (
    TABLE_DIR
    / "Carbon_Breadth_BUSCO_protein_summary_420.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

# ================================================================
# SAVE FAILED RECORDS
# ================================================================

failed_file = (
    TABLE_DIR
    / "Carbon_Breadth_BUSCO_protein_extraction_failures_420.csv"
)

failed_df.to_csv(
    failed_file,
    index=False
)

# ================================================================
# JSON SUMMARY
# ================================================================

json_summary = {
    "total_BUSCO_candidates": len(candidate_buscos),
    "coordinate_valid_records": len(coord_valid),
    "target_assemblies": len(target_assemblies),
    "assembly_directories_found": len(assembly_dirs),
    "protein_records_recovered": len(protein_df),
    "unique_BUSCO_assembly_proteins": len(unique_proteins),
    "failed_records": len(failed_df),
    "BUSCOs_with_protein": (
        int(
            protein_df["BUSCO_ID"].nunique()
        )
        if len(protein_df) > 0
        else 0
    )
}

json_file = (
    TABLE_DIR
    / "Carbon_Breadth_BUSCO_protein_extraction_summary_420.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_summary,
        f,
        indent=2
    )

# ================================================================
# REPORT
# ================================================================

report_file = (
    REPORT_DIR
    / "Carbon_Breadth_candidate_BUSCO_protein_extraction_420.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CARBON BREADTH — CANDIDATE BUSCO PROTEIN EXTRACTION\n"
    )
    f.write("=" * 80 + "\n\n")

    for key, value in json_summary.items():
        f.write(
            f"{key}: {value}\n"
        )

    f.write("\n")
    f.write(
        "IMPORTANT:\n"
    )
    f.write(
        "Protein sequences were recovered from existing BUSCO "
        "results. No NCBI query was performed.\n"
    )
    f.write(
        "No global GFF/GTF scan was performed.\n"
    )

# ================================================================
# FINAL OUTPUT
# ================================================================

print()
print("=" * 80)
print("CANDIDATE BUSCO PROTEIN EXTRACTION COMPLETE")
print("=" * 80)

print()
print(
    f"BUSCO candidates: "
    f"{len(candidate_buscos)}"
)

print(
    f"Protein records recovered: "
    f"{len(protein_df)}"
)

print(
    f"Unique BUSCO/assembly proteins: "
    f"{len(unique_proteins)}"
)

print(
    f"BUSCOs with recovered proteins: "
    f"{json_summary['BUSCOs_with_protein']}"
)

print(
    f"Failed records: "
    f"{len(failed_df)}"
)

print()
print("OUTPUTS")
print("-" * 80)

if protein_table:
    print(
        f"Protein table:\n{protein_table}"
    )

print(
    f"\nMulti-FASTA:\n{multi_fasta}"
)

print(
    f"\nBUSCO protein summary:\n{summary_file}"
)

print(
    f"\nFailed extraction records:\n{failed_file}"
)

print(
    f"\nJSON summary:\n{json_file}"
)

print(
    f"\nReport:\n{report_file}"
)

print()
print("=" * 80)
print("NEXT STEP")
print("=" * 80)

print(
    """
BUSCO
  ↓
candidate protein sequences
  ↓
protein/ortholog identification
  ↓
GO / InterPro / KEGG / EC annotation
  ↓
pathway enrichment
  ↓
functional network
  ↓
carbon-breadth chassis interpretation
"""
)

print("=" * 80)