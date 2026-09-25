#!/usr/bin/env python3

"""
STAGE 4B — POST-BUSCO PHYLOGENY CONSTRUCTION v2

IMPORTANT:
    BUSCO has ALREADY been completed for all 436 genomes.

This script DOES NOT rerun BUSCO.

It starts from the existing BUSCO output and performs:

    1. Validate 436 BUSCO single-copy sequence directories
    2. Collect single-copy BUSCO *.faa files
    3. Count BUSCO family occupancy
    4. Retain BUSCO families present in >=90% of taxa
    5. Build one multi-FASTA per qualifying BUSCO family
    6. MAFFT multiple sequence alignment
    7. Concatenate aligned BUSCO families
    8. Build partition file
    9. IQ-TREE maximum-likelihood phylogeny
   10. Verify Pareto representation
   11. Write final report

Expected phylogeny set:
    436 species
    436 unique accessions

Occupancy threshold:
    90%
    ceil(436 * 0.90) = 393 taxa

BUSCO output structure confirmed for this project:

    busco/
        Species__ACCESSION/
            run_saccharomycetes_odb10/
                busco_sequences/
                    single_copy_busco_sequences/
                        BUSCO_ID.faa
                        BUSCO_ID.gff

Genome ZIP files are NOT modified.
Genome FASTA files are NOT modified.
BUSCO results are NOT deleted.
BUSCO is NOT rerun.
"""

from pathlib import Path
import csv
import os
import re
import sys
import math
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path("/mnt/c/Y1000_chassis_project")

RESULTS = ROOT / "results"

STAGE4A_DIR = (
    RESULTS /
    "stage4A_genome_annotation_inventory"
)

STAGE4B_DIR = (
    RESULTS /
    "stage4B_phylogeny"
)

BUSCO_DIR = (
    STAGE4B_DIR /
    "busco"
)

POSTBUSCO_DIR = (
    STAGE4B_DIR /
    "postbusco_phylogeny_v2"
)

RAW_FAMILY_DIR = (
    POSTBUSCO_DIR /
    "busco_family_fastas"
)

ALIGNMENT_DIR = (
    POSTBUSCO_DIR /
    "mafft_alignments"
)

CONCAT_DIR = (
    POSTBUSCO_DIR /
    "concatenated"
)

TREE_DIR = (
    POSTBUSCO_DIR /
    "iqtree"
)


MANIFEST = (
    STAGE4A_DIR /
    "stage4A_final_436_phylogeny_input_manifest.csv"
)


# ============================================================
# 2. DATASET CONSTANTS
# ============================================================

EXPECTED_TAXA = 436

OCCUPANCY_FRACTION = 0.90

MIN_TAXA_PER_FAMILY = math.ceil(
    EXPECTED_TAXA * OCCUPANCY_FRACTION
)

# Explicit safeguard
if MIN_TAXA_PER_FAMILY != 393:

    raise RuntimeError(
        f"Occupancy calculation error: "
        f"expected 393 but got {MIN_TAXA_PER_FAMILY}"
    )


# ============================================================
# 3. COMPUTATIONAL SETTINGS
# ============================================================

TOTAL_CPUS = os.cpu_count() or 8

# MAFFT itself gets a small number of threads.
# Several independent families can therefore be aligned
# concurrently.
MAFFT_THREADS = min(
    2,
    TOTAL_CPUS
)

MAFFT_WORKERS = min(
    4,
    max(
        1,
        TOTAL_CPUS // MAFFT_THREADS
    )
)

IQTREE_THREADS = min(
    8,
    TOTAL_CPUS
)

BOOTSTRAPS = 1000

SH_ALRT = 1000


# ============================================================
# 4. BASIC HELPERS
# ============================================================

def clean_text(value):

    if value is None:
        return ""

    return str(value).strip()


def safe_taxon_name(
    species,
    accession
):

    """
    Generate a stable taxon ID.

    Example:
        Teunomyces funiuensis
        +
        GCA_030558095.1

    becomes:

        Teunomyces_funiuensis__GCA_030558095.1
    """

    species = clean_text(
        species
    )

    label = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        species
    )

    label = re.sub(
        r"_+",
        "_",
        label
    )

    label = label.strip("_")

    return (
        f"{label}__{accession}"
    )


# ============================================================
# 5. READ 436-SPECIES MANIFEST
# ============================================================

def read_manifest():

    print("\n" + "=" * 80)
    print("READING STAGE 4A PHYLOGENY MANIFEST")
    print("=" * 80)

    if not MANIFEST.exists():

        raise FileNotFoundError(
            f"Required manifest not found:\n{MANIFEST}"
        )

    rows = []

    with open(
        MANIFEST,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        for row in reader:
            rows.append(row)

    print(
        f"Rows loaded: {len(rows)}"
    )

    if len(rows) != EXPECTED_TAXA:

        raise RuntimeError(
            f"Expected {EXPECTED_TAXA} rows "
            f"but found {len(rows)}."
        )

    if not rows:

        raise RuntimeError(
            "Manifest is empty."
        )

    # --------------------------------------------------------
    # Identify accession column
    # --------------------------------------------------------

    accession_candidates = [
        "Assembly_Accession",
        "Assembly",
        "Accession",
        "Assembly_Accession_Key",
        "Stage3B2F_Assembly_Key"
    ]

    accession_col = None

    for col in accession_candidates:

        if col in rows[0]:

            accession_col = col
            break

    if accession_col is None:

        raise RuntimeError(
            "Could not identify accession column.\n"
            f"Available columns:\n"
            f"{list(rows[0].keys())}"
        )

    # --------------------------------------------------------
    # Identify species column
    # --------------------------------------------------------

    species_candidates = [
        "Canonical_Species",
        "Final_Canonical_Name",
        "Final_Canonical_Species",
        "Species",
        "Organism_Name",
        "NCBI_Organism",
        "Phenotype_Species",
        "Species_Name"
    ]

    species_col = None

    for col in species_candidates:

        if col not in rows[0]:
            continue

        values = [
            clean_text(
                row.get(col)
            )
            for row in rows
        ]

        nonempty = [
            value
            for value in values
            if value
        ]

        if len(nonempty) == len(rows):

            species_col = col
            break

    if species_col is None:

        raise RuntimeError(
            "Could not identify a fully populated "
            "species column.\n"
            f"Available columns:\n"
            f"{list(rows[0].keys())}"
        )

    print(
        f"Species column: {species_col}"
    )

    print(
        f"Accession column: {accession_col}"
    )

    # --------------------------------------------------------
    # Build accession -> species record
    # --------------------------------------------------------

    manifest = {}

    species_seen = set()

    accession_seen = set()

    for row in rows:

        species = clean_text(
            row.get(species_col)
        )

        accession = clean_text(
            row.get(accession_col)
        )

        if not species:

            raise RuntimeError(
                "Empty species encountered."
            )

        if not accession:

            raise RuntimeError(
                f"Empty accession for {species}"
            )

        if species in species_seen:

            raise RuntimeError(
                f"Duplicate species in 436-set:\n"
                f"{species}"
            )

        if accession in accession_seen:

            raise RuntimeError(
                f"Duplicate accession in 436-set:\n"
                f"{accession}"
            )

        species_seen.add(
            species
        )

        accession_seen.add(
            accession
        )

        taxon = safe_taxon_name(
            species,
            accession
        )

        manifest[accession] = {
            "species": species,
            "accession": accession,
            "taxon": taxon
        }

    print(
        f"Unique species: {len(species_seen)}"
    )

    print(
        f"Unique accessions: {len(accession_seen)}"
    )

    if len(species_seen) != EXPECTED_TAXA:

        raise RuntimeError(
            "Species uniqueness validation failed."
        )

    if len(accession_seen) != EXPECTED_TAXA:

        raise RuntimeError(
            "Accession uniqueness validation failed."
        )

    print(
        "\n436-species phylogeny manifest validated."
    )

    return manifest


# ============================================================
# 6. FIND ACTUAL BUSCO SINGLE-COPY DIRECTORY
# ============================================================

def locate_single_copy_directory(
    species,
    accession
):

    """
    Locate the actual BUSCO 6.1.0 single-copy directory.

    Confirmed structure:

    BUSCO_DIR/
        Species__ACCESSION/
            run_saccharomycetes_odb10/
                busco_sequences/
                    single_copy_busco_sequences/
    """

    taxon = safe_taxon_name(
        species,
        accession
    )

    output_dir = (
        BUSCO_DIR /
        taxon
    )

    if not output_dir.exists():

        return None

    # --------------------------------------------------------
    # Primary expected path
    # --------------------------------------------------------

    expected = (
        output_dir /
        "run_saccharomycetes_odb10" /
        "busco_sequences" /
        "single_copy_busco_sequences"
    )

    if expected.is_dir():

        return expected

    # --------------------------------------------------------
    # Fallback search.
    # This does NOT modify anything.
    # --------------------------------------------------------

    matches = list(
        output_dir.glob(
            "**/single_copy_busco_sequences"
        )
    )

    matches = [
        path
        for path in matches
        if path.is_dir()
    ]

    if len(matches) == 1:

        return matches[0]

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple single-copy BUSCO directories "
            f"found for {species}:\n"
            f"{matches}"
        )

    return None


# ============================================================
# 7. VALIDATE BUSCO SINGLE-COPY OUTPUT
# ============================================================

def validate_busco_sequence_outputs(
    manifest
):

    print("\n" + "=" * 80)
    print("VALIDATING BUSCO SINGLE-COPY SEQUENCE OUTPUT")
    print("=" * 80)

    valid_taxa = []

    missing_directories = []

    empty_directories = []

    total_faa_files = 0

    for index, (
        accession,
        info
    ) in enumerate(
        manifest.items(),
        start=1
    ):

        species = info["species"]

        single_copy_dir = (
            locate_single_copy_directory(
                species,
                accession
            )
        )

        if single_copy_dir is None:

            missing_directories.append(
                (
                    species,
                    accession
                )
            )

            continue

        faa_count = 0

        # ----------------------------------------------------
        # Efficient directory scan
        # ----------------------------------------------------

        with os.scandir(
            single_copy_dir
        ) as entries:

            for entry in entries:

                if not entry.is_file():
                    continue

                if not entry.name.endswith(".faa"):
                    continue

                faa_count += 1

        if faa_count == 0:

            empty_directories.append(
                (
                    species,
                    accession,
                    str(single_copy_dir)
                )
            )

        else:

            valid_taxa.append(
                accession
            )

            total_faa_files += faa_count

        if (
            index % 25 == 0
            or index == EXPECTED_TAXA
        ):

            print(
                f"Checked {index}/{EXPECTED_TAXA} taxa | "
                f"valid={len(valid_taxa)} | "
                f"FAA files={total_faa_files:,}"
            )

    print("\n" + "-" * 80)

    print(
        f"Taxa with single-copy BUSCO sequences: "
        f"{len(valid_taxa)}/{EXPECTED_TAXA}"
    )

    print(
        f"Total single-copy *.faa files: "
        f"{total_faa_files:,}"
    )

    print(
        f"Missing directories: "
        f"{len(missing_directories)}"
    )

    print(
        f"Empty directories: "
        f"{len(empty_directories)}"
    )

    # --------------------------------------------------------
    # Fail only if actual sequence directories are missing.
    # --------------------------------------------------------

    if missing_directories:

        print(
            "\nMISSING SINGLE-COPY DIRECTORIES:"
        )

        for species, accession in (
            missing_directories
        ):

            print(
                f"  {species} | {accession}"
            )

        raise RuntimeError(
            "One or more BUSCO single-copy directories "
            "are missing."
        )

    if empty_directories:

        print(
            "\nEMPTY SINGLE-COPY DIRECTORIES:"
        )

        for (
            species,
            accession,
            path
        ) in empty_directories:

            print(
                f"  {species} | {accession}\n"
                f"      {path}"
            )

        raise RuntimeError(
            "One or more BUSCO single-copy directories "
            "contain no .faa files."
        )

    if len(valid_taxa) != EXPECTED_TAXA:

        raise RuntimeError(
            f"Expected {EXPECTED_TAXA} taxa with "
            f"single-copy sequences but found "
            f"{len(valid_taxa)}."
        )

    print(
        "\nBUSCO single-copy sequence validation PASSED."
    )

    print(
        "BUSCO will NOT be rerun."
    )

    return total_faa_files


# ============================================================
# 8. COUNT BUSCO FAMILY OCCUPANCY
# ============================================================

def count_busco_families(
    manifest
):

    print("\n" + "=" * 80)
    print("COUNTING BUSCO FAMILY OCCUPANCY")
    print("=" * 80)

    print(
        f"Minimum taxa required: "
        f"{MIN_TAXA_PER_FAMILY}"
    )

    print(
        f"Occupancy threshold: "
        f"{OCCUPANCY_FRACTION:.0%}"
    )

    # --------------------------------------------------------
    # family -> set(accessions)
    # --------------------------------------------------------

    family_accessions = defaultdict(
        set
    )

    total_files = 0

    for index, (
        accession,
        info
    ) in enumerate(
        manifest.items(),
        start=1
    ):

        species = info["species"]

        single_copy_dir = (
            locate_single_copy_directory(
                species,
                accession
            )
        )

        with os.scandir(
            single_copy_dir
        ) as entries:

            for entry in entries:

                if not entry.is_file():
                    continue

                filename = entry.name

                if not filename.endswith(".faa"):
                    continue

                family = filename[:-4]

                family_accessions[
                    family
                ].add(
                    accession
                )

                total_files += 1

        if (
            index % 25 == 0
            or index == EXPECTED_TAXA
        ):

            print(
                f"Scanned {index}/{EXPECTED_TAXA} taxa | "
                f"files={total_files:,} | "
                f"families={len(family_accessions):,}"
            )

    # --------------------------------------------------------
    # Occupancy calculation
    # --------------------------------------------------------

    occupancy = {}

    for family, accessions in (
        family_accessions.items()
    ):

        occupancy[family] = len(
            accessions
        )

    qualifying = {
        family
        for family, count in occupancy.items()
        if count >= MIN_TAXA_PER_FAMILY
    }

    print("\n" + "-" * 80)

    print(
        f"Individual BUSCO *.faa files: "
        f"{total_files:,}"
    )

    print(
        f"Unique BUSCO families found: "
        f"{len(family_accessions):,}"
    )

    print(
        f"BUSCO families meeting 90% occupancy: "
        f"{len(qualifying):,}"
    )

    print(
        f"Minimum taxa required: "
        f"{MIN_TAXA_PER_FAMILY}"
    )

    if len(qualifying) == 0:

        raise RuntimeError(
            "Zero BUSCO families meet the 90% "
            "occupancy threshold."
        )

    # --------------------------------------------------------
    # Save occupancy table
    # --------------------------------------------------------

    occupancy_file = (
        POSTBUSCO_DIR /
        "busco_family_occupancy.csv"
    )

    with open(
        occupancy_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow([
            "BUSCO_Family",
            "Taxa_Present",
            "Total_Taxa",
            "Occupancy",
            "Pass_90pct"
        ])

        for family, count in sorted(
            occupancy.items(),
            key=lambda x: (
                -x[1],
                x[0]
            )
        ):

            writer.writerow([
                family,
                count,
                EXPECTED_TAXA,
                count / EXPECTED_TAXA,
                "YES"
                if family in qualifying
                else "NO"
            ])

    print(
        f"\nOccupancy table:\n"
        f"{occupancy_file}"
    )

    return (
        family_accessions,
        qualifying
    )


# ============================================================
# 9. READ SINGLE-RECORD FASTA
# ============================================================

def read_single_fasta(
    path
):

    header = None

    sequence_parts = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                if header is not None:

                    raise RuntimeError(
                        f"More than one FASTA "
                        f"record found in:\n{path}"
                    )

                header = line[1:].strip()

            else:

                sequence_parts.append(
                    line
                )

    if header is None:

        raise RuntimeError(
            f"No FASTA header found:\n{path}"
        )

    sequence = "".join(
        sequence_parts
    )

    if not sequence:

        raise RuntimeError(
            f"Empty FASTA sequence:\n{path}"
        )

    return header, sequence


# ============================================================
# 10. WRITE QUALIFYING BUSCO FAMILY FASTAS
# ============================================================

def write_qualifying_family_fastas(
    manifest,
    qualifying
):

    print("\n" + "=" * 80)
    print("BUILDING QUALIFYING BUSCO FAMILY FASTAS")
    print("=" * 80)

    RAW_FAMILY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # We only delete files generated by this post-BUSCO
    # stage. BUSCO output itself is untouched.
    # --------------------------------------------------------

    for old_file in RAW_FAMILY_DIR.glob(
        "*.faa"
    ):

        old_file.unlink()

    # --------------------------------------------------------
    # Build family -> accession -> source file
    # --------------------------------------------------------

    family_files = {
        family: {}
        for family in qualifying
    }

    for index, (
        accession,
        info
    ) in enumerate(
        manifest.items(),
        start=1
    ):

        species = info["species"]

        single_copy_dir = (
            locate_single_copy_directory(
                species,
                accession
            )
        )

        with os.scandir(
            single_copy_dir
        ) as entries:

            for entry in entries:

                if not entry.is_file():
                    continue

                if not entry.name.endswith(".faa"):
                    continue

                family = entry.name[:-4]

                if family not in qualifying:
                    continue

                if accession in family_files[
                    family
                ]:

                    raise RuntimeError(
                        f"Duplicate sequence for "
                        f"{family} / {accession}"
                    )

                family_files[
                    family
                ][accession] = Path(
                    entry.path
                )

        if (
            index % 25 == 0
            or index == EXPECTED_TAXA
        ):

            print(
                f"Processed {index}/{EXPECTED_TAXA} taxa"
            )

    # --------------------------------------------------------
    # Write multi-FASTA files
    # --------------------------------------------------------

    written_families = 0

    total_sequences = 0

    for family in sorted(
        qualifying
    ):

        records = family_files[
            family
        ]

        if len(records) < MIN_TAXA_PER_FAMILY:

            continue

        output = (
            RAW_FAMILY_DIR /
            f"{family}.faa"
        )

        with open(
            output,
            "w",
            encoding="utf-8"
        ) as handle:

            for accession in sorted(
                records
            ):

                info = manifest[
                    accession
                ]

                _header, sequence = (
                    read_single_fasta(
                        records[accession]
                    )
                )

                handle.write(
                    f">{info['taxon']}\n"
                )

                for start in range(
                    0,
                    len(sequence),
                    80
                ):

                    handle.write(
                        sequence[
                            start:start + 80
                        ]
                        + "\n"
                    )

                total_sequences += 1

        written_families += 1

    print(
        f"\nQualifying BUSCO family FASTAs: "
        f"{written_families:,}"
    )

    print(
        f"Sequences written: "
        f"{total_sequences:,}"
    )

    if written_families == 0:

        raise RuntimeError(
            "No qualifying BUSCO family FASTAs "
            "were produced."
        )

    return written_families


# ============================================================
# 11. MAFFT ALIGNMENT
# ============================================================

def align_one_family(
    raw_file
):

    family = raw_file.stem

    output = (
        ALIGNMENT_DIR /
        f"{family}.aln.faa"
    )

    log_file = (
        ALIGNMENT_DIR /
        f"{family}.mafft.log"
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if (
        output.exists()
        and output.stat().st_size > 0
    ):

        return (
            family,
            "SKIPPED"
        )

    command = [
        "mafft",
        "--auto",
        "--thread",
        str(MAFFT_THREADS),
        str(raw_file)
    ]

    try:

        with open(
            output,
            "w",
            encoding="utf-8"
        ) as out_handle, open(
            log_file,
            "w",
            encoding="utf-8"
        ) as log_handle:

            process = subprocess.run(
                command,
                stdout=out_handle,
                stderr=log_handle,
                text=True
            )

        if process.returncode != 0:

            if output.exists():

                output.unlink()

            return (
                family,
                "FAILED"
            )

        return (
            family,
            "COMPLETED"
        )

    except Exception:

        if output.exists():

            output.unlink()

        return (
            family,
            "FAILED"
        )


def run_mafft():

    print("\n" + "=" * 80)
    print("MAFFT MULTIPLE SEQUENCE ALIGNMENT")
    print("=" * 80)

    ALIGNMENT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_files = sorted(
        RAW_FAMILY_DIR.glob(
            "*.faa"
        )
    )

    if not raw_files:

        raise RuntimeError(
            "No qualifying BUSCO family FASTAs "
            "were found."
        )

    print(
        f"Families to align: "
        f"{len(raw_files):,}"
    )

    print(
        f"MAFFT threads/job: "
        f"{MAFFT_THREADS}"
    )

    print(
        f"Parallel MAFFT jobs: "
        f"{MAFFT_WORKERS}"
    )

    completed = 0

    skipped = 0

    failed = []

    with ThreadPoolExecutor(
        max_workers=MAFFT_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                align_one_family,
                raw_file
            ): raw_file
            for raw_file in raw_files
        }

        for future in as_completed(
            futures
        ):

            family, status = (
                future.result()
            )

            if status == "COMPLETED":

                completed += 1

            elif status == "SKIPPED":

                skipped += 1

            else:

                failed.append(
                    family
                )

            done = (
                completed
                + skipped
                + len(failed)
            )

            if (
                done % 25 == 0
                or done == len(raw_files)
            ):

                print(
                    f"MAFFT progress: "
                    f"{done}/{len(raw_files)} | "
                    f"completed={completed} | "
                    f"skipped={skipped} | "
                    f"failed={len(failed)}"
                )

    if failed:

        print(
            "\nFAILED ALIGNMENTS:"
        )

        for family in failed:

            print(
                f"  {family}"
            )

        raise RuntimeError(
            f"{len(failed)} MAFFT alignments failed."
        )

    print(
        "\nMAFFT stage complete."
    )


# ============================================================
# 12. READ ALIGNED FASTA
# ============================================================

def read_fasta(
    path
):

    records = {}

    current = None

    sequence_parts = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                if current is not None:

                    records[
                        current
                    ] = "".join(
                        sequence_parts
                    )

                current = (
                    line[1:].strip()
                )

                sequence_parts = []

            else:

                sequence_parts.append(
                    line
                )

        if current is not None:

            records[
                current
            ] = "".join(
                sequence_parts
            )

    return records


# ============================================================
# 13. CONCATENATE ALIGNMENTS
# ============================================================

def concatenate_alignments(
    manifest
):

    print("\n" + "=" * 80)
    print("CONCATENATING BUSCO ALIGNMENTS")
    print("=" * 80)

    CONCAT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    alignment_files = sorted(
        ALIGNMENT_DIR.glob(
            "*.aln.faa"
        )
    )

    if not alignment_files:

        raise RuntimeError(
            "No MAFFT alignments found."
        )

    expected_taxa = [
        info["taxon"]
        for info in manifest.values()
    ]

    concatenated = {
        taxon: []
        for taxon in expected_taxa
    }

    partitions = []

    current_position = 1

    family_count = 0

    for alignment_file in alignment_files:

        records = read_fasta(
            alignment_file
        )

        if not records:
            continue

        lengths = {
            len(sequence)
            for sequence in records.values()
        }

        if len(lengths) != 1:

            raise RuntimeError(
                f"Inconsistent sequence lengths "
                f"in alignment:\n"
                f"{alignment_file}"
            )

        alignment_length = next(
            iter(lengths)
        )

        if alignment_length == 0:
            continue

        # ----------------------------------------------------
        # Validate taxa
        # ----------------------------------------------------

        duplicate_taxa = (
            len(records)
            != len(set(records.keys()))
        )

        if duplicate_taxa:

            raise RuntimeError(
                f"Duplicate taxa in:\n"
                f"{alignment_file}"
            )

        # ----------------------------------------------------
        # Add sequence or gap.
        # ----------------------------------------------------

        for taxon in expected_taxa:

            if taxon in records:

                sequence = records[
                    taxon
                ]

            else:

                sequence = (
                    "-"
                    * alignment_length
                )

            concatenated[
                taxon
            ].append(
                sequence
            )

        start = current_position

        end = (
            current_position
            + alignment_length
            - 1
        )

        family = (
            alignment_file.name
            .replace(
                ".aln.faa",
                ""
            )
        )

        partitions.append(
            (
                family,
                start,
                end
            )
        )

        current_position = (
            end + 1
        )

        family_count += 1

    total_length = (
        current_position - 1
    )

    print(
        f"Families concatenated: "
        f"{family_count:,}"
    )

    print(
        f"Concatenated alignment length: "
        f"{total_length:,} aa"
    )

    if family_count == 0:

        raise RuntimeError(
            "No alignments were available "
            "for concatenation."
        )

    # --------------------------------------------------------
    # Write concatenated FASTA
    # --------------------------------------------------------

    concatenated_fasta = (
        CONCAT_DIR /
        "stage4B_concatenated_protein_alignment.faa"
    )

    with open(
        concatenated_fasta,
        "w",
        encoding="utf-8"
    ) as handle:

        for taxon in expected_taxa:

            sequence = "".join(
                concatenated[
                    taxon
                ]
            )

            handle.write(
                f">{taxon}\n"
            )

            for start in range(
                0,
                len(sequence),
                80
            ):

                handle.write(
                    sequence[
                        start:start + 80
                    ]
                    + "\n"
                )

    # --------------------------------------------------------
    # Write partition file
    # --------------------------------------------------------

    partition_file = (
        CONCAT_DIR /
        "stage4B_iqtree_partitions.nex"
    )

    with open(
        partition_file,
        "w",
        encoding="utf-8"
    ) as handle:

        handle.write(
            "#nexus\n\n"
        )

        handle.write(
            "begin sets;\n"
        )

        for (
            family,
            start,
            end
        ) in partitions:

            safe_family = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                family
            )

            handle.write(
                f"    charset {safe_family} = "
                f"{start}-{end};\n"
            )

        handle.write(
            "end;\n"
        )

    # --------------------------------------------------------
    # Write taxon mapping
    # --------------------------------------------------------

    mapping_file = (
        CONCAT_DIR /
        "stage4B_taxon_mapping.csv"
    )

    with open(
        mapping_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow([
            "Taxon_ID",
            "Species",
            "Assembly_Accession"
        ])

        for accession, info in (
            manifest.items()
        ):

            writer.writerow([
                info["taxon"],
                info["species"],
                accession
            ])

    print(
        f"\nConcatenated FASTA:\n"
        f"{concatenated_fasta}"
    )

    print(
        f"\nPartition file:\n"
        f"{partition_file}"
    )

    return (
        concatenated_fasta,
        partition_file
    )


# ============================================================
# 14. VERIFY PARETO REPRESENTATION
# ============================================================

def verify_pareto_representation(
    manifest
):

    print("\n" + "=" * 80)
    print("VERIFYING PARETO REPRESENTATION")
    print("=" * 80)

    pareto_file = (
        STAGE4A_DIR /
        "stage4A_pareto_genome_inventory.csv"
    )

    if not pareto_file.exists():

        print(
            "WARNING: Pareto inventory not found."
        )

        return

    rows = []

    with open(
        pareto_file,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        for row in reader:

            rows.append(row)

    if not rows:

        print(
            "WARNING: Pareto inventory is empty."
        )

        return

    accession_candidates = [
        "Assembly_Accession",
        "Accession",
        "Assembly",
        "Stage3B2F_Assembly_Key"
    ]

    accession_col = None

    for col in accession_candidates:

        if col in rows[0]:

            accession_col = col

            break

    if accession_col is None:

        print(
            "WARNING: Could not identify "
            "Pareto accession column."
        )

        return

    pareto_accessions = {
        clean_text(
            row.get(accession_col)
        )
        for row in rows
        if clean_text(
            row.get(accession_col)
        )
    }

    represented = (
        pareto_accessions
        &
        set(manifest.keys())
    )

    print(
        f"Pareto assemblies represented: "
        f"{len(represented)}/"
        f"{len(pareto_accessions)}"
    )

    for accession in sorted(
        pareto_accessions
    ):

        if accession in represented:

            print(
                f"  ✓ "
                f"{manifest[accession]['species']} "
                f"| {accession}"
            )

        else:

            print(
                f"  ✗ NOT REPRESENTED: "
                f"{accession}"
            )

    if (
        len(represented)
        != len(pareto_accessions)
    ):

        raise RuntimeError(
            "Not all retained Pareto assemblies "
            "are represented in the phylogeny set."
        )

    print(
        "\nPareto representation validated."
    )


# ============================================================
# 15. IQ-TREE
# ============================================================

def run_iqtree(
    concatenated_fasta,
    partition_file
):

    print("\n" + "=" * 80)
    print("IQ-TREE MAXIMUM-LIKELIHOOD PHYLOGENY")
    print("=" * 80)

    TREE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prefix = (
        TREE_DIR /
        "Y1000_phylogeny"
    )

    tree_file = Path(
        str(prefix)
        + ".treefile"
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    if (
        tree_file.exists()
        and tree_file.stat().st_size > 0
    ):

        print(
            "Existing IQ-TREE tree detected."
        )

        print(
            f"{tree_file}"
        )

        print(
            "IQ-TREE will be skipped."
        )

        return tree_file

    command = [
        "iqtree3",

        "-s",
        str(concatenated_fasta),

        "-p",
        str(partition_file),

        "-m",
        "MFP+MERGE",

        "-B",
        str(BOOTSTRAPS),

        "--alrt",
        str(SH_ALRT),

        "-T",
        str(IQTREE_THREADS),

        "--prefix",
        str(prefix),

        "-redo"
    ]

    log_file = (
        TREE_DIR /
        "iqtree.log"
    )

    print(
        f"IQ-TREE threads: "
        f"{IQTREE_THREADS}"
    )

    print(
        f"UFBoot replicates: "
        f"{BOOTSTRAPS}"
    )

    print(
        f"SH-aLRT replicates: "
        f"{SH_ALRT}"
    )

    run_command(
        command,
        log_file
    )

    if not tree_file.exists():

        raise RuntimeError(
            "IQ-TREE did not produce the "
            "expected .treefile."
        )

    print(
        "\nIQ-TREE completed."
    )

    print(
        f"Final tree:\n{tree_file}"
    )

    return tree_file


# ============================================================
# 16. FINAL REPORT
# ============================================================

def write_report(
    manifest,
    qualifying,
    tree_file
):

    report = (
        POSTBUSCO_DIR /
        "stage4B_postbusco_phylogeny_report.txt"
    )

    alignment_count = len(
        list(
            ALIGNMENT_DIR.glob(
                "*.aln.faa"
            )
        )
    )

    with open(
        report,
        "w",
        encoding="utf-8"
    ) as handle:

        handle.write(
            "STAGE 4B — POST-BUSCO PHYLOGENY\n"
        )

        handle.write(
            "=" * 80
            + "\n\n"
        )

        handle.write(
            f"Phylogeny taxa: "
            f"{len(manifest)}\n"
        )

        handle.write(
            f"Occupancy threshold: "
            f"{OCCUPANCY_FRACTION:.0%}\n"
        )

        handle.write(
            f"Minimum taxa/family: "
            f"{MIN_TAXA_PER_FAMILY}\n"
        )

        handle.write(
            f"Qualifying BUSCO families: "
            f"{len(qualifying)}\n"
        )

        handle.write(
            f"MAFFT alignments: "
            f"{alignment_count}\n"
        )

        handle.write(
            f"IQ-TREE tree: "
            f"{tree_file}\n"
        )

        handle.write(
            "\n"
        )

        handle.write(
            "BUSCO was NOT rerun.\n"
        )

        handle.write(
            "Existing BUSCO results were NOT modified.\n"
        )

        handle.write(
            "Genome ZIP files were NOT modified.\n"
        )

        handle.write(
            "Genome FASTA files were NOT modified.\n"
        )

    print(
        f"\nFinal report:\n{report}"
    )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print(
        "STAGE 4B — POST-BUSCO PHYLOGENY CONSTRUCTION v2"
    )
    print("=" * 80)

    print(
        "\nBUSCO STATUS:"
    )

    print(
        "    436/436 BUSCO runs already completed"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "    BUSCO WILL NOT BE RERUN."
    )

    print(
        "    Existing BUSCO output WILL NOT be deleted."
    )

    print(
        "\nExpected taxa:"
        f" {EXPECTED_TAXA}"
    )

    print(
        "Required occupancy:"
        f" {OCCUPANCY_FRACTION:.0%}"
    )

    print(
        "Minimum taxa per BUSCO family:"
        f" {MIN_TAXA_PER_FAMILY}"
    )

    # --------------------------------------------------------
    # Create derived output directories
    # --------------------------------------------------------

    POSTBUSCO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RAW_FAMILY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ALIGNMENT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CONCAT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TREE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Stage A — 436-species manifest
    # --------------------------------------------------------

    manifest = read_manifest()

    # --------------------------------------------------------
    # Stage B — Validate actual BUSCO sequence output
    # --------------------------------------------------------

    total_faa = (
        validate_busco_sequence_outputs(
            manifest
        )
    )

    # --------------------------------------------------------
    # Stage C — Count BUSCO families
    # --------------------------------------------------------

    (
        family_accessions,
        qualifying
    ) = count_busco_families(
        manifest
    )

    # --------------------------------------------------------
    # Stage D — Write qualifying families
    # --------------------------------------------------------

    write_qualifying_family_fastas(
        manifest,
        qualifying
    )

    # --------------------------------------------------------
    # Stage E — Pareto verification
    # --------------------------------------------------------

    verify_pareto_representation(
        manifest
    )

    # --------------------------------------------------------
    # Stage F — MAFFT
    # --------------------------------------------------------

    run_mafft()

    # --------------------------------------------------------
    # Stage G — Concatenation
    # --------------------------------------------------------

    (
        concatenated_fasta,
        partition_file
    ) = concatenate_alignments(
        manifest
    )

    # --------------------------------------------------------
    # Stage H — IQ-TREE
    # --------------------------------------------------------

    tree_file = run_iqtree(
        concatenated_fasta,
        partition_file
    )

    # --------------------------------------------------------
    # Stage I — Report
    # --------------------------------------------------------

    write_report(
        manifest,
        qualifying,
        tree_file
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print(
        "STAGE 4B PHYLOGENY COMPLETE"
    )
    print("=" * 80)

    print(
        f"\n436-species phylogeny: COMPLETE"
    )

    print(
        f"Single-copy BUSCO *.faa files scanned: "
        f"{total_faa:,}"
    )

    print(
        f"Qualifying BUSCO families: "
        f"{len(qualifying):,}"
    )

    print(
        f"\nFinal tree:"
    )

    print(
        tree_file
    )

    print(
        "\nNext major stage:"
    )

    print(
        "STAGE 5 — PHYLOGENY-AWARE ML"
    )

    print(
        "\nNO BUSCO RUNS WERE RERUN."
    )

    print(
        "NO BUSCO RESULTS WERE DELETED."
    )

    print(
        "NO GENOME ZIP FILES WERE MODIFIED."
    )

    print(
        "NO GENOME FASTA FILES WERE MODIFIED."
    )

    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nPipeline interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print(
            "\n\n" + "=" * 80
        )

        print(
            "ERROR"
        )

        print(
            "=" * 80
        )

        print(
            str(exc)
        )

        sys.exit(1)