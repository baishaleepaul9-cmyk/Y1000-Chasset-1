#!/usr/bin/env python3

"""
STAGE 4B — PHYLOGENY CONSTRUCTION
Y1000+ Non-Conventional Yeast Chassis Project

Environment:
    WSL2
    BUSCO 6.1.0
    Saccharomycetes ODB10
    MAFFT
    IQ-TREE 3

INPUT:
    Stage 4A final 436-species phylogeny manifest

OUTPUT:
    BUSCO completeness
    Single-copy BUSCO ortholog set
    Multiple sequence alignments
    Concatenated phylogenomic matrix
    IQ-TREE maximum-likelihood phylogeny
    Pareto candidate verification

IMPORTANT:
    Genome ZIP files are NEVER modified, deleted, or downloaded.

RESUME BEHAVIOR:
    - Complete BUSCO runs are skipped.
    - Incomplete BUSCO runs are cleaned and rerun.
    - Three specifically identified historical BUSCO runs are
      forcibly rerun once:
          Candida takamatsuzukensis
          Candida tropicalis
          Candida tumulicola
    - After successful rerun, those three are treated normally
      as completed results on future restarts.
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
from collections import Counter, defaultdict
import os
import sys
import csv
import math
import time
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

ROOT = Path("/mnt/c/Y1000_chassis_project")

DATA = ROOT / "data"
RESULTS = ROOT / "results"
SCRIPTS = ROOT / "scripts"

GENOMES_DIR = DATA / "genomes"

STAGE4A_DIR = (
    RESULTS /
    "stage4A_genome_annotation_inventory"
)

STAGE4B_DIR = (
    RESULTS /
    "stage4B_phylogeny"
)

GENOME_FASTA_DIR = (
    STAGE4B_DIR /
    "genome_fastas"
)

BUSCO_DIR = (
    STAGE4B_DIR /
    "busco"
)

BUSCO_LOG_DIR = (
    STAGE4B_DIR /
    "busco_logs"
)

SINGLE_COPY_DIR = (
    STAGE4B_DIR /
    "single_copy_busco_sequences"
)

ALIGNED_DIR = (
    STAGE4B_DIR /
    "aligned_busco"
)

STAGE4A_MANIFEST = (
    STAGE4A_DIR /
    "stage4A_final_436_phylogeny_input_manifest.csv"
)

STAGE4A_PARETO = (
    STAGE4A_DIR /
    "stage4A_pareto_genome_inventory.csv"
)

STAGE4A_RECONCILIATION = (
    STAGE4A_DIR /
    "stage4A_pareto_reconciliation_snapshot.csv"
)

BUSCO_MANIFEST = (
    STAGE4B_DIR /
    "stage4B_busco_run_manifest.csv"
)

BUSCO_FAILED_MANIFEST = (
    STAGE4B_DIR /
    "stage4B_busco_failed.csv"
)

REPORT = (
    STAGE4B_DIR /
    "stage4B_report.txt"
)

BUSCO_LINEAGE = (
    ROOT /
    "busco_downloads" /
    "lineages" /
    "saccharomycetes_odb10"
)


# =============================================================================
# 2. EXPECTED COHORT STRUCTURE
# =============================================================================

TOTAL_SPECIES_EXPECTED = 436

# Original Stage 2B Pareto structure:
ORIGINAL_PARETO_ASSEMBLIES_EXPECTED = 11

# After taxonomy reconciliation:
PARETO_CANONICAL_SPECIES_EXPECTED = 10

# Final retained Pareto genomes for phylogeny:
RETAINED_PARETO_GENOMES_EXPECTED = 10


# =============================================================================
# 3. COMPUTE SETTINGS
# =============================================================================

TOTAL_CPUS = os.cpu_count() or 8

# Four BUSCO jobs × two CPUs = eight CPUs
BUSCO_CPU = 2

BUSCO_WORKERS = min(
    4,
    max(
        1,
        TOTAL_CPUS // BUSCO_CPU
    )
)

MAFFT_CPU = min(
    8,
    TOTAL_CPUS
)

IQTREE_CPU = min(
    8,
    TOTAL_CPUS
)


# =============================================================================
# 4. ORTHOLOG OCCUPANCY THRESHOLD
# =============================================================================
#
# 90% of 436 = 392.4
#
# Therefore strict >=90% requires:
#
#       ceil(392.4) = 393
#
# =============================================================================

MIN_TAXON_COMPLETENESS = 0.90

MIN_TAXA_PER_ORTHOLOG = math.ceil(
    TOTAL_SPECIES_EXPECTED *
    MIN_TAXON_COMPLETENESS
)


# =============================================================================
# 5. HISTORICAL BUSCO RESULTS TO FORCE-RERUN
# =============================================================================
#
# These three results were present on disk as "EXISTING", but they belong
# to the portion of the previous run whose completion state was uncertain.
#
# They are therefore explicitly invalidated before the next BUSCO run.
#
# IMPORTANT:
# After successful rerun, they are NOT permanently forced.
# busco_is_complete() will recognize the new valid output normally.
# =============================================================================

FORCE_RERUN_SPECIES = {
    "Candida takamatsuzukensis",
    "Candida tropicalis",
    "Candida tumulicola",
}

FORCE_RERUN_ACCESSIONS = {
    "GCA_030572035.1",
    "GCA_030582595.1",
    "GCA_030578535.1",
}


# =============================================================================
# 6. GENERAL HELPERS
# =============================================================================

def die(message):

    print()
    print("=" * 80)
    print("ERROR")
    print("=" * 80)
    print(message)
    print()
    sys.exit(1)


def ensure_dirs():

    STAGE4B_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    GENOME_FASTA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    BUSCO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    BUSCO_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    SINGLE_COPY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ALIGNED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def load_csv(path):

    if not path.exists():

        die(
            f"Required file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(
            csv.DictReader(f)
        )


def write_csv(
    path,
    rows,
    fieldnames
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def run_command(
    command,
    stdout_path=None,
    stderr_path=None
):

    print()
    print("COMMAND:")
    print(
        " ".join(
            str(x)
            for x in command
        )
    )
    print()

    stdout_handle = None
    stderr_handle = None

    try:

        if stdout_path:

            stdout_handle = open(
                stdout_path,
                "w",
                encoding="utf-8"
            )

        if stderr_path:

            stderr_handle = open(
                stderr_path,
                "w",
                encoding="utf-8"
            )

        result = subprocess.run(
            command,
            stdout=(
                stdout_handle
                if stdout_handle
                else None
            ),
            stderr=(
                stderr_handle
                if stderr_handle
                else None
            ),
            check=False
        )

        return result.returncode

    finally:

        if stdout_handle:
            stdout_handle.close()

        if stderr_handle:
            stderr_handle.close()


# =============================================================================
# 7. SPECIES COLUMN DETECTION
# =============================================================================
#
# The Stage 4A manifest is authoritative, but its species field is not
# assumed to have one particular column name.
#
# The function searches plausible Stage 4A species fields and chooses one
# that is populated for all 436 rows.
# =============================================================================

def detect_species_column(rows):

    if not rows:

        die(
            "Stage 4A manifest is empty."
        )

    possible_columns = [

        "Species",

        "Canonical_Species",

        "Final_Canonical_Name",

        "Final_Canonical_Species",

        "Organism_Name",

        "NCBI_Organism",

        "Phenotype_Species",

        "Species_Name",

    ]

    for column in possible_columns:

        if column not in rows[0]:
            continue

        values = [
            str(
                row.get(
                    column,
                    ""
                )
            ).strip()

            for row in rows
        ]

        nonempty = [
            x
            for x in values
            if x
        ]

        if len(nonempty) == len(rows):

            return column

    print()
    print(
        "Available Stage 4A manifest columns:"
    )

    for column in rows[0].keys():

        print(
            f"  {column}"
        )

    die(
        "Could not identify a populated species "
        "column in the Stage 4A manifest."
    )


# =============================================================================
# 8. LOAD AND VALIDATE STAGE 4A MANIFEST
# =============================================================================

def load_phylogeny_manifest():

    rows = load_csv(
        STAGE4A_MANIFEST
    )

    if len(rows) != TOTAL_SPECIES_EXPECTED:

        die(
            f"Stage 4A manifest contains "
            f"{len(rows)} rows, expected "
            f"{TOTAL_SPECIES_EXPECTED}."
        )

    print()
    print("=" * 80)
    print("STAGE 4A INPUT VALIDATION")
    print("=" * 80)

    print(
        f"Rows loaded: {len(rows)}"
    )

    species_column = detect_species_column(
        rows
    )

    print(
        f"Species column detected: "
        f"{species_column}"
    )

    species = [
        str(
            row.get(
                species_column,
                ""
            )
        ).strip()

        for row in rows
    ]

    accessions = [
        str(
            row.get(
                "Assembly_Accession",
                ""
            )
        ).strip()

        for row in rows
    ]

    if any(
        not x
        for x in species
    ):

        die(
            f"Species column '{species_column}' "
            f"contains empty values."
        )

    if any(
        not x
        for x in accessions
    ):

        die(
            "Assembly_Accession contains "
            "empty values."
        )

    unique_species = set(
        species
    )

    unique_accessions = set(
        accessions
    )

    print(
        f"Unique species: "
        f"{len(unique_species)}"
    )

    print(
        f"Unique accessions: "
        f"{len(unique_accessions)}"
    )

    # ---------------------------------------------------------
    # Duplicate species diagnostic
    # ---------------------------------------------------------

    species_counts = Counter(
        species
    )

    duplicate_species = {
        name: count

        for name, count
        in species_counts.items()

        if count > 1
    }

    if duplicate_species:

        print()
        print(
            "Duplicate species detected:"
        )

        for name, count in sorted(
            duplicate_species.items()
        ):

            print(
                f"  {name}: {count}"
            )

    # ---------------------------------------------------------
    # Strict checks
    # ---------------------------------------------------------

    if len(unique_species) != TOTAL_SPECIES_EXPECTED:

        die(
            "Stage 4A species uniqueness check failed."
        )

    if len(unique_accessions) != TOTAL_SPECIES_EXPECTED:

        die(
            "Stage 4A accession uniqueness check failed."
        )

    # ---------------------------------------------------------
    # Normalize species name for downstream code
    # ---------------------------------------------------------
    #
    # Downstream functions expect:
    #
    #     row["Species"]
    #
    # We add that field to memory only.
    #
    # The Stage 4A CSV itself is NOT modified.
    # ---------------------------------------------------------

    for row in rows:

        row["Species"] = str(
            row[species_column]
        ).strip()

    print()
    print(
        "436-species phylogeny manifest validated."
    )

    print(
        f"Using species field: "
        f"{species_column}"
    )

    return rows


# =============================================================================
# 9. VALIDATE PARETO INPUTS
# =============================================================================

def validate_pareto():

    pareto = load_csv(
        STAGE4A_PARETO
    )

    reconciliation = load_csv(
        STAGE4A_RECONCILIATION
    )

    print()
    print("=" * 80)
    print("PARETO VALIDATION")
    print("=" * 80)

    print(
        f"Retained Pareto genome rows: "
        f"{len(pareto)}"
    )

    if len(pareto) != (
        RETAINED_PARETO_GENOMES_EXPECTED
    ):

        die(
            f"Expected "
            f"{RETAINED_PARETO_GENOMES_EXPECTED} "
            f"retained Pareto genomes, found "
            f"{len(pareto)}."
        )

    print(
        f"Reconciliation rows: "
        f"{len(reconciliation)}"
    )

    pareto_accessions = {
        str(
            row.get(
                "Assembly_Accession",
                ""
            )
        ).strip()

        for row in pareto

        if str(
            row.get(
                "Assembly_Accession",
                ""
            )
        ).strip()
    }

    print(
        f"Unique retained Pareto accessions: "
        f"{len(pareto_accessions)}"
    )

    if len(pareto_accessions) != (
        RETAINED_PARETO_GENOMES_EXPECTED
    ):

        die(
            "Pareto accession uniqueness check failed."
        )

    return pareto, reconciliation


# =============================================================================
# 10. EXTRACT GENOME FASTA
# =============================================================================

def extract_genome_fastas(
    manifest
):

    import zipfile

    print()
    print("=" * 80)
    print("GENOME FASTA EXTRACTION")
    print("=" * 80)

    extracted = 0
    already = 0
    failed = []

    for row in manifest:

        species = str(
            row.get(
                "Species",
                ""
            )
        ).strip()

        accession = str(
            row.get(
                "Assembly_Accession",
                ""
            )
        ).strip()

        if not species or not accession:

            failed.append(
                f"{species}|{accession}|"
                f"missing metadata"
            )

            continue

        safe_species = (
            species
            .replace(" ", "_")
            .replace("/", "_")
        )

        run_id = (
            safe_species
            + "__"
            + accession
        )

        output_fasta = (
            GENOME_FASTA_DIR /
            f"{run_id}.fna"
        )

        if (
            output_fasta.exists()
            and
            output_fasta.stat().st_size > 0
        ):

            already += 1

            continue

        zip_path = (
            GENOMES_DIR /
            f"{accession}.zip"
        )

        if not zip_path.exists():

            failed.append(
                f"{species}|{accession}|"
                f"ZIP missing"
            )

            continue

        try:

            with zipfile.ZipFile(
                zip_path,
                "r"
            ) as z:

                fasta_members = [

                    name

                    for name
                    in z.namelist()

                    if (
                        name.lower().endswith(
                            ".fna"
                        )
                        or
                        name.lower().endswith(
                            ".fa"
                        )
                        or
                        name.lower().endswith(
                            ".fasta"
                        )
                        or
                        name.lower().endswith(
                            ".fna.gz"
                        )
                        or
                        name.lower().endswith(
                            ".fa.gz"
                        )
                        or
                        name.lower().endswith(
                            ".fasta.gz"
                        )
                    )
                ]

                if not fasta_members:

                    failed.append(
                        f"{species}|{accession}|"
                        f"no genome FASTA in ZIP"
                    )

                    continue

                member = fasta_members[0]

                with z.open(
                    member
                ) as source, open(
                    output_fasta,
                    "wb"
                ) as destination:

                    shutil.copyfileobj(
                        source,
                        destination
                    )

                extracted += 1

        except Exception as e:

            failed.append(
                f"{species}|{accession}|"
                f"{repr(e)}"
            )

    print(
        f"Extracted: {extracted}"
    )

    print(
        f"Already present: {already}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    if failed:

        print()
        print(
            "Extraction failures:"
        )

        for failure in failed[:20]:

            print(
                " ",
                failure
            )

        die(
            f"{len(failed)} genome FASTA "
            f"extractions failed."
        )

    available = list(
        GENOME_FASTA_DIR.glob(
            "*.fna"
        )
    )

    print(
        f"Genome FASTAs available: "
        f"{len(available)}/"
        f"{TOTAL_SPECIES_EXPECTED}"
    )

    if len(available) != (
        TOTAL_SPECIES_EXPECTED
    ):

        die(
            "Genome FASTA availability check failed."
        )


# =============================================================================
# 11. BUSCO PATH HELPERS
# =============================================================================

def busco_run_id(
    species,
    accession
):

    safe_species = (
        species
        .replace(" ", "_")
        .replace("/", "_")
    )

    return (
        safe_species
        + "__"
        + accession
    )


def busco_run_dir(
    run_id
):

    return (
        BUSCO_DIR /
        run_id
    )


def busco_inner_dir(
    run_id
):

    return (
        BUSCO_DIR /
        run_id /
        "run_saccharomycetes_odb10"
    )


# =============================================================================
# 12. BUSCO COMPLETION DETECTION
# =============================================================================
#
# A BUSCO result is accepted as complete only if BOTH:
#
#   short_summary.txt
#   full_table.tsv
#
# exist and are non-empty.
# =============================================================================

def busco_is_complete(
    run_id
):

    run_dir = busco_run_dir(
        run_id
    )

    inner_dir = busco_inner_dir(
        run_id
    )

    if not run_dir.exists():
        return False

    short_summary = (
        inner_dir /
        "short_summary.txt"
    )

    full_table = (
        inner_dir /
        "full_table.tsv"
    )

    if not short_summary.exists():
        return False

    if (
        short_summary.stat().st_size
        == 0
    ):
        return False

    if not full_table.exists():
        return False

    if (
        full_table.stat().st_size
        == 0
    ):
        return False

    return True


# =============================================================================
# 13. INVALIDATE THREE UNTRUSTED RESULTS
# =============================================================================

def invalidate_untrusted_results(
    manifest
):

    print()
    print("=" * 80)
    print(
        "CHECKING UNTRUSTED HISTORICAL "
        "BUSCO RESULTS"
    )
    print("=" * 80)

    invalidated = []

    for row in manifest:

        species = str(
            row.get(
                "Species",
                ""
            )
        ).strip()

        accession = str(
            row.get(
                "Assembly_Accession",
                ""
            )
        ).strip()

        if (
            species not in
            FORCE_RERUN_SPECIES

            and

            accession not in
            FORCE_RERUN_ACCESSIONS
        ):

            continue

        run_id = busco_run_id(
            species,
            accession
        )

        run_dir = busco_run_dir(
            run_id
        )

        print()
        print(
            f"FORCED RERUN: "
            f"{species} | {accession}"
        )

        if run_dir.exists():

            print(
                "Removing old BUSCO result:"
            )

            print(
                run_dir
            )

            try:

                shutil.rmtree(
                    run_dir
                )

                invalidated.append(
                    run_id
                )

            except Exception as e:

                die(
                    f"Could not remove untrusted "
                    f"BUSCO directory:\n"
                    f"{run_dir}\n\n"
                    f"Error: {e}"
                )

        else:

            print(
                "No existing BUSCO directory. "
                "Will run normally."
            )

    print()
    print(
        f"Untrusted BUSCO results invalidated: "
        f"{len(invalidated)}"
    )

    print()
    print(
        "Genome ZIP files were NOT modified."
    )

    print(
        "Genome FASTA files were NOT modified."
    )


# =============================================================================
# 14. RUN ONE BUSCO JOB
# =============================================================================

def run_one_busco(
    row
):

    species = str(
        row.get(
            "Species",
            ""
        )
    ).strip()

    accession = str(
        row.get(
            "Assembly_Accession",
            ""
        )
    ).strip()

    run_id = busco_run_id(
        species,
        accession
    )

    fasta = (
        GENOME_FASTA_DIR /
        f"{run_id}.fna"
    )

    run_dir = busco_run_dir(
        run_id
    )

    stdout_log = (
        BUSCO_LOG_DIR /
        f"{run_id}.stdout.log"
    )

    stderr_log = (
        BUSCO_LOG_DIR /
        f"{run_id}.stderr.log"
    )

    # ---------------------------------------------------------
    # Resume check
    # ---------------------------------------------------------

    if busco_is_complete(
        run_id
    ):

        return {
            "species": species,
            "accession": accession,
            "status": "SKIPPED_COMPLETE",
            "run_id": run_id,
            "message": (
                "Existing complete BUSCO "
                "result retained."
            )
        }

    # ---------------------------------------------------------
    # Incomplete result cleanup
    # ---------------------------------------------------------

    if run_dir.exists():

        print()
        print(
            f"Removing incomplete BUSCO run:"
        )

        print(
            run_dir
        )

        try:

            shutil.rmtree(
                run_dir
            )

        except Exception as e:

            return {
                "species": species,
                "accession": accession,
                "status": "FAILED",
                "run_id": run_id,
                "message": (
                    "Could not remove incomplete "
                    f"run: {e}"
                )
            }

    # ---------------------------------------------------------
    # Verify FASTA
    # ---------------------------------------------------------

    if not fasta.exists():

        return {
            "species": species,
            "accession": accession,
            "status": "FAILED",
            "run_id": run_id,
            "message": (
                f"Genome FASTA missing: "
                f"{fasta}"
            )
        }

    # ---------------------------------------------------------
    # BUSCO command
    # ---------------------------------------------------------

    command = [

        "busco",

        "-i",
        str(fasta),

        "-o",
        run_id,

        "-m",
        "genome",

        "-l",
        str(BUSCO_LINEAGE),

        "--cpu",
        str(BUSCO_CPU),

        "--out_path",
        str(BUSCO_DIR),

        "--skip_bbtools",

        "--offline",
    ]

    start = time.time()

    return_code = run_command(
        command,
        stdout_path=stdout_log,
        stderr_path=stderr_log
    )

    elapsed = (
        time.time()
        - start
    )

    # ---------------------------------------------------------
    # Actual completion validation
    # ---------------------------------------------------------

    if (
        return_code == 0
        and
        busco_is_complete(
            run_id
        )
    ):

        return {
            "species": species,
            "accession": accession,
            "status": "COMPLETED",
            "run_id": run_id,
            "message": (
                f"BUSCO completed in "
                f"{elapsed / 60:.1f} min"
            )
        }

    return {
        "species": species,
        "accession": accession,
        "status": "FAILED",
        "run_id": run_id,
        "message": (
            f"BUSCO failed or remained "
            f"incomplete. Return code="
            f"{return_code}"
        )
    }


# =============================================================================
# 15. PARALLEL BUSCO
# =============================================================================

def run_busco_all(
    manifest
):

    print()
    print("=" * 80)
    print(
        "BUSCO — PARALLEL / OFFLINE / RESUMABLE"
    )
    print("=" * 80)

    print(
        f"Total genomes: "
        f"{len(manifest)}"
    )

    print(
        f"BUSCO CPUs/job: "
        f"{BUSCO_CPU}"
    )

    print(
        f"Parallel jobs: "
        f"{BUSCO_WORKERS}"
    )

    print(
        f"Total BUSCO CPU allocation: "
        f"{BUSCO_WORKERS * BUSCO_CPU}"
    )

    # ---------------------------------------------------------
    # Invalidate the three known-untrusted results
    # ---------------------------------------------------------

    invalidate_untrusted_results(
        manifest
    )

    # ---------------------------------------------------------
    # Determine trusted completed runs
    # ---------------------------------------------------------

    completed = []
    pending = []

    for row in manifest:

        species = str(
            row.get(
                "Species",
                ""
            )
        ).strip()

        accession = str(
            row.get(
                "Assembly_Accession",
                ""
            )
        ).strip()

        run_id = busco_run_id(
            species,
            accession
        )

        if busco_is_complete(
            run_id
        ):

            completed.append(
                row
            )

        else:

            pending.append(
                row
            )

    print()
    print(
        f"Trusted completed BUSCO runs: "
        f"{len(completed)}"
    )

    print(
        f"Remaining: "
        f"{len(pending)}"
    )

    if not pending:

        print()
        print(
            "All BUSCO runs are complete."
        )

        return

    print()
    print(
        "Starting parallel BUSCO processing..."
    )

    print(
        "Existing complete BUSCO outputs "
        "will be skipped."
    )

    print(
        "Incomplete BUSCO outputs "
        "will be restarted."
    )

    # ---------------------------------------------------------
    # Initialize manifest
    # ---------------------------------------------------------

    manifest_rows = []

    for row in manifest:

        species = str(
            row.get(
                "Species",
                ""
            )
        ).strip()

        accession = str(
            row.get(
                "Assembly_Accession",
                ""
            )
        ).strip()

        run_id = busco_run_id(
            species,
            accession
        )

        if busco_is_complete(
            run_id
        ):

            status = "EXISTING"

        else:

            status = "PENDING"

        manifest_rows.append({

            "Species":
                species,

            "Assembly_Accession":
                accession,

            "Genome_FASTA":
                str(
                    GENOME_FASTA_DIR /
                    f"{run_id}.fna"
                ),

            "BUSCO_Run_ID":
                run_id,

            "Status":
                status,

            "Message":
                "",
        })

    manifest_fields = [

        "Species",

        "Assembly_Accession",

        "Genome_FASTA",

        "BUSCO_Run_ID",

        "Status",

        "Message",
    ]

    write_csv(
        BUSCO_MANIFEST,
        manifest_rows,
        manifest_fields
    )

    # ---------------------------------------------------------
    # Counters
    # ---------------------------------------------------------

    completed_count = len(
        completed
    )

    failed_results = []

    start_time = time.time()

    # ---------------------------------------------------------
    # Parallel execution
    # ---------------------------------------------------------

    executor = ThreadPoolExecutor(
        max_workers=BUSCO_WORKERS
    )

    futures = {}

    try:

        for row in pending:

            future = executor.submit(
                run_one_busco,
                row
            )

            futures[
                future
            ] = row

        for future in as_completed(
            futures
        ):

            row = futures[
                future
            ]

            try:

                result = future.result()

            except Exception as e:

                result = {

                    "species":
                        row.get(
                            "Species",
                            ""
                        ),

                    "accession":
                        row.get(
                            "Assembly_Accession",
                            ""
                        ),

                    "status":
                        "FAILED",

                    "run_id":
                        "",

                    "message":
                        repr(e),
                }

            # -------------------------------------------------
            # Progress accounting
            # -------------------------------------------------

            if result["status"] in {

                "COMPLETED",

                "SKIPPED_COMPLETE",

            }:

                completed_count += 1

            else:

                failed_results.append(
                    result
                )

            elapsed = (
                time.time()
                -
                start_time
            )

            processed = (
                completed_count
                -
                len(completed)
            )

            if processed > 0:

                rate = (
                    processed /
                    (elapsed / 3600)
                )

                remaining = (
                    TOTAL_SPECIES_EXPECTED
                    -
                    completed_count
                )

                eta_hours = (
                    remaining / rate
                    if rate > 0
                    else float("inf")
                )

            else:

                rate = 0
                eta_hours = float("inf")

            print()
            print("=" * 80)

            print(
                f"BUSCO PROGRESS: "
                f"{completed_count}/"
                f"{TOTAL_SPECIES_EXPECTED}"
            )

            print(
                f"Completed: "
                f"{result['species']}"
            )

            print(
                f"Status: "
                f"{result['status']}"
            )

            print(
                f"Current average: "
                f"{rate:.2f} genomes/hour"
            )

            if eta_hours != float(
                "inf"
            ):

                print(
                    f"Estimated remaining: "
                    f"{eta_hours:.1f} hours"
                )

            print("=" * 80)

            # -------------------------------------------------
            # Update persistent manifest
            # -------------------------------------------------

            for manifest_row in manifest_rows:

                if (

                    manifest_row[
                        "Species"
                    ]
                    ==
                    result[
                        "species"
                    ]

                    and

                    manifest_row[
                        "Assembly_Accession"
                    ]
                    ==
                    result[
                        "accession"
                    ]

                ):

                    if (
                        result[
                            "status"
                        ]
                        ==
                        "COMPLETED"
                    ):

                        manifest_row[
                            "Status"
                        ] = "COMPLETED"

                    elif (
                        result[
                            "status"
                        ]
                        ==
                        "SKIPPED_COMPLETE"
                    ):

                        manifest_row[
                            "Status"
                        ] = "EXISTING"

                    else:

                        manifest_row[
                            "Status"
                        ] = "FAILED"

                    manifest_row[
                        "Message"
                    ] = result[
                        "message"
                    ]

                    break

            # Save immediately
            write_csv(
                BUSCO_MANIFEST,
                manifest_rows,
                manifest_fields
            )

    except KeyboardInterrupt:

        print()
        print(
            "Pipeline interrupted by user."
        )

        print(
            "Completed BUSCO outputs "
            "are preserved."
        )

        print(
            "Any currently running BUSCO "
            "jobs may need to be rerun."
        )

        executor.shutdown(
            wait=False,
            cancel_futures=True
        )

        raise

    finally:

        executor.shutdown(
            wait=True
        )

    # ---------------------------------------------------------
    # Save failed manifest
    # ---------------------------------------------------------

    if failed_results:

        failed_rows = []

        for result in failed_results:

            failed_rows.append({

                "Species":
                    result[
                        "species"
                    ],

                "Assembly_Accession":
                    result[
                        "accession"
                    ],

                "BUSCO_Run_ID":
                    result[
                        "run_id"
                    ],

                "Message":
                    result[
                        "message"
                    ],
            })

        write_csv(
            BUSCO_FAILED_MANIFEST,
            failed_rows,
            [
                "Species",
                "Assembly_Accession",
                "BUSCO_Run_ID",
                "Message",
            ]
        )

        print()
        print(
            f"WARNING: "
            f"{len(failed_results)} "
            f"BUSCO runs failed."
        )

    # ---------------------------------------------------------
    # Final BUSCO validation
    # ---------------------------------------------------------

    final_completed = []
    final_missing = []

    for row in manifest:

        species = str(
            row.get(
                "Species",
                ""
            )
        ).strip()

        accession = str(
            row.get(
                "Assembly_Accession",
                ""
            )
        ).strip()

        run_id = busco_run_id(
            species,
            accession
        )

        if busco_is_complete(
            run_id
        ):

            final_completed.append(
                row
            )

        else:

            final_missing.append(
                row
            )

    print()
    print("=" * 80)
    print("BUSCO FINAL VALIDATION")
    print("=" * 80)

    print(
        f"Complete BUSCO runs: "
        f"{len(final_completed)}/"
        f"{TOTAL_SPECIES_EXPECTED}"
    )

    print(
        f"Missing/incomplete: "
        f"{len(final_missing)}"
    )

    if final_missing:

        print()
        print(
            "Missing/incomplete genomes:"
        )

        for row in final_missing[:30]:

            print(
                " ",
                row.get(
                    "Species"
                ),
                row.get(
                    "Assembly_Accession"
                )
            )

        die(
            "BUSCO stage is incomplete. "
            "Phylogeny construction will not proceed."
        )

    print()
    print(
        "BUSCO stage complete."
    )


# =============================================================================
# 16. COLLECT SINGLE-COPY BUSCO ORTHOLOGS
# =============================================================================

def collect_single_copy():

    print()
    print("=" * 80)
    print(
        "COLLECTING SINGLE-COPY BUSCO SEQUENCES"
    )
    print("=" * 80)

    # Clean only our derived ortholog directory.
    # This does NOT touch genome files or BUSCO results.
    SINGLE_COPY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    collected = {}

    busco_runs = sorted(
        [
            d
            for d in BUSCO_DIR.iterdir()
            if d.is_dir()
        ]
    )

    for run_dir in busco_runs:

        inner_dir = (
            run_dir /
            "run_saccharomycetes_odb10"
        )

        sc_dir = (
            inner_dir /
            "single_copy_busco_sequences"
        )

        if not sc_dir.exists():

            continue

        for fasta in sc_dir.glob(
            "*.faa"
        ):

            busco_id = fasta.stem

            collected.setdefault(
                busco_id,
                []
            ).append(
                fasta
            )

    print(
        f"BUSCO families found: "
        f"{len(collected)}"
    )

    selected = {}

    for busco_id, files in (
        collected.items()
    ):

        taxa_count = len(
            files
        )

        if taxa_count >= (
            MIN_TAXA_PER_ORTHOLOG
        ):

            selected[
                busco_id
            ] = files

    print(
        f"BUSCO families meeting "
        f"{MIN_TAXON_COMPLETENESS * 100:.0f}% "
        f"occupancy: "
        f"{len(selected)}"
    )

    print(
        f"Minimum taxa required: "
        f"{MIN_TAXA_PER_ORTHOLOG}"
    )

    if not selected:

        die(
            "No BUSCO orthologs satisfy "
            "the occupancy threshold."
        )

    # ---------------------------------------------------------
    # Build normalized ortholog FASTA files
    # ---------------------------------------------------------

    for busco_id, files in (
        selected.items()
    ):

        output = (
            SINGLE_COPY_DIR /
            f"{busco_id}.faa"
        )

        with open(
            output,
            "w",
            encoding="utf-8"
        ) as handle:

            for fasta in files:

                text = fasta.read_text(
                    encoding="utf-8"
                )

                handle.write(
                    text
                )

                if not text.endswith(
                    "\n"
                ):

                    handle.write(
                        "\n"
                    )

    print(
        "Single-copy BUSCO collection complete."
    )

    return selected


# =============================================================================
# 17. MAFFT ALIGNMENT
# =============================================================================

def align_busco_families():

    print()
    print("=" * 80)
    print(
        "MULTIPLE SEQUENCE ALIGNMENT — MAFFT"
    )
    print("=" * 80)

    ALIGNED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        SINGLE_COPY_DIR.glob(
            "*.faa"
        )
    )

    print(
        f"Ortholog families: "
        f"{len(files)}"
    )

    aligned = []

    for i, fasta in enumerate(
        files,
        1
    ):

        output = (
            ALIGNED_DIR /
            fasta.name
        )

        if (
            output.exists()
            and
            output.stat().st_size > 0
        ):

            aligned.append(
                output
            )

            continue

        command = [

            "mafft",

            "--auto",

            "--thread",
            str(MAFFT_CPU),

            str(fasta),
        ]

        print(
            f"[MAFFT {i}/{len(files)}] "
            f"{fasta.name}"
        )

        try:

            result = subprocess.run(

                command,

                stdout=subprocess.PIPE,

                stderr=subprocess.PIPE,

                text=True
            )

            if result.returncode != 0:

                print(
                    result.stderr[-2000:]
                )

                die(
                    f"MAFFT failed for "
                    f"{fasta.name}"
                )

            output.write_text(
                result.stdout,
                encoding="utf-8"
            )

            aligned.append(
                output
            )

        except Exception as e:

            die(
                f"MAFFT exception for "
                f"{fasta.name}: {e}"
            )

    print(
        f"Aligned families: "
        f"{len(aligned)}"
    )

    return aligned


# =============================================================================
# 18. PARSE FASTA
# =============================================================================

def read_fasta(
    path
):

    records = []

    current_id = None
    current_seq = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:

            continue

        if line.startswith(">"):

            if current_id is not None:

                records.append(
                    (
                        current_id,
                        "".join(
                            current_seq
                        )
                    )
                )

            current_id = (
                line[1:]
                .split()[0]
            )

            current_seq = []

        else:

            current_seq.append(
                line
            )

    if current_id is not None:

        records.append(
            (
                current_id,
                "".join(
                    current_seq
                )
            )
        )

    return records


# =============================================================================
# 19. CONCATENATE ALIGNMENTS
# =============================================================================

def concatenate_alignments(
    aligned_files
):

    print()
    print("=" * 80)
    print(
        "CONCATENATING PHYLOGENOMIC MATRIX"
    )
    print("=" * 80)

    output_fasta = (
        STAGE4B_DIR /
        "concatenated_phylogenomic_matrix.faa"
    )

    partition_file = (
        STAGE4B_DIR /
        "iqtree_partitions.txt"
    )

    sequences = defaultdict(
        str
    )

    partitions = []

    current_start = 1

    for fasta in aligned_files:

        records = read_fasta(
            fasta
        )

        if not records:

            continue

        lengths = {
            len(seq)
            for _, seq in records
        }

        if len(lengths) != 1:

            die(
                f"Alignment has inconsistent "
                f"sequence lengths: "
                f"{fasta}"
            )

        length = next(
            iter(lengths)
        )

        current_end = (
            current_start
            +
            length
            -
            1
        )

        gene_name = (
            fasta.stem
        )

        partitions.append(
            (
                gene_name,
                current_start,
                current_end
            )
        )

        present_ids = set()

        for seq_id, seq in records:

            sequences[
                seq_id
            ] += seq

            present_ids.add(
                seq_id
            )

        # -----------------------------------------------------
        # Fill gaps for taxa absent from this ortholog
        # -----------------------------------------------------

        for seq_id in list(
            sequences.keys()
        ):

            if seq_id not in present_ids:

                sequences[
                    seq_id
                ] += "-" * length

        current_start = (
            current_end + 1
        )

    # ---------------------------------------------------------
    # Write concatenated matrix
    # ---------------------------------------------------------

    with open(
        output_fasta,
        "w",
        encoding="utf-8"
    ) as handle:

        for seq_id in sorted(
            sequences
        ):

            handle.write(
                f">{seq_id}\n"
            )

            seq = sequences[
                seq_id
            ]

            for i in range(
                0,
                len(seq),
                80
            ):

                handle.write(
                    seq[
                        i:i + 80
                    ]
                    +
                    "\n"
                )

    # ---------------------------------------------------------
    # Write IQ-TREE partition file
    # ---------------------------------------------------------

    with open(
        partition_file,
        "w",
        encoding="utf-8"
    ) as handle:

        for gene, start, end in (
            partitions
        ):

            handle.write(
                f"LG, {gene} = "
                f"{start}-{end}\n"
            )

    print(
        f"Taxa: "
        f"{len(sequences)}"
    )

    print(
        f"Concatenated length: "
        f"{current_start - 1}"
    )

    print(
        f"Partitions: "
        f"{len(partitions)}"
    )

    return (
        output_fasta,
        partition_file
    )


# =============================================================================
# 20. IQ-TREE
# =============================================================================

def run_iqtree(
    matrix,
    partition_file
):

    print()
    print("=" * 80)
    print(
        "IQ-TREE PHYLOGENOMIC INFERENCE"
    )
    print("=" * 80)

    prefix = (
        STAGE4B_DIR /
        "Y1000_phylogeny"
    )

    tree_file = Path(
        str(prefix)
        +
        ".treefile"
    )

    if (
        tree_file.exists()
        and
        tree_file.stat().st_size > 0
    ):

        print(
            "Existing IQ-TREE tree detected."
        )

        print(
            tree_file
        )

        return tree_file

    command = [

        "iqtree3",

        "-s",
        str(matrix),

        "-p",
        str(partition_file),

        "-m",
        "MFP+MERGE",

        "-B",
        "1000",

        "-alrt",
        "1000",

        "-T",
        str(IQTREE_CPU),

        "--prefix",
        str(prefix),
    ]

    return_code = run_command(
        command
    )

    if return_code != 0:

        die(
            "IQ-TREE failed."
        )

    if not tree_file.exists():

        die(
            "IQ-TREE completed but "
            "treefile was not created."
        )

    print(
        f"Phylogeny created:\n"
        f"{tree_file}"
    )

    return tree_file


# =============================================================================
# 21. PARETO TAXON VERIFICATION
# =============================================================================

def verify_pareto_in_tree(
    tree_file,
    pareto
):

    print()
    print("=" * 80)
    print(
        "PARETO TAXA VERIFICATION"
    )
    print("=" * 80)

    tree_text = tree_file.read_text(
        encoding="utf-8"
    )

    found = 0

    for row in pareto:

        species = str(
            row.get(
                "Species",
                ""
            )
        ).strip()

        if not species:

            species = str(
                row.get(
                    "Final_Canonical_Name",
                    ""
                )
            ).strip()

        token = (
            species
            .replace(" ", "_")
        )

        if token in tree_text:

            found += 1

        else:

            print(
                "Missing Pareto taxon:",
                species
            )

    print(
        f"Pareto taxa represented: "
        f"{found}/"
        f"{len(pareto)}"
    )

    if found != len(pareto):

        print()
        print(
            "WARNING:"
        )

        print(
            "Not all retained Pareto taxa "
            "were detected by simple name "
            "matching."
        )


# =============================================================================
# 22. FINAL REPORT
# =============================================================================

def write_report(
    manifest,
    pareto,
    busco_complete_count,
    selected_ortholog_count,
    tree_file=None
):

    lines = []

    lines.append(
        "STAGE 4B — PHYLOGENY CONSTRUCTION REPORT"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")

    lines.append(
        f"Phylogeny candidate species: "
        f"{len(manifest)}"
    )

    lines.append(
        f"Expected: "
        f"{TOTAL_SPECIES_EXPECTED}"
    )

    lines.append("")

    lines.append(
        f"BUSCO complete: "
        f"{busco_complete_count}/"
        f"{TOTAL_SPECIES_EXPECTED}"
    )

    lines.append(
        f"BUSCO lineage: "
        f"{BUSCO_LINEAGE}"
    )

    lines.append(
        f"BUSCO CPUs/job: "
        f"{BUSCO_CPU}"
    )

    lines.append(
        f"BUSCO parallel jobs: "
        f"{BUSCO_WORKERS}"
    )

    lines.append("")

    lines.append(
        f"Minimum taxa per ortholog: "
        f"{MIN_TAXA_PER_ORTHOLOG}"
    )

    lines.append(
        f"Selected ortholog families: "
        f"{selected_ortholog_count}"
    )

    lines.append("")

    lines.append(
        f"Retained Pareto genomes: "
        f"{len(pareto)}"
    )

    lines.append(
        f"Canonical Pareto species expected: "
        f"{PARETO_CANONICAL_SPECIES_EXPECTED}"
    )

    lines.append("")

    lines.append(
        "Forced historical BUSCO reruns:"
    )

    for species in sorted(
        FORCE_RERUN_SPECIES
    ):

        lines.append(
            f"  {species}"
        )

    lines.append("")

    if tree_file:

        lines.append(
            f"IQ-TREE tree: "
            f"{tree_file}"
        )

    lines.append("")

    lines.append(
        "Genome ZIP files were not modified."
    )

    lines.append(
        "Genome ZIP files were not deleted."
    )

    lines.append(
        "Genome ZIP files were not downloaded."
    )

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# =============================================================================
# 23. MAIN
# =============================================================================

def main():

    pipeline_start = time.time()

    print()
    print("=" * 80)
    print(
        "STAGE 4B — PHYLOGENY CONSTRUCTION"
    )
    print("=" * 80)

    print(
        "Y1000+ Non-Conventional Yeast "
        "Chassis Project"
    )

    print()

    print(
        f"Project root: "
        f"{ROOT}"
    )

    print(
        f"BUSCO lineage: "
        f"{BUSCO_LINEAGE}"
    )

    print(
        f"Minimum taxa per ortholog: "
        f"{MIN_TAXA_PER_ORTHOLOG}"
    )

    print(
        f"BUSCO workers: "
        f"{BUSCO_WORKERS}"
    )

    print(
        f"BUSCO CPUs/job: "
        f"{BUSCO_CPU}"
    )

    # ---------------------------------------------------------
    # Directory setup
    # ---------------------------------------------------------

    ensure_dirs()

    # ---------------------------------------------------------
    # Tool validation
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("TOOL VALIDATION")
    print("=" * 80)

    required_tools = [
        "busco",
        "mafft",
        "iqtree3",
    ]

    for tool in required_tools:

        path = shutil.which(
            tool
        )

        if path is None:

            die(
                f"Required executable not found: "
                f"{tool}"
            )

        print(
            f"{tool}: {path}"
        )

    # ---------------------------------------------------------
    # BUSCO lineage validation
    # ---------------------------------------------------------

    if not BUSCO_LINEAGE.exists():

        die(
            "BUSCO Saccharomycetes ODB10 lineage "
            f"not found:\n{BUSCO_LINEAGE}"
        )

    print()
    print(
        "BUSCO ODB10 lineage found."
    )

    # ---------------------------------------------------------
    # Stage 4A manifest
    # ---------------------------------------------------------

    manifest = (
        load_phylogeny_manifest()
    )

    # ---------------------------------------------------------
    # Pareto validation
    # ---------------------------------------------------------

    pareto, reconciliation = (
        validate_pareto()
    )

    # ---------------------------------------------------------
    # Genome FASTA extraction
    # ---------------------------------------------------------

    extract_genome_fastas(
        manifest
    )

    # ---------------------------------------------------------
    # BUSCO
    # ---------------------------------------------------------

    run_busco_all(
        manifest
    )

    # ---------------------------------------------------------
    # Single-copy BUSCO orthologs
    # ---------------------------------------------------------

    selected = (
        collect_single_copy()
    )

    # ---------------------------------------------------------
    # MAFFT
    # ---------------------------------------------------------

    aligned = (
        align_busco_families()
    )

    # ---------------------------------------------------------
    # Concatenate
    # ---------------------------------------------------------

    matrix, partition_file = (
        concatenate_alignments(
            aligned
        )
    )

    # ---------------------------------------------------------
    # IQ-TREE
    # ---------------------------------------------------------

    tree = (
        run_iqtree(
            matrix,
            partition_file
        )
    )

    # ---------------------------------------------------------
    # Pareto verification
    # ---------------------------------------------------------

    verify_pareto_in_tree(
        tree,
        pareto
    )

    # ---------------------------------------------------------
    # Final BUSCO count
    # ---------------------------------------------------------

    final_busco_count = 0

    for row in manifest:

        species = str(
            row.get(
                "Species",
                ""
            )
        ).strip()

        accession = str(
            row.get(
                "Assembly_Accession",
                ""
            )
        ).strip()

        run_id = busco_run_id(
            species,
            accession
        )

        if busco_is_complete(
            run_id
        ):

            final_busco_count += 1

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    write_report(

        manifest,

        pareto,

        final_busco_count,

        len(selected),

        tree
    )

    elapsed = (
        time.time()
        -
        pipeline_start
    )

    # ---------------------------------------------------------
    # Final console summary
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "STAGE 4B COMPLETE"
    )
    print("=" * 80)

    print()

    print(
        f"Phylogeny species: "
        f"{len(manifest)}/"
        f"{TOTAL_SPECIES_EXPECTED}"
    )

    print(
        f"BUSCO complete: "
        f"{final_busco_count}/"
        f"{TOTAL_SPECIES_EXPECTED}"
    )

    print(
        f"Ortholog families: "
        f"{len(selected)}"
    )

    print(
        f"Tree: "
        f"{tree}"
    )

    print(
        f"Report: "
        f"{REPORT}"
    )

    print(
        f"Elapsed: "
        f"{elapsed / 3600:.2f} hours"
    )

    print()

    print(
        "NO GENOME ZIP FILES WERE MODIFIED."
    )

    print(
        "NO GENOME ZIP FILES WERE DELETED."
    )

    print(
        "NO GENOME ZIP FILES WERE DOWNLOADED."
    )

    print()

    print(
        "Stage 4B completed successfully."
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()