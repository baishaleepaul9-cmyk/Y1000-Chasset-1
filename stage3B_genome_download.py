from pathlib import Path
import csv
import time
import urllib.request
import urllib.error
import sys

# ============================================================
# STAGE 3B — FULL GENOME ACQUISITION
# Project: Y1000_chassis_project
# ============================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")

INPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "stage3A5C_taxonomic_assembly_validation"
    / "assemblies_KEEP.csv"
)

GENOME_DIR = PROJECT_ROOT / "data" / "genomes"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage3B_genome_download"
)

LOG_FILE = RESULTS_DIR / "stage3B_genome_download_log.csv"
SUMMARY_FILE = RESULTS_DIR / "stage3B_genome_download_summary.txt"

GENOME_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_keep_rows():
    """Read all assemblies with Validation_Decision = KEEP."""

    with INPUT_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(csv.DictReader(f))

    keep = []

    for row in rows:

        decision = (
            row.get("Validation_Decision", "")
            .strip()
            .upper()
        )

        if decision == "KEEP":
            keep.append(row)

    return keep


def download_file(url, output_path):

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Y1000-chassis-project/1.0"
        }
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=300
        ) as response:

            with output_path.open("wb") as out:

                while True:

                    chunk = response.read(
                        1024 * 1024
                    )

                    if not chunk:
                        break

                    out.write(chunk)

        if not output_path.exists():
            return False, "Output file was not created."

        size = output_path.stat().st_size

        if size == 0:
            output_path.unlink()
            return False, "Downloaded file has zero size."

        return True, ""

    except Exception as e:

        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                pass

        return False, str(e)


def main():

    print("=" * 75)
    print("STAGE 3B — FULL GENOME ACQUISITION")
    print("=" * 75)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input CSV    : {INPUT_CSV}")
    print(f"Genome dir   : {GENOME_DIR}")
    print(f"Log file     : {LOG_FILE}")
    print()

    if not INPUT_CSV.exists():

        print("ERROR: Input CSV not found:")
        print(INPUT_CSV)
        sys.exit(1)

    rows = read_keep_rows()

    print(f"KEEP assemblies: {len(rows)}")

    if not rows:

        print("ERROR: No KEEP assemblies found.")
        sys.exit(1)

    # --------------------------------------------------------
    # Load existing log if present
    # --------------------------------------------------------

    existing_log = {}

    if LOG_FILE.exists():

        try:

            with LOG_FILE.open(
                "r",
                encoding="utf-8-sig",
                newline=""
            ) as f:

                for row in csv.DictReader(f):

                    accession = row.get(
                        "Assembly_Accession",
                        ""
                    ).strip()

                    if accession:
                        existing_log[accession] = row

        except Exception as e:

            print(
                f"WARNING: Could not read existing log: {e}"
            )

    log_rows = []

    total = len(rows)

    downloaded = 0
    already_exists = 0
    failed = 0
    skipped_invalid = 0

    start_time = time.time()

    for index, row in enumerate(
        rows,
        start=1
    ):

        species = row.get(
            "Phenotype_Species",
            ""
        ).strip()

        accession = (
            row.get(
                "Assembly_Accession_NCBI",
                ""
            ).strip()
            or
            row.get(
                "Assembly_Accession",
                ""
            ).strip()
        )

        pareto = row.get(
            "Pareto_Optimal",
            ""
        ).strip()

        taxonomic_status = row.get(
            "Taxonomic_Status",
            ""
        ).strip()

        print()
        print("-" * 75)
        print(
            f"[{index}/{total}] {species}"
        )
        print(
            f"Accession: {accession}"
        )

        if not accession.startswith(
            ("GCA_", "GCF_")
        ):

            print(
                "SKIPPED: Invalid accession."
            )

            skipped_invalid += 1

            log_rows.append({
                "Phenotype_Species": species,
                "Assembly_Accession": accession,
                "Pareto_Optimal": pareto,
                "Taxonomic_Status": taxonomic_status,
                "Status": "INVALID_ACCESSION",
                "File": "",
                "File_Size_Bytes": "",
                "Error": "Invalid accession"
            })

            continue

        output_zip = (
            GENOME_DIR
            / f"{accession}.zip"
        )

        # ----------------------------------------------------
        # Skip already downloaded valid files
        # ----------------------------------------------------

        if (
            output_zip.exists()
            and output_zip.stat().st_size > 0
        ):

            size = output_zip.stat().st_size

            print(
                f"ALREADY EXISTS: "
                f"{size:,} bytes"
            )

            already_exists += 1

            status = "ALREADY_EXISTS"
            error = ""

        else:

            url = (
                "https://api.ncbi.nlm.nih.gov/"
                "datasets/v2alpha/genome/accession/"
                f"{accession}/download"
                "?include_annotation_type=GENOME_FASTA"
            )

            print("Downloading...")

            success, error = download_file(
                url,
                output_zip
            )

            if success:

                size = output_zip.stat().st_size

                print(
                    f"SUCCESS: {size:,} bytes"
                )

                downloaded += 1
                status = "DOWNLOADED"

            else:

                size = 0

                print(
                    f"FAILED: {error}"
                )

                failed += 1
                status = "FAILED"

        log_rows.append({
            "Phenotype_Species": species,
            "Assembly_Accession": accession,
            "Pareto_Optimal": pareto,
            "Taxonomic_Status": taxonomic_status,
            "Status": status,
            "File": str(output_zip)
            if output_zip.exists()
            else "",
            "File_Size_Bytes": size,
            "Error": error
        })

        # ----------------------------------------------------
        # Save progress after EVERY genome
        # ----------------------------------------------------

        with LOG_FILE.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            fieldnames = [
                "Phenotype_Species",
                "Assembly_Accession",
                "Pareto_Optimal",
                "Taxonomic_Status",
                "Status",
                "File",
                "File_Size_Bytes",
                "Error"
            ]

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(log_rows)

        # Small delay to avoid hammering NCBI
        time.sleep(1)

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    valid_files = []

    for row in rows:

        accession = (
            row.get(
                "Assembly_Accession_NCBI",
                ""
            ).strip()
            or
            row.get(
                "Assembly_Accession",
                ""
            ).strip()
        )

        if not accession:
            continue

        file_path = (
            GENOME_DIR
            / f"{accession}.zip"
        )

        if (
            file_path.exists()
            and file_path.stat().st_size > 0
        ):

            valid_files.append(accession)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = [
        "STAGE 3B — GENOME ACQUISITION SUMMARY",
        "=" * 60,
        "",
        f"KEEP assemblies in input : {len(rows)}",
        f"Downloaded this run       : {downloaded}",
        f"Already existed           : {already_exists}",
        f"Failed                    : {failed}",
        f"Invalid accessions        : {skipped_invalid}",
        f"Valid genome ZIP files    : {len(valid_files)}",
        "",
        f"Elapsed time (seconds)    : {elapsed:.1f}",
        "",
        f"Genome directory:",
        str(GENOME_DIR),
        "",
        f"Download log:",
        str(LOG_FILE),
    ]

    SUMMARY_FILE.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print()
    print("=" * 75)
    print("STAGE 3B COMPLETE")
    print("=" * 75)

    for line in summary:
        print(line)


if __name__ == "__main__":
    main()