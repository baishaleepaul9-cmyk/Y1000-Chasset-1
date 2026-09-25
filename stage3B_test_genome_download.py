from pathlib import Path
import csv
import time
import urllib.request
import urllib.error

# ============================================================
# Stage 3B — TEST genome acquisition
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
RESULTS_DIR = PROJECT_ROOT / "results" / "stage3B_genome_download"

TEST_N = 3

GENOME_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def read_keep_rows():
    """Read validated KEEP assemblies."""
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    keep = [
        row for row in rows
        if row.get("Validation_Decision", "").strip().upper() == "KEEP"
    ]

    return keep


def download_file(url, output_path):
    """Download one file from NCBI."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Y1000-chassis-project/1.0"
        }
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with output_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)

        return True, ""

    except Exception as e:
        return False, str(e)


def main():

    print("=" * 70)
    print("STAGE 3B — TEST GENOME ACQUISITION")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input CSV    : {INPUT_CSV}")
    print(f"Genome dir   : {GENOME_DIR}")
    print()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{INPUT_CSV}"
        )

    rows = read_keep_rows()

    print(f"KEEP assemblies found: {len(rows)}")

    if len(rows) == 0:
        raise RuntimeError("No KEEP assemblies found.")

    test_rows = rows[:TEST_N]

    print(f"Testing first {len(test_rows)} assemblies:")
    print()

    log = []

    for i, row in enumerate(test_rows, start=1):

        species = row.get("Phenotype_Species", "").strip()
        accession = (
            row.get("Assembly_Accession_NCBI", "").strip()
            or row.get("Assembly_Accession", "").strip()
        )

        print(f"[{i}/{len(test_rows)}] {species}")
        print(f"Accession: {accession}")

        if not accession.startswith(("GCA_", "GCF_")):
            print("ERROR: Invalid assembly accession")
            print()
            continue

        # NCBI assembly FTP directory
        prefix = accession.rsplit(".", 1)[0]

        # NCBI Datasets download endpoint
        url = (
            "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/"
            f"{accession}/download"
            "?include_annotation_type=GENOME_FASTA"
        )

        output_zip = GENOME_DIR / f"{accession}.zip"

        if output_zip.exists() and output_zip.stat().st_size > 0:
            print("Already downloaded; skipping.")
            status = "ALREADY_EXISTS"
            error = ""

        else:
            print("Downloading...")
            success, error = download_file(url, output_zip)

            if success:
                print(
                    f"SUCCESS: {output_zip.name} "
                    f"({output_zip.stat().st_size:,} bytes)"
                )
                status = "DOWNLOADED"
            else:
                print(f"FAILED: {error}")
                status = "FAILED"

                if output_zip.exists():
                    output_zip.unlink()

        log.append({
            "Phenotype_Species": species,
            "Assembly_Accession": accession,
            "Status": status,
            "File": str(output_zip),
            "Error": error
        })

        print()

        time.sleep(1)

    log_file = RESULTS_DIR / "stage3B_test_download_log.csv"

    with log_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Phenotype_Species",
                "Assembly_Accession",
                "Status",
                "File",
                "Error"
            ]
        )
        writer.writeheader()
        writer.writerows(log)

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Download directory: {GENOME_DIR}")
    print(f"Log file          : {log_file}")
    print()

    for item in log:
        print(
            f"{item['Phenotype_Species']} | "
            f"{item['Assembly_Accession']} | "
            f"{item['Status']}"
        )


if __name__ == "__main__":
    main()