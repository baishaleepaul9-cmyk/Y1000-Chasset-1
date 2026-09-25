from pathlib import Path
import pandas as pd
import sys
import re


# =============================================================================
# Y1000+ BUSCO GENOMIC FEATURE MATRIX
# =============================================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

STAGE4B = PROJECT / "results" / "stage4B_phylogeny"

MANIFEST_FILE = (
    STAGE4B / "stage4B_genome_fasta_manifest.csv"
)

BUSCO_ROOT = (
    STAGE4B / "busco"
)

OUTPUT_DIR = (
    STAGE4B
    / "postbusco_phylogeny_v2"
    / "genomic_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXPECTED_GENOMES = 436

VALID_STATUSES = {
    "Complete",
    "Duplicated",
    "Fragmented",
    "Missing"
}


# =============================================================================
# ERROR HANDLER
# =============================================================================

def stop(message):

    print()
    print("=" * 80)
    print("ERROR")
    print("=" * 80)
    print(message)
    print()

    sys.exit(1)


# =============================================================================
# EXTRACT ACCESSION FROM BUSCO DIRECTORY
# =============================================================================

def accession_from_path(path):

    """
    Expected BUSCO structure:

    busco/
        Species_name__GCA_XXXXXXXXX.X/
            run_saccharomycetes_odb10/
                full_table.tsv

    Extracts GCA/GCF accession from the parent genome directory.
    """

    text = str(path)

    match = re.search(
        r"(GC[AF]_\d+\.\d+)",
        text
    )

    if match:
        return match.group(1)

    return None


# =============================================================================
# READ BUSCO TABLE
# =============================================================================

def read_busco_table(path):

    """
    BUSCO full_table.tsv is read as a raw tab-separated file.

    We intentionally do NOT use pandas header inference because
    BUSCO full_table.tsv files contain comment/header conventions
    that can differ between BUSCO versions.

    Required columns:

        column 0 = BUSCO ID
        column 1 = Status
    """

    records = []

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as handle:

        for raw_line in handle:

            line = raw_line.rstrip(
                "\r\n"
            )

            if not line.strip():
                continue

            # BUSCO comments
            if line.startswith("#"):
                continue

            fields = line.split("\t")

            if len(fields) < 2:
                continue

            busco_id = fields[0].strip()
            status = fields[1].strip()

            if not busco_id:
                continue

            if status not in VALID_STATUSES:
                continue

            records.append(
                (
                    busco_id,
                    status
                )
            )

    if not records:

        raise RuntimeError(
            f"No valid BUSCO records found in:\n{path}"
        )

    df = pd.DataFrame(
        records,
        columns=[
            "BUSCO_ID",
            "Status"
        ]
    )

    # -------------------------------------------------------------------------
    # Handle duplicate BUSCO IDs
    # -------------------------------------------------------------------------

    priority = {
        "Missing": 0,
        "Fragmented": 1,
        "Complete": 2,
        "Duplicated": 3
    }

    df["_priority"] = (
        df["Status"]
        .map(priority)
        .fillna(-1)
    )

    df = (
        df
        .sort_values(
            [
                "BUSCO_ID",
                "_priority"
            ]
        )
        .drop_duplicates(
            subset="BUSCO_ID",
            keep="last"
        )
    )

    df = df.drop(
        columns="_priority"
    )

    return df


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("Y1000+ BUSCO GENOMIC FEATURE MATRIX")
print("=" * 80)

print()
print("Project:")
print(PROJECT)

print()
print("Manifest:")
print(MANIFEST_FILE)

print()
print("BUSCO root:")
print(BUSCO_ROOT)

print()
print("Output:")
print(OUTPUT_DIR)


# =============================================================================
# CHECK PATHS
# =============================================================================

if not MANIFEST_FILE.exists():

    stop(
        f"Manifest not found:\n{MANIFEST_FILE}"
    )

if not BUSCO_ROOT.exists():

    stop(
        f"BUSCO directory not found:\n{BUSCO_ROOT}"
    )


# =============================================================================
# LOAD MANIFEST
# =============================================================================

print()
print("=" * 80)
print("LOADING FINAL 436-GENOME MANIFEST")
print("=" * 80)

manifest = pd.read_csv(
    MANIFEST_FILE,
    dtype=str
)

manifest["Species"] = (
    manifest["Species"]
    .astype(str)
    .str.strip()
)

manifest["Assembly_Accession"] = (
    manifest["Assembly_Accession"]
    .astype(str)
    .str.strip()
)

print()
print(
    f"Manifest rows: {len(manifest)}"
)

print()
print("Manifest columns:")

for column in manifest.columns:
    print(
        f"  {column}"
    )


# =============================================================================
# MANIFEST QC
# =============================================================================

print()
print("=" * 80)
print("MANIFEST QC")
print("=" * 80)

if len(manifest) != EXPECTED_GENOMES:

    stop(
        f"Expected exactly {EXPECTED_GENOMES} representatives, "
        f"but found {len(manifest)}."
    )

print()
print(
    f"✓ Manifest contains exactly "
    f"{EXPECTED_GENOMES} representatives."
)


# =============================================================================
# SCAN BUSCO DIRECTORY ONCE
# =============================================================================

print()
print("=" * 80)
print("LOCATING BUSCO RESULTS")
print("=" * 80)

print()
print(
    "Scanning BUSCO directory once..."
)

print(
    "This is much faster than searching recursively "
    "for each of the 436 genomes."
)

# -------------------------------------------------------------------------
# ONE recursive scan
# -------------------------------------------------------------------------

all_full_tables = list(
    BUSCO_ROOT.rglob("full_table.tsv")
)

print()
print(
    f"Total full_table.tsv files discovered: "
    f"{len(all_full_tables)}"
)


# =============================================================================
# BUILD ACCESSION → BUSCO FILE INDEX
# =============================================================================

print()
print(
    "Building BUSCO accession index..."
)

busco_index = {}

unidentified_tables = []

duplicate_accessions = []


for path in all_full_tables:

    accession = accession_from_path(
        path
    )

    if accession is None:

        unidentified_tables.append(
            str(path)
        )

        continue

    if accession in busco_index:

        duplicate_accessions.append(
            accession
        )

        continue

    busco_index[
        accession
    ] = path


print(
    f"Unique BUSCO accessions indexed: "
    f"{len(busco_index)}"
)

print(
    f"Unidentified BUSCO files: "
    f"{len(unidentified_tables)}"
)

print(
    f"Duplicate accession entries: "
    f"{len(duplicate_accessions)}"
)


# =============================================================================
# MATCH THE 436 REPRESENTATIVES
# =============================================================================

print()
print(
    "Matching the 436 representative assemblies..."
)


busco_paths = []

missing_busco = []


for _, row in manifest.iterrows():

    species = row[
        "Species"
    ]

    accession = row[
        "Assembly_Accession"
    ]

    path = busco_index.get(
        accession
    )

    if path is None:

        missing_busco.append(
            {
                "Species":
                    species,

                "Assembly_Accession":
                    accession
            }
        )

    else:

        busco_paths.append(
            {
                "Species":
                    species,

                "Assembly_Accession":
                    accession,

                "BUSCO_Full_Table":
                    str(path)
            }
        )


# =============================================================================
# BUSCO DISCOVERY SUMMARY
# =============================================================================

print()
print(
    f"BUSCO full_table.tsv files "
    f"matching representatives: "
    f"{len(busco_paths)}/{len(manifest)}"
)

print(
    f"BUSCO results missing: "
    f"{len(missing_busco)}"
)


if missing_busco:

    missing_df = pd.DataFrame(
        missing_busco
    )

    missing_file = (
        OUTPUT_DIR
        / "busco_missing_436.csv"
    )

    missing_df.to_csv(
        missing_file,
        index=False
    )

    print()
    print(
        "Missing BUSCO results written to:"
    )

    print(
        missing_file
    )

    print()

    for item in missing_busco:

        print(
            f"  MISSING: "
            f"{item['Assembly_Accession']} "
            f"{item['Species']}"
        )

    stop(
        "BUSCO results are missing for "
        "one or more representative genomes."
    )


print()
print(
    "✓ BUSCO results found for every "
    "representative genome."
)


# =============================================================================
# READ BUSCO TABLES
# =============================================================================

print()
print("=" * 80)
print("READING BUSCO TABLES")
print("=" * 80)


all_records = []

summary_records = []

total = len(
    busco_paths
)


for i, item in enumerate(
    busco_paths,
    start=1
):

    species = item[
        "Species"
    ]

    accession = item[
        "Assembly_Accession"
    ]

    full_table = Path(
        item[
            "BUSCO_Full_Table"
        ]
    )

    print(
        f"[{i:>3}/{total}] "
        f"{species}"
    )

    try:

        df = read_busco_table(
            full_table
        )

    except Exception as error:

        raise RuntimeError(
            f"\nFailed reading BUSCO table:\n"
            f"{full_table}\n\n"
            f"Reason:\n"
            f"{error}"
        )

    counts = (
        df["Status"]
        .value_counts()
        .to_dict()
    )

    complete = counts.get(
        "Complete",
        0
    )

    duplicated = counts.get(
        "Duplicated",
        0
    )

    fragmented = counts.get(
        "Fragmented",
        0
    )

    missing = counts.get(
        "Missing",
        0
    )

    total_buscos = len(
        df
    )

    summary_records.append(
        {
            "Species":
                species,

            "Assembly_Accession":
                accession,

            "BUSCO_Total":
                total_buscos,

            "Complete":
                complete,

            "Duplicated":
                duplicated,

            "Fragmented":
                fragmented,

            "Missing":
                missing
        }
    )

    for _, busco in df.iterrows():

        all_records.append(
            {
                "Species":
                    species,

                "Assembly_Accession":
                    accession,

                "BUSCO_ID":
                    busco["BUSCO_ID"],

                "Status":
                    busco["Status"]
            }
        )


# =============================================================================
# LONG BUSCO TABLE
# =============================================================================

busco_long = pd.DataFrame(
    all_records
)


if busco_long.empty:

    stop(
        "No BUSCO records were extracted."
    )


print()
print(
    f"BUSCO records extracted: "
    f"{len(busco_long):,}"
)


# =============================================================================
# BUSCO FEATURE UNIVERSE
# =============================================================================

print()
print("=" * 80)
print("BUSCO FEATURE UNIVERSE")
print("=" * 80)


features = sorted(
    busco_long[
        "BUSCO_ID"
    ]
    .unique()
)


print()
print(
    f"Unique BUSCO features: "
    f"{len(features):,}"
)


# =============================================================================
# STATUS MATRIX
# =============================================================================

print()
print("=" * 80)
print("BUILDING BUSCO STATUS MATRIX")
print("=" * 80)


status_matrix = (
    busco_long
    .pivot_table(
        index=[
            "Species",
            "Assembly_Accession"
        ],
        columns="BUSCO_ID",
        values="Status",
        aggfunc="first"
    )
    .reset_index()
)


feature_columns = [
    column
    for column in status_matrix.columns
    if column not in {
        "Species",
        "Assembly_Accession"
    }
]


feature_columns = sorted(
    feature_columns
)


status_matrix = status_matrix[
    [
        "Species",
        "Assembly_Accession"
    ]
    +
    feature_columns
]


status_matrix[
    feature_columns
] = (
    status_matrix[
        feature_columns
    ]
    .fillna("Missing")
)


# =============================================================================
# PRESENCE / ABSENCE MATRIX
# =============================================================================

print()
print("=" * 80)
print("BUILDING PRESENCE/ABSENCE MATRIX")
print("=" * 80)


presence_matrix = (
    status_matrix.copy()
)


for feature in feature_columns:

    presence_matrix[
        feature
    ] = (
        presence_matrix[
            feature
        ]
        .isin(
            [
                "Complete",
                "Duplicated",
                "Fragmented"
            ]
        )
        .astype(int)
    )


# =============================================================================
# COMPLETE-ONLY MATRIX
# =============================================================================

print()
print("=" * 80)
print("BUILDING COMPLETE-ONLY MATRIX")
print("=" * 80)


complete_matrix = (
    status_matrix.copy()
)


for feature in feature_columns:

    complete_matrix[
        feature
    ] = (
        complete_matrix[
            feature
        ]
        .eq("Complete")
        .astype(int)
    )


# =============================================================================
# SUMMARY TABLE
# =============================================================================

summary_df = pd.DataFrame(
    summary_records
)


summary_df[
    "BUSCO_Complete_Pct"
] = (
    summary_df["Complete"]
    /
    summary_df["BUSCO_Total"]
    *
    100
).round(2)


summary_df[
    "BUSCO_Present_Pct"
] = (
    (
        summary_df["Complete"]
        +
        summary_df["Duplicated"]
        +
        summary_df["Fragmented"]
    )
    /
    summary_df["BUSCO_Total"]
    *
    100
).round(2)


# =============================================================================
# TAXON RECONCILIATION
# =============================================================================

print()
print("=" * 80)
print("MATRIX TAXON QC")
print("=" * 80)


expected_taxa = set(
    manifest[
        "Assembly_Accession"
    ]
)


actual_taxa = set(
    presence_matrix[
        "Assembly_Accession"
    ]
)


missing_taxa = (
    expected_taxa
    -
    actual_taxa
)

unexpected_taxa = (
    actual_taxa
    -
    expected_taxa
)


print()
print(
    f"Expected representative genomes: "
    f"{len(expected_taxa)}"
)

print(
    f"Genomes in matrix: "
    f"{len(actual_taxa)}"
)

print(
    f"Missing: "
    f"{len(missing_taxa)}"
)

print(
    f"Unexpected: "
    f"{len(unexpected_taxa)}"
)


if missing_taxa:

    for accession in sorted(
        missing_taxa
    ):

        print(
            f"  MISSING: {accession}"
        )


if unexpected_taxa:

    for accession in sorted(
        unexpected_taxa
    ):

        print(
            f"  UNEXPECTED: {accession}"
        )


if missing_taxa or unexpected_taxa:

    stop(
        "Matrix taxon reconciliation failed."
    )


print()
print(
    "✓ All 436 representative taxa "
    "are present in the matrix."
)


# =============================================================================
# WRITE OUTPUTS
# =============================================================================

print()
print("=" * 80)
print("WRITING OUTPUT FILES")
print("=" * 80)


status_file = (
    OUTPUT_DIR
    / "y1000_busco_status_matrix.csv"
)

presence_file = (
    OUTPUT_DIR
    / "y1000_busco_presence_absence_matrix.csv"
)

complete_file = (
    OUTPUT_DIR
    / "y1000_busco_complete_matrix.csv"
)

summary_file = (
    OUTPUT_DIR
    / "y1000_busco_feature_summary.csv"
)


status_matrix.to_csv(
    status_file,
    index=False
)

presence_matrix.to_csv(
    presence_file,
    index=False
)

complete_matrix.to_csv(
    complete_file,
    index=False
)

summary_df.to_csv(
    summary_file,
    index=False
)


# =============================================================================
# FINAL VALIDATION
# =============================================================================

print()
print("=" * 80)
print("FINAL VALIDATION")
print("=" * 80)


n_species = len(
    presence_matrix
)

n_features = len(
    feature_columns
)


print()
print(
    f"Final species: "
    f"{n_species}"
)

print(
    f"Final BUSCO features: "
    f"{n_features}"
)

print(
    f"Final matrix dimensions: "
    f"{n_species} × {n_features}"
)


if n_species != EXPECTED_GENOMES:

    stop(
        f"Expected {EXPECTED_GENOMES} species "
        f"but found {n_species}."
    )


binary_values = set(
    presence_matrix[
        feature_columns
    ]
    .stack()
    .unique()
)


if not binary_values.issubset(
    {0, 1}
):

    stop(
        "Presence/absence matrix contains "
        "values other than 0 and 1."
    )


if (
    presence_matrix[
        feature_columns
    ]
    .isna()
    .any()
    .any()
):

    stop(
        "NaN values found in "
        "presence/absence matrix."
    )


# =============================================================================
# FINAL REPORT
# =============================================================================

print()
print("=" * 80)
print("GENOMIC FEATURE MATRIX COMPLETE")
print("=" * 80)

print()
print(
    "✓ Existing BUSCO results used."
)

print(
    "✓ BUSCO was NOT rerun."
)

print(
    f"✓ {n_species} representative genomes included."
)

print(
    f"✓ {n_features} BUSCO features detected."
)

print(
    "✓ All 436 taxa reconciled."
)

print(
    "✓ Presence/absence matrix is binary."
)

print()
print(
    "PRIMARY ML MATRIX:"
)

print(
    presence_file
)

print()
print(
    "COMPLETE-ONLY MATRIX:"
)

print(
    complete_file
)

print()
print(
    "STATUS MATRIX:"
)

print(
    status_file
)

print()
print(
    "FEATURE SUMMARY:"
)

print(
    summary_file
)

print()
print(
    "✓ Genomic feature matrix generation finished."
)