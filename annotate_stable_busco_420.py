from pathlib import Path
import pandas as pd
import re
import sys
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")

BUSCO_ROOT = (
    PROJECT_ROOT
    / "results"
    / "stage4B_phylogeny"
    / "busco"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
)

STABLE_CANDIDATE_FILE = (
    FEATURE_DIR
    / "tables"
    / "Carbon_Breadth_stable_BUSCO_candidates_420.csv"
)

OUTPUT_DIR = (
    FEATURE_DIR
    / "functional_annotation_420"
)

OUTPUT_TABLE_DIR = OUTPUT_DIR / "tables"

OUTPUT_TABLE = (
    OUTPUT_TABLE_DIR
    / "Carbon_Breadth_BUSCO_functional_annotation_420.csv"
)


# =============================================================================
# SETTINGS
# =============================================================================

# The stable table contains the most stable candidates.
# We also recover the top-20 BUSCO IDs from the feature interpretation data.
TOP_N = 20

# BUSCO full_table.tsv normally has these columns:
BUSCO_COLUMNS = [
    "busco_id",
    "status",
    "sequence",
    "start",
    "end",
    "strand",
    "score",
    "length",
    "ortho_db_url",
    "description",
]


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def parse_species_and_accession(dirname):
    """
    Example:
    Aciculoconidium_aculeatum__GCA_030579395.1

    Returns:
        species_name
        assembly_accession
    """

    if "__" in dirname:
        species, accession = dirname.rsplit("__", 1)
    else:
        species = dirname
        accession = ""

    return species, accession


def normalize_busco_id(value):
    """
    Normalize BUSCO IDs while preserving the actual identifier.

    Example:
        12468at4891 -> 12468at4891
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    # Remove accidental whitespace
    value = re.sub(r"\s+", "", value)

    return value


def find_full_table(genome_dir):
    """
    BUSCO structure confirmed from the project:

    genome_dir/
        run_saccharomycetes_odb10/
            full_table.tsv

    We search only inside the individual genome directory.
    """

    # Fast path: expected BUSCO layout
    candidates = list(genome_dir.glob("run_*/full_table.tsv"))

    if candidates:
        return candidates[0]

    # Fallback in case a directory has an unexpected run name
    candidates = list(genome_dir.glob("**/full_table.tsv"))

    if candidates:
        return candidates[0]

    return None


def read_busco_full_table(full_table):
    """
    Read a BUSCO full_table.tsv.

    BUSCO full_table files contain comment lines beginning with '#'.
    The actual data are tab-separated.
    """

    try:
        df = pd.read_csv(
            full_table,
            sep="\t",
            comment="#",
            header=None,
            dtype=str,
            engine="python"
        )

    except Exception as e:
        print(f"ERROR reading {full_table}")
        print(e)
        return None

    # Keep only rows with at least one column
    df = df.dropna(how="all")

    if df.empty:
        return None

    # BUSCO full_table normally has 10 fields.
    # Some versions may contain a slightly different layout.
    if df.shape[1] >= 10:

        df = df.iloc[:, :10].copy()
        df.columns = BUSCO_COLUMNS

    else:
        print(
            f"WARNING: Unexpected number of columns "
            f"({df.shape[1]}) in {full_table}"
        )
        return None

    # Normalize BUSCO ID
    df["busco_id"] = df["busco_id"].map(normalize_busco_id)

    return df


# =============================================================================
# LOAD STABLE CANDIDATES
# =============================================================================

print_header("STABLE BUSCO FUNCTIONAL ANNOTATION")

print("Project:")
print(PROJECT_ROOT)

print()
print("BUSCO root:")
print(BUSCO_ROOT)

print()
print("Stable candidate file:")
print(STABLE_CANDIDATE_FILE)


if not STABLE_CANDIDATE_FILE.exists():
    print()
    print("ERROR: Stable candidate file not found:")
    print(STABLE_CANDIDATE_FILE)
    sys.exit(1)


stable_df = pd.read_csv(STABLE_CANDIDATE_FILE)

print_header("LOADING STABLE BUSCO CANDIDATES")

print("Stable candidate table:")
print(stable_df.shape)

print()
print("Columns:")
print(stable_df.columns.tolist())


# =============================================================================
# LOAD / IDENTIFY BUSCO CANDIDATES
# =============================================================================

candidate_ids = []

# Stable candidate BUSCOs
if "Feature" in stable_df.columns:

    for value in stable_df["Feature"].dropna():

        value = normalize_busco_id(value)

        if value and value not in candidate_ids:
            candidate_ids.append(value)


# If stable table does not contain all 20 candidates,
# recover them from the feature interpretation directory.
#
# We deliberately search ONLY inside feature_interpretation_420,
# not the entire project.

print_header("IDENTIFYING TOP BUSCO CANDIDATES")

# Search CSV files only inside the feature interpretation directory.
candidate_csvs = list(FEATURE_DIR.rglob("*.csv"))

print(f"CSV files found inside feature interpretation directory: {len(candidate_csvs)}")


# Collect BUSCO-like feature names from CSV files
all_candidate_features = []

for csv_file in candidate_csvs:

    try:
        temp = pd.read_csv(csv_file, nrows=5000)

    except Exception:
        continue

    for column in ["Feature", "feature", "BUSCO", "BUSCO_ID", "busco_id"]:

        if column in temp.columns:

            values = temp[column].dropna().astype(str)

            for value in values:

                value = normalize_busco_id(value)

                if value and re.fullmatch(r"\d+at\d+", value):

                    all_candidate_features.append(value)


# Count candidate frequency
if all_candidate_features:

    feature_counts = (
        pd.Series(all_candidate_features)
        .value_counts()
        .reset_index()
    )

    feature_counts.columns = ["BUSCO_ID", "Count"]

    # Add candidates not already in stable list
    for busco_id in feature_counts["BUSCO_ID"]:

        if busco_id not in candidate_ids:
            candidate_ids.append(busco_id)

        if len(candidate_ids) >= TOP_N:
            break


# If we still have fewer than 20, explicitly use the known
# top-20 candidates from the previous analysis.
known_top20 = [
    "12468at4891",
    "12523at4891",
    "1679at4891",
    "18913at4891",
    "22532at4891",
    "2307at4891",
    "23379at4891",
    "24318at4891",
    "2471at4891",
    "27915at4891",
    "28058at4891",
    "29457at4891",
    "33531at4891",
    "34052at4891",
    "3574at4891",
    "36839at4891",
    "4322at4891",
    "5746at4891",
    "6711at4891",
    "9782at4891",
]

for busco_id in known_top20:

    if busco_id not in candidate_ids:
        candidate_ids.append(busco_id)


# Restrict to the intended 20 candidates
candidate_ids = [
    busco_id
    for busco_id in candidate_ids
    if busco_id in known_top20
]

# Preserve the known top-20 ordering
candidate_ids = [
    busco_id
    for busco_id in known_top20
    if busco_id in candidate_ids
]

print()
print("Unique BUSCO candidates:")
print(len(candidate_ids))

print()
print(candidate_ids)


# =============================================================================
# CREATE ML INFORMATION LOOKUP
# =============================================================================

print_header("PREPARING ML FEATURE INFORMATION")

ml_lookup = {}

if "Feature" in stable_df.columns:

    for _, row in stable_df.iterrows():

        feature = normalize_busco_id(row.get("Feature"))

        if not feature:
            continue

        ml_lookup[feature] = {
            "Mean_Importance": row.get("Mean_Importance"),
            "SD_Importance": row.get("SD_Importance"),
            "Median_Importance": row.get("Median_Importance"),
            "Min_Importance": row.get("Min_Importance"),
            "Max_Importance": row.get("Max_Importance"),
            "Mean_Baseline_R2": row.get("Mean_Baseline_R2"),
            "Positive_Importance_Folds": row.get(
                "Positive_Importance_Folds"
            ),
            "Fold_Stability": row.get("Fold_Stability"),
            "Importance_Rank": row.get("Importance_Rank"),
        }


print(f"ML feature records loaded: {len(ml_lookup)}")


# =============================================================================
# FIND BUSCO GENOME DIRECTORIES
# =============================================================================

print_header("SEARCHING BUSCO RESULT DIRECTORIES")

if not BUSCO_ROOT.exists():

    print("ERROR: BUSCO root does not exist:")
    print(BUSCO_ROOT)
    sys.exit(1)


# IMPORTANT:
# We only inspect direct child directories.
# No project-wide os.walk().
genome_dirs = [
    p for p in BUSCO_ROOT.iterdir()
    if p.is_dir()
]

genome_dirs.sort(key=lambda x: x.name.lower())

print()
print("BUSCO result directories found:")
print(len(genome_dirs))


if not genome_dirs:

    print("ERROR: No BUSCO genome directories found.")
    sys.exit(1)


# =============================================================================
# ANNOTATE ALL GENOMES
# =============================================================================

print_header("EXTRACTING BUSCO FUNCTIONAL ANNOTATIONS")

records = []

start_time = time.time()

tables_found = 0
tables_missing = 0
tables_failed = 0

for i, genome_dir in enumerate(genome_dirs, start=1):

    species_name, assembly_accession = parse_species_and_accession(
        genome_dir.name
    )

    full_table = find_full_table(genome_dir)

    # Progress every genome
    print(
        f"[{i:03d}/{len(genome_dirs):03d}] "
        f"{species_name}"
    )

    if full_table is None:

        tables_missing += 1

        # Even if the BUSCO table is missing, preserve the candidate
        # structure so the final dataset remains explicit.
        for busco_id in candidate_ids:

            ml_info = ml_lookup.get(busco_id, {})

            records.append({
                "Species": species_name,
                "Assembly_Accession": assembly_accession,
                "BUSCO_ID": busco_id,
                "Status": "BUSCO_TABLE_NOT_FOUND",
                "Sequence": pd.NA,
                "Start": pd.NA,
                "End": pd.NA,
                "Strand": pd.NA,
                "Score": pd.NA,
                "Length": pd.NA,
                "OrthoDB_URL": pd.NA,
                "Functional_Description": pd.NA,
                "Mean_Importance": ml_info.get("Mean_Importance"),
                "SD_Importance": ml_info.get("SD_Importance"),
                "Median_Importance": ml_info.get("Median_Importance"),
                "Min_Importance": ml_info.get("Min_Importance"),
                "Max_Importance": ml_info.get("Max_Importance"),
                "Mean_Baseline_R2": ml_info.get("Mean_Baseline_R2"),
                "Positive_Importance_Folds": ml_info.get(
                    "Positive_Importance_Folds"
                ),
                "Fold_Stability": ml_info.get("Fold_Stability"),
                "Importance_Rank": ml_info.get("Importance_Rank"),
                "BUSCO_Table": pd.NA,
            })

        continue

    tables_found += 1

    busco_df = read_busco_full_table(full_table)

    if busco_df is None:

        tables_failed += 1
        continue


    # Create lookup for this genome
    busco_lookup = {}

    for _, row in busco_df.iterrows():

        busco_id = normalize_busco_id(row["busco_id"])

        if busco_id in candidate_ids:

            busco_lookup[busco_id] = row


    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Iterate over ALL candidate IDs.
    #
    # This means Missing BUSCOs are explicitly retained.
    # -------------------------------------------------------------------------

    for busco_id in candidate_ids:

        ml_info = ml_lookup.get(busco_id, {})

        if busco_id in busco_lookup:

            row = busco_lookup[busco_id]

            status = row["status"]

            sequence = row["sequence"]
            start = row["start"]
            end = row["end"]
            strand = row["strand"]
            score = row["score"]
            length = row["length"]
            ortho_db_url = row["ortho_db_url"]
            description = row["description"]

        else:

            # Candidate not explicitly present in full_table.
            # Treat as Missing, rather than dropping it.
            status = "Missing"

            sequence = pd.NA
            start = pd.NA
            end = pd.NA
            strand = pd.NA
            score = pd.NA
            length = pd.NA
            ortho_db_url = pd.NA
            description = pd.NA


        records.append({
            "Species": species_name,
            "Assembly_Accession": assembly_accession,
            "BUSCO_ID": busco_id,
            "Status": status,
            "Sequence": sequence,
            "Start": start,
            "End": end,
            "Strand": strand,
            "Score": score,
            "Length": length,
            "OrthoDB_URL": ortho_db_url,
            "Functional_Description": description,

            # ML information
            "Mean_Importance": ml_info.get("Mean_Importance"),
            "SD_Importance": ml_info.get("SD_Importance"),
            "Median_Importance": ml_info.get("Median_Importance"),
            "Min_Importance": ml_info.get("Min_Importance"),
            "Max_Importance": ml_info.get("Max_Importance"),
            "Mean_Baseline_R2": ml_info.get("Mean_Baseline_R2"),
            "Positive_Importance_Folds": ml_info.get(
                "Positive_Importance_Folds"
            ),
            "Fold_Stability": ml_info.get("Fold_Stability"),
            "Importance_Rank": ml_info.get("Importance_Rank"),

            "BUSCO_Table": str(full_table),
        })


# =============================================================================
# CREATE MASTER TABLE
# =============================================================================

print_header("MASTER BUSCO ANNOTATION TABLE")

if not records:

    print("ERROR: No BUSCO annotation records were generated.")
    sys.exit(1)


annotation_df = pd.DataFrame(records)


# Convert numeric columns where appropriate
numeric_columns = [
    "Start",
    "End",
    "Score",
    "Length",
    "Mean_Importance",
    "SD_Importance",
    "Median_Importance",
    "Min_Importance",
    "Max_Importance",
    "Mean_Baseline_R2",
    "Positive_Importance_Folds",
    "Fold_Stability",
    "Importance_Rank",
]

for col in numeric_columns:

    if col in annotation_df.columns:

        annotation_df[col] = pd.to_numeric(
            annotation_df[col],
            errors="coerce"
        )


# Sort consistently
annotation_df = annotation_df.sort_values(
    by=[
        "Species",
        "Importance_Rank",
        "BUSCO_ID"
    ],
    na_position="last"
).reset_index(drop=True)


# =============================================================================
# SAVE
# =============================================================================

OUTPUT_TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

annotation_df.to_csv(
    OUTPUT_TABLE,
    index=False
)


# =============================================================================
# SUMMARY
# =============================================================================

elapsed = time.time() - start_time

print()
print("Saved:")
print(OUTPUT_TABLE)

print()
print("Shape:")
print(annotation_df.shape)

print()
print("BUSCO result directories:")
print(len(genome_dirs))

print()
print("Full BUSCO tables found:")
print(tables_found)

print()
print("BUSCO tables missing:")
print(tables_missing)

print()
print("BUSCO tables failed to read:")
print(tables_failed)

print()
print("Candidate BUSCO IDs:")
print(len(candidate_ids))

print()
print("Expected records if all tables were available:")
print(len(genome_dirs) * len(candidate_ids))

print()
print("Actual records:")
print(len(annotation_df))


# =============================================================================
# STATUS SUMMARY
# =============================================================================

print_header("BUSCO STATUS SUMMARY")

status_summary = (
    annotation_df["Status"]
    .value_counts(dropna=False)
)

print(status_summary)


# =============================================================================
# CANDIDATE SUMMARY
# =============================================================================

print_header("CANDIDATE BUSCO SUMMARY")

candidate_summary = (
    annotation_df
    .groupby("BUSCO_ID", dropna=False)
    .agg(
        Genomes=("Species", "nunique"),
        Complete=("Status", lambda x: (x == "Complete").sum()),
        Duplicated=("Status", lambda x: (x == "Duplicated").sum()),
        Fragmented=("Status", lambda x: (x == "Fragmented").sum()),
        Missing=("Status", lambda x: (x == "Missing").sum()),
        Table_Not_Found=(
            "Status",
            lambda x: (x == "BUSCO_TABLE_NOT_FOUND").sum()
        ),
    )
    .reset_index()
)


print(candidate_summary.to_string(index=False))


# Save candidate summary
candidate_summary_file = (
    OUTPUT_TABLE_DIR
    / "Carbon_Breadth_BUSCO_candidate_summary_420.csv"
)

candidate_summary.to_csv(
    candidate_summary_file,
    index=False
)

print()
print("Candidate summary saved:")
print(candidate_summary_file)


# =============================================================================
# FUNCTIONAL ANNOTATION SUMMARY
# =============================================================================

print_header("FUNCTIONAL ANNOTATION SUMMARY")

functional_df = (
    annotation_df[
        annotation_df["Functional_Description"].notna()
    ][
        [
            "BUSCO_ID",
            "Functional_Description"
        ]
    ]
    .drop_duplicates()
    .sort_values("BUSCO_ID")
)

print(
    functional_df.to_string(index=False)
)


print()
print("=" * 80)
print("ANNOTATION COMPLETE")
print("=" * 80)

print()
print(f"Elapsed time: {elapsed:.2f} seconds")

print()
print("Main output:")
print(OUTPUT_TABLE)

print()
print("Candidate summary:")
print(candidate_summary_file)

print()
print("The BUSCO annotation table is now ready for downstream")
print("functional interpretation, pathway analysis, and network analysis.")

print("=" * 80)