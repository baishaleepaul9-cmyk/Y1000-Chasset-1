from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# GENOME ↔ PHENOTYPE RECONCILIATION
# WITH EXPLICIT TAXONOMIC MAPPING
# ============================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

MANIFEST_FILE = (
    PROJECT
    / "results"
    / "stage4B_phylogeny"
    / "stage4B_genome_fasta_manifest.csv"
)

GENOMIC_MATRIX_FILE = (
    PROJECT
    / "results"
    / "stage4B_phylogeny"
    / "postbusco_phylogeny_v2"
    / "genomic_features"
    / "y1000_busco_presence_absence_matrix.csv"
)

PHENOTYPE_FILE = (
    PROJECT
    / "results"
    / "stage2A_multitrait_profile"
    / "species_multitrait_profile.csv"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "genome_phenotype_reconciliation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

MAIN_OUTPUT = (
    OUTPUT_DIR
    / "genome_phenotype_ml_dataset.csv"
)

GENOME_ONLY_OUTPUT = (
    OUTPUT_DIR
    / "genome_only_species.csv"
)

PHENOTYPE_ONLY_OUTPUT = (
    OUTPUT_DIR
    / "phenotype_only_species.csv"
)

AGGREGATION_LOG_OUTPUT = (
    OUTPUT_DIR
    / "phenotype_aggregation_log.csv"
)

TAXON_MAPPING_OUTPUT = (
    OUTPUT_DIR
    / "explicit_taxonomic_mappings.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "reconciliation_summary.csv"
)

# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_GENOMES = 436

PHENOTYPE_COLUMNS = [
    "N_Strains",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Carbon_Breadth_SD",
    "Nitrogen_Breadth_SD",
]

# ============================================================
# HELPERS
# ============================================================

def normalize_species(value):
    """
    Normalize species names for matching.

    This does NOT perform taxonomic synonym resolution.
    It only normalizes whitespace, case and underscores.
    """

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = value.replace("_", " ")

    # Collapse repeated spaces
    value = " ".join(value.split())

    return value


def clean_accession(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def require_columns(df, columns, name):

    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{name} is missing required columns:\n"
            + "\n".join(f"  {x}" for x in missing)
        )


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("GENOME ↔ PHENOTYPE RECONCILIATION")
print("=" * 80)

print()
print("Project:")
print(PROJECT)

print()
print("Output:")
print(OUTPUT_DIR)

# ============================================================
# LOAD MANIFEST
# ============================================================

print()
print("=" * 80)
print("LOADING FINAL 436-GENOME MANIFEST")
print("=" * 80)

manifest = pd.read_csv(MANIFEST_FILE)

require_columns(
    manifest,
    [
        "Species",
        "Assembly_Accession",
        "Genome_FASTA",
        "Status",
        "Error",
    ],
    "Genome manifest"
)

print()
print(f"Manifest rows: {len(manifest)}")

if len(manifest) != EXPECTED_GENOMES:
    raise RuntimeError(
        f"Expected {EXPECTED_GENOMES} manifest rows, "
        f"found {len(manifest)}."
    )

print(
    f"✓ Exactly {EXPECTED_GENOMES} representative genomes found."
)

manifest["_species_key"] = (
    manifest["Species"]
    .apply(normalize_species)
)

manifest["_accession_key"] = (
    manifest["Assembly_Accession"]
    .apply(clean_accession)
)

if manifest["_species_key"].duplicated().any():

    dup = manifest[
        manifest["_species_key"].duplicated(
            keep=False
        )
    ]

    raise RuntimeError(
        "Manifest contains duplicate species:\n"
        + dup[
            [
                "Species",
                "Assembly_Accession"
            ]
        ].to_string(index=False)
    )

print("✓ Manifest species are unique.")

# ============================================================
# LOAD GENOMIC MATRIX
# ============================================================

print()
print("=" * 80)
print("LOADING BUSCO GENOMIC FEATURE MATRIX")
print("=" * 80)

genomic = pd.read_csv(
    GENOMIC_MATRIX_FILE
)

require_columns(
    genomic,
    [
        "Species",
        "Assembly_Accession",
    ],
    "Genomic feature matrix"
)

print()
print(
    f"Genomic matrix shape: {genomic.shape}"
)

genomic["_species_key"] = (
    genomic["Species"]
    .apply(normalize_species)
)

genomic["_accession_key"] = (
    genomic["Assembly_Accession"]
    .apply(clean_accession)
)

# BUSCO features = everything except identifiers
identifier_columns = [
    "Species",
    "Assembly_Accession",
]

busco_features = [
    c for c in genomic.columns
    if c not in identifier_columns
    and not c.startswith("_")
]

print(
    f"Genomic genomes: {len(genomic)}"
)

print(
    f"BUSCO features: {len(busco_features)}"
)

if len(genomic) != EXPECTED_GENOMES:

    raise RuntimeError(
        f"Expected {EXPECTED_GENOMES} genomic rows, "
        f"found {len(genomic)}."
    )

print("✓ Genomic matrix loaded.")

# ============================================================
# MANIFEST ↔ GENOMIC MATRIX
# ============================================================

print()
print("=" * 80)
print("MANIFEST ↔ GENOMIC MATRIX RECONCILIATION")
print("=" * 80)

manifest_species = set(
    manifest["_species_key"]
)

genomic_species = set(
    genomic["_species_key"]
)

missing_from_genomic = (
    manifest_species
    - genomic_species
)

unexpected_genomic = (
    genomic_species
    - manifest_species
)

print()
print(
    f"Manifest species: {len(manifest_species)}"
)

print(
    f"Genomic matrix species: {len(genomic_species)}"
)

print(
    f"Missing from genomic matrix: "
    f"{len(missing_from_genomic)}"
)

print(
    f"Unexpected genomic species: "
    f"{len(unexpected_genomic)}"
)

if missing_from_genomic:
    raise RuntimeError(
        "Some manifest species are missing "
        "from the genomic matrix."
    )

if unexpected_genomic:
    raise RuntimeError(
        "Unexpected species found in genomic matrix."
    )

print()
print(
    "✓ Manifest and genomic matrix reconcile perfectly."
)

# ============================================================
# LOAD PHENOTYPE
# ============================================================

print()
print("=" * 80)
print("LOADING FINAL PHENOTYPE TABLE")
print("=" * 80)

phenotype = pd.read_csv(
    PHENOTYPE_FILE
)

require_columns(
    phenotype,
    ["Species"] + PHENOTYPE_COLUMNS,
    "Phenotype table"
)

print()
print(
    f"Phenotype rows: {len(phenotype)}"
)

print(
    f"Phenotype columns: {len(phenotype.columns)}"
)

phenotype["_species_key"] = (
    phenotype["Species"]
    .apply(normalize_species)
)

print("✓ Phenotype table loaded.")

# ============================================================
# PHENOTYPE DUPLICATE HANDLING
# ============================================================

print()
print("=" * 80)
print("PHENOTYPE DUPLICATE HANDLING")
print("=" * 80)

species_counts = (
    phenotype["_species_key"]
    .value_counts()
)

duplicate_species = species_counts[
    species_counts > 1
]

print()
print(
    f"Unique phenotype species: "
    f"{phenotype['_species_key'].nunique()}"
)

print(
    f"Species with multiple phenotype rows: "
    f"{len(duplicate_species)}"
)

if len(duplicate_species) > 0:

    print()
    print(
        "Duplicated phenotype species:"
    )

    for key, count in duplicate_species.items():

        original_names = phenotype.loc[
            phenotype["_species_key"] == key,
            "Species"
        ].unique()

        print(
            f"  {original_names[0]}: "
            f"{count} rows"
        )

# ============================================================
# AGGREGATE DUPLICATE PHENOTYPE RECORDS
# ============================================================

print()
print("=" * 80)
print("CREATING ONE PHENOTYPE PROFILE PER SPECIES")
print("=" * 80)

aggregation_log = []

aggregated_rows = []

for species_key, group in phenotype.groupby(
    "_species_key",
    sort=False
):

    if len(group) == 1:

        row = group.iloc[0].copy()

        aggregated_rows.append(row)

        aggregation_log.append({
            "Species": row["Species"],
            "Species_Key": species_key,
            "Original_Rows": 1,
            "Aggregation_Method": "None",
            "Source_Species_Names": row["Species"],
        })

    else:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Duplicate phenotype rows are aggregated using the
        # arithmetic mean for numerical phenotype variables.
        #
        # This is only for genuine duplicate species records.
        # ----------------------------------------------------

        row = group.iloc[0].copy()

        for col in PHENOTYPE_COLUMNS:

            row[col] = pd.to_numeric(
                group[col],
                errors="coerce"
            ).mean()

        source_names = "; ".join(
            group["Species"]
            .astype(str)
            .unique()
        )

        row["Species"] = (
            group["Species"]
            .iloc[0]
        )

        aggregated_rows.append(row)

        aggregation_log.append({
            "Species": row["Species"],
            "Species_Key": species_key,
            "Original_Rows": len(group),
            "Aggregation_Method": "Mean",
            "Source_Species_Names": source_names,
        })

phenotype_agg = pd.DataFrame(
    aggregated_rows
)

aggregation_log_df = pd.DataFrame(
    aggregation_log
)

# Save aggregation log
aggregation_log_df.to_csv(
    AGGREGATION_LOG_OUTPUT,
    index=False
)

print()
print(
    f"Original phenotype rows: "
    f"{len(phenotype)}"
)

print(
    f"Unique species after aggregation: "
    f"{len(phenotype_agg)}"
)

print()
print(
    "✓ Exactly one phenotype profile per species."
)

print()
print("Aggregation log:")
print(AGGREGATION_LOG_OUTPUT)

# ============================================================
# EXPLICIT TAXONOMIC MAPPINGS
# ============================================================

print()
print("=" * 80)
print("EXPLICIT TAXONOMIC MAPPING")
print("=" * 80)

# ------------------------------------------------------------
# IMPORTANT
#
# These mappings are NOT generic synonym matching.
# They are explicitly declared because the genome/ASTRAL
# taxon uses the species-level name while the phenotype table
# contains the corresponding nominal variety.
# ------------------------------------------------------------

EXPLICIT_MAPPINGS = {

    normalize_species(
        "Schwanniomyces polymorphus"
    ): normalize_species(
        "Schwanniomyces polymorphus var. polymorphus"
    ),

}

mapping_rows = []

for genome_species_key, phenotype_species_key in (
    EXPLICIT_MAPPINGS.items()
):

    genome_display = (
        manifest.loc[
            manifest["_species_key"]
            == genome_species_key,
            "Species"
        ]
        .iloc[0]
        if (
            manifest["_species_key"]
            == genome_species_key
        ).any()
        else genome_species_key
    )

    phenotype_display = (
        phenotype_agg.loc[
            phenotype_agg["_species_key"]
            == phenotype_species_key,
            "Species"
        ]
        .iloc[0]
        if (
            phenotype_agg["_species_key"]
            == phenotype_species_key
        ).any()
        else phenotype_species_key
    )

    mapping_rows.append({
        "Genome_Species": genome_display,
        "Genome_Species_Key": genome_species_key,
        "Phenotype_Species": phenotype_display,
        "Phenotype_Species_Key": phenotype_species_key,
        "Mapping_Type": "Explicit_nominal_variety_mapping",
        "Reason": (
            "Genome/ASTRAL species-level name mapped "
            "explicitly to nominal variety "
            "var. polymorphus in phenotype table."
        ),
    })

taxon_mapping_df = pd.DataFrame(
    mapping_rows
)

taxon_mapping_df.to_csv(
    TAXON_MAPPING_OUTPUT,
    index=False
)

print()

for row in mapping_rows:

    print(
        f"  {row['Genome_Species']}"
    )

    print(
        f"      ↓"
    )

    print(
        f"  {row['Phenotype_Species']}"
    )

    print(
        f"      [{row['Mapping_Type']}]"
    )

print()
print(
    "Mapping table:"
)
print(TAXON_MAPPING_OUTPUT)

# ============================================================
# VALIDATE MAPPINGS
# ============================================================

for genome_key, phenotype_key in (
    EXPLICIT_MAPPINGS.items()
):

    if not (
        manifest["_species_key"]
        == genome_key
    ).any():

        raise RuntimeError(
            "Explicit genome mapping refers to "
            "a species not present in the 436-genome manifest:\n"
            f"{genome_key}"
        )

    if not (
        phenotype_agg["_species_key"]
        == phenotype_key
    ).any():

        raise RuntimeError(
            "Explicit phenotype mapping refers to "
            "a species not present in phenotype table:\n"
            f"{phenotype_key}"
        )

print()
print(
    "✓ Explicit taxonomic mappings validated."
)

# ============================================================
# CREATE EFFECTIVE PHENOTYPE MATCH KEY
# ============================================================

phenotype_agg["_match_key"] = (
    phenotype_agg["_species_key"]
)

# Build reverse mapping:
# phenotype species → genome species
reverse_mapping = {
    phenotype_key: genome_key
    for genome_key, phenotype_key
    in EXPLICIT_MAPPINGS.items()
}

# Change the phenotype match key only for the
# explicitly documented mapping.

for phenotype_key, genome_key in (
    reverse_mapping.items()
):

    mask = (
        phenotype_agg["_species_key"]
        == phenotype_key
    )

    phenotype_agg.loc[
        mask,
        "_match_key"
    ] = genome_key

# ============================================================
# SPECIES RECONCILIATION
# ============================================================

print()
print("=" * 80)
print("GENOME ↔ PHENOTYPE SPECIES RECONCILIATION")
print("=" * 80)

genomic_species = set(
    genomic["_species_key"]
)

phenotype_match_species = set(
    phenotype_agg["_match_key"]
)

matched_species = (
    genomic_species
    & phenotype_match_species
)

genome_only_species = (
    genomic_species
    - phenotype_match_species
)

phenotype_only_species = (
    phenotype_match_species
    - genomic_species
)

print()
print(
    f"Genomic species: "
    f"{len(genomic_species)}"
)

print(
    f"Phenotype species: "
    f"{len(phenotype_match_species)}"
)

print(
    f"Matched species: "
    f"{len(matched_species)}"
)

print(
    f"Genome-only species: "
    f"{len(genome_only_species)}"
)

print(
    f"Phenotype-only species: "
    f"{len(phenotype_only_species)}"
)

# ============================================================
# SAVE GENOME-ONLY
# ============================================================

print()
print("=" * 80)
print("SAVING GENOME-ONLY SPECIES")
print("=" * 80)

genome_only_df = manifest[
    manifest["_species_key"].isin(
        genome_only_species
    )
].copy()

genome_only_df = genome_only_df[
    [
        "Species",
        "Assembly_Accession",
        "Genome_FASTA",
        "Status",
        "Error",
    ]
]

genome_only_df.to_csv(
    GENOME_ONLY_OUTPUT,
    index=False
)

print()
print(
    f"Genome-only species: "
    f"{len(genome_only_df)}"
)

print("Output:")
print(GENOME_ONLY_OUTPUT)

# ============================================================
# SAVE PHENOTYPE-ONLY
# ============================================================

print()
print("=" * 80)
print("SAVING PHENOTYPE-ONLY SPECIES")
print("=" * 80)

phenotype_only_df = phenotype_agg[
    phenotype_agg["_match_key"].isin(
        phenotype_only_species
    )
].copy()

phenotype_only_df = phenotype_only_df[
    [
        "Species",
        *PHENOTYPE_COLUMNS,
    ]
]

phenotype_only_df.to_csv(
    PHENOTYPE_ONLY_OUTPUT,
    index=False
)

print()
print(
    f"Phenotype-only species: "
    f"{len(phenotype_only_df)}"
)

print("Output:")
print(PHENOTYPE_ONLY_OUTPUT)

# ============================================================
# CREATE MATCHED DATASET
# ============================================================

print()
print("=" * 80)
print("CREATING MATCHED GENOME–PHENOTYPE DATASET")
print("=" * 80)

# ------------------------------------------------------------
# Genomic side
# ------------------------------------------------------------

genomic_matched = genomic[
    genomic["_species_key"].isin(
        matched_species
    )
].copy()

# ------------------------------------------------------------
# Phenotype side
# ------------------------------------------------------------

phenotype_matched = phenotype_agg[
    phenotype_agg["_match_key"].isin(
        matched_species
    )
].copy()

# Make the phenotype merge key exactly equal to the
# genome species key.

phenotype_matched_for_merge = (
    phenotype_matched.copy()
)

phenotype_matched_for_merge[
    "_species_key"
] = phenotype_matched_for_merge[
    "_match_key"
]

# ------------------------------------------------------------
# Remove helper columns from phenotype
# ------------------------------------------------------------

phenotype_columns_for_merge = [
    "_species_key",
    "Species",
    *PHENOTYPE_COLUMNS,
]

phenotype_for_merge = (
    phenotype_matched_for_merge[
        phenotype_columns_for_merge
    ]
    .copy()
)

# Rename phenotype species so the original genome species
# name remains authoritative in the final ML dataset.

phenotype_for_merge = (
    phenotype_for_merge
    .rename(
        columns={
            "Species": "Phenotype_Source_Species"
        }
    )
)

# ------------------------------------------------------------
# Validate one-to-one
# ------------------------------------------------------------

if phenotype_for_merge[
    "_species_key"
].duplicated().any():

    duplicates = phenotype_for_merge[
        phenotype_for_merge[
            "_species_key"
        ].duplicated(keep=False)
    ]

    raise RuntimeError(
        "Phenotype merge keys are still duplicated:\n"
        + duplicates.to_string(index=False)
    )

if genomic_matched[
    "_species_key"
].duplicated().any():

    raise RuntimeError(
        "Genomic merge keys are duplicated."
    )

# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

reconciled_df = genomic_matched.merge(
    phenotype_for_merge,
    on="_species_key",
    how="inner",
    validate="one_to_one",
)

print()
print(
    f"Matched genomic rows: "
    f"{len(genomic_matched)}"
)

print(
    f"Matched phenotype rows: "
    f"{len(phenotype_for_merge)}"
)

print()
print(
    f"Merged dataset shape: "
    f"{reconciled_df.shape}"
)

# ============================================================
# CLEAN FINAL DATASET
# ============================================================

# Remove internal helper columns
reconciled_df = reconciled_df.drop(
    columns=[
        "_species_key",
        "_accession_key",
        "_match_key",
    ],
    errors="ignore"
)

# ------------------------------------------------------------
# Put phenotype source column near Species
# ------------------------------------------------------------

preferred_front = [
    "Species",
    "Assembly_Accession",
    "Phenotype_Source_Species",
    *PHENOTYPE_COLUMNS,
]

remaining_columns = [
    c for c in reconciled_df.columns
    if c not in preferred_front
]

final_columns = [
    c for c in preferred_front
    if c in reconciled_df.columns
] + remaining_columns

reconciled_df = (
    reconciled_df[
        final_columns
    ]
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

reconciled_df.to_csv(
    MAIN_OUTPUT,
    index=False
)

print()
print(
    "✓ Matched genome–phenotype dataset written."
)

print()
print("Output:")
print(MAIN_OUTPUT)

# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

final_species = (
    reconciled_df["Species"]
    .apply(normalize_species)
)

final_accessions = (
    reconciled_df["Assembly_Accession"]
    .apply(clean_accession)
)

actual_matched_species = (
    final_species.nunique()
)

print()
print("Final ML dataset:")
print(
    f"  Rows:    {len(reconciled_df)}"
)

print(
    f"  Columns: {len(reconciled_df.columns)}"
)

print()
print(
    f"Expected matched species: "
    f"{len(matched_species)}"
)

print(
    f"Actual matched species: "
    f"{actual_matched_species}"
)

if actual_matched_species != len(
    matched_species
):

    raise RuntimeError(
        "Final species count does not match "
        "expected matched species."
    )

if final_species.duplicated().any():

    raise RuntimeError(
        "Duplicate species remain in final ML dataset."
    )

print("✓ Species are unique.")

if final_accessions.duplicated().any():

    raise RuntimeError(
        "Duplicate assembly accessions remain."
    )

print(
    "✓ Assembly accessions are unique."
)

# ============================================================
# MISSING VALUE CHECK
# ============================================================

phenotype_missing = (
    reconciled_df[
        PHENOTYPE_COLUMNS
    ]
    .isna()
    .sum()
    .sum()
)

if phenotype_missing != 0:

    raise RuntimeError(
        f"Final dataset contains "
        f"{phenotype_missing} missing phenotype cells."
    )

print()
print(
    f"Missing phenotype cells: "
    f"{phenotype_missing}"
)

print(
    "✓ No missing phenotype values."
)

# BUSCO missing values

busco_columns_final = [
    c for c in busco_features
    if c in reconciled_df.columns
]

busco_missing = (
    reconciled_df[
        busco_columns_final
    ]
    .isna()
    .sum()
    .sum()
)

if busco_missing != 0:

    raise RuntimeError(
        f"Final dataset contains "
        f"{busco_missing} missing BUSCO cells."
    )

print()
print(
    f"Missing BUSCO feature cells: "
    f"{busco_missing}"
)

print(
    "✓ No missing BUSCO values."
)

# ============================================================
# CHECK SCHWANNIOMYCES MAPPING
# ============================================================

print()
print("=" * 80)
print("EXPLICIT MAPPING VALIDATION")
print("=" * 80)

schwanniomyces_key = normalize_species(
    "Schwanniomyces polymorphus"
)

schwann = reconciled_df[
    final_species == schwanniomyces_key
]

if len(schwann) == 1:

    row = schwann.iloc[0]

    print()
    print(
        "✓ Schwanniomyces polymorphus "
        "is now included."
    )

    print()
    print(
        "Genome species:"
    )

    print(
        f"  {row['Species']}"
    )

    print()
    print(
        "Phenotype source:"
    )

    print(
        f"  {row['Phenotype_Source_Species']}"
    )

    print()
    print(
        "Phenotype values:"
    )

    for col in PHENOTYPE_COLUMNS:

        print(
            f"  {col}: {row[col]}"
        )

else:

    raise RuntimeError(
        "Schwanniomyces polymorphus "
        "was expected to be present exactly once "
        "after explicit mapping."
    )

# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "Metric": "Genomic species",
        "Value": len(genomic_species),
    },
    {
        "Metric": "Original phenotype species",
        "Value": phenotype["_species_key"].nunique(),
    },
    {
        "Metric": "Matched species",
        "Value": actual_matched_species,
    },
    {
        "Metric": "Genome-only species",
        "Value": len(genome_only_species),
    },
    {
        "Metric": "Phenotype-only species",
        "Value": len(phenotype_only_species),
    },
    {
        "Metric": "BUSCO genomic features",
        "Value": len(busco_columns_final),
    },
    {
        "Metric": "Phenotype variables",
        "Value": len(PHENOTYPE_COLUMNS),
    },
    {
        "Metric": "Final ML rows",
        "Value": len(reconciled_df),
    },
    {
        "Metric": "Final ML columns",
        "Value": len(reconciled_df.columns),
    },
    {
        "Metric": "Explicit taxonomic mappings",
        "Value": len(EXPLICIT_MAPPINGS),
    },
])

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("RECONCILIATION COMPLETE")
print("=" * 80)

print()
print(
    f"Genomic species:                "
    f"{len(genomic_species)}"
)

print(
    f"Phenotype species:              "
    f"{phenotype['_species_key'].nunique()}"
)

print(
    f"Matched species:                "
    f"{actual_matched_species}"
)

print(
    f"Genome-only species:            "
    f"{len(genome_only_species)}"
)

print(
    f"Phenotype-only species:         "
    f"{len(phenotype_only_species)}"
)

print(
    f"BUSCO genomic features:         "
    f"{len(busco_columns_final)}"
)

print(
    f"Phenotype variables:            "
    f"{len(PHENOTYPE_COLUMNS)}"
)

print(
    f"Final ML rows:                  "
    f"{len(reconciled_df)}"
)

print(
    f"Final ML columns:               "
    f"{len(reconciled_df.columns)}"
)

print()
print(
    "MAIN ML DATASET:"
)

print(
    MAIN_OUTPUT
)

print()
print(
    "GENOME-ONLY:"
)

print(
    GENOME_ONLY_OUTPUT
)

print()
print(
    "PHENOTYPE-ONLY:"
)

print(
    PHENOTYPE_ONLY_OUTPUT
)

print()
print(
    "AGGREGATION LOG:"
)

print(
    AGGREGATION_LOG_OUTPUT
)

print()
print(
    "TAXONOMIC MAPPINGS:"
)

print(
    TAXON_MAPPING_OUTPUT
)

print()
print(
    "SUMMARY:"
)

print(
    SUMMARY_OUTPUT
)

print()
print(
    "✓ 436 genomic representatives were reconciled."
)

print(
    "✓ Duplicate phenotype species were explicitly handled."
)

print(
    "✓ Schwanniomyces polymorphus was explicitly mapped "
    "to the nominal variety phenotype record."
)

print(
    "✓ No phenotype row was silently discarded."
)

print(
    "✓ Matched dataset contains one row per genomic species."
)

print(
    "✓ BUSCO genomic features were retained."
)

print(
    "✓ Phenotype variables were retained."
)

print(
    "✓ Input files were not modified."
)

print()
print(
    "READY FOR PHYLOGENY-AWARE ML"
)