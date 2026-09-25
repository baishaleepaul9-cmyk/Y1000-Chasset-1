from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
import time

# ============================================================
# CARBON BREADTH
# BUSCO -> LOCUS / PROTEIN / FUNCTIONAL IDENTITY MAPPING
#
# IMPORTANT:
# This script does NOT perform a global GFF/GTF search.
# It uses the BUSCO full_table.tsv files that have already
# been successfully identified in the project.
#
# It NEVER invents gene names.
# If a true gene/protein identifier is unavailable in BUSCO,
# the BUSCO functional description is retained as the
# functional protein identity.
# ============================================================

START = time.time()

ROOT = Path(r"C:\Y1000_chassis_project")

# ------------------------------------------------------------
# EXISTING TOP-20 FEATURE TABLE
# ------------------------------------------------------------

TOP20 = (
    ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
    / "tables"
    / "Carbon_Breadth_top20_BUSCO_features_420.csv"
)

# ------------------------------------------------------------
# BUSCO ROOT
# ------------------------------------------------------------

BUSCO_ROOT = (
    ROOT
    / "results"
    / "stage4B_phylogeny"
    / "busco"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUT_ROOT = (
    ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
    / "functional_annotation_420"
    / "busco_gene_mapping_420"
)

TABLE_DIR = OUT_ROOT / "tables"
REPORT_DIR = OUT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MAPPING_OUT = (
    TABLE_DIR /
    "Carbon_Breadth_BUSCO_gene_protein_mapping_420.csv"
)

SUMMARY_OUT = (
    TABLE_DIR /
    "Carbon_Breadth_BUSCO_gene_mapping_summary_420.csv"
)

CANDIDATE_OUT = (
    TABLE_DIR /
    "Carbon_Breadth_gene_protein_candidates_420.csv"
)

UNMAPPED_OUT = (
    TABLE_DIR /
    "Carbon_Breadth_unmapped_BUSCO_candidates_420.csv"
)

JSON_OUT = (
    TABLE_DIR /
    "Carbon_Breadth_BUSCO_gene_mapping_summary_420.json"
)

REPORT_OUT = (
    REPORT_DIR /
    "Carbon_Breadth_BUSCO_gene_mapping_report_420.txt"
)


# ============================================================
# HELPERS
# ============================================================

def clean(value):
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def extract_busco_id(value):
    value = clean(value)

    if not value:
        return ""

    # Normal BUSCO form:
    # 36839at4891
    m = re.search(r"\b\d+at\d+\b", value)

    if m:
        return m.group(0)

    return value


def parse_busco_table(path):
    """
    Parse BUSCO full_table.tsv.

    BUSCO normally has:
    0 BUSCO_ID
    1 Status
    2 Sequence
    3 Start
    4 End
    5 Strand
    6 Score
    7 Length
    8 OrthoDB_URL
    9 Functional_Description

    Some files may contain comments/header lines.
    """

    records = []

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:

            for line in handle:

                line = line.rstrip("\n\r")

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                fields = line.split("\t")

                if len(fields) < 2:
                    continue

                busco_id = extract_busco_id(fields[0])

                if not busco_id:
                    continue

                # Only process candidate BUSCOs
                if busco_id not in CANDIDATES:
                    continue

                status = clean(fields[1])

                sequence = clean(fields[2]) if len(fields) > 2 else ""

                start = clean(fields[3]) if len(fields) > 3 else ""

                end = clean(fields[4]) if len(fields) > 4 else ""

                strand = clean(fields[5]) if len(fields) > 5 else ""

                score = clean(fields[6]) if len(fields) > 6 else ""

                length = clean(fields[7]) if len(fields) > 7 else ""

                orthodb = clean(fields[8]) if len(fields) > 8 else ""

                description = (
                    clean(fields[9])
                    if len(fields) > 9
                    else ""
                )

                records.append({
                    "BUSCO_ID": busco_id,
                    "Status": status,
                    "Sequence": sequence,
                    "Start": start,
                    "End": end,
                    "Strand": strand,
                    "Score": score,
                    "Length": length,
                    "OrthoDB_URL": orthodb,
                    "Functional_Description": description
                })

    except Exception as exc:

        print(
            f"Could not parse BUSCO table: "
            f"{path}\n  {exc}"
        )

    return records


# ============================================================
# LOAD TOP-20
# ============================================================

print("=" * 80)
print("CARBON BREADTH — BUSCO → GENE / PROTEIN FUNCTIONAL MAPPING")
print("=" * 80)

print("\nLoading top-20 BUSCO feature table:")
print(TOP20)

if not TOP20.exists():
    raise FileNotFoundError(
        f"\nTop-20 feature table not found:\n{TOP20}"
    )

top20 = pd.read_csv(TOP20)

if "Feature" not in top20.columns:
    raise KeyError(
        "The top-20 table does not contain the 'Feature' column."
    )

top20["BUSCO_ID"] = top20["Feature"].astype(str).map(
    extract_busco_id
)

top20 = top20[
    top20["BUSCO_ID"].ne("")
].copy()

top20 = top20.drop_duplicates(
    subset=["BUSCO_ID"]
)

CANDIDATES = set(top20["BUSCO_ID"])

print("\nUnique candidates:", len(CANDIDATES))

print("\nCandidates:")
for x in sorted(CANDIDATES):
    print(" ", x)


# ============================================================
# SEARCH ONLY FULL_TABLE.TSV
# ============================================================

print("\n" + "=" * 80)
print("SEARCHING BUSCO FULL_TABLE.TSV FILES")
print("=" * 80)

print("\nBUSCO root:")
print(BUSCO_ROOT)

if not BUSCO_ROOT.exists():
    raise FileNotFoundError(
        f"\nBUSCO directory not found:\n{BUSCO_ROOT}"
    )

# This is the ONLY recursive search.
# It searches for full_table.tsv, not GFF/GTF.
full_tables = list(
    BUSCO_ROOT.rglob("full_table.tsv")
)

print(
    f"\nFull BUSCO tables found: "
    f"{len(full_tables)}"
)

if not full_tables:
    raise FileNotFoundError(
        "No full_table.tsv files found."
    )


# ============================================================
# PARSE
# ============================================================

all_records = []

tables_with_candidates = 0

for i, table in enumerate(full_tables, start=1):

    records = parse_busco_table(table)

    if records:

        tables_with_candidates += 1

        # ----------------------------------------------------
        # Recover species / assembly from directory name
        # ----------------------------------------------------

        # Example:
        #
        # Ambrosiozyma_llanquihuensis__GCA_030568425.1
        #
        # run_saccharomycetes_odb10
        #

        busco_run_dir = table.parent

        assembly_dir = busco_run_dir.parent

        assembly_folder = assembly_dir.name

        assembly_accession = ""

        m = re.search(
            r"(GCA_\d+\.\d+|GCF_\d+\.\d+)",
            assembly_folder
        )

        if m:
            assembly_accession = m.group(1)

        species = assembly_folder

        if "__" in assembly_folder:
            species = assembly_folder.split(
                "__"
            )[0]

        for record in records:

            record["Species"] = species

            record["Assembly_Accession"] = (
                assembly_accession
            )

            record["Assembly_Folder"] = (
                assembly_folder
            )

            record["BUSCO_Run_Directory"] = (
                str(busco_run_dir)
            )

            record["Full_Table"] = str(table)

            all_records.append(record)

    if i % 50 == 0:

        elapsed = time.time() - START

        print(
            f"Processed {i}/{len(full_tables)} "
            f"tables ({elapsed:.1f}s)"
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

mapping = pd.DataFrame(all_records)

print("\n" + "=" * 80)
print("BUSCO TABLE SEARCH COMPLETE")
print("=" * 80)

print(
    "\nFull tables searched        :",
    len(full_tables)
)

print(
    "Tables containing candidates:",
    tables_with_candidates
)

print(
    "Mapping records             :",
    len(mapping)
)


if mapping.empty:

    raise RuntimeError(
        "\nNo candidate BUSCOs were recovered."
    )


# ============================================================
# ADD MODEL INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("ADDING PREDICTIVE MODEL INFORMATION")
print("=" * 80)

model_cols = [
    "BUSCO_ID",
    "Phenotype",
    "Feature_Set",
    "Model",
    "Mean_Importance",
    "SD_Importance",
    "Median_Importance",
    "Min_Importance",
    "Max_Importance",
    "Mean_Baseline_R2",
    "Positive_Importance_Folds",
    "Fold_Stability",
    "Importance_Rank"
]

available_model_cols = [
    c for c in model_cols
    if c in top20.columns
]

model_info = top20[
    available_model_cols
].copy()

mapping = mapping.merge(
    model_info,
    on="BUSCO_ID",
    how="left",
    suffixes=("", "_MODEL")
)


# ============================================================
# TRUE GENE/PROTEIN IDS
# ============================================================

print("\n" + "=" * 80)
print("DETERMINING GENE / PROTEIN IDENTIFIERS")
print("=" * 80)

# BUSCO full_table.tsv generally DOES NOT contain a gene ID
# or protein accession.
#
# Therefore:
# Gene_ID = blank unless genuinely available.
# Protein_ID = blank unless genuinely available.
#
# We do NOT convert a contig accession into a fake gene ID.

mapping["Gene_ID"] = ""

mapping["Protein_ID"] = ""

mapping["Locus_Tag"] = ""

mapping["Gene_Name"] = ""

mapping["Protein_Name"] = ""


# ============================================================
# FUNCTIONAL IDENTITY
# ============================================================

mapping["Functional_Protein_Description"] = (
    mapping["Functional_Description"]
)

mapping["Functional_Identity_Source"] = np.where(
    mapping["Functional_Description"].fillna("").astype(str).str.strip().ne(""),
    "BUSCO functional description",
    "BUSCO ID only"
)


# ============================================================
# NORMALIZE IMPORTANT FIELDS
# ============================================================

for col in [
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
    "Importance_Rank"
]:

    if col in mapping.columns:

        mapping[col] = pd.to_numeric(
            mapping[col],
            errors="coerce"
        )


# ============================================================
# UNIQUE BUSCO SUMMARY
# ============================================================

summary_rows = []

for busco in sorted(CANDIDATES):

    subset = mapping[
        mapping["BUSCO_ID"] == busco
    ].copy()

    candidate_model = top20[
        top20["BUSCO_ID"] == busco
    ]

    description = ""

    if not subset.empty:

        descs = (
            subset["Functional_Description"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        descs = [
            x for x in descs
            if x and x.lower() != "nan"
        ]

        if descs:

            # Most common description
            description = (
                pd.Series(descs)
                .value_counts()
                .index[0]
            )

    importance = np.nan
    stability = np.nan
    rank = np.nan

    if not candidate_model.empty:

        if "Mean_Importance" in candidate_model:
            importance = candidate_model[
                "Mean_Importance"
            ].iloc[0]

        if "Fold_Stability" in candidate_model:
            stability = candidate_model[
                "Fold_Stability"
            ].iloc[0]

        if "Importance_Rank" in candidate_model:
            rank = candidate_model[
                "Importance_Rank"
            ].iloc[0]

    summary_rows.append({

        "BUSCO_ID": busco,

        "Genomes_Found": len(subset),

        "Complete_Count": (
            (subset["Status"] == "Complete").sum()
            if not subset.empty
            else 0
        ),

        "Duplicated_Count": (
            (subset["Status"] == "Duplicated").sum()
            if not subset.empty
            else 0
        ),

        "Fragmented_Count": (
            (subset["Status"] == "Fragmented").sum()
            if not subset.empty
            else 0
        ),

        "Missing_Count": (
            (subset["Status"] == "Missing").sum()
            if not subset.empty
            else 0
        ),

        "Unique_Assemblies": (
            subset["Assembly_Accession"]
            .nunique()
            if not subset.empty
            else 0
        ),

        "Unique_Sequences": (
            subset["Sequence"]
            .nunique()
            if not subset.empty
            else 0
        ),

        "Gene_ID_Available": False,

        "Protein_ID_Available": False,

        "Gene_Name_Available": False,

        "Functional_Description": description,

        "Mean_Importance": importance,

        "Fold_Stability": stability,

        "Importance_Rank": rank

    })


summary = pd.DataFrame(summary_rows)


# ============================================================
# UNMAPPED BUSCOs
# ============================================================

unmapped = summary[
    summary["Genomes_Found"] == 0
].copy()


# ============================================================
# UNIQUE FUNCTIONAL CANDIDATE TABLE
# ============================================================

candidate_table = (
    summary[
        [
            "BUSCO_ID",
            "Functional_Description",
            "Mean_Importance",
            "Fold_Stability",
            "Importance_Rank",
            "Genomes_Found",
            "Unique_Assemblies",
            "Unique_Sequences",
            "Gene_ID_Available",
            "Protein_ID_Available",
            "Gene_Name_Available"
        ]
    ]
    .copy()
)


# ============================================================
# SAVE
# ============================================================

mapping.to_csv(
    MAPPING_OUT,
    index=False
)

summary.to_csv(
    SUMMARY_OUT,
    index=False
)

candidate_table.to_csv(
    CANDIDATE_OUT,
    index=False
)

unmapped.to_csv(
    UNMAPPED_OUT,
    index=False
)


# ============================================================
# JSON
# ============================================================

json_summary = {

    "total_busco_candidates": int(
        len(CANDIDATES)
    ),

    "full_tables_searched": int(
        len(full_tables)
    ),

    "tables_containing_candidates": int(
        tables_with_candidates
    ),

    "busco_candidates_mapped": int(
        summary[
            summary["Genomes_Found"] > 0
        ].shape[0]
    ),

    "busco_candidates_unmapped": int(
        len(unmapped)
    ),

    "mapping_records": int(
        len(mapping)
    ),

    "unique_species": int(
        mapping["Species"].nunique()
    ),

    "unique_assemblies": int(
        mapping["Assembly_Accession"].nunique()
    ),

    "unique_sequence_ids": int(
        mapping["Sequence"].replace(
            "",
            np.nan
        ).nunique()
    ),

    "unique_gene_ids": 0,

    "unique_protein_ids": 0,

    "unique_gene_names": 0,

    "functional_descriptions_available": int(
        mapping[
            "Functional_Description"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    ),

    "important_note": (
        "BUSCO full_table.tsv provides BUSCO hit coordinates "
        "and functional descriptions but does not reliably "
        "provide gene IDs, locus tags, protein accessions, "
        "or gene names. These identifiers were therefore "
        "not inferred or invented."
    )

}

with open(
    JSON_OUT,
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        json_summary,
        handle,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

elapsed = time.time() - START

with open(
    REPORT_OUT,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "CARBON BREADTH — BUSCO GENE/PROTEIN MAPPING REPORT\n"
    )

    handle.write("=" * 70 + "\n\n")

    handle.write(
        f"Total BUSCO candidates: "
        f"{len(CANDIDATES)}\n"
    )

    handle.write(
        f"Full tables searched: "
        f"{len(full_tables)}\n"
    )

    handle.write(
        f"Tables containing candidates: "
        f"{tables_with_candidates}\n"
    )

    handle.write(
        f"BUSCO candidates mapped: "
        f"{len(CANDIDATES) - len(unmapped)}\n"
    )

    handle.write(
        f"BUSCO candidates unmapped: "
        f"{len(unmapped)}\n"
    )

    handle.write(
        f"Mapping records: "
        f"{len(mapping)}\n"
    )

    handle.write(
        f"Unique species: "
        f"{mapping['Species'].nunique()}\n"
    )

    handle.write(
        f"Unique assemblies: "
        f"{mapping['Assembly_Accession'].nunique()}\n"
    )

    handle.write(
        f"Unique sequence IDs: "
        f"{mapping['Sequence'].replace('', np.nan).nunique()}\n"
    )

    handle.write(
        "\nGene IDs mapped: 0\n"
    )

    handle.write(
        "Protein IDs mapped: 0\n"
    )

    handle.write(
        "Gene names mapped: 0\n"
    )

    handle.write(
        "\nIMPORTANT:\n"
    )

    handle.write(
        "The BUSCO full_table.tsv format does not reliably "
        "contain gene IDs, protein accessions, or gene names. "
        "The script therefore retains the BUSCO functional "
        "description rather than inventing gene identities.\n"
    )

    handle.write(
        "\nFunctional descriptions can now be used for "
        "candidate-level interpretation, while true "
        "gene/protein identifiers require an independent "
        "genome annotation source.\n"
    )

    handle.write(
        f"\nRuntime: {elapsed:.2f} seconds\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("BUSCO → GENE / PROTEIN FUNCTIONAL MAPPING COMPLETE")
print("=" * 80)

print(
    "\nTotal BUSCO candidates       :",
    len(CANDIDATES)
)

print(
    "Full tables searched         :",
    len(full_tables)
)

print(
    "Tables containing candidates:",
    tables_with_candidates
)

print(
    "BUSCO candidates mapped      :",
    len(CANDIDATES) - len(unmapped)
)

print(
    "BUSCO candidates unmapped    :",
    len(unmapped)
)

print(
    "Mapping records              :",
    len(mapping)
)

print(
    "Unique species               :",
    mapping["Species"].nunique()
)

print(
    "Unique assemblies            :",
    mapping["Assembly_Accession"].nunique()
)

print(
    "Unique sequence IDs          :",
    mapping["Sequence"]
    .replace("", np.nan)
    .nunique()
)

print(
    "Unique gene IDs              : 0"
)

print(
    "Unique protein IDs           : 0"
)

print(
    "Unique gene names            : 0"
)

print(
    "Functional descriptions      :",
    mapping[
        "Functional_Description"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
    .sum()
)

print("\n" + "=" * 80)
print("OUTPUTS")
print("=" * 80)

print("\nBUSCO mapping:")
print(MAPPING_OUT)

print("\nBUSCO summary:")
print(SUMMARY_OUT)

print("\nCandidate functional table:")
print(CANDIDATE_OUT)

print("\nUnmapped BUSCOs:")
print(UNMAPPED_OUT)

print("\nJSON:")
print(JSON_OUT)

print("\nReport:")
print(REPORT_OUT)

print("\n" + "=" * 80)
print("NEXT STEP")
print("=" * 80)

print("""
BUSCO
  ↓
BUSCO functional identity
  ↓
candidate functional grouping
  ↓
GO / InterPro / EC / KEGG
  ↓
pathway grouping
  ↓
network analysis
  ↓
candidate chassis interpretation

TRUE gene/protein identifiers were NOT fabricated.
""")

print(
    f"\nRuntime: {elapsed:.2f} seconds"
)

print("\nDONE.")