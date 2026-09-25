# ================================================================
# CARBON BREADTH 420
# TARGETED BUSCO -> GENE / PROTEIN MAPPING
#
# IMPORTANT:
# - Does NOT scan all GFF/GTF files in the project
# - Does NOT use NCBI Entrez
# - Does NOT require an NCBI email
# - Searches only annotation directories relevant to the
#   assemblies represented by the 20 predictive BUSCO candidates
# ================================================================

import os
import sys
import json
import glob
import re
import time
from collections import defaultdict

import pandas as pd
import numpy as np

# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = r"C:\Y1000_chassis_project"

FEATURE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage5_phylogeny_ml_dataset",
    "phylogeny_aware_ml",
    "phylogeny_cv",
    "model_training",
    "feature_interpretation_420"
)

BUSCO_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage4B_phylogeny",
    "busco"
)

TOP20_FILE = os.path.join(
    FEATURE_DIR,
    "tables",
    "Carbon_Breadth_top20_BUSCO_features_420.csv"
)

MAPPING_FILE = os.path.join(
    FEATURE_DIR,
    "functional_annotation_420",
    "busco_gene_mapping_420",
    "tables",
    "Carbon_Breadth_BUSCO_gene_protein_mapping_420.csv"
)

OUT_DIR = os.path.join(
    FEATURE_DIR,
    "functional_annotation_420",
    "targeted_gene_mapping_420"
)

TABLE_DIR = os.path.join(
    OUT_DIR,
    "tables"
)

REPORT_DIR = os.path.join(
    OUT_DIR,
    "reports"
)

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MASTER_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_targeted_BUSCO_gene_protein_mapping_420.csv"
)

SUMMARY_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_targeted_mapping_summary_420.csv"
)

UNMAPPED_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_targeted_unmapped_BUSCOs_420.csv"
)

REPORT_OUT = os.path.join(
    REPORT_DIR,
    "Carbon_Breadth_targeted_mapping_report_420.txt"
)

# ================================================================
# SETTINGS
# ================================================================

# Only annotation file extensions we care about.
ANNOTATION_EXTENSIONS = (
    ".gff",
    ".gff3",
    ".gtf"
)

# Maximum number of annotation files to inspect.
# This is a safety mechanism.
MAX_ANNOTATION_FILES = 5000

# ================================================================
# HELPERS
# ================================================================

def clean(x):

    if pd.isna(x):
        return ""

    return str(x).strip()


def first_existing(df, names):

    for name in names:

        if name in df.columns:
            return name

    return None


def normalize_contig(value):

    value = clean(value)

    if not value:
        return ""

    return value.strip()


def normalize_accession(value):

    value = clean(value)

    if not value:
        return ""

    return value.split()[0]


def parse_attributes(attribute_string):

    """
    Handles both GFF3 and GTF-style attributes.
    """

    attrs = {}

    attribute_string = clean(attribute_string)

    if not attribute_string:
        return attrs

    # ------------------------------------------------------------
    # GFF3:
    # key=value;key=value
    # ------------------------------------------------------------

    if "=" in attribute_string:

        pieces = attribute_string.split(";")

        for piece in pieces:

            piece = piece.strip()

            if not piece:
                continue

            if "=" in piece:

                key, value = piece.split(
                    "=",
                    1
                )

                attrs[key.strip()] = (
                    value.strip()
                    .strip('"')
                )

    # ------------------------------------------------------------
    # GTF:
    # key "value"; key "value";
    # ------------------------------------------------------------

    else:

        pattern = r'(\S+)\s+"([^"]*)"'

        for match in re.finditer(
            pattern,
            attribute_string
        ):

            key = match.group(1).strip()
            value = match.group(2).strip()

            attrs[key] = value

    return attrs


def choose_value(attrs, possible_keys):

    for key in possible_keys:

        if key in attrs:

            value = clean(attrs[key])

            if value:
                return value

    return ""


def coordinate_overlap(
    query_start,
    query_end,
    feature_start,
    feature_end
):

    try:

        query_start = int(query_start)
        query_end = int(query_end)

        feature_start = int(feature_start)
        feature_end = int(feature_end)

    except Exception:

        return False

    return not (
        feature_end < query_start
        or
        feature_start > query_end
    )


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("CARBON BREADTH 420")
print("TARGETED BUSCO → GENE / PROTEIN MAPPING")
print("=" * 80)

print("\nBUSCO root:")
print(BUSCO_ROOT)

# ================================================================
# CHECK INPUTS
# ================================================================

if not os.path.exists(TOP20_FILE):

    raise FileNotFoundError(
        f"\nTop-20 file not found:\n{TOP20_FILE}"
    )


if not os.path.exists(MAPPING_FILE):

    raise FileNotFoundError(
        f"\nMapping file not found:\n{MAPPING_FILE}"
    )


if not os.path.exists(BUSCO_ROOT):

    raise FileNotFoundError(
        f"\nBUSCO root not found:\n{BUSCO_ROOT}"
    )


# ================================================================
# LOAD TOP 20
# ================================================================

print("\n" + "=" * 80)
print("LOADING TOP-20 BUSCO CANDIDATES")
print("=" * 80)

top20 = pd.read_csv(
    TOP20_FILE,
    low_memory=False
)

candidate_col = first_existing(
    top20,
    [
        "Feature",
        "BUSCO_ID"
    ]
)

if candidate_col is None:

    raise KeyError(
        "Could not find Feature or BUSCO_ID in top-20 table."
    )

candidate_buscos = (
    top20[candidate_col]
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

print(
    "Unique candidates:",
    len(candidate_buscos)
)

# ================================================================
# LOAD EXISTING MAPPING
# ================================================================

print("\n" + "=" * 80)
print("LOADING EXISTING BUSCO MAPPING")
print("=" * 80)

mapping = pd.read_csv(
    MAPPING_FILE,
    low_memory=False
)

print(
    "Mapping shape:",
    mapping.shape
)

# ================================================================
# REQUIRED COLUMNS
# ================================================================

required_columns = [
    "BUSCO_ID",
    "Assembly_Accession",
    "Sequence",
    "Start",
    "End",
    "Strand"
]

missing = [
    c
    for c in required_columns
    if c not in mapping.columns
]

if missing:

    raise KeyError(
        "Required columns missing from mapping:\n"
        + "\n".join(missing)
    )

# ================================================================
# FILTER TOP 20
# ================================================================

mapping["BUSCO_ID"] = (
    mapping["BUSCO_ID"]
    .astype(str)
    .str.strip()
)

mapping["Assembly_Accession"] = (
    mapping["Assembly_Accession"]
    .astype(str)
    .str.strip()
)

mapping["Sequence"] = (
    mapping["Sequence"]
    .fillna("")
    .astype(str)
    .str.strip()
)

mapping["Start"] = pd.to_numeric(
    mapping["Start"],
    errors="coerce"
)

mapping["End"] = pd.to_numeric(
    mapping["End"],
    errors="coerce"
)

candidate_mapping = mapping[
    mapping["BUSCO_ID"].isin(candidate_buscos)
].copy()

print(
    "\nRows belonging to top-20 BUSCOs:",
    len(candidate_mapping)
)

# ================================================================
# BUILD TARGET ASSEMBLY LIST
# ================================================================

print("\n" + "=" * 80)
print("BUILDING TARGET ASSEMBLY LIST")
print("=" * 80)

target_assemblies = (
    candidate_mapping[
        "Assembly_Accession"
    ]
    .dropna()
    .astype(str)
    .str.strip()
)

target_assemblies = sorted(
    set(
        x for x in target_assemblies
        if x
        and x.lower() not in {
            "nan",
            "none",
            "na"
        }
    )
)

print(
    "Target assemblies:",
    len(target_assemblies)
)

print("\nExample assemblies:")

for assembly in target_assemblies[:20]:

    print(
        " ",
        assembly
    )

# ================================================================
# BUILD BUSCO QUERY INDEX
# ================================================================

print("\n" + "=" * 80)
print("BUILDING BUSCO COORDINATE INDEX")
print("=" * 80)

queries = defaultdict(list)

for _, row in candidate_mapping.iterrows():

    assembly = clean(
        row["Assembly_Accession"]
    )

    sequence = normalize_contig(
        row["Sequence"]
    )

    if not assembly:
        continue

    if not sequence:
        continue

    queries[assembly].append(
        {
            "BUSCO_ID": clean(
                row["BUSCO_ID"]
            ),
            "Sequence": sequence,
            "Start": row["Start"],
            "End": row["End"],
            "Strand": clean(
                row["Strand"]
            ),
            "Species": clean(
                row.get("Species", "")
            ),
            "Functional_Description": clean(
                row.get(
                    "Functional_Description",
                    ""
                )
            )
        }
    )

print(
    "Assemblies with coordinate queries:",
    len(queries)
)

# ================================================================
# TARGETED DIRECTORY DISCOVERY
# ================================================================

print("\n" + "=" * 80)
print("SEARCHING ONLY TARGET ASSEMBLY DIRECTORIES")
print("=" * 80)

print(
    "This is NOT a global GFF/GTF scan."
)

# ---------------------------------------------------------------
# First look for directories whose names contain assembly IDs.
# ---------------------------------------------------------------

assembly_dir_map = defaultdict(list)

# We prune aggressively.
# We only retain paths that could plausibly correspond to
# one of the target assemblies.

target_set = set(target_assemblies)

visited_dirs = 0

for root, dirs, files in os.walk(BUSCO_ROOT):

    visited_dirs += 1

    # ------------------------------------------------------------
    # Check whether current path contains an assembly accession.
    # ------------------------------------------------------------

    matched_assemblies = []

    normalized_root = root.lower()

    for assembly in target_assemblies:

        if assembly.lower() in normalized_root:

            matched_assemblies.append(
                assembly
            )

    if matched_assemblies:

        for assembly in matched_assemblies:

            if root not in assembly_dir_map[assembly]:

                assembly_dir_map[assembly].append(
                    root
                )

print(
    "\nTarget assembly directories found:",
    sum(
        len(v)
        for v in assembly_dir_map.values()
    )
)

# ================================================================
# SECOND ROUTE:
# SEARCH ONLY FILES UNDER MATCHED ASSEMBLY DIRECTORIES
# ================================================================

print("\n" + "=" * 80)
print("SEARCHING TARGETED ANNOTATION FILES")
print("=" * 80)

annotation_files = []

seen_files = set()

for assembly, directories in assembly_dir_map.items():

    for directory in directories:

        for root, dirs, files in os.walk(directory):

            for filename in files:

                lower = filename.lower()

                if not lower.endswith(
                    ANNOTATION_EXTENSIONS
                ):
                    continue

                path = os.path.join(
                    root,
                    filename
                )

                path = os.path.abspath(path)

                if path in seen_files:
                    continue

                seen_files.add(path)

                annotation_files.append(
                    (
                        assembly,
                        path
                    )
                )

                if len(annotation_files) >= MAX_ANNOTATION_FILES:

                    break

            if len(annotation_files) >= MAX_ANNOTATION_FILES:
                break

        if len(annotation_files) >= MAX_ANNOTATION_FILES:
            break

    if len(annotation_files) >= MAX_ANNOTATION_FILES:
        break


print(
    "Targeted GFF/GTF files found:",
    len(annotation_files)
)

if len(annotation_files) == 0:

    print(
        "\nWARNING:"
    )

    print(
        "No GFF/GTF files were found inside directories"
        " associated with the target assembly IDs."
    )

    print(
        "\nThis does NOT mean the BUSCO candidates are unmapped."
    )

    print(
        "It means the current BUSCO directory structure does not"
        " expose genome annotations under the assembly names."
    )

    print(
        "\nThe script will create an evidence table using the"
        " existing BUSCO functional descriptions and coordinates."
    )

# ================================================================
# TARGETED GFF PARSING
# ================================================================

print("\n" + "=" * 80)
print("MAPPING BUSCO COORDINATES TO GENE / PROTEIN FEATURES")
print("=" * 80)

results = []

processed_files = 0

# ---------------------------------------------------------------
# For each annotation file, only inspect records on sequences
# that are actually required by our BUSCO candidates.
# ---------------------------------------------------------------

for assembly, gff_path in annotation_files:

    processed_files += 1

    print(
        f"[{processed_files}/{len(annotation_files)}] "
        f"{assembly} :: "
        f"{os.path.basename(gff_path)}"
    )

    relevant_queries = queries.get(
        assembly,
        []
    )

    if not relevant_queries:
        continue

    # ------------------------------------------------------------
    # Index queries by contig
    # ------------------------------------------------------------

    query_by_contig = defaultdict(list)

    for query in relevant_queries:

        query_by_contig[
            query["Sequence"]
        ].append(
            query
        )

    try:

        with open(
            gff_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as handle:

            for line in handle:

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                fields = line.rstrip(
                    "\n"
                ).split("\t")

                if len(fields) < 9:
                    continue

                seqid = fields[0]
                feature_type = fields[2]

                try:

                    feature_start = int(
                        fields[3]
                    )

                    feature_end = int(
                        fields[4]
                    )

                except Exception:

                    continue

                strand = fields[6]

                attributes = parse_attributes(
                    fields[8]
                )

                # ------------------------------------------------
                # Only consider gene / transcript / CDS / protein
                # relevant records.
                # ------------------------------------------------

                useful_types = {
                    "gene",
                    "mRNA",
                    "transcript",
                    "CDS",
                    "protein",
                    "ncRNA",
                    "tRNA",
                    "rRNA"
                }

                if feature_type not in useful_types:
                    continue

                # ------------------------------------------------
                # Exact contig matching first.
                # ------------------------------------------------

                candidate_contigs = [
                    seqid
                ]

                if seqid not in query_by_contig:
                    continue

                for query in query_by_contig[seqid]:

                    if not coordinate_overlap(
                        query["Start"],
                        query["End"],
                        feature_start,
                        feature_end
                    ):
                        continue

                    gene_id = choose_value(
                        attributes,
                        [
                            "gene_id",
                            "geneID",
                            "GeneID",
                            "gene"
                        ]
                    )

                    gene_name = choose_value(
                        attributes,
                        [
                            "gene_name",
                            "geneName",
                            "Name",
                            "gene"
                        ]
                    )

                    locus_tag = choose_value(
                        attributes,
                        [
                            "locus_tag",
                            "locusTag"
                        ]
                    )

                    protein_id = choose_value(
                        attributes,
                        [
                            "protein_id",
                            "proteinId",
                            "ProteinID",
                            "protein"
                        ]
                    )

                    transcript_id = choose_value(
                        attributes,
                        [
                            "transcript_id",
                            "transcriptId",
                            "Parent"
                        ]
                    )

                    product = choose_value(
                        attributes,
                        [
                            "product",
                            "Product",
                            "description",
                            "Name"
                        ]
                    )

                    results.append(
                        {
                            "BUSCO_ID":
                                query["BUSCO_ID"],

                            "Assembly_Accession":
                                assembly,

                            "Sequence":
                                query["Sequence"],

                            "BUSCO_Start":
                                query["Start"],

                            "BUSCO_End":
                                query["End"],

                            "BUSCO_Strand":
                                query["Strand"],

                            "Feature_Type":
                                feature_type,

                            "Feature_Start":
                                feature_start,

                            "Feature_End":
                                feature_end,

                            "Feature_Strand":
                                strand,

                            "Gene_ID":
                                gene_id,

                            "Gene_Name":
                                gene_name,

                            "Locus_Tag":
                                locus_tag,

                            "Protein_ID":
                                protein_id,

                            "Transcript_ID":
                                transcript_id,

                            "Product":
                                product,

                            "Source_GFF":
                                gff_path,

                            "Functional_Description":
                                query[
                                    "Functional_Description"
                                ]
                        }
                    )

    except Exception as e:

        print(
            "  Could not parse:",
            str(e)
        )

# ================================================================
# CREATE ANNOTATION DATAFRAME
# ================================================================

annotation_df = pd.DataFrame(
    results
)

print("\n" + "=" * 80)
print("TARGETED MAPPING RESULT")
print("=" * 80)

print(
    "Raw annotation matches:",
    len(annotation_df)
)

if len(annotation_df) > 0:

    # Remove exact duplicates
    annotation_df = (
        annotation_df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(
        "Unique mapping records:",
        len(annotation_df)
    )

else:

    annotation_df = pd.DataFrame(
        columns=[
            "BUSCO_ID",
            "Assembly_Accession",
            "Sequence",
            "BUSCO_Start",
            "BUSCO_End",
            "BUSCO_Strand",
            "Feature_Type",
            "Feature_Start",
            "Feature_End",
            "Feature_Strand",
            "Gene_ID",
            "Gene_Name",
            "Locus_Tag",
            "Protein_ID",
            "Transcript_ID",
            "Product",
            "Source_GFF",
            "Functional_Description"
        ]
    )

# ================================================================
# MERGE PREDICTIVE INFORMATION
# ================================================================

print("\n" + "=" * 80)
print("ADDING PREDICTIVE MODEL INFORMATION")
print("=" * 80)

predictive_columns = [
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

available_predictive = [
    c
    for c in predictive_columns
    if c in top20.columns
]

predictive = top20[
    [candidate_col] + available_predictive
].copy()

predictive = predictive.rename(
    columns={
        candidate_col: "BUSCO_ID"
    }
)

if len(annotation_df) > 0:

    annotation_df = annotation_df.merge(
        predictive,
        on="BUSCO_ID",
        how="left"
    )

# ================================================================
# BUSCO-LEVEL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("CREATING BUSCO-LEVEL SUMMARY")
print("=" * 80)

summary_rows = []

for busco in candidate_buscos:

    busco_rows = annotation_df[
        annotation_df["BUSCO_ID"] == busco
    ]

    original_rows = candidate_mapping[
        candidate_mapping["BUSCO_ID"] == busco
    ]

    row = {
        "BUSCO_ID": busco,

        "Assembly_Count":
            original_rows[
                "Assembly_Accession"
            ].nunique(),

        "Sequence_Count":
            original_rows[
                "Sequence"
            ].nunique(),

        "Annotation_Matches":
            len(busco_rows),

        "Gene_ID_Count":
            busco_rows[
                "Gene_ID"
            ].replace("", pd.NA).dropna().nunique()
            if len(busco_rows)
            else 0,

        "Gene_Name_Count":
            busco_rows[
                "Gene_Name"
            ].replace("", pd.NA).dropna().nunique()
            if len(busco_rows)
            else 0,

        "Locus_Tag_Count":
            busco_rows[
                "Locus_Tag"
            ].replace("", pd.NA).dropna().nunique()
            if len(busco_rows)
            else 0,

        "Protein_ID_Count":
            busco_rows[
                "Protein_ID"
            ].replace("", pd.NA).dropna().nunique()
            if len(busco_rows)
            else 0,

        "Product_Count":
            busco_rows[
                "Product"
            ].replace("", pd.NA).dropna().nunique()
            if len(busco_rows)
            else 0,

        "Functional_Description":
            clean(
                original_rows[
                    "Functional_Description"
                ].iloc[0]
            )
            if "Functional_Description"
            in original_rows.columns
            and len(original_rows)
            else ""
    }

    # Add predictive values
    for col in available_predictive:

        value = top20.loc[
            top20[candidate_col].astype(str).str.strip()
            == busco,
            col
        ]

        row[col] = (
            value.iloc[0]
            if len(value)
            else None
        )

    summary_rows.append(row)

busco_summary = pd.DataFrame(
    summary_rows
)

# ================================================================
# SAVE OUTPUTS
# ================================================================

annotation_df.to_csv(
    MASTER_OUT,
    index=False
)

busco_summary.to_csv(
    SUMMARY_OUT,
    index=False
)

# ================================================================
# UNMAPPED BUSCOs
# ================================================================

mapped_buscos = set(
    annotation_df[
        "BUSCO_ID"
    ].unique()
)

unmapped_buscos = [
    b
    for b in candidate_buscos
    if b not in mapped_buscos
]

unmapped_df = busco_summary[
    busco_summary["BUSCO_ID"].isin(
        unmapped_buscos
    )
].copy()

unmapped_df.to_csv(
    UNMAPPED_OUT,
    index=False
)

# ================================================================
# FINAL STATISTICS
# ================================================================

mapped_count = len(
    mapped_buscos
)

unmapped_count = len(
    unmapped_buscos
)

gene_records = 0
gene_ids = 0
gene_names = 0
protein_ids = 0
locus_tags = 0

if len(annotation_df) > 0:

    gene_records = int(
        (
            annotation_df["Gene_ID"]
            .fillna("")
            .astype(str)
            .str.strip()
            != ""
        ).sum()
    )

    gene_ids = annotation_df[
        "Gene_ID"
    ].replace(
        "",
        pd.NA
    ).dropna().nunique()

    gene_names = annotation_df[
        "Gene_Name"
    ].replace(
        "",
        pd.NA
    ).dropna().nunique()

    protein_ids = annotation_df[
        "Protein_ID"
    ].replace(
        "",
        pd.NA
    ).dropna().nunique()

    locus_tags = annotation_df[
        "Locus_Tag"
    ].replace(
        "",
        pd.NA
    ).dropna().nunique()

# ================================================================
# REPORT
# ================================================================

report = f"""
CARBON BREADTH 420
TARGETED BUSCO → GENE / PROTEIN MAPPING
========================================

Total BUSCO candidates:
{len(candidate_buscos)}

Target assemblies:
{len(target_assemblies)}

Targeted annotation files found:
{len(annotation_files)}

Annotation files processed:
{processed_files}

BUSCO candidates mapped to annotation:
{mapped_count}

BUSCO candidates without annotation match:
{unmapped_count}

Raw annotation mapping records:
{len(annotation_df)}

Unique Gene IDs:
{gene_ids}

Unique Gene names:
{gene_names}

Unique Locus tags:
{locus_tags}

Unique Protein IDs:
{protein_ids}

Gene-containing annotation records:
{gene_records}

IMPORTANT:

This analysis did NOT use NCBI Entrez and does NOT require
an NCBI email.

The script searched only annotation directories associated
with the assemblies represented by the 20 predictive BUSCOs.

BUSCO functional descriptions are retained independently
of gene/protein mapping.

No GO, KEGG, InterPro or EC annotation is assigned by this
script unless a genuine gene/protein feature is recovered.

Next stage:

BUSCO
  ↓
assembly-specific gene/protein identifier
  ↓
UniProt / InterPro / KEGG / GO
  ↓
pathway annotation
  ↓
pathway grouping
  ↓
network analysis
"""

with open(
    REPORT_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n" + "=" * 80)
print("TARGETED BUSCO → GENE / PROTEIN MAPPING COMPLETE")
print("=" * 80)

print(
    "\nTotal BUSCO candidates       :",
    len(candidate_buscos)
)

print(
    "Target assemblies            :",
    len(target_assemblies)
)

print(
    "Targeted GFF/GTF files       :",
    len(annotation_files)
)

print(
    "BUSCO candidates mapped      :",
    mapped_count
)

print(
    "BUSCO candidates unmapped    :",
    unmapped_count
)

print(
    "Mapping records              :",
    len(annotation_df)
)

print(
    "Unique gene IDs              :",
    gene_ids
)

print(
    "Unique gene names            :",
    gene_names
)

print(
    "Unique locus tags            :",
    locus_tags
)

print(
    "Unique protein IDs           :",
    protein_ids
)

print("\n" + "=" * 80)
print("OUTPUTS")
print("=" * 80)

print("\nMaster mapping:")
print(MASTER_OUT)

print("\nBUSCO summary:")
print(SUMMARY_OUT)

print("\nUnmapped BUSCOs:")
print(UNMAPPED_OUT)

print("\nReport:")
print(REPORT_OUT)

print("\n" + "=" * 80)
print("NEXT STEP")
print("=" * 80)

print(
    """
Do NOT run a global GFF/GTF scan.

If gene/protein IDs are recovered:
    BUSCO → gene/protein → GO/InterPro/EC/KEGG

If no gene/protein IDs are recovered:
    inspect the exact assembly annotation location
    for the affected assemblies before proceeding.
"""
)

print("=" * 80)