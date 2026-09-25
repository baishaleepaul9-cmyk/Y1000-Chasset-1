# ================================================================
# BUSCO -> GENE / PROTEIN MAPPING
# CARBON BREADTH — Y1000+ PROJECT
# ================================================================
#
# Purpose:
#   Map the 20 predictive BUSCO candidates to the actual
#   gene/protein/sequence identifiers available in the BUSCO
#   results and BUSCO lineage dataset.
#
# IMPORTANT:
#   BUSCO IDs are ortholog-group identifiers, NOT necessarily
#   gene names. Therefore this script preserves:
#
#       BUSCO_ID
#       Species
#       Assembly
#       Status
#       Sequence / Gene ID
#       Functional description
#
#   It does NOT discard a BUSCO merely because a gene name
#   cannot immediately be assigned.
#
# ================================================================

import os
import re
import csv
import json
import gzip
import time
from pathlib import Path

import pandas as pd


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")

BUSCO_ROOT = (
    PROJECT_ROOT
    / "results"
    / "stage4B_phylogeny"
    / "busco"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
)

TOP20_FILE = (
    FEATURE_ROOT
    / "tables"
    / "Carbon_Breadth_top20_BUSCO_features_420.csv"
)

ANNOTATION_FILE = (
    FEATURE_ROOT
    / "functional_annotation_420"
    / "tables"
    / "Carbon_Breadth_BUSCO_functional_annotation_420.csv"
)

OUTPUT_ROOT = (
    FEATURE_ROOT
    / "functional_annotation_420"
    / "busco_gene_mapping_420"
)

TABLE_ROOT = OUTPUT_ROOT / "tables"
REPORT_ROOT = OUTPUT_ROOT / "reports"

TABLE_ROOT.mkdir(parents=True, exist_ok=True)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def clean_string(value):
    """Convert NaN/None into empty string."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def safe_read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")

    return pd.read_csv(path)


def normalize_busco_id(value):
    """
    Normalize BUSCO identifiers while preserving the original
    identifier format.
    """
    value = clean_string(value)

    if not value:
        return ""

    value = value.strip()

    # Remove accidental whitespace
    value = re.sub(r"\s+", "", value)

    return value


def extract_busco_id_from_text(text):
    """
    Look for BUSCO IDs such as:
        36839at4891
        5746at4891
    """

    text = clean_string(text)

    if not text:
        return None

    match = re.search(
        r"\b\d+at\d+\b",
        text
    )

    if match:
        return match.group(0)

    return None


def safe_read_tsv(path):
    """
    BUSCO full_table.tsv is tab-delimited and may contain
    comment lines.
    """

    try:
        return pd.read_csv(
            path,
            sep="\t",
            comment="#",
            dtype=str,
            header=None
        )
    except Exception:
        return None


# ================================================================
# HEADER DETECTION FOR BUSCO FULL TABLE
# ================================================================

def read_busco_full_table(path):

    # First attempt: BUSCO standard header handling
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            dtype=str
        )

        if len(df.columns) >= 2:

            # Detect whether BUSCO column exists
            possible_busco_columns = [
                c for c in df.columns
                if str(c).lower() in {
                    "busco id",
                    "busco_id",
                    "buscoid",
                    "busco"
                }
            ]

            if possible_busco_columns:
                return df

    except Exception:
        pass

    # Second approach: no header
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            dtype=str,
            header=None
        )

        # Standard BUSCO full_table layout:
        #
        # BUSCO ID
        # Status
        # Sequence
        # Start
        # End
        # Strand
        # Score
        # Length
        # OrthoDB URL
        # Description
        #
        if df.shape[1] >= 2:

            ncols = df.shape[1]

            standard_names = [
                "BUSCO_ID",
                "Status",
                "Sequence",
                "Start",
                "End",
                "Strand",
                "Score",
                "Length",
                "OrthoDB_URL",
                "Functional_Description"
            ]

            names = []

            for i in range(ncols):
                if i < len(standard_names):
                    names.append(standard_names[i])
                else:
                    names.append(f"Column_{i+1}")

            df.columns = names

            return df

    except Exception as e:
        print(f"Could not parse BUSCO table: {e}")

    return None


# ================================================================
# DISCOVER BUSCO FULL TABLES
# ================================================================

def discover_full_tables():

    print("=" * 80)
    print("SEARCHING BUSCO RESULT TABLES")
    print("=" * 80)

    if not BUSCO_ROOT.exists():
        raise FileNotFoundError(
            f"BUSCO root does not exist:\n{BUSCO_ROOT}"
        )

    tables = []

    # Only search for full_table.tsv.
    # This is much faster than searching every file in the project.
    for path in BUSCO_ROOT.rglob("full_table.tsv"):

        tables.append(path)

    print()
    print(f"BUSCO full_table.tsv files found: {len(tables)}")

    return tables


# ================================================================
# LOAD TOP-20 BUSCO CANDIDATES
# ================================================================

print()
print("=" * 80)
print("CARBON BREADTH — BUSCO → GENE / PROTEIN MAPPING")
print("=" * 80)

print()
print("Project:")
print(PROJECT_ROOT)

print()
print("BUSCO root:")
print(BUSCO_ROOT)

print()
print("Top-20 feature file:")
print(TOP20_FILE)


top20 = safe_read_csv(TOP20_FILE)

if "Feature" not in top20.columns:
    raise KeyError(
        "Top-20 feature file does not contain 'Feature'."
    )

top20["BUSCO_ID"] = (
    top20["Feature"]
    .astype(str)
    .map(normalize_busco_id)
)

top20 = top20[
    top20["BUSCO_ID"] != ""
].copy()

top20 = top20.drop_duplicates(
    subset=["BUSCO_ID"]
)

candidate_buscos = top20["BUSCO_ID"].tolist()

print()
print("=" * 80)
print("TOP-20 BUSCO CANDIDATES")
print("=" * 80)

for i, busco in enumerate(candidate_buscos, 1):
    print(f"{i:2d}. {busco}")

print()
print(f"Unique candidates: {len(candidate_buscos)}")


# ================================================================
# LOAD EXISTING FUNCTIONAL ANNOTATION
# ================================================================

annotation_df = None

if ANNOTATION_FILE.exists():

    print()
    print("=" * 80)
    print("LOADING EXISTING BUSCO FUNCTIONAL ANNOTATION")
    print("=" * 80)

    try:
        annotation_df = pd.read_csv(
            ANNOTATION_FILE,
            dtype=str
        )

        print(
            f"Existing annotation shape: "
            f"{annotation_df.shape}"
        )

    except Exception as e:

        print(
            "Warning: could not load existing annotation:"
        )
        print(e)


# ================================================================
# DISCOVER BUSCO FULL TABLES
# ================================================================

full_tables = discover_full_tables()


# ================================================================
# MAP BUSCO IDs THROUGH FULL_TABLE.TSV
# ================================================================

print()
print("=" * 80)
print("MAPPING BUSCO IDs THROUGH FULL_TABLE.TSV")
print("=" * 80)

candidate_set = set(candidate_buscos)

mapping_rows = []

processed_tables = 0
matched_tables = 0

start_time = time.time()

for table_path in full_tables:

    processed_tables += 1

    if processed_tables % 25 == 0:

        elapsed = time.time() - start_time

        print(
            f"Processed {processed_tables}/"
            f"{len(full_tables)} tables "
            f"({elapsed:.1f}s)"
        )

    df = read_busco_full_table(table_path)

    if df is None:
        continue

    if df.empty:
        continue

    # ------------------------------------------------------------
    # Identify BUSCO column
    # ------------------------------------------------------------

    busco_col = None

    for c in df.columns:

        c_lower = str(c).lower()

        if (
            "busco" in c_lower
            or c_lower == "busco id"
            or c_lower == "busco_id"
        ):
            busco_col = c
            break

    # If no obvious column exists, assume first column
    if busco_col is None:

        if len(df.columns) >= 1:
            busco_col = df.columns[0]
        else:
            continue

    # ------------------------------------------------------------
    # Identify relevant columns
    # ------------------------------------------------------------

    status_col = None
    sequence_col = None
    start_col = None
    end_col = None
    strand_col = None
    score_col = None
    length_col = None
    url_col = None
    desc_col = None

    for c in df.columns:

        cl = str(c).lower()

        if status_col is None and "status" in cl:
            status_col = c

        if sequence_col is None and (
            "sequence" in cl
            or "gene" in cl
            or "protein" in cl
            or cl == "contig"
        ):
            sequence_col = c

        if start_col is None and "start" in cl:
            start_col = c

        if end_col is None and "end" in cl:
            end_col = c

        if strand_col is None and "strand" in cl:
            strand_col = c

        if score_col is None and "score" in cl:
            score_col = c

        if length_col is None and "length" in cl:
            length_col = c

        if url_col is None and (
            "url" in cl
            or "orthodb" in cl
        ):
            url_col = c

        if desc_col is None and (
            "description" in cl
            or "function" in cl
        ):
            desc_col = c

    # Standard BUSCO table fallback positions
    cols = list(df.columns)

    if sequence_col is None and len(cols) >= 3:
        sequence_col = cols[2]

    if status_col is None and len(cols) >= 2:
        status_col = cols[1]

    if start_col is None and len(cols) >= 4:
        start_col = cols[3]

    if end_col is None and len(cols) >= 5:
        end_col = cols[4]

    if strand_col is None and len(cols) >= 6:
        strand_col = cols[5]

    if score_col is None and len(cols) >= 7:
        score_col = cols[6]

    if length_col is None and len(cols) >= 8:
        length_col = cols[7]

    if url_col is None and len(cols) >= 9:
        url_col = cols[8]

    if desc_col is None and len(cols) >= 10:
        desc_col = cols[9]

    # ------------------------------------------------------------
    # Normalize BUSCO IDs
    # ------------------------------------------------------------

    df["_BUSCO_ID_NORMALIZED"] = (
        df[busco_col]
        .astype(str)
        .map(normalize_busco_id)
    )

    hits = df[
        df["_BUSCO_ID_NORMALIZED"].isin(candidate_set)
    ].copy()

    if hits.empty:
        continue

    matched_tables += 1

    # ------------------------------------------------------------
    # Infer species / assembly from directory
    # ------------------------------------------------------------

    table_parts = table_path.parts

    species = ""

    # Expected structure:
    #
    # busco\
    #   Species__GCA_xxx\
    #       run_...\
    #           full_table.tsv
    #

    for part in reversed(table_parts):

        if "__GCA_" in part:

            species = part
            break

    assembly = ""

    assembly_match = re.search(
        r"(GCA_\d+\.\d+|GCF_\d+\.\d+)",
        species
    )

    if assembly_match:
        assembly = assembly_match.group(1)

    if not species:

        # fallback
        species = table_path.parent.parent.name

        assembly_match = re.search(
            r"(GCA_\d+\.\d+|GCF_\d+\.\d+)",
            species
        )

        if assembly_match:
            assembly = assembly_match.group(1)

    # ------------------------------------------------------------
    # Extract each hit
    # ------------------------------------------------------------

    for _, row in hits.iterrows():

        busco_id = row["_BUSCO_ID_NORMALIZED"]

        sequence = (
            clean_string(row[sequence_col])
            if sequence_col in row
            else ""
        )

        status = (
            clean_string(row[status_col])
            if status_col in row
            else ""
        )

        start = (
            clean_string(row[start_col])
            if start_col in row
            else ""
        )

        end = (
            clean_string(row[end_col])
            if end_col in row
            else ""
        )

        strand = (
            clean_string(row[strand_col])
            if strand_col in row
            else ""
        )

        score = (
            clean_string(row[score_col])
            if score_col in row
            else ""
        )

        length = (
            clean_string(row[length_col])
            if length_col in row
            else ""
        )

        orthodb_url = (
            clean_string(row[url_col])
            if url_col in row
            else ""
        )

        description = (
            clean_string(row[desc_col])
            if desc_col in row
            else ""
        )

        # --------------------------------------------------------
        # Classify the sequence identifier
        # --------------------------------------------------------

        sequence_type = "Unknown"

        if sequence:

            if re.search(
                r"(gene|cds|transcript|mRNA)",
                sequence,
                re.I
            ):
                sequence_type = "Gene_or_transcript"

            elif re.search(
                r"(protein|prot)",
                sequence,
                re.I
            ):
                sequence_type = "Protein"

            elif re.search(
                r"\.\d+$",
                sequence
            ):
                sequence_type = "Assembly_sequence"

            else:
                sequence_type = "Sequence_or_gene_ID"

        mapping_rows.append({

            "BUSCO_ID": busco_id,

            "Species": species,

            "Assembly_Accession": assembly,

            "BUSCO_Status": status,

            "Sequence_or_Gene_ID": sequence,

            "Identifier_Type": sequence_type,

            "Start": start,

            "End": end,

            "Strand": strand,

            "Score": score,

            "Length": length,

            "OrthoDB_URL": orthodb_url,

            "Functional_Description": description,

            "Full_Table": str(table_path),

        })


# ================================================================
# BUILD MAPPING TABLE
# ================================================================

mapping_df = pd.DataFrame(mapping_rows)


if mapping_df.empty:

    print()
    print("=" * 80)
    print("NO BUSCO MATCHES FOUND")
    print("=" * 80)

    print(
        """
The BUSCO candidates were not recovered from the discovered
full_table.tsv files.

This does NOT mean that the BUSCOs are invalid.

The next diagnostic is to inspect the exact BUSCO lineage
dataset and the format of its info/ogs.id.info file.
"""
    )

else:

    # Remove exact duplicates
    mapping_df = mapping_df.drop_duplicates()

    # Sort
    mapping_df = mapping_df.sort_values(
        [
            "BUSCO_ID",
            "Species",
            "Assembly_Accession"
        ]
    )


# ================================================================
# MERGE PREDICTIVE IMPORTANCE
# ================================================================

print()
print("=" * 80)
print("ADDING PREDICTIVE MODEL INFORMATION")
print("=" * 80)

importance_columns = [
    "BUSCO_ID",
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

available_importance_columns = [
    c for c in importance_columns
    if c in top20.columns
]

importance_df = top20[
    available_importance_columns
].drop_duplicates(
    subset=["BUSCO_ID"]
)


if not mapping_df.empty:

    mapping_df = mapping_df.merge(
        importance_df,
        on="BUSCO_ID",
        how="left"
    )


# ================================================================
# MERGE EXISTING FUNCTIONAL DESCRIPTIONS
# ================================================================

if (
    annotation_df is not None
    and not annotation_df.empty
    and "BUSCO_ID" in annotation_df.columns
):

    annotation_columns = [
        "BUSCO_ID",
        "Functional_Description"
    ]

    annotation_columns = [
        c for c in annotation_columns
        if c in annotation_df.columns
    ]

    existing_desc = annotation_df[
        annotation_columns
    ].drop_duplicates(
        subset=["BUSCO_ID"]
    )

    if "Functional_Description" in existing_desc.columns:

        existing_desc = existing_desc.rename(
            columns={
                "Functional_Description":
                "Existing_Functional_Description"
            }
        )

        if not mapping_df.empty:

            mapping_df = mapping_df.merge(
                existing_desc,
                on="BUSCO_ID",
                how="left"
            )


# ================================================================
# SUMMARY PER BUSCO
# ================================================================

summary_rows = []

for busco in candidate_buscos:

    if mapping_df.empty:

        subset = pd.DataFrame()

    else:

        subset = mapping_df[
            mapping_df["BUSCO_ID"] == busco
        ]

    n_records = len(subset)

    n_species = (
        subset["Species"].nunique()
        if n_records > 0
        else 0
    )

    n_sequence_ids = (
        subset["Sequence_or_Gene_ID"]
        .replace("", pd.NA)
        .dropna()
        .nunique()
        if n_records > 0
        else 0
    )

    if n_records > 0:

        identifiers = sorted(
            set(
                x
                for x in subset[
                    "Sequence_or_Gene_ID"
                ].astype(str)
                if x.strip()
            )
        )

        identifier_string = "; ".join(
            identifiers[:20]
        )

    else:

        identifier_string = ""

    summary_rows.append({

        "BUSCO_ID": busco,

        "Mapping_Record_Count": n_records,

        "Species_With_Mapping": n_species,

        "Unique_Sequence_or_Gene_IDs":
            n_sequence_ids,

        "Example_Sequence_or_Gene_IDs":
            identifier_string,

        "Mapping_Status":
            (
                "Mapped"
                if n_records > 0
                else "Not_mapped_from_full_tables"
            )

    })


summary_df = pd.DataFrame(summary_rows)


# ================================================================
# SAVE MAIN MAPPING TABLE
# ================================================================

mapping_file = (
    TABLE_ROOT
    / "Carbon_Breadth_BUSCO_gene_protein_mapping_420.csv"
)

mapping_df.to_csv(
    mapping_file,
    index=False
)


summary_file = (
    TABLE_ROOT
    / "Carbon_Breadth_BUSCO_gene_mapping_summary_420.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ================================================================
# SAVE UNMAPPED BUSCO LIST
# ================================================================

unmapped_df = summary_df[
    summary_df["Mapping_Status"] != "Mapped"
].copy()

unmapped_file = (
    TABLE_ROOT
    / "Carbon_Breadth_unmapped_BUSCO_candidates_420.csv"
)

unmapped_df.to_csv(
    unmapped_file,
    index=False
)


# ================================================================
# CREATE GENE/PROTEIN-FOCUSED TABLE
# ================================================================

if not mapping_df.empty:

    gene_protein_df = mapping_df[
        [
            "BUSCO_ID",
            "Species",
            "Assembly_Accession",
            "BUSCO_Status",
            "Sequence_or_Gene_ID",
            "Identifier_Type",
            "Functional_Description",
            "Mean_Importance",
            "Fold_Stability",
            "Importance_Rank",
            "Full_Table"
        ]
    ].copy()

    gene_protein_df = gene_protein_df[
        gene_protein_df[
            "Sequence_or_Gene_ID"
        ].astype(str).str.strip() != ""
    ]

else:

    gene_protein_df = pd.DataFrame(
        columns=[
            "BUSCO_ID",
            "Species",
            "Assembly_Accession",
            "BUSCO_Status",
            "Sequence_or_Gene_ID",
            "Identifier_Type",
            "Functional_Description",
            "Mean_Importance",
            "Fold_Stability",
            "Importance_Rank",
            "Full_Table"
        ]
    )


gene_protein_file = (
    TABLE_ROOT
    / "Carbon_Breadth_gene_protein_candidates_420.csv"
)

gene_protein_df.to_csv(
    gene_protein_file,
    index=False
)


# ================================================================
# JSON SUMMARY
# ================================================================

mapped_count = int(
    (
        summary_df["Mapping_Status"] == "Mapped"
    ).sum()
)

unmapped_count = int(
    (
        summary_df["Mapping_Status"]
        != "Mapped"
    ).sum()
)

summary_json = {

    "Project": "Y1000+",

    "Phenotype": "Carbon_Breadth",

    "Total_BUSCO_candidates":
        len(candidate_buscos),

    "BUSCO_full_tables_found":
        len(full_tables),

    "BUSCO_full_tables_with_candidate_hits":
        matched_tables,

    "BUSCO_candidates_mapped":
        mapped_count,

    "BUSCO_candidates_not_mapped":
        unmapped_count,

    "Unique_mapping_records":
        int(len(mapping_df)),

    "Unique_gene_or_protein_records":
        int(len(gene_protein_df)),

    "BUSCO_root":
        str(BUSCO_ROOT),

    "Top20_file":
        str(TOP20_FILE),

    "Mapping_file":
        str(mapping_file),

    "Gene_protein_file":
        str(gene_protein_file),

    "Unmapped_file":
        str(unmapped_file)

}


json_file = (
    TABLE_ROOT
    / "Carbon_Breadth_BUSCO_gene_mapping_summary_420.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        summary_json,
        handle,
        indent=4
    )


# ================================================================
# WRITE REPORT
# ================================================================

report_file = (
    REPORT_ROOT
    / "Carbon_Breadth_BUSCO_gene_mapping_report_420.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "CARBON BREADTH — BUSCO → GENE / PROTEIN MAPPING\n"
    )

    handle.write("=" * 80 + "\n\n")

    handle.write(
        f"Total BUSCO candidates: "
        f"{len(candidate_buscos)}\n"
    )

    handle.write(
        f"BUSCO full tables found: "
        f"{len(full_tables)}\n"
    )

    handle.write(
        f"Full tables containing candidates: "
        f"{matched_tables}\n"
    )

    handle.write(
        f"BUSCO candidates mapped: "
        f"{mapped_count}\n"
    )

    handle.write(
        f"BUSCO candidates not mapped: "
        f"{unmapped_count}\n"
    )

    handle.write(
        f"Mapping records: "
        f"{len(mapping_df)}\n"
    )

    handle.write(
        f"Gene/protein records: "
        f"{len(gene_protein_df)}\n\n"
    )

    handle.write(
        "IMPORTANT INTERPRETATION\n"
    )

    handle.write(
        "BUSCO IDs represent orthologous groups and should not "
        "automatically be treated as unique gene names.\n"
    )

    handle.write(
        "Sequence/gene identifiers are retained at the level "
        "reported by the BUSCO results.\n"
    )

    handle.write(
        "No predictive candidate was discarded merely because "
        "a gene name was unavailable.\n"
    )


# ================================================================
# FINAL OUTPUT
# ================================================================

print()
print("=" * 80)
print("BUSCO → GENE / PROTEIN MAPPING COMPLETE")
print("=" * 80)

print()
print(f"Total BUSCO candidates       : {len(candidate_buscos)}")
print(f"Full tables searched        : {len(full_tables)}")
print(f"Tables containing candidates: {matched_tables}")
print(f"BUSCO candidates mapped     : {mapped_count}")
print(f"BUSCO candidates unmapped   : {unmapped_count}")
print(f"Mapping records             : {len(mapping_df)}")
print(
    f"Gene/protein records        : "
    f"{len(gene_protein_df)}"
)

print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print()
print("BUSCO → sequence/gene/protein mapping:")
print(mapping_file)

print()
print("BUSCO mapping summary:")
print(summary_file)

print()
print("Gene/protein candidate table:")
print(gene_protein_file)

print()
print("Unmapped BUSCO candidates:")
print(unmapped_file)

print()
print("JSON summary:")
print(json_file)

print()
print("Report:")
print(report_file)

print()
print("=" * 80)
print("NEXT STAGE")
print("=" * 80)

if mapped_count > 0:

    print(
        """
BUSCO candidates have been recovered from the BUSCO results.

Next:
BUSCO
  ↓
gene/protein IDs
  ↓
gene-level annotation
  ↓
GO / InterPro / EC / KEGG
  ↓
pathway grouping
  ↓
network analysis
  ↓
candidate chassis interpretation
"""
    )

else:

    print(
        """
No candidate BUSCOs were recovered from full_table.tsv.

DO NOT discard the candidates.

Inspect the BUSCO lineage dataset's:
    info/ogs.id.info
    info/species.info
    links_to_odb*.txt

The exact BUSCO dataset/version must then be used for
ortholog-group → constituent-gene mapping.
"""
    )

print()
print("Existing models and OOF predictions were NOT overwritten.")
print()