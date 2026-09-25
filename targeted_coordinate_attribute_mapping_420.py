# ================================================================
# TARGETED BUSCO → GENE / PROTEIN MAPPING
# CARBON BREADTH — 420
#
# Purpose:
#   1. Load existing BUSCO mapping
#   2. Extract the 20 predictive BUSCO candidates
#   3. Recover target assemblies
#   4. Locate ONLY GFF/GTF files belonging to those assemblies
#   5. Map BUSCO coordinates to annotation features
#   6. Recover protein/CDS/gene information
#
# Does NOT:
#   - search all GFF files blindly
#   - overwrite existing model results
#   - require NCBI email
# ================================================================

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")

BUSCO_MAPPING = (
    PROJECT_ROOT
    / r"results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml"
    / r"phylogeny_cv\model_training\feature_interpretation_420"
    / r"functional_annotation_420\busco_gene_mapping_420\tables"
    / r"Carbon_Breadth_BUSCO_gene_protein_mapping_420.csv"
)

BUSCO_ROOT = (
    PROJECT_ROOT
    / r"results\stage4B_phylogeny\busco"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / r"results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml"
    / r"phylogeny_cv\model_training\feature_interpretation_420"
    / r"functional_annotation_420\busco_gene_mapping_420"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
REPORT_DIR = OUTPUT_ROOT / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# TOP-20 BUSCO CANDIDATES
# ================================================================

TOP_BUSCOS = [
    "36839at4891",
    "5746at4891",
    "23379at4891",
    "6711at4891",
    "2471at4891",
    "29457at4891",
    "27915at4891",
    "1679at4891",
    "18913at4891",
    "22532at4891",
    "28058at4891",
    "4322at4891",
    "33531at4891",
    "3574at4891",
    "12523at4891",
    "12468at4891",
    "2307at4891",
    "24318at4891",
    "9782at4891",
    "34052at4891",
]

TOP_BUSCOS = set(TOP_BUSCOS)

# ================================================================
# HELPERS
# ================================================================

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_accession(x):
    """
    Normalize assembly accession.

    Examples:
        GCA_030570635.1
        GCA_030570635
    """
    x = clean(x)

    if not x:
        return ""

    m = re.search(r"(GC[AF]_\d+\.\d+)", x)

    if m:
        return m.group(1)

    m = re.search(r"(GC[AF]_\d+)", x)

    if m:
        return m.group(1)

    return x


def parse_attributes(attr_string):
    """
    Parse GFF3/GTF attribute strings.
    """

    result = {}

    if not attr_string:
        return result

    text = str(attr_string).strip()

    # GFF3 style:
    # ID=x;Parent=y;gene=z
    if "=" in text:
        for field in text.split(";"):
            field = field.strip()

            if "=" in field:
                key, value = field.split("=", 1)
                result[key.strip()] = value.strip().strip('"')

    # GTF style:
    # gene_id "x"; transcript_id "y";
    else:
        matches = re.findall(
            r'(\S+)\s+"([^"]*)"',
            text
        )

        for key, value in matches:
            result[key.strip()] = value.strip()

    return result


# ================================================================
# LOAD EXISTING BUSCO MAPPING
# ================================================================

print("=" * 80)
print("LOADING EXISTING BUSCO MAPPING")
print("=" * 80)

print(BUSCO_MAPPING)

if not BUSCO_MAPPING.exists():
    raise FileNotFoundError(
        f"BUSCO mapping not found:\n{BUSCO_MAPPING}"
    )

mapping = pd.read_csv(BUSCO_MAPPING, low_memory=False)

print("\nMapping shape:")
print(mapping.shape)

print("\nColumns:")
print(list(mapping.columns))

# ================================================================
# FILTER TOP-20
# ================================================================

mapping["BUSCO_ID"] = mapping["BUSCO_ID"].astype(str).str.strip()

candidate_mapping = mapping[
    mapping["BUSCO_ID"].isin(TOP_BUSCOS)
].copy()

print("\nRows belonging to candidate BUSCOs:")
print(len(candidate_mapping))

# ================================================================
# COORDINATE-VALID RECORDS
# ================================================================

print("\n" + "=" * 80)
print("COORDINATE-VALID BUSCO RECORDS")
print("=" * 80)

for col in ["Sequence", "Start", "End"]:
    if col not in candidate_mapping.columns:
        candidate_mapping[col] = np.nan

candidate_mapping["Sequence"] = (
    candidate_mapping["Sequence"]
    .fillna("")
    .astype(str)
    .str.strip()
)

candidate_mapping["Start"] = pd.to_numeric(
    candidate_mapping["Start"],
    errors="coerce"
)

candidate_mapping["End"] = pd.to_numeric(
    candidate_mapping["End"],
    errors="coerce"
)

coordinate_records = candidate_mapping[
    (candidate_mapping["Sequence"] != "") &
    candidate_mapping["Start"].notna() &
    candidate_mapping["End"].notna()
].copy()

print("Candidate BUSCOs:",
      candidate_mapping["BUSCO_ID"].nunique())

print("BUSCOs with coordinates:",
      coordinate_records["BUSCO_ID"].nunique())

print("Coordinate records:",
      len(coordinate_records))

# ================================================================
# RECOVER TARGET ASSEMBLIES
# ================================================================

print("\n" + "=" * 80)
print("RECOVERING TARGET ASSEMBLIES")
print("=" * 80)

coordinate_records["Assembly_Accession"] = (
    coordinate_records["Assembly_Accession"]
    .fillna("")
    .astype(str)
    .map(normalize_accession)
)

target_assemblies = sorted(
    set(
        x for x in coordinate_records["Assembly_Accession"]
        if x
    )
)

print("\nTarget assemblies:")
print(len(target_assemblies))

for accession in target_assemblies[:20]:
    print(" ", accession)

if len(target_assemblies) > 20:
    print(
        f" ... {len(target_assemblies) - 20} additional assemblies"
    )

# ================================================================
# BUILD TARGET ASSEMBLY DIRECTORY INDEX
#
# IMPORTANT:
# We only inspect directories immediately under BUSCO_ROOT.
# We DO NOT walk the entire GFF universe.
# ================================================================

print("\n" + "=" * 80)
print("INDEXING BUSCO ASSEMBLY DIRECTORIES")
print("=" * 80)

if not BUSCO_ROOT.exists():
    raise FileNotFoundError(
        f"BUSCO root does not exist:\n{BUSCO_ROOT}"
    )

assembly_dirs = []

for p in BUSCO_ROOT.iterdir():

    if not p.is_dir():
        continue

    name = p.name

    accession_match = re.search(
        r"(GC[AF]_\d+(?:\.\d+)?)",
        name
    )

    if accession_match:
        accession = accession_match.group(1)

        if accession in target_assemblies:
            assembly_dirs.append(p)

print("\nTarget assembly directories found:")
print(len(assembly_dirs))

for p in assembly_dirs[:20]:
    print(" ", p)

# ================================================================
# FALLBACK:
# If assembly directories are nested, inspect only a shallow level.
# ================================================================

if len(assembly_dirs) == 0:

    print("\nNo direct target directories found.")
    print("Performing LIMITED targeted directory search...")

    found = {}

    # only search up to 3 directory levels
    queue = [(BUSCO_ROOT, 0)]

    while queue:

        current, depth = queue.pop()

        if depth > 3:
            continue

        try:
            children = list(current.iterdir())
        except Exception:
            continue

        for child in children:

            if not child.is_dir():
                continue

            name = child.name

            accession_match = re.search(
                r"(GC[AF]_\d+(?:\.\d+)?)",
                name
            )

            if accession_match:

                accession = accession_match.group(1)

                if accession in target_assemblies:
                    found[accession] = child

            queue.append(
                (child, depth + 1)
            )

    assembly_dirs = list(found.values())

print("\nFinal target assembly directories:")
print(len(assembly_dirs))

# ================================================================
# LOCATE GFF/GTF ONLY INSIDE TARGET ASSEMBLIES
# ================================================================

print("\n" + "=" * 80)
print("LOCATING TARGET GFF / GTF FILES")
print("=" * 80)

annotation_files = []

for i, assembly_dir in enumerate(assembly_dirs, 1):

    accession_match = re.search(
        r"(GC[AF]_\d+(?:\.\d+)?)",
        assembly_dir.name
    )

    accession = (
        accession_match.group(1)
        if accession_match
        else ""
    )

    print(
        f"[{i}/{len(assembly_dirs)}] {accession}",
        end="\r"
    )

    # Search ONLY inside this assembly directory.
    # We stop at the first useful annotation files.

    try:
        for root, dirs, files in os.walk(assembly_dir):

            for fname in files:

                lower = fname.lower()

                if not (
                    lower.endswith(".gff") or
                    lower.endswith(".gff3") or
                    lower.endswith(".gtf") or
                    lower.endswith(".gff.gz") or
                    lower.endswith(".gff3.gz") or
                    lower.endswith(".gtf.gz")
                ):
                    continue

                path = Path(root) / fname

                annotation_files.append(
                    (
                        accession,
                        path
                    )
                )

    except Exception as e:
        print(
            f"\nWARNING: Could not inspect {assembly_dir}: {e}"
        )

print("\n\nTarget annotation files found:")
print(len(annotation_files))

# ================================================================
# IF NO GFF FOUND
# ================================================================

if len(annotation_files) == 0:

    print("\n" + "=" * 80)
    print("NO TARGET GFF/GTF FILES FOUND")
    print("=" * 80)

    print(
        """
The BUSCO results contain coordinates, but the corresponding
GFF/GTF annotation files are not located inside the BUSCO
assembly directories.

DO NOT perform another 878,000-file global search.

The next correct source is the genome/annotation directory
associated with the 436 assemblies.
"""
    )

    raise SystemExit(0)

# ================================================================
# BUILD BUSCO QUERY INDEX
# ================================================================

print("\n" + "=" * 80)
print("BUILDING BUSCO COORDINATE INDEX")
print("=" * 80)

queries = defaultdict(list)

for _, row in coordinate_records.iterrows():

    accession = clean(
        row["Assembly_Accession"]
    )

    sequence = clean(
        row["Sequence"]
    )

    start = int(
        min(
            row["Start"],
            row["End"]
        )
    )

    end = int(
        max(
            row["Start"],
            row["End"]
        )
    )

    queries[
        (
            accession,
            sequence
        )
    ].append(
        {
            "BUSCO_ID": row["BUSCO_ID"],
            "Start": start,
            "End": end,
            "BUSCO_Start": row["Start"],
            "BUSCO_End": row["End"],
            "BUSCO_Strand": clean(row.get("Strand", "")),
        }
    )

print(
    "Assembly/sequence query groups:",
    len(queries)
)

# ================================================================
# PARSE ANNOTATION
# ================================================================

print("\n" + "=" * 80)
print("MAPPING BUSCO COORDINATES TO ANNOTATIONS")
print("=" * 80)

mapped_records = []

files_processed = 0

for accession, annotation_path in annotation_files:

    files_processed += 1

    print(
        f"Processing annotation "
        f"{files_processed}/{len(annotation_files)}: "
        f"{accession} | {annotation_path.name}"
    )

    # ------------------------------------------------------------
    # Handle gzip
    # ------------------------------------------------------------

    if str(annotation_path).lower().endswith(".gz"):
        import gzip
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"

    try:

        with opener(
            annotation_path,
            mode,
            encoding="utf-8",
            errors="replace"
        ) as handle:

            for line in handle:

                if not line.strip():
                    continue

                if line.startswith("#"):
                    continue

                parts = line.rstrip("\n").split("\t")

                if len(parts) < 9:
                    continue

                seqid = parts[0]

                try:
                    feature_start = int(parts[3])
                    feature_end = int(parts[4])
                except ValueError:
                    continue

                strand = parts[6]
                feature_type = parts[2]
                attributes = parse_attributes(parts[8])

                key = (
                    accession,
                    seqid
                )

                if key not in queries:
                    continue

                # ------------------------------------------------
                # Check overlap with candidate BUSCO coordinates
                # ------------------------------------------------

                for q in queries[key]:

                    overlap = (
                        feature_start <= q["End"] and
                        feature_end >= q["Start"]
                    )

                    if not overlap:
                        continue

                    # ------------------------------------------------
                    # Recover common annotation identifiers
                    # ------------------------------------------------

                    gene_id = (
                        attributes.get("gene_id")
                        or attributes.get("gene")
                        or attributes.get("GeneID")
                        or attributes.get("geneID")
                        or ""
                    )

                    gene_name = (
                        attributes.get("gene_name")
                        or attributes.get("gene")
                        or attributes.get("Name")
                        or ""
                    )

                    locus_tag = (
                        attributes.get("locus_tag")
                        or attributes.get("locus")
                        or ""
                    )

                    protein_id = (
                        attributes.get("protein_id")
                        or attributes.get("protein")
                        or attributes.get("proteinId")
                        or ""
                    )

                    transcript_id = (
                        attributes.get("transcript_id")
                        or attributes.get("transcript")
                        or ""
                    )

                    feature_id = (
                        attributes.get("ID")
                        or ""
                    )

                    parent = (
                        attributes.get("Parent")
                        or ""
                    )

                    product = (
                        attributes.get("product")
                        or ""
                    )

                    name = (
                        attributes.get("Name")
                        or ""
                    )

                    dbxref = (
                        attributes.get("Dbxref")
                        or attributes.get("db_xref")
                        or ""
                    )

                    mapped_records.append(
                        {
                            "BUSCO_ID":
                                q["BUSCO_ID"],

                            "Assembly_Accession":
                                accession,

                            "Sequence":
                                seqid,

                            "BUSCO_Start":
                                q["BUSCO_Start"],

                            "BUSCO_End":
                                q["BUSCO_End"],

                            "BUSCO_Strand":
                                q["BUSCO_Strand"],

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

                            "Feature_ID":
                                feature_id,

                            "Parent":
                                parent,

                            "Product":
                                product,

                            "Name":
                                name,

                            "Dbxref":
                                dbxref,

                            "Annotation_File":
                                str(annotation_path)
                        }
                    )

    except Exception as e:

        print(
            f"WARNING: failed to parse "
            f"{annotation_path}: {e}"
        )

# ================================================================
# CREATE RESULT TABLE
# ================================================================

print("\n" + "=" * 80)
print("CREATING MAPPING TABLE")
print("=" * 80)

mapped = pd.DataFrame(mapped_records)

if mapped.empty:

    print(
        """
No coordinate-to-annotation matches were recovered.

This means the GFF/GTF files found do not use the same
sequence identifiers as the BUSCO coordinate records.

The BUSCO mapping itself is NOT invalid.
It means the annotation source needs identifier reconciliation.
"""
    )

    raise SystemExit(0)

# ================================================================
# MERGE PREDICTIVE INFORMATION
# ================================================================

predictive_cols = [
    "BUSCO_ID",
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

available_predictive = [
    c for c in predictive_cols
    if c in candidate_mapping.columns
]

predictive = (
    candidate_mapping[available_predictive]
    .drop_duplicates("BUSCO_ID")
)

mapped = mapped.merge(
    predictive,
    on="BUSCO_ID",
    how="left"
)

# ================================================================
# CLEAN IDENTIFIERS
# ================================================================

identifier_cols = [
    "Gene_ID",
    "Gene_Name",
    "Locus_Tag",
    "Protein_ID",
    "Transcript_ID",
    "Feature_ID",
    "Parent",
    "Product",
    "Name",
    "Dbxref",
]

for col in identifier_cols:

    if col in mapped.columns:

        mapped[col] = (
            mapped[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("TARGETED BUSCO → GENE / PROTEIN MAPPING SUMMARY")
print("=" * 80)

print(
    "Total BUSCO candidates       :",
    len(TOP_BUSCOS)
)

print(
    "BUSCO candidates mapped      :",
    mapped["BUSCO_ID"].nunique()
)

print(
    "BUSCO candidates unmapped    :",
    len(
        TOP_BUSCOS -
        set(mapped["BUSCO_ID"].unique())
    )
)

print(
    "Mapping records              :",
    len(mapped)
)

for col, label in [
    ("Gene_ID", "Unique gene IDs"),
    ("Gene_Name", "Unique gene names"),
    ("Locus_Tag", "Unique locus tags"),
    ("Protein_ID", "Unique protein IDs"),
    ("Transcript_ID", "Unique transcript IDs"),
]:

    if col in mapped.columns:

        values = mapped[col]

        values = values[
            values != ""
        ]

        print(
            f"{label:<28}:",
            values.nunique()
        )

# ================================================================
# SAVE COMPLETE MAPPING
# ================================================================

complete_path = (
    TABLE_DIR /
    "Carbon_Breadth_targeted_BUSCO_annotation_mapping_420.csv"
)

mapped.to_csv(
    complete_path,
    index=False
)

# ================================================================
# CREATE GENE/PROTEIN CANDIDATE TABLE
# ================================================================

print("\n" + "=" * 80)
print("CREATING GENE / PROTEIN CANDIDATE TABLE")
print("=" * 80)

gene_candidate = mapped.copy()

# Keep records that have ANY useful identifier
identifier_mask = pd.Series(
    False,
    index=gene_candidate.index
)

for col in [
    "Gene_ID",
    "Gene_Name",
    "Locus_Tag",
    "Protein_ID",
    "Product",
    "Name"
]:

    if col in gene_candidate.columns:

        identifier_mask |= (
            gene_candidate[col]
            .fillna("")
            .astype(str)
            .str.strip()
            != ""
        )

gene_candidate = gene_candidate[
    identifier_mask
].copy()

gene_candidate_path = (
    TABLE_DIR /
    "Carbon_Breadth_gene_protein_candidates_420.csv"
)

gene_candidate.to_csv(
    gene_candidate_path,
    index=False
)

# ================================================================
# BUSCO SUMMARY
# ================================================================

busco_summary = []

for busco in sorted(TOP_BUSCOS):

    sub = mapped[
        mapped["BUSCO_ID"] == busco
    ]

    row = {
        "BUSCO_ID": busco,
        "Mapping_Records": len(sub),
        "Assemblies": (
            sub["Assembly_Accession"]
            .nunique()
            if len(sub)
            else 0
        ),
        "Gene_ID_Count": (
            sub.loc[
                sub["Gene_ID"] != "",
                "Gene_ID"
            ].nunique()
            if "Gene_ID" in sub
            else 0
        ),
        "Gene_Name_Count": (
            sub.loc[
                sub["Gene_Name"] != "",
                "Gene_Name"
            ].nunique()
            if "Gene_Name" in sub
            else 0
        ),
        "Locus_Tag_Count": (
            sub.loc[
                sub["Locus_Tag"] != "",
                "Locus_Tag"
            ].nunique()
            if "Locus_Tag" in sub
            else 0
        ),
        "Protein_ID_Count": (
            sub.loc[
                sub["Protein_ID"] != "",
                "Protein_ID"
            ].nunique()
            if "Protein_ID" in sub
            else 0
        ),
        "Product_Count": (
            sub.loc[
                sub["Product"] != "",
                "Product"
            ].nunique()
            if "Product" in sub
            else 0
        )
    }

    busco_summary.append(row)

busco_summary = pd.DataFrame(
    busco_summary
)

summary_path = (
    TABLE_DIR /
    "Carbon_Breadth_targeted_BUSCO_annotation_summary_420.csv"
)

busco_summary.to_csv(
    summary_path,
    index=False
)

# ================================================================
# UNMAPPED BUSCOS
# ================================================================

mapped_buscos = set(
    mapped["BUSCO_ID"].unique()
)

unmapped = sorted(
    TOP_BUSCOS -
    mapped_buscos
)

unmapped_df = pd.DataFrame(
    {
        "BUSCO_ID": unmapped
    }
)

unmapped_path = (
    TABLE_DIR /
    "Carbon_Breadth_unmapped_BUSCO_candidates_420.csv"
)

unmapped_df.to_csv(
    unmapped_path,
    index=False
)

# ================================================================
# REPORT
# ================================================================

report_path = (
    REPORT_DIR /
    "Carbon_Breadth_targeted_BUSCO_annotation_report_420.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CARBON BREADTH — TARGETED BUSCO ANNOTATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total BUSCO candidates: {len(TOP_BUSCOS)}\n"
    )

    f.write(
        f"Mapped BUSCO candidates: "
        f"{len(mapped_buscos)}\n"
    )

    f.write(
        f"Unmapped BUSCO candidates: "
        f"{len(unmapped)}\n"
    )

    f.write(
        f"Mapping records: {len(mapped)}\n"
    )

    f.write(
        f"Target annotation files: "
        f"{len(annotation_files)}\n"
    )

    f.write("\nUnmapped BUSCOs:\n")

    for x in unmapped:
        f.write(f"  {x}\n")

# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n" + "=" * 80)
print("TARGETED BUSCO ANNOTATION COMPLETE")
print("=" * 80)

print(
    "\nBUSCO → annotation mapping:"
)
print(complete_path)

print(
    "\nGene/protein candidate table:"
)
print(gene_candidate_path)

print(
    "\nBUSCO summary:"
)
print(summary_path)

print(
    "\nUnmapped BUSCOs:"
)
print(unmapped_path)

print(
    "\nReport:"
)
print(report_path)

print("\n" + "=" * 80)
print("NEXT SCIENTIFIC STAGE")
print("=" * 80)

print(
"""
BUSCO
  ↓
coordinate-validated annotation
  ↓
gene / locus / protein identifiers
  ↓
functional annotation
  ↓
GO
InterPro
EC
KEGG
  ↓
pathway enrichment / pathway grouping
  ↓
functional network
  ↓
candidate chassis interpretation

IMPORTANT:
Do NOT treat BUSCO functional descriptions alone as
gene-level pathway evidence.
"""
)