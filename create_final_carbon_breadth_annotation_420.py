import os
import pandas as pd
import numpy as np

# ============================================================
# CARBON BREADTH — FINAL INTEGRATED ANNOTATION REPORT
# ============================================================

BASE = (
    r"C:\Y1000_chassis_project\results"
    r"\stage5_phylogeny_ml_dataset"
    r"\phylogeny_aware_ml"
    r"\phylogeny_cv"
    r"\model_training"
    r"\feature_interpretation_420"
    r"\functional_annotation_420"
    r"\BUSCO20_annotation"
)

UNIPROT_DIR = os.path.join(BASE, "UniProt")
PATHWAY_DIR = os.path.join(BASE, "pathway_analysis_420")

# ------------------------------------------------------------
# INPUT FILES
# ------------------------------------------------------------

ANNOTATION_FILE = os.path.join(
    UNIPROT_DIR,
    "Carbon_Breadth_top20_UniProt_annotation_inspection_420.csv"
)

PATHWAY_FILE = os.path.join(
    PATHWAY_DIR,
    "Carbon_Breadth_top20_KEGG_gene_pathway_mapping_420.csv"
)

BUSCO_SUMMARY_FILE = os.path.join(
    PATHWAY_DIR,
    "Carbon_Breadth_top20_BUSCO_pathway_summary_420.csv"
)

UNIQUE_PATHWAY_FILE = os.path.join(
    PATHWAY_DIR,
    "Carbon_Breadth_top20_unique_pathways_420.csv"
)

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

OUTPUT_DIR = os.path.join(BASE, "FINAL_CARBON_BREADTH_ANNOTATION_420")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

FINAL_TABLE = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_top20_FINAL_integrated_annotation_420.csv"
)

PATHWAY_TABLE = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_top20_FINAL_pathway_annotation_420.csv"
)

CANDIDATE_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_top20_FINAL_candidate_summary_420.csv"
)

PATHWAY_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_FINAL_pathway_summary_420.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_FINAL_integrated_annotation_report_420.txt"
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(x):
    """
    Convert NaN/empty values to a consistent representation.
    """
    if pd.isna(x):
        return ""

    x = str(x).strip()

    if x.lower() in {"nan", "none", "null"}:
        return ""

    return x


def unique_join(values, separator="; "):
    """
    Combine values while preserving order and removing duplicates.
    """
    result = []

    for value in values:
        value = clean_value(value)

        if not value:
            continue

        # Some fields may already contain semicolon-separated values
        parts = [p.strip() for p in value.split(";")]

        for part in parts:
            if part and part not in result:
                result.append(part)

    return separator.join(result)


def count_items(value):
    """
    Count semicolon-separated annotation items.
    """
    value = clean_value(value)

    if not value:
        return 0

    return len(
        [x.strip() for x in value.split(";") if x.strip()]
    )


# ============================================================
# LOAD ANNOTATION TABLE
# ============================================================

print("=" * 80)
print("CARBON BREADTH — FINAL INTEGRATED ANNOTATION")
print("=" * 80)

print("\nLoading UniProt / GO / KEGG / EC annotation table...")

if not os.path.exists(ANNOTATION_FILE):
    raise FileNotFoundError(
        f"\nAnnotation file not found:\n{ANNOTATION_FILE}"
    )

annotation = pd.read_csv(ANNOTATION_FILE)

print(f"Annotation shape: {annotation.shape}")

print("\nAnnotation columns:")

for col in annotation.columns:
    print(f"  {col}")


# ============================================================
# LOAD KEGG PATHWAY MAPPING
# ============================================================

print("\n" + "=" * 80)
print("LOADING KEGG PATHWAY MAPPING")
print("=" * 80)

if not os.path.exists(PATHWAY_FILE):
    raise FileNotFoundError(
        f"\nKEGG pathway mapping file not found:\n{PATHWAY_FILE}"
    )

pathway = pd.read_csv(PATHWAY_FILE)

print(f"\nPathway mapping shape: {pathway.shape}")

print("\nPathway columns:")

for col in pathway.columns:
    print(f"  {col}")


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

annotation.columns = [
    str(c).strip()
    for c in annotation.columns
]

pathway.columns = [
    str(c).strip()
    for c in pathway.columns
]

if "BUSCO_ID" not in annotation.columns:
    raise ValueError(
        "BUSCO_ID column missing from annotation table."
    )

if "BUSCO_ID" not in pathway.columns:
    raise ValueError(
        "BUSCO_ID column missing from pathway mapping table."
    )


# ============================================================
# CREATE BUSCO-LEVEL KEGG PATHWAY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("BUILDING BUSCO-LEVEL PATHWAY ANNOTATIONS")
print("=" * 80)

pathway_records = []

for busco, group in pathway.groupby("BUSCO_ID"):

    kegg_genes = []
    pathway_ids = []
    pathway_names = []

    # Detect likely columns dynamically
    for _, row in group.iterrows():

        for col in pathway.columns:

            col_lower = col.lower()

            value = clean_value(row[col])

            if not value:
                continue

            if "kegg" in col_lower and (
                "gene" in col_lower
                or "id" in col_lower
            ):
                if value not in kegg_genes:
                    kegg_genes.append(value)

            if "pathway" in col_lower:

                # Avoid treating generic BUSCO/pathway columns as IDs
                if "id" in col_lower:
                    if value not in pathway_ids:
                        pathway_ids.append(value)

                elif (
                    "name" in col_lower
                    or "description" in col_lower
                ):
                    if value not in pathway_names:
                        pathway_names.append(value)

    pathway_records.append({
        "BUSCO_ID": busco,
        "KEGG_Gene_IDs": "; ".join(kegg_genes),
        "KEGG_Gene_Count": len(kegg_genes),
        "KEGG_Pathway_IDs": "; ".join(pathway_ids),
        "KEGG_Pathway_Count": len(pathway_ids),
        "KEGG_Pathway_Names": "; ".join(pathway_names),
    })


pathway_busco = pd.DataFrame(pathway_records)

print(
    f"\nBUSCOs represented in pathway mapping: "
    f"{len(pathway_busco)}"
)


# ============================================================
# IF PATHWAY FILE HAS STANDARD COLUMN NAMES, IMPROVE EXTRACTION
# ============================================================

# Look for common pathway column names

possible_gene_cols = [
    "KEGG_Gene",
    "KEGG_Gene_ID",
    "KEGG_ID",
    "KEGG"
]

possible_pathway_id_cols = [
    "Pathway_ID",
    "KEGG_Pathway",
    "KEGG_Pathway_ID"
]

possible_pathway_name_cols = [
    "Pathway_Name",
    "KEGG_Pathway_Name",
    "Pathway"
]


gene_col = next(
    (c for c in possible_gene_cols if c in pathway.columns),
    None
)

pathway_id_col = next(
    (c for c in possible_pathway_id_cols if c in pathway.columns),
    None
)

pathway_name_col = next(
    (c for c in possible_pathway_name_cols if c in pathway.columns),
    None
)

if gene_col or pathway_id_col or pathway_name_col:

    print("\nDetected pathway columns:")

    if gene_col:
        print(f"  KEGG gene column    : {gene_col}")

    if pathway_id_col:
        print(f"  Pathway ID column   : {pathway_id_col}")

    if pathway_name_col:
        print(f"  Pathway name column : {pathway_name_col}")

    rows = []

    for busco, group in pathway.groupby("BUSCO_ID"):

        genes = []

        if gene_col:
            for x in group[gene_col]:
                x = clean_value(x)
                if x and x not in genes:
                    genes.append(x)

        pids = []

        if pathway_id_col:
            for x in group[pathway_id_col]:
                x = clean_value(x)
                if x and x not in pids:
                    pids.append(x)

        pnames = []

        if pathway_name_col:
            for x in group[pathway_name_col]:
                x = clean_value(x)
                if x and x not in pnames:
                    pnames.append(x)

        rows.append({
            "BUSCO_ID": busco,
            "KEGG_Gene_IDs": "; ".join(genes),
            "KEGG_Gene_Count": len(genes),
            "KEGG_Pathway_IDs": "; ".join(pids),
            "KEGG_Pathway_Count": len(pids),
            "KEGG_Pathway_Names": "; ".join(pnames),
        })

    pathway_busco = pd.DataFrame(rows)


# ============================================================
# MERGE ANNOTATION + PATHWAY DATA
# ============================================================

print("\n" + "=" * 80)
print("MERGING FUNCTIONAL AND PATHWAY ANNOTATIONS")
print("=" * 80)

final = annotation.copy()

final = final.merge(
    pathway_busco,
    on="BUSCO_ID",
    how="left"
)


# ============================================================
# FILL MISSING VALUES
# ============================================================

for col in [
    "KEGG_Gene_IDs",
    "KEGG_Pathway_IDs",
    "KEGG_Pathway_Names"
]:

    if col in final.columns:
        final[col] = final[col].fillna("")


for col in [
    "KEGG_Gene_Count",
    "KEGG_Pathway_Count"
]:

    if col in final.columns:
        final[col] = (
            final[col]
            .fillna(0)
            .astype(int)
        )


# ============================================================
# ANNOTATION COUNTS
# ============================================================

print("\nCalculating annotation statistics...")

if "GO" in final.columns:
    final["GO_Count_Final"] = final["GO"].apply(
        count_items
    )

if "KEGG" in final.columns:
    final["KEGG_Annotation_Count_Final"] = final["KEGG"].apply(
        count_items
    )

if "EC" in final.columns:
    final["EC_Count_Final"] = final["EC"].apply(
        count_items
    )

if "UniProt" in final.columns:
    final["UniProt_Count_Final"] = final["UniProt"].apply(
        count_items
    )


# ============================================================
# ANNOTATION STATUS
# ============================================================

def annotation_status(row):

    pathway_count = int(
        row.get("KEGG_Pathway_Count", 0)
    )

    kegg_count = int(
        row.get("KEGG_Annotation_Count_Final", 0)
    )

    go_count = int(
        row.get("GO_Count_Final", 0)
    )

    ec_count = int(
        row.get("EC_Count_Final", 0)
    )

    if pathway_count > 0:
        return "KEGG pathway mapped"

    if kegg_count > 0:
        return "KEGG annotated; pathway not retrieved"

    if go_count > 0:
        return "GO annotated only"

    if ec_count > 0:
        return "EC annotated only"

    return "Functional annotation only"


final["Annotation_Status"] = final.apply(
    annotation_status,
    axis=1
)


# ============================================================
# PATHWAY MAPPED FLAG
# ============================================================

final["Pathway_Mapped"] = (
    final["KEGG_Pathway_Count"] > 0
)


# ============================================================
# REORDER IMPORTANT COLUMNS
# ============================================================

preferred_columns = [
    "BUSCO_ID",
    "Functional_Description",
    "UniProt",
    "Protein",
    "Gene",
    "GO",
    "KEGG",
    "EC",
    "KEGG_Gene_IDs",
    "KEGG_Gene_Count",
    "KEGG_Pathway_IDs",
    "KEGG_Pathway_Count",
    "KEGG_Pathway_Names",
    "GO_Count_Final",
    "KEGG_Annotation_Count_Final",
    "EC_Count_Final",
    "UniProt_Count_Final",
    "Pathway_Mapped",
    "Annotation_Status"
]

existing_preferred = [
    c for c in preferred_columns
    if c in final.columns
]

remaining = [
    c for c in final.columns
    if c not in existing_preferred
]

final = final[
    existing_preferred + remaining
]


# ============================================================
# SAVE FINAL INTEGRATED TABLE
# ============================================================

final.to_csv(
    FINAL_TABLE,
    index=False
)

print("\nFinal integrated table written:")
print(FINAL_TABLE)

print(f"\nFinal table shape: {final.shape}")


# ============================================================
# CREATE PATHWAY-FOCUSED TABLE
# ============================================================

print("\n" + "=" * 80)
print("CREATING PATHWAY-FOCUSED TABLE")
print("=" * 80)

pathway_columns = [
    "BUSCO_ID",
    "Functional_Description",
    "KEGG",
    "EC",
    "KEGG_Gene_IDs",
    "KEGG_Gene_Count",
    "KEGG_Pathway_IDs",
    "KEGG_Pathway_Count",
    "KEGG_Pathway_Names",
    "Pathway_Mapped"
]

pathway_columns = [
    c for c in pathway_columns
    if c in final.columns
]

pathway_table = final[pathway_columns].copy()

pathway_table = pathway_table[
    pathway_table["Pathway_Mapped"] == True
]

pathway_table.to_csv(
    PATHWAY_TABLE,
    index=False
)

print(
    f"\nBUSCOs with pathway mappings: "
    f"{len(pathway_table)}"
)

print(PATHWAY_TABLE)


# ============================================================
# CREATE CANDIDATE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CREATING CANDIDATE SUMMARY")
print("=" * 80)

candidate_summary_columns = [
    "BUSCO_ID",
    "Functional_Description",
    "GO",
    "KEGG",
    "EC",
    "KEGG_Gene_Count",
    "KEGG_Pathway_Count",
    "KEGG_Pathway_Names",
    "Pathway_Mapped",
    "Annotation_Status"
]

candidate_summary_columns = [
    c for c in candidate_summary_columns
    if c in final.columns
]

candidate_summary = final[
    candidate_summary_columns
].copy()

candidate_summary.to_csv(
    CANDIDATE_SUMMARY,
    index=False
)

print(CANDIDATE_SUMMARY)


# ============================================================
# CREATE UNIQUE PATHWAY SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CREATING UNIQUE PATHWAY SUMMARY")
print("=" * 80)

pathway_rows = []

for _, row in final.iterrows():

    busco = clean_value(
        row.get("BUSCO_ID", "")
    )

    function = clean_value(
        row.get("Functional_Description", "")
    )

    pathway_ids = clean_value(
        row.get("KEGG_Pathway_IDs", "")
    )

    pathway_names = clean_value(
        row.get("KEGG_Pathway_Names", "")
    )

    ids = [
        x.strip()
        for x in pathway_ids.split(";")
        if x.strip()
    ]

    names = [
        x.strip()
        for x in pathway_names.split(";")
        if x.strip()
    ]

    for i, pid in enumerate(ids):

        pname = (
            names[i]
            if i < len(names)
            else ""
        )

        pathway_rows.append({
            "Pathway_ID": pid,
            "Pathway_Name": pname,
            "BUSCO_ID": busco,
            "Functional_Description": function
        })


if pathway_rows:

    pathway_long = pd.DataFrame(
        pathway_rows
    )

    # Count unique candidates per pathway
    pathway_summary = (
        pathway_long
        .groupby(
            ["Pathway_ID", "Pathway_Name"],
            dropna=False
        )
        .agg(
            Candidate_BUSCO_Count=(
                "BUSCO_ID",
                "nunique"
            ),
            Candidate_BUSCOs=(
                "BUSCO_ID",
                lambda x: "; ".join(
                    dict.fromkeys(x)
                )
            ),
            Functions=(
                "Functional_Description",
                lambda x: "; ".join(
                    dict.fromkeys(
                        y for y in x
                        if y
                    )
                )
            )
        )
        .reset_index()
    )

else:

    pathway_summary = pd.DataFrame(
        columns=[
            "Pathway_ID",
            "Pathway_Name",
            "Candidate_BUSCO_Count",
            "Candidate_BUSCOs",
            "Functions"
        ]
    )


pathway_summary.to_csv(
    PATHWAY_SUMMARY,
    index=False
)

print(
    f"\nUnique pathways in final table: "
    f"{len(pathway_summary)}"
)

print(PATHWAY_SUMMARY)


# ============================================================
# FINAL STATISTICS
# ============================================================

total_buscos = len(final)

uniprot_buscos = 0
go_buscos = 0
kegg_buscos = 0
ec_buscos = 0
pathway_buscos = 0

if "UniProt_Count_Final" in final.columns:
    uniprot_buscos = (
        final["UniProt_Count_Final"] > 0
    ).sum()

if "GO_Count_Final" in final.columns:
    go_buscos = (
        final["GO_Count_Final"] > 0
    ).sum()

if "KEGG_Annotation_Count_Final" in final.columns:
    kegg_buscos = (
        final["KEGG_Annotation_Count_Final"] > 0
    ).sum()

if "EC_Count_Final" in final.columns:
    ec_buscos = (
        final["EC_Count_Final"] > 0
    ).sum()

if "KEGG_Pathway_Count" in final.columns:
    pathway_buscos = (
        final["KEGG_Pathway_Count"] > 0
    ).sum()

unique_kegg_genes = set()

if "KEGG_Gene_IDs" in final.columns:

    for value in final["KEGG_Gene_IDs"]:

        value = clean_value(value)

        for gene in value.split(";"):

            gene = gene.strip()

            if gene:
                unique_kegg_genes.add(gene)


# ============================================================
# WRITE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "=" * 80 + "\n"
    )

    report.write(
        "CARBON BREADTH — FINAL INTEGRATED ANNOTATION REPORT\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        "CANDIDATE SUMMARY\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        f"Total candidate BUSCOs       : {total_buscos}\n"
    )

    report.write(
        f"BUSCOs with UniProt          : {uniprot_buscos}\n"
    )

    report.write(
        f"BUSCOs with GO               : {go_buscos}\n"
    )

    report.write(
        f"BUSCOs with KEGG             : {kegg_buscos}\n"
    )

    report.write(
        f"BUSCOs with EC               : {ec_buscos}\n"
    )

    report.write(
        f"BUSCOs mapped to pathways    : {pathway_buscos}\n"
    )

    report.write(
        f"Unique KEGG gene IDs         : {len(unique_kegg_genes)}\n"
    )

    report.write(
        f"Unique KEGG pathways         : {len(pathway_summary)}\n"
    )

    report.write("\n")

    report.write(
        "BUSCO-LEVEL INTERPRETATION\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    for _, row in final.iterrows():

        busco = clean_value(
            row.get("BUSCO_ID", "")
        )

        function = clean_value(
            row.get(
                "Functional_Description",
                ""
            )
        )

        status = clean_value(
            row.get(
                "Annotation_Status",
                ""
            )
        )

        pathway_count = row.get(
            "KEGG_Pathway_Count",
            0
        )

        report.write(
            f"\n{busco}\n"
        )

        report.write(
            f"  Function          : {function}\n"
        )

        report.write(
            f"  KEGG pathways     : {pathway_count}\n"
        )

        report.write(
            f"  Annotation status : {status}\n"
        )

    report.write("\n\n")

    report.write(
        "OUTPUTS\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        f"Final integrated table:\n{FINAL_TABLE}\n\n"
    )

    report.write(
        f"Pathway-focused table:\n{PATHWAY_TABLE}\n\n"
    )

    report.write(
        f"Candidate summary:\n{CANDIDATE_SUMMARY}\n\n"
    )

    report.write(
        f"Unique pathway summary:\n{PATHWAY_SUMMARY}\n"
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL INTEGRATED ANNOTATION COMPLETE")
print("=" * 80)

print(f"\nTotal candidate BUSCOs       : {total_buscos}")
print(f"BUSCOs with UniProt          : {uniprot_buscos}")
print(f"BUSCOs with GO               : {go_buscos}")
print(f"BUSCOs with KEGG             : {kegg_buscos}")
print(f"BUSCOs with EC               : {ec_buscos}")
print(f"BUSCOs mapped to pathways    : {pathway_buscos}")
print(f"Unique KEGG gene IDs         : {len(unique_kegg_genes)}")
print(f"Unique KEGG pathways         : {len(pathway_summary)}")

print("\n" + "=" * 80)
print("OUTPUTS")
print("=" * 80)

print(f"\nFinal integrated annotation:")
print(FINAL_TABLE)

print(f"\nPathway-focused annotation:")
print(PATHWAY_TABLE)

print(f"\nCandidate summary:")
print(CANDIDATE_SUMMARY)

print(f"\nUnique pathway summary:")
print(PATHWAY_SUMMARY)

print(f"\nReport:")
print(REPORT_FILE)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)