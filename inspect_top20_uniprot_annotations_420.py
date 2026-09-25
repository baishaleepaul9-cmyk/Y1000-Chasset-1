import os
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\BUSCO20_annotation\UniProt\Carbon_Breadth_top20_UniProt_functional_annotations_420.csv"

OUTPUT_DIR = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\BUSCO20_annotation\UniProt"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_top20_UniProt_annotation_inspection_420.csv"
)

REPORT_FILE = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_top20_UniProt_annotation_inspection_420.txt"
)


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("INSPECTING TOP-20 BUSCO UNIPROT ANNOTATIONS")
print("=" * 80)

print("\nInput:")
print(INPUT_FILE)


# =============================================================================
# CHECK INPUT
# =============================================================================

if not os.path.exists(INPUT_FILE):

    print("\nERROR: Input file does not exist.")
    print(INPUT_FILE)

    raise SystemExit(1)


# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(INPUT_FILE)

print("\nFile loaded successfully.")

print("\nShape:")
print(df.shape)


# =============================================================================
# SHOW COLUMNS
# =============================================================================

print("\n" + "=" * 80)
print("AVAILABLE COLUMNS")
print("=" * 80)

for i, col in enumerate(df.columns, start=1):
    print(f"{i:3d}. {col}")


# =============================================================================
# NORMALIZE COLUMN NAMES FOR DETECTION
# =============================================================================

column_lookup = {
    str(col).strip().lower(): col
    for col in df.columns
}


def find_column(possible_names=None, contains=None):

    possible_names = possible_names or []
    contains = contains or []

    # Exact matches first
    for name in possible_names:

        key = name.lower()

        if key in column_lookup:
            return column_lookup[key]

    # Then substring matches
    for col in df.columns:

        low = str(col).lower()

        for term in contains:

            if term.lower() in low:
                return col

    return None


# =============================================================================
# DETECT IMPORTANT COLUMNS
# =============================================================================

BUSCO_COL = find_column(
    possible_names=[
        "BUSCO_ID",
        "BUSCO",
        "busco_id"
    ],
    contains=["busco"]
)

UNIPROT_COL = find_column(
    possible_names=[
        "UniProt",
        "UniProt_ID",
        "UniProt_Accession",
        "UniProt_Accession_ID"
    ],
    contains=["uniprot"]
)

FUNCTION_COL = find_column(
    possible_names=[
        "Function",
        "Functional_Description",
        "Description",
        "Protein_Function"
    ],
    contains=[
        "function",
        "description"
    ]
)

GO_COL = find_column(
    possible_names=[
        "GO",
        "GO_ID",
        "GO_Terms",
        "GO_Term",
        "Gene_Ontology"
    ],
    contains=[
        "go",
        "gene ontology"
    ]
)

KEGG_COL = find_column(
    possible_names=[
        "KEGG",
        "KEGG_ID",
        "KEGG_Pathway",
        "KEGG_Pathways",
        "KEGG_Orthology",
        "KO"
    ],
    contains=[
        "kegg",
        "ko"
    ]
)

EC_COL = find_column(
    possible_names=[
        "EC",
        "EC_Number",
        "EC_Numbers",
        "Enzyme_Commission"
    ],
    contains=[
        "ec",
        "enzyme commission"
    ]
)

PROTEIN_COL = find_column(
    possible_names=[
        "Protein",
        "Protein_ID",
        "Protein_Name"
    ],
    contains=[
        "protein"
    ]
)

GENE_COL = find_column(
    possible_names=[
        "Gene",
        "Gene_ID",
        "Gene_Name"
    ],
    contains=[
        "gene"
    ]
)


# =============================================================================
# DISPLAY DETECTED COLUMNS
# =============================================================================

print("\n" + "=" * 80)
print("DETECTED COLUMNS")
print("=" * 80)

print(f"BUSCO       : {BUSCO_COL}")
print(f"UniProt     : {UNIPROT_COL}")
print(f"Function    : {FUNCTION_COL}")
print(f"Protein     : {PROTEIN_COL}")
print(f"Gene        : {GENE_COL}")
print(f"GO          : {GO_COL}")
print(f"KEGG        : {KEGG_COL}")
print(f"EC          : {EC_COL}")


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def clean_values(series):

    if series is None:
        return []

    values = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    values = [
        x for x in values
        if x
        and x.lower() not in {
            "nan",
            "none",
            "na",
            "n/a",
            "-"
        }
    ]

    return list(dict.fromkeys(values))


def values_for_group(group, column):

    if column is None:
        return []

    return clean_values(group[column])


# =============================================================================
# OVERALL ANNOTATION COUNTS
# =============================================================================

print("\n" + "=" * 80)
print("OVERALL ANNOTATION AVAILABILITY")
print("=" * 80)


def nonempty_count(column):

    if column is None:
        return 0

    return len(clean_values(df[column]))


print(f"\nTotal records: {len(df)}")

print(
    f"UniProt records : "
    f"{nonempty_count(UNIPROT_COL)}"
)

print(
    f"Function records: "
    f"{nonempty_count(FUNCTION_COL)}"
)

print(
    f"GO records      : "
    f"{nonempty_count(GO_COL)}"
)

print(
    f"KEGG records    : "
    f"{nonempty_count(KEGG_COL)}"
)

print(
    f"EC records      : "
    f"{nonempty_count(EC_COL)}"
)


# =============================================================================
# UNIQUE BUSCOs
# =============================================================================

if BUSCO_COL is not None:

    buscos = clean_values(df[BUSCO_COL])

else:

    buscos = []


print("\nUnique BUSCO candidates:", len(buscos))


# =============================================================================
# BUSCO-LEVEL INSPECTION
# =============================================================================

print("\n" + "=" * 80)
print("BUSCO-LEVEL ANNOTATION SUMMARY")
print("=" * 80)


inspection_rows = []


for index, busco in enumerate(buscos, start=1):

    group = df[
        df[BUSCO_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        == busco
    ]

    uniprot_values = values_for_group(
        group,
        UNIPROT_COL
    )

    function_values = values_for_group(
        group,
        FUNCTION_COL
    )

    go_values = values_for_group(
        group,
        GO_COL
    )

    kegg_values = values_for_group(
        group,
        KEGG_COL
    )

    ec_values = values_for_group(
        group,
        EC_COL
    )

    protein_values = values_for_group(
        group,
        PROTEIN_COL
    )

    gene_values = values_for_group(
        group,
        GENE_COL
    )


    print("\n" + "-" * 80)

    print(
        f"[{index}/{len(buscos)}] {busco}"
    )

    print(
        "Function:",
        "; ".join(function_values)
        if function_values
        else "None"
    )

    print(
        "UniProt:",
        "; ".join(uniprot_values)
        if uniprot_values
        else "None"
    )

    print(
        "GO:",
        "; ".join(go_values)
        if go_values
        else "None"
    )

    print(
        "KEGG:",
        "; ".join(kegg_values)
        if kegg_values
        else "None"
    )

    print(
        "EC:",
        "; ".join(ec_values)
        if ec_values
        else "None"
    )


    # Store result
    inspection_rows.append({

        "BUSCO_ID":
            busco,

        "Functional_Description":
            "; ".join(function_values),

        "UniProt":
            "; ".join(uniprot_values),

        "Protein":
            "; ".join(protein_values),

        "Gene":
            "; ".join(gene_values),

        "GO":
            "; ".join(go_values),

        "KEGG":
            "; ".join(kegg_values),

        "EC":
            "; ".join(ec_values),

        "UniProt_Match_Count":
            len(uniprot_values),

        "GO_Count":
            len(go_values),

        "KEGG_Count":
            len(kegg_values),

        "EC_Count":
            len(ec_values)
    })


# =============================================================================
# SAVE INSPECTION TABLE
# =============================================================================

inspection_df = pd.DataFrame(
    inspection_rows
)

inspection_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

buscos_with_uniprot = sum(
    inspection_df["UniProt_Match_Count"] > 0
)

buscos_with_go = sum(
    inspection_df["GO_Count"] > 0
)

buscos_with_kegg = sum(
    inspection_df["KEGG_Count"] > 0
)

buscos_with_ec = sum(
    inspection_df["EC_Count"] > 0
)


# =============================================================================
# REPORT
# =============================================================================

report_lines = []

report_lines.append(
    "TOP-20 BUSCO UNIPROT ANNOTATION INSPECTION"
)

report_lines.append("=" * 80)

report_lines.append(
    f"Total BUSCO candidates: {len(buscos)}"
)

report_lines.append(
    f"BUSCOs with UniProt matches: {buscos_with_uniprot}"
)

report_lines.append(
    f"BUSCOs with GO annotations: {buscos_with_go}"
)

report_lines.append(
    f"BUSCOs with KEGG annotations: {buscos_with_kegg}"
)

report_lines.append(
    f"BUSCOs with EC annotations: {buscos_with_ec}"
)

report_lines.append("")

report_lines.append(
    "BUSCO-level results:"
)

report_lines.append("=" * 80)


for _, row in inspection_df.iterrows():

    report_lines.append("")

    report_lines.append(
        f"BUSCO: {row['BUSCO_ID']}"
    )

    report_lines.append(
        f"Function: {row['Functional_Description'] or 'None'}"
    )

    report_lines.append(
        f"UniProt: {row['UniProt'] or 'None'}"
    )

    report_lines.append(
        f"GO: {row['GO'] or 'None'}"
    )

    report_lines.append(
        f"KEGG: {row['KEGG'] or 'None'}"
    )

    report_lines.append(
        f"EC: {row['EC'] or 'None'}"
    )


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "\n".join(report_lines)
    )


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

print(
    f"\nBUSCO candidates: {len(buscos)}"
)

print(
    f"BUSCOs with UniProt matches: "
    f"{buscos_with_uniprot}"
)

print(
    f"BUSCOs with GO: "
    f"{buscos_with_go}"
)

print(
    f"BUSCOs with KEGG: "
    f"{buscos_with_kegg}"
)

print(
    f"BUSCOs with EC: "
    f"{buscos_with_ec}"
)

print("\nOutputs:")

print(
    "\nInspection table:"
)

print(OUTPUT_FILE)

print(
    "\nReport:"
)

print(REPORT_FILE)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)