# ================================================================
# CARBON BREADTH 420
# INSPECT TARGETED GFF/GTF ATTRIBUTES
#
# Purpose:
#   Inspect ONLY the already identified BUSCO -> GFF/GTF mappings.
#
#   We do NOT scan the 436 assemblies again.
#   We do NOT scan thousands of GFF files again.
#
#   The goal is to determine which identifiers are actually present
#   in the mapped GFF/GTF records:
#
#       ID
#       Parent
#       gene
#       gene_id
#       gene_name
#       locus_tag
#       protein_id
#       transcript_id
#       Dbxref
#       Name
#       product
#
# ================================================================

import os
import re
import json
import pandas as pd
from collections import Counter

print("=" * 80)
print("CARBON BREADTH — TARGETED GFF/GTF ATTRIBUTE INSPECTION")
print("=" * 80)


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = r"C:\Y1000_chassis_project"

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage5_phylogeny_ml_dataset",
    "phylogeny_aware_ml",
    "phylogeny_cv",
    "model_training",
    "feature_interpretation_420",
    "functional_annotation_420",
    "busco_gene_mapping_420"
)

TABLE_DIR = os.path.join(BASE_DIR, "tables")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ================================================================
# POSSIBLE INPUT FILES
# ================================================================

possible_files = [
    os.path.join(
        TABLE_DIR,
        "Carbon_Breadth_BUSCO_gene_protein_mapping_420.csv"
    ),

    os.path.join(
        TABLE_DIR,
        "Carbon_Breadth_targeted_BUSCO_GFF_mapping_420.csv"
    ),

    os.path.join(
        TABLE_DIR,
        "Carbon_Breadth_BUSCO_coordinate_mapping_420.csv"
    ),

    os.path.join(
        TABLE_DIR,
        "Carbon_Breadth_BUSCO_GFF_mapping_420.csv"
    ),
]


# ================================================================
# FIND THE EXISTING MAPPING TABLE
# ================================================================

mapping_file = None

for f in possible_files:
    if os.path.exists(f):
        mapping_file = f
        break


# If exact filenames differ, search only this output directory.
if mapping_file is None:

    candidates = []

    for name in os.listdir(TABLE_DIR):

        lname = name.lower()

        if (
            name.endswith(".csv")
            and (
                "mapping" in lname
                or "gff" in lname
                or "busco" in lname
            )
        ):
            candidates.append(
                os.path.join(TABLE_DIR, name)
            )

    if len(candidates) == 1:
        mapping_file = candidates[0]

    elif len(candidates) > 1:

        print("\nPossible mapping files found:")

        for i, f in enumerate(candidates, 1):
            print(f"  [{i}] {os.path.basename(f)}")

        # Prefer files containing targeted/GFF terminology
        preferred = [
            f for f in candidates
            if (
                "targeted" in os.path.basename(f).lower()
                or "gff" in os.path.basename(f).lower()
            )
        ]

        if preferred:
            mapping_file = preferred[0]

        else:
            mapping_file = candidates[0]


if mapping_file is None:

    raise FileNotFoundError(
        "\nCould not find the targeted BUSCO/GFF mapping table.\n"
        f"Expected directory:\n{TABLE_DIR}\n\n"
        "Run the targeted BUSCO mapping script first."
    )


print("\nMapping table:")
print(mapping_file)


# ================================================================
# LOAD TABLE
# ================================================================

df = pd.read_csv(
    mapping_file,
    low_memory=False
)

print("\nMapping table shape:")
print(df.shape)

print("\nColumns:")
for c in df.columns:
    print(" ", c)


# ================================================================
# IDENTIFY BUSCO COLUMN
# ================================================================

busco_candidates = [
    "BUSCO_ID",
    "busco_id",
    "Busco_ID",
    "busco"
]

busco_col = None

for c in busco_candidates:
    if c in df.columns:
        busco_col = c
        break

if busco_col is None:
    raise KeyError(
        "BUSCO_ID column could not be identified."
    )


# ================================================================
# IDENTIFY ATTRIBUTE-LIKE COLUMNS
# ================================================================

attribute_candidates = [
    "Attributes",
    "Attribute",
    "attributes",
    "attribute",
    "GFF_Attributes",
    "GTF_Attributes",
    "gff_attributes",
    "Attributes_Raw",
    "Raw_Attributes",
    "raw_attributes"
]

attribute_col = None

for c in attribute_candidates:
    if c in df.columns:
        attribute_col = c
        break


print("\n" + "=" * 80)
print("IDENTIFYING RAW ATTRIBUTE COLUMN")
print("=" * 80)

if attribute_col:

    print("Attribute column found:")
    print(attribute_col)

else:

    print(
        "No dedicated attribute column was found."
    )

    print(
        "\nRelevant columns available in the mapping table:"
    )

    for c in df.columns:

        lc = c.lower()

        if any(
            x in lc
            for x in [
                "id",
                "gene",
                "protein",
                "locus",
                "parent",
                "name",
                "product",
                "attribute",
                "gff",
                "gtf",
                "feature",
                "coordinate",
                "seq"
            ]
        ):
            print(" ", c)


# ================================================================
# SHOW SAMPLE RECORDS
# ================================================================

print("\n" + "=" * 80)
print("SAMPLE MAPPED RECORDS")
print("=" * 80)

show_cols = [
    c for c in [
        busco_col,
        "Species",
        "Assembly_Accession",
        "Sequence",
        "Start",
        "End",
        "Strand",
        "Feature",
        "GFF_File",
        "GTF_File",
        attribute_col
    ]
    if c is not None and c in df.columns
]

if show_cols:

    print(
        df[show_cols]
        .head(20)
        .to_string(index=False)
    )

else:

    print(
        df.head(20)
        .to_string(index=False)
    )


# ================================================================
# EXTRACT IDENTIFIER-LIKE VALUES FROM ALL STRING COLUMNS
# ================================================================

print("\n" + "=" * 80)
print("SEARCHING ALL COLUMNS FOR IDENTIFIER-LIKE INFORMATION")
print("=" * 80)


identifier_patterns = {

    "gene_id": [
        r'gene_id[=\s"]+([^";\s]+)',
        r'geneID[=\s"]+([^";\s]+)',
        r'gene[=\s"]+([^";\s]+)',
    ],

    "gene_name": [
        r'gene_name[=\s"]+([^";\s]+)',
        r'gene_name[=\s"]+([^";\s]+)',
        r'Name[=\s"]+([^";\s]+)',
    ],

    "locus_tag": [
        r'locus_tag[=\s"]+([^";\s]+)',
        r'locus[=\s"]+([^";\s]+)',
    ],

    "protein_id": [
        r'protein_id[=\s"]+([^";\s]+)',
        r'protein[=\s"]+([^";\s]+)',
        r'protein_accession[=\s"]+([^";\s]+)',
    ],

    "transcript_id": [
        r'transcript_id[=\s"]+([^";\s]+)',
        r'transcript[=\s"]+([^";\s]+)',
    ],

    "parent": [
        r'Parent[=\s"]+([^";\s]+)',
        r'parent[=\s"]+([^";\s]+)',
    ],

    "feature_id": [
        r'ID[=\s"]+([^";\s]+)',
        r'id[=\s"]+([^";\s]+)',
    ],

    "dbxref": [
        r'Dbxref[=\s"]+([^";\s]+)',
        r'dbxref[=\s"]+([^";\s]+)',
    ],

    "product": [
        r'product[=\s"]+([^";]+)',
        r'Product[=\s"]+([^";]+)',
    ],

    "name": [
        r'Name[=\s"]+([^";\s]+)',
        r'name[=\s"]+([^";\s]+)',
    ],
}


def extract_value(text, patterns):

    if pd.isna(text):
        return None

    text = str(text)

    for pattern in patterns:

        m = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if m:

            value = m.group(1).strip()

            if value:
                return value

    return None


# ================================================================
# BUILD DIAGNOSTIC TABLE
# ================================================================

diagnostic = df.copy()


# Combine ALL textual columns into one searchable string.
text_columns = []

for c in df.columns:

    if (
        df[c].dtype == "object"
        or pd.api.types.is_string_dtype(df[c])
    ):
        text_columns.append(c)


if text_columns:

    diagnostic["_ALL_TEXT"] = (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" | ".join, axis=1)
    )

else:

    diagnostic["_ALL_TEXT"] = ""


# Extract identifiers
for identifier, patterns in identifier_patterns.items():

    diagnostic[identifier] = diagnostic[
        "_ALL_TEXT"
    ].apply(
        lambda x: extract_value(
            x,
            patterns
        )
    )


# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("IDENTIFIER RECOVERY SUMMARY")
print("=" * 80)

identifier_summary = []

for identifier in identifier_patterns:

    nonempty = (
        diagnostic[identifier]
        .notna()
        & (
            diagnostic[identifier]
            .astype(str)
            .str.strip()
            != ""
        )
    )

    identifier_summary.append({

        "Identifier_Type": identifier,

        "Records_With_Value":
            int(nonempty.sum()),

        "Unique_Values":
            int(
                diagnostic.loc[
                    nonempty,
                    identifier
                ].nunique()
            )
    })


summary_df = pd.DataFrame(
    identifier_summary
)

print(
    summary_df.to_string(
        index=False
    )
)


# ================================================================
# SHOW RECOVERED VALUES
# ================================================================

print("\n" + "=" * 80)
print("RECOVERED IDENTIFIER EXAMPLES")
print("=" * 80)

for identifier in identifier_patterns:

    vals = (
        diagnostic[identifier]
        .dropna()
        .astype(str)
        .str.strip()
    )

    vals = vals[
        vals != ""
    ].unique()

    print(
        f"\n{identifier.upper()}: "
        f"{len(vals)} unique"
    )

    for value in vals[:15]:

        print(
            "  ",
            value
        )


# ================================================================
# BUSCO-LEVEL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("BUSCO-LEVEL IDENTIFIER SUMMARY")
print("=" * 80)


busco_summary = []

for busco, group in diagnostic.groupby(
    busco_col,
    dropna=True
):

    row = {
        "BUSCO_ID": busco
    }

    for identifier in identifier_patterns:

        values = (
            group[identifier]
            .dropna()
            .astype(str)
            .str.strip()
        )

        values = [
            x for x in values
            if x
        ]

        unique_values = sorted(
            set(values)
        )

        row[
            identifier
        ] = ";".join(
            unique_values[:20]
        )

    busco_summary.append(row)


busco_summary_df = pd.DataFrame(
    busco_summary
)


# ================================================================
# SAVE DIAGNOSTIC RECORDS
# ================================================================

diagnostic_output = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_targeted_GFF_identifier_diagnostic_420.csv"
)

diagnostic.to_csv(
    diagnostic_output,
    index=False
)


# ================================================================
# SAVE BUSCO SUMMARY
# ================================================================

busco_output = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_identifier_summary_420.csv"
)

busco_summary_df.to_csv(
    busco_output,
    index=False
)


# ================================================================
# SAVE IDENTIFIER SUMMARY
# ================================================================

summary_output = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_identifier_recovery_summary_420.csv"
)

summary_df.to_csv(
    summary_output,
    index=False
)


# ================================================================
# SAVE RAW SAMPLE
# ================================================================

sample_output = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_targeted_GFF_raw_sample_420.csv"
)

df.head(
    min(100, len(df))
).to_csv(
    sample_output,
    index=False
)


# ================================================================
# TEXT REPORT
# ================================================================

report_output = os.path.join(
    REPORT_DIR,
    "Carbon_Breadth_targeted_GFF_identifier_diagnostic_420.txt"
)

with open(
    report_output,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CARBON BREADTH — TARGETED GFF/GTF IDENTIFIER DIAGNOSTIC\n"
    )

    f.write("=" * 80 + "\n\n")

    f.write(
        f"Input mapping file:\n{mapping_file}\n\n"
    )

    f.write(
        f"Total mapping records: {len(df)}\n"
    )

    f.write(
        f"Unique BUSCOs: {df[busco_col].nunique()}\n\n"
    )

    f.write(
        "IDENTIFIER SUMMARY\n"
    )

    f.write(
        summary_df.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "BUSCO-LEVEL SUMMARY\n"
    )

    f.write(
        busco_summary_df.to_string(
            index=False
        )
    )


# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

print("\nNo new GFF/GTF search was performed.")
print(
    f"Records inspected: {len(df)}"
)

print(
    f"BUSCOs inspected: "
    f"{df[busco_col].nunique()}"
)

print("\nOutputs:")

print(
    "\nDiagnostic records:"
)
print(
    diagnostic_output
)

print(
    "\nBUSCO identifier summary:"
)
print(
    busco_output
)

print(
    "\nIdentifier recovery summary:"
)
print(
    summary_output
)

print(
    "\nRaw sample:"
)
print(
    sample_output
)

print(
    "\nReport:"
)
print(
    report_output
)

print("\n" + "=" * 80)
print("NEXT STEP")
print("=" * 80)

print(
    """
Inspect the IDENTIFIER RECOVERY SUMMARY.

If gene/protein IDs are present, we will use them directly.

If only IDs/Parent/Dbxref are present, we will trace those
relationships to CDS/protein records.

If only locus/feature identifiers are present, we will use
the corresponding assembly annotation to recover the protein
product.

DO NOT run another large GFF search yet.
"""
)

print("=" * 80)