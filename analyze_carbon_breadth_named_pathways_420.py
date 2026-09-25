import os
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\BUSCO20_annotation"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "FINAL_CARBON_BREADTH_INTERPRETATION_NAMED_420",
    "tables",
    "Carbon_Breadth_BUSCO_named_pathway_summary_420.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "FINAL_CARBON_BREADTH_INTERPRETATION_NAMED_420",
    "analysis_corrected"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("CARBON BREADTH FINAL NAMED PATHWAY ANALYSIS")
print("=" * 80)
print()

print("Input:")
print(INPUT_FILE)
print()

# ============================================================
# CHECK FILE
# ============================================================

if not os.path.isfile(INPUT_FILE):
    print("ERROR: Input file not found.")
    raise SystemExit(1)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Input loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print()

print("Columns:")
for col in df.columns:
    print(f"  {col}")

print()

# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "BUSCO_ID",
    "KEGG_Gene_Count",
    "KEGG_Pathway_Count",
    "KEGG_Pathway_IDs",
    "KEGG_Pathway_Names",
    "Pathway_Mapped"
]

missing = [c for c in required if c not in df.columns]

if missing:
    print("ERROR: Missing required columns:")
    for c in missing:
        print(" ", c)
    raise SystemExit(1)

# ============================================================
# IMPORTANT:
# DO NOT USE KEGG_Pathway_Count AS THE PATHWAY COLUMN
# ============================================================

busco_col = "BUSCO_ID"
pathway_id_col = "KEGG_Pathway_IDs"
pathway_name_col = "KEGG_Pathway_Names"
mapped_col = "Pathway_Mapped"

# ============================================================
# NORMALIZE MAPPED FLAG
# ============================================================

df[mapped_col] = (
    df[mapped_col]
    .fillna(False)
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False
    })
    .fillna(False)
)

# ============================================================
# BASIC COVERAGE
# ============================================================

total_candidates = df[busco_col].nunique()

mapped_df = df[df[mapped_col] == True].copy()

mapped_candidates = mapped_df[busco_col].nunique()

not_mapped = total_candidates - mapped_candidates

print("=" * 80)
print("PATHWAY COVERAGE")
print("=" * 80)

print(f"Total candidates       : {total_candidates}")
print(f"Pathway-mapped         : {mapped_candidates}")
print(f"Not pathway-mapped     : {not_mapped}")

# ============================================================
# EXTRACT PATHWAY RECORDS
# ============================================================

records = []

for _, row in df.iterrows():

    busco = row[busco_col]

    pathway_ids = row[pathway_id_col]
    pathway_names = row[pathway_name_col]

    if pd.isna(pathway_ids):
        continue

    pathway_ids = str(pathway_ids).strip()

    if not pathway_ids or pathway_ids.lower() == "nan":
        continue

    # --------------------------------------------------------
    # Split pathway IDs
    # --------------------------------------------------------

    ids = [
        x.strip()
        for x in pathway_ids.replace("|", ";").split(";")
        if x.strip()
    ]

    # --------------------------------------------------------
    # Split pathway names
    # --------------------------------------------------------

    if pd.isna(pathway_names):
        names = []
    else:
        names = [
            x.strip()
            for x in str(pathway_names)
            .replace("|", ";")
            .split(";")
            if x.strip()
        ]

    # --------------------------------------------------------
    # Match IDs and names by position
    # --------------------------------------------------------

    for i, pathway_id in enumerate(ids):

        if i < len(names):
            pathway_name = names[i]
        else:
            pathway_name = ""

        records.append({
            "BUSCO_ID": busco,
            "KEGG_Pathway_ID": pathway_id,
            "KEGG_Pathway_Name": pathway_name
        })

# ============================================================
# CREATE PATHWAY DATAFRAME
# ============================================================

pathway_df = pd.DataFrame(records)

# Remove duplicates
if not pathway_df.empty:

    pathway_df = pathway_df.drop_duplicates(
        subset=[
            "BUSCO_ID",
            "KEGG_Pathway_ID",
            "KEGG_Pathway_Name"
        ]
    )

# ============================================================
# PATHWAY SUMMARY
# ============================================================

unique_pathways = (
    pathway_df["KEGG_Pathway_ID"]
    .nunique()
    if not pathway_df.empty
    else 0
)

print()
print("=" * 80)
print("VALIDATED PATHWAY SUMMARY")
print("=" * 80)

print(f"Unique KEGG pathways : {unique_pathways}")
print(f"Pathway records      : {len(pathway_df)}")

# ============================================================
# PATHWAY FREQUENCY
# ============================================================

if not pathway_df.empty:

    pathway_frequency = (
        pathway_df
        .groupby(
            [
                "KEGG_Pathway_ID",
                "KEGG_Pathway_Name"
            ]
        )
        .agg(
            BUSCO_Count=("BUSCO_ID", "nunique")
        )
        .reset_index()
        .sort_values(
            "BUSCO_Count",
            ascending=False
        )
    )

else:

    pathway_frequency = pd.DataFrame(
        columns=[
            "KEGG_Pathway_ID",
            "KEGG_Pathway_Name",
            "BUSCO_Count"
        ]
    )

# ============================================================
# BUSCO → PATHWAY SUMMARY
# ============================================================

busco_summary_records = []

for busco in df[busco_col].unique():

    sub = pathway_df[
        pathway_df["BUSCO_ID"] == busco
    ]

    if len(sub) > 0:

        ids = "; ".join(
            sub["KEGG_Pathway_ID"]
            .drop_duplicates()
            .tolist()
        )

        names = "; ".join(
            sub["KEGG_Pathway_Name"]
            .drop_duplicates()
            .tolist()
        )

        mapped = True

    else:

        ids = ""
        names = ""
        mapped = False

    original = df[
        df[busco_col] == busco
    ].iloc[0]

    busco_summary_records.append({

        "BUSCO_ID": busco,

        "KEGG_Gene_Count":
            original["KEGG_Gene_Count"],

        "KEGG_Pathway_Count":
            len(sub["KEGG_Pathway_ID"].unique())
            if len(sub) > 0
            else 0,

        "KEGG_Pathway_IDs":
            ids,

        "KEGG_Pathway_Names":
            names,

        "Pathway_Mapped":
            mapped
    })

busco_summary = pd.DataFrame(
    busco_summary_records
)

# ============================================================
# BUSCO × PATHWAY MATRIX
# ============================================================

if not pathway_df.empty:

    pathway_matrix = pd.crosstab(
        pathway_df["BUSCO_ID"],
        pathway_df["KEGG_Pathway_ID"]
    )

else:

    pathway_matrix = pd.DataFrame()

# ============================================================
# SAVE FILES
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_BUSCO_named_pathway_summary_CORRECTED_420.csv"
)

frequency_file = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_named_pathway_frequency_CORRECTED_420.csv"
)

records_file = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_named_pathway_records_CORRECTED_420.csv"
)

matrix_file = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_BUSCO_named_pathway_matrix_CORRECTED_420.csv"
)

busco_summary.to_csv(
    summary_file,
    index=False
)

pathway_frequency.to_csv(
    frequency_file,
    index=False
)

pathway_df.to_csv(
    records_file,
    index=False
)

pathway_matrix.to_csv(
    matrix_file
)

# ============================================================
# REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_corrected_named_pathway_analysis_report_420.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "FINAL CARBON BREADTH NAMED PATHWAY ANALYSIS\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Total candidates       : {total_candidates}\n"
    )

    f.write(
        f"Pathway-mapped         : {mapped_candidates}\n"
    )

    f.write(
        f"Not pathway-mapped     : {not_mapped}\n"
    )

    f.write(
        f"Unique KEGG pathways   : {unique_pathways}\n"
    )

    f.write(
        f"Pathway records        : {len(pathway_df)}\n"
    )

    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("PATHWAY FREQUENCY\n")
    f.write("=" * 80 + "\n\n")

    for _, row in pathway_frequency.iterrows():

        f.write(
            f"{row['KEGG_Pathway_ID']}\t"
            f"{row['KEGG_Pathway_Name']}\t"
            f"{row['BUSCO_Count']}\n"
        )

# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("CORRECTED ANALYSIS COMPLETE")
print("=" * 80)

print()
print(f"Total candidates       : {total_candidates}")
print(f"Pathway-mapped         : {mapped_candidates}")
print(f"Not pathway-mapped     : {not_mapped}")
print(f"Unique KEGG pathways   : {unique_pathways}")
print(f"Pathway records        : {len(pathway_df)}")

print()
print("OUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print()
print("OUTPUT FILES:")
print(summary_file)
print(frequency_file)
print(records_file)
print(matrix_file)
print(report_file)

print()
print("=" * 80)
print("DONE")
print("=" * 80)