import os
import re
import time
import requests
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = (
    r"C:\Y1000_chassis_project\results"
    r"\stage5_phylogeny_ml_dataset"
    r"\phylogeny_aware_ml"
    r"\phylogeny_cv"
    r"\model_training"
    r"\feature_interpretation_420"
    r"\functional_annotation_420"
    r"\BUSCO20_annotation"
)

PATHWAY_DIR = os.path.join(BASE_DIR, "pathway_analysis_420")

INPUT_FILE = os.path.join(
    PATHWAY_DIR,
    "Carbon_Breadth_top20_KEGG_gene_pathway_mapping_420.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "FINAL_CARBON_BREADTH_ANNOTATION_420",
    "pathway_name_resolution_420"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DETAILED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_top20_KEGG_gene_pathway_mapping_named_420.csv"
)

BUSCO_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_top20_BUSCO_pathway_summary_named_420.csv"
)

PATHWAY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_top20_unique_pathways_named_420.csv"
)

REPORT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "Carbon_Breadth_KEGG_pathway_name_resolution_report_420.txt"
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_pathway_id(value):
    """
    Convert pathway values such as:
        path:sce00414
        sce00414
        path:ko01100
    into:
        sce00414
        ko01100
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    # Remove possible prefixes
    value = re.sub(r"^path:", "", value)

    # Remove whitespace
    value = value.strip()

    return value


def get_pathway_name(pathway_id, session):
    """
    Retrieve KEGG pathway name using KEGG REST API.
    """

    if pathway_id is None:
        return None

    url = f"https://rest.kegg.jp/get/path:{pathway_id}"

    try:
        response = session.get(url, timeout=20)

        if response.status_code != 200:
            return None

        text = response.text

        # KEGG format:
        # NAME        Citrate cycle (TCA cycle)
        match = re.search(
            r"^NAME\s+(.+)$",
            text,
            flags=re.MULTILINE
        )

        if match:
            name = match.group(1).strip()

            # Remove trailing semicolon if present
            name = name.rstrip(";").strip()

            return name

    except requests.RequestException:
        return None

    return None


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("KEGG PATHWAY NAME RESOLUTION")
print("=" * 80)

print("\nInput:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    print("\nERROR: Input file not found.")
    print(INPUT_FILE)
    raise SystemExit(1)

# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(INPUT_FILE)

print("\nInput shape:")
print(df.shape)

print("\nColumns:")
for col in df.columns:
    print(" ", col)


# =============================================================================
# FIND PATHWAY COLUMN
# =============================================================================

possible_pathway_columns = [
    "Pathway_ID",
    "KEGG_Pathway",
    "KEGG_Pathway_ID",
    "Pathway",
    "pathway_id",
    "pathway",
    "PathwayID"
]

pathway_column = None

for col in possible_pathway_columns:
    if col in df.columns:
        pathway_column = col
        break

if pathway_column is None:

    # Try to find automatically
    for col in df.columns:
        col_lower = col.lower()

        if "pathway" in col_lower:
            pathway_column = col
            break

if pathway_column is None:
    print("\nERROR: Could not identify the KEGG pathway column.")
    print("Available columns:")
    print(list(df.columns))
    raise SystemExit(1)

print("\nDetected pathway column:")
print(pathway_column)


# =============================================================================
# CLEAN PATHWAY IDS
# =============================================================================

df["KEGG_Pathway_ID_Clean"] = df[pathway_column].apply(
    clean_pathway_id
)

unique_pathways = sorted(
    df["KEGG_Pathway_ID_Clean"]
    .dropna()
    .unique()
)

print("\nUnique KEGG pathways found:")
print(len(unique_pathways))


# =============================================================================
# KEGG API SESSION
# =============================================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "Carbon-Breadth-Annotation/1.0"
})


# =============================================================================
# RETRIEVE PATHWAY NAMES
# =============================================================================

print("\n" + "=" * 80)
print("RETRIEVING KEGG PATHWAY NAMES")
print("=" * 80)

pathway_names = {}

successful = 0
failed = 0

for i, pathway_id in enumerate(unique_pathways, start=1):

    print(
        f"[{i}/{len(unique_pathways)}] "
        f"{pathway_id}"
    )

    name = get_pathway_name(
        pathway_id,
        session
    )

    if name:

        pathway_names[pathway_id] = name

        successful += 1

        print(
            f"   Name: {name}"
        )

    else:

        pathway_names[pathway_id] = "Name not retrieved"

        failed += 1

        print(
            "   Name: FAILED"
        )

    # Avoid hammering the API
    time.sleep(0.2)


# =============================================================================
# ADD PATHWAY NAMES
# =============================================================================

df["KEGG_Pathway_Name"] = df[
    "KEGG_Pathway_ID_Clean"
].map(pathway_names)


# =============================================================================
# CLEAN OUTPUT COLUMN ORDER
# =============================================================================

# Put important columns first
preferred_columns = [
    "BUSCO_ID",
    "KEGG_Gene_ID",
    "KEGG_Gene",
    "Gene",
    "Protein",
    "Functional_Description",
    pathway_column,
    "KEGG_Pathway_ID_Clean",
    "KEGG_Pathway_Name"
]

ordered_columns = []

for col in preferred_columns:
    if col in df.columns and col not in ordered_columns:
        ordered_columns.append(col)

for col in df.columns:
    if col not in ordered_columns:
        ordered_columns.append(col)

df = df[ordered_columns]


# =============================================================================
# SAVE DETAILED TABLE
# =============================================================================

df.to_csv(
    DETAILED_OUTPUT,
    index=False
)


# =============================================================================
# BUSCO-LEVEL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("CREATING BUSCO-LEVEL PATHWAY SUMMARY")
print("=" * 80)

if "BUSCO_ID" not in df.columns:
    print("\nERROR: BUSCO_ID column not found.")
    raise SystemExit(1)

summary_records = []

for busco_id, group in df.groupby("BUSCO_ID"):

    pathway_ids = sorted(
        group["KEGG_Pathway_ID_Clean"]
        .dropna()
        .unique()
    )

    pathway_names_for_busco = []

    for pid in pathway_ids:

        pname = pathway_names.get(
            pid,
            "Name not retrieved"
        )

        pathway_names_for_busco.append(
            f"{pid} | {pname}"
        )

    kegg_genes = []

    if "KEGG_Gene_ID" in group.columns:

        for value in group["KEGG_Gene_ID"].dropna():

            if str(value).strip():

                kegg_genes.append(
                    str(value).strip()
                )

    # Remove duplicates
    kegg_genes = sorted(set(kegg_genes))

    functional_description = ""

    if "Functional_Description" in group.columns:

        values = (
            group["Functional_Description"]
            .dropna()
            .astype(str)
            .tolist()
        )

        if values:
            functional_description = values[0]

    summary_records.append({

        "BUSCO_ID": busco_id,

        "Functional_Description":
            functional_description,

        "KEGG_Gene_Count":
            len(kegg_genes),

        "KEGG_Pathway_Count":
            len(pathway_ids),

        "KEGG_Gene_IDs":
            "; ".join(kegg_genes),

        "KEGG_Pathway_IDs":
            "; ".join(pathway_ids),

        "KEGG_Pathway_Names":
            "; ".join(pathway_names_for_busco),

        "Pathway_Mapped":
            len(pathway_ids) > 0

    })


busco_summary = pd.DataFrame(
    summary_records
)

busco_summary.to_csv(
    BUSCO_OUTPUT,
    index=False
)


# =============================================================================
# PATHWAY-LEVEL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("CREATING PATHWAY-LEVEL SUMMARY")
print("=" * 80)

pathway_records = []

for pathway_id in unique_pathways:

    subset = df[
        df["KEGG_Pathway_ID_Clean"] == pathway_id
    ]

    buscos = sorted(
        subset["BUSCO_ID"]
        .dropna()
        .unique()
    )

    kegg_genes = []

    if "KEGG_Gene_ID" in subset.columns:

        for value in subset["KEGG_Gene_ID"].dropna():

            if str(value).strip():
                kegg_genes.append(
                    str(value).strip()
                )

    kegg_genes = sorted(set(kegg_genes))

    pathway_records.append({

        "KEGG_Pathway_ID":
            pathway_id,

        "KEGG_Pathway_Name":
            pathway_names.get(
                pathway_id,
                "Name not retrieved"
            ),

        "Carbon_Breadth_BUSCO_Count":
            len(buscos),

        "Carbon_Breadth_BUSCOs":
            "; ".join(buscos),

        "KEGG_Gene_Count":
            len(kegg_genes),

        "KEGG_Gene_IDs":
            "; ".join(kegg_genes)

    })


pathway_summary = pd.DataFrame(
    pathway_records
)

# Sort by number of Carbon Breadth candidates
pathway_summary = pathway_summary.sort_values(
    by=[
        "Carbon_Breadth_BUSCO_Count",
        "KEGG_Gene_Count"
    ],
    ascending=False
)

pathway_summary.to_csv(
    PATHWAY_OUTPUT,
    index=False
)


# =============================================================================
# REPORT
# =============================================================================

mapped_buscos = busco_summary[
    busco_summary["Pathway_Mapped"] == True
].shape[0]

total_buscos = busco_summary.shape[0]

unique_pathway_count = len(unique_pathways)

print("\n" + "=" * 80)
print("KEGG PATHWAY NAME RESOLUTION COMPLETE")
print("=" * 80)

print(
    f"\nTotal BUSCO candidates       : {total_buscos}"
)

print(
    f"BUSCOs mapped to pathways   : {mapped_buscos}"
)

print(
    f"Unique KEGG pathways        : {unique_pathway_count}"
)

print(
    f"Pathway names retrieved     : {successful}"
)

print(
    f"Pathway names failed        : {failed}"
)

print("\nOutputs:")

print(
    "\nDetailed mapping:"
)

print(
    DETAILED_OUTPUT
)

print(
    "\nBUSCO-level summary:"
)

print(
    BUSCO_OUTPUT
)

print(
    "\nPathway-level summary:"
)

print(
    PATHWAY_OUTPUT
)


with open(
    REPORT_OUTPUT,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "=" * 80 + "\n"
    )

    report.write(
        "CARBON BREADTH KEGG PATHWAY NAME RESOLUTION REPORT\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        f"Total BUSCO candidates: {total_buscos}\n"
    )

    report.write(
        f"BUSCOs mapped to pathways: {mapped_buscos}\n"
    )

    report.write(
        f"Unique KEGG pathways: {unique_pathway_count}\n"
    )

    report.write(
        f"Pathway names retrieved: {successful}\n"
    )

    report.write(
        f"Pathway names failed: {failed}\n\n"
    )

    report.write(
        "Pathway list:\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    for pathway_id in unique_pathways:

        report.write(
            f"{pathway_id}\t"
            f"{pathway_names.get(pathway_id, 'Name not retrieved')}\n"
        )


print("\nReport:")
print(REPORT_OUTPUT)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)