import os
import re
import time
import requests
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset"
    r"\phylogeny_aware_ml\phylogeny_cv\model_training"
    r"\feature_interpretation_420\functional_annotation_420"
    r"\BUSCO20_annotation\UniProt"
    r"\Carbon_Breadth_top20_UniProt_annotation_inspection_420.csv"
)

BASE_OUTPUT = (
    r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset"
    r"\phylogeny_aware_ml\phylogeny_cv\model_training"
    r"\feature_interpretation_420\functional_annotation_420"
    r"\BUSCO20_annotation\pathway_analysis_420"
)

os.makedirs(BASE_OUTPUT, exist_ok=True)

DETAIL_FILE = os.path.join(
    BASE_OUTPUT,
    "Carbon_Breadth_top20_KEGG_gene_pathway_mapping_420.csv"
)

BUSCO_FILE = os.path.join(
    BASE_OUTPUT,
    "Carbon_Breadth_top20_BUSCO_pathway_summary_420.csv"
)

PATHWAY_FILE = os.path.join(
    BASE_OUTPUT,
    "Carbon_Breadth_top20_unique_pathways_420.csv"
)

REPORT_FILE = os.path.join(
    BASE_OUTPUT,
    "Carbon_Breadth_top20_KEGG_pathway_mapping_report_420.txt"
)

FAILED_FILE = os.path.join(
    BASE_OUTPUT,
    "Carbon_Breadth_top20_KEGG_pathway_failures_420.csv"
)

# KEGG REST API
KEGG_BASE = "https://rest.kegg.jp"

# KEGG requests should be kept below the API rate limit.
REQUEST_DELAY = 0.4

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """Convert NaN/blank values to empty string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_kegg_gene_ids(value):
    """
    Extract KEGG organism-specific gene IDs.

    Examples:
        sce:YNL123W
        cgr:2890766
        ago:AGOS_ABR134C
        kaf:KAFR_0I01310
    """

    text = clean_value(value)

    if not text:
        return []

    # KEGG organism code followed by colon and gene/accession.
    #
    # Organism codes are normally 3-4 characters.
    # Gene identifiers may contain letters, numbers,
    # underscores, hyphens and periods.
    pattern = r"\b([A-Za-z0-9]{3,4}:[A-Za-z0-9_.-]+)\b"

    matches = re.findall(pattern, text)

    # Remove duplicates while preserving order.
    seen = set()
    result = []

    for match in matches:
        match = match.strip()

        if match not in seen:
            seen.add(match)
            result.append(match)

    return result


def query_kegg_pathways(kegg_gene):
    """
    Query KEGG REST API:

        /link/pathway/<organism>:<gene>

    Returns pathway IDs.
    """

    url = f"{KEGG_BASE}/link/pathway/{kegg_gene}"

    try:
        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "Carbon-Breadth-Project/420"
            }
        )

        if response.status_code != 200:
            return [], f"HTTP_{response.status_code}"

        pathways = []

        for line in response.text.strip().splitlines():

            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) >= 2:

                pathway_id = parts[1].strip()

                if pathway_id:
                    pathways.append(pathway_id)

        return sorted(set(pathways)), ""

    except Exception as e:
        return [], str(e)


def get_kegg_pathway_names(pathway_ids):
    """
    Retrieve pathway names from KEGG.

    KEGG accepts multiple IDs in a single GET request,
    with a maximum of 10 entries per request.
    """

    pathway_names = {}

    pathway_ids = sorted(set(pathway_ids))

    for i in range(0, len(pathway_ids), 10):

        batch = pathway_ids[i:i + 10]

        if not batch:
            continue

        joined = "+".join(batch)

        url = f"{KEGG_BASE}/list/{joined}"

        try:

            response = requests.get(
                url,
                timeout=30,
                headers={
                    "User-Agent": "Carbon-Breadth-Project/420"
                }
            )

            if response.status_code != 200:
                continue

            for line in response.text.strip().splitlines():

                parts = line.split("\t", 1)

                if len(parts) == 2:

                    pid = parts[0].strip()
                    pname = parts[1].strip()

                    pathway_names[pid] = pname

        except Exception:
            pass

        time.sleep(REQUEST_DELAY)

    return pathway_names


# ============================================================
# LOAD INPUT
# ============================================================

print("=" * 80)
print("CARBON BREADTH TOP-20 KEGG → PATHWAY MAPPING")
print("=" * 80)

print("\nInput:")
print(INPUT_FILE)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nInput shape:")
print(df.shape)

print("\nColumns:")
for col in df.columns:
    print(" ", col)

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "BUSCO_ID" not in df.columns:
    raise ValueError(
        "BUSCO_ID column not found in input file."
    )

if "KEGG" not in df.columns:
    raise ValueError(
        "KEGG column not found in input file."
    )

# ============================================================
# EXTRACT KEGG IDS
# ============================================================

print("\n" + "=" * 80)
print("EXTRACTING KEGG GENE IDENTIFIERS")
print("=" * 80)

records = []

total_kegg_ids = 0
buscos_with_kegg = 0

for _, row in df.iterrows():

    busco = clean_value(row["BUSCO_ID"])
    kegg_value = clean_value(row["KEGG"])

    kegg_ids = extract_kegg_gene_ids(kegg_value)

    if kegg_ids:
        buscos_with_kegg += 1

    total_kegg_ids += len(kegg_ids)

    for kegg_gene in kegg_ids:

        records.append({
            "BUSCO_ID": busco,
            "KEGG_Gene_ID": kegg_gene
        })

print("\nBUSCOs with KEGG annotations :", buscos_with_kegg)
print("Total KEGG gene IDs          :", total_kegg_ids)

# ============================================================
# STOP IF NO KEGG IDS
# ============================================================

if not records:

    raise ValueError(
        "\nNo KEGG gene identifiers were recovered.\n"
        "Check the KEGG column format."
    )

# ============================================================
# REMOVE DUPLICATES
# ============================================================

records_df = pd.DataFrame(records)

records_df = records_df.drop_duplicates(
    subset=["BUSCO_ID", "KEGG_Gene_ID"]
).reset_index(drop=True)

print(
    "\nUnique BUSCO/KEGG gene pairs:",
    len(records_df)
)

# ============================================================
# QUERY KEGG
# ============================================================

print("\n" + "=" * 80)
print("QUERYING KEGG FOR PATHWAYS")
print("=" * 80)

mapping_records = []
failed_records = []

cache = {}

unique_genes = records_df["KEGG_Gene_ID"].unique().tolist()

print("\nUnique KEGG genes to query:", len(unique_genes))

for i, kegg_gene in enumerate(unique_genes, start=1):

    print(
        f"[{i}/{len(unique_genes)}] {kegg_gene}",
        flush=True
    )

    if kegg_gene in cache:

        pathways, error = cache[kegg_gene]

    else:

        pathways, error = query_kegg_pathways(
            kegg_gene
        )

        cache[kegg_gene] = (
            pathways,
            error
        )

        time.sleep(REQUEST_DELAY)

    matching_buscos = records_df.loc[
        records_df["KEGG_Gene_ID"] == kegg_gene,
        "BUSCO_ID"
    ].unique()

    if error:

        print(
            f"   ERROR: {error}"
        )

        for busco in matching_buscos:

            failed_records.append({
                "BUSCO_ID": busco,
                "KEGG_Gene_ID": kegg_gene,
                "Error": error
            })

        continue

    if not pathways:

        print(
            "   Pathways: 0"
        )

        for busco in matching_buscos:

            failed_records.append({
                "BUSCO_ID": busco,
                "KEGG_Gene_ID": kegg_gene,
                "Error": "No KEGG pathway links returned"
            })

        continue

    print(
        f"   Pathways: {len(pathways)}"
    )

    for busco in matching_buscos:

        for pathway in pathways:

            mapping_records.append({
                "BUSCO_ID": busco,
                "KEGG_Gene_ID": kegg_gene,
                "KEGG_Pathway_ID": pathway
            })

# ============================================================
# BUILD MAPPING TABLE
# ============================================================

mapping_df = pd.DataFrame(
    mapping_records
)

if mapping_df.empty:

    print("\nNo pathway mappings were recovered.")

    pd.DataFrame(
        failed_records
    ).to_csv(
        FAILED_FILE,
        index=False
    )

    raise ValueError(
        "\nKEGG gene IDs were found, but KEGG returned "
        "no pathway mappings."
    )

mapping_df = mapping_df.drop_duplicates()

# ============================================================
# PATHWAY NAMES
# ============================================================

print("\n" + "=" * 80)
print("RETRIEVING KEGG PATHWAY NAMES")
print("=" * 80)

unique_pathways = mapping_df[
    "KEGG_Pathway_ID"
].dropna().unique().tolist()

print(
    "\nUnique pathway IDs:",
    len(unique_pathways)
)

pathway_names = get_kegg_pathway_names(
    unique_pathways
)

mapping_df["KEGG_Pathway_Name"] = (
    mapping_df["KEGG_Pathway_ID"]
    .map(pathway_names)
)

# ============================================================
# BUSCO-LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("BUILDING BUSCO-LEVEL SUMMARY")
print("=" * 80)

summary_records = []

all_buscos = df[
    "BUSCO_ID"
].dropna().astype(str).unique().tolist()

for busco in all_buscos:

    subset = mapping_df[
        mapping_df["BUSCO_ID"] == busco
    ]

    genes = sorted(
        subset["KEGG_Gene_ID"]
        .dropna()
        .unique()
        .tolist()
    )

    pathways = sorted(
        subset["KEGG_Pathway_ID"]
        .dropna()
        .unique()
        .tolist()
    )

    pathway_names_busco = sorted(
        subset["KEGG_Pathway_Name"]
        .dropna()
        .unique()
        .tolist()
    )

    summary_records.append({

        "BUSCO_ID": busco,

        "KEGG_Gene_Count": len(genes),

        "KEGG_Gene_IDs": "; ".join(genes),

        "Pathway_Count": len(pathways),

        "KEGG_Pathway_IDs": "; ".join(pathways),

        "KEGG_Pathway_Names": "; ".join(
            pathway_names_busco
        )

    })

busco_summary_df = pd.DataFrame(
    summary_records
)

# ============================================================
# PATHWAY-LEVEL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("BUILDING PATHWAY-LEVEL SUMMARY")
print("=" * 80)

pathway_records = []

for pathway_id, group in mapping_df.groupby(
    "KEGG_Pathway_ID"
):

    buscos = sorted(
        group["BUSCO_ID"]
        .dropna()
        .unique()
        .tolist()
    )

    genes = sorted(
        group["KEGG_Gene_ID"]
        .dropna()
        .unique()
        .tolist()
    )

    names = (
        group["KEGG_Pathway_Name"]
        .dropna()
        .unique()
        .tolist()
    )

    pathway_name = (
        names[0]
        if names
        else ""
    )

    pathway_records.append({

        "KEGG_Pathway_ID": pathway_id,

        "KEGG_Pathway_Name": pathway_name,

        "BUSCO_Count": len(buscos),

        "BUSCO_IDs": "; ".join(buscos),

        "KEGG_Gene_Count": len(genes),

        "KEGG_Gene_IDs": "; ".join(genes)

    })

pathway_summary_df = pd.DataFrame(
    pathway_records
)

pathway_summary_df = pathway_summary_df.sort_values(
    by=["BUSCO_Count", "KEGG_Pathway_ID"],
    ascending=[False, True]
).reset_index(drop=True)

# ============================================================
# SAVE OUTPUTS
# ============================================================

mapping_df.to_csv(
    DETAIL_FILE,
    index=False
)

busco_summary_df.to_csv(
    BUSCO_FILE,
    index=False
)

pathway_summary_df.to_csv(
    PATHWAY_FILE,
    index=False
)

pd.DataFrame(
    failed_records
).to_csv(
    FAILED_FILE,
    index=False
)

# ============================================================
# REPORT
# ============================================================

buscos_mapped = mapping_df[
    "BUSCO_ID"
].nunique()

pathway_count = mapping_df[
    "KEGG_Pathway_ID"
].nunique()

gene_count = mapping_df[
    "KEGG_Gene_ID"
].nunique()

report_lines = [

    "=" * 80,

    "CARBON BREADTH TOP-20 KEGG PATHWAY MAPPING REPORT",

    "=" * 80,

    "",

    f"Total candidate BUSCOs       : {len(all_buscos)}",

    f"BUSCOs with KEGG annotations : {buscos_with_kegg}",

    f"Unique KEGG gene IDs         : {gene_count}",

    f"BUSCOs mapped to pathways    : {buscos_mapped}",

    f"Unique KEGG pathways         : {pathway_count}",

    f"Pathway mapping records      : {len(mapping_df)}",

    f"Failed query records         : {len(failed_records)}",

    "",

    "=" * 80,

    "BUSCO → PATHWAY SUMMARY",

    "=" * 80,

    ""

]

for _, row in busco_summary_df.iterrows():

    report_lines.append(
        f"{row['BUSCO_ID']}"
    )

    report_lines.append(
        f"  KEGG genes: {row['KEGG_Gene_Count']}"
    )

    report_lines.append(
        f"  Pathways : {row['Pathway_Count']}"
    )

    if row["KEGG_Pathway_Names"]:

        report_lines.append(
            f"  Names    : {row['KEGG_Pathway_Names']}"
        )

    else:

        report_lines.append(
            "  Names    : None"
        )

    report_lines.append("")

report_lines.extend([

    "=" * 80,

    "OUTPUTS",

    "=" * 80,

    "",

    f"Detailed mapping:",
    f"{DETAIL_FILE}",

    "",

    f"BUSCO summary:",
    f"{BUSCO_FILE}",

    "",

    f"Pathway summary:",
    f"{PATHWAY_FILE}",

    "",

    f"Failed queries:",
    f"{FAILED_FILE}",

    "",

    f"Report:",
    f"{REPORT_FILE}",

    "",

    "=" * 80,

    "COMPLETE",

    "=" * 80

])

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report_lines)
    )

# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("KEGG / PATHWAY MAPPING COMPLETE")
print("=" * 80)

print(
    f"\nTotal candidate BUSCOs       : {len(all_buscos)}"
)

print(
    f"BUSCOs with KEGG annotations : {buscos_with_kegg}"
)

print(
    f"Unique KEGG gene IDs         : {gene_count}"
)

print(
    f"BUSCOs mapped to pathways    : {buscos_mapped}"
)

print(
    f"Unique KEGG pathways         : {pathway_count}"
)

print(
    f"Pathway mapping records      : {len(mapping_df)}"
)

print("\n" + "=" * 80)
print("OUTPUTS")
print("=" * 80)

print("\nDetailed mapping:")
print(DETAIL_FILE)

print("\nBUSCO-level summary:")
print(BUSCO_FILE)

print("\nPathway-level summary:")
print(PATHWAY_FILE)

print("\nFailed queries:")
print(FAILED_FILE)

print("\nReport:")
print(REPORT_FILE)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)