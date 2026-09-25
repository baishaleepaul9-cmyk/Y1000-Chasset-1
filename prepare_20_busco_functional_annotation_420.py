# ================================================================
# PREPARE TOP-20 BUSCO FUNCTIONAL ANNOTATION INPUT
# Carbon Breadth project - 420
# ================================================================

import os
import re
import pandas as pd

# ----------------------------------------------------------------
# INPUT
# ----------------------------------------------------------------

MAPPING_FILE = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\busco_gene_mapping_420\tables\Carbon_Breadth_BUSCO_gene_protein_mapping_420.csv"

# ----------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------

OUT_DIR = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\BUSCO20_annotation"

os.makedirs(OUT_DIR, exist_ok=True)

OUT_TABLE = os.path.join(
    OUT_DIR,
    "Carbon_Breadth_top20_BUSCO_functional_annotation_input_420.csv"
)

OUT_REPORT = os.path.join(
    OUT_DIR,
    "Carbon_Breadth_top20_BUSCO_functional_annotation_report_420.txt"
)

# ----------------------------------------------------------------
# TOP 20 BUSCOs
# ----------------------------------------------------------------

TOP20 = [
    "1679at4891",
    "2307at4891",
    "2471at4891",
    "3574at4891",
    "4322at4891",
    "5746at4891",
    "6711at4891",
    "9782at4891",
    "12468at4891",
    "12523at4891",
    "18913at4891",
    "22532at4891",
    "23379at4891",
    "24318at4891",
    "27915at4891",
    "28058at4891",
    "29457at4891",
    "33531at4891",
    "34052at4891",
    "36839at4891",
]

print("=" * 80)
print("PREPARING TOP-20 BUSCO FUNCTIONAL ANNOTATION INPUT")
print("=" * 80)

# ----------------------------------------------------------------
# LOAD
# ----------------------------------------------------------------

print("\nLoading mapping table:")
print(MAPPING_FILE)

df = pd.read_csv(MAPPING_FILE, low_memory=False)

print("\nMapping shape:")
print(df.shape)

if "BUSCO_ID" not in df.columns:
    raise ValueError("BUSCO_ID column not found.")

# ----------------------------------------------------------------
# FILTER TO TOP 20
# ----------------------------------------------------------------

top = df[df["BUSCO_ID"].astype(str).isin(TOP20)].copy()

print("\nRows belonging to top-20 BUSCOs:")
print(len(top))

# ----------------------------------------------------------------
# ONE REPRESENTATIVE FUNCTIONAL RECORD PER BUSCO
# ----------------------------------------------------------------

# Prefer records containing functional descriptions.
description_cols = [
    "Functional_Protein_Description",
    "Functional_Description",
    "Protein_Name",
    "Gene_Name",
    "Protein_ID",
    "Gene_ID",
    "Locus_Tag",
]

for col in description_cols:
    if col not in top.columns:
        top[col] = ""

# Normalize missing values
for col in description_cols:
    top[col] = top[col].fillna("").astype(str).str.strip()

# Create a combined functional description.
def choose_function(row):

    for col in [
        "Functional_Protein_Description",
        "Functional_Description",
        "Protein_Name",
        "Gene_Name",
    ]:
        value = str(row.get(col, "")).strip()

        if value and value.lower() not in [
            "nan",
            "none",
            "unknown",
            "na",
        ]:
            return value

    return ""

top["Selected_Function"] = top.apply(
    choose_function,
    axis=1
)

# ----------------------------------------------------------------
# SELECT REPRESENTATIVE RECORD
# ----------------------------------------------------------------

representatives = []

for busco in TOP20:

    sub = top[top["BUSCO_ID"].astype(str) == busco].copy()

    if sub.empty:
        representatives.append({
            "BUSCO_ID": busco,
            "Representative_Species": "",
            "Assembly_Accession": "",
            "Sequence": "",
            "Selected_Function": "",
            "GO_Terms": "",
            "KEGG_ID": "",
            "EC_Number": "",
            "Annotation_Status": "BUSCO not found"
        })
        continue

    # Prefer a row with a functional description.
    sub["_has_function"] = (
        sub["Selected_Function"].str.strip() != ""
    )

    sub = sub.sort_values(
        "_has_function",
        ascending=False
    )

    r = sub.iloc[0]

    representatives.append({
        "BUSCO_ID": busco,
        "Representative_Species": r.get("Species", ""),
        "Assembly_Accession": r.get("Assembly_Accession", ""),
        "Sequence": r.get("Sequence", ""),
        "Selected_Function": r.get("Selected_Function", ""),
        "GO_Terms": "",
        "KEGG_ID": "",
        "EC_Number": "",
        "Annotation_Status": (
            "Functional description available"
            if r.get("Selected_Function", "").strip()
            else "Functional description unavailable"
        )
    })

result = pd.DataFrame(representatives)

# ----------------------------------------------------------------
# ADD ORDER / ML LINK
# ----------------------------------------------------------------

result["Candidate_Rank"] = range(1, len(result) + 1)

# Reorder
result = result[
    [
        "Candidate_Rank",
        "BUSCO_ID",
        "Representative_Species",
        "Assembly_Accession",
        "Sequence",
        "Selected_Function",
        "GO_Terms",
        "KEGG_ID",
        "EC_Number",
        "Annotation_Status",
    ]
]

# ----------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------

result.to_csv(
    OUT_TABLE,
    index=False
)

# ----------------------------------------------------------------
# REPORT
# ----------------------------------------------------------------

with open(OUT_REPORT, "w", encoding="utf-8") as f:

    f.write("=" * 80 + "\n")
    f.write("CARBON BREADTH TOP-20 BUSCO FUNCTIONAL ANNOTATION INPUT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Total candidate BUSCOs : {len(TOP20)}\n")
    f.write(
        f"BUSCOs recovered       : "
        f"{result['BUSCO_ID'].isin(TOP20).sum()}\n"
    )

    f.write(
        f"Functional descriptions: "
        f"{(result['Selected_Function'].str.strip() != '').sum()}\n"
    )

    f.write("\n\nTOP-20 BUSCO FUNCTIONAL RECORDS\n")
    f.write("-" * 80 + "\n")

    for _, r in result.iterrows():

        f.write(
            f"\n{r['Candidate_Rank']}. "
            f"{r['BUSCO_ID']}\n"
        )

        f.write(
            f"Species: {r['Representative_Species']}\n"
        )

        f.write(
            f"Assembly: {r['Assembly_Accession']}\n"
        )

        f.write(
            f"Function: {r['Selected_Function']}\n"
        )

        f.write(
            "GO: [TO BE MAPPED]\n"
        )

        f.write(
            "KEGG: [TO BE MAPPED]\n"
        )

        f.write(
            "EC: [TO BE MAPPED]\n"
        )

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------

print("\n" + "=" * 80)
print("TOP-20 BUSCO ANNOTATION INPUT READY")
print("=" * 80)

print(f"\nTotal candidate BUSCOs : {len(TOP20)}")

print(
    "BUSCOs recovered       :",
    result["BUSCO_ID"].isin(TOP20).sum()
)

print(
    "Functional descriptions:",
    (result["Selected_Function"].str.strip() != "").sum()
)

print("\nBUSCO → FUNCTION")

for _, r in result.iterrows():

    print(
        f"{r['BUSCO_ID']:15s} → "
        f"{r['Selected_Function']}"
    )

print("\n" + "=" * 80)
print("OUTPUTS")
print("=" * 80)

print("\n20-BUSCO annotation table:")
print(OUT_TABLE)

print("\nReport:")
print(OUT_REPORT)

print("\nGO / KEGG / EC columns are intentionally blank.")
print("They will be populated by the actual annotation step.")
print("=" * 80)