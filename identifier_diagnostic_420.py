import pandas as pd

MAPPING_FILE = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\busco_gene_mapping_420\tables\Carbon_Breadth_BUSCO_gene_protein_mapping_420.csv"

df = pd.read_csv(MAPPING_FILE, low_memory=False)

print("=" * 80)
print("IDENTIFIER DIAGNOSTIC")
print("=" * 80)

for col in [
    "BUSCO_ID",
    "Sequence",
    "Gene_ID",
    "Gene_Name",
    "Locus_Tag",
    "Protein_ID",
    "Functional_Description"
]:

    if col not in df.columns:
        print(f"\n{col}: COLUMN NOT PRESENT")
        continue

    print(f"\n{'=' * 40}")
    print(col)

    values = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values != ""
    ]

    print("Non-empty:", len(values))
    print("Unique:", values.nunique())

    print("\nExamples:")

    for value in values.drop_duplicates().head(15):
        print(" ", repr(value))

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)