from pathlib import Path
import pandas as pd
import re
import sys

# ============================================================
# STAGE 5 — PHYLOGENY-AWARE ML DATASET PREPARATION
# ============================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

RESULTS = PROJECT / "results"

ASTRAL_TREE = (
    RESULTS
    / "stage4B_phylogeny"
    / "postbusco_phylogeny_v2"
    / "astral"
    / "stage4B_ASTRAL_species_tree.nwk"
)

GENOME_MANIFEST = (
    RESULTS
    / "stage4B_phylogeny"
    / "stage4B_genome_fasta_manifest.csv"
)

OUTPUT_DIR = (
    RESULTS
    / "stage5_phylogeny_ml"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMPORTANT:
# SET THESE AFTER LOOKING AT THE DISCOVERED FILES
# ============================================================

PHENOTYPE_FILE = None
GENOMIC_FEATURE_FILE = None


# ============================================================
# HELPERS
# ============================================================

def norm(x):
    """Normalize strings for matching."""
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def find_accession_column(df):
    """Find Assembly Accession column."""
    candidates = [
        "Assembly_Accession",
        "Assembly Accession",
        "assembly_accession",
        "assembly accession",
        "accession",
        "Assembly"
    ]

    for c in candidates:
        if c in df.columns:
            return c

    # More flexible search
    for c in df.columns:
        cl = str(c).lower()
        if "assembly" in cl and "accession" in cl:
            return c

    for c in df.columns:
        cl = str(c).lower()
        if cl == "accession":
            return c

    return None


def extract_tree_accessions(tree_text):
    """
    Extract GCA/GCF accessions from ASTRAL leaf labels.

    Example:
    Sugiyamaella_americana__GCA_030583405.1
    """

    accessions = re.findall(
        r"(GC[AF]_\d+\.\d+)",
        tree_text
    )

    return list(dict.fromkeys(accessions))


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("STAGE 5 — PHYLOGENY-AWARE ML DATASET PREPARATION")
print("=" * 80)


# ============================================================
# 1. CHECK ASTRAL TREE
# ============================================================

print("\n" + "=" * 80)
print("1. ASTRAL TREE")
print("=" * 80)

if not ASTRAL_TREE.exists():
    print("ERROR: ASTRAL tree not found:")
    print(ASTRAL_TREE)
    sys.exit(1)

tree_text = ASTRAL_TREE.read_text(
    encoding="utf-8",
    errors="ignore"
).strip()

tree_accessions = extract_tree_accessions(tree_text)

print(f"ASTRAL tree:")
print(f"  {ASTRAL_TREE}")

print(f"\nASTRAL taxa detected: {len(tree_accessions)}")


if len(tree_accessions) != 436:
    print(
        f"WARNING: Expected 436 taxa, "
        f"but detected {len(tree_accessions)}."
    )
else:
    print("✓ 436 taxa detected.")


# ============================================================
# 2. LOAD REPRESENTATIVE GENOME MANIFEST
# ============================================================

print("\n" + "=" * 80)
print("2. REPRESENTATIVE GENOME MANIFEST")
print("=" * 80)

if not GENOME_MANIFEST.exists():
    print("ERROR: Genome manifest not found:")
    print(GENOME_MANIFEST)
    sys.exit(1)

manifest = pd.read_csv(GENOME_MANIFEST)

print(f"Manifest:")
print(f"  {GENOME_MANIFEST}")

print(f"\nRows: {len(manifest)}")
print("Columns:")
for c in manifest.columns:
    print(f"  {c}")

manifest_acc_col = find_accession_column(manifest)

if manifest_acc_col is None:
    print("\nERROR: Could not identify Assembly Accession column.")
    sys.exit(1)

print(f"\nAccession column: {manifest_acc_col}")


manifest_accessions = set(
    manifest[manifest_acc_col]
    .dropna()
    .astype(str)
    .str.strip()
)

print(f"Manifest accessions: {len(manifest_accessions)}")


# ============================================================
# 3. ASTRAL ↔ MANIFEST CHECK
# ============================================================

print("\n" + "=" * 80)
print("3. ASTRAL ↔ REPRESENTATIVE GENOME RECONCILIATION")
print("=" * 80)

tree_set = set(tree_accessions)

missing_from_tree = sorted(
    manifest_accessions - tree_set
)

unexpected_in_tree = sorted(
    tree_set - manifest_accessions
)

print(f"Manifest taxa:       {len(manifest_accessions)}")
print(f"ASTRAL taxa:         {len(tree_set)}")
print(f"Missing from tree:   {len(missing_from_tree)}")
print(f"Unexpected in tree:  {len(unexpected_in_tree)}")


if missing_from_tree:
    print("\nMissing from ASTRAL:")
    for x in missing_from_tree:
        print(" ", x)

if unexpected_in_tree:
    print("\nUnexpected in ASTRAL:")
    for x in unexpected_in_tree:
        print(" ", x)


if not missing_from_tree and not unexpected_in_tree:
    print("\n✓ ASTRAL and representative genome manifest match.")


# ============================================================
# 4. DISCOVER CSV FILES
# ============================================================

print("\n" + "=" * 80)
print("4. DISCOVERING POSSIBLE INPUT TABLES")
print("=" * 80)

csv_files = list(RESULTS.rglob("*.csv"))

print(f"\nCSV files found: {len(csv_files)}")


phenotype_keywords = [
    "carbon",
    "nitrogen",
    "growth",
    "phenotype",
    "trait",
    "utilization",
    "stress",
    "breadth",
    "pareto"
]

genomic_keywords = [
    "ko",
    "pathway",
    "gene",
    "family",
    "transporter",
    "cazyme",
    "ortholog",
    "busco",
    "genomic",
    "genome"
]


phenotype_candidates = []
genomic_candidates = []


for f in csv_files:

    name = f.name.lower()

    try:
        df_tmp = pd.read_csv(
            f,
            nrows=5
        )

        columns = " ".join(
            str(c).lower()
            for c in df_tmp.columns
        )

        combined = name + " " + columns

        if any(k in combined for k in phenotype_keywords):
            phenotype_candidates.append(f)

        if any(k in combined for k in genomic_keywords):
            genomic_candidates.append(f)

    except Exception:
        pass


print("\n--- POSSIBLE PHENOTYPE TABLES ---")

for f in phenotype_candidates:
    print(" ", f)


print("\n--- POSSIBLE GENOMIC FEATURE TABLES ---")

for f in genomic_candidates:
    print(" ", f)


# ============================================================
# 5. STOP HERE IF INPUT FILES NOT SELECTED
# ============================================================

if PHENOTYPE_FILE is None or GENOMIC_FEATURE_FILE is None:

    print("\n" + "=" * 80)
    print("INPUT FILES NEED TO BE SET")
    print("=" * 80)

    print("""
Open this script and set:

PHENOTYPE_FILE = Path(r"...")

GENOMIC_FEATURE_FILE = Path(r"...")

Use the FINAL phenotype table and FINAL genomic feature table,
not intermediate QC files.

Then run the script again.
""")

    sys.exit(0)


# Convert to Path
PHENOTYPE_FILE = Path(PHENOTYPE_FILE)
GENOMIC_FEATURE_FILE = Path(GENOMIC_FEATURE_FILE)


# ============================================================
# 6. LOAD PHENOTYPE TABLE
# ============================================================

print("\n" + "=" * 80)
print("5. PHENOTYPE DATA")
print("=" * 80)

if not PHENOTYPE_FILE.exists():
    print("ERROR: Phenotype file not found:")
    print(PHENOTYPE_FILE)
    sys.exit(1)

phenotype = pd.read_csv(PHENOTYPE_FILE)

print(f"File: {PHENOTYPE_FILE}")
print(f"Rows: {len(phenotype)}")
print(f"Columns: {len(phenotype.columns)}")

print("\nColumns:")
for c in phenotype.columns:
    print(f"  {c}")


pheno_acc_col = find_accession_column(phenotype)

if pheno_acc_col is None:
    print(
        "\nERROR: Phenotype table does not contain "
        "an Assembly Accession column."
    )
    sys.exit(1)

print(f"\nPhenotype accession column: {pheno_acc_col}")


# ============================================================
# 7. LOAD GENOMIC FEATURE TABLE
# ============================================================

print("\n" + "=" * 80)
print("6. GENOMIC FEATURES")
print("=" * 80)

if not GENOMIC_FEATURE_FILE.exists():
    print("ERROR: Genomic feature file not found:")
    print(GENOMIC_FEATURE_FILE)
    sys.exit(1)

genomic = pd.read_csv(GENOMIC_FEATURE_FILE)

print(f"File: {GENOMIC_FEATURE_FILE}")
print(f"Rows: {len(genomic)}")
print(f"Columns: {len(genomic.columns)}")

print("\nColumns:")
for c in genomic.columns:
    print(f"  {c}")


genomic_acc_col = find_accession_column(genomic)

if genomic_acc_col is None:
    print(
        "\nERROR: Genomic table does not contain "
        "an Assembly Accession column."
    )
    sys.exit(1)

print(f"\nGenomic accession column: {genomic_acc_col}")


# ============================================================
# 8. NORMALIZE ACCESSIONS
# ============================================================

phenotype["_ACCESSION"] = (
    phenotype[pheno_acc_col]
    .astype(str)
    .str.strip()
)

genomic["_ACCESSION"] = (
    genomic[genomic_acc_col]
    .astype(str)
    .str.strip()
)


# ============================================================
# 9. CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("7. DUPLICATE ACCESSION CHECK")
print("=" * 80)

pheno_dup = (
    phenotype["_ACCESSION"]
    .value_counts()
)

pheno_dup = pheno_dup[
    pheno_dup > 1
]

genomic_dup = (
    genomic["_ACCESSION"]
    .value_counts()
)

genomic_dup = genomic_dup[
    genomic_dup > 1
]


print(
    f"Phenotype duplicate accessions: "
    f"{len(pheno_dup)}"
)

print(
    f"Genomic duplicate accessions: "
    f"{len(genomic_dup)}"
)


if len(pheno_dup):
    print("\nPhenotype duplicates:")
    print(pheno_dup)

if len(genomic_dup):
    print("\nGenomic duplicates:")
    print(genomic_dup)


# ============================================================
# 10. RESTRICT TO ASTRAL TAXA
# ============================================================

print("\n" + "=" * 80)
print("8. RESTRICTING DATA TO ASTRAL TAXA")
print("=" * 80)

phenotype_astral = phenotype[
    phenotype["_ACCESSION"].isin(tree_set)
].copy()

genomic_astral = genomic[
    genomic["_ACCESSION"].isin(tree_set)
].copy()


print(
    f"Phenotype rows matching ASTRAL: "
    f"{len(phenotype_astral)}"
)

print(
    f"Genomic rows matching ASTRAL: "
    f"{len(genomic_astral)}"
)


# ============================================================
# 11. CHECK MISSING TAXA
# ============================================================

pheno_accessions = set(
    phenotype_astral["_ACCESSION"]
)

genomic_accessions = set(
    genomic_astral["_ACCESSION"]
)


missing_pheno = sorted(
    tree_set - pheno_accessions
)

missing_genomic = sorted(
    tree_set - genomic_accessions
)


print("\nMissing phenotype taxa:")
print(f"  {len(missing_pheno)}")

for x in missing_pheno[:50]:
    print(" ", x)


print("\nMissing genomic taxa:")
print(f"  {len(missing_genomic)}")

for x in missing_genomic[:50]:
    print(" ", x)


# ============================================================
# 12. INTEGRATE
# ============================================================

print("\n" + "=" * 80)
print("9. INTEGRATING PHENOTYPE + GENOMIC DATA")
print("=" * 80)


# Remove original accession columns to avoid duplicate names
phenotype_merge = phenotype_astral.drop(
    columns=[pheno_acc_col],
    errors="ignore"
)

genomic_merge = genomic_astral.drop(
    columns=[genomic_acc_col],
    errors="ignore"
)


integrated = pd.merge(
    phenotype_merge,
    genomic_merge,
    on="_ACCESSION",
    how="inner",
    suffixes=("_phenotype", "_genomic")
)


print(f"Integrated rows:    {len(integrated)}")
print(f"Integrated columns: {len(integrated.columns)}")


# ============================================================
# 13. FINAL TREE-COMPATIBLE DATASET
# ============================================================

final_accessions = set(
    integrated["_ACCESSION"]
)

missing_final = sorted(
    tree_set - final_accessions
)

print("\n" + "=" * 80)
print("10. FINAL TREE ↔ ML DATASET CHECK")
print("=" * 80)

print(f"ASTRAL taxa:             {len(tree_set)}")
print(f"Integrated taxa:         {len(final_accessions)}")
print(f"Missing from ML table:   {len(missing_final)}")


if missing_final:
    print("\nMissing taxa:")
    for x in missing_final[:100]:
        print(" ", x)

else:
    print(
        "\n✓ All 436 ASTRAL taxa have "
        "phenotype + genomic records."
    )


# ============================================================
# 14. SAVE OUTPUTS
# ============================================================

integrated_file = (
    OUTPUT_DIR /
    "stage5_phylogeny_ml_integrated.csv"
)

missing_file = (
    OUTPUT_DIR /
    "stage5_missing_taxa.csv"
)

tree_accession_file = (
    OUTPUT_DIR /
    "stage5_astral_taxa.csv"
)


integrated.to_csv(
    integrated_file,
    index=False
)

pd.DataFrame({
    "Assembly_Accession": missing_final
}).to_csv(
    missing_file,
    index=False
)

pd.DataFrame({
    "Assembly_Accession": tree_accessions
}).to_csv(
    tree_accession_file,
    index=False
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("STAGE 5 DATASET PREPARATION COMPLETE")
print("=" * 80)

print(f"""
ASTRAL taxa:                 {len(tree_set)}
Integrated taxa:             {len(final_accessions)}
Missing phenotype taxa:      {len(missing_pheno)}
Missing genomic taxa:        {len(missing_genomic)}
Missing final ML taxa:       {len(missing_final)}

Integrated dataset:
{integrated_file}

Missing taxa report:
{missing_file}

ASTRAL accession list:
{tree_accession_file}
""")

if len(missing_final) == 0:
    print("✓ READY FOR PHYLOGENY-AWARE ML")
else:
    print(
        "⚠ DATASET REQUIRES TAXON/FEATURE QC "
        "BEFORE PHYLOGENY-AWARE ML"
    )

print("\nStage 5 finished.")