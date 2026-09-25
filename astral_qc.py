from pathlib import Path
import pandas as pd
import re

# ============================================================
# ASTRAL SPECIES TREE QC
# Normal Windows Python version
# ============================================================

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE = Path(
    r"C:\Y1000_chassis_project\results\stage4B_phylogeny"
    r"\postbusco_phylogeny_v2"
)

ASTRAL_DIR = BASE / "astral"

TREE_FILE = ASTRAL_DIR / "stage4B_ASTRAL_species_tree.nwk"

MANIFEST_FILE = (
    BASE.parent / "stage4B_genome_fasta_manifest.csv"
)

OUT_FILE = (
    ASTRAL_DIR / "astral_species_tree_qc.csv"
)


# ============================================================
# 2. PARETO CANDIDATES
#    From your Stage 4A.1 reconciliation
# ============================================================

pareto = {
    "GCA_003705225.1": "Ambrosiozyma vanderkliftii",
    "GCA_003123585.1": "Barnettozyma californica",
    "GCA_003709245.3": "Cyberlindnera saturnus",
    "GCA_030558845.1": "Kodamaea laetipori",
    "GCA_030563145.1":
        "Schwanniomyces polymorphus var. africanus",
    "GCA_030463025.1":
        "Schwanniomyces polymorphus var. polymorphus",
    "GCA_030583345.1":
        "Schwanniomyces pseudopolymorphus",
    "GCA_030583405.1":
        "Sugiyamaella americana",
    "GCA_030579815.1":
        "Sugiyamaella smithiae",
    "GCA_030558095.1":
        "Teunomyces funiuensis",
    "GCA_030564625.1":
        "Zygoascus hellenicus",
}


# ============================================================
# 3. CHECK FILES
# ============================================================

print("=" * 80)
print("ASTRAL SPECIES TREE QC")
print("=" * 80)

print("\nChecking files...")

if not TREE_FILE.exists():
    raise FileNotFoundError(
        f"\nASTRAL tree was not found:\n{TREE_FILE}\n\n"
        "Check that C:\\Y1000_chassis_project exists and "
        "that the ASTRAL output is in the expected folder."
    )

if TREE_FILE.stat().st_size == 0:
    raise RuntimeError(
        f"\nASTRAL tree exists but is EMPTY:\n{TREE_FILE}"
    )

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(
        f"\nManifest was not found:\n{MANIFEST_FILE}"
    )

print("✓ ASTRAL tree found")
print(
    f"  Size: {TREE_FILE.stat().st_size:,} bytes"
)

print("✓ Genome manifest found")


# ============================================================
# 4. READ MANIFEST
# ============================================================

manifest = pd.read_csv(MANIFEST_FILE)

expected_accessions = set(
    manifest["Assembly_Accession"]
    .astype(str)
    .str.strip()
)

expected_species = dict(
    zip(
        manifest["Assembly_Accession"]
        .astype(str)
        .str.strip(),

        manifest["Species"]
        .astype(str)
        .str.strip()
    )
)

print("\n" + "=" * 80)
print("EXPECTED TAXA")
print("=" * 80)

print(
    f"Representative taxa in manifest: "
    f"{len(expected_accessions)}"
)


# ============================================================
# 5. READ ASTRAL TREE
# ============================================================

text = TREE_FILE.read_text(
    encoding="utf-8"
).strip()

if not text.endswith(";"):
    print(
        "\nWARNING: Tree does not appear to end with ';'"
    )

# Extract leaf labels.
#
# Tree labels in your ASTRAL output look like:
#
# Species_name__GCA_030558095.1
#
# This pattern captures labels occurring after
# '(' or ',' and before ':', ',' or ')'.

labels = re.findall(
    r'(?:(?<=\()|(?<=,))([^():,]+)(?=:|,|\))',
    text
)

labels = [
    x.strip()
    for x in labels
    if x.strip()
]

tree_labels = set(labels)

print("\n" + "=" * 80)
print("ASTRAL TREE")
print("=" * 80)

print(
    f"Leaf labels detected: {len(labels)}"
)

print(
    f"Unique taxa detected: {len(tree_labels)}"
)


# ============================================================
# 6. EXTRACT ASSEMBLY ACCESSIONS
# ============================================================

tree_accessions = set()

for label in tree_labels:

    match = re.search(
        r'(GCA_\d+\.\d+|GCF_\d+\.\d+)',
        label
    )

    if match:
        tree_accessions.add(
            match.group(1)
        )


print(
    f"Assembly accessions detected: "
    f"{len(tree_accessions)}"
)


# ============================================================
# 7. MANIFEST ↔ ASTRAL RECONCILIATION
# ============================================================

missing = sorted(
    expected_accessions - tree_accessions
)

unexpected = sorted(
    tree_accessions - expected_accessions
)

present = sorted(
    expected_accessions & tree_accessions
)

print("\n" + "=" * 80)
print("MANIFEST ↔ ASTRAL RECONCILIATION")
print("=" * 80)

print(
    f"Expected taxa:      {len(expected_accessions)}"
)

print(
    f"Present in ASTRAL:  {len(present)}"
)

print(
    f"Missing:            {len(missing)}"
)

print(
    f"Unexpected:         {len(unexpected)}"
)


# ------------------------------------------------------------
# Missing taxa
# ------------------------------------------------------------

if missing:

    print("\nMISSING FROM ASTRAL:")

    for acc in missing:

        print(
            f"  {acc}\t"
            f"{expected_species.get(acc, 'UNKNOWN')}"
        )

else:

    print(
        "\n✓ All expected representative taxa "
        "are present in ASTRAL."
    )


# ------------------------------------------------------------
# Unexpected taxa
# ------------------------------------------------------------

if unexpected:

    print("\nUNEXPECTED TAXA:")

    for acc in unexpected:
        print(f"  {acc}")

else:

    print(
        "\n✓ No unexpected assembly accessions detected."
    )


# ============================================================
# 8. PARETO CANDIDATE CHECK
# ============================================================

print("\n" + "=" * 80)
print("PARETO CANDIDATE CHECK")
print("=" * 80)

pareto_rows = []

for order, (acc, name) in enumerate(
    pareto.items(),
    start=1
):

    in_manifest = (
        acc in expected_accessions
    )

    in_tree = (
        acc in tree_accessions
    )

    if in_tree and in_manifest:

        status = "PRESENT_IN_ASTRAL"

    elif in_manifest and not in_tree:

        status = "MISSING_FROM_ASTRAL"

    elif in_tree and not in_manifest:

        status = "UNEXPECTED_IN_ASTRAL"

    else:

        status = (
            "NOT_IN_MANIFEST_OR_ASTRAL"
        )

    pareto_rows.append({
        "Pareto_Order": order,
        "Assembly_Accession": acc,
        "Pareto_Name": name,
        "In_436_Manifest": in_manifest,
        "In_ASTRAL": in_tree,
        "Status": status
    })

    symbol = (
        "✓"
        if status == "PRESENT_IN_ASTRAL"
        else "!"
    )

    print(
        f"{symbol} "
        f"{acc:18s}  "
        f"{name:48s}  "
        f"{status}"
    )


pareto_df = pd.DataFrame(
    pareto_rows
)


# ============================================================
# 9. PARETO SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PARETO SUMMARY")
print("=" * 80)

print(
    f"Pareto candidates:       "
    f"{len(pareto_df)}"
)

print(
    f"Present in 436 manifest: "
    f"{pareto_df['In_436_Manifest'].sum()}"
)

print(
    f"Present in ASTRAL:       "
    f"{pareto_df['In_ASTRAL'].sum()}"
)

print(
    f"Missing from ASTRAL:     "
    f"{(~pareto_df['In_ASTRAL']).sum()}"
)


# ============================================================
# 10. SAVE QC TABLE
# ============================================================

OUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

pareto_df.to_csv(
    OUT_FILE,
    index=False
)


# ============================================================
# 11. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("QC COMPLETE")
print("=" * 80)

print(
    f"\nQC table written to:\n"
    f"{OUT_FILE}"
)

print("\nFINAL CHECKS:")

if len(missing) == 0:

    print(
        "✓ All representative taxa are present "
        "in the ASTRAL tree."
    )

else:

    print(
        f"! {len(missing)} representative taxa "
        "are missing from the ASTRAL tree."
    )


if len(unexpected) == 0:

    print(
        "✓ No unexpected taxa were detected."
    )

else:

    print(
        f"! {len(unexpected)} unexpected taxa "
        "were detected."
    )


if pareto_df["In_ASTRAL"].all():

    print(
        "✓ All 11 Pareto candidates are represented "
        "in the ASTRAL tree."
    )

else:

    print(
        "! At least one Pareto candidate is missing "
        "from the ASTRAL tree."
    )


print("\nQC finished.")