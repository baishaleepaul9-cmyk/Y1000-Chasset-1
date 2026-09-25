from pathlib import Path
import pandas as pd
import numpy as np
from Bio import Phylo
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

# ============================================================
# Y1000+ PHYLOGENY ↔ ML DATASET QC
# ============================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

ML_FILE = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "feature_qc"
    / "y1000_419_variable_busco_ml_matrix.csv"
)

TREE_FILE = (
    PROJECT
    / "results"
    / "stage4B_phylogeny"
    / "postbusco_phylogeny_v2"
    / "astral"
    / "stage4B_ASTRAL_species_tree.nwk"
)

ROOTED_TREE_FILE = (
    PROJECT
    / "results"
    / "stage4B_phylogeny"
    / "postbusco_phylogeny_v2"
    / "astral"
    / "stage4B_ASTRAL_species_tree_midpoint_rooted.nwk"
)

OUTPUT_DIR = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_qc"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_TAXA = 419

# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("Y1000+ PHYLOGENY ↔ ML DATASET QC")
print("=" * 80)

print()
print("ML matrix:")
print(ML_FILE)

print()
print("ASTRAL tree:")
print(TREE_FILE)

# ============================================================
# CHECK INPUTS
# ============================================================

if not ML_FILE.exists():
    raise FileNotFoundError(
        f"ML matrix not found:\n{ML_FILE}"
    )

if not TREE_FILE.exists():
    raise FileNotFoundError(
        f"ASTRAL tree not found:\n{TREE_FILE}"
    )

# ============================================================
# LOAD ML DATASET
# ============================================================

print()
print("=" * 80)
print("LOADING ML DATASET")
print("=" * 80)

df = pd.read_csv(ML_FILE)

print()
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

if len(df) != EXPECTED_TAXA:
    raise RuntimeError(
        f"Expected {EXPECTED_TAXA} taxa but found {len(df)}"
    )

# ============================================================
# LOAD TREE
# ============================================================

print()
print("=" * 80)
print("LOADING ASTRAL TREE")
print("=" * 80)

tree = Phylo.read(str(TREE_FILE), "newick")

tree_taxa = [
    str(terminal.name).strip()
    for terminal in tree.get_terminals()
    if terminal.name
]

tree_taxa = list(dict.fromkeys(tree_taxa))

print()
print(f"ASTRAL tree taxa: {len(tree_taxa)}")
print(f"ML taxa:          {len(df)}")

if len(tree_taxa) != 436:
    print(
        f"WARNING: ASTRAL tree contains {len(tree_taxa)} taxa "
        f"instead of the expected 436."
    )

# ============================================================
# TAXON NAME NORMALIZATION
# ============================================================

def normalize_taxon(name):
    """
    Normalize names conservatively.

    The tree labels are expected to contain the assembly
    accession, while the ML dataset contains Species and
    Assembly_Accession separately.
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
    )


tree_normalized = {
    normalize_taxon(x): x
    for x in tree_taxa
}

ml_accessions = (
    df["Assembly_Accession"]
    .astype(str)
    .str.strip()
)

ml_species = (
    df["Species"]
    .astype(str)
    .str.strip()
)

# ============================================================
# ACCESSION-BASED TREE MATCHING
# ============================================================

print()
print("=" * 80)
print("MATCHING ML TAXA TO ASTRAL TREE")
print("=" * 80)

matched = []
missing = []

for species, accession in zip(ml_species, ml_accessions):

    accession_key = normalize_taxon(accession)

    candidates = [
        original
        for normalized, original in tree_normalized.items()
        if accession_key in normalized
    ]

    if candidates:
        matched.append({
            "Species": species,
            "Assembly_Accession": accession,
            "ASTRAL_Label": candidates[0],
            "Matched": True
        })
    else:
        missing.append({
            "Species": species,
            "Assembly_Accession": accession,
            "ASTRAL_Label": "",
            "Matched": False
        })

matching_df = pd.DataFrame(
    matched + missing
)

n_matched = len(matched)
n_missing = len(missing)

print()
print(f"ML taxa:               {len(df)}")
print(f"Matched to ASTRAL:     {n_matched}")
print(f"Missing from ASTRAL:   {n_missing}")

# ============================================================
# SAVE MATCHING TABLE
# ============================================================

matching_output = (
    OUTPUT_DIR
    / "ml_taxa_astral_reconciliation.csv"
)

matching_df.to_csv(
    matching_output,
    index=False
)

if n_missing > 0:

    print()
    print("Missing taxa:")

    for row in missing:
        print(
            f"  {row['Assembly_Accession']}  "
            f"{row['Species']}"
        )

    raise RuntimeError(
        "Some ML taxa could not be mapped to the ASTRAL tree."
    )

print()
print("✓ All 419 ML taxa are represented in ASTRAL.")

# ============================================================
# PRUNE TREE TO ML TAXA
# ============================================================

print()
print("=" * 80)
print("CREATING 419-TAXON ML TREE")
print("=" * 80)

ml_tree = tree.clone()

ml_accession_set = {
    normalize_taxon(x)
    for x in ml_accessions
}

tips_to_remove = []

for terminal in ml_tree.get_terminals():

    label = normalize_taxon(terminal.name)

    accession_match = any(
        accession in label
        for accession in ml_accession_set
    )

    if not accession_match:
        tips_to_remove.append(terminal)

print()
print(f"Original ASTRAL tips: {len(tree_taxa)}")
print(f"Removing non-ML taxa: {len(tips_to_remove)}")

for terminal in tips_to_remove:
    ml_tree.prune(terminal)

remaining_ml_tree_taxa = [
    terminal.name
    for terminal in ml_tree.get_terminals()
]

print()
print(f"Final ML tree tips: {len(remaining_ml_tree_taxa)}")

if len(remaining_ml_tree_taxa) != EXPECTED_TAXA:
    raise RuntimeError(
        f"Expected a 419-taxon ML tree but obtained "
        f"{len(remaining_ml_tree_taxa)} taxa."
    )

ml_tree_output = (
    OUTPUT_DIR
    / "y1000_419_astral_ml_tree.nwk"
)

Phylo.write(
    ml_tree,
    str(ml_tree_output),
    "newick"
)

print()
print("✓ 419-taxon ML tree created.")

# ============================================================
# TREE ROOTING INFORMATION
# ============================================================

print()
print("=" * 80)
print("TREE ROOTING INFORMATION")
print("=" * 80)

print()
print(f"Original ASTRAL rooted: {tree.rooted}")
print(f"419-taxon tree rooted:  {ml_tree.rooted}")

if ROOTED_TREE_FILE.exists():

    rooted_tree = Phylo.read(
        str(ROOTED_TREE_FILE),
        "newick"
    )

    rooted_tips = [
        x.name
        for x in rooted_tree.get_terminals()
        if x.name
    ]

    print()
    print("Midpoint-rooted tree available:")
    print(ROOTED_TREE_FILE)

    print(
        f"Midpoint-rooted tree tips: {len(rooted_tips)}"
    )

else:

    print()
    print("No midpoint-rooted tree file found.")

# ============================================================
# TREE DISTANCE MATRIX
# ============================================================

print()
print("=" * 80)
print("CALCULATING PHYLOGENETIC DISTANCES")
print("=" * 80)

# Use branch-length distances when available.
# If branch lengths are missing, this will identify the issue.

terminals = ml_tree.get_terminals()

distance_records = []

for i, a in enumerate(terminals):

    if i % 50 == 0:
        print(
            f"  Processing {i + 1}/{len(terminals)}"
        )

    for j in range(i + 1, len(terminals)):

        b = terminals[j]

        distance = ml_tree.distance(a, b)

        distance_records.append({
            "Taxon_1": a.name,
            "Taxon_2": b.name,
            "Distance": distance
        })

distance_df = pd.DataFrame(distance_records)

distance_output = (
    OUTPUT_DIR
    / "y1000_419_phylogenetic_distances.csv"
)

distance_df.to_csv(
    distance_output,
    index=False
)

print()
print(
    f"Pairwise distances calculated: "
    f"{len(distance_df):,}"
)

# ============================================================
# DISTANCE SUMMARY
# ============================================================

distance_summary = pd.DataFrame([{
    "Taxa": len(terminals),
    "Pairwise_Comparisons": len(distance_df),
    "Minimum_Distance": distance_df["Distance"].min(),
    "Median_Distance": distance_df["Distance"].median(),
    "Mean_Distance": distance_df["Distance"].mean(),
    "Maximum_Distance": distance_df["Distance"].max()
}])

distance_summary_output = (
    OUTPUT_DIR
    / "phylogenetic_distance_summary.csv"
)

distance_summary.to_csv(
    distance_summary_output,
    index=False
)

print()
print(distance_summary.to_string(index=False))

# ============================================================
# ML TREE VISUALIZATION
# ============================================================

print()
print("=" * 80)
print("CREATING ML TREE VISUALIZATION")
print("=" * 80)

# Large figure because 419 taxa cannot be interpreted well
# on a tiny figure.

fig_height = max(
    16,
    len(terminals) * 0.055
)

fig = plt.figure(
    figsize=(20, fig_height)
)

ax = fig.add_subplot(1, 1, 1)

Phylo.draw(
    ml_tree,
    axes=ax,
    do_show=False,
    label_func=lambda x: None,
    show_confidence=False
)

ax.set_title(
    "Y1000+ 419-Taxon ASTRAL Subtree\n"
    "Matched Genome–Phenotype ML Dataset",
    fontsize=18,
    pad=20
)

ax.set_xlabel(
    "Phylogenetic distance",
    fontsize=12
)

ax.set_ylabel(
    "Taxa",
    fontsize=12
)

plt.tight_layout()

tree_png = (
    OUTPUT_DIR
    / "y1000_419_astral_ml_tree.png"
)

plt.savefig(
    tree_png,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Tree visualization:")
print(tree_png)

# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([{
    "ASTRAL_Taxa": len(tree_taxa),
    "ML_Taxa": len(df),
    "ML_Taxa_Matched": n_matched,
    "ML_Taxa_Missing": n_missing,
    "ML_Tree_Taxa": len(remaining_ml_tree_taxa),
    "Original_Tree_Rooted": tree.rooted,
    "Midpoint_Rooted_Tree_Available": ROOTED_TREE_FILE.exists()
}])

summary_output = (
    OUTPUT_DIR
    / "phylogeny_ml_qc_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)

print()
print("=" * 80)
print("PHYLOGENY ↔ ML QC COMPLETE")
print("=" * 80)

print()
print(f"ASTRAL taxa:                 {len(tree_taxa)}")
print(f"ML taxa:                     {len(df)}")
print(f"ML taxa matched:             {n_matched}")
print(f"ML taxa missing:             {n_missing}")
print(f"Final ML tree taxa:          {len(remaining_ml_tree_taxa)}")

print()
print("419-TAXON TREE:")
print(ml_tree_output)

print()
print("TREE IMAGE:")
print(tree_png)

print()
print("RECONCILIATION:")
print(matching_output)

print()
print("DISTANCE SUMMARY:")
print(distance_summary_output)

print()
print("✓ All 419 ML taxa mapped to the ASTRAL phylogeny.")
print("✓ 419-taxon phylogenetic subtree generated.")
print("✓ Original ASTRAL tree was not modified.")