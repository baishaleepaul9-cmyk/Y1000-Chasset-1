from pathlib import Path
from Bio import Phylo

# ============================================================
# ASTRAL TREE ROOTING
# ============================================================

BASE = Path(
    r"C:\Y1000_chassis_project\results\stage4B_phylogeny"
)

ASTRAL_DIR = (
    BASE
    / "postbusco_phylogeny_v2"
    / "astral"
)

INPUT_TREE = (
    ASTRAL_DIR
    / "stage4B_ASTRAL_species_tree.nwk"
)

OUTPUT_TREE = (
    ASTRAL_DIR
    / "stage4B_ASTRAL_species_tree_midpoint_rooted.nwk"
)

print("=" * 80)
print("ASTRAL TREE — MIDPOINT ROOTING")
print("=" * 80)

print(f"\nInput:")
print(INPUT_TREE)

print(f"\nOutput:")
print(OUTPUT_TREE)

# ------------------------------------------------------------
# Load original ASTRAL tree
# ------------------------------------------------------------

tree = Phylo.read(INPUT_TREE, "newick")

print("\nOriginal tree:")
print(f"  Tips: {len(tree.get_terminals())}")
print(f"  Rooted: {tree.rooted}")

# ------------------------------------------------------------
# Preserve original
# ------------------------------------------------------------

original_taxa = {
    terminal.name
    for terminal in tree.get_terminals()
}

# ------------------------------------------------------------
# Midpoint rooting
# ------------------------------------------------------------

print("\nApplying midpoint rooting...")

tree.root_at_midpoint()

# Mark explicitly as rooted
tree.rooted = True

# ------------------------------------------------------------
# Validate
# ------------------------------------------------------------

rooted_taxa = {
    terminal.name
    for terminal in tree.get_terminals()
}

print("\nRooted tree:")
print(f"  Tips: {len(tree.get_terminals())}")
print(f"  Rooted: {tree.rooted}")

print("\nTaxon reconciliation:")

missing = original_taxa - rooted_taxa
unexpected = rooted_taxa - original_taxa

print(f"  Original taxa:  {len(original_taxa)}")
print(f"  Rooted taxa:    {len(rooted_taxa)}")
print(f"  Missing taxa:   {len(missing)}")
print(f"  Unexpected:     {len(unexpected)}")

if missing:
    print("\nERROR — taxa disappeared:")
    for x in sorted(missing):
        print(" ", x)

    raise SystemExit(1)

if unexpected:
    print("\nERROR — unexpected taxa appeared:")
    for x in sorted(unexpected):
        print(" ", x)

    raise SystemExit(1)

# ------------------------------------------------------------
# Save rooted tree
# ------------------------------------------------------------

Phylo.write(
    tree,
    OUTPUT_TREE,
    "newick"
)

print("\n" + "=" * 80)
print("ROOTING COMPLETE")
print("=" * 80)

print(f"\n✓ Original unrooted tree preserved:")
print(f"  {INPUT_TREE}")

print(f"\n✓ Midpoint-rooted tree created:")
print(f"  {OUTPUT_TREE}")

print("\n✓ Same number of taxa")
print("✓ No taxa lost")
print("✓ No unexpected taxa added")
print("✓ Original ASTRAL tree was not modified")