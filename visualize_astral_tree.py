from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from Bio import Phylo


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"C:\Y1000_chassis_project\results\stage4B_phylogeny"
    r"\postbusco_phylogeny_v2"
)

TREE_FILE = (
    BASE
    / "astral"
    / "stage4B_ASTRAL_species_tree.nwk"
)

OUTPUT = (
    BASE
    / "astral"
    / "astral_pareto_tree_circular.png"
)


# ============================================================
# PARETO REPRESENTATIVES
# ============================================================

PARETO = {
    "GCA_003705225.1": "Ambrosiozyma vanderkliftii",
    "GCA_003123585.1": "Barnettozyma californica",
    "GCA_003709245.3": "Cyberlindnera saturnus",
    "GCA_030558845.1": "Kodamaea laetipori",
    "GCA_030463025.1": "Schwanniomyces polymorphus",
    "GCA_030583345.1": "Schwanniomyces pseudopolymorphus",
    "GCA_030583405.1": "Sugiyamaella americana",
    "GCA_030579815.1": "Sugiyamaella smithiae",
    "GCA_030558095.1": "Teunomyces funiuensis",
    "GCA_030564625.1": "Zygoascus hellenicus",
}


# ============================================================
# LOAD TREE
# ============================================================

print("=" * 80)
print("CIRCULAR ASTRAL TREE VISUALIZATION")
print("=" * 80)

if not TREE_FILE.exists():
    raise FileNotFoundError(
        f"\nASTRAL tree not found:\n{TREE_FILE}"
    )

tree = Phylo.read(str(TREE_FILE), "newick")

terminals = tree.get_terminals()

print(f"\nTree loaded successfully.")
print(f"Total taxa: {len(terminals)}")


# ============================================================
# FIND PARETO TERMINALS
# ============================================================

pareto_nodes = {}

for terminal in terminals:

    label = terminal.name or ""

    for accession, species in PARETO.items():

        if accession in label:
            pareto_nodes[accession] = terminal


print(f"Pareto taxa detected: {len(pareto_nodes)}/10")

for accession, node in pareto_nodes.items():
    print(
        f"  ✓ {accession:<18} "
        f"{PARETO[accession]}"
    )


# ============================================================
# CALCULATE DISTANCE FROM ROOT
# ============================================================

def get_depths(clade, current_depth=0.0, depths=None):

    if depths is None:
        depths = {}

    branch_length = clade.branch_length or 0.0

    depth = current_depth + branch_length

    depths[clade] = depth

    for child in clade.clades:
        get_depths(child, depth, depths)

    return depths


depths = get_depths(tree.root)


# ============================================================
# ASSIGN ANGLES TO TERMINAL TAXA
# ============================================================

# Keep the natural order from the ASTRAL tree.

leaf_angles = {}

n_leaves = len(terminals)

for i, terminal in enumerate(terminals):

    # Start at top and move clockwise
    angle = math.pi / 2 - (2 * math.pi * i / n_leaves)

    leaf_angles[terminal] = angle


# ============================================================
# INTERNAL NODE ANGLES
# ============================================================

def calculate_angle(clade):

    if clade in leaf_angles:
        return leaf_angles[clade]

    child_angles = [
        calculate_angle(child)
        for child in clade.clades
    ]

    # Circular mean
    x = sum(math.cos(a) for a in child_angles)
    y = sum(math.sin(a) for a in child_angles)

    angle = math.atan2(y, x)

    leaf_angles[clade] = angle

    return angle


calculate_angle(tree.root)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(18, 18),
    dpi=200
)

ax = fig.add_subplot(
    111,
    projection="polar"
)

# Remove normal polar decorations
ax.set_xticks([])
ax.set_yticks([])

ax.grid(False)

ax.spines["polar"].set_visible(False)


# ============================================================
# TREE DRAWING
# ============================================================

# We scale the tree so that labels have room outside it.

max_depth = max(depths.values())

label_radius = max_depth * 1.18


def draw_clade(clade):

    if not clade.clades:
        return

    parent_angle = leaf_angles[clade]
    parent_radius = depths[clade]

    child_angles = []
    child_radii = []

    for child in clade.clades:

        child_angle = leaf_angles[child]
        child_radius = depths[child]

        child_angles.append(child_angle)
        child_radii.append(child_radius)

        # ----------------------------------------------------
        # Radial branch
        # ----------------------------------------------------

        ax.plot(
            [child_angle, child_angle],
            [parent_radius, child_radius],
            linewidth=0.45,
            color="0.45",
            alpha=0.75,
            zorder=1
        )

        draw_clade(child)

    # --------------------------------------------------------
    # Horizontal / circular connecting branch
    # --------------------------------------------------------

    if child_angles:

        # Sort angles around the circle
        angles = sorted(child_angles)

        # Use the mean angle of children
        mean_angle = math.atan2(
            sum(math.sin(a) for a in child_angles),
            sum(math.cos(a) for a in child_angles)
        )

        # Draw each child connection around parent radius
        for angle in child_angles:

            ax.plot(
                [mean_angle, angle],
                [parent_radius, parent_radius],
                linewidth=0.45,
                color="0.45",
                alpha=0.75,
                zorder=1
            )


draw_clade(tree.root)


# ============================================================
# TERMINAL POINTS
# ============================================================

for terminal in terminals:

    angle = leaf_angles[terminal]
    radius = depths[terminal]

    accession = None

    for acc in PARETO:

        if acc in (terminal.name or ""):
            accession = acc
            break

    if accession:

        # Pareto candidate
        ax.scatter(
            angle,
            radius,
            s=45,
            color="red",
            edgecolor="black",
            linewidth=0.5,
            zorder=10
        )

    else:

        # Normal taxa
        ax.scatter(
            angle,
            radius,
            s=5,
            color="black",
            alpha=0.65,
            zorder=5
        )


# ============================================================
# PARETO LABELS
# ============================================================

for accession, terminal in pareto_nodes.items():

    angle = leaf_angles[terminal]
    radius = depths[terminal]

    species = PARETO[accession]

    # --------------------------------------------------------
    # Convert angle to readable text orientation
    # --------------------------------------------------------

    angle_deg = math.degrees(angle)

    if angle_deg < -90 or angle_deg > 90:

        rotation = angle_deg + 180
        alignment = "right"

    else:

        rotation = angle_deg
        alignment = "left"

    # --------------------------------------------------------
    # Leader line
    # --------------------------------------------------------

    ax.plot(
        [angle, angle],
        [radius, label_radius],
        color="red",
        linewidth=0.8,
        alpha=0.8,
        zorder=8
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    ax.text(
        angle,
        label_radius,
        species,
        fontsize=8.5,
        fontweight="bold",
        color="darkred",
        rotation=rotation,
        rotation_mode="anchor",
        horizontalalignment=alignment,
        verticalalignment="center",
        zorder=20
    )


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "ASTRAL Species Tree of 436 Representative Yeast Genomes\n"
    "Pareto-Optimal Industrial Chassis Candidates Highlighted",
    fontsize=19,
    fontweight="bold",
    pad=35
)


# ============================================================
# LEGEND
# ============================================================

legend_elements = [

    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=9,
        label="Pareto candidate"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="black",
        markersize=5,
        label="Other representative genome"
    ),

    Line2D(
        [0],
        [0],
        color="0.45",
        linewidth=1,
        label="ASTRAL species-tree branch"
    ),
]


ax.legend(
    handles=legend_elements,
    loc="upper left",
    bbox_to_anchor=(-0.08, 1.02),
    frameon=True,
    fontsize=10
)


# ============================================================
# INFORMATION BOX
# ============================================================

info = (
    "TREE SUMMARY\n"
    "────────────────────────\n"
    f"Representative taxa: {len(terminals)}\n"
    f"Gene trees: 1,605\n"
    f"Pareto candidates shown: {len(pareto_nodes)}/10\n"
    "Inference: ASTRAL\n"
    "Display: circular, unrooted topology\n"
    "\n"
    "Note:\n"
    "S. polymorphus var. africanus\n"
    "is represented by the selected\n"
    "GCA_030463025.1 assembly."
)

fig.text(
    0.02,
    0.03,
    info,
    fontsize=9.5,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round,pad=0.6",
        facecolor="white",
        edgecolor="0.7"
    )
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=400,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)

print(f"\nOutput:")
print(OUTPUT)

print(
    f"\nPareto representatives highlighted: "
    f"{len(pareto_nodes)}/10"
)

if len(pareto_nodes) == 10:
    print("✓ All represented Pareto candidates highlighted.")
else:
    print("! Some Pareto candidates were not detected.")