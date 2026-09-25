import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STAGE 2B
# Pareto / Multi-Trait Chassis Analysis
# ============================================================

INPUT_FILE = (
    r"C:\Y1000_chassis_project\results"
    r"\stage2A_multitrait_profile"
    r"\species_multitrait_profile.csv"
)

OUTPUT_DIR = (
    r"C:\Y1000_chassis_project\results"
    r"\stage2B_pareto_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD SPECIES-LEVEL PROFILE
# ============================================================

print("=" * 70)
print("STAGE 2B — PARETO / MULTI-TRAIT CHASSIS ANALYSIS")
print("=" * 70)

print("\nLoading species-level phenotype profile...")

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset shape: {df.shape}")

print(
    f"Number of species: "
    f"{df['Species'].nunique()}"
)


# ============================================================
# 2. DEFINE THE THREE PHENOTYPES
# ============================================================

traits = [
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth"
]

print("\nSelected optimization traits:")

for trait in traits:
    print(f" - {trait}")


# ============================================================
# 3. CHECK FOR MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print("\nMissing values:")

print(
    df[
        ["Species"] + traits
    ].isna().sum()
)

missing_rows = df[traits].isna().any(axis=1).sum()

print(
    f"\nSpecies with missing values in any "
    f"Pareto trait: {missing_rows}"
)

if missing_rows > 0:

    print(
        "\nWARNING: Species with missing Pareto traits "
        "will be excluded from Pareto analysis."
    )

    pareto_df = df.dropna(
        subset=traits
    ).copy()

else:

    pareto_df = df.copy()


print(
    f"Species entering Pareto analysis: "
    f"{len(pareto_df)}"
)


# ============================================================
# 4. PARETO DOMINANCE FUNCTION
# ============================================================

def dominates(row_a, row_b, trait_columns):

    values_a = row_a[trait_columns].values
    values_b = row_b[trait_columns].values

    # A must be at least as good as B
    # in every trait.
    no_worse = np.all(
        values_a >= values_b
    )

    # A must be strictly better in
    # at least one trait.
    strictly_better = np.any(
        values_a > values_b
    )

    return no_worse and strictly_better


# ============================================================
# 5. CALCULATE PARETO FRONT
# ============================================================

print("\n" + "=" * 70)
print("PARETO ANALYSIS")
print("=" * 70)

n = len(pareto_df)

pareto_flags = np.ones(
    n,
    dtype=bool
)

dominating_counts = np.zeros(
    n,
    dtype=int
)

# Convert to numpy array for efficiency
values = pareto_df[
    traits
].to_numpy(
    dtype=float
)

print(
    f"\nComparing {n} species "
    f"across {len(traits)} traits..."
)


# ------------------------------------------------------------
# Pairwise dominance calculation
# ------------------------------------------------------------

for i in range(n):

    # If another species is at least as good
    # in all dimensions and strictly better
    # in at least one, species i is dominated.

    for j in range(n):

        if i == j:
            continue

        a = values[j]
        b = values[i]

        no_worse = np.all(
            a >= b
        )

        strictly_better = np.any(
            a > b
        )

        if no_worse and strictly_better:

            dominating_counts[i] += 1

            pareto_flags[i] = False


# ============================================================
# 6. ADD RESULTS TO DATAFRAME
# ============================================================

pareto_df[
    "Pareto_Optimal"
] = pareto_flags

pareto_df[
    "Number_of_Dominating_Species"
] = dominating_counts


# ============================================================
# 7. ADD PARETO RANK
# ============================================================

# Rank 1 = Pareto frontier
# Rank 2 = after removing frontier
# etc.
#
# This is a non-dominated sorting procedure.

remaining = pareto_df.index.tolist()

pareto_rank = pd.Series(
    np.nan,
    index=pareto_df.index
)

current_rank = 1

while len(remaining) > 0:

    current_values = values[
        [pareto_df.index.get_loc(idx)
         for idx in remaining]
    ]

    current_front = []

    for local_i in range(
        len(remaining)
    ):

        dominated = False

        for local_j in range(
            len(remaining)
        ):

            if local_i == local_j:
                continue

            a = current_values[
                local_j
            ]

            b = current_values[
                local_i
            ]

            no_worse = np.all(
                a >= b
            )

            strictly_better = np.any(
                a > b
            )

            if (
                no_worse
                and strictly_better
            ):

                dominated = True
                break

        if not dominated:

            current_front.append(
                remaining[local_i]
            )

    if len(current_front) == 0:
        break

    for idx in current_front:
        pareto_rank.loc[idx] = current_rank

    remaining = [
        idx
        for idx in remaining
        if idx not in current_front
    ]

    current_rank += 1


pareto_df[
    "Pareto_Rank"
] = pareto_rank.astype(int)


# ============================================================
# 8. SORT RESULTS
# ============================================================

pareto_df = pareto_df.sort_values(
    [
        "Pareto_Rank",
        "Carbon_Breadth",
        "Nitrogen_Breadth",
        "Utilized_Median_Growth"
    ],
    ascending=[
        True,
        False,
        False,
        False
    ]
)


# ============================================================
# 9. SAVE COMPLETE RESULTS
# ============================================================

complete_output = os.path.join(
    OUTPUT_DIR,
    "species_pareto_results.csv"
)

pareto_df.to_csv(
    complete_output,
    index=False
)


# ============================================================
# 10. EXTRACT PARETO FRONTIER
# ============================================================

frontier = pareto_df[
    pareto_df["Pareto_Optimal"]
].copy()

frontier = frontier.sort_values(
    [
        "Carbon_Breadth",
        "Nitrogen_Breadth",
        "Utilized_Median_Growth"
    ],
    ascending=False
)


frontier_output = os.path.join(
    OUTPUT_DIR,
    "pareto_frontier.csv"
)

frontier.to_csv(
    frontier_output,
    index=False
)


# ============================================================
# 11. EXTRACT DOMINATED SPECIES
# ============================================================

dominated = pareto_df[
    ~pareto_df["Pareto_Optimal"]
].copy()

dominated_output = os.path.join(
    OUTPUT_DIR,
    "dominated_species.csv"
)

dominated.to_csv(
    dominated_output,
    index=False
)


# ============================================================
# 12. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PARETO SUMMARY")
print("=" * 70)

print(
    f"\nTotal species analyzed: "
    f"{len(pareto_df)}"
)

print(
    f"Pareto-optimal species: "
    f"{len(frontier)}"
)

print(
    f"Dominated species: "
    f"{len(dominated)}"
)

print(
    f"Percentage Pareto-optimal: "
    f"{100 * len(frontier) / len(pareto_df):.2f}%"
)


# ============================================================
# 13. DISPLAY PARETO FRONTIER
# ============================================================

print("\n" + "=" * 70)
print("PARETO FRONTIER")
print("=" * 70)

display_columns = [
    "Species",
    "N_Strains",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Number_of_Dominating_Species",
    "Pareto_Rank"
]

print(
    frontier[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# 14. TRAIT CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("TRAIT CORRELATIONS")
print("=" * 70)

correlation_matrix = pareto_df[
    traits
].corr(
    method="spearman"
)

print(
    correlation_matrix.round(4)
)


correlation_output = os.path.join(
    OUTPUT_DIR,
    "trait_spearman_correlations.csv"
)

correlation_matrix.to_csv(
    correlation_output
)


# ============================================================
# 15. 3D PARETO PLOT
# ============================================================

print("\nCreating 3D Pareto plot...")

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

dominated_plot = pareto_df[
    ~pareto_df["Pareto_Optimal"]
]

frontier_plot = pareto_df[
    pareto_df["Pareto_Optimal"]
]


ax.scatter(
    dominated_plot[
        "Carbon_Breadth"
    ],
    dominated_plot[
        "Nitrogen_Breadth"
    ],
    dominated_plot[
        "Utilized_Median_Growth"
    ],
    alpha=0.35,
    s=20,
    label="Dominated"
)


ax.scatter(
    frontier_plot[
        "Carbon_Breadth"
    ],
    frontier_plot[
        "Nitrogen_Breadth"
    ],
    frontier_plot[
        "Utilized_Median_Growth"
    ],
    s=55,
    label="Pareto-optimal"
)


ax.set_xlabel(
    "Carbon Breadth"
)

ax.set_ylabel(
    "Nitrogen Breadth"
)

ax.set_zlabel(
    "Utilized Median Growth"
)

ax.set_title(
    "Y1000+ Multi-Trait Phenotype Space"
)

ax.legend()


plot_3d = os.path.join(
    OUTPUT_DIR,
    "pareto_frontier_3d.png"
)

plt.tight_layout()

plt.savefig(
    plot_3d,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. CARBON vs NITROGEN
# ============================================================

print(
    "Creating Carbon vs Nitrogen plot..."
)

plt.figure(
    figsize=(9, 7)
)

plt.scatter(
    dominated_plot[
        "Carbon_Breadth"
    ],
    dominated_plot[
        "Nitrogen_Breadth"
    ],
    alpha=0.35,
    s=25,
    label="Dominated"
)

plt.scatter(
    frontier_plot[
        "Carbon_Breadth"
    ],
    frontier_plot[
        "Nitrogen_Breadth"
    ],
    s=55,
    label="Pareto-optimal"
)

plt.xlabel(
    "Carbon Breadth"
)

plt.ylabel(
    "Nitrogen Breadth"
)

plt.title(
    "Carbon vs Nitrogen Breadth"
)

plt.legend()

plt.tight_layout()

plot_cn = os.path.join(
    OUTPUT_DIR,
    "pareto_frontier_carbon_nitrogen.png"
)

plt.savefig(
    plot_cn,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. CARBON vs GROWTH
# ============================================================

print(
    "Creating Carbon vs Growth plot..."
)

plt.figure(
    figsize=(9, 7)
)

plt.scatter(
    dominated_plot[
        "Carbon_Breadth"
    ],
    dominated_plot[
        "Utilized_Median_Growth"
    ],
    alpha=0.35,
    s=25,
    label="Dominated"
)

plt.scatter(
    frontier_plot[
        "Carbon_Breadth"
    ],
    frontier_plot[
        "Utilized_Median_Growth"
    ],
    s=55,
    label="Pareto-optimal"
)

plt.xlabel(
    "Carbon Breadth"
)

plt.ylabel(
    "Utilized Median Growth"
)

plt.title(
    "Carbon Breadth vs Utilized Median Growth"
)

plt.legend()

plt.tight_layout()

plot_cg = os.path.join(
    OUTPUT_DIR,
    "pareto_frontier_carbon_growth.png"
)

plt.savefig(
    plot_cg,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. NITROGEN vs GROWTH
# ============================================================

print(
    "Creating Nitrogen vs Growth plot..."
)

plt.figure(
    figsize=(9, 7)
)

plt.scatter(
    dominated_plot[
        "Nitrogen_Breadth"
    ],
    dominated_plot[
        "Utilized_Median_Growth"
    ],
    alpha=0.35,
    s=25,
    label="Dominated"
)

plt.scatter(
    frontier_plot[
        "Nitrogen_Breadth"
    ],
    frontier_plot[
        "Utilized_Median_Growth"
    ],
    s=55,
    label="Pareto-optimal"
)

plt.xlabel(
    "Nitrogen Breadth"
)

plt.ylabel(
    "Utilized Median Growth"
)

plt.title(
    "Nitrogen Breadth vs Utilized Median Growth"
)

plt.legend()

plt.tight_layout()

plot_ng = os.path.join(
    OUTPUT_DIR,
    "pareto_frontier_nitrogen_growth.png"
)

plt.savefig(
    plot_ng,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. SAVE REPORT
# ============================================================

report_output = os.path.join(
    OUTPUT_DIR,
    "stage2B_report.txt"
)

with open(
    report_output,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 2B — PARETO / MULTI-TRAIT "
        "CHASSIS ANALYSIS\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        f"Species analyzed: {len(pareto_df)}\n"
    )

    f.write(
        f"Pareto-optimal species: "
        f"{len(frontier)}\n"
    )

    f.write(
        f"Dominated species: "
        f"{len(dominated)}\n"
    )

    f.write(
        f"Pareto percentage: "
        f"{100 * len(frontier) / len(pareto_df):.2f}%\n\n"
    )

    f.write(
        "Optimization traits:\n"
    )

    for trait in traits:
        f.write(
            f"- {trait}\n"
        )

    f.write("\n")

    f.write(
        "Pareto frontier:\n\n"
    )

    f.write(
        frontier[
            display_columns
        ].to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        "Spearman correlation matrix:\n\n"
    )

    f.write(
        correlation_matrix.round(4).to_string()
    )

    f.write("\n")


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2B COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

print(
    "1. species_pareto_results.csv"
)

print(
    "2. pareto_frontier.csv"
)

print(
    "3. dominated_species.csv"
)

print(
    "4. trait_spearman_correlations.csv"
)

print(
    "5. pareto_frontier_3d.png"
)

print(
    "6. pareto_frontier_carbon_nitrogen.png"
)

print(
    "7. pareto_frontier_carbon_growth.png"
)

print(
    "8. pareto_frontier_nitrogen_growth.png"
)

print(
    "9. stage2B_report.txt"
)

print("\nNext step:")
print(
    "Stage 3 — Genomic Feature Acquisition"
)