import os
import pandas as pd
import numpy as np

# ============================================================
# STAGE 2A
# Build Final Species-Level Multi-Trait Phenotype Profile
# ============================================================

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_FILE = r"C:\Y1000_chassis_project\data\phenotype\media-3.xlsx"

OUTPUT_DIR = (
    r"C:\Y1000_chassis_project\results"
    r"\stage2A_multitrait_profile"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("STAGE 2A — MULTI-TRAIT PHENOTYPE PROFILE")
print("=" * 70)

print("\nLoading phenotype dataset...")

# Row 1 of the Excel file contains the actual column headers
df = pd.read_excel(
    INPUT_FILE,
    header=1
)

print(f"\nOriginal dataset shape: {df.shape}")

print("\nOriginal columns:")
print(df.columns.tolist())


# ============================================================
# 2. STANDARDIZE STRAIN ID COLUMN
# ============================================================

# The original Y1000+ Excel file uses:
# yHMPu50000
#
# We rename it internally to:
# Strain_ID

if "yHMPu50000" in df.columns:

    df = df.rename(
        columns={
            "yHMPu50000": "Strain_ID"
        }
    )

    print("\nStrain ID column:")
    print("yHMPu50000 -> Strain_ID")

elif "Strain_ID" in df.columns:

    print("\nStrain ID column already named Strain_ID.")

else:

    print("\nERROR: Could not identify the strain ID column.")

    print("\nAvailable columns:")
    for col in df.columns:
        print(" -", col)

    raise ValueError(
        "Neither 'yHMPu50000' nor 'Strain_ID' "
        "was found."
    )


# ============================================================
# 3. DEFINE PHENOTYPE CONDITIONS
# ============================================================

carbon_conditions = [

    "Cellobiose",
    "Citrate",
    "D-Glucosamine",
    "Fructose",
    "Galactose",
    "Glucose",
    "Glycerol",
    "L-Arabinose",
    "L-Sorbose",
    "Lactose",
    "Maltose",
    "Mannose",
    "myo-Inositol",
    "Rhamnose",
    "Xylose",
    "Raffinose",
    "Sucrose",
    "DL-Lactate"

]


nitrogen_conditions = [

    "Allantoin",
    "Creatinine",
    "L-Lysine",
    "Nitrate",
    "Nitrite",
    "Urea"

]


# ============================================================
# 4. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = (

    [
        "Species",
        "Strain_ID"
    ]

    + carbon_conditions

    + nitrogen_conditions

)


missing_columns = [

    col
    for col in required_columns
    if col not in df.columns

]


if missing_columns:

    print("\nERROR: Missing columns:")

    for col in missing_columns:
        print(" -", col)

    raise ValueError(
        "One or more required phenotype columns "
        "are missing."
    )


print("\nAll required phenotype columns found.")


# ============================================================
# 5. BASIC DATA INFORMATION
# ============================================================

print("\nDataset information")
print("-" * 70)

print(
    f"Rows: {len(df)}"
)

print(
    f"Unique strains: "
    f"{df['Strain_ID'].nunique()}"
)

print(
    f"Unique species: "
    f"{df['Species'].nunique()}"
)


# ============================================================
# 6. CALCULATE CARBON BREADTH
# ============================================================

# Carbon Breadth =
# number of carbon conditions with positive growth (> 0)

df["Carbon_Breadth_Calculated"] = (

    df[carbon_conditions] > 0

).sum(axis=1)


# ============================================================
# 7. CALCULATE NITROGEN BREADTH
# ============================================================

# Nitrogen Breadth =
# number of nitrogen conditions with positive growth (> 0)

df["Nitrogen_Breadth_Calculated"] = (

    df[nitrogen_conditions] > 0

).sum(axis=1)


# ============================================================
# 8. CALCULATE UTILIZED MEDIAN GROWTH
# ============================================================

# For each strain:
#
# 1. Take all carbon-growth measurements
# 2. Remove missing values
# 3. Keep only positive-growth conditions
# 4. Calculate the median
#
# This gives:
#
# Utilized Median Growth
#
# It measures typical growth performance among
# substrates that the organism actually utilizes.

def calculate_utilized_median(row):

    values = row[carbon_conditions].dropna()

    positive_values = values[
        values > 0
    ]

    if len(positive_values) == 0:

        return np.nan

    return positive_values.median()


df["Utilized_Median_Growth"] = (

    df.apply(
        calculate_utilized_median,
        axis=1
    )

)


# ============================================================
# 9. CALCULATE DATA COVERAGE
# ============================================================

df["Carbon_Conditions_Available"] = (

    df[carbon_conditions]
    .notna()
    .sum(axis=1)

)


df["Nitrogen_Conditions_Available"] = (

    df[nitrogen_conditions]
    .notna()
    .sum(axis=1)

)


# ============================================================
# 10. VALIDATE CARBON BREADTH
# ============================================================

print("\n" + "=" * 70)
print("CARBON BREADTH VALIDATION")
print("=" * 70)


if "Carbon Breadth" in df.columns:

    carbon_difference = (

        df["Carbon_Breadth_Calculated"]
        - df["Carbon Breadth"]

    )

    exact_matches = (

        carbon_difference == 0

    ).sum()

    total_rows = len(df)

    print(
        f"\nExact matches: "
        f"{exact_matches}/{total_rows}"
    )

    print(
        f"Mismatch count: "
        f"{(carbon_difference != 0).sum()}"
    )

    if (carbon_difference != 0).sum() > 0:

        print(
            "\nWARNING:"
            "\nSome calculated Carbon Breadth values "
            "differ from the published values."
        )

        mismatch_table = df.loc[
            carbon_difference != 0,
            [
                "Species",
                "Strain_ID",
                "Carbon Breadth",
                "Carbon_Breadth_Calculated"
            ]
        ]

        print("\nFirst mismatches:")

        print(
            mismatch_table
            .head(10)
            .to_string(index=False)
        )

else:

    print(
        "\nPublished 'Carbon Breadth' column "
        "was not found."
    )


# ============================================================
# 11. VALIDATE NITROGEN BREADTH
# ============================================================

print("\n" + "=" * 70)
print("NITROGEN BREADTH VALIDATION")
print("=" * 70)


if "Nitrogen Breadth" in df.columns:

    nitrogen_difference = (

        df["Nitrogen_Breadth_Calculated"]
        - df["Nitrogen Breadth"]

    )

    exact_matches = (

        nitrogen_difference == 0

    ).sum()

    total_rows = len(df)

    print(
        f"\nExact matches: "
        f"{exact_matches}/{total_rows}"
    )

    print(
        f"Mismatch count: "
        f"{(nitrogen_difference != 0).sum()}"
    )

    if (nitrogen_difference != 0).sum() > 0:

        print(
            "\nWARNING:"
            "\nSome calculated Nitrogen Breadth values "
            "differ from the published values."
        )

        mismatch_table = df.loc[
            nitrogen_difference != 0,
            [
                "Species",
                "Strain_ID",
                "Nitrogen Breadth",
                "Nitrogen_Breadth_Calculated"
            ]
        ]

        print("\nFirst mismatches:")

        print(
            mismatch_table
            .head(10)
            .to_string(index=False)
        )

else:

    print(
        "\nPublished 'Nitrogen Breadth' column "
        "was not found."
    )


# ============================================================
# 12. STRAIN-LEVEL MULTI-TRAIT PROFILE
# ============================================================

print("\n" + "=" * 70)
print("STRAIN-LEVEL PROFILE")
print("=" * 70)


strain_columns = [

    "Species",
    "Strain_ID",

    "Carbon_Breadth_Calculated",

    "Nitrogen_Breadth_Calculated",

    "Utilized_Median_Growth",

    "Carbon_Conditions_Available",

    "Nitrogen_Conditions_Available"

]


strain_profile = df[
    strain_columns
].copy()


# Rename calculated traits to final names

strain_profile = strain_profile.rename(

    columns={

        "Carbon_Breadth_Calculated":
            "Carbon_Breadth",

        "Nitrogen_Breadth_Calculated":
            "Nitrogen_Breadth"

    }

)


# Save strain-level table

strain_output = os.path.join(

    OUTPUT_DIR,
    "strain_multitrait_profile.csv"

)


strain_profile.to_csv(

    strain_output,
    index=False

)


print(
    f"\nStrain-level profile saved to:"
    f"\n{strain_output}"
)


# ============================================================
# 13. SPECIES-LEVEL AGGREGATION
# ============================================================

print("\n" + "=" * 70)
print("SPECIES-LEVEL AGGREGATION")
print("=" * 70)


# We aggregate strains within each species.
#
# Carbon Breadth:
#     mean across strains
#
# Nitrogen Breadth:
#     mean across strains
#
# Utilized Median Growth:
#     median across strains
#
# We also retain within-species SD so that
# strain-level variation is not completely hidden.

species_profile = (

    df.groupby("Species")

      .agg(

          N_Strains=(
              "Strain_ID",
              "nunique"
          ),

          Carbon_Breadth=(
              "Carbon_Breadth_Calculated",
              "mean"
          ),

          Nitrogen_Breadth=(
              "Nitrogen_Breadth_Calculated",
              "mean"
          ),

          Utilized_Median_Growth=(
              "Utilized_Median_Growth",
              "median"
          ),

          Carbon_Breadth_SD=(
              "Carbon_Breadth_Calculated",
              "std"
          ),

          Nitrogen_Breadth_SD=(
              "Nitrogen_Breadth_Calculated",
              "std"
          )

      )

      .reset_index()

)


# For species represented by only one strain,
# SD cannot be calculated and becomes NaN.
#
# Replace this with zero because there is no
# observed within-species variation in the dataset.

species_profile[
    "Carbon_Breadth_SD"
] = species_profile[
    "Carbon_Breadth_SD"
].fillna(0)


species_profile[
    "Nitrogen_Breadth_SD"
] = species_profile[
    "Nitrogen_Breadth_SD"
].fillna(0)


# ============================================================
# 14. ROUND NUMERICAL VALUES
# ============================================================

numeric_columns = [

    "Carbon_Breadth",

    "Nitrogen_Breadth",

    "Utilized_Median_Growth",

    "Carbon_Breadth_SD",

    "Nitrogen_Breadth_SD"

]


species_profile[
    numeric_columns
] = species_profile[
    numeric_columns
].round(4)


# ============================================================
# 15. SAVE SPECIES-LEVEL PROFILE
# ============================================================

species_output = os.path.join(

    OUTPUT_DIR,
    "species_multitrait_profile.csv"

)


species_profile.to_csv(

    species_output,
    index=False

)


print(
    f"\nSpecies-level profile saved to:"
    f"\n{species_output}"
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROFILE SUMMARY")
print("=" * 70)


print(
    f"\nNumber of rows: "
    f"{len(df)}"
)


print(
    f"Unique strains: "
    f"{df['Strain_ID'].nunique()}"
)


print(
    f"Unique species: "
    f"{species_profile['Species'].nunique()}"
)


print(
    f"Species represented by >1 strain: "
    f"{(species_profile['N_Strains'] > 1).sum()}"
)


print("\nThree selected phenotype dimensions:")

print(
    "1. Carbon Breadth"
)

print(
    "2. Nitrogen Breadth"
)

print(
    "3. Utilized Median Growth"
)


# ============================================================
# 17. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SPECIES-LEVEL PHENOTYPE STATISTICS")
print("=" * 70)


print(

    species_profile[
        [
            "Carbon_Breadth",
            "Nitrogen_Breadth",
            "Utilized_Median_Growth"
        ]
    ].describe()

)


# ============================================================
# 18. TOP SPECIES — CARBON BREADTH
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 SPECIES — CARBON BREADTH")
print("=" * 70)


top_carbon = (

    species_profile

    .sort_values(

        [
            "Carbon_Breadth",
            "Nitrogen_Breadth",
            "Utilized_Median_Growth"
        ],

        ascending=False

    )

    [
        [
            "Species",
            "N_Strains",
            "Carbon_Breadth",
            "Nitrogen_Breadth",
            "Utilized_Median_Growth"
        ]
    ]

    .head(10)

)


print(
    top_carbon.to_string(
        index=False
    )
)


# ============================================================
# 19. TOP SPECIES — NITROGEN BREADTH
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 SPECIES — NITROGEN BREADTH")
print("=" * 70)


top_nitrogen = (

    species_profile

    .sort_values(

        [
            "Nitrogen_Breadth",
            "Carbon_Breadth",
            "Utilized_Median_Growth"
        ],

        ascending=False

    )

    [
        [
            "Species",
            "N_Strains",
            "Carbon_Breadth",
            "Nitrogen_Breadth",
            "Utilized_Median_Growth"
        ]
    ]

    .head(10)

)


print(
    top_nitrogen.to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE SUMMARY REPORT
# ============================================================

report_file = os.path.join(

    OUTPUT_DIR,
    "stage2A_report.txt"

)


with open(

    report_file,
    "w",
    encoding="utf-8"

) as f:

    f.write(
        "STAGE 2A — MULTI-TRAIT PHENOTYPE PROFILE\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Original rows: {len(df)}\n"
    )

    f.write(
        f"Unique strains: "
        f"{df['Strain_ID'].nunique()}\n"
    )

    f.write(
        f"Unique species: "
        f"{species_profile['Species'].nunique()}\n"
    )

    f.write(
        f"Species with >1 strain: "
        f"{(species_profile['N_Strains'] > 1).sum()}\n\n"
    )

    f.write(
        "Selected phenotype dimensions:\n"
    )

    f.write(
        "1. Carbon Breadth\n"
    )

    f.write(
        "2. Nitrogen Breadth\n"
    )

    f.write(
        "3. Utilized Median Growth\n\n"
    )

    f.write(
        "Species-level descriptive statistics:\n\n"
    )

    f.write(
        species_profile[
            [
                "Carbon_Breadth",
                "Nitrogen_Breadth",
                "Utilized_Median_Growth"
            ]
        ]
        .describe()
        .to_string()
    )

    f.write("\n")


# ============================================================
# 21. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2A COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

print(
    "1.",
    strain_output
)

print(
    "2.",
    species_output
)

print(
    "3.",
    report_file
)

print("\nNext step:")
print(
    "Stage 2B — Pareto / Multi-Trait Chassis Analysis"
)