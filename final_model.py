import pandas as pd

path = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\tables\feature_importance_stability_420.csv"

df = pd.read_csv(path)

print("=" * 80)
print("VALIDATED ML FILE INSPECTION")
print("=" * 80)

print("\nShape:", df.shape)

print("\nColumns:")
for c in df.columns:
    print(" ", c)

print("\nFeature_Set values:")
print(df["Feature_Set"].value_counts(dropna=False))

print("\nModel values:")
print(df["Model"].value_counts(dropna=False))

print("\nPhenotype values:")
print(df["Phenotype"].value_counts(dropna=False))

print("\nTop 30 rows by Mean_Importance:")
print(
    df.sort_values(
        ["Mean_Importance", "Fold_Stability"],
        ascending=[False, False]
    ).head(30).to_string(index=False)
)

print("\nBest rows by Fold_Stability:")
print(
    df.sort_values(
        ["Fold_Stability", "Mean_Importance"],
        ascending=[False, False]
    ).head(30).to_string(index=False)
)