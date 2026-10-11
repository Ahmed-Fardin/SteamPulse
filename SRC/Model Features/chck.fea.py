import pandas as pd

file = "Data/Processed/steam_features.csv"

df = pd.read_csv(
    file,
    encoding="utf-8",
    encoding_errors="replace",
    low_memory=False
)

print("=" * 50)
print("STEAMPULSE FEATURE CHECK")
print("=" * 50)

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nHIT distribution:")
print(df["hit"].value_counts())

print("\nSplit:")
print(df["split"].value_counts())

print("\nMissing values:")

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values")
else:
    print(missing)

print("\nNumeric features:")

numeric = df.select_dtypes(include="number").columns

for column in numeric:
    print(
        f"{column}: "
        f"min={df[column].min()} | "
        f"max={df[column].max()}"
    )

print("\nDone.")