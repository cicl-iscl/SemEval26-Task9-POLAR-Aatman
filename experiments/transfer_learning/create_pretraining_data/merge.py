import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
DATASETS_DIR = Path("datasets")
OUTPUT_FILE = DATASETS_DIR / "pretrain.csv"
CSV_FILES = [
    "hate_check_hi.csv",
    "hatexplain.csv",
    "implicit_hate.csv",
    "macd.csv",
    "olid.csv",
    "toxigen.csv",
    "uli.csv"
]
# ---------------------------------------

# 1. Load all CSVs into a list
dfs = []
for csv_file in CSV_FILES:
    path = DATASETS_DIR / csv_file
    df = pd.read_csv(path)
    df['source'] = df.get('source', csv_file.split('.')[0])  # fallback source
    dfs.append(df)

# 2. Check column names and types
first_columns = dfs[0].columns
first_dtypes = dfs[0].dtypes
for i, df in enumerate(dfs[1:], start=1):
    if not all(df.columns == first_columns):
        print(f"Warning: Column names mismatch in {CSV_FILES[i]}")
    if not all(df.dtypes == first_dtypes):
        print(f"Warning: Column types mismatch in {CSV_FILES[i]}")

# 3. Concatenate all dataframes
df_combined = pd.concat(dfs, ignore_index=True)

# 4. Print overall annotation distribution (%)
annotation_counts = df_combined['annotation'].value_counts(normalize=True) * 100
print("Overall annotation distribution (%):")
print(annotation_counts.round(2))

# 5. Print annotation distribution per source (%)
print("\nAnnotation distribution per source (%):")
source_groups = df_combined.groupby('source')['annotation']
for source, group in source_groups:
    dist = group.value_counts(normalize=True) * 100
    print(f"\nSource: {source}")
    print(dist.round(2))

# 6. Rename annotation column to labels
df_combined = df_combined.rename(columns={'annotation': 'labels'})

# 7. Keep only text and labels columns
df_final = df_combined[['text', 'labels']]

# 8. Save final combined CSV
df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"\nSaved combined pretrain dataset to: {OUTPUT_FILE}")
print("Final shape:", df_final.shape)

'''
Overall annotation distribution (%):
annotation
0    54.42
1    45.58
Name: proportion, dtype: float64

Annotation distribution per source (%):

Source: hate_check_hi
annotation
1    70.21
0    29.79
Name: proportion, dtype: float64

Source: hatexplain
annotation
1    61.22
0    38.78
Name: proportion, dtype: float64

Source: implicit_hate
annotation
0    61.88
1    38.12
Name: proportion, dtype: float64

Source: macd
annotation
0    51.98
1    48.02
Name: proportion, dtype: float64

Source: olid
annotation
0    67.1
1    32.9
Name: proportion, dtype: float64

Source: toxigen
annotation
0    65.83
1    34.17
Name: proportion, dtype: float64

Source: uli_dataset
annotation
0    58.64
1    41.36
Name: proportion, dtype: float64

Saved combined pretrain dataset to: datasets/pretrain.csv
Final shape: (118802, 2)
'''