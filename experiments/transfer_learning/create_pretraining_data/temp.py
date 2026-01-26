from datasets import load_dataset
import pandas as pd
from pathlib import Path
import re
import emoji

# ---------------- CONFIG ----------------
OUTPUT_CSV = Path("datasets/toxigen.csv")
SOURCE_NAME = "toxigen"
TOXICITY_THRESHOLD = 3
# ---------------------------------------

# ---------------- CLEAN TEXT FUNCTION ----------------
def clean_text(text):
    if pd.isna(text):
        return text

    # 1. lowercase
    text = text.lower()

    # 2. remove @USER mentions
    text = re.sub(r'@user', '', text, flags=re.IGNORECASE)

    # 3. remove URLs (actual links or placeholder "URL")
    text = re.sub(r'http\S+|https\S+|url', '', text, flags=re.IGNORECASE)

    # 4. remove underscores, repeated underscores
    text = re.sub(r'_+', ' ', text)

    # 5. remove slashes
    text = text.replace('\\', ' ').replace('/', ' ')

    # 6. remove emojis
    text = emoji.replace_emoji(text, replace="")

    # 7. remove quotation marks (normal + smart)
    text = re.sub(r"[\"“”]", "", text)

    # 8. normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ---------------- LOAD DATASET ----------------
dataset = load_dataset("toxigen/toxigen-data")

# 1. Convert train and test splits to pandas
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# 2. Select relevant columns
train_df = train_df[['text', 'toxicity_human']]
test_df = test_df[['text', 'toxicity_human']]

# 3. Convert toxicity to binary annotation
train_df['annotation'] = (train_df['toxicity_human'] > TOXICITY_THRESHOLD).astype(int)
test_df['annotation'] = (test_df['toxicity_human'] > TOXICITY_THRESHOLD).astype(int)

# 4. Drop original toxicity column
train_df = train_df[['text', 'annotation']]
test_df = test_df[['text', 'annotation']]

# 5. Apply clean_text to text column
train_df['text'] = train_df['text'].astype(str).apply(clean_text)
test_df['text'] = test_df['text'].astype(str).apply(clean_text)

# 6. Add source column
train_df['source'] = SOURCE_NAME
test_df['source'] = SOURCE_NAME

# 7. Merge train + test
df_final = pd.concat([train_df, test_df], ignore_index=True)

# 8. Print annotation distribution
print("Annotation value counts:")
print(df_final['annotation'].value_counts())

# 9. Save to CSV
df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\nSaved cleaned Toxigen dataset to: {OUTPUT_CSV}")
