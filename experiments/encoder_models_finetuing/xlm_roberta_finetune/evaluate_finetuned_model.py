import argparse
import logging
import os
import re

import emoji
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("evaluation_xlm_roberta.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Finetuned XLM-RoBERTa on Subtask 1"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./finetune_results_xlm_roberta/final_model",
        help="Path to finetuned model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../../data/subtask1/dev",
        help="Path to dev dataset directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for prediction"
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="Maximum sequence length"
    )
    return parser.parse_args()


def clean_text(text):
    if pd.isna(text):
        return text

    # 1. lowercase
    text = text.lower()

    # 2. remove @USER mentions
    text = re.sub(r"@user", "", text, flags=re.IGNORECASE)
    text = re.sub(r"@url", "", text, flags=re.IGNORECASE)

    # 3. remove URLs (actual links or placeholder "URL")
    # text = re.sub(r'http\S+|https\S+|url', '', text, flags=re.IGNORECASE)

    # 4. remove underscores, repeated underscores
    text = re.sub(r"_+", " ", text)

    # 5. remove slashes
    text = text.replace("\\", " ").replace("/", " ")

    # 6. remove emojis
    text = emoji.replace_emoji(text, replace="")

    # 7. remove quotation marks (normal + smart)
    text = re.sub(r'["""]', "", text)

    # 8. normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    args = parse_args()

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Model & Tokenizer
    logger.info(f"Loading model from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    model.to(device)

    # Create trainer for easy prediction
    trainer = Trainer(model=model)

    # 2. Iterate over files
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return

    csv_files = [f for f in os.listdir(args.data_dir) if f.endswith(".csv")]
    logger.info(f"Found {len(csv_files)} files in {args.data_dir}")

    all_predictions = []
    all_labels = []

    results = []

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    for file in sorted(csv_files):
        lang = file.replace(".csv", "")
        file_path = os.path.join(args.data_dir, file)

        logger.info(f"Processing {lang} ({file})...")

        try:
            df = pd.read_csv(file_path)

            # Clean text
            df["text"] = df["text"].astype(str).apply(clean_text)

            # Prepare dataset
            # Determine label column
            label_col = "polarization" if "polarization" in df.columns else "labels"
            if label_col not in df.columns:
                logger.warning(
                    f"Label column not found in {file}. Skipping evaluation for this file."
                )
                continue

            dataset = Dataset.from_pandas(df[["text"]], preserve_index=False)
            tokenized = dataset.map(tokenize_function, batched=True)

            # Predict
            preds_output = trainer.predict(tokenized)
            pred_labels = np.argmax(preds_output.predictions, axis=1)
            true_labels = df[label_col].values

            # Calculate metrics
            f1 = f1_score(true_labels, pred_labels, average="macro")
            acc = accuracy_score(true_labels, pred_labels)

            logger.info(
                f"--> {lang}: F1-Macro={f1:.4f}, Accuracy={acc:.4f}, Support={len(df)}"
            )

            results.append(
                {"lang": lang, "f1_macro": f1, "accuracy": acc, "support": len(df)}
            )

            all_predictions.extend(pred_labels)
            all_labels.extend(true_labels)

        except Exception as e:
            logger.error(f"Error processing {file}: {e}")

    # 3. Overall Evaluation
    if all_labels:
        logger.info("\n=== Overall Results ===")
        overall_f1 = f1_score(all_labels, all_predictions, average="macro")
        overall_acc = accuracy_score(all_labels, all_predictions)

        logger.info(f"Overall F1-Macro: {overall_f1:.4f}")
        logger.info(f"Overall Accuracy: {overall_acc:.4f}")
        logger.info(f"Total Examples: {len(all_labels)}")

        logger.info("\nClassification Report:\n")
        logger.info(classification_report(all_labels, all_predictions, digits=4))

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv("evaluation_results_per_lang_xlm_roberta.csv", index=False)
        logger.info(
            "Saved per-language results to evaluation_results_per_lang_xlm_roberta.csv"
        )
    else:
        logger.warning("No data found or processed.")


if __name__ == "__main__":
    main()
