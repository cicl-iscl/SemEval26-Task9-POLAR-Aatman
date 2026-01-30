
import os
import shutil
import argparse
import numpy as np
import pandas as pd
import torch
import evaluate
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune mDeBERTa on Subtask 1")
    parser.add_argument("--model_path", type=str, default="./pretrain_mdeberta", help="Path to pretrained model")
    parser.add_argument("--data_dir", type=str, default="../../../data/subtask1", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./finetune_results", help="Output directory for results")
    parser.add_argument("--submission_dir", type=str, default="./subtask_1", help="Directory for submission files")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def load_split(split_dir):
    dfs = []
    if not os.path.exists(split_dir):
        print(f"Directory not found: {split_dir}")
        return pd.DataFrame()
        
    for file in os.listdir(split_dir):
        if file.endswith(".csv"):
            lang = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(split_dir, file))
            df["lang"] = lang
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Pretrained Model & Tokenizer
    print(f"Loading model from {args.model_path}...")
    try:
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_path)
        model = DebertaV2ForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying AutoTokenizer/AutoModel...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
        
    model.to(device)
    
    # 2. Load Data
    train_dir = os.path.join(args.data_dir, "train")
    dev_dir = os.path.join(args.data_dir, "dev")
    test_dir = os.path.join(args.data_dir, "test")
    
    print("Loading Train Data...")
    raw_train_df = load_split(train_dir)
    print(f"Loaded {len(raw_train_df)} training examples")
    
    print("Loading Dev Data (Used as internal Test)...")
    raw_dev_df = load_split(dev_dir)
    print(f"Loaded {len(raw_dev_df)} dev examples")
    
    print("Loading Test Data (For Submission)...")
    raw_test_df = load_split(test_dir)
    print(f"Loaded {len(raw_test_df)} test examples")
    
    # 3. Data Processing & Splitting
    # Rename 'polarization' to 'labels'
    if "polarization" in raw_train_df.columns:
        raw_train_df = raw_train_df.rename(columns={"polarization": "labels"})
    if "polarization" in raw_dev_df.columns:
        raw_dev_df = raw_dev_df.rename(columns={"polarization": "labels"})
        
    # Split Train into 95% Train / 5% Val
    train_df, val_df = train_test_split(
        raw_train_df,
        test_size=0.05,
        stratify=raw_train_df["labels"],
        random_state=args.seed,
        shuffle=True
    )
    
    # Use Dev as Internal Test
    test_df = raw_dev_df.copy()
    
    print("Shape after split:")
    print(f"Train:      {train_df.shape}")
    print(f"Validation: {val_df.shape}")
    print(f"Test (Dev): {test_df.shape}")
    
    # Create Dataset Objects
    train_dataset = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df[["text", "labels"]], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[["text", "labels"]], preserve_index=False)
    
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    # 4. Tokenization
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=256
        )
    
    print("Tokenizing datasets...")
    encoded_dataset = dataset.map(tokenize_function, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # 5. Training Setup
    steps_per_epoch = len(encoded_dataset["train"]) // (args.batch_size * args.gradient_accumulation_steps)
    eval_steps = steps_per_epoch
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=1000,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=eval_steps,
        eval_steps=eval_steps,
        save_steps=eval_steps * 20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        eval_strategy="steps",
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
        # fp16=torch.cuda.is_available(),
    )
    
    metric_f1 = evaluate.load("f1")
    metric_acc = evaluate.load("accuracy")
    
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
        acc = metric_acc.compute(predictions=predictions, references=labels)["accuracy"]
        return {"f1": f1, "accuracy": acc}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # 6. Train
    print("Starting training...")
    trainer.train()
    
    # 7. Evaluation on Internal Test Set (Dev Folder)
    print("Evaluating on Internal Test Set (Dev folder data)...")
    preds_output = trainer.predict(encoded_dataset["test"])
    
    pred_labels = np.argmax(preds_output.predictions, axis=1)
    true_labels = preds_output.label_ids
    
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=["Not Polar (0)", "Polar (1)"], digits=4))
    
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    print(f"Macro F1: {macro_f1:.4f}")
    
    # Per-Language Analysis
    test_df["preds"] = pred_labels
    print("\n=== Macro F1 per Language ===")
    results = []
    for lang in sorted(test_df["lang"].unique()):
        lang_df = test_df[test_df["lang"] == lang]
        f1 = f1_score(lang_df["labels"], lang_df["preds"], average="macro")
        acc = accuracy_score(lang_df["labels"], lang_df["preds"])
        print(f"{lang}: F1={f1:.4f}, Acc={acc:.4f}, Support={len(lang_df)}")
        results.append({"lang": lang, "f1_macro": f1, "accuracy": acc, "count": len(lang_df)})
        
    results_df = pd.DataFrame(results)
    print(f"\nAverage Macro F1 across languages: {results_df['f1_macro'].mean():.4f}")
    
    # 8. Generate Submission (Test Folder)
    submission_dir = args.submission_dir
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir)
    
    print("Generating predictions for submission...")
    
    # tokenize test set for submission
    submission_dataset = Dataset.from_pandas(raw_test_df[["text"]], preserve_index=False)
    submission_tokenized = submission_dataset.map(tokenize_function, batched=True)
    submission_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Predict
    submission_preds_output = trainer.predict(submission_tokenized)
    submission_labels = np.argmax(submission_preds_output.predictions, axis=1)
    
    # Add predictions back to dataframe
    raw_test_df["polarization"] = submission_labels
    
    # Save individual files
    languages = sorted(raw_test_df["lang"].unique())
    print(f"Processing {len(languages)} languages for submission...")
    
    for lang in languages:
        lang_df = raw_test_df[raw_test_df["lang"] == lang]
        output_df = lang_df[["id", "polarization"]]
        
        output_path = os.path.join(submission_dir, f"pred_{lang}.csv")
        output_df.to_csv(output_path, index=False)
        
    print("Zipping prediction files...")
    shutil.make_archive("subtask_1", "zip", submission_dir)
    print(f"Created subtask_1.zip in {os.getcwd()}")
    
    # Also save the final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    print(f"Saving final model to {final_model_path}...")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

if __name__ == "__main__":
    main()
