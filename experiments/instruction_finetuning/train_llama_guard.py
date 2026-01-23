"""
Instruction Fine-tuning: Llama-Guard-3-8B for Polarization Detection

This script fine-tunes the meta-llama/Llama-Guard-3-8B model on the polarization
detection dataset using QLoRA (Quantized Low-Rank Adaptation).

Usage:
    python train_llama_guard.py

Requirements:
    - .env file with HF_TOKEN variable
    - GPU with 16GB+ VRAM (tested on A100)
    - Data in ../../subtask1/train and ../../subtask1/dev
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================


class Config:
    """Training configuration"""

    # Model
    MODEL_NAME = "meta-llama/Llama-Guard-3-8B"
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Paths
    TRAIN_DATA_PATH = "../../subtask1/train"
    DEV_DATA_PATH = "../../subtask1/dev"
    OUTPUT_DIR = "./llama-guard-3-8b-polarization"
    PREDICTIONS_DIR = "./predictions"

    # Training hyperparameters
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    WARMUP_RATIO = 0.03

    # LoRA configuration
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05

    # Evaluation
    EVAL_STEPS = 500
    SAVE_STEPS = 500
    LOGGING_STEPS = 50

    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.HF_TOKEN:
            raise ValueError("HF_TOKEN not found in .env file. Please add it.")
        if not Path(cls.TRAIN_DATA_PATH).exists():
            raise ValueError(f"Training data path not found: {cls.TRAIN_DATA_PATH}")
        if not Path(cls.DEV_DATA_PATH).exists():
            raise ValueError(f"Dev data path not found: {cls.DEV_DATA_PATH}")

        # Create output directories
        Path(cls.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Loading and Preparation
# ============================================================================


def load_split(split_dir):
    """Load all CSV files from a directory and combine them."""
    logger.info(f"Loading data from {split_dir}")
    dfs = []
    for file in os.listdir(split_dir):
        if file.endswith(".csv"):
            lang = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(split_dir, file))
            df["lang"] = lang
            dfs.append(df)
            logger.debug(f"Loaded {len(df)} examples for language: {lang}")

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total examples loaded: {len(combined_df)}")
    return combined_df


def format_instruction(text, lang, label=None):
    """Format a single example as an instruction-following conversation."""
    system_prompt = """You are an expert content moderator specializing in detecting polarized content in social media posts.

Polarized content includes:
- Hate speech
- Toxicity
- Misogyny or gender-based violence
- Sarcastic or offensive speech
- Strong us-vs-them divisions
- Extreme opinions that create hostility between groups

Your task is to classify whether the given text contains polarized content. Respond with only 'Yes' or 'No'."""

    user_message = f"""Language: {lang}
Text: {text}

Does this text contain polarized content? Answer with only 'Yes' or 'No'."""

    if label is not None:
        assistant_message = "Yes" if label == 1 else "No"
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        }
    else:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        }


def prepare_dataset(df):
    """Format dataset for instruction fine-tuning."""
    logger.info(f"Preparing dataset with {len(df)} examples")
    formatted_data = []
    for _, row in df.iterrows():
        formatted_data.append(
            format_instruction(row["text"], row["lang"], row["polarization"])
        )
    return Dataset.from_list(formatted_data)


def load_and_prepare_data():
    """Load and prepare all datasets."""
    logger.info("=" * 80)
    logger.info("Loading and preparing data")
    logger.info("=" * 80)

    # Load data
    train_df = load_split(Config.TRAIN_DATA_PATH)
    dev_df = load_split(Config.DEV_DATA_PATH)

    logger.info(f"Train size: {train_df.shape}")
    logger.info(f"Dev size: {dev_df.shape}")
    logger.info(f"Languages: {train_df['lang'].nunique()}")
    logger.info(
        f"Polarization distribution (train):\n{train_df['polarization'].value_counts(normalize=True)}"
    )

    # Create stratified splits
    train_df["lang_label"] = (
        train_df["lang"].astype(str) + "_" + train_df["polarization"].astype(str)
    )

    train_data, temp_data = train_test_split(
        train_df,
        test_size=0.10,
        stratify=train_df["lang_label"],
        random_state=42,
        shuffle=True,
    )

    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        stratify=temp_data["lang_label"],
        random_state=42,
        shuffle=True,
    )

    logger.info(
        f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
    )

    # Prepare datasets
    train_dataset = prepare_dataset(train_data)
    val_dataset = prepare_dataset(val_data)
    test_dataset = prepare_dataset(test_data)

    return train_dataset, val_dataset, test_dataset, test_data, dev_df


# ============================================================================
# Model Loading
# ============================================================================


def load_model_and_tokenizer():
    """Load model and tokenizer with quantization."""
    logger.info("=" * 80)
    logger.info("Loading model and tokenizer")
    logger.info("=" * 80)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {Config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, token=Config.HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info(f"Tokenizer loaded - Vocab size: {len(tokenizer)}")

    # Configure 4-bit quantization
    logger.info("Configuring 4-bit quantization (QLoRA)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    logger.info(f"Loading model: {Config.MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=Config.HF_TOKEN,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    logger.info(f"Model loaded on device: {model.device}")

    return model, tokenizer


def apply_lora(model):
    """Apply LoRA adapters to the model."""
    logger.info("=" * 80)
    logger.info("Applying LoRA configuration")
    logger.info("=" * 80)

    peft_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )
    logger.info(f"Total parameters: {total_params:,}")

    return model


# ============================================================================
# Training
# ============================================================================


def formatting_prompts_func(tokenizer):
    """Create formatting function for the trainer."""

    def format_func(example):
        output_texts = []
        for i in range(len(example["messages"])):
            text = tokenizer.apply_chat_template(
                example["messages"][i], tokenize=False, add_generation_prompt=False
            )
            output_texts.append(text)
        return output_texts

    return format_func


def train_model(model, tokenizer, train_dataset, val_dataset):
    """Train the model."""
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info("=" * 80)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=Config.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=Config.WARMUP_RATIO,
        logging_steps=Config.LOGGING_STEPS,
        save_strategy="steps",
        save_steps=Config.SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=Config.EVAL_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        weight_decay=0.001,
        report_to="none",
        seed=42,
    )

    logger.info("Training configuration:")
    logger.info(f"  Epochs: {Config.NUM_EPOCHS}")
    logger.info(f"  Batch size: {Config.BATCH_SIZE}")
    logger.info(f"  Gradient accumulation: {Config.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(
        f"  Effective batch size: {Config.BATCH_SIZE * Config.GRADIENT_ACCUMULATION_STEPS}"
    )
    logger.info(f"  Learning rate: {Config.LEARNING_RATE}")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func(tokenizer),
        max_seq_length=Config.MAX_SEQ_LENGTH,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    logger.info("Starting training loop...")
    train_result = trainer.train()

    logger.info("Training completed!")
    logger.info(f"Training metrics: {train_result.metrics}")

    return trainer


# ============================================================================
# Evaluation and Inference
# ============================================================================


def predict_polarization(text, lang, model, tokenizer):
    """Predict polarization for a single text."""
    # Format the input
    formatted = format_instruction(text, lang)

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        formatted["messages"], tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract answer
    answer = response.split("assistant")[-1].strip().lower()

    # Parse answer
    if "yes" in answer:
        return 1
    elif "no" in answer:
        return 0
    else:
        logger.warning(f"Unclear response: {answer}")
        return 0


def evaluate_on_test_set(model, tokenizer, test_data, sample_size=500):
    """Evaluate model on test set."""
    logger.info("=" * 80)
    logger.info("Evaluating on test set")
    logger.info("=" * 80)

    # Sample for faster evaluation
    test_sample = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)
    logger.info(f"Evaluating on {len(test_sample)} test examples")

    predictions = []
    true_labels = []

    for idx, row in test_sample.iterrows():
        pred = predict_polarization(row["text"], row["lang"], model, tokenizer)
        predictions.append(pred)
        true_labels.append(row["polarization"])

        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(test_sample)} examples")

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="binary")

    logger.info("=" * 80)
    logger.info("Test Set Evaluation Results")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info("\nClassification Report:")
    logger.info(
        "\n"
        + classification_report(
            true_labels, predictions, target_names=["Not Polarized", "Polarized"]
        )
    )

    # Save results
    results = {"accuracy": accuracy, "f1_score": f1, "num_samples": len(test_sample)}

    results_path = Path(Config.PREDICTIONS_DIR) / "test_results.txt"
    with open(results_path, "w") as f:
        f.write("Test Set Evaluation Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(
            classification_report(
                true_labels, predictions, target_names=["Not Polarized", "Polarized"]
            )
        )

    logger.info(f"Test results saved to {results_path}")

    return results


def generate_dev_predictions(model, tokenizer, dev_df):
    """Generate predictions for the dev set."""
    logger.info("=" * 80)
    logger.info("Generating predictions for dev set")
    logger.info("=" * 80)

    predictions = []

    logger.info(f"Processing {len(dev_df)} dev examples")
    for idx, row in dev_df.iterrows():
        pred = predict_polarization(row["text"], row["lang"], model, tokenizer)
        predictions.append({"id": row["id"], "polarization": pred})

        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(dev_df)} examples")

    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_path = Path(Config.PREDICTIONS_DIR) / "dev_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    logger.info(f"Dev predictions saved to {pred_path}")
    logger.info(
        f"Prediction distribution:\n{pred_df['polarization'].value_counts(normalize=True)}"
    )

    return pred_df


# ============================================================================
# Main
# ============================================================================


def main():
    """Main training pipeline."""
    try:
        # Validate configuration
        logger.info("Validating configuration...")
        Config.validate()

        # Check GPU
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            logger.warning("No GPU detected! Training will be very slow.")

        # Load data
        train_dataset, val_dataset, test_dataset, test_data, dev_df = (
            load_and_prepare_data()
        )

        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()

        # Apply LoRA
        model = apply_lora(model)

        # Train
        trainer = train_model(model, tokenizer, train_dataset, val_dataset)

        # Save model
        logger.info("=" * 80)
        logger.info("Saving model")
        logger.info("=" * 80)
        trainer.model.save_pretrained(Config.OUTPUT_DIR)
        tokenizer.save_pretrained(Config.OUTPUT_DIR)
        logger.info(f"Model saved to {Config.OUTPUT_DIR}")

        # Evaluate on test set
        test_results = evaluate_on_test_set(model, tokenizer, test_data)

        # Generate dev predictions
        dev_predictions = generate_dev_predictions(model, tokenizer, dev_df)

        logger.info("=" * 80)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {Config.OUTPUT_DIR}")
        logger.info(f"Predictions saved to: {Config.PREDICTIONS_DIR}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
