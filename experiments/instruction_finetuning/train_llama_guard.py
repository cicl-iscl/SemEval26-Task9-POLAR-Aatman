"""
Instruction Fine-tuning: Llama-Guard-3-8B for Polarization Detection

This script fine-tunes the meta-llama/Llama-Guard-3-8B model on the polarization
detection dataset using QLoRA (Quantized Low-Rank Adaptation).
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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


@dataclass
class TrainingConfig:
    """Configuration for model training"""

    # Model
    model_name: str = "meta-llama/Llama-Guard-3-8B"
    hf_token: str = field(default_factory=lambda: None)

    # Data paths
    train_data_path: str = None
    dev_data_path: str = None
    output_dir: str = "./llama-guard-3-8b-polarization"
    predictions_dir: str = "./predictions"

    # Training hyperparameters
    max_seq_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.03

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 50
    test_sample_size: int = 500

    def __post_init__(self):
        """Validate configuration after initialization"""
        import os

        if self.hf_token is None:
            self.hf_token = os.getenv("HF_TOKEN")
            if not self.hf_token:
                raise ValueError("HF_TOKEN not found in environment variables")

        if not self.train_data_path:
            raise ValueError("train_data_path is required")
        if not self.dev_data_path:
            raise ValueError("dev_data_path is required")

        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.predictions_dir).mkdir(parents=True, exist_ok=True)


class DataProcessor:
    """Handles data loading and preprocessing"""

    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_split(self, split_dir: str) -> pd.DataFrame:
        """Load all CSV files from a directory and combine them"""
        self.logger.info(f"Loading data from {split_dir}")
        dfs = []
        for file in Path(split_dir).glob("*.csv"):
            lang = file.stem
            df = pd.read_csv(file)
            df["lang"] = lang
            dfs.append(df)
            self.logger.debug(f"Loaded {len(df)} examples for language: {lang}")

        combined_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Total examples loaded: {len(combined_df)}")
        return combined_df

    @staticmethod
    def format_instruction(text: str, lang: str, label: int = None) -> Dict:
        """Format a single example following LlamaGuard's official prompt format"""

        # Define polarization categories in LlamaGuard format
        unsafe_categories = """P1: Hate Speech.
Content that demeans, dehumanizes, or incites hatred against individuals or groups based on protected characteristics such as race, ethnicity, religion, gender, sexual orientation, or other identity markers.

P2: Toxicity.
Content that is insulting, threatening, profane, or otherwise disrespectful in a way that disrupts constructive dialogue or targets individuals or groups with hostility.

P3: Misogyny and Gender-Based Violence.
Content that promotes discrimination, violence, or harm against individuals based on their gender, including but not limited to sexist language, threats, or objectification.

P4: Sarcastic or Offensive Speech.
Content that uses sarcasm, mockery, or offensive language to belittle, ridicule, or demean others in harmful ways.

P5: Polarizing Rhetoric.
Content that creates strong us-vs-them divisions, promotes extreme opinions, or fosters hostility between groups through divisive language or framing.

P6: Implicit Hate Speech
Subtle expressions of prejudice, stereotyping, or derogatory characterizations of groups based on identity, religion, ethnicity, or nationality."""

        # User message in the conversation
        user_message = text

        # Construct the prompt following LlamaGuard format
        # Note: We evaluate the User message, not Agent
        prompt_parts = [
            "Task: Check if there is unsafe content in 'User' messages in conversations according to our safety policy with the below categories.",
            "",
            "<BEGIN UNSAFE CONTENT CATEGORIES>",
            unsafe_categories,
            "<END UNSAFE CONTENT CATEGORIES>",
            "",
            "<BEGIN CONVERSATION>",
            "",
            f"User: {user_message}",
            "",
            "<END CONVERSATION>",
            "",
            "Provide your safety assessment for ONLY THE LAST User in the above conversation:",
            " - First line must read 'safe' or 'unsafe'.",
            " - If unsafe, a second line must include a comma-separated list of violated categories.",
        ]

        user_prompt = "\n".join(prompt_parts)

        if label is not None:
            # Generate the assistant response in LlamaGuard format
            if label == 1:  # Polarized
                assistant_message = "unsafe\nP1,P2,P3,P4,P5,P6"
            else:  # Not polarized
                assistant_message = "safe"

            return {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_message},
                ]
            }
        else:
            return {
                "messages": [
                    {"role": "user", "content": user_prompt},
                ]
            }

    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """Format dataset for instruction fine-tuning"""
        self.logger.info(f"Preparing dataset with {len(df)} examples")
        formatted_data = [
            self.format_instruction(row["text"], row["lang"], row["polarization"])
            for _, row in df.iterrows()
        ]
        return Dataset.from_list(formatted_data)

    def load_and_prepare_data(
        self,
    ) -> Tuple[Dataset, Dataset, Dataset, pd.DataFrame, pd.DataFrame]:
        """Load and prepare all datasets with 90%-5%-5% split"""
        self.logger.info("=" * 80)
        self.logger.info("Loading and preparing data")
        self.logger.info("=" * 80)

        # Load data
        train_df = self.load_split(self.config.train_data_path)
        dev_df = self.load_split(self.config.dev_data_path)

        self.logger.info(f"Train size: {train_df.shape}")
        self.logger.info(f"Dev size: {dev_df.shape}")
        self.logger.info(f"Languages: {train_df['lang'].nunique()}")
        self.logger.info(
            f"Polarization distribution (train):\n{train_df['polarization'].value_counts(normalize=True)}"
        )

        # Create stratified 90%-5%-5% splits
        train_df["lang_label"] = (
            train_df["lang"].astype(str) + "_" + train_df["polarization"].astype(str)
        )

        # First split: 90% train, 10% temp
        train_data, temp_data = train_test_split(
            train_df,
            test_size=0.10,
            stratify=train_df["lang_label"],
            random_state=42,
            shuffle=True,
        )

        # Second split: 5% validation, 5% test (50-50 split of the 10% temp)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.50,
            stratify=temp_data["lang_label"],
            random_state=42,
            shuffle=True,
        )

        self.logger.info("=" * 80)
        self.logger.info("Data Split Information")
        self.logger.info("=" * 80)
        self.logger.info(
            f"Train set: {len(train_data)} samples ({len(train_data) / len(train_df) * 100:.1f}%)"
        )
        self.logger.info(
            f"Validation set: {len(val_data)} samples ({len(val_data) / len(train_df) * 100:.1f}%)"
        )
        self.logger.info(
            f"Test set: {len(test_data)} samples ({len(test_data) / len(train_df) * 100:.1f}%)"
        )
        self.logger.info(f"Dev set (pure test): {len(dev_df)} samples")

        # Log language distribution for each split
        self.logger.info("\nLanguage distribution:")
        self.logger.info(f"Train: {dict(train_data['lang'].value_counts())}")
        self.logger.info(f"Val: {dict(val_data['lang'].value_counts())}")
        self.logger.info(f"Test: {dict(test_data['lang'].value_counts())}")

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_data)
        val_dataset = self.prepare_dataset(val_data)
        test_dataset = self.prepare_dataset(test_data)

        return train_dataset, val_dataset, test_dataset, test_data, dev_df


class ModelManager:
    """Handles model loading and configuration"""

    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer"""
        self.logger.info(f"Loading tokenizer: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, token=self.config.hf_token
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.logger.info(f"Tokenizer loaded - Vocab size: {len(tokenizer)}")
        return tokenizer

    def load_model(self) -> AutoModelForCausalLM:
        """Load model with quantization"""
        self.logger.info("=" * 80)
        self.logger.info("Loading model with quantization")
        self.logger.info("=" * 80)

        # Configure 4-bit quantization
        self.logger.info("Configuring 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load model
        self.logger.info(f"Loading model: {self.config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=self.config.hf_token,
        )

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        self.logger.info(f"Model loaded on device: {model.device}")

        return model

    def apply_lora(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """Apply LoRA adapters to the model"""
        self.logger.info("=" * 80)
        self.logger.info("Applying LoRA configuration")
        self.logger.info("=" * 80)

        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
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
        self.logger.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        self.logger.info(f"Total parameters: {total_params:,}")

        return model


class Trainer:
    """Handles model training"""

    def __init__(
        self,
        config: TrainingConfig,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        logger: logging.Logger,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger

    def formatting_prompts_func(self, example: Dict) -> List[str]:
        """Format examples using the chat template"""
        output_texts = []
        for i in range(len(example["messages"])):
            text = self.tokenizer.apply_chat_template(
                example["messages"][i], tokenize=False, add_generation_prompt=False
            )
            output_texts.append(text)
        return output_texts

    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> SFTTrainer:
        """Train the model"""
        self.logger.info("=" * 80)
        self.logger.info("Starting training")
        self.logger.info("=" * 80)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
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

        self.logger.info("Training configuration:")
        self.logger.info(f"  Epochs: {self.config.num_epochs}")
        self.logger.info(f"  Batch size: {self.config.batch_size}")
        self.logger.info(
            f"  Gradient accumulation: {self.config.gradient_accumulation_steps}"
        )
        self.logger.info(
            f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}"
        )
        self.logger.info(f"  Learning rate: {self.config.learning_rate}")
        self.logger.info(f"  Evaluation steps: {self.config.eval_steps}")
        self.logger.info("  Early stopping patience: 3")

        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # tokenizer=self.tokenizer,
            processing_class=self.tokenizer,
            formatting_func=self.formatting_prompts_func,
            max_length=self.config.max_seq_length,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train
        self.logger.info("Starting training loop...")
        train_result = trainer.train()

        self.logger.info("Training completed!")
        self.logger.info(f"Training metrics: {train_result.metrics}")

        return trainer


class Evaluator:
    """Handles model evaluation and inference"""

    def __init__(
        self,
        config: TrainingConfig,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        logger: logging.Logger,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger

    def predict_polarization(self, text: str, lang: str) -> int:
        """Predict polarization for a single text"""
        # Format the input
        formatted = DataProcessor.format_instruction(text, lang)

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            formatted["messages"], tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer (look for the last assistant response)
        answer = response.split("assistant")[-1].strip().lower()

        # Parse answer - LlamaGuard format is "safe" or "unsafe"
        if "unsafe" in answer:
            return 1  # Polarized
        elif "safe" in answer:
            return 0  # Not polarized
        else:
            self.logger.warning(f"Unclear response: {answer}")
            return 0  # Default to safe

    def evaluate_on_test_set(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate model on test set"""
        self.logger.info("=" * 80)
        self.logger.info("Evaluating on test set")
        self.logger.info("=" * 80)

        # Use entire test set (it's already 5% of data)
        self.logger.info(f"Evaluating on {len(test_data)} test examples")

        predictions = []
        true_labels = []

        for idx, row in test_data.iterrows():
            pred = self.predict_polarization(row["text"], row["lang"])
            predictions.append(pred)
            true_labels.append(row["polarization"])

            if (idx + 1) % 50 == 0:
                self.logger.info(f"Processed {idx + 1}/{len(test_data)} examples")

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average="binary")

        self.logger.info("=" * 80)
        self.logger.info("Test Set Evaluation Results")
        self.logger.info("=" * 80)
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        self.logger.info("\nClassification Report:")
        self.logger.info(
            "\n"
            + classification_report(
                true_labels, predictions, target_names=["Not Polarized", "Polarized"]
            )
        )

        # Save results
        results_path = Path(self.config.predictions_dir) / "test_results.txt"
        with open(results_path, "w") as f:
            f.write("Test Set Evaluation Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(
                classification_report(
                    true_labels,
                    predictions,
                    target_names=["Not Polarized", "Polarized"],
                )
            )

        self.logger.info(f"Test results saved to {results_path}")

        return {"accuracy": accuracy, "f1_score": f1, "num_samples": len(test_data)}

    def generate_dev_predictions(self, dev_df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for the dev set"""
        self.logger.info("=" * 80)
        self.logger.info("Generating predictions for dev set")
        self.logger.info("=" * 80)

        predictions = []

        self.logger.info(f"Processing {len(dev_df)} dev examples")
        for idx, row in dev_df.iterrows():
            pred = self.predict_polarization(row["text"], row["lang"])
            predictions.append({"id": row["id"], "polarization": pred})

            if (idx + 1) % 100 == 0:
                self.logger.info(f"Processed {idx + 1}/{len(dev_df)} examples")

        # Save predictions
        pred_df = pd.DataFrame(predictions)
        pred_path = Path(self.config.predictions_dir) / "dev_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        self.logger.info(f"Dev predictions saved to {pred_path}")
        self.logger.info(
            f"Prediction distribution:\n{pred_df['polarization'].value_counts(normalize=True)}"
        )

        return pred_df


def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-Guard-3-8B for polarization detection"
    )

    # Data paths
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to training data directory",
    )
    parser.add_argument(
        "--dev_data_path", type=str, required=True, help="Path to dev data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./llama-guard-3-8b-polarization",
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        default="./predictions",
        help="Directory to save predictions",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )

    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    return parser.parse_args()


def main():
    """Main training pipeline"""
    # Setup
    logger = setup_logging()
    args = parse_args()

    try:
        # Create configuration
        config = TrainingConfig(
            train_data_path=args.train_data_path,
            dev_data_path=args.dev_data_path,
            output_dir=args.output_dir,
            predictions_dir=args.predictions_dir,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            max_seq_length=args.max_seq_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        # Check GPU
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            logger.warning("No GPU detected! Training will be very slow.")

        # Load and prepare data
        data_processor = DataProcessor(config, logger)
        train_dataset, val_dataset, test_dataset, test_data, dev_df = (
            data_processor.load_and_prepare_data()
        )

        # Load model and tokenizer
        model_manager = ModelManager(config, logger)
        tokenizer = model_manager.load_tokenizer()
        model = model_manager.load_model()
        model = model_manager.apply_lora(model)

        # Train
        trainer_obj = Trainer(config, model, tokenizer, logger)
        trainer = trainer_obj.train(train_dataset, val_dataset)

        # Save model
        logger.info("=" * 80)
        logger.info("Saving model")
        logger.info("=" * 80)
        trainer.model.save_pretrained(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        logger.info(f"Model saved to {config.output_dir}")

        # Evaluate
        evaluator = Evaluator(config, model, tokenizer, logger)
        evaluator.evaluate_on_test_set(test_data)
        evaluator.generate_dev_predictions(dev_df)

        logger.info("=" * 80)
        logger.info("Training pipeline completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {config.output_dir}")
        logger.info(f"Predictions saved to: {config.predictions_dir}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
