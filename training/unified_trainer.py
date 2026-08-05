"""Unified Fine-Tuning Module for DSS Restoration Models.

Provides a clean, single trainer for fine-tuning character-level MLMs (dictabert-char, TavBERT)
and WordPiece MLMs (MsBERT) on the canonical frozen scroll splits.
"""

import argparse
import logging
from pathlib import Path
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import numpy as np
from utils.splits import load_frozen_scroll_splits, get_scroll_sets
from utils.tokenizer_compat import load_tokenizer

logger = logging.getLogger(__name__)


def choose_word_span(
    text: str,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    """Select a contiguous 1-3 word span within text for span-corruption training."""
    words = text.split()
    max_words = min(3, len(words))
    probabilities = np.array([0.5, 0.3, 0.2][:max_words], dtype=float)
    probabilities /= probabilities.sum()
    word_count = int(rng.choice(np.arange(1, max_words + 1), p=probabilities))
    start_word = int(rng.integers(0, len(words) - word_count + 1))
    prefix = " ".join(words[:start_word])
    target = " ".join(words[start_word : start_word + word_count])
    start_character = len(prefix) + (1 if prefix else 0)
    return start_character, start_character + len(target), word_count



def train_dss_model(
    model_name_or_path: str = "dicta-il/dictabert-char",
    output_dir: str = "models/ft_dictabert_char",
    num_train_epochs: int = 5,
    per_device_train_batch_size: int = 8,
    learning_rate: float = 5e-5,
    seed: int = 42,
):
    """Fine-tune a language model on the canonical DSS train partition."""
    logger.info(f"Loading tokenizer and model for {model_name_or_path}...")
    tokenizer = load_tokenizer(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)

    # Load frozen scroll splits
    scroll_sets = get_scroll_sets()
    train_scrolls = scroll_sets["train"]
    val_scrolls = scroll_sets["val"]

    logger.info(
        f"Partition loaded: {len(train_scrolls)} train scrolls, "
        f"{len(val_scrolls)} val scrolls."
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        seed=seed,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
    )

    logger.info("Ready for fine-tuning execution.")
    return model, tokenizer, training_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified DSS Fine-Tuning CLI")
    parser.add_argument("--model", type=str, default="dicta-il/dictabert-char")
    parser.add_argument("--output_dir", type=str, default="models/ft_dictabert_char")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dss_model(
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
