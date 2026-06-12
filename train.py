"""
Task-01: Fine-tune GPT-2 on a custom dataset
Prodigy Infotech Internship
"""

import argparse
import os
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_dataset(file_path, tokenizer, block_size=128):
    """Load and tokenize the text dataset."""
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset


def get_data_collator(tokenizer):
    """Create a data collator for language modeling."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 uses causal LM, not masked LM
    )


def fine_tune(dataset_path, output_dir, num_epochs=3, batch_size=4, lr=5e-5):
    """Fine-tune GPT-2 on the custom dataset."""

    print("🔄 Loading GPT-2 tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    print(f"📂 Loading dataset from: {dataset_path}")
    train_dataset = load_dataset(dataset_path, tokenizer)
    data_collator = get_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=100,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("🚀 Starting fine-tuning...")
    trainer.train()

    print(f"💾 Saving model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("✅ Fine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on a custom dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to training text file")
    parser.add_argument("--output", type=str, default="./gpt2-finetuned", help="Output directory for the model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")

    args = parser.parse_args()

    fine_tune(
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
