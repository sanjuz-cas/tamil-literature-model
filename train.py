import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed as xd
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from optimum.tpu import fsdp_v2  # <<< WE ARE USING THIS AGAIN
import os

# --- 1. (CRITICAL) SET YOUR BUCKET HERE ---
MY_BUCKET_NAME = "gs://sanjay-trc-bucket-2025"
PROJECT_NAME = "tamil-model"
CHECKPOINT_DIR = os.path.join(MY_BUCKET_NAME, PROJECT_NAME, "checkpoints")

# --- 2. CHOOSE YOUR BASE MODEL ---
MODEL_NAME = "xlm-roberta-base"


def setup_tpu_training():
    print("Setting up FSDPv2 for TPU sharding...")
    fsdp_v2.use_fsdp_v2()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    fsdp_training_args = fsdp_v2.get_fsdp_training_args(model)
    return model, tokenizer, fsdp_training_args


def prepare_dataset(tokenizer):
    print("Loading and preparing dataset...")
    dataset = load_dataset("thamizhi/thirukkural", split="train")
    dataset = dataset.shuffle(seed=42).select(range(2000))

    def tokenize_function(examples):
        return tokenizer(examples["Kural"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )
    return tokenized_datasets


def main():
    model, tokenizer, fsdp_args = setup_tpu_training()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        dataloader_drop_last=True,
        **fsdp_args,  # <<< THIS IS THE OPTIMUM CODE
    )

    tokenized_data = prepare_dataset(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        data_collator=data_collator,
    )

    print(f"--- Starting Fine-Tuning for {PROJECT_NAME} ---")
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")

    trainer.train(resume_from_checkpoint=True)

    print("--- Fine-Tuning Complete ---")
    print("Saving final model...")
    trainer.save_model(os.path.join(CHECKPOINT_DIR, "final-model"))
    print(f"All model checkpoints are safe in your GCS bucket!")


if __name__ == "__main__":
    main()
