import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import evaluate

# ============== CONFIG ==============
MODEL_NAME = "google/gemma-2b"
PROJECT_NAME = "tamil-gemma-2b"
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "tamil-llm")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
# ====================================

if WANDB_API_KEY:
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

perplexity_metric = evaluate.load("perplexity", module_type="metric")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return perplexity_metric.compute(predictions=logits, model_id=MODEL_NAME)


def prepare_dataset(tokenizer):
    dataset = load_dataset("thamizhi/thirukkural", split="train").shuffle(seed=42)

    def tokenize_fn(examples):
        return tokenizer(examples["Kural"])

    tokenized = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names
    )
    split = tokenized.train_test_split(test_size=0.1)
    return split["train"], split["test"]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_ds, eval_ds = prepare_dataset(tokenizer)

    args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        report_to="wandb" if WANDB_API_KEY else "none",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_drop_last=True,
        fp16=True,
        gradient_checkpointing=True,
        resume_from_checkpoint=True if os.path.exists(CHECKPOINT_DIR) else None,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=os.path.exists(CHECKPOINT_DIR))
    trainer.save_model(os.path.join(CHECKPOINT_DIR, "final-model"))
    print("✅ Training complete. Model saved to", CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
