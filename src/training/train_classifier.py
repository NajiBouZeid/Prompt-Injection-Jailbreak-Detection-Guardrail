import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512


def load_split(name: str, max_examples: int | None = None) -> Dataset:
    """Load a train/val/test parquet split as a HuggingFace Dataset with only
    the columns the model needs (text, label). max_examples truncates the
    split for quick smoke tests - leave None for real training runs."""
    df = pd.read_parquet(f"data/processed/{name}.parquet")[["text", "label"]]
    if max_examples is not None:
        df = df.iloc[:max_examples]
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize_split(dataset: Dataset, tokenizer) -> Dataset:
    """Tokenize text with head truncation (default HF behavior for a single
    sequence: keep the first MAX_LENGTH tokens, drop the rest). No fixed
    padding here - DataCollatorWithPadding pads dynamically per batch instead,
    since most texts are far shorter than MAX_LENGTH."""
    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    return dataset.map(tokenize_batch, batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train(
    output_dir: str = "models/deberta-v3-base-guardrail",
    train_split: str = "train",
    val_split: str = "val",
    num_train_epochs: float = 2,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    per_device_eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_examples: int | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # deberta-v3-base's checkpoint is stored in fp16, and its disentangled
    # attention (XSoftmax) is known to produce NaN logits under pure fp16.
    # Load master weights in fp32; bf16=True below handles mixed precision
    # safely during the forward/backward pass instead.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, dtype=torch.float32
    )

    train_dataset = tokenize_split(load_split(train_split, max_examples), tokenizer)
    val_dataset = tokenize_split(load_split(val_split, max_examples), tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        bf16=True,
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return trainer
