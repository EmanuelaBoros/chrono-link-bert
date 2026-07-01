from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def normalize_label(label: str) -> str:
    label = label.strip()
    if label == "_" or label == "":
        return "O"
    return label


def read_hipe_tsv(
    path: str | Path,
    token_column: str = "TOKEN",
    label_column: str = "NE-COARSE-LIT",
    misc_column: str = "MISC",
) -> list[dict[str, Any]]:
    path = Path(path)

    examples = []
    current_tokens = []
    current_labels = []

    current_doc_id = None
    current_date = None

    header = None
    col2idx = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line:
                continue

            if line.startswith("#"):
                if line.startswith("# hipe2022:document_id"):
                    current_doc_id = line.split("=", 1)[-1].strip()
                elif line.startswith("# hipe2022:date"):
                    current_date = line.split("=", 1)[-1].strip()
                continue

            parts = line.split("\t")

            if header is None:
                header = parts
                col2idx = {name: i for i, name in enumerate(header)}

                missing = [
                    col
                    for col in [token_column, label_column, misc_column]
                    if col not in col2idx
                ]
                if missing:
                    raise ValueError(
                        f"Missing columns in {path}: {missing}. "
                        f"Available columns: {header}"
                    )
                continue

            if len(parts) < len(header):
                continue

            token = parts[col2idx[token_column]]
            label = normalize_label(parts[col2idx[label_column]])
            misc = parts[col2idx[misc_column]]

            current_tokens.append(token)
            current_labels.append(label)

            if "EndOfSentence" in misc:
                if current_tokens:
                    examples.append(
                        {
                            "tokens": current_tokens,
                            "ner_tags_str": current_labels,
                            "document_id": current_doc_id,
                            "date": current_date,
                        }
                    )
                current_tokens = []
                current_labels = []

    if current_tokens:
        examples.append(
            {
                "tokens": current_tokens,
                "ner_tags_str": current_labels,
                "document_id": current_doc_id,
                "date": current_date,
            }
        )

    return examples


def build_label_list(*splits: list[dict[str, Any]]) -> list[str]:
    labels = set()

    for split in splits:
        for ex in split:
            labels.update(ex["ner_tags_str"])

    # Keep O first, then sorted BIO labels.
    other = sorted(label for label in labels if label != "O")
    return ["O"] + other


def encode_labels(
    examples: list[dict[str, Any]],
    label2id: dict[str, int],
) -> list[dict[str, Any]]:
    encoded = []

    for ex in examples:
        new_ex = dict(ex)
        new_ex["ner_tags"] = [label2id[label] for label in ex["ner_tags_str"]]
        encoded.append(new_ex)

    return encoded


def tokenize_and_align_labels(
    examples: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    label_all_tokens: bool = False,
) -> dict[str, Any]:
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    aligned_labels = []

    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(labels[word_id])
            else:
                if label_all_tokens:
                    label_ids.append(labels[word_id])
                else:
                    label_ids.append(-100)

            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def compute_metrics_builder(id2label: dict[int, str]):
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for pred_row, label_row in zip(predictions, labels):
            current_preds = []
            current_labels = []

            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue

                current_preds.append(id2label[int(pred_id)])
                current_labels.append(id2label[int(label_id)])

            true_predictions.append(current_preds)
            true_labels.append(current_labels)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", required=True)
    parser.add_argument("--test_file", default=None)

    parser.add_argument("--output_dir", default="chrono-link-bert-ner")
    parser.add_argument("--hub_model_id", default=None)

    parser.add_argument("--token_column", default="TOKEN")
    parser.add_argument("--label_column", default="NE-COARSE-LIT")
    parser.add_argument("--misc_column", default="MISC")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_token", default=None)

    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    print("Reading TSV files...")

    train_examples = read_hipe_tsv(
        args.train_file,
        token_column=args.token_column,
        label_column=args.label_column,
        misc_column=args.misc_column,
    )

    valid_examples = read_hipe_tsv(
        args.validation_file,
        token_column=args.token_column,
        label_column=args.label_column,
        misc_column=args.misc_column,
    )

    test_examples = []
    if args.test_file:
        test_examples = read_hipe_tsv(
            args.test_file,
            token_column=args.token_column,
            label_column=args.label_column,
            misc_column=args.misc_column,
        )

    label_list = build_label_list(train_examples, valid_examples, test_examples)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    print("Labels:")
    for i, label in id2label.items():
        print(i, label)

    train_examples = encode_labels(train_examples, label2id)
    valid_examples = encode_labels(valid_examples, label2id)
    if test_examples:
        test_examples = encode_labels(test_examples, label2id)

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_examples),
            "validation": Dataset.from_list(valid_examples),
        }
    )

    if test_examples:
        dataset["test"] = Dataset.from_list(test_examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tokenized = dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples=examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(id2label),
    )

    trainer.train()

    print("Validation evaluation:")
    print(trainer.evaluate(tokenized["validation"]))

    if "test" in tokenized:
        print("Test evaluation:")
        print(trainer.evaluate(tokenized["test"]))

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
