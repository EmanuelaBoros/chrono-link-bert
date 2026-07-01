from __future__ import annotations

import argparse
from typing import Any

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def get_label_names(dataset, tags_column: str) -> list[str]:
    features = dataset["train"].features[tags_column]

    # Common HF token-classification format:
    # Sequence(ClassLabel)
    if hasattr(features, "feature") and hasattr(features.feature, "names"):
        return features.feature.names

    # Fallback: infer numeric labels.
    label_ids = set()
    for row in dataset["train"]:
        label_ids.update(row[tags_column])
    max_id = max(label_ids)
    return [str(i) for i in range(max_id + 1)]


def tokenize_and_align_labels(
    examples: dict[str, Any],
    tokenizer: Any,
    tokens_column: str,
    tags_column: str,
    fact_column: str,
    max_length: int,
) -> dict[str, Any]:
    """
    Input format:
    [tokens of original sentence] [SEP] fact_context

    Labels:
    - original tokens get NER labels
    - [SEP] and fact_context tokens get -100
    """
    all_tokens = examples[tokens_column]
    all_tags = examples[tags_column]
    all_facts = examples[fact_column]

    batch_input_tokens = []
    batch_labels = []

    sep_token = tokenizer.sep_token or "[SEP]"

    for tokens, tags, fact in zip(all_tokens, all_tags, all_facts):
        fact_words = str(fact).split()

        combined_tokens = list(tokens) + [sep_token] + fact_words
        combined_labels = list(tags) + [-100] + [-100] * len(fact_words)

        batch_input_tokens.append(combined_tokens)
        batch_labels.append(combined_labels)

    tokenized = tokenizer(
        batch_input_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    aligned_labels = []

    for i, labels in enumerate(batch_labels):
        word_ids = tokenized.word_ids(batch_index=i)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                if word_id < len(labels):
                    label_ids.append(labels[word_id])
                else:
                    label_ids.append(-100)
            else:
                # For subword continuation, ignore by default.
                label_ids.append(-100)

            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def compute_metrics_builder(label_names: list[str]):
    seqeval = evaluate.load("seqeval")

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

                current_preds.append(label_names[pred_id])
                current_labels.append(label_names[label_id])

            true_predictions.append(current_preds)
            true_labels.append(current_labels)

        results = seqeval.compute(
            predictions=true_predictions,
            references=true_labels,
        )

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_name", default="bert-base-multilingual-cased")
    parser.add_argument("--output_dir", default="wikitemporal-bert-ner")

    parser.add_argument("--tokens_column", default="tokens")
    parser.add_argument("--tags_column", default="ner_tags")
    parser.add_argument("--fact_column", default="fact_context")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)

    label_names = get_label_names(dataset, args.tags_column)
    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {label: i for i, label in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenized = dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples=examples,
            tokenizer=tokenizer,
            tokens_column=args.tokens_column,
            tags_column=args.tags_column,
            fact_column=args.fact_column,
            max_length=args.max_length,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(label_names),
    )

    trainer.train()

    if "test" in tokenized:
        print("Evaluating on test...")
        print(trainer.evaluate(tokenized["test"]))

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
