from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_name", default="bert-base-cased")
    parser.add_argument("--output_dir", default="chrono-link-bert")
    parser.add_argument("--hub_model_id", default=None)

    parser.add_argument("--text_column", default="text")
    parser.add_argument("--fact_column", default="fact_context")

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
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

    dataset = load_dataset(args.dataset_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def combine_text_and_facts(examples):
        combined = []
        sep = tokenizer.sep_token or "[SEP]"

        for text, facts in zip(examples[args.text_column], examples[args.fact_column]):
            text = str(text)
            facts = str(facts)
            combined.append(f"{text} {sep} {facts}")

        return {"combined_text": combined}

    dataset = dataset.map(
        combine_text_and_facts,
        batched=True,
    )

    def tokenize(examples):
        return tokenizer(
            examples["combined_text"],
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=True,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
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
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    if "test" in tokenized:
        print("Evaluating on test...")
        print(trainer.evaluate(tokenized["test"]))

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
