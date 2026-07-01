from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

DEFAULT_MODELS = [
    "emanuelaboros/chrono-link-bert-americanstories-1850-1890",
    "Livingwithmachines/bert_1760_1850",
    "Livingwithmachines/bert_1850_1875",
    "Livingwithmachines/bert_1875_1890",
    "Livingwithmachines/bert_1890_1900",
    "Livingwithmachines/bert_1760_1900",
]


@dataclass
class Example:
    text: str
    fact_context: str
    year: int


def parse_year(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"(\d{4})", str(value))
    if not match:
        return None
    return int(match.group(1))


def replace_document_year(fact_context: str, wrong_year: int) -> str:
    """
    Replace 'Document year: 1850.' with a wrong year.
    If the pattern is not found, prepend it.
    """
    pattern = r"Document year:\s*\d{4}\."

    if re.search(pattern, fact_context):
        return re.sub(pattern, f"Document year: {wrong_year}.", fact_context)

    return f"Document year: {wrong_year}. {fact_context}"


def choose_wrong_year(
    true_year: int, candidate_years: list[int], min_distance: int = 20
) -> int:
    far_years = [y for y in candidate_years if abs(y - true_year) >= min_distance]

    if far_years:
        return random.choice(far_years)

    return random.choice([y for y in candidate_years if y != true_year])


def build_input(
    text: str,
    fact_context: str,
    tokenizer: Any,
    use_fact_context: bool = True,
) -> str:
    sep = tokenizer.sep_token or "[SEP]"

    if use_fact_context:
        return f"{text} {sep} {fact_context}"

    return text


def load_examples(
    dataset_name: str,
    split: str,
    text_column: str,
    fact_column: str,
    year_column: str,
    max_examples: int,
    seed: int,
) -> list[Example]:
    dataset = load_dataset(dataset_name, split=split)

    if max_examples and max_examples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_examples))

    examples = []

    for row in dataset:
        year = parse_year(row.get(year_column))
        text = str(row.get(text_column, "")).strip()
        fact_context = str(row.get(fact_column, "")).strip()

        if not text or year is None:
            continue

        examples.append(
            Example(
                text=text,
                fact_context=fact_context,
                year=year,
            )
        )

    return examples


def get_scored_positions(
    input_ids: torch.Tensor,
    tokenizer: Any,
    max_scored_tokens: int,
    seed: int,
) -> list[int]:
    """
    Select non-special token positions to score.
    We do pseudo-log-likelihood by masking one token at a time.
    """
    special_ids = set(tokenizer.all_special_ids)
    positions = []

    ids = input_ids.tolist()

    for i, tok_id in enumerate(ids):
        if tok_id in special_ids:
            continue
        positions.append(i)

    if max_scored_tokens and len(positions) > max_scored_tokens:
        rng = random.Random(seed)
        positions = rng.sample(positions, max_scored_tokens)
        positions = sorted(positions)

    return positions


def pseudo_log_likelihood(
    text: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int = 256,
    max_scored_tokens: int = 64,
    batch_size: int = 16,
    seed: int = 13,
) -> float:
    """
    Compute approximate pseudo-log-likelihood for a masked language model.

    Higher = better.
    """
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]

    positions = get_scored_positions(
        input_ids=input_ids,
        tokenizer=tokenizer,
        max_scored_tokens=max_scored_tokens,
        seed=seed,
    )

    if not positions:
        return float("-inf")

    log_probs = []

    model.eval()

    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]

        batch_input_ids = input_ids.unsqueeze(0).repeat(len(batch_positions), 1)
        batch_attention_mask = attention_mask.unsqueeze(0).repeat(
            len(batch_positions), 1
        )

        labels = []

        for row_idx, pos in enumerate(batch_positions):
            original_token_id = int(batch_input_ids[row_idx, pos].item())
            labels.append(original_token_id)
            batch_input_ids[row_idx, pos] = tokenizer.mask_token_id

        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )

        logits = outputs.logits

        for row_idx, pos in enumerate(batch_positions):
            token_logits = logits[row_idx, pos]
            token_log_probs = torch.log_softmax(token_logits, dim=-1)
            gold_id = labels[row_idx]
            log_probs.append(float(token_log_probs[gold_id].detach().cpu()))

    return float(np.mean(log_probs))


def evaluate_temporal_context_preference(
    model_name: str,
    examples: list[Example],
    candidate_years: list[int],
    max_length: int,
    max_scored_tokens: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    use_fact_context: bool = True,
) -> dict[str, Any]:
    """
    Main Chrono-Link-BERT evaluation.

    For every example:
      correct = text + correct fact_context
      wrong   = text + fact_context with wrong document year

    A temporally sensitive model should prefer correct > wrong.
    """
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)

    deltas = []
    correct_wins = 0
    evaluated = 0

    for idx, ex in enumerate(tqdm(examples, desc=f"Temporal preference: {model_name}")):
        wrong_year = choose_wrong_year(
            true_year=ex.year,
            candidate_years=candidate_years,
            min_distance=20,
        )

        wrong_fact_context = replace_document_year(
            fact_context=ex.fact_context,
            wrong_year=wrong_year,
        )

        correct_input = build_input(
            text=ex.text,
            fact_context=ex.fact_context,
            tokenizer=tokenizer,
            use_fact_context=use_fact_context,
        )

        wrong_input = build_input(
            text=ex.text,
            fact_context=wrong_fact_context,
            tokenizer=tokenizer,
            use_fact_context=use_fact_context,
        )

        correct_score = pseudo_log_likelihood(
            text=correct_input,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
            max_scored_tokens=max_scored_tokens,
            batch_size=batch_size,
            seed=seed + idx,
        )

        wrong_score = pseudo_log_likelihood(
            text=wrong_input,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=max_length,
            max_scored_tokens=max_scored_tokens,
            batch_size=batch_size,
            seed=seed + idx + 10000,
        )

        if not math.isfinite(correct_score) or not math.isfinite(wrong_score):
            continue

        delta = correct_score - wrong_score
        deltas.append(delta)

        if delta > 0:
            correct_wins += 1

        evaluated += 1

    if evaluated == 0:
        return {
            "model": model_name,
            "temporal_preference_accuracy": None,
            "mean_delta_correct_minus_wrong": None,
            "median_delta_correct_minus_wrong": None,
            "num_examples": 0,
        }

    return {
        "model": model_name,
        "temporal_preference_accuracy": correct_wins / evaluated,
        "mean_delta_correct_minus_wrong": float(np.mean(deltas)),
        "median_delta_correct_minus_wrong": float(np.median(deltas)),
        "num_examples": evaluated,
    }


def infer_period_from_model_name(model_name: str) -> tuple[int, int] | None:
    """
    Extract period from names like:
    Livingwithmachines/bert_1850_1875
    """
    match = re.search(r"bert_(\d{4})_(\d{4})", model_name)

    if not match:
        return None

    return int(match.group(1)), int(match.group(2))


def evaluate_period_model_selection(
    model_names: list[str],
    examples: list[Example],
    max_length: int,
    max_scored_tokens: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    """
    Evaluation specifically for Livingwithmachines period BERTs.

    For each article, score raw text under each period model.
    The best-scoring model should ideally correspond to the article's year.
    """
    period_models = []

    for name in model_names:
        period = infer_period_from_model_name(name)
        if period is not None:
            period_models.append((name, period))

    if not period_models:
        return pd.DataFrame()

    loaded = []

    for name, period in period_models:
        print(f"Loading period model: {name}")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForMaskedLM.from_pretrained(name).to(device)
        model.eval()
        loaded.append((name, period, tokenizer, model))

    rows = []

    for idx, ex in enumerate(tqdm(examples, desc="Period model selection")):
        scores = []

        for name, period, tokenizer, model in loaded:
            score = pseudo_log_likelihood(
                text=ex.text,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
                max_scored_tokens=max_scored_tokens,
                batch_size=batch_size,
                seed=seed + idx,
            )

            scores.append((name, period, score))

        best_name, best_period, best_score = max(scores, key=lambda x: x[2])

        true_matching_models = [
            name for name, (start, end), score in scores if start <= ex.year <= end
        ]

        is_correct_period = best_name in true_matching_models

        rows.append(
            {
                "year": ex.year,
                "best_model": best_name,
                "best_period": f"{best_period[0]}-{best_period[1]}",
                "best_score": best_score,
                "true_matching_models": ";".join(true_matching_models),
                "correct_period": is_correct_period,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        default="emanuelaboros/americanstories-1850-1890",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--fact_column", default="fact_context")
    parser.add_argument("--year_column", default="year")

    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )

    parser.add_argument("--max_examples", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_scored_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)

    parser.add_argument("--output_csv", default="temporal_comparison_results.csv")
    parser.add_argument(
        "--period_output_csv", default="period_model_selection_results.csv"
    )

    parser.add_argument(
        "--skip_period_selection",
        action="store_true",
        help="Skip the Livingwithmachines period-model dating evaluation.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading examples from {args.dataset_name}/{args.split}")
    examples = load_examples(
        dataset_name=args.dataset_name,
        split=args.split,
        text_column=args.text_column,
        fact_column=args.fact_column,
        year_column=args.year_column,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    print(f"Loaded {len(examples)} examples.")

    candidate_years = sorted(set(ex.year for ex in examples))

    results = []

    for model_name in args.models:
        try:
            result = evaluate_temporal_context_preference(
                model_name=model_name,
                examples=examples,
                candidate_years=candidate_years,
                max_length=args.max_length,
                max_scored_tokens=args.max_scored_tokens,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed,
                use_fact_context=True,
            )
            results.append(result)
        except Exception as e:
            print(f"Skipping model {model_name} because of error: {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="temporal_preference_accuracy",
        ascending=False,
        na_position="last",
    )

    print("\nTemporal context preference results")
    print(results_df)

    results_df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")

    if not args.skip_period_selection:
        period_df = evaluate_period_model_selection(
            model_names=args.models,
            examples=examples,
            max_length=args.max_length,
            max_scored_tokens=args.max_scored_tokens,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )

        if len(period_df) > 0:
            print("\nPeriod model selection accuracy")
            print(period_df["correct_period"].mean())

            period_df.to_csv(args.period_output_csv, index=False)
            print(f"Saved: {args.period_output_csv}")


if __name__ == "__main__":
    main()
