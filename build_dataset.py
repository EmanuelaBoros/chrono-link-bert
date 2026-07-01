from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
import spacy
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
from tqdm import tqdm

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

HEADERS = {"User-Agent": "Chrono-Link-BERT/0.1 historical NLP research prototype"}


def parse_year(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"([+-]?\d{3,4})", text)
    if not match:
        return None
    try:
        return int(match.group(1).replace("+", ""))
    except ValueError:
        return None


def load_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict[str, Any], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def wikidata_search_entity(
    mention: str,
    language: str = "en",
    limit: int = 3,
    sleep: float = 0.05,
) -> list[dict[str, Any]]:
    params = {
        "action": "wbsearchentities",
        "search": mention,
        "language": language,
        "format": "json",
        "limit": limit,
    }

    try:
        response = requests.get(
            WIKIDATA_API,
            params=params,
            headers=HEADERS,
            timeout=20,
        )
        response.raise_for_status()
        time.sleep(sleep)
        return response.json().get("search", [])
    except Exception:
        return []


def wikidata_get_temporal_facts(qid: str, sleep: float = 0.05) -> dict[str, Any]:
    query = f"""
    SELECT ?item ?itemLabel ?itemDescription
           ?birth ?death ?inception ?dissolved ?start ?end
           ?instanceLabel ?occupationLabel ?countryLabel
    WHERE {{
      BIND(wd:{qid} AS ?item)

      OPTIONAL {{ ?item wdt:P569 ?birth. }}
      OPTIONAL {{ ?item wdt:P570 ?death. }}
      OPTIONAL {{ ?item wdt:P571 ?inception. }}
      OPTIONAL {{ ?item wdt:P576 ?dissolved. }}
      OPTIONAL {{ ?item wdt:P580 ?start. }}
      OPTIONAL {{ ?item wdt:P582 ?end. }}

      OPTIONAL {{ ?item wdt:P31 ?instance. }}
      OPTIONAL {{ ?item wdt:P106 ?occupation. }}
      OPTIONAL {{ ?item wdt:P17 ?country. }}

      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en".
        ?item rdfs:label ?itemLabel.
        ?item schema:description ?itemDescription.
        ?instance rdfs:label ?instanceLabel.
        ?occupation rdfs:label ?occupationLabel.
        ?country rdfs:label ?countryLabel.
      }}
    }}
    LIMIT 10
    """

    try:
        response = requests.get(
            WIKIDATA_SPARQL,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        time.sleep(sleep)
        data = response.json()
    except Exception:
        return {"qid": qid}

    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return {"qid": qid}

    labels = []
    descriptions = []
    instances = set()
    occupations = set()
    countries = set()

    years = {
        "birth_year": None,
        "death_year": None,
        "inception_year": None,
        "dissolved_year": None,
        "start_year": None,
        "end_year": None,
    }

    for b in bindings:
        if "itemLabel" in b:
            labels.append(b["itemLabel"]["value"])
        if "itemDescription" in b:
            descriptions.append(b["itemDescription"]["value"])

        if "instanceLabel" in b:
            instances.add(b["instanceLabel"]["value"])
        if "occupationLabel" in b:
            occupations.add(b["occupationLabel"]["value"])
        if "countryLabel" in b:
            countries.add(b["countryLabel"]["value"])

        mapping = {
            "birth": "birth_year",
            "death": "death_year",
            "inception": "inception_year",
            "dissolved": "dissolved_year",
            "start": "start_year",
            "end": "end_year",
        }

        for sparql_name, out_name in mapping.items():
            if sparql_name in b and years[out_name] is None:
                years[out_name] = parse_year(b[sparql_name]["value"])

    return {
        "qid": qid,
        "label": labels[0] if labels else None,
        "description": descriptions[0] if descriptions else None,
        "instances": sorted(instances)[:5],
        "occupations": sorted(occupations)[:5],
        "countries": sorted(countries)[:5],
        **years,
    }


def temporal_status(facts: dict[str, Any], doc_year: int | None) -> str:
    if doc_year is None:
        return "unknown_doc_year"

    intervals = []

    birth = facts.get("birth_year")
    death = facts.get("death_year")
    inception = facts.get("inception_year")
    dissolved = facts.get("dissolved_year")
    start = facts.get("start_year")
    end = facts.get("end_year")

    if birth is not None or death is not None:
        intervals.append((birth, death))
    if inception is not None or dissolved is not None:
        intervals.append((inception, dissolved))
    if start is not None or end is not None:
        intervals.append((start, end))

    if not intervals:
        return "no_temporal_facts"

    for left, right in intervals:
        if left is not None and doc_year < left:
            continue
        if right is not None and doc_year > right:
            continue
        return "active_or_alive"

    return "temporally_mismatched"


def candidate_mentions_from_spacy(
    text: str, nlp: Any, max_mentions: int = 8
) -> list[str]:
    doc = nlp(text[:5000])
    mentions = []

    for ent in doc.ents:
        mention = ent.text.strip()
        if len(mention) < 3:
            continue
        if len(mention.split()) > 5:
            continue
        if mention.lower() in {"the", "and", "for", "said"}:
            continue
        mentions.append(mention)

    seen = set()
    clean = []

    for mention in mentions:
        key = mention.lower()
        if key not in seen:
            seen.add(key)
            clean.append(mention)

    return clean[:max_mentions]


def choose_best_candidate(
    mention: str,
    candidates: list[dict[str, Any]],
    doc_year: int | None,
    fact_cache: dict[str, Any],
) -> dict[str, Any] | None:
    enriched = []

    for cand in candidates:
        qid = cand.get("id")
        if not qid:
            continue

        if qid not in fact_cache:
            fact_cache[qid] = wikidata_get_temporal_facts(qid)

        facts = dict(fact_cache[qid])
        facts["mention"] = mention
        facts["search_label"] = cand.get("label")
        facts["search_description"] = cand.get("description")
        facts["temporal_status"] = temporal_status(facts, doc_year)

        enriched.append(facts)

    if not enriched:
        return None

    plausible = [x for x in enriched if x["temporal_status"] == "active_or_alive"]
    if plausible:
        return plausible[0]

    no_temporal = [x for x in enriched if x["temporal_status"] == "no_temporal_facts"]
    if no_temporal:
        return no_temporal[0]

    return enriched[0]


def build_fact_context(
    doc_year: int | None,
    linked_entities: list[dict[str, Any]],
    max_chars: int = 700,
) -> str:
    parts = []

    if doc_year is not None:
        parts.append(f"Document year: {doc_year}.")
    else:
        parts.append("Document year: unknown.")

    bits = []

    for ent in linked_entities:
        label = ent.get("label") or ent.get("search_label") or ent.get("mention")
        qid = ent.get("qid")
        status = ent.get("temporal_status", "unknown")

        dates = []
        if ent.get("birth_year") is not None or ent.get("death_year") is not None:
            dates.append(f"{ent.get('birth_year', '?')}-{ent.get('death_year', '?')}")
        if (
            ent.get("inception_year") is not None
            or ent.get("dissolved_year") is not None
        ):
            dates.append(
                f"inception={ent.get('inception_year', '?')}, dissolved={ent.get('dissolved_year', '?')}"
            )

        types = []
        types.extend(ent.get("instances", [])[:2])
        types.extend(ent.get("occupations", [])[:2])
        types.extend(ent.get("countries", [])[:1])

        bit = str(label)
        if qid:
            bit += f" [{qid}]"
        if types:
            bit += " type=" + ", ".join(types)
        if dates:
            bit += " dates=" + "; ".join(dates)
        bit += f" temporal_status={status}"

        bits.append(bit)

    if bits:
        parts.append("Linked entities: " + " ; ".join(bits) + ".")
    else:
        parts.append("Linked entities: none.")

    return " ".join(parts)[:max_chars]


def detect_text_column(row: dict[str, Any], preferred: str | None = None) -> str:
    if preferred and preferred in row:
        return preferred

    candidates = [
        "article",
        "text",
        "article_text",
        "raw_text",
        "ocr",
        "content",
    ]

    for col in candidates:
        if col in row and isinstance(row[col], str) and len(row[col]) > 20:
            return col

    for key, value in row.items():
        if isinstance(value, str) and len(value) > 50:
            return key

    raise ValueError(
        f"Could not detect text column. Available columns: {list(row.keys())}"
    )


def clean_article_text(text: str, max_chars: int = 2000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def enrich_article(
    row: dict[str, Any],
    text_column: str,
    doc_year: int,
    nlp: Any,
    entity_cache: dict[str, Any],
    fact_cache: dict[str, Any],
    max_mentions: int,
) -> dict[str, Any]:
    text = clean_article_text(str(row[text_column]))

    mentions = candidate_mentions_from_spacy(
        text=text,
        nlp=nlp,
        max_mentions=max_mentions,
    )

    linked_entities = []

    for mention in mentions:
        cache_key = f"en::{mention}"

        if cache_key not in entity_cache:
            entity_cache[cache_key] = wikidata_search_entity(
                mention=mention,
                language="en",
                limit=3,
            )

        best = choose_best_candidate(
            mention=mention,
            candidates=entity_cache[cache_key],
            doc_year=doc_year,
            fact_cache=fact_cache,
        )

        if best is not None:
            linked_entities.append(best)

    fact_context = build_fact_context(
        doc_year=doc_year,
        linked_entities=linked_entities,
    )

    return {
        "id": str(row.get("id", row.get("article_id", ""))),
        "year": doc_year,
        "text": text,
        "fact_context": fact_context,
        "linked_entities": linked_entities,
        "source_dataset": "dell-research-harvard/AmericanStories",
    }


def load_americanstories_year(year: int):
    return load_dataset(
        "dell-research-harvard/AmericanStories",
        str(year),
        split="train",
        streaming=True,
        trust_remote_code=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--max_examples_per_year", type=int, default=1000)
    parser.add_argument("--text_column", default=None)
    parser.add_argument("--max_mentions", type=int, default=8)
    parser.add_argument("--cache_dir", default="chronolink_cache")
    parser.add_argument("--output_repo", required=True)
    parser.add_argument("--hf_token", default=None)

    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    print("Loading spaCy...")
    nlp = spacy.load("xx_ent_wiki_sm")

    cache_dir = Path(args.cache_dir)
    entity_cache_path = cache_dir / "entity_search_cache.json"
    fact_cache_path = cache_dir / "wikidata_fact_cache.json"

    entity_cache = load_cache(entity_cache_path)
    fact_cache = load_cache(fact_cache_path)

    all_rows = []

    for year in args.years:
        print(f"Loading AmericanStories year {year}...")
        stream = load_americanstories_year(year)

        text_column = args.text_column
        count = 0

        for row in tqdm(stream, desc=f"Year {year}"):
            row = dict(row)

            if text_column is None:
                text_column = detect_text_column(row, preferred=args.text_column)
                print(f"Detected text column for {year}: {text_column}")

            try:
                enriched = enrich_article(
                    row=row,
                    text_column=text_column,
                    doc_year=year,
                    nlp=nlp,
                    entity_cache=entity_cache,
                    fact_cache=fact_cache,
                    max_mentions=args.max_mentions,
                )
            except Exception as e:
                print(f"Skipping row due to error: {e}")
                continue

            if len(enriched["text"]) < 50:
                continue

            all_rows.append(enriched)
            count += 1

            if count >= args.max_examples_per_year:
                break

        save_cache(entity_cache, entity_cache_path)
        save_cache(fact_cache, fact_cache_path)

    dataset = Dataset.from_list(all_rows)

    # simple split
    dataset = dataset.shuffle(seed=42)
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    valid_test = train_test["test"].train_test_split(test_size=0.5, seed=42)

    dataset_dict = DatasetDict(
        {
            "train": train_test["train"],
            "validation": valid_test["train"],
            "test": valid_test["test"],
        }
    )

    print(dataset_dict)
    print(f"Pushing to {args.output_repo}")
    dataset_dict.push_to_hub(args.output_repo)
    print("Done.")


if __name__ == "__main__":
    main()
