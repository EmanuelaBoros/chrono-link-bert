# Chrono-Link-BERT

**Chrono-Link-BERT** is an experimental BERT-style model for historical NLP. It is trained on historical newspaper text enriched with lightweight temporal knowledge derived from detected entities and Wikidata facts.

The idea is simple: historical documents are not only sequences of tokens. They also contain people, places, organizations, events, dates, and entities whose meaning depends on time. Chrono-Link-BERT explores whether adding lightweight temporal/entity grounding during continued pretraining can improve downstream historical NLP tasks such as named entity recognition.

---

## Model idea

Chrono-Link-BERT follows a two-stage pipeline:

1. **Dataset enrichment**
   - Historical newspaper articles are sampled from `dell-research-harvard/AmericanStories`.
   - Candidate entities are detected with spaCy.
   - Mentions are linked to Wikidata using lightweight search.
   - Temporal facts such as birth year, death year, inception year, dissolution year, start time, and end time are extracted.
   - Each article is augmented with a textual `fact_context`.

2. **Continued pretraining**
   - A BERT masked-language model is further trained on:

```text
article text [SEP] temporal Wikidata fact context
```

The resulting model is then used as a base model for downstream NER fine-tuning.

---

## Example enriched input

```text
Steam to Australia under 60 Days. PASSAGE MONEY £14 AND UPWARDS ...
[SEP]
Document year: 1850. Linked entities: Australia [Q408] type=country, sovereign state temporal_status=no_temporal_facts.
```

Each enriched example contains fields such as:

```json
{
  "year": 1850,
  "text": "historical newspaper article text...",
  "fact_context": "Document year: 1850. Linked entities: ...",
  "linked_entities": [
    {
      "mention": "Australia",
      "qid": "Q408",
      "label": "Australia",
      "instances": ["country", "sovereign state"],
      "temporal_status": "no_temporal_facts"
    }
  ]
}
```

---

## Installation

```bash
pip install -U pip
pip install -r requirements.txt
python -m spacy download xx_ent_wiki_sm
```

For loading `AmericanStories`, use a `datasets` version that supports dataset scripts, for example:

```bash
pip install "datasets==3.6.0" "huggingface_hub==0.36.2"
```

---

## 1. Build the enriched dataset

The dataset-building script loads selected years from AmericanStories, detects entity mentions, retrieves lightweight Wikidata facts, and creates an enriched Hugging Face dataset.

```bash
python build_dataset.py \
  --years 1850 1860 1870 1880 1890 \
  --max_examples_per_year 500 \
  --max_mentions 4 \
  --output_repo emanuelaboros/americanstories-1850-1890
```

The script produces a dataset with `train`, `validation`, and `test` splits.

Important implementation detail for AmericanStories:

```python
def load_americanstories_year(year: int):
    dataset = load_dataset(
        "dell-research-harvard/AmericanStories",
        "subset_years",
        year_list=[str(year)],
        trust_remote_code=True,
    )
    return dataset[str(year)]
```

---

## 2. Continue pretraining with MLM

Once the enriched dataset is built, continue pretraining BERT using masked language modeling.

```bash
python train_mlm.py \
  --dataset_name emanuelaboros/americanstories-1850-1890 \
  --model_name bert-base-cased \
  --output_dir chrono-link-bert-americanstories-1850-1890 \
  --hub_model_id emanuelaboros/chrono-link-bert-americanstories-1850-1890 \
  --num_train_epochs 3 \
  --max_length 256 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --push_to_hub
```

This produces the continued-pretrained base model:

```text
emanuelaboros/chrono-link-bert-americanstories-1850-1890
```

---

## 3. Fine-tune for historical NER

Chrono-Link-BERT is first trained as a masked-language model. To use it for NER, it must be fine-tuned on labeled token-classification data.

Example with HIPE/topres19th TSV files:

```bash
python train_ner.py \
  --model_name_or_path emanuelaboros/chrono-link-bert-americanstories-1850-1890 \
  --train_file data/topres19th/en/HIPE-2022-v2.1-topres19th-train-en.tsv \
  --validation_file data/topres19th/en/HIPE-2022-v2.1-topres19th-dev-en.tsv \
  --test_file data/topres19th/en/HIPE-2022-v2.1-topres19th-test-en.tsv \
  --label_column NE-COARSE-LIT \
  --output_dir chrono-link-bert-topres19th-ner \
  --hub_model_id emanuelaboros/chrono-link-bert-topres19th-ner \
  --num_train_epochs 5 \
  --learning_rate 3e-5 \
  --max_length 256 \
  --per_device_train_batch_size 16 \
  --push_to_hub
```

---

## 4. Evaluate NER

```bash
python evaluate_ner.py \
  --model_name_or_path emanuelaboros/chrono-link-bert-topres19th-ner \
  --test_file data/topres19th/en/HIPE-2022-v2.1-topres19th-test-en.tsv \
  --label_column NE-COARSE-LIT
```

The evaluation script reads HIPE-style TSV files with columns such as:

```text
TOKEN	NE-COARSE-LIT	NE-COARSE-METO	NE-FINE-LIT	NE-FINE-METO	NE-FINE-COMP	NE-NESTED	NEL-LIT	NEL-METO	MISC
```

Sentences are reconstructed using `EndOfSentence` markers in the `MISC` column.

---

## 5. Compare with other historical BERT models

The comparison script fine-tunes several base models on the same NER training data and evaluates them on the same test set.

```bash
python compare_ner_models.py \
  --models \
    bert-base-cased \
    Livingwithmachines/bert_1760_1850 \
    Livingwithmachines/bert_1850_1875 \
    Livingwithmachines/bert_1875_1890 \
    Livingwithmachines/bert_1890_1900 \
    Livingwithmachines/bert_1760_1900 \
    emanuelaboros/chrono-link-bert-americanstories-1850-1890 \
  --train_file data/topres19th/en/HIPE-2022-v2.1-topres19th-train-en.tsv \
  --validation_file data/topres19th/en/HIPE-2022-v2.1-topres19th-dev-en.tsv \
  --test_file data/topres19th/en/HIPE-2022-v2.1-topres19th-test-en.tsv \
  --label_column NE-COARSE-LIT \
  --output_dir ner_model_comparison \
  --summary_csv ner_model_comparison_results.csv \
  --num_train_epochs 5 \
  --learning_rate 3e-5 \
  --max_length 256 \
  --per_device_train_batch_size 16
```

---

## NER comparison results

The following results compare Chrono-Link-BERT against generic BERT and several Living with Machines historical BERT models. All models were fine-tuned and evaluated on the same HIPE/topres19th NER setup using the `NE-COARSE-LIT` column.

| Rank | Model | Valid Precision | Valid Recall | Valid F1 | Test Precision | Test Recall | Test F1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `Livingwithmachines/bert_1760_1900` | 0.7915 | 0.8723 | 0.8300 | 0.7854 | **0.8635** | **0.8226** |
| 2 | `Livingwithmachines/bert_1850_1875` | 0.7871 | **0.8809** | **0.8313** | **0.7971** | 0.8307 | 0.8135 |
| 3 | `Livingwithmachines/bert_1760_1850` | 0.7876 | 0.8681 | 0.8259 | 0.7948 | 0.8222 | 0.8083 |
| 4 | `Livingwithmachines/bert_1890_1900` | **0.7969** | 0.8681 | 0.8310 | 0.7858 | 0.8315 | 0.8080 |
| 5 | `Livingwithmachines/bert_1875_1890` | 0.7841 | **0.8809** | 0.8297 | 0.7754 | 0.8172 | 0.7957 |
| 6 | `bert-base-cased` | 0.7773 | 0.8170 | 0.7967 | 0.7819 | 0.7793 | 0.7806 |
| 7 | `emanuelaboros/chrono-link-bert-americanstories-1850-1890` | 0.7520 | 0.8128 | 0.7812 | 0.7490 | 0.7970 | 0.7722 |

### Interpretation

The best-performing model in this experiment is `Livingwithmachines/bert_1760_1900`, with a test F1 of **0.8226**. All Living with Machines historical BERT variants outperform both `bert-base-cased` and the current Chrono-Link-BERT checkpoint on this NER benchmark.

The current Chrono-Link-BERT model obtains a test F1 of **0.7722**, below the generic `bert-base-cased` baseline at **0.7806**. The gap is relatively small, but the result suggests that the current lightweight Wikidata-based temporal enrichment setup does not yet improve downstream historical NER over generic BERT or broad historical-domain pretraining.

This result shows that broad historical-domain pretraining remains a stronger signal for this task than the current temporal-enrichment strategy. A more promising next experiment is to apply Chrono-Link enrichment on top of an already historical model such as `Livingwithmachines/bert_1760_1900`, rather than starting from `bert-base-cased`.

---

## Limitations

Chrono-Link-BERT is an early prototype.
---

## Citation

If you use this repository or build on this idea, please cite the repository or the associated paper/preprint once available.

```bibtex
@misc{boros2026chronolinkbert,
  title        = {Chrono-Link-BERT: Lightweight Temporal Knowledge Linking for Historical Language Models},
  author       = {Boros, Emanuela},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/EmanuelaBoros/chrono-link-bert}
}
```

---

## License

Please check the licenses of all upstream resources before redistributing data or models. In particular, the enriched dataset derives from AmericanStories and includes Wikidata-derived metadata. Downstream HIPE/topres19th evaluation files may have their own licensing constraints.
