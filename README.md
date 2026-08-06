# HinEmo-9

An emotion dataset for code-mixed Hindi–English across two scripts.

30,436 YouTube comments annotated for nine emotions across three language
varieties: English, Devanagari Hindi, and romanized Hinglish.

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
<!-- Add after the Zenodo release: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

---

## Read this before using the data

The dataset has **two tiers**, and every record carries a `label_source`
field.

| Tier | Items | How labels were produced | Recommended use |
|---|---:|---|---|
| `human_gold` | 3,363 | Three annotators, independently, predictions withheld. Fleiss κ = 0.87 (κ = 0.78 on a blind control set) | evaluation |
| `llm` | 27,073 | LLM pre-labeler, reviewed by an annotator with the prediction displayed. 78.4% agreement with the gold tier | training |

**Evaluate on the gold tier.** The model-derived tier is roughly 78% accurate
overall, but only 52% for Joy and 59% for Love. A system trained on it will
inherit those error rates unevenly across categories.

---

## Quick start

```python
import json

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

train = load("data/train.jsonl")   # 28,417
dev   = load("data/dev.jsonl")     #  1,009  gold
test  = load("data/test.jsonl")    #  1,010  gold
gold  = load("data/gold.jsonl")    #  3,363  incl. individual rater labels

print(train[0])
# {'id': '...', 'text': '...', 'label': 'Devotion',
#  'language_label': 'Hindi', 'label_source': 'llm', ...}
```

With Hugging Face `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("<username>/hinemo9")
```

Splits are fixed and disjoint by `id`. Verify integrity with
`checksums.txt`.

---

## Files

| Path | Rows | Contents |
|---|---:|---|
| `data/train.jsonl` | 28,417 | 27,073 model-derived + 1,344 gold |
| `data/dev.jsonl` | 1,009 | gold, held out |
| `data/test.jsonl` | 1,010 | gold, held out |
| `data/gold.jsonl` | 3,363 | all gold items, with each rater's label |
| `data/full.jsonl` | 30,436 | the complete curated set |
| `reference/` | — | per-cell metrics, confusion matrices, agreement report |
| `docs/annotation_guidelines.md` | — | the document annotators worked from |
| `code/` | — | collection, processing, training, evaluation, figures |

### Field schema

| Field | Type | Notes |
|---|---|---|
| `id` | string | unique across the dataset |
| `text` | string | comment text, **PII-masked** (`[NAME]`, `[USER]`) |
| `label` | string | primary emotion, one of nine |
| `language_label` | string | `English` / `Hindi` / `Hinglish` |
| `genre` | string | collection genre seed |
| `label_source` | string | `human_gold` or `llm` |
| `n_annotators` | int | 3 for gold, 1 otherwise |
| `gold_source` | string | gold only: `unanimous` / `majority` / `adjudicated` |
| `rater_labels` | list | gold only: the three individual labels |
| `llm_prediction` | string | gold only: what the pre-labeler predicted |
| `code_mix_index` | int | 0–25; higher = more Hindi-side tokens |
| `n_tokens` | int | 3–30 by construction |

Unmasked text is not distributed.

---

## Labels

Seven conventional categories plus two that recur in Indian online discourse
and are absent from existing emotion resources.

| Label | Definition |
|---|---|
| Love | Affection toward a person or thing, at peer level |
| Joy | Happiness, delight, amusement |
| Anger | Irritation, rage, blame, hostility |
| Sadness | Grief, hurt, disappointment about the present |
| Fear | Fright, dread, feeling unsafe |
| Surprise | Shock, astonishment, being caught off-guard |
| **Nostalgia** | Bittersweet longing for the past; fond remembering with an ache |
| **Devotion** | Reverence toward something higher — deity, guru, nation |
| Neutral | No emotional expression |

Both added categories achieve the highest inter-annotator agreement in the
corpus (κ = 0.93 and 0.89) and are the best-classified emotions for every
model benchmarked.

---

## Composition

| Emotion | English | Hindi | Hinglish | Total |
|---|---:|---:|---:|---:|
| Love | 1,200 | 1,200 | 1,200 | 3,600 |
| Joy | 1,200 | 1,200 | 1,200 | 3,600 |
| Anger | 1,200 | 1,098 | 1,200 | 3,498 |
| Sadness | 1,200 | 1,200 | 1,200 | 3,600 |
| Fear | 1,191 | 280 | 1,201 | 2,672 |
| Surprise | 1,113 | 161 | 492 | 1,766 |
| Nostalgia | 1,200 | 1,200 | 1,200 | 3,600 |
| Devotion | 1,200 | 1,200 | 1,200 | 3,600 |
| Neutral | 1,500 | 1,500 | 1,500 | 4,500 |
| **Total** | **11,004** | **9,039** | **10,393** | **30,436** |

Fear and Surprise were exempt from the 1,200 per-cell cap, so their rows show
the complete available pool. Devanagari Fear (280) and Surprise (161) are
scarce because reactive emotions in casual Hindi are written in romanized
script or English — see the paper.

---

## Benchmark

Macro-F1 on the gold test set, mean ± std over three seeds:

| Model | Macro-F1 |
|---|---|
| Human ceiling* | 0.942 |
| XLM-R base | 0.630 ± 0.010 |
| IndicBERT-v2 | 0.624 ± 0.003 |
| TF-IDF + char n-gram SVM | 0.561 |
| MuRIL base | 0.554 ± 0.012 |
| mBERT | 0.496 ± 0.004 |

\* Leave-one-annotator-out on judgements where the other two agreed; an upper
bound, since ambiguous items are excluded.

Reproduce with `code/train_model.py` and `code/aggregate_results.py` — see
`code/README.md`.

---

## Known limitations

1. 27,073 labels are model-derived with low-intervention human review;
   estimated 78.4% accurate, and considerably lower for Joy and Love.
2. Gold agreement (κ = 0.87) rests on items annotators had seen before; a
   blind control gives κ = 0.78, the better estimate.
3. Devanagari Fear and Surprise are scarce, and the gold subset over-samples
   them, so evaluation splits do not reflect natural prevalence.
4. Devotion spans religious, guru-directed and nationalist veneration under
   one label; the composition is not reported.
5. Single platform (YouTube), genre-seeded sampling, three annotators of
   shared regional background.

---

## Ethics and intended use

Built from publicly visible YouTube comments. Identifying metadata was
discarded at collection; personal names and URLs within comment text are
masked. No IRB approval was required: no human-participant interaction, no
intentional collection of personal data.

**In scope:** research on emotion detection in code-mixed and multi-script
text; studies of script and register effects; annotation methodology.

**Out of scope:** inferring emotional states of identifiable individuals;
profiling users or communities; any deployment where a misclassification
could affect a person without human review. Devotion in particular should not
be used to classify individuals or groups by religious expression.

---

## Licence

Annotations, taxonomy, splits and curation: **CC BY 4.0**.

The licence covers our contributions. It does not extend to the underlying
comment text, which was authored by third parties and remains subject to the
terms of the platform on which it was posted.

---

## Citation

```bibtex
@article{sahani2026hinemo9,
  title   = {HinEmo-9: An Emotion Dataset for Code-Mixed Hindi--English Across Two Scripts},
  author  = {Sahani, Rakesh and A, Firos},
  year    = {2026},
  note    = {Under review}
}
```

Update on acceptance.

---

## Contact

Rakesh Sahani — rakesh.sahani@rgu.ac.in
Department of Computer Science & Engineering, Rajiv Gandhi University,
Doimukh, India–791112
