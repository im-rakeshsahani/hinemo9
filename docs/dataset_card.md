# HinEmo-9 — Dataset Card

An emotion dataset for code-mixed Hindi–English across two scripts: 30,436
YouTube comments annotated for nine emotions across three language varieties.

> **Before release:** fill in the citation, contact and repository URL at the
> foot of this file.

---

## At a glance

| | |
|---|---|
| Comments | 30,436 |
| Labels | 9 (Love, Joy, Anger, Sadness, Fear, Surprise, Nostalgia, Devotion, Neutral) |
| Varieties | English, Hindi (Devanagari), Hinglish (romanized) |
| Gold subset | 3,363 items, triple-annotated, Fleiss κ = 0.87 (κ = 0.78 blind control) |
| Model-derived subset | 27,073 items, 78.4% agreement with gold |
| Sources | 1,496 videos across 1,014 channels, 39 genre seeds |
| Splits | train 28,417 / dev 1,009 / test 1,010 |
| Domain | Public YouTube comments |
| Licence | CC BY 4.0 (annotations and curation) |

---

## Label provenance — read this first

The dataset has **two tiers**, and every record carries a `label_source`
field.

**`human_gold` (3,363 items).** Annotated independently by three annotators
with no model predictions displayed. Fleiss κ = 0.87. Labels assigned by
majority vote — 2,812 unanimous, 528 majority — with 23 no-majority items
adjudicated by the first author against the annotation guidelines.

Because these items had been seen earlier in the model-assisted stage, a blind
control was run on 501 comments no annotator had encountered. Agreement on
the control is **κ = 0.78** against a composition-matched gold baseline of
0.835, so the headline figure is modestly inflated by prior exposure and 0.78
is the better estimate of independent agreement.

**`llm` (27,073 items).** Labels produced by an LLM pre-labeler
(claude-haiku-4-5). Each item was reviewed and submitted by an annotator with
the prediction displayed; verification agreement was 99.6–99.97%, so these are
**model-derived with human verification** rather than independent annotation.

**Estimated accuracy of the `llm` tier: 78.4%**, measured against gold. The
blind control gives 79.1%, so this figure is not inflated by prior exposure.
Per-category accuracy varies substantially:

| Emotion | Accuracy | Emotion | Accuracy |
|---|---:|---|---:|
| Surprise | 92% | Sadness | 82% |
| Nostalgia | 87% | Fear | 80% |
| Devotion | 87% | Love | 59% |
| Neutral | 86% | Joy | 52% |
| Anger | 82% | **Overall** | **78.4%** |

**Recommendation:** evaluate on `human_gold` items only. The `llm` tier suits
training and augmentation.

---

## Files

| File | Rows | Contents |
|---|---:|---|
| `data/train.jsonl` | 28,417 | 27,073 model-derived + 1,344 gold |
| `data/dev.jsonl` | 1,009 | gold, held out |
| `data/test.jsonl` | 1,010 | gold, held out |
| `data/gold.jsonl` | 3,363 | all gold items, incl. individual rater labels |
| `data/full.jsonl` | 30,436 | the complete curated set |
| `data/checksums.txt` | — | sha256 for each file above |
| `reference/` | — | per-cell metrics, agreement reports, lexical tables |
| `docs/annotation_guidelines.md` | — | the document annotators worked from |

Splits are fixed and disjoint by `id`.

---

## Field schema

| Field | Type | Notes |
|---|---|---|
| `id` | string | unique across the dataset |
| `text` | string | comment text, **PII-masked** (`[NAME]`, `[USER]`) |
| `label` | string | primary emotion, one of the nine |
| `language_label` | string | `English` / `Hindi` / `Hinglish` |
| `genre` | string | collection genre seed |
| `label_source` | string | `human_gold` or `llm` |
| `n_annotators` | int | 3 for gold, 1 otherwise |
| `gold_source` | string | gold only: `unanimous` / `majority` / `adjudicated` |
| `rater_labels` | list | gold only: the three individual annotator labels |
| `llm_prediction` | string | gold only: the pre-labeler's label, for comparison |
| `code_mix_index` | int | 0–25; higher = more Hindi-side tokens |
| `n_tokens` | int | 3–30 by construction |

Unmasked text (`raw_text`) is **not** distributed.

---

## Taxonomy

Seven conventional categories plus two that recur in Indian online discourse
and are absent from existing emotion resources.

**Nostalgia** — bittersweet longing for the past; missing something gone, fond
remembering with an ache. Distinguished from Sadness by warmth toward what is
remembered.

**Devotion** — reverence toward something higher: deity, guru, nation, revered
figure. Distinguished from Love by asymmetry — the writer looks *up* at the
object rather than across.

Both achieve the highest inter-annotator agreement of any category (κ = 0.93
and 0.89) while showing small, non-zero confusion with their nearest
neighbours — 39 Nostalgia↔Sadness and 59 Devotion↔Love disagreement pairs,
1.2% and 1.8% of the gold set. Distinct but linguistically adjacent, as
intended.

---

## Composition

| Emotion | English | Hindi | Hinglish | Total |
|---|---:|---:|---:|---:|
| Love | 1,200 | 1,200 | 1,200 | 3,600 |
| Joy | 1,200 | 1,200 | 1,200 | 3,600 |
| Anger | 1,200 | **1,098** | 1,200 | 3,498 |
| Sadness | 1,200 | 1,200 | 1,200 | 3,600 |
| Fear | 1,191 | **280** | 1,201 | 2,672 |
| Surprise | 1,113 | **161** | 492 | 1,766 |
| Nostalgia | 1,200 | 1,200 | 1,200 | 3,600 |
| Devotion | 1,200 | 1,200 | 1,200 | 3,600 |
| Neutral | 1,500 | 1,500 | 1,500 | 4,500 |
| **Total** | **11,004** | **9,039** | **10,393** | **30,436** |

Target was 1,200 per emotion×variety cell, Neutral 1,500 as the background
class. Fear and Surprise were exempt from the cap, so their rows show the
complete available pool rather than a curation choice. Anger-Devanagari is the
one capped cell whose pool ran out below target.

---

## Construction

**Collection.** 312,953 public YouTube comments gathered through genre-seeded
search across 39 genres, filtered to 3–30 tokens, deduplicated corpus-wide,
and PII-masked — leaving 210,405. Author names, channel identifiers and avatar
URLs were discarded at collection; names and URLs inside comment text were
replaced at preprocessing.

**Variety assignment.** Devanagari with no substantial Latin content → Hindi;
Latin script containing Hindi lexis → Hinglish; otherwise English.

**Pre-labelling.** 111,496 comments labelled by an LLM with a fixed nine-way
prompt and seventeen few-shot examples.

**Annotation.** Two stages — model-assisted verification of all items, then
independent annotation of a 3,363-item subset with predictions withheld.

**Splits.** Stratified by emotion×variety. All evaluation items are gold and
unseen in training. 40% of gold was allocated to training because gold
construction had consumed the entire available pool for the scarcest cells,
which would otherwise have left them without training support.

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

\* Leave-one-annotator-out, scored only on judgements where the other two
annotators agreed (90% of judgements). Excludes the most ambiguous items and
should be read as an **upper bound**.

---

## Known limitations

1. **27,073 labels are model-derived** with low-intervention human review;
   estimated 78.4% accurate, and considerably lower for Joy (52%) and Love
   (59%).
2. **Gold agreement reflects prior exposure.** κ = 0.87 rests on items
   annotators had seen in the model-assisted stage; the blind control gives
   κ = 0.78. The control cannot cover Fear or Surprise, whose unlabelled pools
   were exhausted during curation.
3. **The gold subset over-samples scarce classes**, so neither it nor the dev
   and test splits reflect natural emotion prevalence. Several
   emotion×variety cells in those splits hold 12–17 items.
4. **Devanagari Fear (280) and Surprise (161) are scarce.** Seven targeted
   collection strategies each reached Devanagari-writing audiences but
   returned target-emotion yields below 1%. This is reported as a property of
   the register — reactive emotions in casual Hindi appear in romanized script
   or English — rather than a collection failure.
5. **Devotion is internally heterogeneous**, covering religious, guru-directed
   and nationalist veneration under one label. The composition is not
   reported, and systems should not be assumed to distinguish them.
6. **Single platform** (YouTube) and genre-seeded sampling.
7. **Single primary label.** Secondary emotions were recorded during
   annotation but are not analysed here.
8. **Three annotators** of shared regional and linguistic background.

---

## Ethics

Comments were collected from publicly visible YouTube threads. No interaction
with human participants took place and no personally identifiable information
was intentionally collected. Identifying metadata was discarded at collection
and in-text names and URLs are masked; only masked text is distributed. On
this basis the study did not require institutional ethics board approval.

Three annotators, all Indian nationals with working knowledge of Hindi and
English, participated voluntarily and were not paid. They were informed of the
research purpose and of the intention to release the annotated data publicly.

**Intended use:** research on emotion detection in code-mixed and multi-script
text; studies of script and register effects; annotation methodology.

**Out of scope:** identifying, profiling or making inferences about specific
individuals; any deployment where a misclassification could affect a person
without human review. Devotion is a religiously and politically inflected
category in Indian discourse and should not be used to classify individuals or
groups by religious expression.

---

## Licence

**CC BY 4.0**, covering the annotations, taxonomy, splits and curation.

The licence does not extend to the underlying comment text, which was authored
by third parties and remains subject to the terms of the platform on which it
was posted. Users redistributing or building on the dataset should attribute
this work and observe those platform terms independently.

---

## Citation

> **TODO** — BibTeX entry once the paper has a venue.

## Repository

> **TODO** — repository URL and archival DOI.

## Contact

Rakesh Sahani — rakesh.sahani@rgu.ac.in
Department of Computer Science & Engineering, Rajiv Gandhi University,
Doimukh, India–791112
