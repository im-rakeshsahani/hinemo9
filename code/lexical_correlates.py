#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lexical_correlates.py — tokens most associated with each emotion.

Implements the log-odds ratio with informative Dirichlet prior (Monroe,
Colaresi & Quinn 2008), the method used for the GoEmotions lexical table.
Each emotion is contrasted against all other emotions; the corpus-wide token
distribution serves as the prior. Values are z-scores, so |z| > 3 indicates a
highly significant association.

Two outputs:
  paper_tables/lexical_correlates.csv        top tokens per emotion
  paper_tables/lexical_correlates_by_lang.csv  same, split by register

The second is the code-mixed extension: it shows whether an emotion's markers
are carried by Devanagari, romanized Hindi, or English tokens.
"""
import json, math, re, collections
from pathlib import Path

import pandas as pd

OUT = Path("paper_tables"); OUT.mkdir(exist_ok=True)
EMOTIONS = ["Love", "Joy", "Anger", "Sadness", "Fear",
            "Surprise", "Nostalgia", "Devotion", "Neutral"]
LANGS = ["English", "Hindi", "Hinglish"]
TOP_N = 8
MIN_COUNT = 20          # ignore very rare tokens
Z_THRESHOLD = 3.0

# keep Devanagari, Latin letters and emoji as tokens; drop punctuation
TOKEN = re.compile(r"[\u0900-\u097F]+|[A-Za-z]+|[\U0001F300-\U0001FAFF\u2600-\u27BF]")
PLACEHOLDER = re.compile(r"\[[A-Z]+\]")


def tokenize(text):
    text = PLACEHOLDER.sub(" ", text or "")
    return [t.lower() for t in TOKEN.findall(text)]


def log_odds(counts_i, counts_j, prior):
    """z-scored log odds ratio with informative Dirichlet prior."""
    n_i = sum(counts_i.values())
    n_j = sum(counts_j.values())
    a0 = sum(prior.values())
    out = {}
    for w, a_w in prior.items():
        y_i = counts_i.get(w, 0)
        y_j = counts_j.get(w, 0)
        if y_i + y_j < MIN_COUNT:
            continue
        num_i = y_i + a_w
        den_i = n_i + a0 - y_i - a_w
        num_j = y_j + a_w
        den_j = n_j + a0 - y_j - a_w
        if min(num_i, den_i, num_j, den_j) <= 0:
            continue
        delta = math.log(num_i / den_i) - math.log(num_j / den_j)
        var = 1.0 / num_i + 1.0 / num_j
        out[w] = delta / math.sqrt(var)
    return out


def load_rows():
    """Prefer human-gold labels; fall back to the full curated set."""
    rows = []
    gold_p = Path("data/gold_dataset.json")
    if gold_p.exists():
        for g in json.load(open(gold_p, encoding="utf-8")):
            rows.append({"text": g.get("masked_text", ""),
                         "label": g["gold_emotion"],
                         "lang": g.get("language_label")})
    full_p = Path("data/final_annotation_set.json")
    if full_p.exists():
        gold_ids = set()
        if gold_p.exists():
            gold_ids = {g["id"] for g in json.load(open(gold_p, encoding="utf-8"))}
        for t in json.load(open(full_p, encoding="utf-8")):
            d = t["data"]
            if d.get("id") in gold_ids:
                continue
            prim = None
            for r in (t.get("predictions") or [{}])[0].get("result", []):
                if r.get("from_name") == "primary_emotion":
                    prim = (r.get("value", {}).get("choices") or [None])[0]
            if prim:
                rows.append({"text": d.get("masked_text", ""),
                             "label": prim,
                             "lang": d.get("language_label")})
    return rows


def table(rows, tag):
    by_emo = collections.defaultdict(collections.Counter)
    prior = collections.Counter()
    for r in rows:
        toks = tokenize(r["text"])
        by_emo[r["label"]].update(toks)
        prior.update(toks)
    prior = {w: c for w, c in prior.items() if c >= MIN_COUNT}
    if not prior:
        return []

    out = []
    for e in EMOTIONS:
        if e not in by_emo:
            continue
        others = collections.Counter()
        for e2, c in by_emo.items():
            if e2 != e:
                others.update(c)
        z = log_odds(by_emo[e], others, prior)
        top = sorted(z.items(), key=lambda kv: -kv[1])[:TOP_N]
        for w, score in top:
            if score >= Z_THRESHOLD:
                out.append({"scope": tag, "emotion": e, "token": w,
                            "z": round(score, 2),
                            "count_in_emotion": by_emo[e][w]})
    return out


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} labelled comments\n")

    overall = table(rows, "all")
    df = pd.DataFrame(overall)
    df.to_csv(OUT / "lexical_correlates.csv", index=False)

    print("=== TOP TOKENS PER EMOTION (z-scored log odds, threshold 3.0) ===")
    for e in EMOTIONS:
        sub = df[df["emotion"] == e]
        if len(sub):
            toks = "  ".join(f"{r.token} ({r.z:.0f})" for r in sub.itertuples())
            print(f"  {e:10s} {toks}")
        else:
            print(f"  {e:10s} —")

    per_lang = []
    for lg in LANGS:
        sub = [r for r in rows if r["lang"] == lg]
        per_lang.extend(table(sub, lg))
    dl = pd.DataFrame(per_lang)
    dl.to_csv(OUT / "lexical_correlates_by_lang.csv", index=False)

    print("\n=== BY REGISTER ===")
    for lg in LANGS:
        print(f"\n--- {lg} ---")
        for e in EMOTIONS:
            sub = dl[(dl["scope"] == lg) & (dl["emotion"] == e)]
            if len(sub):
                toks = "  ".join(f"{r.token}" for r in sub.itertuples())
                print(f"  {e:10s} {toks}")

    print(f"\nwritten to {OUT}/lexical_correlates.csv and _by_lang.csv")
    print("\nEmotions with strong lexical markers usually show higher")
    print("inter-annotator agreement; emotions without them tend to require")
    print("context. Compare this list against the per-label kappa in §5.3 —")
    print("that comparison is the interpretation worth writing up.")


if __name__ == "__main__":
    main()
