#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_stats.py — the non-training numbers the paper needs.

Produces
  paper_tables/human_ceiling.csv        leave-one-annotator-out human performance
  paper_tables/source_diversity.csv     videos / channels / genres covered
  paper_tables/cooccurrence.csv         primary x secondary emotion matrix

Human ceiling method
--------------------
For each triple-annotated item and each rater r: treat r's label as the
"prediction" and the agreement of the *other two* raters as the reference.
Items where the other two disagree have no reference and are skipped (counted
and reported). Averaging over the three raters gives a human macro-F1 directly
comparable to the model numbers, computed on the same test items.
"""
import json, collections
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

OUT = Path("paper_tables"); OUT.mkdir(exist_ok=True)
EMOTIONS = ["Love", "Joy", "Anger", "Sadness", "Fear",
            "Surprise", "Nostalgia", "Devotion", "Neutral"]
LANGS = ["English", "Hindi", "Hinglish"]


# ===================================================== 1. HUMAN CEILING
def human_ceiling():
    merged = json.load(open("data/gold_merged.json", encoding="utf-8"))
    test_ids = set()
    with open("data/splits/test.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_ids.add(json.loads(line)["id"])
    dev_ids = set()
    with open("data/splits/dev.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dev_ids.add(json.loads(line)["id"])

    rows = []
    for scope, ids in [("test", test_ids), ("dev", dev_ids),
                       ("all_gold", None)]:
        preds, refs, skipped, total = [], [], 0, 0
        for item in merged:
            if ids is not None and item["id"] not in ids:
                continue
            labels = [a["primary_emotion"] for a in item.get("annotations", [])]
            if len(labels) != 3:
                continue
            total += 1
            for i in range(3):
                others = [labels[j] for j in range(3) if j != i]
                if others[0] != others[1]:
                    skipped += 1          # no reference available
                    continue
                preds.append(labels[i])
                refs.append(others[0])
        if not preds:
            continue
        rows.append({
            "scope": scope,
            "items": total,
            "judgements_used": len(preds),
            "judgements_skipped_no_reference": skipped,
            "human_macro_f1": round(f1_score(refs, preds, average="macro",
                                             labels=EMOTIONS, zero_division=0), 4),
            "human_accuracy": round(accuracy_score(refs, preds), 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "human_ceiling.csv", index=False)
    print("=== HUMAN CEILING (leave-one-annotator-out) ===")
    print(df.to_string(index=False))
    print("\nUse the 'test' row as the ceiling alongside model results —")
    print("it is computed on the same items the models were evaluated on.\n")


# ================================================== 2. SOURCE DIVERSITY
def source_diversity():
    data = json.load(open("data/final_annotation_set.json", encoding="utf-8"))
    recs = [t["data"] for t in data]
    df = pd.DataFrame(recs)

    overall = {
        "comments": len(df),
        "distinct_videos": df["video_id"].nunique(),
        "distinct_channels": df["channel"].nunique(),
        "distinct_genres": df["genre"].nunique(),
        "median_comments_per_video": float(df.groupby("video_id").size().median()),
        "max_comments_per_video": int(df.groupby("video_id").size().max()),
        "pct_from_top_10_videos": round(
            100 * df.groupby("video_id").size().nlargest(10).sum() / len(df), 2),
        "pct_from_top_10_channels": round(
            100 * df.groupby("channel").size().nlargest(10).sum() / len(df), 2),
    }
    print("=== SOURCE DIVERSITY ===")
    for k, v in overall.items():
        print(f"  {k:32s} {v}")

    per_lang = df.groupby("language_label").agg(
        comments=("id", "size"),
        videos=("video_id", "nunique"),
        channels=("channel", "nunique")).reset_index()
    print("\nby register:")
    print(per_lang.to_string(index=False))

    pd.concat([
        pd.DataFrame([overall]).assign(scope="overall"),
        per_lang.rename(columns={"language_label": "scope"})
    ], ignore_index=True).to_csv(OUT / "source_diversity.csv", index=False)

    print("\nThe concentration figures answer the reviewer question")
    print("'is this really one viral thread?' — keep them in the paper.\n")


# ================================================== 3. CO-OCCURRENCE
def _secondaries(result):
    """Pull other_emotions from a Label Studio style result list."""
    out = []
    for r in result:
        if r.get("from_name") == "other_emotions":
            out.extend(r.get("value", {}).get("choices") or [])
    return out


def cooccurrence():
    # secondary labels live in the LLM predictions; human rounds captured a
    # primary label per item, so co-occurrence is reported from predictions
    # over the curated 30,436 and flagged as such in the paper.
    src = Path("data/final_annotation_set.json")
    data = json.load(open(src, encoding="utf-8"))

    mat = pd.DataFrame(0, index=EMOTIONS, columns=EMOTIONS)
    n_with_secondary = 0
    n_total = 0
    for t in data:
        preds = t.get("predictions") or []
        if not preds:
            continue
        res = preds[0].get("result", [])
        prim = None
        for r in res:
            if r.get("from_name") == "primary_emotion":
                prim = (r.get("value", {}).get("choices") or [None])[0]
        if prim is None:
            continue
        n_total += 1
        secs = [s for s in _secondaries(res) if s != prim]
        if secs:
            n_with_secondary += 1
        for s in secs:
            if prim in mat.index and s in mat.columns:
                mat.loc[prim, s] += 1

    mat.to_csv(OUT / "cooccurrence.csv")
    print("=== PRIMARY x SECONDARY CO-OCCURRENCE (LLM labels, n=%d) ===" % n_total)
    print(f"items carrying at least one secondary emotion: {n_with_secondary} "
          f"({100*n_with_secondary/max(n_total,1):.1f}%)")
    print(mat.to_string())

    pairs = []
    for a in EMOTIONS:
        for b in EMOTIONS:
            if a != b:
                pairs.append((a, b, int(mat.loc[a, b])))
    pairs.sort(key=lambda x: -x[2])
    print("\nstrongest 10 pairs:")
    for a, b, n in pairs[:10]:
        print(f"  {a:10s} -> {b:10s} {n}")
    print("\nCaveat for the paper: these are model-assigned secondary labels,")
    print("not human ones. Report as an exploratory analysis, and compare the")
    print("Nostalgia-Sadness and Devotion-Love entries against the human")
    print("boundary confusions (39 and 59 pairs) from the agreement round.\n")


if __name__ == "__main__":
    human_ceiling()
    source_diversity()
    cooccurrence()
    print(f"tables written to {OUT}/")
