#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_svm.py — classical non-neural floor for the benchmark table.

Character n-grams are used rather than word n-grams because the corpus mixes
Devanagari and romanized Hindi, where word-level features fragment badly and
the same word appears in two scripts. char_wb (2,5) is the standard choice for
code-mixed Indic text.

Writes the same artefacts as train_model.py so aggregate_results.py picks it
up without modification.

  python train_svm.py --seed 42
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (f1_score, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

EMOTIONS = ["Love", "Joy", "Anger", "Sadness", "Fear",
            "Surprise", "Nostalgia", "Devotion", "Neutral"]
LANGS = ["English", "Hindi", "Hinglish"]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate(preds, golds, rows, outdir, tag):
    macro = f1_score(golds, preds, average="macro", labels=EMOTIONS, zero_division=0)
    acc = accuracy_score(golds, preds)
    weighted = f1_score(golds, preds, average="weighted", labels=EMOTIONS, zero_division=0)
    print(f"\n[{tag}]  accuracy={acc:.4f}  macro-F1={macro:.4f}  weighted-F1={weighted:.4f}")

    p, r, f, s = precision_recall_fscore_support(
        golds, preds, labels=EMOTIONS, zero_division=0)
    per_emo = pd.DataFrame({"emotion": EMOTIONS, "precision": p,
                            "recall": r, "f1": f, "support": s})
    print("\nper-emotion:")
    print(per_emo.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    langs = [x["language_label"] for x in rows]
    d = pd.DataFrame({"gold": golds, "pred": preds, "lang": langs})
    lang_rows = []
    for lg in LANGS:
        sub = d[d["lang"] == lg]
        if len(sub):
            lang_rows.append({
                "language": lg, "n": len(sub),
                "macro_f1": f1_score(sub["gold"], sub["pred"], average="macro",
                                     labels=EMOTIONS, zero_division=0),
                "accuracy": accuracy_score(sub["gold"], sub["pred"])})
    per_lang = pd.DataFrame(lang_rows)
    print("\nper-language:")
    print(per_lang.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    cell_rows = []
    for e in EMOTIONS:
        for lg in LANGS:
            sub = d[d["lang"] == lg]
            n = int((sub["gold"] == e).sum())
            if n == 0:
                cell_rows.append({"emotion": e, "language": lg, "n": 0,
                                  "f1": float("nan")})
                continue
            yt = (sub["gold"] == e).astype(int)
            yp = (sub["pred"] == e).astype(int)
            cell_rows.append({"emotion": e, "language": lg, "n": n,
                              "f1": f1_score(yt, yp, zero_division=0)})
    per_cell = pd.DataFrame(cell_rows)

    outdir.mkdir(parents=True, exist_ok=True)
    per_emo.to_csv(outdir / f"{tag}_per_emotion.csv", index=False)
    per_lang.to_csv(outdir / f"{tag}_per_language.csv", index=False)
    per_cell.to_csv(outdir / f"{tag}_per_cell.csv", index=False)
    pd.DataFrame(confusion_matrix(golds, preds, labels=EMOTIONS),
                 index=EMOTIONS, columns=EMOTIONS).to_csv(
        outdir / f"{tag}_confusion.csv")
    with open(outdir / f"{tag}_predictions.jsonl", "w", encoding="utf-8") as f:
        for pi, gi, row in zip(preds, golds, rows):
            f.write(json.dumps({"id": row["id"], "gold": gi, "pred": pi,
                                "language_label": row["language_label"]},
                               ensure_ascii=False) + "\n")
    with open(outdir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "macro_f1": macro,
                   "weighted_f1": weighted}, f, indent=2)
    return {"accuracy": acc, "macro_f1": macro, "weighted_f1": weighted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="data/splits")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    d = Path(args.data)
    train = load(d / "train.jsonl")
    dev = load(d / "dev.jsonl")
    test = load(d / "test.jsonl")
    print(f"train={len(train)}  dev={len(dev)}  test={len(test)}")

    Xtr = [r["text"] for r in train]; ytr = [r["label"] for r in train]
    Xdv = [r["text"] for r in dev];   ydv = [r["label"] for r in dev]
    Xte = [r["text"] for r in test];  yte = [r["label"] for r in test]

    clf = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                        min_df=2, max_features=300_000, sublinear_tf=True),
        LinearSVC(C=1.0, class_weight="balanced",
                  random_state=args.seed, max_iter=5000),
    )
    print("fitting TF-IDF(char_wb 2-5) + LinearSVC ...")
    clf.fit(Xtr, ytr)

    outdir = Path(args.out) / f"svm_seed{args.seed}"
    dev_res = evaluate(list(clf.predict(Xdv)), ydv, dev, outdir, "dev")
    test_res = evaluate(list(clf.predict(Xte)), yte, test, outdir, "test")

    master = Path(args.out) / "all_results.csv"
    row = {"model": "svm", "seed": args.seed,
           "dev_macro_f1": dev_res["macro_f1"],
           **{f"test_{k}": v for k, v in test_res.items()}}
    pd.DataFrame([row]).to_csv(master, mode="a", index=False,
                               header=not master.exists())
    print(f"\nwritten -> {outdir}\nappended -> {master}")


if __name__ == "__main__":
    main()
