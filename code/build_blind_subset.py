#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_blind_subset.py — blind agreement check, matched to testable cells.

Why this exists
---------------
Stage 1 showed every annotator all 30,436 curated items with the model's label
displayed. The gold subset was then drawn from those same items, so kappa on
it cannot be separated from recall of the earlier exposure. This script draws
a subset from pre-labelled items that were never curated and never annotated,
giving a contamination-free estimate.

Coverage limit
--------------
Fear and Surprise were exhausted during curation: no unlabelled items remain
in any variety. Anger-Devanagari is likewise empty. The blind check therefore
covers 20 of 27 cells and cannot speak to Fear or Surprise.

Because those two labels have high agreement (0.91, 0.89) and make up 36% of
the gold subset, a raw blind-vs-gold comparison would be confounded. This
script prints the gold kappa RESTRICTED TO THE SAME CELLS, so the comparison
is like-for-like.

  python build_blind_subset.py --n 500
"""
import argparse, json, random, collections, itertools
from pathlib import Path

random.seed(2024)
EMOTIONS = ["Love", "Joy", "Anger", "Sadness", "Fear",
            "Surprise", "Nostalgia", "Devotion", "Neutral"]
LANGS = ["English", "Hindi", "Hinglish"]
MIN_CELL = 30           # a cell needs this many spare items to be testable


def primary_of(task):
    for r in (task.get("predictions") or [{}])[0].get("result", []):
        if r.get("from_name") == "primary_emotion":
            return (r.get("value", {}).get("choices") or [None])[0]
    return None


def fleiss_kappa(items):
    """items: list of lists of labels (one list per item, 3 labels each)."""
    items = [i for i in items if len(i) == 3]
    n = len(items)
    if n == 0:
        return None, 0
    cats = sorted({l for i in items for l in i})
    idx = {c: k for k, c in enumerate(cats)}
    counts = [[0] * len(cats) for _ in range(n)]
    for r, lab in enumerate(items):
        for l in lab:
            counts[r][idx[l]] += 1
    m = 3
    P_i = [(sum(c * c for c in row) - m) / (m * (m - 1)) for row in counts]
    P_bar = sum(P_i) / n
    p_j = [sum(counts[r][j] for r in range(n)) / (n * m) for j in range(len(cats))]
    P_e = sum(p * p for p in p_j)
    if P_e >= 1:
        return 1.0, n
    return (P_bar - P_e) / (1 - P_e), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    curated = json.load(open("data/final_annotation_set.json", encoding="utf-8"))
    curated_ids = {t["data"]["id"] for t in curated}

    src = Path("data/main_prelabel.json")
    if not src.exists():
        print(f"ERROR: {src} not found.")
        return
    pool = [t for t in json.load(open(src, encoding="utf-8"))
            if t["data"].get("id") not in curated_ids]

    cells = collections.defaultdict(list)
    for t in pool:
        e, lg = primary_of(t), t["data"].get("language_label")
        if e in EMOTIONS and lg in LANGS:
            cells[(e, lg)].append(t)

    testable = {k: v for k, v in cells.items() if len(v) >= MIN_CELL}
    missing = [(e, l) for e in EMOTIONS for l in LANGS
               if (e, l) not in testable]

    print(f"eligible pool: {len(pool)}")
    print(f"testable cells: {len(testable)} of 27")
    print(f"cells with no usable spare items ({len(missing)}):")
    for e, l in missing:
        print(f"    {e}-{l}  (spare: {len(cells.get((e,l), []))})")

    # ---- draw, proportional to the GOLD subset composition on testable cells
    gold = json.load(open("data/gold_dataset.json", encoding="utf-8"))
    gold_cells = collections.Counter(
        (g["gold_emotion"], g["language_label"]) for g in gold)
    gold_testable_total = sum(v for k, v in gold_cells.items() if k in testable)

    subset = []
    for key in testable:
        share = round(args.n * gold_cells.get(key, 0) / max(gold_testable_total, 1))
        share = max(5, min(share, len(testable[key])))
        random.shuffle(testable[key])
        subset.extend(testable[key][:share])
    random.shuffle(subset)

    c = collections.Counter((primary_of(t), t["data"].get("language_label"))
                            for t in subset)
    print(f"\nBLIND SUBSET (n={len(subset)}), matched to gold composition:")
    print(f"{'emotion':10s}" + "".join(f"{l:>10s}" for l in LANGS) + f"{'TOTAL':>8s}")
    for e in EMOTIONS:
        vals = [c.get((e, l), 0) for l in LANGS]
        if sum(vals):
            print(f"{e:10s}" + "".join(f"{v:>10d}" for v in vals) + f"{sum(vals):>8d}")

    clean = [{"data": t["data"]} for t in subset]
    for i in (1, 2, 3):
        p = Path(f"data/blind_r{i}.json")
        json.dump(clean, open(p, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"  wrote {p}")
    json.dump(subset, open("data/blind_subset_withpred.json", "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)
    print("  wrote data/blind_subset_withpred.json (predictions kept — do NOT import)")

    # ---- matched baseline: gold kappa on the same cells ----
    merged = json.load(open("data/gold_merged.json", encoding="utf-8"))
    gold_lang = {g["id"]: (g["gold_emotion"], g["language_label"]) for g in gold}
    all_items, matched_items = [], []
    for it in merged:
        labs = [a["primary_emotion"] for a in it.get("annotations", [])]
        if len(labs) != 3:
            continue
        all_items.append(labs)
        if gold_lang.get(it["id"]) in testable:
            matched_items.append(labs)

    k_all, n_all = fleiss_kappa(all_items)
    k_match, n_match = fleiss_kappa(matched_items)
    print("\n=== BASELINE FOR COMPARISON ===")
    print(f"  gold kappa, all cells          : {k_all:.3f}  (n={n_all})")
    print(f"  gold kappa, testable cells only: {k_match:.3f}  (n={n_match})")
    print(f"\nCompare the blind subset's kappa against {k_match:.3f}, NOT")
    print(f"against {k_all:.3f}. The difference between the two baselines is")
    print("composition, not contamination.")
    print("\nAfter annotation:")
    print("  python merge_exports.py exports/blind_r1.json exports/blind_r2.json "
          "exports/blind_r3.json -o data/blind_merged.json")
    print("  python agreement_analysis.py data/blind_merged.json -o agreement_out_blind")


if __name__ == "__main__":
    main()
