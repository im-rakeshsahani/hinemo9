#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_results.py — turn results/ into paper-ready tables.

Produces
  paper_tables/table_main.csv        model x macro-F1 mean+-std, accuracy, per-language
  paper_tables/table_per_emotion.csv model x emotion F1 mean+-std
  paper_tables/table_per_cell.csv    model x emotion x language F1 mean+-std with n
  paper_tables/significance.csv      pairwise paired t-test + Wilcoxon across seeds
  paper_tables/bootstrap_ci.csv      95% CI on macro-F1 per model (from saved predictions)

Handles duplicate rows in all_results.csv (e.g. a model re-run with new
hyperparameters) by keeping the LAST entry per (model, seed).
"""
import json, itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

RESULTS = Path("results")
OUT = Path("paper_tables")
OUT.mkdir(exist_ok=True)

EMOTIONS = ["Love", "Joy", "Anger", "Sadness", "Fear",
            "Surprise", "Nostalgia", "Devotion", "Neutral"]
LANGS = ["English", "Hindi", "Hinglish"]
N_BOOT = 2000
rng = np.random.default_rng(42)


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


# ---------------------------------------------------------------- main table
master = pd.read_csv(RESULTS / "all_results.csv")
# a re-run appends a second row for the same (model, seed); keep the latest
master = master.drop_duplicates(subset=["model", "seed"], keep="last")

rows = []
for model, grp in master.groupby("model"):
    f1s = grp["test_macro_f1"].values
    accs = grp["test_accuracy"].values
    rows.append({
        "model": model,
        "n_seeds": len(f1s),
        "macro_f1": fmt(f1s.mean(), f1s.std(ddof=1) if len(f1s) > 1 else 0.0),
        "macro_f1_mean": f1s.mean(),
        "accuracy": fmt(accs.mean(), accs.std(ddof=1) if len(accs) > 1 else 0.0),
    })
main = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)

# per-language columns, averaged across seeds
lang_cols = {lg: [] for lg in LANGS}
for _, r in main.iterrows():
    per_model = []
    for seed in master[master["model"] == r["model"]]["seed"]:
        p = RESULTS / f"{r['model']}_seed{seed}" / "test_per_language.csv"
        if p.exists():
            per_model.append(pd.read_csv(p).set_index("language")["macro_f1"])
    if per_model:
        d = pd.concat(per_model, axis=1)
        for lg in LANGS:
            if lg in d.index:
                v = d.loc[lg].values
                lang_cols[lg].append(fmt(v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0))
            else:
                lang_cols[lg].append("")
    else:
        for lg in LANGS:
            lang_cols[lg].append("")
for lg in LANGS:
    main[lg] = lang_cols[lg]

main.drop(columns=["macro_f1_mean"]).to_csv(OUT / "table_main.csv", index=False)
print("=== MAIN TABLE ===")
print(main.drop(columns=["macro_f1_mean"]).to_string(index=False))


# ------------------------------------------------------------ per-emotion
emo_rows = []
for model, grp in master.groupby("model"):
    frames = []
    for seed in grp["seed"]:
        p = RESULTS / f"{model}_seed{seed}" / "test_per_emotion.csv"
        if p.exists():
            frames.append(pd.read_csv(p).set_index("emotion")["f1"])
    if not frames:
        continue
    d = pd.concat(frames, axis=1)
    row = {"model": model}
    for e in EMOTIONS:
        if e in d.index:
            v = d.loc[e].values
            row[e] = fmt(v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)
    emo_rows.append(row)
per_emo = pd.DataFrame(emo_rows)
per_emo.to_csv(OUT / "table_per_emotion.csv", index=False)
print("\n=== PER-EMOTION F1 ===")
print(per_emo.to_string(index=False))


# --------------------------------------------------------------- per-cell
cell_rows = []
for model, grp in master.groupby("model"):
    frames = []
    for seed in grp["seed"]:
        p = RESULTS / f"{model}_seed{seed}" / "test_per_cell.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        continue
    allc = pd.concat(frames)
    g = allc.groupby(["emotion", "language"]).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), n=("n", "first")).reset_index()
    for _, r in g.iterrows():
        cell_rows.append({
            "model": model, "emotion": r["emotion"], "language": r["language"],
            "n": int(r["n"]),
            "f1": fmt(r["f1_mean"], 0.0 if pd.isna(r["f1_std"]) else r["f1_std"]),
            "reliable": "yes" if r["n"] >= 30 else "INDICATIVE (n<30)",
        })
pd.DataFrame(cell_rows).to_csv(OUT / "table_per_cell.csv", index=False)
print(f"\nper-cell table -> {OUT/'table_per_cell.csv'}")


# ---------------------------------------------------------- significance
sig = []
models = list(master["model"].unique())
for a, b in itertools.combinations(models, 2):
    fa = master[master["model"] == a].sort_values("seed")["test_macro_f1"].values
    fb = master[master["model"] == b].sort_values("seed")["test_macro_f1"].values
    if len(fa) != len(fb) or len(fa) < 2:
        continue
    t, pt = stats.ttest_rel(fa, fb)
    try:
        w, pw = stats.wilcoxon(fa, fb)
    except ValueError:
        w, pw = float("nan"), float("nan")
    sig.append({
        "model_a": a, "model_b": b,
        "mean_a": round(fa.mean(), 4), "mean_b": round(fb.mean(), 4),
        "diff": round(fa.mean() - fb.mean(), 4),
        "t_stat": round(float(t), 3), "p_ttest": round(float(pt), 4),
        "p_wilcoxon": round(float(pw), 4) if pw == pw else "",
        "significant_05": "yes" if pt < 0.05 else "no",
    })
sigdf = pd.DataFrame(sig)
sigdf.to_csv(OUT / "significance.csv", index=False)
print("\n=== PAIRWISE SIGNIFICANCE (paired across seeds) ===")
if len(sigdf):
    print(sigdf.to_string(index=False))
print("\nNote: with 3 seeds these tests have very low power. Treat p-values as")
print("indicative; a non-significant result is not evidence of equivalence.")


# ------------------------------------------------------------- bootstrap
boot_rows = []
for model, grp in master.groupby("model"):
    seed = sorted(grp["seed"])[0]
    p = RESULTS / f"{model}_seed{seed}" / "test_predictions.jsonl"
    if not p.exists():
        continue
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    gold = np.array([r["gold"] for r in recs])
    pred = np.array([r["pred"] for r in recs])
    n = len(gold)
    stats_ = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        stats_.append(f1_score(gold[idx], pred[idx], average="macro",
                               labels=EMOTIONS, zero_division=0))
    lo, hi = np.percentile(stats_, [2.5, 97.5])
    boot_rows.append({
        "model": model, "seed_used": seed,
        "macro_f1": round(f1_score(gold, pred, average="macro",
                                   labels=EMOTIONS, zero_division=0), 4),
        "ci_low": round(lo, 4), "ci_high": round(hi, 4),
        "ci_width": round(hi - lo, 4),
    })
bdf = pd.DataFrame(boot_rows)
bdf.to_csv(OUT / "bootstrap_ci.csv", index=False)
print("\n=== BOOTSTRAP 95% CI (single seed, 2000 resamples) ===")
if len(bdf):
    print(bdf.to_string(index=False))

print(f"\nall tables written to {OUT}/")
