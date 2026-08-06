#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agreement_analysis.py  —  Pilot inter-annotator agreement & guideline triage
=============================================================================

Consumes a 3-rater export and answers the question the pilot exists to answer:
*which labels and which guideline distinctions are failing, and in what order
should I fix them?*

Inputs (auto-detected):
  (a) a Label Studio JSON export  -- list of tasks, each task.annotations[] a
      rater, each annotation.result[] keyed by the from_name contract
      (primary_emotion / other_emotions / emotion_carrying_language /
       code_switch_is_emotional / confidence / difficult / notes); OR
  (b) the project's native schema  -- records with annotations[] where each
      entry has primary_emotion / emotions[] / ... (the sample_dataset.json shape).

Outputs (to --outdir, default ./agreement_out):
  - report printed to stdout (human-readable)
  - per_label_metrics.csv     (κ, Spearman ρ, prevalence, difficult-rate)
  - confusion_primary.csv      (full pairwise primary-emotion confusion matrix)
  - no_majority_items.csv      (items to re-route to +2 raters)
  - summary.json               (everything, machine-readable)

Statistics — and WHY each (viva-defensible):
  * Fleiss' κ on primary_emotion  -> single headline agreement number across
    all 3 raters on the 9-way forced-primary choice.
  * Per-label Cohen's κ (binary present/absent, averaged over rater pairs)
    -> shows WHICH labels are hard, not just the global average.
  * Per-label Spearman ρ (each rater vs. the mean of the others) -> reported
    ALONGSIDE κ on purpose: κ can collapse toward 0 for rare labels even when
    raters agree (the "kappa paradox" / prevalence problem). Nostalgia, Fear
    and Devotion are low-prevalence by design, so κ alone would understate
    them; ρ is the cross-check.
  * Targeted confusion Nostalgia<->Sadness and Devotion<->Love -> the two
    thesis-critical distinctions. If these cells are hot, the novel labels are
    not yet separable and the GUIDELINES (anchors "needs warmth" / "needs
    reverence") need sharpening before scaling.
  * Code-mix fields (emotion_carrying_language, code_switch_is_emotional)
    -> percent agreement, because they are the project's novelty; the pilot
    must show raters can apply them consistently.

Deps: numpy, scipy, scikit-learn  (no statsmodels needed)
"""

from __future__ import annotations

import argparse
import json
import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

# Canonical label set (locked taxonomy). Order fixed for stable matrices.
LABELS = ["Love", "Joy", "Anger", "Sadness", "Fear",
          "Surprise", "Nostalgia", "Devotion", "Neutral"]
MULTILABEL = [l for l in LABELS if l != "Neutral"]  # other_emotions excludes Neutral

# The two thesis-critical distinctions to interrogate explicitly.
TARGET_PAIRS = [("Nostalgia", "Sadness"), ("Devotion", "Love")]


# --------------------------------------------------------------------------- #
# INTERNAL REPRESENTATION                                                      #
# --------------------------------------------------------------------------- #
class RaterLabel:
    __slots__ = ("rater", "primary", "emotions", "ecl", "csie",
                 "confidence", "difficult")

    def __init__(self, rater, primary, emotions, ecl, csie, confidence, difficult):
        self.rater = str(rater)
        self.primary = primary                # str in LABELS or None
        self.emotions = set(emotions)         # set incl. primary
        self.ecl = ecl                        # en/hi/both/neither or None
        self.csie = csie                      # bool or None
        self.confidence = confidence          # int 1-5 or None
        self.difficult = bool(difficult)


class Item:
    __slots__ = ("item_id", "text", "raters")

    def __init__(self, item_id, text):
        self.item_id = item_id
        self.text = text
        self.raters: list[RaterLabel] = []


# --------------------------------------------------------------------------- #
# LOADERS                                                                      #
# --------------------------------------------------------------------------- #
def _first_choice(value: dict):
    ch = value.get("choices")
    if ch:
        return ch[0]
    if "rating" in value:
        return value["rating"]
    if "text" in value:
        t = value["text"]
        return t[0] if isinstance(t, list) and t else t
    return None


def _all_choices(value: dict) -> list:
    return list(value.get("choices") or [])


def _parse_ls_annotation(ann: dict) -> RaterLabel:
    """One Label Studio annotation (= one rater's labeling of one task)."""
    by_name = defaultdict(list)
    for r in ann.get("result", []):
        by_name[r.get("from_name")].append(r.get("value", {}))

    def one(name):
        vals = by_name.get(name)
        return _first_choice(vals[0]) if vals else None

    primary = one("primary_emotion")
    others = []
    if by_name.get("other_emotions"):
        others = _all_choices(by_name["other_emotions"][0])
    emotions = set(others) | ({primary} if primary else set())

    ecl = one("emotion_carrying_language")
    csie_raw = one("code_switch_is_emotional")
    csie = None if csie_raw is None else str(csie_raw).strip().lower() in ("yes", "true", "1")
    conf = one("confidence")
    try:
        conf = int(conf) if conf is not None else None
    except (ValueError, TypeError):
        conf = None
    difficult = bool(by_name.get("difficult"))  # flag present == difficult

    rater = ann.get("completed_by")
    if isinstance(rater, dict):
        rater = rater.get("email") or rater.get("id")
    return RaterLabel(rater, primary, emotions, ecl, csie, conf, difficult)


def _parse_native_annotation(a: dict) -> RaterLabel:
    primary = a.get("primary_emotion")
    emotions = set(a.get("emotions") or []) | ({primary} if primary else set())
    return RaterLabel(
        rater=a.get("annotator_id"),
        primary=primary,
        emotions=emotions,
        ecl=a.get("emotion_carrying_language"),
        csie=a.get("code_switch_is_emotional"),
        confidence=a.get("confidence"),
        difficult=a.get("difficult", False),
    )


def load_items(path: Path) -> list[Item]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    items: list[Item] = []
    for i, task in enumerate(data):
        anns = task.get("annotations", [])
        is_ls = bool(anns) and isinstance(anns[0], dict) and "result" in anns[0]
        data_block = task.get("data", task)
        item_id = data_block.get("id") or task.get("id") or f"item_{i}"
        text = data_block.get("masked_text") or data_block.get("raw_text") or ""
        it = Item(item_id, text)
        for a in anns:
            it.raters.append(_parse_ls_annotation(a) if is_ls
                             else _parse_native_annotation(a))
        if it.raters:
            items.append(it)
    return items


# --------------------------------------------------------------------------- #
# STATISTICS                                                                   #
# --------------------------------------------------------------------------- #
def fleiss_kappa_primary(items: list[Item]) -> tuple[float | None, int, int]:
    """Fleiss' κ on primary_emotion. Requires a fixed number of raters/item,
    so we use the modal rater-count and report coverage."""
    counts = Counter(len([r for r in it.raters if r.primary]) for it in items)
    if not counts:
        return None, 0, 0
    n = counts.most_common(1)[0][0]
    if n < 2:
        return None, 0, n
    subset = [it for it in items
              if len([r for r in it.raters if r.primary]) == n]
    if not subset:
        return None, 0, n
    k = len(LABELS)
    idx = {lab: j for j, lab in enumerate(LABELS)}
    M = np.zeros((len(subset), k))
    for i, it in enumerate(subset):
        for r in it.raters:
            if r.primary in idx:
                M[i, idx[r.primary]] += 1
    N = len(subset)
    P_i = (np.sum(M ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    p_j = M.sum(axis=0) / (N * n)
    P_e = np.sum(p_j ** 2)
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 1e-12 else None
    return (float(kappa) if kappa is not None else None), N, n


def per_label_kappa_spearman(items: list[Item]):
    """For each label: averaged pairwise Cohen's κ (binary present/absent) and
    a rater-vs-consensus Spearman ρ. Returns dict label -> metrics."""
    out = {}
    for lab in LABELS:
        # Build, per rater, their binary judgments keyed by item.
        rater_vec = defaultdict(dict)   # rater -> {item_id: 0/1}
        prevalence_hits = total_judgments = 0
        for it in items:
            for r in it.raters:
                v = int(lab in r.emotions)
                rater_vec[r.rater][it.item_id] = v
                prevalence_hits += v
                total_judgments += 1

        # Pairwise Cohen's κ over shared items.
        kappas = []
        raters = list(rater_vec)
        for a, b in combinations(raters, 2):
            shared = set(rater_vec[a]) & set(rater_vec[b])
            if len(shared) < 2:
                continue
            ya = [rater_vec[a][i] for i in shared]
            yb = [rater_vec[b][i] for i in shared]
            if len(set(ya)) == 1 and len(set(yb)) == 1 and ya[0] == yb[0]:
                kappas.append(1.0)        # perfect, degenerate -> treat as 1
            else:
                try:
                    kappas.append(cohen_kappa_score(ya, yb))
                except Exception:
                    pass
        kappa = float(np.nanmean(kappas)) if kappas else None

        # Rater-vs-consensus Spearman: each rater's vector vs mean of others.
        rhos = []
        for a in raters:
            items_a = list(rater_vec[a])
            xs, ys = [], []
            for iid in items_a:
                others = [rater_vec[b][iid] for b in raters
                          if b != a and iid in rater_vec[b]]
                if not others:
                    continue
                xs.append(rater_vec[a][iid])
                ys.append(float(np.mean(others)))
            if len(xs) >= 3 and len(set(xs)) > 1 and len(set(ys)) > 1:
                rho, _ = spearmanr(xs, ys)
                if not np.isnan(rho):
                    rhos.append(rho)
        rho = float(np.mean(rhos)) if rhos else None

        out[lab] = {
            "kappa": kappa,
            "spearman": rho,
            "prevalence": prevalence_hits / total_judgments if total_judgments else 0.0,
            "n_positive_judgments": prevalence_hits,
        }
    return out


def primary_confusion(items: list[Item]) -> np.ndarray:
    """Symmetric pairwise confusion matrix over primary_emotion."""
    idx = {lab: j for j, lab in enumerate(LABELS)}
    C = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for it in items:
        prims = [r.primary for r in it.raters if r.primary in idx]
        for a, b in combinations(prims, 2):
            C[idx[a], idx[b]] += 1
            C[idx[b], idx[a]] += 1
    return C


def targeted_confusion(C: np.ndarray):
    idx = {lab: j for j, lab in enumerate(LABELS)}
    res = {}
    for x, y in TARGET_PAIRS:
        cross = int(C[idx[x], idx[y]])
        agree_x = int(C[idx[x], idx[x]])  # both raters chose x
        agree_y = int(C[idx[y], idx[y]])
        denom = cross + agree_x + agree_y
        confusion_share = cross / denom if denom else 0.0
        res[f"{x}<->{y}"] = {
            "cross_pairs": cross,
            "both_chose_first": agree_x,
            "both_chose_second": agree_y,
            "confusion_share": confusion_share,
        }
    return res


def difficult_rate_by_primary(items: list[Item]):
    diff = Counter()
    tot = Counter()
    for it in items:
        for r in it.raters:
            if r.primary:
                tot[r.primary] += 1
                diff[r.primary] += int(r.difficult)
    return {lab: (diff[lab] / tot[lab] if tot[lab] else 0.0) for lab in LABELS}


def categorical_pct_agreement(items: list[Item], attr: str):
    """Percent of rater-pairs agreeing on a categorical attribute."""
    agree = total = 0
    for it in items:
        vals = [getattr(r, attr) for r in it.raters if getattr(r, attr) is not None]
        for a, b in combinations(vals, 2):
            total += 1
            agree += int(a == b)
    return (agree / total) if total else None


def no_majority_items(items: list[Item]):
    flagged = []
    for it in items:
        prims = [r.primary for r in it.raters if r.primary]
        if not prims:
            continue
        top, cnt = Counter(prims).most_common(1)[0]
        if cnt <= len(prims) / 2:   # no strict majority
            flagged.append((it.item_id, dict(Counter(prims)), it.text))
    return flagged


# --------------------------------------------------------------------------- #
# GUIDELINE-TRIAGE RANKING                                                     #
# --------------------------------------------------------------------------- #
def rank_fix_first(per_label, difficult_rates, targeted):
    """Priority = which guidelines to sharpen first. Higher score = more urgent.
    Combines low agreement, high difficulty, and (for the two target pairs)
    high cross-confusion."""
    pair_penalty = defaultdict(float)
    for pair, m in targeted.items():
        a, b = pair.split("<->")
        pair_penalty[a] += m["confusion_share"]
        pair_penalty[b] += m["confusion_share"]

    rows = []
    for lab in LABELS:
        k = per_label[lab]["kappa"]
        s = per_label[lab]["spearman"]
        d = difficult_rates.get(lab, 0.0)
        # Missing metric -> treat as worst (1.0 disagreement) so it surfaces.
        dis_k = 1.0 - k if k is not None else 1.0
        dis_s = 1.0 - s if s is not None else 1.0
        score = 0.45 * dis_k + 0.25 * dis_s + 0.15 * d + 0.15 * pair_penalty[lab]
        reasons = []
        if k is not None and k < 0.4:
            reasons.append(f"low κ={k:.2f}")
        if s is not None and s < 0.4:
            reasons.append(f"low ρ={s:.2f}")
        if d > 0.15:
            reasons.append(f"difficult-rate={d:.0%}")
        if pair_penalty[lab] > 0.2:
            reasons.append("confused with its contrast label")
        rows.append((score, lab, reasons,
                     {"kappa": k, "spearman": s, "difficult_rate": d}))
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows


# --------------------------------------------------------------------------- #
# REPORT + OUTPUT                                                              #
# --------------------------------------------------------------------------- #
def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def run(path: Path, outdir: Path, raters_expected: int = 3):
    items = load_items(path)
    outdir.mkdir(parents=True, exist_ok=True)

    n_items = len(items)
    rater_counts = Counter(len(it.raters) for it in items)
    all_raters = sorted({r.rater for it in items for r in it.raters})

    fleiss, fl_cov, fl_n = fleiss_kappa_primary(items)
    per_label = per_label_kappa_spearman(items)
    C = primary_confusion(items)
    targeted = targeted_confusion(C)
    diff_rates = difficult_rate_by_primary(items)
    ecl_agree = categorical_pct_agreement(items, "ecl")
    csie_agree = categorical_pct_agreement(items, "csie")
    no_maj = no_majority_items(items)
    ranking = rank_fix_first(per_label, diff_rates, targeted)

    # ---- stdout report ----
    P = print
    P("=" * 70)
    P("PILOT AGREEMENT REPORT")
    P("=" * 70)
    P(f"items: {n_items}   raters seen: {len(all_raters)} ({', '.join(map(str, all_raters))})")
    P(f"raters-per-item distribution: {dict(rater_counts)}")
    P()
    P("HEADLINE")
    P(f"  Fleiss' κ (primary_emotion, {fl_n} raters/item, {fl_cov} items): {fmt(fleiss)}")
    P(f"  emotion_carrying_language  pairwise agreement: {fmt(ecl_agree)}")
    P(f"  code_switch_is_emotional   pairwise agreement: {fmt(csie_agree)}")
    P()
    P("PER-LABEL  (κ = pairwise Cohen, ρ = rater-vs-consensus Spearman)")
    P(f"  {'label':10s} {'κ':>6s} {'ρ':>6s} {'prev':>6s} {'diff':>6s}")
    for lab in LABELS:
        m = per_label[lab]
        P(f"  {lab:10s} {fmt(m['kappa']):>6s} {fmt(m['spearman']):>6s} "
          f"{m['prevalence']*100:5.1f}% {diff_rates[lab]*100:5.1f}%")
    P()
    P("THESIS-CRITICAL CONFUSION")
    for pair, m in targeted.items():
        P(f"  {pair:22s} cross-pairs={m['cross_pairs']:3d}  "
          f"confusion-share={m['confusion_share']:.0%}")
    P()
    P("FIX THESE GUIDELINES FIRST  (most urgent at top)")
    for rank, (score, lab, reasons, _m) in enumerate(ranking, 1):
        why = "; ".join(reasons) if reasons else "ok"
        P(f"  {rank}. {lab:10s} score={score:.2f}  [{why}]")
    P()
    if no_maj:
        P(f"NO-MAJORITY ITEMS to re-route to +2 raters: {len(no_maj)} "
          f"(see no_majority_items.csv)")
    P("=" * 70)

    # ---- files ----
    with (outdir / "per_label_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "cohen_kappa", "spearman_rho", "prevalence",
                    "difficult_rate", "n_positive_judgments"])
        for lab in LABELS:
            m = per_label[lab]
            w.writerow([lab, m["kappa"], m["spearman"], m["prevalence"],
                        diff_rates[lab], m["n_positive_judgments"]])

    with (outdir / "confusion_primary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + LABELS)
        for i, lab in enumerate(LABELS):
            w.writerow([lab] + list(map(int, C[i])))

    with (outdir / "no_majority_items.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "primary_vote_counts", "text"])
        for iid, votes, text in no_maj:
            w.writerow([iid, json.dumps(votes, ensure_ascii=False), text])

    summary = {
        "n_items": n_items,
        "raters": all_raters,
        "raters_per_item": dict(rater_counts),
        "fleiss_kappa_primary": fleiss,
        "fleiss_coverage_items": fl_cov,
        "ecl_pct_agreement": ecl_agree,
        "csie_pct_agreement": csie_agree,
        "per_label": per_label,
        "difficult_rate": diff_rates,
        "targeted_confusion": targeted,
        "fix_first_ranking": [
            {"rank": i + 1, "label": lab, "score": score, "reasons": reasons}
            for i, (score, lab, reasons, _m) in enumerate(ranking)
        ],
        "no_majority_item_ids": [iid for iid, _, _ in no_maj],
    }
    (outdir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Pilot inter-annotator agreement analysis")
    p.add_argument("export", help="Label Studio JSON export OR native-schema JSON")
    p.add_argument("-o", "--outdir", default="agreement_out")
    p.add_argument("--raters", type=int, default=3)
    args = p.parse_args(argv)
    run(Path(args.export), Path(args.outdir), args.raters)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
