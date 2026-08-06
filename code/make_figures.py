#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py — paper figures.

Figure 1 distinguishes three situations, matching Section 4.4 and 6.1:
  * cells at the per-cell target                      (neutral)
  * cells below target because the pool ran out       (darker shading)
  * Fear and Surprise, exempt from the cap, where the
    number shown is the complete available pool       (light shading)

The earlier version shaded purely on "below 1,200", which implied that
Fear-Hinglish (1,201) had met a target while Fear-English (1,191) had not.
Both are complete pools; the distinction was an artefact.

Figure 2 puts the annotator and model confusion matrices on one shared scale
with a single colourbar, so the two panels are directly comparable.

Outputs 300-dpi PNG and vector PDF to figures/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG = Path("figures"); FIG.mkdir(exist_ok=True)

EMOTIONS = ["Love", "Joy", "Anger", "Sadness", "Fear",
            "Surprise", "Nostalgia", "Devotion", "Neutral"]
NOVEL = {"Nostalgia", "Devotion"}
TAKE_ALL = {"Fear", "Surprise"}          # exempt from the per-cell cap
LANGS = ["English", "Hindi", "Hinglish"]
TARGET = {e: 1200 for e in EMOTIONS}
TARGET["Neutral"] = 1500
BEST_MODEL_DIR = Path("results/xlmr_seed42")

INK        = "#22303f"
MUTED      = "#8a97a5"
AT_TARGET  = "#dde6ee"
POOL       = "#fdf0e4"     # exempt from cap: complete pool
POOL_EDGE  = "#e0a878"
BELOW      = "#f6d9c8"     # capped emotion, pool ran out
BELOW_EDGE = "#c2703f"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print(f"  figures/{name}.png / .pdf")


# ------------------------------------------------------------------ F1
def fig1():
    data = json.load(open("data/final_annotation_set.json", encoding="utf-8"))
    rows = []
    for t in data:
        prim = None
        for r in (t.get("predictions") or [{}])[0].get("result", []):
            if r.get("from_name") == "primary_emotion":
                prim = (r.get("value", {}).get("choices") or [None])[0]
        if prim:
            rows.append((prim, t["data"].get("language_label")))
    df = pd.DataFrame(rows, columns=["emotion", "lang"])

    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    n_r, n_c = len(EMOTIONS), len(LANGS)

    for i, e in enumerate(EMOTIONS):
        for j, lg in enumerate(LANGS):
            v = int(((df.emotion == e) & (df.lang == lg)).sum())
            if e in TAKE_ALL:
                face, edge, txt, emph = POOL, POOL_EDGE, BELOW_EDGE, True
            elif v < TARGET[e]:
                face, edge, txt, emph = BELOW, BELOW_EDGE, BELOW_EDGE, True
            else:
                face, edge, txt, emph = AT_TARGET, "white", INK, False
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor=edge,
                linewidth=1.4 if emph else 1.2, zorder=1))
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                    zorder=3, color=txt,
                    fontweight="bold" if emph else "normal")

    ax.set_xlim(-0.5, n_c - 0.5)
    ax.set_ylim(n_r - 0.5, -0.5)
    ax.set_xticks(range(n_c)); ax.set_xticklabels(LANGS)
    ax.set_yticks(range(n_r))
    for lab, e in zip(ax.set_yticklabels(EMOTIONS), EMOTIONS):
        if e in NOVEL:
            lab.set_fontweight("bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("variety", labelpad=8, color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.text(-0.5, n_r - 0.05,
            "light shading = exempt from the cap, complete pool shown\n"
            "darker shading = capped category, pool exhausted below target\n"
            "bold row labels = added categories",
            fontsize=7.5, color=MUTED, va="top", linespacing=1.5)
    save(fig, "fig1_composition")


# ------------------------------------------------------------------ F2
def _row_norm(m):
    s = m.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, m / s, np.nan)


def _human_confusion():
    merged = json.load(open("data/gold_merged.json", encoding="utf-8"))
    gold = {g["id"]: g["gold_emotion"]
            for g in json.load(open("data/gold_dataset.json", encoding="utf-8"))}
    idx = {e: i for i, e in enumerate(EMOTIONS)}
    m = np.zeros((len(EMOTIONS), len(EMOTIONS)))
    for it in merged:
        g = gold.get(it["id"])
        if g is None:
            continue
        for a in it.get("annotations", []):
            lab = a["primary_emotion"]
            if g in idx and lab in idx:
                m[idx[g], idx[lab]] += 1
    return _row_norm(m)


def _model_confusion():
    p = BEST_MODEL_DIR / "test_confusion.csv"
    if not p.exists():
        return None
    cm = pd.read_csv(p, index_col=0).reindex(index=EMOTIONS, columns=EMOTIONS)
    return _row_norm(cm.values.astype(float))


def _panel(ax, mat, title, cmap):
    """Every cell is annotated. Values that round to .00 are shown as 0 in a
    lighter tone, so a blank cell never has to be interpreted."""
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center",
                        fontsize=6.5, color=MUTED)
                continue
            if v < 0.005:
                ax.text(j, i, "0", ha="center", va="center",
                        fontsize=6, color="#b9c4ce")
                continue
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=6.5, color="white" if v > 0.55 else INK)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS, fontsize=8)
    ax.set_title(title, pad=10, fontweight="bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig2():
    human = _human_confusion()
    model = _model_confusion()
    if model is None:
        print("  [skip] model confusion matrix not found")
        return
    blues = LinearSegmentedColormap.from_list("b", ["#ffffff", "#2c5f88"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    _panel(axes[0], human, "Annotators", blues)
    im = _panel(axes[1], model, "Best model (XLM-R)", blues)
    axes[0].set_ylabel("gold label")
    for ax in axes:
        ax.set_xlabel("assigned label")
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("share of row", color=MUTED)
    save(fig, "fig2_confusion_human_vs_model")

    mask = ~np.eye(len(EMOTIONS), dtype=bool)
    h, m = human[mask], model[mask]
    ok = ~(np.isnan(h) | np.isnan(m))
    if ok.sum() > 2:
        print(f"\n  off-diagonal correlation, human vs model: "
              f"r = {np.corrcoef(h[ok], m[ok])[0,1]:.3f}")


if __name__ == "__main__":
    print("generating figures ...")
    fig1()
    fig2()
    print(f"\nwritten to {FIG}/")
