#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_appendix_b.py — build the per-cell results table for Appendix B.

Reads paper_tables/table_per_cell.csv (written by aggregate_results.py),
filters to one model, and emits an emotion x variety table with each cell
showing F1 followed by n in parentheses. Cells with n < 30 are marked so a
reader can see which numbers rest on thin support.

  python make_appendix_b.py                 # defaults to xlmr
  python make_appendix_b.py --model muril

Writes paper_tables/appendix_b.md (paste into the paper) and
paper_tables/appendix_b.csv.
"""
import argparse
from pathlib import Path

import pandas as pd

EMOTIONS = ["Love", "Joy", "Anger", "Sadness", "Fear",
            "Surprise", "Nostalgia", "Devotion", "Neutral"]
LANGS = ["English", "Hindi", "Hinglish"]
THIN = 30
OUT = Path("paper_tables")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlmr")
    ap.add_argument("--src", default="paper_tables/table_per_cell.csv")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: {src} not found. Run aggregate_results.py first.")
        return

    df = pd.read_csv(src)
    sub = df[df["model"] == args.model]
    if sub.empty:
        print(f"ERROR: no rows for model '{args.model}'.")
        print("available:", sorted(df["model"].unique()))
        return

    # f1 column may be "0.732 ± 0.021" or a bare float
    def f1_of(v):
        s = str(v)
        return s.split("±")[0].strip() if "±" in s else s

    grid, thin_cells = {}, []
    for _, r in sub.iterrows():
        e, lg, n = r["emotion"], r["language"], int(r["n"])
        cell = f"{f1_of(r['f1'])} ({n})"
        if n < THIN:
            cell += "†"
            thin_cells.append(f"{e}-{lg}")
        grid[(e, lg)] = cell

    # ---- markdown ----
    lines = [f"| Emotion | {' | '.join(LANGS)} |",
             "|---" * (len(LANGS) + 1) + "|"]
    for e in EMOTIONS:
        row = [grid.get((e, lg), "—") for lg in LANGS]
        lines.append(f"| {e} | {' | '.join(row)} |")
    md = "\n".join(lines)

    caption = (
        f"**Table B1:** Per-emotion, per-variety F1 for {args.model.upper()} on "
        f"the human-gold test split, mean over three seeds, with cell counts in "
        f"parentheses. Cells marked † have n < {THIN}; at that support the "
        f"95% confidence interval exceeds ±0.20, so these values are indicative "
        f"only."
    )

    OUT.mkdir(exist_ok=True)
    (OUT / "appendix_b.md").write_text(caption + "\n\n" + md + "\n",
                                       encoding="utf-8")

    wide = pd.DataFrame(
        [[grid.get((e, lg), "") for lg in LANGS] for e in EMOTIONS],
        index=EMOTIONS, columns=LANGS)
    wide.to_csv(OUT / "appendix_b.csv")

    print(caption)
    print()
    print(md)
    print()
    print(f"thin cells (n < {THIN}): {', '.join(thin_cells) if thin_cells else 'none'}")
    print(f"\nwritten to {OUT}/appendix_b.md and appendix_b.csv")


if __name__ == "__main__":
    main()
