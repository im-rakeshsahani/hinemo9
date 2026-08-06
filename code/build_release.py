#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_release.py — assemble the public distribution.

Writes release/ containing:
  data/train.jsonl  dev.jsonl  test.jsonl     the benchmark splits
  data/gold.jsonl                             3,363 gold items + rater labels
  data/full.jsonl                             all 30,436 curated items
  data/checksums.txt                          sha256 of every data file
  reference/                                  per-cell tables, agreement reports

PRIVACY: raw_text is dropped everywhere. Only masked_text (PII-substituted)
is published. The script verifies this at the end and says so.

  python build_release.py
"""
import argparse, hashlib, json, shutil
from pathlib import Path

REL = Path("release")

PUBLIC_FIELDS = [
    "id", "text", "label", "language_label", "genre",
    "label_source", "n_annotators", "gold_source",
    "rater_labels", "llm_prediction", "code_mix_index", "n_tokens",
]


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def clean(rec, include_raw=False):
    out = {k: rec[k] for k in PUBLIC_FIELDS if k in rec}
    if include_raw and "raw_text" in rec:
        out["raw_text"] = rec["raw_text"]
    return out


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path}  ({len(rows)} rows)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def primary_of(task):
    for r in (task.get("predictions") or [{}])[0].get("result", []):
        if r.get("from_name") == "primary_emotion":
            return (r.get("value", {}).get("choices") or [None])[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-raw", action="store_true",
                    help="publish unmasked raw_text (NOT recommended)")
    args = ap.parse_args()

    if REL.exists():
        shutil.rmtree(REL)
    (REL / "data").mkdir(parents=True)
    (REL / "reference").mkdir(parents=True)

    counts = {}

    print("splits:")
    for name in ["train", "dev", "test"]:
        src = Path("data/splits") / f"{name}.jsonl"
        if not src.exists():
            print(f"  [missing] {src}")
            continue
        rows = [clean(r, args.include_raw) for r in load_jsonl(src)]
        write_jsonl(rows, REL / "data" / f"{name}.jsonl")
        counts[name] = len(rows)

    print("gold subset:")
    gp = Path("data/gold_dataset.json")
    if gp.exists():
        gold = json.load(open(gp, encoding="utf-8"))
        grows = [{
            "id": g["id"],
            "text": g.get("masked_text", ""),
            "label": g["gold_emotion"],
            "language_label": g.get("language_label"),
            "genre": g.get("genre"),
            "label_source": "human_gold",
            "gold_source": g.get("gold_source"),
            "n_annotators": 3,
            "rater_labels": g.get("rater_labels"),
            "llm_prediction": g.get("llm_prediction"),
        } for g in gold]
        write_jsonl(grows, REL / "data" / "gold.jsonl")
        counts["gold"] = len(grows)
    else:
        print(f"  [missing] {gp}")

    print("full curated set:")
    fp = Path("data/final_annotation_set.json")
    if fp.exists():
        full = json.load(open(fp, encoding="utf-8"))
        frows = []
        for t in full:
            d = t["data"]
            rec = {
                "id": d.get("id"),
                "text": d.get("masked_text", ""),
                "label": primary_of(t),
                "language_label": d.get("language_label"),
                "genre": d.get("genre"),
                "code_mix_index": d.get("code_mix_index"),
                "n_tokens": d.get("n_tokens"),
                "label_source": "llm",
            }
            if args.include_raw:
                rec["raw_text"] = d.get("raw_text")
            frows.append(rec)
        write_jsonl(frows, REL / "data" / "full.jsonl")
        counts["full"] = len(frows)
    else:
        print(f"  [missing] {fp}")

    print("reference tables:")
    for src in [Path("paper_tables"), Path("agreement_out_gold"),
                Path("agreement_out_blind")]:
        if src.exists():
            for f in src.glob("*.csv"):
                shutil.copy(f, REL / "reference" / f"{src.name}__{f.name}")
                print(f"  reference/{src.name}__{f.name}")

    for doc in ["DATASET_CARD.md", "README.md", "LICENSE", "CITATION.cff",
                "HinEmo9_Annotation_Guidelines.md"]:
        if Path(doc).exists():
            shutil.copy(doc, REL / doc)
            print(f"  {doc}")

    print("checksums:")
    with open(REL / "data" / "checksums.txt", "w", encoding="utf-8") as f:
        for p in sorted((REL / "data").glob("*.jsonl")):
            f.write(f"{sha256(p)}  {p.name}\n")
    print(f"  {REL/'data'/'checksums.txt'}")

    print("\nrelease summary:", counts)

    leaked = None
    for p in (REL / "data").glob("*.jsonl"):
        with open(p, encoding="utf-8") as f:
            if any('"raw_text"' in line for line in f):
                leaked = p.name
                break
    if leaked:
        print(f"\n*** raw_text FOUND in {leaked} — do not publish this folder ***")
    else:
        print("privacy check: no raw_text in any published file — OK")


if __name__ == "__main__":
    main()
