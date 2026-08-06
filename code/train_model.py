#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py — baseline trainer for the HinEmo-9 dataset.

One script, every baseline. Selects the model by name, trains with
class-weighted loss, evaluates on the held-out human-gold test split, and
writes per-emotion and per-emotion x per-language metrics to CSV so the
results table assembles itself across runs.

Usage
-----
  python train_model.py --model muril      --seed 42
  python train_model.py --model xlmr       --seed 42
  python train_model.py --model mbert      --seed 42
  python train_model.py --model muril_hybrid --seed 42

  # repeat with --seed 43 --seed 44 for mean +- std

Tuned for a 6 GB GPU (RTX 4050): batch 16, max_len 128, fp16.
If you hit CUDA OOM, drop to --batch 8 --grad_accum 2 (same effective batch).
"""
import argparse, json, os, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             accuracy_score, classification_report,
                             confusion_matrix)
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# --------------------------------------------------------------------------
MODELS = {
    "muril":        "google/muril-base-cased",
    "xlmr":         "xlm-roberta-base",
    "mbert":        "bert-base-multilingual-cased",
    "indicbert":    "ai4bharat/IndicBERTv2-MLM-only",
    # hybrid variants: transformer encoder + BiLSTM + BiGRU over the sequence
    "muril_hybrid": "google/muril-base-cased",
    "xlmr_hybrid":  "xlm-roberta-base",
}
EMOTIONS = ["Love", "Joy", "Anger", "Sadness", "Fear",
            "Surprise", "Nostalgia", "Devotion", "Neutral"]
LANGS = ["English", "Hindi", "Hinglish"]
E2I = {e: i for i, e in enumerate(EMOTIONS)}


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_split(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class EmoDataset(Dataset):
    def __init__(self, rows, tok, max_len):
        self.rows, self.tok, self.max_len = rows, tok, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["text"], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(E2I[r["label"]], dtype=torch.long),
            "idx": i,
        }


class Classifier(nn.Module):
    """Plain transformer + linear head on the CLS token."""

    def __init__(self, name, n_labels, dropout=0.2):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(h, n_labels)

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]           # CLS
        return self.fc(self.drop(cls))


class HybridClassifier(nn.Module):
    """Transformer + BiLSTM + BiGRU, concatenated with the CLS vector.

    Sequence output feeds a BiLSTM, then a BiGRU; the final forward/backward
    hidden states are concatenated with the transformer CLS representation.
    """

    def __init__(self, name, n_labels, rnn_hidden=128, dropout=0.3):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.lstm = nn.LSTM(h, rnn_hidden, batch_first=True,
                            bidirectional=True)
        self.gru = nn.GRU(rnn_hidden * 2, rnn_hidden, batch_first=True,
                          bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(h + rnn_hidden * 2, n_labels)

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        cls = seq[:, 0]
        lstm_out, _ = self.lstm(seq)
        _, gru_h = self.gru(lstm_out)               # gru_h: (2, B, H)
        gru_cat = torch.cat([gru_h[0], gru_h[1]], dim=1)
        joint = torch.cat([cls, gru_cat], dim=1)
        return self.fc(self.drop(joint))


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, golds, idxs = [], [], []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        preds.extend(logits.argmax(-1).cpu().tolist())
        golds.extend(b["label"].tolist())
        idxs.extend(b["idx"].tolist())
    return np.array(preds), np.array(golds), np.array(idxs)


def evaluate(preds, golds, rows, idxs, outdir, tag):
    """Write overall, per-emotion, per-language and per-cell metrics."""
    macro = f1_score(golds, preds, average="macro", zero_division=0)
    weighted = f1_score(golds, preds, average="weighted", zero_division=0)
    acc = accuracy_score(golds, preds)

    print(f"\n[{tag}]  accuracy={acc:.4f}  macro-F1={macro:.4f}  "
          f"weighted-F1={weighted:.4f}")

    # per-emotion
    p, r, f, s = precision_recall_fscore_support(
        golds, preds, labels=list(range(len(EMOTIONS))), zero_division=0)
    per_emo = pd.DataFrame({
        "emotion": EMOTIONS, "precision": p, "recall": r,
        "f1": f, "support": s})
    print("\nper-emotion:")
    print(per_emo.to_string(index=False,
                            float_format=lambda x: f"{x:.3f}"))

    # per-language and per-cell
    lang_of = [rows[i]["language_label"] for i in idxs]
    df = pd.DataFrame({"gold": golds, "pred": preds, "lang": lang_of})

    lang_rows = []
    for lg in LANGS:
        sub = df[df["lang"] == lg]
        if len(sub) == 0:
            continue
        lang_rows.append({
            "language": lg, "n": len(sub),
            "macro_f1": f1_score(sub["gold"], sub["pred"],
                                 average="macro", zero_division=0),
            "accuracy": accuracy_score(sub["gold"], sub["pred"]),
        })
    per_lang = pd.DataFrame(lang_rows)
    print("\nper-language:")
    print(per_lang.to_string(index=False,
                             float_format=lambda x: f"{x:.3f}"))

    cell_rows = []
    for ei, emo in enumerate(EMOTIONS):
        for lg in LANGS:
            sub = df[df["lang"] == lg]
            n = int((sub["gold"] == ei).sum())
            if n == 0:
                cell_rows.append({"emotion": emo, "language": lg,
                                  "n": 0, "f1": float("nan")})
                continue
            yt = (sub["gold"] == ei).astype(int)
            yp = (sub["pred"] == ei).astype(int)
            cell_rows.append({
                "emotion": emo, "language": lg, "n": n,
                "f1": f1_score(yt, yp, zero_division=0)})
    per_cell = pd.DataFrame(cell_rows)
    print("\nper emotion x language (n shown — cells under ~30 are indicative only):")
    pivot_f1 = per_cell.pivot(index="emotion", columns="language", values="f1")
    pivot_n = per_cell.pivot(index="emotion", columns="language", values="n")
    for emo in EMOTIONS:
        cells = " ".join(
            f"{lg[:4]}={pivot_f1.loc[emo, lg]:.2f}(n={int(pivot_n.loc[emo, lg])})"
            for lg in LANGS if lg in pivot_f1.columns)
        print(f"  {emo:10s} {cells}")

    outdir.mkdir(parents=True, exist_ok=True)
    per_emo.to_csv(outdir / f"{tag}_per_emotion.csv", index=False)
    per_lang.to_csv(outdir / f"{tag}_per_language.csv", index=False)
    per_cell.to_csv(outdir / f"{tag}_per_cell.csv", index=False)
    pd.DataFrame(confusion_matrix(golds, preds,
                                  labels=list(range(len(EMOTIONS)))),
                 index=EMOTIONS, columns=EMOTIONS).to_csv(
        outdir / f"{tag}_confusion.csv")

    # raw predictions — needed later for bootstrap CIs
    with open(outdir / f"{tag}_predictions.jsonl", "w", encoding="utf-8") as f:
        for pi, gi, ix in zip(preds, golds, idxs):
            f.write(json.dumps({
                "id": rows[ix]["id"],
                "gold": EMOTIONS[gi],
                "pred": EMOTIONS[pi],
                "language_label": rows[ix]["language_label"],
            }, ensure_ascii=False) + "\n")

    with open(outdir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "macro_f1": macro,
                   "weighted_f1": weighted}, f, indent=2)

    return {"accuracy": acc, "macro_f1": macro, "weighted_f1": weighted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--data", default="data/splits")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no_class_weight", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model}  seed: {args.seed}")

    d = Path(args.data)
    train_rows = load_split(d / "train.jsonl")
    dev_rows = load_split(d / "dev.jsonl")
    test_rows = load_split(d / "test.jsonl")
    print(f"train={len(train_rows)}  dev={len(dev_rows)}  test={len(test_rows)}")

    name = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)

    train_ds = EmoDataset(train_rows, tok, args.max_len)
    dev_ds = EmoDataset(dev_rows, tok, args.max_len)
    test_ds = EmoDataset(test_rows, tok, args.max_len)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch * 2)
    test_dl = DataLoader(test_ds, batch_size=args.batch * 2)

    cls = HybridClassifier if args.model.endswith("_hybrid") else Classifier
    model = cls(name, len(EMOTIONS)).to(device)

    # class weights — the thin cells (Fear-Hindi ~105) need them
    if args.no_class_weight:
        weight = None
    else:
        counts = np.bincount([E2I[r["label"]] for r in train_rows],
                             minlength=len(EMOTIONS)).astype(float)
        w = counts.sum() / (len(EMOTIONS) * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float, device=device)
        print("class weights:", {e: round(float(x), 2)
                                 for e, x in zip(EMOTIONS, w)})

    lossf = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = (len(train_dl) // args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    outdir = Path(args.out) / f"{args.model}_seed{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    best_dev, best_state = -1.0, None

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        opt.zero_grad()
        for step, b in enumerate(train_dl, 1):
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(b["input_ids"].to(device),
                               b["attention_mask"].to(device))
                loss = lossf(logits, b["label"].to(device)) / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum
            if step % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(); sched.step()
            if step % 200 == 0:
                print(f"  ep{ep} step {step}/{len(train_dl)} "
                      f"loss={running/step:.4f}")

        dp, dg, di = predict(model, dev_dl, device)
        dev_macro = f1_score(dg, dp, average="macro", zero_division=0)
        print(f"epoch {ep}: train_loss={running/len(train_dl):.4f}  "
              f"dev_macro_F1={dev_macro:.4f}  ({time.time()-t0:.0f}s)")
        if dev_macro > best_dev:
            best_dev = dev_macro
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            print("  ^ best so far, checkpointed")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nbest dev macro-F1: {best_dev:.4f}")
    dp, dg, di = predict(model, dev_dl, device)
    evaluate(dp, dg, dev_rows, di, outdir, "dev")
    tp, tg, ti = predict(model, test_dl, device)
    res = evaluate(tp, tg, test_rows, ti, outdir, "test")

    # append to the master results table
    master = Path(args.out) / "all_results.csv"
    row = {"model": args.model, "seed": args.seed,
           "dev_macro_f1": best_dev, **{f"test_{k}": v for k, v in res.items()}}
    pd.DataFrame([row]).to_csv(master, mode="a", index=False,
                               header=not master.exists())
    print(f"\nwritten -> {outdir}\nappended -> {master}")


if __name__ == "__main__":
    main()
