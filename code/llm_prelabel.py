#!/usr/bin/env python3
"""
llm_prelabel.py
===============
Stage-1 LLM batch pre-labeling for the Hinglish / code-mixed emotion dataset.

Reads scraped comments, runs each through an LLM ONCE, and writes a
Label-Studio-importable JSON file of *predictions* (read-only suggestions the
human annotators confirm or correct in Label Studio). This is a SPEED aid,
not a labeling authority — humans always make the final call.

Output contract (must match label_config.xml v2):
    Text object name ......... "comment"   -> to_name for every result
    from_name controls predicted .. primary_emotion, other_emotions,
                               emotion_carrying_language, confidence, notes
    NOT predicted (human-only) ..... difficult   (annotators flag this; the LLM
                               must never pre-fill it or it biases the flag)
    (code_switch_is_emotional was removed from the schema this session.)
    emotions[] is reconstructed at export as [primary_emotion] + other_emotions.

    Verified against label_config.xml (v2) by executing to_ls_task +
    parse_prediction and diffing every from_name / to_name / type / choice
    value against the config — full match, no mismatches.

Each output task:
    {
      "data": { ...full schema fields so Label Studio can display + export... },
      "predictions": [{
        "model_version": "<model-id>",
        "score": <float, mean LLM confidence, used for active-learning sort>,
        "result": [ {from_name, to_name, type, value}, ... ]
      }]
    }

Privacy: only masked_text is ever sent to the API. raw_text (which can contain
names/handles) is kept locally in `data` for the annotators but never leaves
the machine. Do not change `TEXT_FIELD` to raw_text.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install anthropic
    python llm_prelabel.py \
        --input  processed_comments.jsonl \
        --output ls_import.json \
        --model  claude-haiku-4-5-20251001 \
        --limit  500            # omit to run the whole file

Resumable: re-running with the same --output appends only unseen ids
(progress is checkpointed in <output>.done.txt).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------
# Verified facts (checked against docs.claude.com, May 2026):
#   Current model strings: claude-haiku-4-5-20251001 (cheap/fast, good default
#   for a 50k offline pass), claude-sonnet-4-6, claude-opus-4-7.
#   For the full ~50k scrape, the Message Batches API gives a ~50% discount and
#   is the recommended scale path — see the note at the bottom of this file.
#   This script uses the synchronous Messages API: simpler + resumable, fine
#   for the pilot and small batches.
# ----------------------------------------------------------------------------

DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Field in the input record that holds the (masked) text to classify.
# DO NOT set to raw_text — see privacy note above.
TEXT_FIELD = "masked_text"

# Allowed label vocabularies (must mirror label_config.xml exactly).
EMOTIONS = {"Love", "Joy", "Anger", "Sadness", "Fear",
            "Surprise", "Nostalgia", "Devotion", "Neutral"}
CARRYING = {"en", "hi", "both", "neither"}
# code_switch_is_emotional REMOVED this session (field dropped from the schema).

# Schema fields copied verbatim into task["data"] so Label Studio can show
# context and the export round-trips your schema.
DATA_FIELDS = [
    "id", "raw_text", "masked_text", "video_id", "channel", "genre",
    "script", "language_label", "code_mix_index", "n_tokens",
]

SYSTEM_PROMPT = """You are a careful annotator for a code-mixed (Hinglish) \
English-Hindi emotion-detection dataset. You label short YouTube comments.

Use EXACTLY this label set (8 emotions + Neutral), no others:
  Love, Joy, Anger, Sadness, Fear, Surprise, Nostalgia, Devotion, Neutral.

Two labels are subtle and are the whole point of this dataset — read carefully:

- NOSTALGIA: bittersweet, MIXED-VALENCE longing for the past. It needs WARMTH.
  Pure negative grief or loss with no fondness is SADNESS, not Nostalgia.
  Cues: "those days", "miss the old times", "bachpan", "purane din", "golden era",
  fond memory + ache.

- DEVOTION: reverential, ASYMMETRIC, venerational attachment toward a sacred or
  admired figure (deity, guru, idol, hero). It needs REVERENCE/WORSHIP.
  Symmetric, peer-level affection is LOVE, not Devotion.
  Cues: worship, "bhagwan", "guru ji", "legend", "GOAT", salute, "you are my god",
  spiritual or fan veneration.

Rules:
- Choose 1 to 3 emotions actually present. Pick exactly one primary_emotion.
- If no emotion is clearly present, use Neutral alone (primary = Neutral).
- Do not invent emotion that is not in the text.

Also judge one CODE-MIX field:
- emotion_carrying_language: which language expresses the emotion?
    "en"=English, "hi"=Hindi (Devanagari or romanized), "both", "neither".

Return ONLY a JSON object, no markdown, no prose, exactly these keys:
{"emotions": [..1-3..], "primary_emotion": "..", "emotion_carrying_language": "..",
 "confidence": 1-5, "note": ".."}
confidence: your own certainty, 1 (low) to 5 (high). note: <=15 words rationale.
"""

# A few trilingual exemplars to anchor the hard distinctions. Kept short.
FEWSHOT = [
    ("Watching this in 2024, miss those college days yaar 🥹 best time of life",
     {"emotions": ["Nostalgia"], "primary_emotion": "Nostalgia",
      "emotion_carrying_language": "both",
      "confidence": 5, "note": "fond bittersweet longing for past"}),
    ("Bhai mera dog mar gaya last week, abhi tak ro raha hu",
     {"emotions": ["Sadness"], "primary_emotion": "Sadness",
      "emotion_carrying_language": "hi",
      "confidence": 5, "note": "pure grief, no fond warmth"}),
    ("Jai Shree Ram 🙏 you are not a man you are a divine soul, we worship you",
     {"emotions": ["Devotion"], "primary_emotion": "Devotion",
      "emotion_carrying_language": "both",
      "confidence": 5, "note": "reverential veneration of a figure"}),
    ("Happy birthday my love ❤️ you mean everything to me",
     {"emotions": ["Love", "Joy"], "primary_emotion": "Love",
      "emotion_carrying_language": "en",
      "confidence": 4, "note": "peer-level affection, not worship"}),
    ("video kab upload hoga? notification on kar diya",
     {"emotions": ["Neutral"], "primary_emotion": "Neutral",
      "emotion_carrying_language": "neither",
      "confidence": 4, "note": "informational, no emotion"}),
    # --- Fear (primary) ---
    ("ye dekh ke mere rongte khade ho gaye, raat ko akela nahi so paunga 😨",
     {"emotions": ["Fear"], "primary_emotion": "Fear",
      "emotion_carrying_language": "hi",
      "confidence": 5, "note": "physiological fear, goosebumps, dread"}),
    ("OMG that jumpscare actually made me scream, my heart is racing",
     {"emotions": ["Fear", "Surprise"], "primary_emotion": "Fear",
      "emotion_carrying_language": "en",
      "confidence": 4, "note": "startle fear with surprise component"}),
    # --- Fear as SECONDARY (recall: scary-but-funny / scary-but-amazing) ---
    ("itna dar laga par end me haste haste pet dukh gaya 😂",
     {"emotions": ["Joy", "Fear"], "primary_emotion": "Joy",
      "emotion_carrying_language": "hi",
      "confidence": 3, "note": "humor primary, fear present secondarily"}),
    # --- Surprise (primary) ---
    ("WHAT just happened?? yakeen hi nahi ho raha, last ball pe chakka 😱",
     {"emotions": ["Surprise"], "primary_emotion": "Surprise",
      "emotion_carrying_language": "both",
      "confidence": 5, "note": "genuine shock at unexpected event"}),
    ("plot twist ne to dimaag hila diya, socha bhi nahi tha aisa hoga",
     {"emotions": ["Surprise"], "primary_emotion": "Surprise",
      "emotion_carrying_language": "hi",
      "confidence": 4, "note": "shock at narrative reversal"}),
    # --- Surprise as SECONDARY (recall) ---
    ("didn't expect this comeback honestly, so happy for the team 🎉",
     {"emotions": ["Joy", "Surprise"], "primary_emotion": "Joy",
      "emotion_carrying_language": "en",
      "confidence": 4, "note": "joy primary, surprise present secondarily"}),
    # --- Anger (non-political) ---
    ("worst service ever, paise waste kar diye, bilkul bakwaas product",
     {"emotions": ["Anger"], "primary_emotion": "Anger",
      "emotion_carrying_language": "hi",
      "confidence": 5, "note": "consumer frustration / outrage"}),
    ("how dare they treat fans like this, absolutely unacceptable 😡",
     {"emotions": ["Anger"], "primary_emotion": "Anger",
      "emotion_carrying_language": "en",
      "confidence": 4, "note": "indignation"}),
    # --- Joy ---
    ("itni pyaari video, dil khush ho gaya, subah subah mood ban gaya 😄",
     {"emotions": ["Joy"], "primary_emotion": "Joy",
      "emotion_carrying_language": "hi",
      "confidence": 4, "note": "uplift, delight"}),
    # --- Nostalgia vs Sadness boundary (the hard one) ---
    ("90s ke din yaad aate hain, ab wo zamana nahi raha 😔",
     {"emotions": ["Nostalgia", "Sadness"], "primary_emotion": "Nostalgia",
      "emotion_carrying_language": "hi",
      "confidence": 4, "note": "bittersweet warmth dominates -> Nostalgia not Sadness"}),
    # --- Devotion vs Love boundary (the hard one) ---
    ("Sir aap mere idol ho, aapko dekh ke hi inspiration milti hai 🙏",
     {"emotions": ["Devotion"], "primary_emotion": "Devotion",
      "emotion_carrying_language": "hi",
      "confidence": 4, "note": "reverence for admired figure -> Devotion not Love"}),
    # --- Neutral (low-content) ---
    ("background music ka naam kya hai koi bata do",
     {"emotions": ["Neutral"], "primary_emotion": "Neutral",
      "emotion_carrying_language": "neither",
      "confidence": 5, "note": "pure information request"}),
]


# ----------------------------------------------------------------------------
# LLM call
# ----------------------------------------------------------------------------
def build_messages(text: str):
    """User-turn messages: few-shot pairs then the target comment.
    A cache_control breakpoint is placed on the LAST few-shot assistant turn,
    so the entire fixed prefix (all few-shot pairs) is cached and reused across
    every call. Only the final target comment varies, so ~88% of input tokens
    are served from cache at 90% lower cost after the first call."""
    msgs = []
    n = len(FEWSHOT)
    for i, (comment, ans) in enumerate(FEWSHOT):
        msgs.append({"role": "user", "content": f"Comment:\n{comment}"})
        assistant_turn = {"role": "assistant",
                          "content": json.dumps(ans, ensure_ascii=False)}
        # mark the end of the fixed few-shot block as the cache breakpoint
        if i == n - 1:
            assistant_turn["content"] = [{
                "type": "text",
                "text": json.dumps(ans, ensure_ascii=False),
                "cache_control": {"type": "ephemeral"},
            }]
        msgs.append(assistant_turn)
    msgs.append({"role": "user", "content": f"Comment:\n{text}"})
    return msgs


def call_anthropic(client, model: str, text: str, max_retries: int = 5) -> dict:
    """Single classification call with exponential backoff on transient errors.
    Uses prompt caching on the system prompt + few-shot prefix (see build_messages)."""
    import anthropic
    # System prompt as a cacheable block (fixed across all calls).
    system_blocks = [{
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }]
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                temperature=0,            # determinism (note: no seed param exists)
                system=system_blocks,
                messages=build_messages(text),
            )
            raw = "".join(b.text for b in resp.content if b.type == "text").strip()
            return parse_prediction(raw)
        except (anthropic.RateLimitError, anthropic.APIStatusError,
                anthropic.APIConnectionError) as e:
            wait = min(2 ** attempt, 30)
            print(f"  retry {attempt+1}/{max_retries} after {wait}s ({type(e).__name__})",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("max retries exceeded")


def parse_prediction(raw: str) -> dict:
    """Strip any stray fences, parse JSON, validate + coerce to the vocab."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
    obj = json.loads(raw)

    emotions = [e for e in obj.get("emotions", []) if e in EMOTIONS][:3]
    if not emotions:
        emotions = ["Neutral"]
    primary = obj.get("primary_emotion")
    if primary not in EMOTIONS:
        primary = emotions[0]
    if primary not in emotions:           # keep primary inside the multi-set
        emotions = [primary] + emotions[:2]

    carrying = obj.get("emotion_carrying_language")
    if carrying not in CARRYING:
        carrying = "neither"
    try:
        conf = int(obj.get("confidence", 3))
    except (TypeError, ValueError):
        conf = 3
    conf = max(1, min(5, conf))
    note = str(obj.get("note", ""))[:200]

    return {"emotions": emotions, "primary_emotion": primary,
            "emotion_carrying_language": carrying,
            "confidence": conf, "note": note}


# ----------------------------------------------------------------------------
# Label Studio task assembly
# ----------------------------------------------------------------------------
def to_ls_task(record: dict, pred: dict, model_version: str) -> dict:
    """Build one Label Studio task with a read-only prediction block."""
    data = {k: record.get(k) for k in DATA_FIELDS if k in record}
    # masked_text must be present — the config displays $masked_text.
    data.setdefault("masked_text", record.get(TEXT_FIELD, ""))

    def choices(from_name, values):
        return {"from_name": from_name, "to_name": "comment",
                "type": "choices", "value": {"choices": values}}

    # Config v2 splits the old single "emotions" control into:
    #   primary_emotion (single)  +  other_emotions (multi, excludes the primary
    #   and excludes Neutral). Reconstruct emotions[] = primary + other at export.
    primary = pred["primary_emotion"]
    other = [e for e in pred["emotions"] if e != primary and e != "Neutral"]

    result = [
        choices("primary_emotion", [primary]),
        choices("emotion_carrying_language", [pred["emotion_carrying_language"]]),
        {"from_name": "confidence", "to_name": "comment",
         "type": "rating", "value": {"rating": pred["confidence"]}},
    ]
    if other:                       # only emit when there's a secondary emotion
        result.append(choices("other_emotions", other))
    if pred["note"]:                # notes lives in a collapsed panel; optional
        result.append({"from_name": "notes", "to_name": "comment",
                       "type": "textarea", "value": {"text": [pred["note"]]}})
    return {
        "data": data,
        "predictions": [{
            "model_version": model_version,
            "score": pred["confidence"] / 5.0,   # 0-1, for active-learning sort
            "result": result,
        }],
    }


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def read_records(path: Path):
    """Yield dict records from .jsonl or .csv."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif path.suffix == ".csv":
        import csv
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yield row
    else:
        raise ValueError(f"unsupported input type: {path.suffix} (use .jsonl or .csv)")


def run_batch(client, args, records, done_ids, tasks_jsonl, done_path):
    """Message Batches API path: ~50% cheaper, asynchronous, CHUNKED.

    Submits pending records in chunks of CHUNK_SIZE (default 8000) so each
    upload is a small, reliable POST rather than one giant 60k+ payload that
    times out on home connections. Each chunk is submitted -> polled -> results
    retrieved before the next. Every result goes through parse_prediction +
    to_ls_task UNCHANGED, so output is identical to sync.

    Resumable at the comment level: succeeded ids are logged to done_path, so a
    re-run skips them and only re-chunks the remainder. If a chunk is mid-flight
    when interrupted, its batch id is saved and re-running resumes that chunk.
    """
    import anthropic

    CHUNK_SIZE = getattr(args, "chunk_size", 8000)
    batch_id_path = args.output.with_suffix(".batch_id.txt")

    # Build the full pending list (id -> record), skipping already-done ids.
    pending = []
    id_to_rec = {}
    for rec in records:
        rid = str(rec.get("id", ""))
        if not rid or rid in done_ids:
            continue
        text = (rec.get(TEXT_FIELD) or "").strip()
        if not text:
            continue
        id_to_rec[rid] = rec
        pending.append(rid)
        if args.limit and len(pending) >= args.limit:
            break

    if not pending and not batch_id_path.exists():
        print("nothing to submit (all done?).", file=sys.stderr)
        return 0, 0

    total = len(pending)
    n_done = n_err = 0
    print(f"{total} comments to label, in chunks of {CHUNK_SIZE}", file=sys.stderr)

    def process_batch_id(batch_id, id_subset):
        """Poll one batch to completion, retrieve, parse, write. Returns (done,err)."""
        nonlocal n_done, n_err
        while True:
            b = client.messages.batches.retrieve(batch_id)
            c = b.request_counts
            print(f"  [{batch_id[:18]}] status={b.processing_status} "
                  f"done={c.succeeded} err={c.errored} proc={c.processing}",
                  file=sys.stderr)
            if b.processing_status == "ended":
                break
            time.sleep(30)
        d = e = 0
        with tasks_jsonl.open("a", encoding="utf-8") as tf, \
             done_path.open("a", encoding="utf-8") as df:
            for result in client.messages.batches.results(batch_id):
                rid = result.custom_id
                try:
                    if result.result.type != "succeeded":
                        raise RuntimeError(f"item {result.result.type}")
                    msg = result.result.message
                    raw = "".join(x.text for x in msg.content if x.type == "text").strip()
                    pred = parse_prediction(raw)
                    rec = id_to_rec.get(rid) or {"id": rid, "masked_text": ""}
                    task = to_ls_task(rec, pred, args.model)
                    tf.write(json.dumps(task, ensure_ascii=False) + "\n"); tf.flush()
                    df.write(rid + "\n"); df.flush()
                    d += 1
                except Exception as ex:                      # noqa: BLE001
                    e += 1
                    print(f"  skip id={rid}: {ex}", file=sys.stderr)
        return d, e

    # If a chunk was mid-flight from a previous run, finish it first.
    if batch_id_path.exists():
        prev_id = batch_id_path.read_text(encoding="utf-8").strip()
        print(f"resuming in-flight batch {prev_id}", file=sys.stderr)
        d, e = process_batch_id(prev_id, None)
        n_done += d; n_err += e
        batch_id_path.unlink()
        # rebuild pending minus what just completed
        done_now = set(done_path.read_text(encoding="utf-8").split())
        pending = [r for r in pending if r not in done_now]

    # Submit remaining pending in chunks.
    for i in range(0, len(pending), CHUNK_SIZE):
        chunk_ids = pending[i:i + CHUNK_SIZE]
        requests = [{
            "custom_id": rid,
            "params": {
                "model": args.model,
                "max_tokens": 512,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "messages": build_messages((id_to_rec[rid].get(TEXT_FIELD) or "").strip()),
            },
        } for rid in chunk_ids]
        cnum = i // CHUNK_SIZE + 1
        ctot = (len(pending) + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"submitting chunk {cnum}/{ctot} ({len(requests)} requests) ...",
              file=sys.stderr)
        # retry the submit itself on transient connection errors
        for attempt in range(5):
            try:
                batch = client.messages.batches.create(requests=requests)
                break
            except (anthropic.APIConnectionError, anthropic.APIStatusError) as ex:
                wait = min(2 ** attempt, 30)
                print(f"  submit retry {attempt+1}/5 after {wait}s ({type(ex).__name__})",
                      file=sys.stderr)
                time.sleep(wait)
        else:
            print(f"  chunk {cnum} failed to submit after retries; "
                  f"re-run to continue from here.", file=sys.stderr)
            return n_done, n_err
        batch_id_path.write_text(batch.id, encoding="utf-8")
        print(f"  submitted {batch.id}", file=sys.stderr)
        d, e = process_batch_id(batch.id, chunk_ids)
        n_done += d; n_err += e
        batch_id_path.unlink()   # chunk fully consumed

    return n_done, n_err


def main():
    ap = argparse.ArgumentParser(description="LLM batch pre-labeling -> Label Studio import JSON")
    ap.add_argument("--input", required=True, type=Path, help="processed comments (.jsonl or .csv)")
    ap.add_argument("--output", required=True, type=Path, help="Label Studio import .json")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"model id (default {DEFAULT_MODEL})")
    ap.add_argument("--limit", type=int, default=None, help="stop after N comments (pilot)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls (rate control)")
    ap.add_argument("--batch", action="store_true",
                    help="use the Message Batches API (~50%% cheaper, async). "
                         "Submits in chunks, polls, retrieves. Resumable.")
    ap.add_argument("--chunk-size", type=int, default=8000, dest="chunk_size",
                    help="requests per batch chunk (default 8000)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: set ANTHROPIC_API_KEY in your environment.")

    try:
        import anthropic
    except ImportError:
        sys.exit("ERROR: pip install anthropic")

    client = anthropic.Anthropic()

    # Resume support: track already-processed ids.
    done_path = args.output.with_suffix(args.output.suffix + ".done.txt")
    done_ids = set()
    if done_path.exists():
        done_ids = set(done_path.read_text(encoding="utf-8").split())
        print(f"resuming: {len(done_ids)} ids already processed", file=sys.stderr)

    # Append newly built tasks to a sidecar .jsonl first (crash-safe),
    # then assemble the final JSON array at the end.
    tasks_jsonl = args.output.with_suffix(".tasks.jsonl")

    if args.batch:
        n_done, n_err = run_batch(client, args, read_records(args.input),
                                  done_ids, tasks_jsonl, done_path)
    else:
        n_done = n_err = 0
        with tasks_jsonl.open("a", encoding="utf-8") as tf, \
             done_path.open("a", encoding="utf-8") as df:
            for rec in read_records(args.input):
                rid = str(rec.get("id", ""))
                if not rid or rid in done_ids:
                    continue
                text = (rec.get(TEXT_FIELD) or "").strip()
                if not text:
                    continue
                try:
                    pred = call_anthropic(client, args.model, text)
                    task = to_ls_task(rec, pred, args.model)
                    tf.write(json.dumps(task, ensure_ascii=False) + "\n"); tf.flush()
                    df.write(rid + "\n"); df.flush()
                    done_ids.add(rid)
                    n_done += 1
                except Exception as e:                      # noqa: BLE001 — log and continue
                    n_err += 1
                    print(f"  skip id={rid}: {e}", file=sys.stderr)
                if n_done % 50 == 0 and n_done:
                    print(f"... {n_done} labeled", file=sys.stderr)
                if args.limit and n_done >= args.limit:
                    break
                if args.sleep:
                    time.sleep(args.sleep)

    # Assemble final import array from the crash-safe sidecar.
    tasks = [json.loads(l) for l in tasks_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDONE  labeled={n_done}  errors={n_err}  total_tasks={len(tasks)}")
    print(f"Import into Label Studio: Project > Import > {args.output}")
    print("Then enable Settings > Machine Learning > 'Show predictions to annotators' "
          "(see the anchoring caveat before you do).")


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
# SCALE NOTE — Message Batches API (recommended for the full ~50k pass)
# ----------------------------------------------------------------------------
# The synchronous loop above is ideal for the pilot and for resumable runs.
# For the full scrape, the Message Batches API processes large jobs
# asynchronously at roughly half the per-token cost. The per-request body
# (system + messages + the JSON output contract) is identical to call_anthropic
# above; you submit them as a batch and poll for results, then feed each result
# string through parse_prediction() and to_ls_task() unchanged.
# Verify the current batch endpoint + limits before relying on it:
#   https://docs.claude.com/en/docs/build-with-claude/batch-processing
# ----------------------------------------------------------------------------
