#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_comments.py  —  Hinglish Emotion Dataset preprocessing pipeline
=======================================================================

Rebuilt this session. The headline change is the LANGUAGE LABEL: it now emits
the locked 3-way scheme  English / Hindi / Hinglish  (was the old 4-way
hi/en/hi-dominant/en-dominant), decided by *language composition* and
*independent of script*.

Pipeline (raw scraped comments -> processed records in the locked schema):
    1. length filter (3-30 tokens)
    2. word-level language tagging (Hindi / English / Other) -- the shared
       signal used by BOTH the language_label and the code_mix_index, so they
       can never disagree
    3. script detection (devanagari / devanagari_mixed / latin)
    4. language_label   <-- 3-way, from word-level tags (NOT from script)
    5. code_mix_index   (0-50) from the same tags
    6. masking ([URL] [USER] [NAME] -- religion handling: see RELIGION_MODE)
    7. dedup
    8. emit JSONL/CSV in the locked schema (annotations[] left empty)

IMPORTANT — accuracy honesty (viva-relevant):
The word-level language tagger here is a transparent lexicon + context tagger,
not a trained model. It is fast, reproducible (no model artifact, no GPU) and
inspectable, but it is APPROXIMATE for romanized Hindi. Before trusting
language_label at 50k scale, hand-tag a ~300-comment slice from Pilot 1 and
measure accuracy (see validate_language_labels() at the bottom). If it is weak
on romanized Hindi, swap _tag_tokens() for a subword-LSTM LID model (e.g. the
SAIL-ICON-trained taggers, ~94% word-level) WITHOUT touching the rest of the
pipeline — that is why tagging is isolated in one function.

Deps: pip install wordfreq regex   (see requirements.txt)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import regex  # better Unicode \p{...} support than stdlib re
from wordfreq import zipf_frequency, top_n_list

# --------------------------------------------------------------------------- #
# CONFIG                                                                       #
# --------------------------------------------------------------------------- #
MIN_TOKENS = 3
MAX_TOKENS = 30

# English-frequency floor. A Latin token counts as plausibly-English only if it
# is at least this common in English. Tuned against the homograph probe: real
# Hindi romanizations (hai 3.57, dil 3.03, kya 2.71) sit BELOW common-English
# mass but ABOVE junk -- so the floor alone is not enough; we ALSO consult the
# Hindi lexicon first and a homograph set (below).
EN_ZIPF_FLOOR = 3.0

# Religion handling. Devotion depends on cues like "jai shree ram"/"guru ji",
# so blanket [RELIGION] masking destroys signal.
#   "keep" -> do not mask religion (annotation view / internal)
#   "mask" -> replace with [RELIGION] (public-release build)
# Operationalises OPEN DECISION #1: run the pipeline twice (--religion keep|mask),
# keep both artifacts. Default "keep" so annotators see the cue.
RELIGION_MODE = "keep"

CMI_MAX = 50  # code_mix_index reported on 0-50 (locked)

# If this fraction (or more) of a comment's tokens are UNRECOGNISED (neither
# Hindi nor English nor a known loanword), the comment is almost certainly in
# some other language (Uzbek, Turkish, Indonesian, ... — common on retro-music
# videos with global audiences). Such comments are labeled "Foreign" so they
# can be filtered out instead of silently polluting the English bucket.
# 0.6 chosen from the data: genuine EN/HI/Hinglish comments sit well below it
# (0-20% other); foreign comments sit at 80-100%. Re-tune from the validation
# slice if needed. Set DROP_FOREIGN=True to discard them during processing.
FOREIGN_OTHER_RATIO = 0.6
DROP_FOREIGN = True


# --------------------------------------------------------------------------- #
# LEXICONS                                                                     #
# --------------------------------------------------------------------------- #
_EN_COMMON = set(top_n_list("en", 50000))

# Romanized-Hindi function/marker words, checked with PRIORITY over English so
# that hai/nahi/kya are Hindi even though wordfreq lists them. Lowercase,
# ASCII-folded. Extend freely from pilot errors.
# NOTE: do NOT put "the" here -- it collides with the English article (very
# common) and the Hindi "थे" romanization is too rare to justify the damage.
ROMAN_HI = {
    "hai", "hain", "tha", "thi", "raha", "rahi", "rahe", "hoga", "hogi",
    # short grammatical particles (these leak into English as KO/KE/RO etc.)
    "ko", "ke", "ki", "se", "ne", "mein", "pe", "tak", "wala",
    # bhakti / devotion cues (high signal for the Devotion label)
    "jai", "shree", "shri", "sri", "bhakt", "bhakto", "bhakti", "prabhu",
    "hanuman", "krishna", "shiv", "shiva", "mata", "devi", "prabhuji", "ji",
    "dekh", "ro", "din", "kya", "aate", "aata", "aati", "raha", "rahe",
    "khaana", "khana", "peena", "sona", "uthna", "chalna", "rukna",
    "hota", "hoti", "hote", "hua", "hui", "huye", "ho", "hu", "hoon", "hun",
    "mera", "meri", "mere", "tera", "teri", "tere", "uska", "uski", "unka",
    "humara", "hamara", "hamari", "tumhara", "iska", "yeh", "ye", "woh", "wo",
    "kaun", "kahan", "kab", "kaise", "kyun", "kyu", "kyon", "kitna", "kitni",
    "nahi", "nahin", "naa", "haan", "han", "bhi", "toh", "bahut", "bohot",
    "bohut", "thoda", "zyada", "kuch", "sab", "sirf", "phir", "fir", "abhi",
    "aaj", "kal", "yaar", "yar", "bhai", "behen", "didi", "dil", "pyaar",
    "pyar", "ishq", "mohabbat", "gaana", "gana", "geet", "awaaz", "aawaz",
    "zindagi", "zindgi", "duniya", "saath", "sath", "dost", "dosti", "khush",
    "khushi", "gam", "gham", "dard", "dukh", "rona", "hasna", "yaad", "yaadein",
    "purane", "purana", "purani", "bachpan", "guru", "bhagwan", "ishwar",
    "accha", "acha", "achha", "bura", "sundar", "pyara", "pyari", "mast",
    "kamaal", "lajawab", "behtreen", "shaandar", "shukriya", "dhanyavad",
    "matlab", "samajh", "pata", "lagta", "lagti", "chahiye", "karna", "karo",
    "karta", "karti", "kar", "diya", "diye", "liya", "gaya", "gayi", "gaye",
    "aana", "aaya", "aayi", "jaana", "jaa", "jao", "milna", "mila", "dekha",
    "dekho", "suna", "suno", "bol", "bolo", "kaha", "kehna", "wala", "wali",
    "wale", "jaisa", "jaise", "itna", "utna", "ekdum", "bilkul", "waqt",
    "samay", "log", "logon", "baat", "baatein", "cheez", "kaam", "ghar",
    "desh", "watan", "maa", "papa", "pita", "beta", "beti", "chu", "feel",
}

# Romanized Hindi tokens that are ALSO common English words (homographs).
# Cannot be decided by lexicon membership; resolved by sentence context.
ROMAN_HI_AMBIG = {
    "main", "to", "me", "do", "us", "is", "are", "ram", "tu", "na",
    "or", "ab", "tum", "hum",
}

# Integrated loanwords: English-origin but fully naturalised in Hindi speech
# (platform/tech vocabulary). Treated as language-NEUTRAL so they neither flip
# a Hindi comment to Hinglish nor an English comment to Hinglish. Kept small
# and auditable; do NOT add content words like "song"/"beautiful" here -- those
# are genuine English and SHOULD count toward code-mixing.
LOANWORDS_NEUTRAL = {
    "video", "channel", "subscribe", "comment", "like", "share", "mobile",
    "internet", "online", "screen", "song",  # song is borderline; common as loanword
    "video", "mobile", "phone", "app", "live", "upload", "view", "views",
}


# --------------------------------------------------------------------------- #
# SCHEMA                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Comment:
    id: str
    raw_text: str
    masked_text: str
    video_id: str
    channel: str
    genre: str
    script: str            # devanagari | devanagari_mixed | latin
    language_label: str    # English | Hindi | Hinglish   <-- 3-way (locked)
    code_mix_index: int    # 0..50
    n_tokens: int
    annotations: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
# TOKENIZATION & SCRIPT                                                        #
# --------------------------------------------------------------------------- #
_WORD_RE = regex.compile(r"[\p{L}\p{M}]+", regex.UNICODE)
_DEVA_RE = regex.compile(r"\p{Script=Devanagari}", regex.UNICODE)
_LATIN_RE = regex.compile(r"\p{Script=Latin}", regex.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _fold(tok: str) -> str:
    """lowercase + strip combining marks for ASCII-ish comparison."""
    tok = tok.lower()
    nfkd = unicodedata.normalize("NFKD", tok)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def detect_script(text: str) -> str:
    has_deva = bool(_DEVA_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_deva and has_latin:
        return "devanagari_mixed"
    if has_deva:
        return "devanagari"
    return "latin"


# --------------------------------------------------------------------------- #
# WORD-LEVEL LANGUAGE TAGGING  (the shared signal)                             #
# --------------------------------------------------------------------------- #
# tags: "hi" | "en" | "other"
def _tag_one(tok_fold: str, is_deva: bool) -> str:
    if is_deva:
        return "hi"
    if not tok_fold:
        return "other"
    if tok_fold in LOANWORDS_NEUTRAL:
        return "other"  # naturalised loanword -> neutral, not code-mixing
    if tok_fold in ROMAN_HI:
        return "hi"
    if tok_fold in _EN_COMMON and zipf_frequency(tok_fold, "en") >= EN_ZIPF_FLOOR:
        return "en"
    return "other"


def _tag_tokens(tokens: list[str]) -> list[str]:
    """Word-level language tags. Homographs (tokens that are valid in both
    languages) are resolved in two stages: first by an unambiguous Hindi
    neighbour, then by the sentence's dominant non-homograph language."""
    folded = [_fold(t) for t in tokens]
    is_deva = [bool(_DEVA_RE.search(t)) for t in tokens]

    # First pass: ambiguous tokens are held as "ambig", everything else tagged.
    tags = []
    for f, d in zip(folded, is_deva):
        if (not d) and f in ROMAN_HI_AMBIG:
            tags.append("ambig")
        else:
            tags.append(_tag_one(f, d))

    # Sentence-level lean from the UNAMBIGUOUS tokens only.
    strong_hi = tags.count("hi")
    strong_en = tags.count("en")
    sentence_leans_hi = strong_hi > strong_en

    # Resolve each ambiguous token.
    for i in range(len(tags)):
        if tags[i] != "ambig":
            continue
        left = tags[i - 1] if i > 0 else None
        right = tags[i + 1] if i + 1 < len(tags) else None
        if "hi" in (left, right):
            tags[i] = "hi"            # an unambiguous Hindi neighbour wins
        elif "en" in (left, right) and not sentence_leans_hi:
            tags[i] = "en"
        else:
            tags[i] = "hi" if sentence_leans_hi else "en"
    return tags


# --------------------------------------------------------------------------- #
# LANGUAGE LABEL (3-way) + CODE-MIX INDEX  — from the same tags                #
# --------------------------------------------------------------------------- #
def language_label_and_cmi(tokens: list[str]) -> tuple[str, int]:
    """
    language_label rule (LOCKED, script-independent, by composition):
        monolingual English             -> "English"
        monolingual Hindi (any script)  -> "Hindi"
        contains both Hindi & English    -> "Hinglish"

    'monolingual' is judged on hi/en content tokens only; 'other' tokens
    (names, numbers, unknown) are ignored for the mono/mixed decision so a
    single unknown token does not flip a clean comment to Hinglish.
    """
    tags = _tag_tokens(tokens)
    n_hi = tags.count("hi")
    n_en = tags.count("en")
    n_other = tags.count("other")
    n_lang = n_hi + n_en
    n_all = len(tags)

    # Foreign-language guard: mostly-unrecognised tokens -> not EN/HI/Hinglish.
    if n_all and (n_other / n_all) >= FOREIGN_OTHER_RATIO and n_lang <= 1:
        return ("Foreign", 0)

    if n_lang == 0:
        return ("English", 0)  # no recognisable Hindi/English content

    if n_hi > 0 and n_en > 0:
        label = "Hinglish"
    elif n_hi > 0:
        label = "Hindi"
    else:
        label = "English"

    # CMI scaled to 0-50: 0 = monolingual, 25 at a perfect 50/50 mix.
    cmi = round(CMI_MAX * (1.0 - max(n_hi, n_en) / n_lang))
    return (label, int(cmi))


# --------------------------------------------------------------------------- #
# MASKING                                                                      #
# --------------------------------------------------------------------------- #
_URL_RE = regex.compile(r"https?://\S+|www\.\S+", regex.IGNORECASE)
_HANDLE_RE = regex.compile(r"@\w+")
_RELIGION_TERMS = [
    "jai shree ram", "jai shri ram", "jai sri ram", "har har mahadev",
    "allah", "bismillah", "alhamdulillah", "om namah shivaya", "waheguru",
    "jai mata di", "radhe radhe", "hare krishna",
]
_RELIGION_RE = regex.compile(
    r"|".join(regex.escape(t) for t in _RELIGION_TERMS), regex.IGNORECASE
)
_NAME_RE = regex.compile(r"\b([A-Z][a-z]{2,})\b")


def mask(text: str, religion_mode: str = RELIGION_MODE) -> str:
    text = _URL_RE.sub("[URL]", text)
    text = _HANDLE_RE.sub("[USER]", text)
    if religion_mode == "mask":
        text = _RELIGION_RE.sub("[RELIGION]", text)

    def _name_sub(m):
        w = m.group(1)
        wf = _fold(w)
        # never mask a common English word, a known romanized-Hindi token, or a
        # devotion/religion cue -- otherwise we destroy the very signal Devotion
        # depends on (e.g. "Shree", "Prabhu", "Krishna") while in keep mode.
        if wf in _EN_COMMON or wf in ROMAN_HI or wf in ROMAN_HI_AMBIG:
            return w
        return "[NAME]"

    text = _NAME_RE.sub(_name_sub, text)
    return text


# --------------------------------------------------------------------------- #
# MAIN PROCESSING                                                              #
# --------------------------------------------------------------------------- #
def make_id(raw_text: str, video_id: str) -> str:
    h = hashlib.sha1(f"{video_id}::{raw_text}".encode("utf-8")).hexdigest()
    return f"c_{h[:12]}"


def process_record(rec: dict, religion_mode: str = RELIGION_MODE) -> Comment | None:
    raw = (rec.get("raw_text") or rec.get("text") or "").strip()
    if not raw:
        return None
    tokens = tokenize(raw)
    n = len(tokens)
    if n < MIN_TOKENS or n > MAX_TOKENS:
        return None

    script = detect_script(raw)
    label, cmi = language_label_and_cmi(tokens)
    if DROP_FOREIGN and label == "Foreign":
        return None
    masked = mask(raw, religion_mode)

    return Comment(
        id=rec.get("id") or make_id(raw, rec.get("video_id", "")),
        raw_text=raw,
        masked_text=masked,
        video_id=rec.get("video_id", ""),
        channel=rec.get("channel", ""),
        genre=rec.get("genre", ""),
        script=script,
        language_label=label,
        code_mix_index=cmi,
        n_tokens=n,
        annotations=[],
    )


def process_stream(records: Iterable[dict], religion_mode=RELIGION_MODE) -> list[Comment]:
    seen: set[str] = set()
    out: list[Comment] = []
    for rec in records:
        c = process_record(rec, religion_mode)
        if c is None:
            continue
        key = c.masked_text.lower()
        if key in seen:
            continue  # dedup on masked text
        seen.add(key)
        out.append(c)
    return out


def _read_input(path: Path) -> Iterable[dict]:
    if path.suffix == ".jsonl":
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)
    elif path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from (data if isinstance(data, list) else [data])
    elif path.suffix == ".csv":
        import csv
        with path.open(encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)
    else:
        raise ValueError(f"unsupported input: {path.suffix}")


def _write_output(comments: list[Comment], out_path: Path) -> None:
    if not comments:
        print("warning: no comments survived processing", file=sys.stderr)
        return
    if out_path.suffix == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for c in comments:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    elif out_path.suffix == ".csv":
        import csv
        with out_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(comments[0]).keys()))
            w.writeheader()
            for c in comments:
                row = asdict(c)
                row["annotations"] = json.dumps(row["annotations"], ensure_ascii=False)
                w.writerow(row)
    else:
        raise ValueError(f"unsupported output: {out_path.suffix}")


# --------------------------------------------------------------------------- #
# VALIDATION  (run after Pilot 1 hand-tagging)                                 #
# --------------------------------------------------------------------------- #
def validate_language_labels(gold_jsonl: Path) -> None:
    """gold_jsonl: records with raw_text + hand 'gold_language_label'. Prints
    accuracy + confusion matrix so you can decide if the lexicon tagger is good
    enough or needs a trained LID model."""
    from collections import Counter
    conf = Counter()
    correct = total = 0
    for rec in _read_input(gold_jsonl):
        gold = rec.get("gold_language_label")
        if not gold:
            continue
        label, _ = language_label_and_cmi(tokenize(rec["raw_text"]))
        conf[(gold, label)] += 1
        total += 1
        correct += int(gold == label)
    print(f"language_label accuracy: {correct}/{total} = {correct/total:.3f}"
          if total else "no gold rows")
    for (g, p), n in sorted(conf.items()):
        flag = "" if g == p else "  <-- error"
        print(f"  gold={g:8s} pred={p:8s} : {n}{flag}")


# --------------------------------------------------------------------------- #
# CLI / SELF-TEST                                                              #
# --------------------------------------------------------------------------- #
_SELFTEST = [
    ("This song is absolutely beautiful and amazing", "English", "latin"),
    ("bhai ye gaana dil ko chu gaya yaar", "Hindi", "latin"),
    ("ye song bahut beautiful hai yaar", "Hinglish", "latin"),
    ("यह गाना बहुत सुंदर है", "Hindi", "devanagari"),
    ("ye gaana bahut nostalgic feel deta hai", "Hinglish", "latin"),
    ("main is video ko dekh ke ro diya", "Hindi", "latin"),
    ("jai shree ram bhakto", "Hindi", "latin"),
    ("OMG this is the best video ever", "English", "latin"),
]


def _run_selftest() -> int:
    print("SELF-TEST  (language_label / script / cmi)")
    fails = 0
    for text, exp_lang, exp_script in _SELFTEST:
        toks = tokenize(text)
        lab, cmi = language_label_and_cmi(toks)
        scr = detect_script(text)
        ok = (lab == exp_lang and scr == exp_script)
        fails += not ok
        mark = "ok " if ok else "FAIL"
        print(f"  [{mark}] lang={lab:8s}(exp {exp_lang:8s}) "
              f"script={scr:16s} cmi={cmi:2d} | {text}")
    print(f"\n{'ALL PASS' if not fails else str(fails)+' FAILED'}")
    return fails


def main(argv=None):
    p = argparse.ArgumentParser(description="Process scraped comments -> locked schema")
    p.add_argument("input", nargs="?", help="input .jsonl/.json/.csv of raw comments")
    p.add_argument("-o", "--output", help="output .jsonl/.csv")
    p.add_argument("--religion", choices=["keep", "mask"], default=RELIGION_MODE)
    p.add_argument("--selftest", action="store_true", help="run built-in tests and exit")
    p.add_argument("--validate", help="gold jsonl with gold_language_label")
    args = p.parse_args(argv)

    if args.selftest:
        return _run_selftest()
    if args.validate:
        validate_language_labels(Path(args.validate))
        return 0
    if not args.input or not args.output:
        p.error("input and -o/--output are required (or use --selftest)")

    comments = process_stream(_read_input(Path(args.input)), religion_mode=args.religion)
    _write_output(comments, Path(args.output))
    print(f"wrote {len(comments)} comments -> {args.output}  (religion={args.religion})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
