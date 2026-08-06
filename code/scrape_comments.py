#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scrape_comments.py  —  YouTube comment scraper for the Hinglish Emotion Dataset
===============================================================================

Gathers raw public YouTube comments (no API key) into a JSONL that
process_comments.py consumes. Channel/playlist/search seeds are expanded with
yt-dlp; comments are pulled with youtube-comment-downloader.

Design goals (match the rest of the pipeline):
  * RESUMABLE & CRASH-SAFE — appends to JSONL, tracks done-videos and seen
    comment-ids in sidecars under SETTINGS["state_dir"]; re-running continues
    instead of re-scraping or duplicating.
  * PRIVACY BY DEFAULT — author display names, commenter channel IDs, and
    avatar URLs are DROPPED at write time. We persist only the comment text
    plus non-PII provenance (video_id, video channel, genre) and an opaque
    comment id for dedup. (Masking of in-text PII happens later in
    process_comments.py.)
  * POLITE — sleeps between requests, caps comments/video and videos/channel,
    isolates per-video failures so one bad video can't kill a long run.

ETHICS — read before running:
  Scraping public comments for research is common but you are responsible for
  (1) your institution's IRB / ethics approval, (2) YouTube's Terms of Service,
  (3) data-protection law (e.g. India's DPDP Act, GDPR if any EU users). This
  script minimises PII at the source, but YOU must confirm your collection is
  approved. Do not redistribute raw author identities. The public release
  should ship masked_text only (process_comments.py --religion mask).

Output record (one JSON object per line):
  {"cid","raw_text","video_id","channel","genre","votes","replies","scraped_at"}
  -> process_comments.py reads raw_text/video_id/channel/genre.

This environment cannot reach youtube.com, so run this on YOUR machine:
    pip install -r requirements.txt
    python scrape_comments.py            # uses config.py
    python scrape_comments.py --dry-run  # expand seeds, print plan, scrape nothing
    python scrape_comments.py --genre nostalgia_retro   # one genre only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import config as CFG


# --------------------------------------------------------------------------- #
# STATE (resume sidecars)                                                      #
# --------------------------------------------------------------------------- #
class ScrapeState:
    def __init__(self, state_dir: Path, out_path: Path):
        self.state_dir = state_dir
        self.out_path = out_path
        state_dir.mkdir(parents=True, exist_ok=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.done_videos_file = state_dir / "done_videos.txt"
        self.seen_cids_file = state_dir / "seen_cids.txt"
        self.counts_file = state_dir / "genre_counts.json"

        self.done_videos: set[str] = self._load_set(self.done_videos_file)
        self.seen_cids: set[str] = self._load_set(self.seen_cids_file)
        self.genre_counts: dict[str, int] = self._load_counts()
        # Rebuild seen_cids from the actual output too, in case a crash happened
        # mid-write before the sidecar flushed (output is source of truth).
        self._reconcile_from_output()

        self._out_fh = out_path.open("a", encoding="utf-8")
        self._dv_fh = self.done_videos_file.open("a", encoding="utf-8")
        self._cid_fh = self.seen_cids_file.open("a", encoding="utf-8")

    @staticmethod
    def _load_set(p: Path) -> set[str]:
        if p.exists():
            return {ln.strip() for ln in p.open(encoding="utf-8") if ln.strip()}
        return set()

    def _load_counts(self) -> dict:
        if self.counts_file.exists():
            return json.loads(self.counts_file.read_text(encoding="utf-8"))
        return {}

    def _reconcile_from_output(self):
        if not self.out_path.exists():
            return
        for ln in self.out_path.open(encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                self.seen_cids.add(json.loads(ln)["cid"])
            except Exception:
                continue

    def total(self) -> int:
        return len(self.seen_cids)

    def write(self, record: dict, genre: str):
        self._out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._out_fh.flush()
        self.seen_cids.add(record["cid"])
        self._cid_fh.write(record["cid"] + "\n")
        self._cid_fh.flush()
        self.genre_counts[genre] = self.genre_counts.get(genre, 0) + 1

    def mark_video_done(self, vid: str):
        if vid not in self.done_videos:
            self.done_videos.add(vid)
            self._dv_fh.write(vid + "\n")
            self._dv_fh.flush()

    def flush_counts(self):
        self.counts_file.write_text(
            json.dumps(self.genre_counts, ensure_ascii=False, indent=2),
            encoding="utf-8")

    def close(self):
        self.flush_counts()
        for fh in (self._out_fh, self._dv_fh, self._cid_fh):
            try:
                fh.close()
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# SEED EXPANSION (yt-dlp)  — returns list of (video_id, video_channel)         #
# --------------------------------------------------------------------------- #
def expand_seed(seed: dict, max_videos: int) -> list[tuple[str, str]]:
    import yt_dlp

    if seed["type"] == "video":
        vid = _video_id_from_url(seed["url"])
        return [(vid, "")] if vid else []

    if seed["type"] == "search":
        target = f"ytsearch{seed.get('n', 20)}:{seed['query']}"
    else:  # channel | playlist
        target = seed["url"]

    ydl_opts = {
        "extract_flat": True,   # list entries without downloading video data
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "playlistend": max_videos,
    }
    out: list[tuple[str, str]] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(target, download=False)
        entries = info.get("entries") or []
        for e in entries[:max_videos]:
            if not e:
                continue
            vid = e.get("id")
            ch = e.get("channel") or e.get("uploader") or info.get("channel") or ""
            if vid:
                out.append((vid, ch))
    return out


def _video_id_from_url(url: str) -> str | None:
    import re
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else (url if len(url) == 11 else None)


# --------------------------------------------------------------------------- #
# COMMENT PULL (youtube-comment-downloader)                                    #
# --------------------------------------------------------------------------- #
def make_downloader():
    from youtube_comment_downloader import (
        YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT)
    sort = SORT_BY_POPULAR if CFG.SETTINGS["sort_by"] == "popular" else SORT_BY_RECENT
    return YoutubeCommentDownloader(), sort


def pull_comments(downloader, sort, video_id: str):
    """Yields the library's raw comment dicts for one video."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    yield from downloader.get_comments_from_url(
        url, sort_by=sort, language=CFG.SETTINGS.get("language"),
        sleep=CFG.SETTINGS["sleep"])


def to_record(raw: dict, video_id: str, channel: str, genre: str) -> dict:
    """Strip PII; keep text + non-PII provenance only."""
    return {
        "cid": raw.get("cid"),
        "raw_text": (raw.get("text") or "").strip(),
        "video_id": video_id,
        "channel": channel,          # the VIDEO's channel (not the commenter's)
        "genre": genre,
        "votes": raw.get("votes", "0"),
        "replies": raw.get("replies", "0"),
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # DROPPED on purpose: author displayName, commenter channelId, avatar URL
    }


# --------------------------------------------------------------------------- #
# DRIVER                                                                       #
# --------------------------------------------------------------------------- #
def scrape(only_genre: str | None = None, dry_run: bool = False):
    s = CFG.SETTINGS
    out_path = Path(s["out_path"])
    state = None if dry_run else ScrapeState(Path(s["state_dir"]), out_path)
    global_target = s["global_raw_target"]
    min_chars = s["min_chars"]
    max_per_video = s["max_comments_per_video"]
    max_videos = s["max_videos_per_channel"]

    genres = CFG.GENRES if not only_genre else {only_genre: CFG.GENRES[only_genre]}

    if not dry_run:
        downloader, sort = make_downloader()
        print(f"[resume] starting from {state.total()} comments already collected")

    for genre, gcfg in genres.items():
        gtarget = gcfg["per_genre_target"]
        have = (state.genre_counts.get(genre, 0) if state else 0)
        print(f"\n=== genre: {genre}  (target {gtarget}, have {have}, "
              f"targeted={gcfg['targeted']}) ===")

        # 1) expand all seeds for this genre
        videos: list[tuple[str, str]] = []
        for seed in gcfg["seeds"]:
            try:
                vids = (expand_seed(seed, max_videos) if not dry_run
                        else [(f"DRY{len(videos)+i:03d}", "dry-channel") for i in range(3)])
            except Exception as ex:
                print(f"  [seed-error] {seed}: {ex}")
                continue
            print(f"  seed {seed.get('query') or seed.get('url')}: +{len(vids)} videos")
            videos.extend(vids)

        if dry_run:
            print(f"  [dry-run] would scrape ~{len(videos)} videos for '{genre}'")
            continue

        # 2) pull comments video by video, honouring caps + targets
        for vid, ch in videos:
            if state.genre_counts.get(genre, 0) >= gtarget:
                print(f"  [genre target reached] {genre}")
                break
            if state.total() >= global_target:
                print("  [global target reached] stopping")
                state.close()
                return
            if vid in state.done_videos:
                continue

            kept = 0
            try:
                for i, raw in enumerate(pull_comments(downloader, sort, vid)):
                    if i >= max_per_video:
                        break
                    cid = raw.get("cid")
                    text = (raw.get("text") or "").strip()
                    if not cid or cid in state.seen_cids:
                        continue
                    if len(text) < min_chars:
                        continue
                    state.write(to_record(raw, vid, ch, genre), genre)
                    kept += 1
                    if state.genre_counts.get(genre, 0) >= gtarget:
                        break
                state.mark_video_done(vid)
                print(f"  {vid} (+{kept})  genre_total={state.genre_counts.get(genre,0)}"
                      f"  grand_total={state.total()}")
            except KeyboardInterrupt:
                print("\n[interrupted] state saved; re-run to resume.")
                state.close()
                return
            except Exception:
                print(f"  [video-error] {vid} skipped:")
                traceback.print_exc(limit=1)
                state.mark_video_done(vid)  # don't retry a broken video forever
                continue

        state.flush_counts()

    if not dry_run:
        print(f"\n[done] total collected: {state.total()}")
        print(f"[counts] {json.dumps(state.genre_counts, ensure_ascii=False)}")
        state.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="Scrape YouTube comments per config.py")
    p.add_argument("--genre", help="scrape only this genre key from config.GENRES")
    p.add_argument("--dry-run", action="store_true",
                   help="expand seeds & print the plan without scraping")
    args = p.parse_args(argv)
    if args.genre and args.genre not in CFG.GENRES:
        p.error(f"unknown genre '{args.genre}'. choices: {list(CFG.GENRES)}")
    scrape(only_genre=args.genre, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
