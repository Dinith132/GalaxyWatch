#!/usr/bin/env python3
"""
Level 1 pipeline: collect movie + TV metadata from TMDB.

Three steps, run in order:

  python fetch_tmdb.py ids      # download TMDB's daily ID export, pick candidates by popularity
  python fetch_tmdb.py fetch    # fetch details/keywords/credits for each candidate (resumable)
  python fetch_tmdb.py report   # apply quality filters, pick final titles, print coverage report

Setup:
  pip install requests
  export TMDB_API_KEY="your key"      # v3 API key, or the long v4 "read access token"

Output layout (all under ./data):
  exports/                 downloaded ID export files
  candidates.json          ordered list of titles we plan to fetch
  raw/movie/<id>.json      one trimmed JSON file per movie
  raw/tv/<id>.json         one trimmed JSON file per TV show
  selected.json            final list after quality filters (input for Level 2)
"""

import argparse
import gzip
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

API = "https://api.themoviedb.org/3"
EXPORT_URL = "https://files.tmdb.org/p/exports/{prefix}_{date}.json.gz"

DATA = Path("data")
EXPORTS = DATA / "exports"
RAW = DATA / "raw"
CANDIDATES = DATA / "candidates.json"
SELECTED = DATA / "selected.json"

MIN_OVERVIEW_CHARS = 40
TOP_CAST = 15


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class RateLimiter:
    """Spaces requests evenly across all threads (about `per_second` per second)."""

    def __init__(self, per_second):
        self.interval = 1.0 / per_second
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_time)
            self.next_time = slot + self.interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


def make_session(pool_size):
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        sys.exit("Set the TMDB_API_KEY environment variable first.")
    session = requests.Session()
    session.mount("https://", HTTPAdapter(pool_maxsize=pool_size))
    if len(key) > 60:  # v4 read-access token (a long JWT)
        session.headers["Authorization"] = f"Bearer {key}"
    else:  # v3 API key
        session.params = {"api_key": key}
    return session


def write_json_atomic(path, obj):
    """Write via a temp file so a crash never leaves a half-written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def api_get(session, limiter, path, params=None, retries=6):
    """GET with retries. Returns JSON, {"_error": code} for permanent errors, None if we gave up."""
    url = API + path
    for attempt in range(retries):
        limiter.wait()
        try:
            r = session.get(url, params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return {"_error": 404}
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
            continue
        if r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        return {"_error": r.status_code}
    return None  # not saved, so the next run tries again


# --------------------------------------------------------------------------
# step 1: ids
# --------------------------------------------------------------------------

def download_export(prefix):
    """Download today's export, falling back to the previous few days (it is published ~8am UTC)."""
    EXPORTS.mkdir(parents=True, exist_ok=True)
    for days_back in range(4):
        day = datetime.now(timezone.utc) - timedelta(days=days_back)
        url = EXPORT_URL.format(prefix=prefix, date=day.strftime("%m_%d_%Y"))
        print(f"  trying {url}")
        r = requests.get(url, timeout=180)
        if r.status_code == 200:
            text = gzip.decompress(r.content).decode("utf-8")
            (EXPORTS / f"{prefix}_{day:%Y_%m_%d}.json").write_text(text, encoding="utf-8")
            return text
    sys.exit(f"Could not download the {prefix} export.")


def cmd_ids(args):
    candidates = []
    for kind, prefix, limit in (
        ("movie", "movie_ids", args.movies),
        ("tv", "tv_series_ids", args.tv),
    ):
        print(f"Downloading {kind} export...")
        rows = []
        for line in download_export(prefix).splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("adult") or row.get("video"):
                continue
            rows.append({"type": kind, "id": row["id"], "popularity": row.get("popularity", 0)})
        rows.sort(key=lambda r: r["popularity"], reverse=True)
        print(f"  {len(rows)} usable {kind} entries, keeping top {limit}")
        candidates.extend(rows[:limit])

    write_json_atomic(CANDIDATES, candidates)
    print(f"Saved {len(candidates)} candidates to {CANDIDATES}")


# --------------------------------------------------------------------------
# step 2: fetch
# --------------------------------------------------------------------------

def names(items, limit=None):
    out = [i["name"] for i in (items or []) if i.get("name")]
    return out[:limit] if limit else out


def trim(kind, d):
    """Keep only the fields later levels need, so 20k+ files stay small."""
    is_movie = kind == "movie"
    date = (d.get("release_date") if is_movie else d.get("first_air_date")) or ""

    kw_block = d.get("keywords") or {}
    keywords = names(kw_block.get("keywords") if is_movie else kw_block.get("results"))

    credits = d.get("credits") or {}
    cast = sorted(credits.get("cast") or [], key=lambda c: c.get("order", 999))
    if is_movie:
        creators = [c["name"] for c in credits.get("crew") or []
                    if c.get("job") == "Director" and c.get("name")]
    else:
        creators = names(d.get("created_by"))

    collection = d.get("belongs_to_collection") or {}
    imdb_id = d.get("imdb_id") if is_movie else (d.get("external_ids") or {}).get("imdb_id")

    return {
        "id": d["id"],
        "type": kind,
        "title": d.get("title") if is_movie else d.get("name"),
        "original_title": d.get("original_title") if is_movie else d.get("original_name"),
        "original_language": d.get("original_language"),
        "date": date,
        "year": int(date[:4]) if date[:4].isdigit() else None,
        "overview": (d.get("overview") or "").strip(),
        "tagline": (d.get("tagline") or "").strip(),
        "genres": names(d.get("genres")),
        "keywords": keywords,
        "cast": names(cast, TOP_CAST),
        "creators": creators,  # directors for movies, creators for TV
        "collection_id": collection.get("id"),
        "collection_name": collection.get("name"),
        "poster_path": d.get("poster_path"),
        "backdrop_path": d.get("backdrop_path"),
        "vote_average": d.get("vote_average"),
        "vote_count": d.get("vote_count", 0),
        "popularity": d.get("popularity"),
        "runtime": d.get("runtime") if is_movie else None,
        "seasons": None if is_movie else d.get("number_of_seasons"),
        "episodes": None if is_movie else d.get("number_of_episodes"),
        "status": d.get("status"),
        "adult": bool(d.get("adult", False)),
        "imdb_id": imdb_id,
    }


def fetch_one(session, limiter, kind, tid):
    path = RAW / kind / f"{tid}.json"
    append = "keywords,credits" + (",external_ids" if kind == "tv" else "")
    data = api_get(session, limiter, f"/{kind}/{tid}",
                   {"language": "en-US", "append_to_response": append})
    if data is None:
        return "failed"
    if "_error" in data:
        # Remember permanent errors (e.g. 404) so we do not retry them forever.
        write_json_atomic(path, {"id": tid, "type": kind, "_error": data["_error"]})
        return "error"
    write_json_atomic(path, trim(kind, data))
    return "ok"


def cmd_fetch(args):
    if not CANDIDATES.exists():
        sys.exit("Run the 'ids' step first.")
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))

    todo = [c for c in candidates if not (RAW / c["type"] / f"{c['id']}.json").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(candidates)} candidates, {len(todo)} still to fetch.")
    if not todo:
        return

    session = make_session(args.workers * 2)
    limiter = RateLimiter(args.rate)

    # Check the key before launching thousands of requests.
    check = session.get(f"{API}/configuration", timeout=30)
    if check.status_code in (401, 403):
        sys.exit("TMDB rejected the API key. Check TMDB_API_KEY.")

    counts = {"ok": 0, "error": 0, "failed": 0}
    started = time.monotonic()
    chunk = 400
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i in range(0, len(todo), chunk):
                batch = todo[i: i + chunk]
                results = pool.map(lambda c: fetch_one(session, limiter, c["type"], c["id"]), batch)
                for res in results:
                    counts[res] += 1
                done = i + len(batch)
                elapsed = time.monotonic() - started
                eta = elapsed / done * (len(todo) - done)
                print(f"  {done}/{len(todo)}  ok={counts['ok']} "
                      f"missing={counts['error']} retry-later={counts['failed']}  "
                      f"ETA {eta / 60:.1f} min", flush=True)
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where it left off.")
        os._exit(0)

    print("Done. If 'retry-later' is above 0, run 'fetch' again to pick those up.")


# --------------------------------------------------------------------------
# step 3: report + selection
# --------------------------------------------------------------------------

def load_all():
    titles = []
    for kind in ("movie", "tv"):
        folder = RAW / kind
        if not folder.exists():
            continue
        for f in folder.glob("*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            if "_error" not in d:
                titles.append(d)
    return titles


def passes(t, min_votes):
    return (
        not t["adult"]
        and bool(t["title"])
        and len(t["overview"]) >= MIN_OVERVIEW_CHARS
        and (t["vote_count"] or 0) >= min_votes
    )


def pct(n, total):
    return f"{n} ({n / total * 100:.0f}%)" if total else "0"


def cmd_report(args):
    titles = load_all()
    if not titles:
        sys.exit("No fetched titles found. Run the 'fetch' step first.")

    print(f"Fetched titles: {len(titles)} "
          f"({sum(t['type'] == 'movie' for t in titles)} movies, "
          f"{sum(t['type'] == 'tv' for t in titles)} TV)")

    good = [t for t in titles if passes(t, args.min_votes)]
    print(f"Pass quality filter (overview >= {MIN_OVERVIEW_CHARS} chars, "
          f"votes >= {args.min_votes}, not adult): {len(good)}")

    selected = []
    for kind, limit in (("movie", args.movies), ("tv", args.tv)):
        pool = sorted((t for t in good if t["type"] == kind),
                      key=lambda t: t["popularity"] or 0, reverse=True)
        selected.extend(pool[:limit])
        print(f"  {kind}: {len(pool)} eligible, selecting {min(limit, len(pool))}")

    n = len(selected)
    print(f"\nSelected: {n} titles")
    print("Coverage in the selected set:")
    print(f"  with keywords:  {pct(sum(bool(t['keywords']) for t in selected), n)}")
    print(f"  with genres:    {pct(sum(bool(t['genres']) for t in selected), n)}")
    print(f"  with cast:      {pct(sum(bool(t['cast']) for t in selected), n)}")
    print(f"  with director/creator: {pct(sum(bool(t['creators']) for t in selected), n)}")
    print(f"  with poster:    {pct(sum(bool(t['poster_path']) for t in selected), n)}")
    print(f"  in a collection (franchise): {pct(sum(bool(t['collection_id']) for t in selected), n)}")
    print(f"  short overview (< 100 chars): {pct(sum(len(t['overview']) < 100 for t in selected), n)}")
    if n:
        avg = sum(len(t["overview"]) for t in selected) / n
        print(f"  average overview length: {avg:.0f} chars")

    write_json_atomic(SELECTED, [{"type": t["type"], "id": t["id"]} for t in selected])
    print(f"\nSaved {SELECTED}")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ids", help="build the candidate list from TMDB's daily export")
    s.add_argument("--movies", type=int, default=22000, help="candidate movies to keep (default 22000)")
    s.add_argument("--tv", type=int, default=11000, help="candidate TV shows to keep (default 11000)")
    s.set_defaults(func=cmd_ids)

    s = sub.add_parser("fetch", help="fetch details for every candidate (resumable)")
    s.add_argument("--rate", type=float, default=15, help="requests per second (default 15)")
    s.add_argument("--workers", type=int, default=10, help="parallel workers (default 10)")
    s.add_argument("--limit", type=int, default=0, help="only fetch this many (use 50 for a first test)")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("report", help="filter, select the final titles, print coverage")
    s.add_argument("--min-votes", type=int, default=50, help="minimum vote count (default 50)")
    s.add_argument("--movies", type=int, default=13000, help="final movie count (default 13000)")
    s.add_argument("--tv", type=int, default=7000, help="final TV count (default 7000)")
    s.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()