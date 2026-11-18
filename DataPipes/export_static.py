#!/usr/bin/env python3
"""
Level 3: export movies.db into small static files the website can load.

  python export_static.py
  python export_static.py --db data/build/movies.db --out data/export

Output folder (default data/export):
  manifest.json          version stamp, counts, shard rules, color legend, file layouts
  galaxy.bin             positions + attributes for every title (binary, 16 bytes per title)
  search-index.json      one short row per title, for client-side search and id lookup
  posters.json           poster path per title (for hover cards, one small file)
  neighbors/NNN.json     top neighbors + scores, 500 titles per shard
  details/NNN.json       overview, cast, rating, links, 500 titles per shard

Every file is indexed by the compact title number `idx` (0..N-1, most popular first).
Shard of a title = idx // shard_size, position inside the shard = idx % shard_size.

Binary layout of galaxy.bin (little-endian), N = manifest.count:
  bytes 0 .. N*12-1      Float32 x,y,z per title            -> new Float32Array(buf, 0, N*3)
  bytes N*12 .. N*16-1   Uint8 x4 per title:                -> new Uint8Array(buf, N*12, N*4)
                           [0] color group id (see manifest.genre_groups)
                           [1] type: 0 = movie, 1 = tv
                           [2] size: 0-255, how well known (log of vote count)
                           [3] reserved (0)

Scores are stored as integers: real cosine similarity = value / 1000.
"""

import argparse
import gzip
import hashlib
import json
import random
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_DB = Path("data/build/movies.db")
DEFAULT_OUT = Path("data/export")

SHARD_SIZE = 500
TOP_K = 30
KEYWORDS_IN_DETAILS = 8
CAST_IN_DETAILS = 10
IMAGE_BASE = "https://image.tmdb.org/t/p/"
SHARD_RAW_WARN_BYTES = 400_000
TOTAL_LIMIT_MB = 60

ATTRIBUTION = ("This product uses the TMDB API but is not endorsed or certified by TMDB. "
               "Movie and TV data and images courtesy of The Movie Database (themoviedb.org).")


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# =============================================================================
# load from SQLite
# =============================================================================

def load_db(db_path, top_k):
    if not db_path.exists():
        sys.exit(f"{db_path} not found. Run build_engine.py build first.")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    titles = [dict(r) for r in con.execute("SELECT * FROM titles ORDER BY idx")]
    n = len(titles)
    if n == 0 or [t["idx"] for t in titles] != list(range(n)):
        sys.exit("titles.idx is not a clean 0..N-1 sequence. Re-run build_engine.py build.")

    neighbors = [[] for _ in range(n)]
    for r in con.execute(
        "SELECT title_idx, neighbor_idx, score, same_collection FROM neighbors "
        "WHERE rank_no < ? ORDER BY title_idx, rank_no", (top_k,)
    ):
        neighbors[r["title_idx"]].append((r["neighbor_idx"], r["score"], r["same_collection"]))

    groups = [dict(r) for r in con.execute("SELECT id, name, color FROM genre_groups ORDER BY id")]
    meta = {r[0]: r[1] for r in con.execute("SELECT key, value FROM meta")}
    con.close()
    return titles, neighbors, groups, meta


# =============================================================================
# build each output
# =============================================================================

def build_galaxy(titles):
    n = len(titles)
    pos = np.array([[t["x"], t["y"], t["z"]] for t in titles], dtype="<f4")
    votes = np.log1p(np.array([t["vote_count"] or 0 for t in titles], dtype=np.float64))
    size = np.round(255 * votes / max(votes.max(), 1e-9)).astype(np.uint8)
    attrs = np.zeros((n, 4), dtype=np.uint8)
    attrs[:, 0] = [t["color_group"] for t in titles]
    attrs[:, 1] = [1 if t["type"] == "tv" else 0 for t in titles]
    attrs[:, 2] = size
    return pos.tobytes() + attrs.tobytes()


def build_search_index(titles):
    rows = []
    for t in titles:
        row = [t["title"], t["year"] or 0, 1 if t["type"] == "tv" else 0, t["tmdb_id"]]
        alt = t.get("original_title") or ""
        if alt and alt != t["title"]:
            row.append(alt)
        rows.append(row)
    return dump({"fields": ["title", "year", "type", "tmdb_id", "alt_title"], "rows": rows})


def build_posters(titles):
    return dump([t["poster_path"] or "" for t in titles])


def neighbor_item(entries):
    item = {"n": [e[0] for e in entries], "s": [int(round(e[1] * 1000)) for e in entries]}
    same = [rank for rank, e in enumerate(entries) if e[2]]
    if same:
        item["f"] = same   # ranks (positions in "n") that belong to the same franchise
    return item


def detail_item(t):
    d = {"i": t["idx"], "id": t["tmdb_id"], "type": t["type"], "title": t["title"]}

    def put(key, value):
        if value is None or value == "" or value == []:
            return
        d[key] = value

    put("year", t["year"])
    put("tagline", t["tagline"])
    put("overview", t["overview"])
    put("genres", json.loads(t["genres"] or "[]"))
    put("keywords", json.loads(t["keywords"] or "[]")[:KEYWORDS_IN_DETAILS])
    put("cast", json.loads(t["cast_names"] or "[]")[:CAST_IN_DETAILS])
    put("creators", json.loads(t["creators"] or "[]"))
    put("rating", round(t["vote_average"], 1) if t["vote_average"] is not None else None)
    put("votes", t["vote_count"])
    put("runtime", t["runtime"])
    put("seasons", t["seasons"])
    put("episodes", t["episodes"])
    put("status", t["status"])
    put("poster", t["poster_path"])
    put("backdrop", t["backdrop_path"])
    put("imdb", t["imdb_id"])
    put("collection", t["collection_name"] if t["collection_id"] else None)
    return d


def build_shards(titles, neighbors, shard_size):
    files = {}
    n = len(titles)
    for shard, start in enumerate(range(0, n, shard_size)):
        end = min(n, start + shard_size)
        name = f"{shard:03d}.json"
        files[f"neighbors/{name}"] = dump(
            {"start": start, "items": [neighbor_item(neighbors[i]) for i in range(start, end)]}
        ).encode("utf-8")
        files[f"details/{name}"] = dump(
            {"start": start, "items": [detail_item(titles[i]) for i in range(start, end)]}
        ).encode("utf-8")
    return files


# =============================================================================
# write + validate
# =============================================================================

def write_all(out, files):
    for sub in ("neighbors", "details"):
        shutil.rmtree(out / sub, ignore_errors=True)
    for stale in ("manifest.json", "galaxy.bin", "search-index.json", "posters.json"):
        (out / stale).unlink(missing_ok=True)
    for name, data in files.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def validate(out, titles, neighbors, groups, shard_size, top_k):
    errors, warnings = [], []
    n = len(titles)
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if m["count"] != n:
        errors.append("manifest count does not match the database")

    # galaxy.bin
    gal = (out / "galaxy.bin").read_bytes()
    if len(gal) != n * 16:
        errors.append(f"galaxy.bin is {len(gal)} bytes, expected {n * 16}")
    else:
        pos = np.frombuffer(gal, dtype="<f4", count=n * 3)
        attrs = np.frombuffer(gal, dtype=np.uint8, offset=n * 12).reshape(n, 4)
        if not np.isfinite(pos).all():
            errors.append("galaxy.bin has NaN/inf positions")
        elif np.abs(pos).max() > 1.001:
            warnings.append(f"positions exceed [-1, 1] (max {np.abs(pos).max():.3f})")
        if attrs[:, 0].max() >= len(groups):
            errors.append("galaxy.bin has a color group id that is not in the legend")

    # search index
    search = json.loads((out / "search-index.json").read_text(encoding="utf-8"))
    if len(search["rows"]) != n:
        errors.append("search-index.json row count is wrong")
    keys = {(r[2], r[3]) for r in search["rows"]}
    if len(keys) != len(search["rows"]):
        errors.append("search-index.json has duplicate (type, tmdb_id) pairs")

    # posters
    if len(json.loads((out / "posters.json").read_text(encoding="utf-8"))) != n:
        errors.append("posters.json length is wrong")

    # shards
    largest = {"neighbors": 0, "details": 0}
    empty_lists = 0
    shard_count = (n + shard_size - 1) // shard_size
    for shard in range(shard_count):
        start = shard * shard_size
        expect = min(shard_size, n - start)
        name = f"{shard:03d}.json"

        nb_path, de_path = out / "neighbors" / name, out / "details" / name
        if not nb_path.exists() or not de_path.exists():
            errors.append(f"missing shard {name}")
            continue
        largest["neighbors"] = max(largest["neighbors"], nb_path.stat().st_size)
        largest["details"] = max(largest["details"], de_path.stat().st_size)

        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        de = json.loads(de_path.read_text(encoding="utf-8"))
        if nb["start"] != start or len(nb["items"]) != expect:
            errors.append(f"neighbors/{name}: wrong start or item count")
            continue
        if de["start"] != start or len(de["items"]) != expect:
            errors.append(f"details/{name}: wrong start or item count")
            continue

        for k in range(expect):
            idx = start + k
            item, det = nb["items"][k], de["items"][k]
            ns, ss = item["n"], item["s"]
            if not ns:
                empty_lists += 1
            if len(ns) != len(ss) or len(ns) > top_k:
                errors.append(f"idx {idx}: neighbor/score lists are inconsistent")
            elif any((j < 0 or j >= n or j == idx) for j in ns):
                errors.append(f"idx {idx}: neighbor index out of range or points to itself")
            elif len(set(ns)) != len(ns):
                errors.append(f"idx {idx}: duplicate neighbors")
            elif any(ss[x] < ss[x + 1] for x in range(len(ss) - 1)):
                errors.append(f"idx {idx}: scores are not sorted high to low")
            if det["i"] != idx or det["title"] != titles[idx]["title"]:
                errors.append(f"idx {idx}: detail record does not match the database")
            if not det.get("overview"):
                warnings.append(f"idx {idx}: empty overview")

    if empty_lists:
        warnings.append(f"{empty_lists} titles have no neighbors")
    for kind, size in largest.items():
        if size > SHARD_RAW_WARN_BYTES:
            warnings.append(f"largest {kind} shard is {size / 1000:.0f} KB (target is under "
                            f"{SHARD_RAW_WARN_BYTES // 1000} KB); use a smaller --shard-size")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if total > TOTAL_LIMIT_MB * 1e6:
        errors.append(f"export is {total / 1e6:.0f} MB, over the {TOTAL_LIMIT_MB} MB target")

    # spot check against the database
    rng = random.Random(0)
    for idx in rng.sample(range(n), min(200, n)):
        shard, pos_in = divmod(idx, shard_size)
        nb = json.loads((out / "neighbors" / f"{shard:03d}.json").read_text(encoding="utf-8"))
        item = nb["items"][pos_in]
        want = neighbors[idx]
        if want and (item["n"][0] != want[0][0] or item["s"][0] != int(round(want[0][1] * 1000))):
            errors.append(f"idx {idx}: top neighbor in the export differs from the database")
    return errors, warnings, total


def gz_size(data):
    return len(gzip.compress(data, compresslevel=6))


# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE, help="titles per shard (default 500)")
    p.add_argument("--top-k", type=int, default=TOP_K, help="neighbors per title to export (default 30)")
    args = p.parse_args()

    started = time.time()
    print(f"Reading {args.db}")
    titles, neighbors, groups, meta = load_db(args.db, args.top_k)
    n = len(titles)
    print(f"  {n} titles")

    print("Building files")
    files = {
        "galaxy.bin": build_galaxy(titles),
        "search-index.json": build_search_index(titles).encode("utf-8"),
        "posters.json": build_posters(titles).encode("utf-8"),
    }
    files.update(build_shards(titles, neighbors, args.shard_size))

    # Version stamp: changes whenever any data byte changes, so the site can bust its cache.
    h = hashlib.sha1()
    for name in sorted(files):
        h.update(name.encode())
        h.update(files[name])
    version = h.hexdigest()[:10]

    shard_count = (n + args.shard_size - 1) // args.shard_size
    manifest = {
        "version": version,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": n,
        "counts": {"movie": sum(t["type"] == "movie" for t in titles),
                   "tv": sum(t["type"] == "tv" for t in titles)},
        "shard_size": args.shard_size,
        "shard_count": shard_count,
        "top_k": args.top_k,
        "score_scale": 1000,
        "genre_groups": groups,
        "type_codes": {"0": "movie", "1": "tv"},
        "image_base": IMAGE_BASE,
        "image_sizes": {"poster_card": "w185", "poster_panel": "w342", "backdrop": "w780"},
        "tmdb_url": {"movie": "https://www.themoviedb.org/movie/", "tv": "https://www.themoviedb.org/tv/"},
        "model": meta.get("model"),
        "attribution": ATTRIBUTION,
        "files": {name: len(data) for name, data in files.items() if "/" not in name},
    }
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {args.out}")
    write_all(args.out, files)

    print("Validating (reading everything back)")
    errors, warnings, total = validate(args.out, titles, neighbors, groups, args.shard_size, args.top_k)

    # ---- size report ----
    def group_sizes(prefix=None, exact=None):
        names = [k for k in files if (k == exact if exact else k.startswith(prefix))]
        raw = sum(len(files[k]) for k in names)
        gz = sum(gz_size(files[k]) for k in names)
        return len(names), raw, gz

    print(f"\nExport version {version}")
    print(f"{'file':22s} {'count':>5s} {'raw':>10s} {'gzipped':>10s}")
    rows = [("galaxy.bin", group_sizes(exact="galaxy.bin")),
            ("search-index.json", group_sizes(exact="search-index.json")),
            ("posters.json", group_sizes(exact="posters.json")),
            ("neighbors/*.json", group_sizes(prefix="neighbors/")),
            ("details/*.json", group_sizes(prefix="details/"))]
    for label, (count, raw, gz) in rows:
        print(f"{label:22s} {count:5d} {raw / 1e6:8.2f}MB {gz / 1e6:8.2f}MB")
    print(f"{'TOTAL on disk':22s} {'':5s} {total / 1e6:8.2f}MB")

    for w in warnings[:10]:
        print(f"  warning: {w}")
    if len(warnings) > 10:
        print(f"  ... and {len(warnings) - 10} more warnings")
    if errors:
        print(f"\nVALIDATION FAILED ({len(errors)} problems):")
        for e in errors[:20]:
            print(f"  - {e}")
        sys.exit(1)

    print(f"\nValidation passed. Done in {time.time() - started:.0f}s.")
    print(f"Copy the contents of {args.out} into the website's public/data/ folder.")


if __name__ == "__main__":
    main()
