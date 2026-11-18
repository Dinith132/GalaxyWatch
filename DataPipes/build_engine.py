#!/usr/bin/env python3
"""
Level 2 pipeline: turn the fetched TMDB data into similarity data + 3D positions + SQLite.

Run everything in one go:
  python build_engine.py build

Check recommendation quality (prints the top neighbors of titles you name):
  python build_engine.py check                       # a default list of well-known titles
  python build_engine.py check "Inception" "Spirited Away" --top 10

Useful options for `build`:
  --stop-after embed|neighbors   stop early (use this while tuning the text recipe)
  --force                        ignore cached embeddings / UMAP and recompute

Setup:
  pip install numpy umap-learn sentence-transformers
  pip install torch --index-url https://download.pytorch.org/whl/cpu     (CPU-only torch, smaller)

Inputs (from Level 1):   data/selected.json, data/raw/<type>/<id>.json
Outputs (data/build/):
  texts_sample.txt         the text recipe applied to the first 30 titles (to eyeball it)
  titles_light.json        compact title list used by the `check` command
  embeddings.npy           N x 384 normalized embeddings (cached)
  neighbors.npz            top-K neighbors per title: idx, score, same-franchise flag
  umap3d.npy               N x 3 galaxy positions, scaled to [-1, 1] (cached)
  movies.db                SQLite database: the source of truth for Level 3
"""

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path("data")
RAW = DATA / "raw"
SELECTED = DATA / "selected.json"
BUILD = DATA / "build"
DB_PATH = BUILD / "movies.db"

# ---- tunable settings -------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256          # the model reads roughly the first ~200 words of each text
TOP_K = 30                    # neighbors stored per title
CANDIDATES = 150              # how many near matches to consider before franchise limiting
MAX_PER_COLLECTION = 3        # at most this many titles from one TMDB collection (franchise) per list
MAX_KEYWORDS_IN_TEXT = 15
MAX_CAST_IN_TEXT = 5
UMAP_PARAMS = dict(n_components=3, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)

DEFAULT_CHECK_TITLES = [
    "Inception", "Breaking Bad", "Spirited Away", "The Dark Knight", "Interstellar",
    "Toy Story", "Parasite", "Stranger Things", "Game of Thrones", "The Notebook",
    "Get Out", "Attack on Titan",
]

# ---- genre groups (for the galaxy colors) ------------------------------------
# A title gets the FIRST group in this list that any of its genres belongs to.
GENRE_GROUPS = [
    ("Animation & Family", "#b8e986", ["Animation", "Family", "Kids"]),
    ("Documentary",        "#c0c7d1", ["Documentary"]),
    ("Horror",             "#d00000", ["Horror"]),
    ("Sci-Fi & Fantasy",   "#a66bff", ["Science Fiction", "Fantasy", "Sci-Fi & Fantasy"]),
    ("Crime & Thriller",   "#2ec4b6", ["Crime", "Thriller", "Mystery"]),
    ("Action & Adventure", "#ff6b35", ["Action", "Adventure", "Action & Adventure", "War",
                                       "War & Politics", "Western"]),
    ("Romance",            "#ff77b7", ["Romance"]),
    ("Comedy",             "#ffd23f", ["Comedy"]),
    ("Drama",              "#4d96ff", ["Drama", "History"]),
    ("Other",              "#6b7280", []),   # music, news, reality, TV movie, no genre, etc.
]


# =============================================================================
# 1. load titles + build the text recipe
# =============================================================================

def load_titles():
    if not SELECTED.exists():
        sys.exit("data/selected.json not found. Finish Level 1 first (fetch_tmdb.py report).")
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    titles = []
    for s in selected:
        f = RAW / s["type"] / f"{s['id']}.json"
        if not f.exists():
            continue
        t = json.loads(f.read_text(encoding="utf-8"))
        if "_error" in t:
            continue
        titles.append(t)
    # Most popular first, so compact number 0 is the most popular title.
    titles.sort(key=lambda t: (-(t.get("popularity") or 0), t["type"], t["id"]))
    return titles


def build_text(t):
    """
    The recipe. Strongest signals come first because the model only reads the start:
    title -> genres -> keywords -> tagline -> overview -> director/creator -> top cast.
    """
    head = t["title"] + (f" ({t['year']})" if t.get("year") else "") + "."
    parts = [head]
    if t["genres"]:
        parts.append("Genres: " + ", ".join(t["genres"]) + ".")
    if t["keywords"]:
        parts.append("Keywords: " + ", ".join(t["keywords"][:MAX_KEYWORDS_IN_TEXT]) + ".")
    if t["tagline"]:
        parts.append(t["tagline"])
    parts.append(t["overview"])
    if t["creators"]:
        label = "Directed by" if t["type"] == "movie" else "Created by"
        parts.append(f"{label} {', '.join(t['creators'][:2])}.")
    if t["cast"]:
        parts.append("Starring " + ", ".join(t["cast"][:MAX_CAST_IN_TEXT]) + ".")
    return " ".join(p for p in parts if p)


# =============================================================================
# 2. embeddings (cached)
# =============================================================================

def get_embeddings(texts, force):
    digest = hashlib.sha1(("\n".join(texts) + MODEL_NAME + str(MAX_SEQ_LENGTH)).encode("utf-8")).hexdigest()
    emb_path, meta_path = BUILD / "embeddings.npy", BUILD / "embeddings_meta.json"

    if not force and emb_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("digest") == digest:
            print("  texts unchanged since last run, using cached embeddings")
            return np.load(emb_path), digest

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("Missing dependency. Run: pip install sentence-transformers")

    print(f"  loading model {MODEL_NAME} (first run downloads about 90 MB)")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    E = model.encode(texts, batch_size=64, normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    np.save(emb_path, E)
    meta_path.write_text(json.dumps({"digest": digest, "model": MODEL_NAME, "count": len(texts)}))
    return E, digest


# =============================================================================
# 3. neighbors (exact cosine similarity, with franchise limiting)
# =============================================================================

def compute_neighbors(E, collection_ids, k=TOP_K, candidates=CANDIDATES,
                      max_per_collection=MAX_PER_COLLECTION, chunk=1000):
    """
    For each title: score against all others, take the best `candidates`, then walk them
    in score order and keep up to `k`, allowing at most `max_per_collection` titles from
    the same TMDB collection. Embeddings are normalized, so dot product = cosine similarity.
    """
    n = len(E)
    candidates = min(candidates, n - 1)
    k = min(k, candidates)
    nbr_idx = np.full((n, k), -1, dtype=np.int32)
    nbr_score = np.zeros((n, k), dtype=np.float32)
    nbr_same = np.zeros((n, k), dtype=bool)

    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        sims = E[start:end] @ E.T
        sims[np.arange(end - start), np.arange(start, end)] = -np.inf   # exclude itself
        part = np.argpartition(-sims, candidates - 1, axis=1)[:, :candidates]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        part = np.take_along_axis(part, order, axis=1)
        part_scores = np.take_along_axis(part_scores, order, axis=1)

        for r in range(end - start):
            i = start + r
            own = collection_ids[i]
            used = {}
            filled = 0
            for j, sc in zip(part[r], part_scores[r]):
                c = collection_ids[j]
                if c >= 0:
                    if used.get(c, 0) >= max_per_collection:
                        continue
                    used[c] = used.get(c, 0) + 1
                nbr_idx[i, filled] = j
                nbr_score[i, filled] = sc
                nbr_same[i, filled] = bool(c >= 0 and c == own)
                filled += 1
                if filled == k:
                    break
        print(f"  neighbors: {end}/{n}", end="\r", flush=True)
    print()
    return nbr_idx, nbr_score, nbr_same


# =============================================================================
# 4. UMAP to 3D (cached)
# =============================================================================

def get_umap(E, emb_digest, force):
    digest = hashlib.sha1((emb_digest + json.dumps(UMAP_PARAMS, sort_keys=True)).encode()).hexdigest()
    path, meta_path = BUILD / "umap3d.npy", BUILD / "umap_meta.json"
    if not force and path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()).get("digest") == digest:
            print("  embeddings unchanged, using cached UMAP positions")
            return np.load(path)

    try:
        import umap
    except ImportError:
        sys.exit("Missing dependency. Run: pip install umap-learn")

    print("  running UMAP (a few minutes for 20k titles)")
    xyz = umap.UMAP(verbose=True, **UMAP_PARAMS).fit_transform(E).astype(np.float32)
    xyz -= xyz.mean(axis=0)
    xyz /= np.abs(xyz).max()           # scale into [-1, 1], same scale on every axis
    np.save(path, xyz)
    meta_path.write_text(json.dumps({"digest": digest}))
    return xyz


# =============================================================================
# 5. genre colors
# =============================================================================

def assign_color_groups(titles):
    lookup = {}
    for gid, (_, _, genres) in enumerate(GENRE_GROUPS):
        for g in genres:
            lookup[g] = gid
    other = len(GENRE_GROUPS) - 1
    out = np.full(len(titles), other, dtype=np.int16)
    for i, t in enumerate(titles):
        found = [lookup[g] for g in t["genres"] if g in lookup]
        if found:
            out[i] = min(found)          # lowest index = highest priority
    return out


# =============================================================================
# 6. SQLite
# =============================================================================

def write_db(titles, xyz, groups, nbr_idx, nbr_score, nbr_same):
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE titles (
            idx INTEGER PRIMARY KEY,
            tmdb_id INTEGER NOT NULL, type TEXT NOT NULL,
            title TEXT NOT NULL, original_title TEXT, year INTEGER,
            overview TEXT, tagline TEXT,
            genres TEXT, keywords TEXT, cast_names TEXT, creators TEXT,
            collection_id INTEGER, collection_name TEXT,
            poster_path TEXT, backdrop_path TEXT,
            vote_average REAL, vote_count INTEGER, popularity REAL,
            runtime INTEGER, seasons INTEGER, episodes INTEGER, status TEXT, imdb_id TEXT,
            color_group INTEGER, x REAL, y REAL, z REAL,
            UNIQUE (type, tmdb_id)
        );
        CREATE TABLE neighbors (
            title_idx INTEGER NOT NULL, rank_no INTEGER NOT NULL,
            neighbor_idx INTEGER NOT NULL, score REAL NOT NULL, same_collection INTEGER NOT NULL,
            PRIMARY KEY (title_idx, rank_no)
        );
        CREATE TABLE genre_groups (id INTEGER PRIMARY KEY, name TEXT, color TEXT);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
    """)

    dumps = lambda v: json.dumps(v, ensure_ascii=False)
    con.executemany(
        "INSERT INTO titles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(
            i, t["id"], t["type"], t["title"], t.get("original_title"), t.get("year"),
            t["overview"], t["tagline"],
            dumps(t["genres"]), dumps(t["keywords"]), dumps(t["cast"]), dumps(t["creators"]),
            t.get("collection_id"), t.get("collection_name"),
            t.get("poster_path"), t.get("backdrop_path"),
            t.get("vote_average"), t.get("vote_count"), t.get("popularity"),
            t.get("runtime"), t.get("seasons"), t.get("episodes"), t.get("status"), t.get("imdb_id"),
            int(groups[i]), float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]),
        ) for i, t in enumerate(titles)],
    )

    rows = []
    for i in range(len(titles)):
        for r in range(nbr_idx.shape[1]):
            j = int(nbr_idx[i, r])
            if j < 0:
                break
            rows.append((i, r, j, float(nbr_score[i, r]), int(nbr_same[i, r])))
    con.executemany("INSERT INTO neighbors VALUES (?,?,?,?,?)", rows)
    con.execute("CREATE INDEX idx_neighbors_title ON neighbors(title_idx)")
    con.execute("CREATE INDEX idx_titles_type ON titles(type)")

    con.executemany("INSERT INTO genre_groups VALUES (?,?,?)",
                    [(i, name, color) for i, (name, color, _) in enumerate(GENRE_GROUPS)])
    con.executemany("INSERT INTO meta VALUES (?,?)", [
        ("model", MODEL_NAME), ("top_k", str(TOP_K)),
        ("max_per_collection", str(MAX_PER_COLLECTION)), ("count", str(len(titles))),
        ("built_at", time.strftime("%Y-%m-%d %H:%M:%S")),
    ])

    try:   # full-text search (nice to have; the site itself will use a JS search library)
        con.execute("""CREATE VIRTUAL TABLE titles_fts USING fts5(
            title, original_title, content='titles', content_rowid='idx',
            tokenize='unicode61 remove_diacritics 2')""")
        con.execute("INSERT INTO titles_fts(titles_fts) VALUES('rebuild')")
    except sqlite3.OperationalError:
        print("  note: this SQLite build has no FTS5, skipped the search table")

    con.commit()
    con.close()


# =============================================================================
# commands
# =============================================================================

def cmd_build(args):
    BUILD.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("[1/6] Loading titles")
    titles = load_titles()
    n = len(titles)
    if n < 100:
        sys.exit(f"Only {n} titles found. Check data/selected.json and data/raw.")
    print(f"  {n} titles")
    light = [{"type": t["type"], "id": t["id"], "title": t["title"], "year": t.get("year"),
              "collection_id": t.get("collection_id")} for t in titles]
    (BUILD / "titles_light.json").write_text(json.dumps(light, ensure_ascii=False), encoding="utf-8")

    print("[2/6] Building texts")
    texts = [build_text(t) for t in titles]
    (BUILD / "texts_sample.txt").write_text("\n\n".join(texts[:30]), encoding="utf-8")
    print(f"  average text length {sum(map(len, texts)) / n:.0f} chars "
          f"(sample saved to {BUILD / 'texts_sample.txt'})")

    print("[3/6] Embeddings")
    E, emb_digest = get_embeddings(texts, args.force)
    if args.stop_after == "embed":
        print("Stopped after embeddings.")
        return

    print("[4/6] Neighbors")
    collection_ids = np.array([t["collection_id"] if t.get("collection_id") else -1 for t in titles],
                              dtype=np.int64)
    nbr_idx, nbr_score, nbr_same = compute_neighbors(E, collection_ids)
    np.savez(BUILD / "neighbors.npz", idx=nbr_idx, score=nbr_score, same=nbr_same)
    if args.stop_after == "neighbors":
        print("Stopped after neighbors. Now run:  python build_engine.py check")
        return

    print("[5/6] UMAP 3D positions")
    xyz = get_umap(E, emb_digest, args.force)

    print("[6/6] Writing SQLite")
    groups = assign_color_groups(titles)
    write_db(titles, xyz, groups, nbr_idx, nbr_score, nbr_same)

    # ---- summary ----
    filled = (nbr_idx >= 0).sum(axis=1)
    print("\nSummary")
    print(f"  titles: {n}   database: {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"  neighbors per title: min {filled.min()}, average {filled.mean():.1f}")
    print(f"  similarity score, rank 1:  average {nbr_score[:, 0].mean():.3f}, lowest {nbr_score[:, 0].min():.3f}")
    last = min(9, nbr_score.shape[1] - 1)
    print(f"  similarity score, rank {last + 1}: average {nbr_score[:, last].mean():.3f}")
    print("  titles per color group:")
    for gid, (name, color, _) in enumerate(GENRE_GROUPS):
        print(f"    {name:20s} {color}  {int((groups == gid).sum())}")
    print(f"\nDone in {(time.time() - started) / 60:.1f} min. "
          f"Check quality with:  python build_engine.py check")


def find_matches(light, query):
    q = query.strip().lower()
    exact = [i for i, t in enumerate(light) if t["title"].lower() == q]
    if exact:
        return exact
    return [i for i, t in enumerate(light) if q in t["title"].lower()]


def label(t):
    return f"{t['title']} ({t['year'] or '?'}, {t['type']})"


def cmd_check(args):
    light_path, nb_path = BUILD / "titles_light.json", BUILD / "neighbors.npz"
    if not light_path.exists() or not nb_path.exists():
        sys.exit("Run `python build_engine.py build --stop-after neighbors` first.")
    light = json.loads(light_path.read_text(encoding="utf-8"))
    nb = np.load(nb_path)
    nbr_idx, nbr_score, nbr_same = nb["idx"], nb["score"], nb["same"]

    for query in (args.titles or DEFAULT_CHECK_TITLES):
        matches = find_matches(light, query)
        if not matches:
            print(f"\n=== {query}: not in the dataset ===")
            continue
        i = matches[0]    # list is sorted by popularity, so this is the most popular match
        print(f"\n=== {label(light[i])} ===")
        if len(matches) > 1:
            others = ", ".join(label(light[m]) for m in matches[1:4])
            print(f"    (other matches: {others})")
        for r in range(min(args.top, nbr_idx.shape[1])):
            j = int(nbr_idx[i, r])
            if j < 0:
                break
            flag = "  [same franchise]" if nbr_same[i, r] else ""
            print(f"  {r + 1:2d}. {nbr_score[i, r]:.3f}  {label(light[j])}{flag}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="run the whole Level 2 pipeline")
    b.add_argument("--stop-after", choices=["embed", "neighbors"], help="stop early")
    b.add_argument("--force", action="store_true", help="recompute cached embeddings and UMAP")
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("check", help="print the nearest neighbors of some titles")
    c.add_argument("titles", nargs="*", help="title names (default: a built-in list)")
    c.add_argument("--top", type=int, default=10, help="neighbors to show (default 10)")
    c.set_defaults(func=cmd_check)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
