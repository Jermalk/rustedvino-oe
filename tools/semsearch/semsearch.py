#!/usr/bin/env python3
"""semsearch — semantic search over a directory of Markdown notes, using a
RustedVINO server's /v1/embeddings (and, optionally, /v1/rerank).

Why Python, not Rust: this is a lookup aid, not a shipped server component. It
has to run cold with zero build step — no `cargo build`, no lint gates. A Rust
rewrite would add a 40s+ compile before first use for a script whose entire
job is "curl, chunk some markdown, do a dot product." Keep it Python.

Corpus: every *.md under --corpus (default: the repo's dev/). Markdown only —
code chunks poorly at a BERT-class model's 512-token ceiling, and grep already
finds code symbols better than embedding similarity would.

Chunking: each file splits on the level of its first sub-header. A file whose
first sub-header is "## " splits into "## " sections (the preamble before it
is kept as its own chunk if substantial). A file whose first sub-header is
"### " is treated as an append-log of "### " entries (decision logs, session
histories) and its preamble — typically a banner or navigation — is skipped.
A file with neither is one chunk.

Store: cached to tools/semsearch/.cache/ (gitignored — local to one machine
and one server). Per-chunk, content-addressed (see _chunk_key /
_chunk_store_path): editing one line of one file only re-embeds that one
chunk, not the whole corpus — append-only logs grow one entry at a time, and a
whole-corpus fingerprint would re-embed everything on every append.

Model choice: servers don't all register the same embedding model (or a
reranking model at all). --model, then SEMSEARCH_MODEL; with neither, the
tool picks the embedding model on --target if there is exactly one, and
otherwise tells you to choose. --list-models shows what --target actually
serves. The corpus is always read locally, so a remote --target only borrows
the model, never the content.

numpy is optional. If present, search runs as a single matrix-vector
product; if absent it falls back to a pure Python loop. Both paths rank
identically — numpy only buys speed. Pure Python stays well under a second
per query for a few thousand chunks.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

try:
    import numpy as np
except ImportError:
    np = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(TOOL_DIR, ".cache")
API_KEYS_PATH = os.path.join(TOOL_DIR, "api_keys.json")

DEFAULT_TARGET = "http://localhost:11437"
DEFAULT_CORPUS = os.path.join(REPO_ROOT, "dev")

BATCH_SIZE = 16  # server-enforced cap (400 too_many_inputs above this) — not a tuning knob
MAX_CHARS = 900  # measured: this corpus's backtick/identifier-dense text tokenizes at
                  # ~2.5 chars/token, not the ~4 chars/token of plain English. 1800 chars
                  # (the naive guess) 400'd on every batch. Individual chunks that still
                  # overflow after this truncation get halved and retried (see embed_one_with_retry).


def resolve_api_key(cli_key, target):
    """Precedence: --api-key flag > SEMSEARCH_API_KEY env var > api_keys.json
    (gitignored, keyed by exact --target string — see api_keys.example.json).
    A key file beats an env var you'd otherwise have to re-set every session,
    and beats hardcoding a secret into a shell history via --api-key."""
    if cli_key:
        return cli_key
    env_key = os.environ.get("SEMSEARCH_API_KEY")
    if env_key:
        return env_key
    if os.path.exists(API_KEYS_PATH):
        try:
            with open(API_KEYS_PATH, encoding="utf-8") as f:
                keys = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"semsearch: WARNING: {API_KEYS_PATH} exists but couldn't be read ({e}) — ignoring it", file=sys.stderr)
            return None
        return keys.get(target)
    return None


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def parse_file(path):
    """One file → [(line, header, text)], line 1-indexed for grep-style
    file:line results. Splits on the level of the file's first sub-header —
    see the module docstring for the rule."""
    text = open(path, encoding="utf-8").read()
    if not text.strip():
        return []
    lines = text.split("\n")
    first = next((i for i, l in enumerate(lines) if l.startswith(("## ", "### "))), None)
    if first is None:
        title = next((l.lstrip("# ").strip() for l in lines if l.strip()), os.path.basename(path))
        return [(1, title, text)]
    marker = "### " if lines[first].startswith("### ") else "## "
    out = []
    preamble = "\n".join(lines[:first]).strip()
    if marker == "## " and len(preamble) > 50:
        title = next((l.lstrip("# ").strip() for l in lines[:first] if l.strip()), os.path.basename(path))
        out.append((1, title, preamble))
    cur, cur_start = [lines[first]], first
    for i, line in enumerate(lines[first + 1:], start=first + 1):
        if line.startswith(marker):
            out.append((cur_start + 1, cur[0].removeprefix(marker).strip(), "\n".join(cur)))
            cur, cur_start = [line], i
        else:
            cur.append(line)
    out.append((cur_start + 1, cur[0].removeprefix(marker).strip(), "\n".join(cur)))
    return out


def split_oversized(entries):
    """Chunks longer than MAX_CHARS used to just get truncated at embed time —
    silently, with no indication the tail was never searchable. Splits any
    oversized chunk into MAX_CHARS-sized windows instead, each embedded and
    indexed separately (same file/line pointer — a hit still points at the
    right place to Read), so full coverage wins over precise chunk
    boundaries."""
    out = []
    for e in entries:
        text = e["text"]
        if len(text) <= MAX_CHARS:
            out.append(e)
            continue
        n_parts = math.ceil(len(text) / MAX_CHARS)
        for i in range(n_parts):
            part = dict(e)
            part["text"] = text[i * MAX_CHARS:(i + 1) * MAX_CHARS]
            if i > 0:
                part["header"] = f"{e['header']} (cont. {i + 1}/{n_parts})"
            out.append(part)
    return out


def _display_path(path):
    """Path as printed in results: relative to the current directory when the
    file lives under it (ready to paste into an editor or a Read), else
    absolute."""
    try:
        rel = os.path.relpath(path)
    except ValueError:  # Windows: path on another drive than the cwd
        return path
    return path if rel.startswith("..") else rel


def load_corpus(corpus):
    """Walk corpus/**/*.md, chunked and tagged by kind — the file's top-level
    subdirectory under the corpus, or "root" for files directly in it."""
    entries = []
    files = sorted(glob.glob(os.path.join(corpus, "**", "*.md"), recursive=True))
    for path in files:
        rel = os.path.relpath(path, corpus)
        kind = rel.split(os.sep, 1)[0] if os.sep in rel else "root"
        shown = _display_path(path)
        for line_no, header, chunk in parse_file(path):
            entries.append({"file": shown, "line": line_no, "kind": kind,
                             "header": header, "text": chunk})
    return split_oversized(entries)


def _chunk_store_path(target, model, corpus):
    """One persistent store per (target, model, corpus). Vectors from
    different models/servers live in different spaces, and each build prunes
    its store to its own chunks, so two corpora sharing a store would evict
    each other on every switch."""
    h = hashlib.sha256(f"{target}\n{model}\n{corpus}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"chunks-{h}.json")


def _chunk_key(model, text):
    """Content-addressed cache key: hash of the exact (truncated) text that
    gets embedded, plus the model id. Per-chunk, not per-file/per-corpus —
    editing one line in one decision entry only invalidates that one chunk's
    key, not every other chunk's. Deliberately not (file, line): a chunk that
    moves within an append-log (or gets reworded verbatim elsewhere) should
    still hit cache; a chunk whose actual text changes should not."""
    return hashlib.sha256(f"{model}\x00{text[:MAX_CHARS]}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Embedding (talks to /v1/embeddings)
# ---------------------------------------------------------------------------

def _auth_headers(api_key):
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _post_embeddings(target, model, texts, api_key=None):
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{target}/v1/embeddings", data=body,
        headers={"Content-Type": "application/json", **_auth_headers(api_key)},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def preflight(target, model, api_key=None):
    """One real embed call before doing any work. Turns 'model not found' or
    'connection refused' into an actionable message instead of a stack trace
    mid-corpus-embed. See README.md for the two remediations."""
    try:
        _post_embeddings(target, model, ["preflight check"], api_key)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"semsearch: {target}/v1/embeddings rejected model '{model}' "
              f"(HTTP {e.code}): {body[:300]}", file=sys.stderr)
        print(file=sys.stderr)
        if e.code in (401, 403):
            print(f"  Looks like an auth failure — pass --api-key or set SEMSEARCH_API_KEY", file=sys.stderr)
            print(f"  (or add an entry to {API_KEYS_PATH}) if this server requires one.", file=sys.stderr)
        else:
            print(f"  '{model}' may not be available on this server, or isn't an embedding model.", file=sys.stderr)
            print(f"  Servers don't necessarily register the same models —", file=sys.stderr)
            print(f"  run this to see what's actually on {target}:", file=sys.stderr)
            print(f"    python3 tools/semsearch/semsearch.py --list-models --target {target}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"semsearch: cannot reach {target} ({e}).", file=sys.stderr)
        print(f"  Is RustedVINO running? Try: curl -s {target}/health", file=sys.stderr)
        sys.exit(1)


def _probe_model_kind(target, model_id, api_key=None):
    """Fast, side-effect-free probe: try /v1/embeddings against model_id.
    rustedvino rejects a non-embedding model in ~20ms without loading it, and
    its error body names the model's actual kind ('... is a reranking model'),
    so this reads that back directly rather than guessing from the model id."""
    try:
        _post_embeddings(target, model_id, ["probe"], api_key)
        return True, None
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            detail = json.loads(body).get("error", {}).get("message", body)
        except json.JSONDecodeError:
            detail = body
        return False, detail[:200]
    except urllib.error.URLError as e:
        return False, f"unreachable: {e}"


def _embedding_models(target, api_key=None, verbose=True):
    """Which model(s) on `target` actually answer /v1/embeddings, right now.
    Servers don't all register the same embedding model (or any at all) —
    this replaces guessing."""
    req = urllib.request.Request(f"{target}/v1/models", headers=_auth_headers(api_key))
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"semsearch: {target}/v1/models rejected the request (HTTP {e.code}): {body[:300]}", file=sys.stderr)
        if e.code in (401, 403):
            print(f"  Pass --api-key, or set SEMSEARCH_API_KEY, or add an entry to api_keys.json.", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"semsearch: cannot reach {target}/v1/models ({e})", file=sys.stderr)
        sys.exit(1)

    ids = [m["id"] for m in data.get("data", [])]
    if verbose:
        print(f"semsearch: probing {len(ids)} model(s) on {target} for /v1/embeddings support:")
    embedders = []
    for mid in ids:
        ok, detail = _probe_model_kind(target, mid, api_key)
        if ok:
            embedders.append(mid)
        if verbose:
            print(f"  {mid}  — {'usable for semsearch (embedding)' if ok else detail}")
    return embedders


def list_models(target, api_key=None):
    """--list-models: print every registered model's verdict, then the flag
    to pass."""
    embedders = _embedding_models(target, api_key)
    print()
    if not embedders:
        print(f"semsearch: no embedding-capable model found on {target}.")
    elif len(embedders) == 1:
        print(f"semsearch: --model {embedders[0]} --target {target}")
    else:
        print(f"semsearch: multiple embedding models available on {target} — pick one with --model:")
        for mid in embedders:
            print(f"  --model {mid}")


def resolve_model(cli_model, target, api_key=None):
    """Precedence: --model > SEMSEARCH_MODEL > the only embedding model on
    `target`. No hardcoded default — a model id is a local registration name,
    and a default that exists on one server is a 404 on the next. With zero
    or several candidates, stop and say how to choose rather than guess."""
    if cli_model:
        return cli_model
    env_model = os.environ.get("SEMSEARCH_MODEL")
    if env_model:
        return env_model
    embedders = _embedding_models(target, api_key, verbose=False)
    if len(embedders) == 1:
        return embedders[0]
    if not embedders:
        print(f"semsearch: no embedding model found on {target}. Register one on the server, "
              f"or point --target at a server that has one.", file=sys.stderr)
    else:
        print(f"semsearch: {len(embedders)} embedding models on {target} ({', '.join(embedders)}) — "
              f"pick one with --model or SEMSEARCH_MODEL.", file=sys.stderr)
    sys.exit(1)


def embed_one_with_retry(target, model, text, stats, api_key=None):
    """Embed a single string, halving its length up to 4x if the server's
    length gate rejects it (400 context_length_exceeded)."""
    t = text[:MAX_CHARS]
    for attempt in range(5):
        try:
            d = _post_embeddings(target, model, [t], api_key)
            if attempt:
                stats["retried"] += 1
            return d["data"][0]["embedding"]
        except urllib.error.HTTPError as e:
            if e.code != 400 or attempt == 4:
                raise
            t = t[: len(t) // 2]
    raise RuntimeError("unreachable")


def embed_batch(target, model, texts, stats, api_key=None):
    truncated = [t[:MAX_CHARS] for t in texts]
    try:
        d = _post_embeddings(target, model, truncated, api_key)
        return [row["embedding"] for row in d["data"]]
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        # One oversized item poisoned the whole batch — fall back to
        # per-item embedding with retry/shrink for this batch only.
        # Still fires occasionally even with MAX_CHARS=900 (measured: 9 of
        # 185 batches on a ~3000-chunk corpus) — keep this path, don't
        # "simplify" it away.
        stats["batches_fallen_back"] += 1
        return [embed_one_with_retry(target, model, t, stats, api_key) for t in truncated]


def build_index(target, model, corpus, rebuild=False, quiet=False, api_key=None):
    """Per-chunk incremental: only chunks whose content-hash isn't already in
    the (target, model, corpus) store get embedded over the network. Editing
    one line of one entry re-embeds just that one changed chunk, not the
    whole corpus.

    load_corpus() itself is always run fresh — it's pure local file parsing,
    no network, negligible cost — so file/line/kind/header are always current
    even when a chunk's vector comes from the store."""
    entries = load_corpus(corpus)
    if not entries:
        return entries

    store_path = _chunk_store_path(target, model, corpus)
    store = {}
    if not rebuild and os.path.exists(store_path):
        with open(store_path) as f:
            store = json.load(f)

    to_embed = []
    for e in entries:
        key = _chunk_key(model, e["text"])
        e["_key"] = key
        cached = store.get(key)
        if cached is not None:
            e["vector"] = cached
        else:
            to_embed.append(e)

    if to_embed:
        stats = {"batches_fallen_back": 0, "retried": 0}
        t0 = time.time()
        for i in range(0, len(to_embed), BATCH_SIZE):
            batch = to_embed[i:i + BATCH_SIZE]
            embs = embed_batch(target, model, [e["text"] for e in batch], stats, api_key)
            for e, v in zip(batch, embs):
                e["vector"] = v
        elapsed = time.time() - t0
        if not quiet:
            reused = len(entries) - len(to_embed)
            print(f"semsearch: {len(to_embed)}/{len(entries)} chunk(s) newly embedded in "
                  f"{elapsed:.1f}s ({reused} reused from cache, "
                  f"{stats['batches_fallen_back']} batch fallbacks, {stats['retried']} item retries)",
                  file=sys.stderr)
    elif not quiet:
        print(f"semsearch: all {len(entries)} chunk(s) reused from cache, nothing to embed", file=sys.stderr)

    # Persist, pruned to exactly this build's chunks — bounds store size to
    # corpus size and drops vectors for content that no longer exists
    # (edited or removed since the last build).
    os.makedirs(CACHE_DIR, exist_ok=True)
    new_store = {e["_key"]: e["vector"] for e in entries}
    with open(store_path, "w") as f:
        json.dump(new_store, f)
    for e in entries:
        del e["_key"]

    # keep the 3 most-recently-used (target, model, corpus) chunk stores
    # (bounds disk use without thrashing when a user alternates
    # --model/--target/--corpus — each
    # combination gets its own store, so a naive "delete every other file"
    # prune would force a full re-embed on every switch back)
    all_stores = sorted(glob.glob(os.path.join(CACHE_DIR, "chunks-*.json")),
                         key=os.path.getmtime, reverse=True)
    for stale in all_stores[3:]:
        os.remove(stale)
    return entries


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def search_py(entries, qvec, k):
    scored = [(cosine(qvec, e["vector"]), e) for e in entries]
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:k]


def search_np(entries, qvec, k):
    mat = np.array([e["vector"] for e in entries], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_normed = mat / norms
    q = np.array(qvec, dtype=np.float32)
    qn = q / (np.linalg.norm(q) or 1.0)
    scores = mat_normed @ qn
    k = min(k, len(entries))
    idx = np.argpartition(-scores, k - 1)[:k] if k < len(entries) else np.arange(len(entries))
    idx = idx[np.argsort(-scores[idx])]
    return [(float(scores[i]), entries[i]) for i in idx]


def search(entries, qvec, k):
    return search_np(entries, qvec, k) if np is not None else search_py(entries, qvec, k)


# ---------------------------------------------------------------------------
# Reranking (talks to /v1/rerank — Cohere-compatible, see
# src/handlers/reranking.rs). Second-stage refinement: embedding search casts
# a wide net (cheap bi-encoder, one vector per chunk, precomputed) and the
# reranker — a cross-encoder that sees the query and each candidate together
# — re-scores just the top handful. Bi-encoder cosine similarity is a weak
# signal for "which of these five near-identical entries actually answers
# the question"; the cross-encoder is much better at exactly that,
# but it's too expensive to run against the whole corpus, hence two stages.
# ---------------------------------------------------------------------------

def _post_rerank(target, model, query, documents, top_n, api_key=None):
    body = json.dumps({"model": model, "query": query, "documents": documents,
                        "top_n": top_n}).encode()
    req = urllib.request.Request(
        f"{target}/v1/rerank", data=body,
        headers={"Content-Type": "application/json", **_auth_headers(api_key)},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def rerank_hits(target, model, query, candidates, k, api_key=None):
    """candidates: list of (cosine_score, entry) from the embedding stage,
    already the wider top-N pull. Returns the same shape, re-scored and
    re-ordered by the cross-encoder's relevance_score, trimmed to k — or None
    if reranking couldn't happen (see caller: --rerank degrades to
    embedding-only ranking rather than failing the whole search, since a
    reranking model is a much less certain fixture on a server than an
    embedding model — many won't have one loaded)."""
    documents = [e["text"] for _, e in candidates]
    try:
        d = _post_rerank(target, model, query, documents, top_n=k, api_key=api_key)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        reason = f"HTTP {e.code}: {body[:200]}"
        if e.code in (401, 403):
            hint = "check --api-key / SEMSEARCH_API_KEY / api_keys.json"
        else:
            hint = f"run --list-models to confirm '{model}' is registered on {target} as a reranking model"
    except urllib.error.URLError as e:
        reason = f"unreachable: {e}"
        hint = f"is {target} reachable? try: curl -s {target}/health"
    else:
        return [(r["relevance_score"], candidates[r["index"]][1]) for r in d["results"]]

    print(f"semsearch: WARNING: reranking requested (--rerank) but reranker "
          f"'{model}' on {target} is unavailable ({reason}) — {hint}.", file=sys.stderr)
    print(f"semsearch: WARNING: falling back to embedding-only (cosine similarity) ranking. "
          f"This can reduce retrieval quality — a cross-encoder reranker distinguishes "
          f"near-duplicate or terse entries that cosine similarity alone conflates; treat "
          f"these results with correspondingly less confidence.", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Query cache / trace log
#
# One append-only file does both jobs: reusing an identical query's vector
# (skip the network call) and leaving a session-investigable trail of what
# was searched, when, and what came back — including the actual embedding,
# not just the query text, so it can be inspected later without the
# embedding server up. Gitignored (tools/semsearch/.cache/), same as the
# corpus index — it holds every query you ran, so it stays on your machine.
# ---------------------------------------------------------------------------

QUERY_LOG_PATH = os.path.join(CACHE_DIR, "queries.jsonl")


def _query_cache_lookup(target, model, query):
    """Most recent matching (target, model, query) entry's vector, or None."""
    if not os.path.exists(QUERY_LOG_PATH):
        return None
    found = None
    with open(QUERY_LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("target") == target and rec.get("model") == model
                    and rec.get("query") == query and "vector" in rec):
                found = rec["vector"]
    return found


def _log_query(target, model, query, k, vector, vector_cached, hits, reranked):
    os.makedirs(CACHE_DIR, exist_ok=True)
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": target,
        "model": model,
        "query": query,
        "k": k,
        "vector_cached": vector_cached,
        "vector": vector,
        "reranked": reranked,
        "hits": [
            {"score": round(score, 4), "file": e["file"], "line": e["line"],
             "kind": e["kind"], "header": e["header"]}
            for score, e in hits
        ],
    }
    with open(QUERY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Semantic search over a directory of Markdown notes via RustedVINO's /v1/embeddings (and, optionally, /v1/rerank).")
    ap.add_argument("query", nargs="?", help="natural-language query (not needed with --list-models)")
    ap.add_argument("--target", default=DEFAULT_TARGET, help=f"server base URL (default {DEFAULT_TARGET})")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                     help="directory whose *.md files are searched, recursively (default: this repo's dev/)")
    ap.add_argument("--model", default=None,
                     help="embedding model id. Falls back to SEMSEARCH_MODEL, then to the only "
                          "embedding model on --target (see --list-models)")
    ap.add_argument("--api-key", default=None,
                     help="bearer token for --target, if it requires auth. Falls back to "
                          "SEMSEARCH_API_KEY env var, then api_keys.json (see api_keys.example.json)")
    ap.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    ap.add_argument("--rebuild", action="store_true", help="force re-embedding, ignoring both the corpus index cache and this query's cached vector")
    ap.add_argument("--snippet", action="store_true", help="show a text preview per result")
    ap.add_argument("--quiet", action="store_true", help="suppress cache/build status on stderr")
    ap.add_argument("--list-models", action="store_true",
                     help="probe --target for which registered model(s) answer /v1/embeddings, then exit")
    ap.add_argument("--rerank", action="store_true",
                     help=f"second-stage rerank via /v1/rerank (cross-encoder) — pulls --candidates "
                          f"results from embedding search, re-scores with --rerank-model, returns top -k")
    ap.add_argument("--rerank-model", default=os.environ.get("SEMSEARCH_RERANK_MODEL"),
                     help="reranking model id (default: SEMSEARCH_RERANK_MODEL). Required for --rerank")
    ap.add_argument("--candidates", type=int, default=20,
                     help="how many embedding-stage hits to pass into --rerank before trimming to -k (default 20)")
    args = ap.parse_args()

    api_key = resolve_api_key(args.api_key, args.target)

    if args.list_models:
        list_models(args.target, api_key)
        return

    if not args.query:
        ap.error("query is required unless --list-models is given")
    if args.rerank and not args.rerank_model:
        ap.error("--rerank needs a reranking model: pass --rerank-model or set SEMSEARCH_RERANK_MODEL")
    corpus = os.path.abspath(args.corpus)
    if not os.path.isdir(corpus):
        print(f"semsearch: corpus directory {corpus} does not exist — pass --corpus DIR", file=sys.stderr)
        sys.exit(1)

    model = resolve_model(args.model, args.target, api_key)
    preflight(args.target, model, api_key)
    entries = build_index(args.target, model, corpus, rebuild=args.rebuild, quiet=args.quiet, api_key=api_key)
    if not entries:
        print(f"semsearch: no Markdown chunks found under {corpus} — nothing to search", file=sys.stderr)
        sys.exit(1)

    vector_cached = False
    qvec = None if args.rebuild else _query_cache_lookup(args.target, model, args.query)
    if qvec is not None:
        vector_cached = True
        if not args.quiet:
            print("semsearch: query embedding cache hit (queries.jsonl)", file=sys.stderr)
    else:
        d = _post_embeddings(args.target, model, [args.query[:MAX_CHARS]], api_key)
        qvec = d["data"][0]["embedding"]

    used_rerank = False
    if args.rerank:
        candidates = search(entries, qvec, max(args.candidates, args.k))
        reranked = rerank_hits(args.target, args.rerank_model, args.query, candidates, args.k, api_key)
        if reranked is not None:
            hits = reranked
            used_rerank = True
        else:
            hits = candidates[:args.k]
    else:
        hits = search(entries, qvec, args.k)

    if not args.quiet:
        print(f"semsearch: ranking method: {'rerank (' + args.rerank_model + ')' if used_rerank else 'embedding cosine similarity only'}", file=sys.stderr)

    for score, e in hits:
        print(f"{score:.3f}  {e['file']}:{e['line']}  [{e['kind']}]  {e['header']}")
        if args.snippet:
            body = e["text"].split("\n", 1)[-1].strip()
            preview = body[:200].replace("\n", " ")
            print(f"         {preview}{'...' if len(body) > 200 else ''}")

    _log_query(args.target, model, args.query, args.k, qvec, vector_cached, hits, used_rerank)


if __name__ == "__main__":
    main()
