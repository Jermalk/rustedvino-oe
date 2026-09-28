# semsearch

Semantic search over a directory of Markdown notes, using a RustedVINO server's own
`/v1/embeddings` endpoint and, optionally, its `/v1/rerank`. Point it at design notes,
decision logs or a session history, and ask in plain language. It suits humans and coding
agents alike: a reader arriving cold can find the right earlier note without grepping
hundreds of files or guessing the words it was written in.

```
python3 tools/semsearch/semsearch.py "a race between two admin calls hitting the same model" --corpus docs/
```

Output is one line per hit: score, `file:line`, kind (the file's top-level subdirectory under
the corpus, or `root`), and the header. The next step is opening the file at that line, not
reading the whole file:

```
0.751  docs/decisions/2026-08.md:31  [decisions]  2026-08-03 — serialise admin load/unload per model
0.724  docs/api/admin.md:110  [api]  Concurrent admin requests
```

Add `--snippet` for a short text preview per hit and `-k N` to change the number of results
(default 5). Use `--rebuild` to force re-embedding if you suspect the cache is stale.

`--corpus` defaults to `dev/` in the repository root. Pass any directory; every `*.md` under
it is searched, recursively. The corpus is always read from your machine. A remote `--target`
only lends its model, never its files.

## Which model

Model ids are local registration names, so there is no built-in default. semsearch picks the
embedding model in this order:

1. `--model ID`
2. the `SEMSEARCH_MODEL` environment variable
3. the only embedding model registered on `--target`, if there is exactly one

With none, or with several to choose from, it stops and says so. To see what a server offers:

```
python3 tools/semsearch/semsearch.py --list-models --target http://localhost:11437
```

This probes every registered model against `/v1/embeddings` and prints the flag to pass.
`--target` defaults to `http://localhost:11437`, RustedVINO's default listen address.

If the target requires auth, pass `--api-key`, set `SEMSEARCH_API_KEY`, or add an entry to
`tools/semsearch/api_keys.json` (gitignored; copy `api_keys.example.json`). Precedence is
flag, then environment variable, then key file.

## Why it's cheap

Keyword grep misses paraphrases. A query about "GPU ran out of memory" won't match a note that
says "Shmem" and "OOM-killed" and never says "memory" at all. Semantic search does, because it
matches meaning, not strings.

On one ~3,000-chunk corpus of dense, backtick-heavy notes, the first run with a small embedding
model embedded everything in about 28 seconds and cached ~17 MB of vectors to disk. A larger
model takes proportionally longer; it's a one-time cost. After that, a query takes under a
second, most of it request latency. Only chunks whose text changed are re-embedded. It's cheap
enough to use wherever you'd otherwise grep.

## Chunking

Each file is split at the level of its first sub-header:

- **`## ` first:** one chunk per `## ` section. Text before the first section becomes its own
  chunk if it's substantial.
- **`### ` first:** the file is read as an append-log (decision records, session histories),
  one chunk per `### ` entry. The preamble, usually a banner or navigation, is skipped.
- **Neither:** the whole file is one chunk.

Chunks longer than ~900 characters are split into windows that share the same `file:line`, so
the tail of a long section stays searchable.

## numpy: optional, not required

If numpy is importable, search runs as one matrix-vector product; otherwise it falls back to a
pure Python loop. Both rank identically, and pure Python stays well under a second per query
at a few thousand chunks. Don't install numpy into a system Python just for this.

## Why Python, not Rust

RustedVINO is Rust, but semsearch is a lookup aid that should run cold with no build step. A
Rust version would add a 40-second compile and a lint gate before first use, for a tool whose
whole job is "call an HTTP endpoint, chunk some Markdown, compute a dot product." If it ever
grows into something long-running or performance-critical, revisit that deliberately.

## Reranking (optional)

`--rerank` adds a second stage. Embedding search first pulls a wider set of `--candidates` hits
(default 20). A cross-encoder reranking model then scores the query against each candidate's
full text and returns the top `-k`. A cross-encoder reads the query and the document together,
so it separates near-duplicate or terse entries that cosine similarity lumps together. It's too
expensive to run against the whole corpus, hence two stages.

```
python3 tools/semsearch/semsearch.py "query" --rerank --rerank-model <reranker-id>
SEMSEARCH_RERANK_MODEL=<reranker-id> python3 tools/semsearch/semsearch.py "query" --rerank --candidates 30
```

**A reranker failure never fails the search.** If the reranking model isn't available on
`--target` (wrong id, not registered, auth failure, unreachable), semsearch prints a `WARNING`
to stderr saying why and falls back to embedding-only ranking. If you call it from a script or
an agent, watch stderr for that warning: the ranking you got is weaker than the one you asked
for.

## Cache

Both caches live under `tools/semsearch/.cache/`, which is gitignored. Don't commit it and don't
copy it between machines.

- **Corpus index** (`chunks-<hash>.json`, one store per target, model and corpus): each chunk
  is keyed by a hash of the model and its exact text. Editing one line of one entry re-embeds
  that one chunk. The three most recently used stores are kept, so switching models, servers or
  corpora doesn't force a full re-embed.
- **Query log** (`queries.jsonl`, append-only): reuses the embedding of a query you've run
  before, and records what was searched, when, which ranking was actually used and what came
  back. It stores each query's text and vector. That's why it stays local and out of git.

## Scope

Markdown only, not source code. Code chunks poorly against a BERT-class model's 512-token limit,
and grep already finds code symbols better than embedding similarity would.
