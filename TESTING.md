# Testing — RustedVINO 0.9.1

How the 0.9.1 release was tested on real hardware before it shipped, what was measured, and what was not
covered. The release notes summarise the same runs in "How it was tested"; this page has the detail.

The release went through three release candidates. rc.1 and rc.2 ran a scripted test kit, one per machine,
each with blocks chosen for that machine's hardware. rc.2 differs from rc.1 by a debug log line that says why
a model's memory was not measured, and by shipping METRICS.md in the bundle. rc.3 differs from rc.2 only by
the Qwen tool-call typing fix (booleans and `None` typed from the request's tool schema); it was checked on two
machines with a targeted tool-call test and with the test-harness scenarios. The released server is rc.3's code.
The bundle itself is rebuilt from the public tree at publish, and its `rv` CLI carries later changes (see
CHANGELOG: model tables ordered by kind, refusals that name the gate), covered by the 232 `rv` tests and a
smoke run of that bundle.

## Machines

| Machine | Hardware | Build | What ran |
|---|---|---|---|
| B60 | Intel Arc Pro B60 24 GB, 48 GB RAM | rc.1 | test kit, plus opencode |
| B50 | Intel Arc Pro B50 16 GB | rc.2 | test kit, plus eviction under waiting requests |
| B70 | Intel Arc Pro B70 32 GB, 62 GB RAM, plus an Intel integrated GPU | rc.2, rc.3 | test kit, extended tests X1–X8, rc.3 tool-call check |
| CPU laptop | i7-11800H, 15.4 GB RAM, Intel UHD integrated graphics + NVIDIA T1200 | rc.2 | test kit: CPU, placement with a non-Intel GPU, a home-assistant mix |
| 140V laptop | Intel Arc 140V (Lunar Lake, integrated GPU) | rc.3 | rc.3 tool-call check, test-harness scenarios |

Every machine ran a release-candidate bundle (the Linux tarball, checksum verified), not a source build.

## Kit blocks and verdicts

PASS means every check in the block held. "PASS (rerun)" means the block as first written could not show the
behaviour (for example, a model loaded before the client hung up) and passed when rerun with corrected timing;
no server change was made for any rerun. "—" means the block was not part of that machine's kit.

| Block | What it checks | B60 | B50 | B70 | CPU laptop |
|---|---|---|---|---|---|
| P0 | bundle checksum, version, devices, existing configs checked against the new id rules | PASS | PASS | PASS | PASS |
| C1 | `--help` exits 0; an unknown argument exits 64 with the usage on stderr | PASS | PASS | PASS | PASS |
| C2 | a non-plain model id and invalid `on_demand_wait_secs` values stop startup with exit 78 | PASS | PASS | PASS | PASS |
| C3 | the server starts with 0.9.1 and no startup warnings | PASS | PASS | PASS | PASS |
| TH1 | text thinking budget: exact stop, both aliases, `reasoning_effort`, the 400 errors | PASS | PASS | PASS | PASS |
| TH2 | vision thinking budget: stop, no placeholder text, tokens per chunk | PASS | PASS | PASS | PASS |
| TH3 | gpt-oss tool call under a budget: the call stays whole | PASS | PASS | PASS | — (no gpt-oss) |
| TH4 | opencode with `reasoning_effort` (also on gpt-oss) | PASS | — | — | — |
| OD1 | an on-demand model is listed `not_loaded` before its first load | PASS | PASS | PASS | PASS |
| OD2 | `/tokenize` never starts a load | PASS | PASS | PASS | PASS |
| OD3 | a load with waiting requests, streamed and not | PASS | PASS | PASS | PASS |
| OD4 | a client that hangs up while waiting does not cancel the load | PASS (rerun) | PASS (rerun) | PASS | PASS (rerun) |
| OD5 | the wait runs out: `503` with `Retry-After`, a stream error frame, the load goes on | — | — | PASS | — |
| ODC | `/v1/completions` starts an on-demand load | PASS | PASS | PASS | PASS |
| EV1–EV3 | three waiters, one load, an idle model evicted; a busy model never evicted | — | PASS | — | — (see below) |
| MX | new counters, `rv metrics`, measured memory in the admin model list | PASS (rerun) | PASS | PASS | PASS |
| MXI | measured memory of a model on an Intel integrated GPU | — | — | — | PASS |
| FU1 | forced unload: the running request finishes during the drain | PASS | PASS | PASS | PASS |
| FU2 | forced unload: requests still running at the deadline end with a `model_unloading` frame | PASS | PASS | PASS | PASS |
| FU3 | forced operations answer a JSON report when asked; a forced resize | PASS | PASS | PASS | PASS |
| AD | admin routes answer a bad JSON body with the `400 invalid_body` envelope; 422/415 unchanged | PASS | PASS | PASS | PASS |
| PL0–PL2 | device tiers; every model on the CPU by default; a bare `GPU` refused | — | — | — | PASS |
| PL3 | naming the NVIDIA card: a clean, bounded error, the CPU keeps serving | — | — | — | PASS |
| PL4 | SpeechT5 on the CPU | — | — | — | PASS |
| PL5 | a vision model on the integrated GPU with a load deadline | — | — | — | PASS |
| HA1–HA3 | speech out, speech in, embedding, rerank and a small tool call, then all five together | — | — | — | PASS (HA3 recorded as figures) |

Notes on the reruns:

- **OD4.** Loads on these machines were faster than the kit's client timeout, so the client got its answer
  before it could hang up. Rerun with a shorter client timeout (5 s on the B60, 3 s on the B50, 1 s on the
  CPU laptop): the client hung up, the load finished, and the next request was served.
- **MX on the B60.** The kit loaded the model 4.4 s after an eviction, inside the 20-second window in which
  the server deliberately skips measuring a model's memory (see Known issues in the release notes). Rerun with
  a 32-second gap: every check passed. The B50 and B70 kits waited 25 s and passed as written.
- **TH4 on the B60** first had no opencode installed; it passed once opencode was installed.
- **EV1 on the B50.** The kit's pre-check ran inside the 2-second eviction grace and reported that the model
  would not fit; repeated 5 s later it reported the expected eviction, and the block passed.
- **EV on the CPU laptop** was not run: the kit ran under a 10 GiB memory cap that the server cannot see, so
  the cap, not the server's eviction, would have acted first.

## Measured numbers

Thinking budget (TH1–TH3, X6):

- Text models stopped at exactly the budget: 256 tokens, and 1,024 for `reasoning_effort: low` (B60, B50,
  CPU laptop; exact on the B70 too). gpt-oss stopped at exactly 128 tokens on its analysis channel; with a budget of 48 the
  analysis fitted (36 tokens) and the tool call was complete (B60, B50).
- Vision stops on qwen3.5-4b-int4-ov, reasoning tokens at a budget of 256 / 512 / 256 with an image:
  B60 201 / 388 / 193, B50 205 / 406 / 193, CPU laptop 205 / 404 / 205. That is 0.75–0.80 × the budget,
  early and never over. Tokens per streamed reasoning chunk: 1.04–1.07 (B60), 1.04–1.09 (B50), 1.08–1.09
  (CPU laptop), against the documented 1.37.
- On the B70, vision stops landed at about 0.78 × the budget at every budget from 64 to 2,048 (X6).

On-demand loads (OD1, OD3, ODC, EV):

- Load time of the on-demand model: 6.7 s for qwen3-14b on the B60 and 7.2 s on the B50; 8.3 s for
  qwen3-4b on the CPU laptop (later loads there about 1.5 s); 13–16 s for a 27B vision model on the B70.
  The first load of qwen3-14b after `rv add` took 55.0 s on the B60 and 31.3 s on the B50.
- The waiting requests were answered when the load ended: waits of 6.6–6.7 s (B60), 7.1 s (B50) and 1.5 s
  (CPU laptop); `/v1/completions` waited 13.9 s, 16.2 s and 1.2 s.
- Eviction under waiting requests (B50): three waiters were served after one eviction (teardown 148–219 ms)
  and one load (4.6–4.9 s). A busy model was not evicted: a request that needed its room got
  `503 model_busy` with `Retry-After: 5`, the busy model's stream finished normally (1,500 tokens), and a
  later request was served in 5.15 s once the model was idle and could be evicted.
- Keep-alive comments: none of the kit loads lasted 15 s, so none were sent. A keep-alive was seen at
  15.028 s during the load of qwen3.6-35b-a3b on the B70 (X1).

Speed and memory (MX, MXI, PL, HA):

| Machine | Model, device | Decode | Effective prefill (p50) | Measured memory |
|---|---|---|---|---|
| B60 | qwen3-8b-int4-ov, B60 | 65.3 tok/s | ~2,500 tok/s | weights 4,669 MiB + KV pool 2,047 MiB |
| B50 | qwen3-8b-int4-ov, B50 | 37.4 tok/s | ~1,750 tok/s | weights 4,669 MiB + KV pool 2,047 MiB |
| B50 | qwen3-14b-int4-ov, B50 | — | — | weights 9,332 MiB + KV pool 2,047 MiB |
| CPU laptop | qwen3-1.7b-int4-ov, CPU | 25.8 tok/s | ~275 tok/s | not measured on the CPU |
| CPU laptop | qwen3-1.7b-int4-ov, UHD integrated GPU | 11.9 tok/s | — | weights 1,448 MiB + KV pool 2,047 MiB |
| CPU laptop | qwen3.5-4b-int4-ov (vision), CPU | 10.3–11.0 tok/s | ~75 tok/s | — |
| CPU laptop | qwen3.5-4b-int4-ov (vision), UHD integrated GPU | 5.4 tok/s | — | weights 3,391 MiB + KV pool 2,033 MiB |

On the CPU laptop, every model with no `device` loaded on the CPU, in both placement modes. Naming the NVIDIA
card failed within 5 s with a clean `500` (`CL_BUILD_PROGRAM_FAILURE`), and the server stayed healthy and kept
serving its CPU models. The vision model loaded on the integrated GPU in 28.5 s. SpeechT5 spoke a 2.6-second
sentence in 1.1 s (real-time factor 0.43–0.44); Whisper turbo transcribed it in 2.0–2.9 s; embedding and rerank
answered in 0.04–0.7 s; the 1.7B model made a correct `get_weather` tool call in 1–4 s. All five
home-assistant models (speech, transcription, embedding, reranking, the 1.7B tool caller) stayed loaded
together; the server's own memory grew 1.77 → 2.17 → 2.57 → 3.60 → 4.32 GiB as they were added, and was
4.42 GiB after one request to each of the five.

Forced unloads (FU): a request that finished during the drain was reported as such (after 30.1 s on the B60,
19.6 s on the B50, 15.6 s on the CPU laptop). A request still running at the 120-second deadline got a
`model_unloading` error frame and `[DONE]`; with three such streams, three were reported aborted and the
counter rose by three. The JSON report's teardown took 38–184 ms.

## Extended tests on the B70 (rc.2)

Beyond the kit, eight further tests ran on the B70 machine:

- **X1:** a keep-alive comment was sent during a 15-second load of qwen3.6-35b-a3b.
- **X2:** a forced unload of a vision model (qwen3.6-27b) during an 8,192-token stream did not stop the
  generation: as documented for non-text engines, the engine finishes its request first. The `rv unload
  --force` call returned `503` after its 150-second wait, the stream ran on to its normal end, and the model's
  memory was released about 95 s after the drain deadline. The aborted-requests counter counted that stream
  although it completed.
- **X3:** under memory pressure, a `503 insufficient_memory` sent while the model to evict was still inside
  its eviction grace carried no `Retry-After` header (a `503` for a loading model does).
- **X4:** one request with 7 large photos to qwen3.6-27b, each reduced to the 4 MP `max_image_pixels` and
  all within the per-request limits, grew the server process to 44 GB on the 62 GB machine, and the kernel
  killed it. 4 such photos worked (200 after 135.7 s); 8 were refused with `400 image_too_large`. The memory
  grows inside the vision engine, after the server's own decoding and downscaling. This is the Known issue in
  the release notes; `config.example.json` sets `max_images_per_request` to 4 for this reason.
- **X5:** the other modalities on the B70: embeddings (multilingual-e5-large, 2 × 1,024 dimensions) and rerank
  (ms-marco-MiniLM, the right document first) worked, and the speech-to-text route answered `200` (whisper-base,
  on a test tone; not a transcription-quality test). Text-to-speech was not tested there: the only speech model on
  that machine, Kokoro, was not in a loadable form.
- **X6:** the vision thinking-budget stops at about 0.78 × the budget, above.
- **X7:** the integrated GPU and the B70 served at the same time; the text thinking budget stayed exact.
- **X8:** a ten-cycle load/unload soak showed no memory leak.

## rc.3: the tool-call fix

rc.3 differs from rc.2 only by the Qwen tool-call typing fix. The test schema had a string `city`, a boolean
`metric`, an integer `days`, a string `zip` and a `note` that is a string or null.

- **140V laptop, qwen3.5-4b-int8-ov, thinking off:** streamed and not, the arguments arrived typed, e.g.
  `{"city":"Boston","metric":false,"days":2,"zip":"02108","note":null}`. The same request to a build from
  before the fix returned `"metric":"False"` and `"note":"None"`.
- **B70, qwen3.6-35b-a3b, temperature 0:** thinking off and on, streamed and not: `false`, `null` and the
  integer `2` typed, `"02108"` kept as a string. Text streaming and a vision request were answered correctly.
- **Test harness, 140V laptop:** the thinking-budget and on-demand scenario passed 9 of 9 steps, and a
  tool-call scenario passed. A new scenario asserts the argument types: it passes on rc.3 and fails on a
  build from before the fix (`tool call argument "metric": expected false, got "False"`).

## Test counts

What the numbers in the release commit ("automated tests: 2,415") consist of, and what ran besides them:

- **Automated tests in this repository** (run on every build with `cargo test --workspace`; anyone can rerun them
  from the source): the server 2,148 passed, 0 failed, 16 ignored (the ignored ones need model files or a GPU);
  the `rv` admin CLI 232 passed (179 unit, 53 command-level). Together **2,380**.
- **The test harness's own tests:** 35 passed. The harness drives a live server through scripted scenarios; it is
  not part of the public source, so these 35 are counted in the 2,415 but cannot be rerun from this repository.
- **Live scenarios against a running server** (the harness): run on the 140V laptop on rc.3 — the thinking-budget
  and on-demand scenario (9 of 9 steps) and the tool-call scenarios, including the new argument-type check.
- **Hardware test kit:** 26 kit blocks on four machines (B60, B50, B70, CPU laptop) across rc.1 and rc.2, the eight
  extended tests X1–X8 on the B70, and the rc.3 tool-call check on two machines (tables above).
- **After rc.3, on the released server build (140V laptop):** an 8-hour overnight run with qwen3.6-35b-a3b, 810
  requests of every kind (text up to ~32k-token prompts, images, tools, aborted streams, refused concurrent
  requests), 0 failures and 0 restarts; and about 4,000 generated requests (property-based testing) on seven models
  of six chat-template families, which found no crash and the request-shape issues listed in the release notes'
  Known issues.

Counted on 2026-11-22 at the release commit's code.

## Not covered

Not yet verified on hardware (each was fixed or added in code and covered by tests):

- the thinking budget and the `reasoning_tokens` recount on NPU models, and the tokens-per-chunk ratio on
  vision models other than qwen3.5-4b;
- a wait behind a proxy across minutes of streaming keep-alives (one keep-alive was seen, X1);
- the OpenAI Python SDK recipe against a real client (the opencode recipe was, TH4; the Open WebUI recipe was
  used with Open WebUI 0.13.4 on rc.3);
- on an Intel integrated GPU (shared memory), the KV-pool credit an unload and `rv resize` now use;
- the new `language` check on `/v1/audio/transcriptions` with Whisper v1/v2 and English-only (`.en`) models.

Also not exercised in these runs:

- a truly cold load (after a reboot, with empty caches): on the B60, the B50 and the CPU laptop the loads
  were warmed by the OS page cache and, on the GPUs, the GPU compiler's kernel cache;
- the `504` load-deadline path on an integrated GPU: the vision model loaded in 28.5 s, inside the 60-second
  deadline, so the deadline never fired;
- placement falling through to a later device when system memory is full, and eviction on a CPU-only machine;
- Windows: this release has no Windows bundle.

Carried over from 0.9.0, not re-tested: `rv deregister` / `rv save-defaults` without a terminal need `-y`;
gemma-4-e4b-it-int8-ov does not load as a vision model; unloading the reranker can crash on some CPU setups.
