# cmpbench - compare RustedVINO with OVMS and OpenArc on your own hardware

`cmpbench.py` runs the same chat workload against one server at a time, on one machine, and records speed, first-token time,
parallel-stream throughput, host memory and GPU memory in the same format for every server. `cmptable.py` puts the results of
several runs into one table. Both are stdlib-only Python and reuse the prompts and measurement code of `rvbench.py` unchanged.

It exists so that you do not have to take our numbers on trust: run it on your card, with your models, and read the versions line
at the top of every result before you read a number.

**What it measures:** single-stream decode speed, time to first token, prompt processing on one ~700-token prompt, 2 and 4 parallel
streams, a fixed 256-token decode, load time with an empty and with a warm compile cache, host RSS and GPU memory (VRAM).
**What it does not measure:** embeddings, speech, vision, image generation, overload behaviour, shutdown behaviour, long soak,
multi-turn prefix reuse, models other than text chat, any GPU other than Intel's `xe` driver for the VRAM column.

## 1. Requirements

- Linux, Python 3 (stdlib only; tested with 3.12), an Intel GPU under OpenVINO. The VRAM column reads `/proc/<pid>/fdinfo`
  (`drm-resident-vram*`), which the `xe` driver provides for **discrete** cards. On an integrated GPU (shared system memory) and on other drivers the VRAM column reads 0; the model then lives in system memory that host RSS does not show either, so memory cannot be compared on such a machine with this tool.
- A **quiet machine**: the tool measures CPU use over 5 s and refuses to start when it is above 15 % (`--max-cpu-pct`; the limit
  and the measured value are stored in the result; the load average is stored but not used, because threads stuck in I/O wait inflate it on an idle machine). It also warns if a GPU already holds more than 0.2 GiB.
  Close other GPU programs (an LLM service left running on the same card invalidates the run). AC power, and a fixed CPU power
  profile that you log yourself for the whole session (for example `performance`); the tool records the CPU governor, the energy-performance preference, the power profile (if `powerprofilesctl` exists) and whether AC is online (`ac_online`). On a laptop, check `ac_online` is `1`: on battery the same machine ran 20-25 % slower in our test, for every stack.
- The Intel GPU compute runtime (Level Zero / OpenCL "compute-runtime") installed on the host: all bare-metal stacks use it.
- Disk: one folder with the models, one scratch folder for compile caches (a few GB per stack and model).
- Free ports on 127.0.0.1: 11598 (RustedVINO), 11436 and 11435 (OVMS), 8000 (OpenArc). The tool stops if one answers.

## 2. Install the three servers (the versions we used)

You can use other versions; the tool prints what it finds. If the OpenVINO GenAI versions differ between servers, any gap you
see is the server **and** the engine under it, not the server alone.

| | what we used | where it comes from | check |
|---|---|---|---|
| RustedVINO | 0.8.4 (`f2532cacfe36`), bundles OpenVINO / GenAI 2026.2.1 | https://github.com/Jermalk/rustedvino-oe/releases/tag/v0.8.4, file `rustedvino-linux-x86_64-v0.8.4.tar.gz` | sha256 `6068295e14e4cedfbb6a00b6c7c8fa6bb0b5a3177d3a19e3959fb1f04d389864` |
| OVMS | 2026.4.1 (`cfc9ac721`), bundles OpenVINO / GenAI 2026.4.1; bare-metal package that uses your machine's own GPU user-mode driver | https://github.com/openvinotoolkit/model_server/releases/tag/v2026.4.1, asset `ovms_ubuntu24_2026.4.1_python_off.tar.gz` | sha256 `2b7198671a9e99e030b408601733fbd61798d8a741337c867b96eb41c295b3f8` (matches the published `.sha256`) |
| OpenArc | commit `9adbb88884e344c7778e4ee124628898e8565834`, installed with `uv sync` from its own lockfile (`uv` 0.11.29, Python 3.12) | https://github.com/SearchSavior/OpenArc | `git rev-parse HEAD` |

```
# RustedVINO: unpack the bundle (folder contains ./rustedvino and config.example.json)
tar xzf rustedvino-linux-x86_64-v0.8.4.tar.gz

# OVMS: unpack the package; the folder that contains bin/ovms and lib/ is what you configure as "ovms_dir"
tar xzf ovms_ubuntu24_2026.4.1_python_off.tar.gz

# OpenArc: clone, pin, create its own virtual environment
git clone https://github.com/SearchSavior/OpenArc openarc
git -C openarc checkout 9adbb88884e344c7778e4ee124628898e8565834
cd openarc && uv sync        # creates openarc/.venv
```

`uv sync` installs the versions pinned in OpenArc's lockfile (on our machine OpenVINO / GenAI 2026.3.1); a pinned nightly wheel can
disappear from the package index later, so a fresh install may differ. Whatever is installed, the tool prints it. Install `uv`
first (https://docs.astral.sh/uv/). If OpenArc needs an API key in your setup (`OPENARC_API_KEY`), unset it for the run: the tool sends none. OVMS can also run from its docker image (set `ovms_docker` and `ovms_render`); the image
ships its **own** GPU user-mode driver, which is one more difference between the stacks, and the tool says so in the versions line.

## 3. Prepare the models

All three servers must read **byte-identical** model folders. Use a **dedicated copy**, not your normal library:

- OpenArc writes a `model_cache/` folder (potentially many GB) into the model folder it loads; RustedVINO up to 0.8.x
  may write a small `.rustedvino_kv_ratchet.json` there (0.9.0+ writes it to `~/.cache/rustedvino/kv_ratchet/`, or beside
  `ov_cache_dir`, never into the model folder). Delete them after each run (before an OpenArc cold load the tool wipes `<model>/model_cache` and prints that it did).
- Compare `sha256sum <model>/openvino_model.bin` across copies if in any doubt. Folder name = the `--model` argument, for
  example `qwen3-1.7b-int4-ov` and `qwen3-8b-int4-ov` (OpenVINO int4 exports of Qwen3 1.7B and 8B). Use any
  OpenVINO export you can obtain and record its source, revision and the `sha256sum` of its weights next to your results; the tool does not.

## 4. Configure

The tool never assumes anything about your machine. Give the settings in a JSON file (`--config cmp.json`, or `cmp.json` in the
current directory) or as environment variables; environment variables win. If something needed for the chosen stack is missing,
the tool prints the list and starts nothing (exit code 2).

| config key | environment | needed for | meaning |
|---|---|---|---|
| `models` | `CMP_MODELS` | all | folder holding `<model>/` directories (the shared copy of section 3) |
| `caches` | `CMP_CACHES` | all | scratch folder for compile caches; each stack and model gets its own sub-folder |
| `gpu` | `CMP_GPU` | all | OpenVINO device name used by every stack, e.g. `GPU.0` or `GPU.1` (list yours with `./rustedvino --device-info`; on a machine with several GPUs a bare `GPU` is ambiguous, and on a machine with one GPU RustedVINO accepts only `GPU`, not `GPU.0`, while OVMS accepts both. Use the same physical device for every stack) |
| `rv_dir` | `CMP_RV_DIR` | `rv` | the unpacked RustedVINO bundle |
| `ovms_dir` | `CMP_OVMS_DIR` | `ovms` | the unpacked OVMS package (contains `bin/ovms`, `lib/`) |
| `ovms_docker` | `CMP_OVMS_DOCKER` | `ovms`, instead of `ovms_dir` | docker image name, e.g. `openvino/model_server:2026.4.1-gpu` |
| `ovms_render` | `CMP_OVMS_RENDER` | with `ovms_docker` | render node passed into the container, e.g. `/dev/dri/renderD128` |
| `ovms_gpu` | `CMP_OVMS_GPU` | optional | device name inside the container (default: same as `gpu`; one mapped render node usually appears as `GPU.0`) |
| `oa_dir` | `CMP_OA_DIR` | `openarc` | the OpenArc checkout that holds `.venv` |
| `pdev` | `CMP_PDEV` | optional | PCI address of the GPU under test (e.g. `0000:03:00.0`, from `lspci`) for the VRAM column; empty = sum over every DRM device the server touches |
| `rv_vram_gb` | `CMP_RV_VRAM` | optional | the `vram_gb` RustedVINO books for the model (recorded in the result) |
| `rv_nocache` | `CMP_RV_NOCACHE` | optional | `true`: RustedVINO runs without a compile-cache folder; then both "cold" and "warm" loads are cold (recorded in the result) |

Paths may be absolute or start with `~`; the tool resolves them. Unknown keys are reported. An empty environment variable does not clear a JSON value.

`cmpbench.example.json` shows the shape.

## 5. Run one stack on one model

```
python3 cmpbench.py --config cmp.json --stack rv      --model qwen3-1.7b-int4-ov --out results/pass1 --repeat 5
python3 cmpbench.py --config cmp.json --stack ovms    --model qwen3-1.7b-int4-ov --out results/pass1 --repeat 5
python3 cmpbench.py --config cmp.json --stack openarc --model qwen3-1.7b-int4-ov --out results/pass1 --repeat 5
```

Per call: the versions of that stack are printed first; then an empty compile cache is created, the server is started and the model
loaded (cold load time), the server is stopped (exit code and time recorded; RustedVINO gets up to 720 s, since a stop waits for a running model load, and is never SIGKILLed: past that it is left running with a WARNING and the run stops), started again with the now-warm cache (warm load
time), an equivalence probe is sent (same greedy prompt on every stack; compare the texts and `prompt_tokens`), a short warm-up
per request shape runs, then the measured workload and the fixed 256-token decode. The server is stopped at the end. One server
runs at a time. `--skip-cold` reuses an existing cache; `--no-shape-warmup` skips the per-shape warm-up (not recommended: it
makes the prompt-processing number read low).

Output: `<out>/<stack>/<model>/results.json` and `results.md` (versions at the top), plus the server's own log in the same folder.

## 6. Build the table

```
python3 cmptable.py results pass1            # one pass
python3 cmptable.py results pass1 pass2      # several passes: cells read "pass1 / pass2"
```

The table starts with the versions each stack ran with and warns when the OpenVINO GenAI versions differ. It then prints one row
per stack and model. It shows numbers only and does not interpret them.

**Do at least two passes and reverse the order in the second** (stack order R-O-A, then A-O-R, with `rv`, `ovms`, `openarc`):
thermal state and background activity drift during a session, and a difference that flips when the order flips is not a difference.

## 7. Fairness rules the tool applies, and the ones you must apply

Applied by the tool:
- **Same request for every stack:** the same prompts as `rvbench.py`, a unique tag at the start of each prompt (defeats prefix
  caching), `temperature: 0`, same `max_tokens`, thinking switched off. Each server receives the thinking switch in the form it
  accepts (RustedVINO: top-level `enable_thinking: false`; the others: `chat_template_kwargs`); the tool counts responses that still
  contain `<think>` text or a `reasoning_content` field (`think_text_seen`, must be 0).
- **Equalised server settings:** KV cache 2 GB, KV precision `u8` (where the server accepts it), 16 sequences, prefix caching on,
  explicit compile-cache folders. Defaults differ between servers, so the tool writes these into each server's own config
  (RustedVINO's config, OVMS's graph file, OpenArc's `add` flags); it cannot verify that each server honours them. The server's
  log and the generated config are in the result folder (OpenArc's own model entry lives in its checkout and is removed after the run).
- **Token counting from the server's `usage` field;** decode speed is shown two ways (`ct/(total-ttft)` and chunk-timed
  ms/token). The fixed-256 row counts only responses that produced exactly 256 tokens, because stop conditions differ between servers.
  A response without a `usage` block falls back to counting chunks: it is counted in `responses_without_usage`, the prefill cell
  becomes `-`, and the run fails its checks (below).
- **Shape warm-up** before measuring (short, long, 256-token and one discarded 2- and 4-stream burst), so one-time kernel
  compilation is not counted against the first stack that sees a request shape.
- **Checks:** every result carries a `checks` block (thinking seen, fixed-256 samples with exactly 256 tokens, responses without
  `usage`, empty compile cache after the cold load, GPU memory resident before the run). The table prints it under each row; the
  tool exits with code 4 when a check failed. A failed check means that run's numbers are not usable as they stand.
- **Cold and warm loads** use each server's own empty / warm compile cache; never compare one stack's warm load with another's cold.

You must apply:
- **Same model files, same GPU, nothing else on that GPU.** The tool's idle check sees only processes of the user that runs it
  (it prints how many it could not inspect; run as root to see every user's GPU memory). Check `gpu_resident_gib_before` is ~0
  and look yourself for GPU services of other users.
- **A fixed power profile for the whole session,** logged by you.
- **Repeat the session** (section 6) and report the spread, not one run.
- **Read the versions line** of every stack before comparing anything (section 8).

## 8. Reading the numbers: limits

- **Versions are not equalised unless you make them so.** Different servers bundle different OpenVINO GenAI versions; the engine
  has a large influence on speed. A gap between two servers on different GenAI versions cannot be assigned to either server. The
  versions line exists so that this is visible; the strongest comparison runs both servers on the same GenAI generation.
- **Small gaps are noise.** With 5 samples per cell, differences below about 3 % (or 0.3 ms per token) in decode speed should not be
  called real. Report medians, and the min-max range of the fixed 256-token row; repeat sessions for the others (section 6).
- **The prompt-processing ("prefill") number is prompt tokens divided by time to first token for one ~700-token prompt.** It
  includes each server's fixed first-token cost and is not a per-token rate.
- **Parallel-stream rows are one burst each** (after one discarded burst of the same width); the tool does not interpret them.
  The prefill row has `repeat - 1` samples; min-max is stored only for the fixed 256-token row.
- **Memory is host RSS** of the server's process tree (RAM the process holds in the host) and GPU memory from `fdinfo`. They are
  not the same quantity; weights placed in GPU memory do not show as host RSS. Sampled every 0.5 s. The `*_peak_workload_*`
  values cover the measured workload only; `*_peak_incl_warm_load_*` also cover the warm load; the cold load is not sampled.
- **Load times** are polled every 0.05 s (resolution about 0.05 s); a warm load of about a second is therefore a rough figure.
- **"Cold" means an empty OpenVINO compile cache only.** The GPU driver's own kernel cache is not cleared, so cold load times are
  optimistic. The operating system's page cache is not cleared either: the first stack to read a model folder after you copied
  it reads from disk, later ones from memory. Run the stacks in different orders (section 6), or drop the page cache between runs.
- **One machine, one client.** The benchmark client runs on the same CPU as the servers and competes with them at high stream
  counts. Results from your hardware describe your hardware; do not generalise to other GPUs or CPUs.
- **Small models only unless you extend it.** We used a 1.7B and an 8B model at int4. Larger or differently shaped models can
  rank the servers differently.
- If you publish numbers that involve someone else's software, show them the setup and the raw results first; it is the
  quickest way to find a mistake in your configuration.

## 9. Troubleshooting

| message | meaning |
|---|---|
| `configuration incomplete: missing 'gpu' ...` | a required key is absent; add it to the JSON file or the environment |
| `port(s) [8000] already answer on 127.0.0.1` | another server is using a port the stack needs; stop it |
| `WARNING: ... GiB of VRAM is already resident` | another program holds GPU memory; stop it before trusting the run |
| `<stack> exited rc=...` / `model not listed after ...` | the server failed to load the model; the last log lines are in the message and the full log is in the result folder |
| `CPU is N% busy ... not quiet` (exit 3) | something is using the CPU (the previous server still shutting down, a browser, an indexer); find it with `top` and wait. Raising `--max-cpu-pct` is possible but is recorded in the result |
| `error while loading shared libraries: libxml2.so.2` / `libicuuc.so.74` (OVMS) | the OVMS package is built for Ubuntu 24.04; newer distributions ship other versions of these libraries. Install the 24.04 `libxml2` and `libicu74` packages, or extract them (`dpkg-deb -x`) into a private folder and copy the `.so` files into the package's `lib/` (the tool already puts that folder on `LD_LIBRARY_PATH`) |
| exit code 4 | the run finished but a check failed (see `checks` in `results.json`); do not use its numbers as they stand |
| `--model must be a folder name` | `--model` is a folder name under the models folder, not a path |
| RustedVINO exits at start: `device = 'GPU.0' is not an available OpenVINO device (enumerated: ["CPU", "GPU", "NPU"])` | set `gpu` to a name from the `enumerated` list |
