#!/usr/bin/env python3
"""livecaptions — live captions (and optional translation) from a microphone,
a call's audio or a media file, using a RustedVINO server's
/v1/audio/transcriptions and, optionally, /v1/chat/completions.

Why Python, not Rust: like semsearch, this is a client of the server, not a
server component. It has to run cold with no build step, and its whole job is
"read audio from ffmpeg, find the pauses, POST each utterance." Stdlib only.

Audio: ffmpeg does all capture and decoding and hands us raw 16 kHz mono
s16le on a pipe — a PulseAudio/PipeWire source (a microphone, or a speaker's
`.monitor` for everyone on a call), or a file played back at real speed
(`-re`) for a repeatable test. One code path for all three.

Segmentation: an energy VAD in pure Python, with the same constants as the
server's /v1/realtime VAD (RMS 0.015, 100 ms onset, 500 ms of silence ends an
utterance), except that the threshold also adapts upward to the recent noise
floor, because broadcast audio never drops to a quiet room's level. Past
--soft-seconds a 300 ms pause is enough; at --max-seconds (default 25) the cut
is forced, because Whisper's window is 30 s. A short pre-roll is kept so the
first syllable isn't clipped by the onset delay.

Threads: capture + VAD run in their own thread and push utterances onto a
queue that the HTTP worker drains in order. If they shared a thread, the
ffmpeg pipe would back up while a request is in flight and audio would drop.

Model choice mirrors semsearch: --model, then LIVECAPTIONS_MODEL, then the
only speech-to-text model on --target. No hardcoded id — model ids are local
registration names. Translation needs an explicit --llm-model.
"""
from __future__ import annotations

import argparse
import array
import collections
import io
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEYS_PATH = os.path.join(TOOL_DIR, "api_keys.json")

DEFAULT_TARGET = "http://localhost:11437"

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2  # s16le mono

# Same values as the server's realtime VAD (src/handlers/realtime.rs).
VAD_THRESHOLD = 0.015
ONSET_MS = 100
OFFSET_MS = 500
PREROLL_MS = 300
MIN_UTTERANCE_MS = 300  # shorter than this is a click, not speech

# The fixed threshold suits a microphone in a quiet room, not broadcast audio:
# a compressed podcast's quietest 5% of frames measured ~0.02 RMS, above the
# threshold, so no pause was ever found. The threshold therefore also adapts
# to the recent noise floor: ADAPT_FACTOR x the ADAPT_PERCENTILE-th percentile
# of the last ADAPT_WINDOW_MS, never below --threshold.
ADAPT_WINDOW_MS = 10_000
ADAPT_PERCENTILE = 10
ADAPT_FACTOR = 2.5
# Past --soft-seconds, a shorter pause ends the utterance, so a speaker who
# never stops is cut at a breath rather than mid-word at --max-seconds.
# Measured trade-off: 200 ms cut inside words (a plosive's closure is silent)
# and 8 s cut mid-phrase often enough that Whisper invented an ending for the
# cut-off phrase; 300 ms and 12 s gave ~5 s median captions on both a podcast
# and an audiobook with one such ending in three minutes.
SOFT_OFFSET_MS = 300

# Digital silence added to both ends of every utterance before it's sent.
# Measured: one 9.7 s chunk of clear speech that began with ~180 ms of room
# noise came back as a single word, every time; the same chunk with 300 ms of
# leading silence transcribed in full. Timestamps are unaffected.
EDGE_PAD = bytes(SAMPLE_RATE * 300 // 1000 * 2)

RETRIES_429 = 3  # the server answers 429 when its admission queue times out


def log(msg):
    print(f"livecaptions: {msg}", file=sys.stderr, flush=True)


def resolve_api_key(cli_key, target):
    """Precedence: --api-key > LIVECAPTIONS_API_KEY > api_keys.json (gitignored,
    keyed by the exact --target string — see api_keys.example.json)."""
    if cli_key:
        return cli_key
    env_key = os.environ.get("LIVECAPTIONS_API_KEY")
    if env_key:
        return env_key
    if os.path.exists(API_KEYS_PATH):
        try:
            with open(API_KEYS_PATH, encoding="utf-8") as f:
                return json.load(f).get(target)
        except (json.JSONDecodeError, OSError) as e:
            log(f"WARNING: {API_KEYS_PATH} exists but couldn't be read ({e}) — ignoring it")
    return None


def _auth_headers(api_key):
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


# ---------------------------------------------------------------------------
# Server calls
# ---------------------------------------------------------------------------

def pcm_to_wav(pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def _multipart(fields, file_bytes):
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    out.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"utterance.wav\"\r\n"
              "Content-Type: audio/wav\r\n\r\n".encode())
    out.write(file_bytes)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def transcribe(target, model, pcm, language, api_key):
    fields = {"model": model, "response_format": "json"}
    if language:
        fields["language"] = language
    body, ctype = _multipart(fields, pcm_to_wav(pcm))
    for attempt in range(RETRIES_429 + 1):
        req = urllib.request.Request(f"{target}/v1/audio/transcriptions", data=body,
                                     headers={"Content-Type": ctype, **_auth_headers(api_key)}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read()).get("text", "").strip()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < RETRIES_429:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise


def translate(target, model, text, source, dest, api_key):
    """One chat call per caption. enable_thinking=false: this is a
    transformation job, and a reasoning model would otherwise spend the
    budget thinking. Returns (translation, finish_reason)."""
    src = f" from {source}" if source else ""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": f"Translate the user's text{src} into {dest}. "
                                          "Reply with the translation only, no notes or quotes."},
            {"role": "user", "content": text},
        ],
        "max_tokens": max(64, len(text)),
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{target}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", **_auth_headers(api_key)},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        choice = json.loads(resp.read())["choices"][0]
    return (choice["message"].get("content") or "").strip(), choice.get("finish_reason")


def _error_detail(e):
    body = e.read().decode(errors="replace")
    try:
        err = json.loads(body)
    except json.JSONDecodeError:
        return body[:300]
    err = err.get("error", err) if isinstance(err, dict) else err
    msg = err.get("message", body) if isinstance(err, dict) else err
    return str(msg)[:300]


def _model_ids(target, api_key):
    req = urllib.request.Request(f"{target}/v1/models", headers=_auth_headers(api_key))
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return [m["id"] for m in json.loads(resp.read()).get("data", [])]
    except urllib.error.HTTPError as e:
        log(f"{target}/v1/models rejected the request (HTTP {e.code}): {_error_detail(e)}")
        if e.code in (401, 403):
            log(f"  pass --api-key, set LIVECAPTIONS_API_KEY, or add an entry to {API_KEYS_PATH}")
        sys.exit(1)
    except urllib.error.URLError as e:
        log(f"cannot reach {target} ({e}). Is RustedVINO running? Try: curl -s {target}/health")
        sys.exit(1)


def _stt_models(target, api_key, verbose):
    """Which models on `target` answer /v1/audio/transcriptions. Probed with
    0.1 s of silence. /v1/models lists only loaded models, and the server
    rejects a loaded model of another kind straight away, so the probe is
    cheap and never loads anything. The flip side: an `on_demand` model that
    isn't loaded yet isn't found — name it with --model."""
    ids = _model_ids(target, api_key)
    silence = bytes(SAMPLE_RATE // 10 * 2)
    if verbose:
        print(f"livecaptions: probing {len(ids)} model(s) on {target} for speech-to-text:")
    found = []
    for mid in ids:
        try:
            transcribe(target, mid, silence, None, api_key)
            found.append(mid)
            verdict = "speech-to-text: usable"
        except urllib.error.HTTPError as e:
            verdict = _error_detail(e)
        except urllib.error.URLError as e:
            verdict = f"unreachable: {e}"
        if verbose:
            print(f"  {mid}  — {verdict}")
    return found


def resolve_model(cli_model, target, api_key):
    if cli_model:
        return cli_model
    if os.environ.get("LIVECAPTIONS_MODEL"):
        return os.environ["LIVECAPTIONS_MODEL"]
    found = _stt_models(target, api_key, verbose=False)
    if len(found) == 1:
        return found[0]
    if not found:
        log(f"no speech-to-text model found on {target}. Run --list-models to see what's there.")
    else:
        log(f"{len(found)} speech-to-text models on {target} ({', '.join(found)}) — pick one with --model.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Capture + VAD (capture thread)
# ---------------------------------------------------------------------------

def ffmpeg_cmd(args):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if args.file:
        cmd += (["-re"] if not args.fast else []) + ["-i", args.file]
    else:
        cmd += ["-f", "pulse", "-i", args.source]
    return cmd + ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]


def pulse_sources():
    """Source names ffmpeg can capture, or None if it can't list them. Lines
    look like `  <name> [<description>] (none)`, the default marked `*`."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-sources", "pulse"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = [line.lstrip("* ").split(" ", 1)[0] for line in out.splitlines()
             if line.startswith(("  ", "* ")) and line.strip()]
    return names or None


def rms(frame):
    samples = array.array("h", frame)
    if sys.byteorder != "little":
        samples.byteswap()
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0


def capture(proc, out_q, threshold, soft_seconds, max_seconds, started):
    """Read 20 ms frames, emit (start_s, end_s, pcm, committed_at) per
    utterance. Times are seconds of audio since capture started."""
    onset_frames = ONSET_MS // FRAME_MS
    offset_frames = OFFSET_MS // FRAME_MS
    preroll_frames = PREROLL_MS // FRAME_MS
    soft_offset_frames = SOFT_OFFSET_MS // FRAME_MS
    soft_frames = int(soft_seconds * 1000) // FRAME_MS
    max_frames = int(max_seconds * 1000) // FRAME_MS
    min_frames = MIN_UTTERANCE_MS // FRAME_MS
    window = collections.deque(maxlen=ADAPT_WINDOW_MS // FRAME_MS)
    level = threshold   # current effective threshold, recomputed every 0.5 s

    history = []        # recent frames while idle (pre-roll + onset candidates)
    loud_run = 0        # consecutive loud frames while idle
    speech = None       # frames of the current utterance
    speech_start = 0
    silent_run = 0
    n = 0               # frames read so far

    def commit(end_frame):
        nonlocal speech
        if speech and len(speech) >= min_frames:
            out_q.put((speech_start * FRAME_MS / 1000, end_frame * FRAME_MS / 1000,
                       b"".join(speech), time.monotonic()))
        speech = None

    while True:
        frame = proc.stdout.read(FRAME_BYTES)
        if len(frame) < FRAME_BYTES:
            break
        if n == 0:
            started.set()
        n += 1
        energy = rms(frame)
        window.append(energy)
        if n % 25 == 0:
            floor = sorted(window)[len(window) * ADAPT_PERCENTILE // 100]
            level = max(threshold, floor * ADAPT_FACTOR)
        loud = energy > level
        if speech is None:
            history.append(frame)
            loud_run = loud_run + 1 if loud else 0
            if loud_run >= onset_frames:
                keep = history[-(preroll_frames + loud_run):]
                speech, speech_start, silent_run = list(keep), n - len(keep), 0
                history, loud_run = [], 0
            else:
                del history[:-(preroll_frames + onset_frames)]
            continue
        speech.append(frame)
        silent_run = 0 if loud else silent_run + 1
        if silent_run >= offset_frames or (len(speech) >= soft_frames and silent_run >= soft_offset_frames):
            commit(n - silent_run)
        elif len(speech) >= max_frames:
            commit(n)
            speech, speech_start, silent_run = [], n, 0
    commit(n)
    started.set()  # also on EOF / ffmpeg failure, so main() doesn't wait for audio that never comes
    out_q.put((None, n))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def srt_time(s):
    ms = int(round(s * 1000))
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def clock(s):
    return f"{int(s // 60):02}:{s % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser(description="Live captions (and optional translation) from a microphone, "
                                             "a call or a file, via a RustedVINO server.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--source", default="default",
                     help="PulseAudio/PipeWire source: a microphone, or a speaker's .monitor for a call "
                          "(default: the default input). See --list-sources")
    src.add_argument("--file", help="caption a media file instead, played at real speed")
    ap.add_argument("--fast", action="store_true", help="with --file: don't pace to real time")
    ap.add_argument("--target", default=DEFAULT_TARGET, help=f"server base URL (default {DEFAULT_TARGET})")
    ap.add_argument("--model", help="speech-to-text model id (else LIVECAPTIONS_MODEL, else the only one on --target)")
    ap.add_argument("--api-key", help="bearer key (else LIVECAPTIONS_API_KEY, else api_keys.json)")
    ap.add_argument("--language", help="spoken language code, e.g. en, pl. Strongly recommended: "
                                       "autodetect is unreliable on short utterances")
    ap.add_argument("--translate-to", metavar="LANG", help="also translate each caption into LANG (e.g. English)")
    ap.add_argument("--llm-model", default=os.environ.get("LIVECAPTIONS_LLM_MODEL"),
                    help="chat model for --translate-to (or LIVECAPTIONS_LLM_MODEL)")
    ap.add_argument("--srt", metavar="PATH", help="also write captions to an SRT file")
    ap.add_argument("--threshold", type=float, default=VAD_THRESHOLD,
                    help=f"minimum speech energy, 0-1 RMS (default {VAD_THRESHOLD}, as the server's VAD). "
                         "It also adapts upward to the recent noise floor")
    ap.add_argument("--soft-seconds", type=float, default=12.0,
                    help="past this, a 300 ms pause is enough to end a caption (default 12)")
    ap.add_argument("--max-seconds", type=float, default=25.0,
                    help="cut an utterance that runs this long without a pause (default 25; Whisper sees 30)")
    ap.add_argument("--stats", action="store_true", help="print per-caption timing to stderr")
    ap.add_argument("--list-models", action="store_true", help="show which models on --target can transcribe")
    ap.add_argument("--list-sources", action="store_true", help="show audio sources ffmpeg can capture")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        log("ffmpeg not found on PATH. It does all the audio capture and decoding; install it first.")
        sys.exit(1)
    if args.list_sources:
        subprocess.run(["ffmpeg", "-hide_banner", "-sources", "pulse"])
        print("\nPass a microphone name, or a speaker's '.monitor' name to caption what you hear (e.g. a call).")
        return

    api_key = resolve_api_key(args.api_key, args.target)
    if args.list_models:
        found = _stt_models(args.target, api_key, verbose=True)
        print()
        print(f"livecaptions: --model {found[0]}" if len(found) == 1 else
              "livecaptions: pick one with --model: " + ", ".join(found) if found else
              f"livecaptions: no speech-to-text model on {args.target}")
        return
    if args.translate_to and not args.llm_model:
        log("--translate-to needs a chat model: pass --llm-model or set LIVECAPTIONS_LLM_MODEL")
        sys.exit(1)
    if args.file and not os.path.exists(args.file):
        log(f"no such file: {args.file}")
        sys.exit(1)
    # PipeWire quietly falls back to the default microphone for a source name
    # it doesn't know, so a typo would caption the wrong device without a word.
    if not args.file and args.source != "default":
        known = pulse_sources()
        if known is not None and args.source not in known:
            log(f"no audio source named '{args.source}'. Run --list-sources to see the names.")
            sys.exit(1)

    model = resolve_model(args.model, args.target, api_key)
    try:  # preflight: one tiny request, so a wrong model fails now, not on the first sentence
        transcribe(args.target, model, bytes(SAMPLE_RATE // 10 * 2), args.language, api_key)
    except urllib.error.HTTPError as e:
        log(f"{args.target} rejected model '{model}' (HTTP {e.code}): {_error_detail(e)}")
        sys.exit(1)
    except urllib.error.URLError as e:
        log(f"cannot reach {args.target} ({e}). Is RustedVINO running? Try: curl -s {args.target}/health")
        sys.exit(1)

    # Own session: Ctrl-C reaches only us, and we stop ffmpeg ourselves
    # (otherwise it prints a screenful of muxer errors on the way out).
    proc = subprocess.Popen(ffmpeg_cmd(args), stdout=subprocess.PIPE, start_new_session=True)
    utterances = queue.Queue()
    started = threading.Event()
    t0 = [None]

    def run_capture():
        capture(proc, utterances, args.threshold, args.soft_seconds, args.max_seconds, started)

    threading.Thread(target=run_capture, daemon=True).start()
    started.wait(10)
    t0[0] = time.monotonic()
    what = args.file or args.source
    log(f"captioning {what} with {model}" + (f", translating into {args.translate_to} with {args.llm_model}"
                                              if args.translate_to else "") + ". Ctrl-C to stop.")

    srt = open(args.srt, "w", encoding="utf-8") if args.srt else None
    out_lock = threading.Lock()
    lags, tlags = [], []
    count = [0]

    def since_end(end_s, committed_at):
        """Seconds from the end of the speech to now. Includes the 500 ms of
        silence that ends an utterance. With --fast there is no real-time
        clock to compare to, so measure from the moment the VAD committed."""
        return time.monotonic() - (committed_at if args.fast else t0[0] + end_s)

    def emit(line, start_s, end_s, srt_text):
        with out_lock:
            print(line, flush=True)
            if srt:
                count[0] += 1
                srt.write(f"{count[0]}\n{srt_time(start_s)} --> {srt_time(end_s)}\n{srt_text}\n\n")
                srt.flush()

    # Translation runs in its own thread, so a slow LLM call never holds up
    # the next transcription: the original caption appears in well under a
    # second, and its translation follows when it's ready.
    to_translate = queue.Queue()

    def run_translator():
        while (item := to_translate.get()) is not None:
            start_s, end_s, text, committed_at = item
            try:
                translation, finish = translate(args.target, args.llm_model, text, args.language,
                                                args.translate_to, api_key)
                if finish == "length":
                    translation += " [cut off]"
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                translation = f"[translation failed: {_error_detail(e) if hasattr(e, 'read') else e}]"
            tlag = since_end(end_s, committed_at)
            tlags.append(tlag)
            emit(f"[{clock(start_s)}] → {translation}", start_s, end_s, translation)
            if args.stats:
                log(f"  translation {tlag:.2f}s after the speech ended")

    translator = None
    if args.translate_to:
        translator = threading.Thread(target=run_translator, daemon=True)
        translator.start()

    frames_read = None
    try:
        while (item := utterances.get())[0] is not None:
            start_s, end_s, pcm, committed_at = item
            try:
                text = transcribe(args.target, model, EDGE_PAD + pcm + EDGE_PAD, args.language, api_key)
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                log(f"transcription failed at {clock(start_s)}: {_error_detail(e) if hasattr(e, 'read') else e}")
                continue
            if not text:
                continue
            lag = since_end(end_s, committed_at)
            lags.append(lag)
            if translator:
                with out_lock:
                    print(f"[{clock(start_s)}] {text}", flush=True)
                to_translate.put((start_s, end_s, text, committed_at))
            else:
                emit(f"[{clock(start_s)}] {text}", start_s, end_s, text)
            if args.stats:
                log(f"  {end_s - start_s:4.1f}s of speech, caption {lag:.2f}s after it ended")
        frames_read = item[1]
    except KeyboardInterrupt:
        pass
    finally:
        proc.kill()  # nothing to flush on a pipe; SIGTERM makes ffmpeg complain
        proc.wait()
        if frames_read == 0:
            log(f"ffmpeg produced no audio from {what} — see its error above "
                "(for a device, check --list-sources)")
        if translator:
            to_translate.put(None)
            try:
                translator.join()
            except KeyboardInterrupt:
                pass
        if srt:
            srt.close()
        if args.stats:
            for name, xs in (("caption", lags), ("translation", tlags)):
                if xs:
                    xs = sorted(xs)
                    log(f"{len(xs)} {name}s; lag median {xs[len(xs) // 2]:.2f}s, max {xs[-1]:.2f}s")

if __name__ == "__main__":
    main()
