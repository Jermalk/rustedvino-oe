# `/v1/realtime` — WebSocket voice protocol

Reference for client developers. It describes the protocol as the server implements it in
this version. For the HTTP routes, see [`INTERFACE.md`](INTERFACE.md); for configuration, see
[`CONFIG.md`](CONFIG.md); for metrics, see [`METRICS.md`](METRICS.md).

## Overview

One WebSocket connection is one voice session. Within it:

1. The client streams microphone audio as binary frames.
2. The server runs voice activity detection (VAD) on that audio and cuts it into utterances.
3. Each utterance is one **turn**: speech-to-text (Whisper), then the LLM, then text-to-speech.
4. The server sends back the transcript and the reply text as JSON text frames, and the spoken
   reply as binary audio frames, one sentence at a time.

Turns run one at a time, in the order the utterances ended. The session keeps a rolling
conversation history for the LLM, and an optional retrieval pool of past exchanges, which is
searched with embeddings.

This is a **bespoke protocol**. It borrows the name "realtime" from OpenAI, but it is not
OpenAI's Realtime API and does not use its event schema. A client written for OpenAI's
Realtime API will not work against it.

## Connecting

**URL:** `ws://<host>:<port>/v1/realtime` (or `wss://` behind a TLS proxy). The upgrade request
is an ordinary `GET`.

**Authentication** follows the inference routes:

- When the server has no inference keys configured, anyone who can reach the port can connect.
- Otherwise the upgrade request must carry `Authorization: Bearer <key>`. Either an inference
  key or an admin key is accepted.
- A missing or unknown key is refused with an HTTP `401` and a JSON error body before the
  upgrade. No WebSocket is opened.
- The header is the only way to pass a key: there is no query parameter or subprotocol form.
  The browser `WebSocket` API cannot set request headers, so a browser client can connect only
  to a server with no keys, or through a proxy that adds the header.

**On connect** the server:

- creates the session. It has no models set yet; see [`config`](#config) below.
- If [`default_embed_model`](CONFIG.md) is configured, sets the session's embedding model to
  it, and uses that model for retrieval if it is already loaded.
- registers the session in the admin session list, under a session id (a UUID). The client is
  never told this id.
- starts VAD on incoming audio at once.

The server sends **no event at connect**: no greeting, no session id, no model list. Send a
`config` event first. Audio sent before any `config` is still processed, and its turns fail
with an `asr` error, because no speech-to-text model is set.

## Frames

### Binary frames, client → server: microphone audio

- **Format:** PCM, signed 16-bit little-endian, mono, **16 000 Hz**. No header, no container.
  The server does not resample input, so audio at any other rate is misread.
- **Frame size:** up to 16 MiB per frame (over eight minutes of audio). A larger frame is a
  WebSocket protocol error and the connection is closed. If a frame has an odd number of
  bytes, the last byte is dropped.
- **VAD works per frame.** The server compares the RMS level of each whole frame with a fixed
  threshold of 0.015 (about −36 dBFS):
  - Speech starts once at least 100 ms of audio has arrived in consecutive frames above the
    threshold. A single frame of 100 ms or more that is above it starts speech at once.
  - An utterance ends after at least 500 ms of consecutive frames below the threshold.
  - The audio from the frames that started speech is kept in the utterance, and so is the
    trailing silence (up to about 500 ms below the threshold) that ended it.

  Timing is therefore only as fine as the frame size. Frames of a few tens of milliseconds
  give the most precise start and end points.
- **Long speech:** an utterance that never falls silent is committed when it reaches 29.5 s
  (472 000 samples): the server sends `speech_stopped`. If the remaining audio of that frame
  is speech, it starts the next utterance at once, and the server sends `speech_started` for
  it. See also the `GET /v1/realtime` section of [`INTERFACE.md`](INTERFACE.md).
- VAD runs at all times, including while a turn is being answered.

### Binary frames, server → client: synthesized speech

- **Format:** PCM, signed 16-bit little-endian, mono, **16 000 Hz**, with no header. This is
  the same format as the input. A TTS model that produces another rate is resampled to 16 kHz
  first.
- **Chunking:** one binary frame per synthesized sentence, holding that sentence's complete
  audio. Frames are as long as the sentence, so expect several seconds of audio (tens to
  hundreds of kilobytes) in one frame.
- **Order relative to text:**
  - Reply text is sent as `text` events as it is generated. When those pieces complete a
    sentence, the server synthesizes it and sends its audio frame. The audio of a sentence
    therefore always follows the `text` events that contain it.
  - While a sentence is being synthesized, no `text` events are sent. Text generated in the
    meantime is sent after that sentence's audio.
  - The last partial sentence is spoken when generation ends, before `done`.
- **What is spoken:** the text sent to TTS is cleaned first, so the audio can differ from the
  `text` events:
  - numbers and symbols are expanded into words for the session's `language`; see the TTS
    notes in [`INTERFACE.md`](INTERFACE.md);
  - the characters `*`, `_`, `#`, `` ` ``, `~` and `|`, and emoji, are removed.
- A sentence whose synthesis fails is skipped without notice. Its text has already been sent.

### Text frames: JSON events

Every text frame in either direction is one JSON object, with a `"type"` field that names the
event. Field names are `snake_case`.

- **Client → server:** a frame that is not valid JSON, has an unknown `type`, lacks a
  required field, or has a field of the wrong type (e.g. `"max_tokens": "120"`) gets an
  [`error`](#error) event with `stage: "protocol"`. The whole event is dropped, and nothing in
  it is applied. Unknown extra fields in a valid event are ignored.
- **Pings:** WebSocket ping and pong frames are handled by the server's WebSocket layer. They
  have no protocol meaning. A client that stops reading can stall them; see Slow readers
  under [Limits](#limits).

## Client events

| `type` | Purpose |
|---|---|
| [`config`](#config) | Set or change session settings and models; optionally replace the history |
| [`cancel`](#cancel) | Stop the turn that is running |
| [`inject_history`](#inject_history) | Add past exchanges to the retrieval pool |
| [`reload_history`](#reload_history) | Replace the retrieval pool |
| [`request_model_load`](#request_model_load) | Ask the server to load a model in advance |

### `config`

Changes session settings. It can be sent at any time, as often as needed.

Every field is optional. An absent field, or `null`, leaves the current value unchanged. A
setting cannot be returned to "unset" this way, except `system_prompt` and `barge_in_phrase`,
which `""` clears.

| Field | Type | Effect |
|---|---|---|
| `stt_model` | string | Speech-to-text model id. Arbitrated by the server; see [Model resolution](#model-resolution) |
| `llm_model` | string | LLM id: a text-generation model or a vision model (used for text only). A text model served by the NPU pipeline is refused at turn time with an `llm` error. Arbitrated |
| `tts_model` | string | Text-to-speech model id. Arbitrated. `""` turns speech off for the session: no TTS model is chosen or loaded, and replies arrive as `text` events only. It stays off until a later `config` names a TTS model |
| `tts_voice` | string | Voice name passed to the TTS model (for example `"af_nicole"` for Kokoro). The valid names depend on the model. Not set: the model's default voice |
| `language` | string | Language code given to speech-to-text, and to the TTS text normaliser. Not set: no language is passed, and the STT model's own default applies. `""` is stored as an empty string, not as "unset" |
| `system_prompt` | string | System prompt for every LLM turn. Replaces the previous one. `""` clears it |
| `barge_in_phrase` | string | Stop phrase; see [Barge-in phrase turn](#barge-in-phrase-turn). `""` clears it |
| `embed_model` | string | Embedding model for retrieval. See [Embedding model](#embedding-model): it takes effect only if no embedding model is in use yet |
| `history` | array of [history entries](#history-entry) | Replaces both the rolling LLM context and the retrieval pool. See [History and retrieval](#history-and-retrieval) |
| `max_tokens` | integer ≥ 0 | Cap on LLM output tokens per turn. Not set: **120** |
| `max_sentences` | integer ≥ 0 | Ends the turn after this many sentences are spoken. Not set: no limit. See [Limits](#limits) |

**When a change takes effect:**

- The session is updated as soon as the event arrives.
- A turn copies the session settings when it starts, so a turn already running keeps its
  settings. The change applies from the next turn.
- This includes `history`: the session's history is replaced at once, but the running turn
  still uses its copy.

**What the server sends back:** after every `config`:

1. one [`model_loading`](#model_loading) event with `stage: "started"` for each slot whose
   resolved model is being loaded in the background;
2. one [`model_selection`](#model_selection) event.

`history` itself produces no event.

#### Model resolution

Every `config` resolves all three model slots (STT, LLM, TTS) again, including slots the event
did not mention. The value requested for a slot is the value now stored in the session: the
new field if one was sent, otherwise the model the previous resolution chose. The server, not
the client, decides which model each slot gets:

1. **The requested model is loaded:** it is granted (`source: "requested"`).
2. **The requested model is known and can be loaded** without evicting a protected or busy
   model: it is granted (`"requested"`), and a background load starts.
3. **Otherwise, or if nothing was requested for the slot:** the server picks an
   already-loaded model of the right kind:
   - It prefers the slot's default from [`realtime_defaults`](CONFIG.md), if that model is
     loaded.
   - Otherwise it picks the eligible model used by the most live realtime sessions.
   - For the LLM slot, the model must meet [`realtime_viable_minimum`](CONFIG.md) when that
     is configured.

   The result is `"default"` if it is the configured default, otherwise `"substituted"`.
4. **Nothing suitable is loaded:** the slot falls back to its configured default
   (`"default"`). The default is loaded in the background if a dry run says it fits, or if it
   is already loading.
5. **No default is configured** for the slot: the slot gets the requested id, or `""` if
   nothing was requested (`"substituted"`). Turns then fail or are silent, as described
   below.

**The TTS slot resolves to none when no TTS model can serve it (since 0.9.1).** When steps
1–5 end with a TTS model that is neither loaded nor loading — nothing requested and no TTS
model available, a configured default that is not loaded and does not fit, a named model
without room, or an unknown id — the slot gets `tts_model: ""`, `tts_source: "substituted"`,
`tts_reason: "none_available"`. Replies then arrive as `text` events only, with no per-sentence
warning and no dropped-sentence count. STT and LLM resolution are unchanged. Two consequences:

- **The id is not kept.** The session stores `""`, not the model that could not be used. A
  `config` that omits `tts_model` resolves the slot again (so a configured default is tried
  again), but a model the client named is forgotten: to ask for it again, send its id again
  (`"tts_model": "X"`).
- **A configured default that cannot load is reported as `substituted`.** 0.9.0 reported it as
  `tts_source: "default"` with the default's id; now it is `""` / `"substituted"` /
  `"none_available"`.

A TTS model that is already being loaded (for another session or a `request_model_load`) counts
as loading for this session too: it is granted (`"requested"`, or `"default"` for the default),
with a `model_loading` `started` event, and is watched like a load this session started.

Consequences for clients:

- **Omitting `tts_model` does not give text-only output.** If any TTS model is loaded, the
  slot is filled with it, and replies are spoken. To get text only, send `"tts_model": ""`.
- A model granted with a background load is not ready yet. Until it is, turns that need it
  fail:
  - STT: an `asr` error;
  - LLM: an `llm` error;
  - TTS: sentences are skipped, with no error.

  Wait for `model_loading` `ready` before speaking.
- **There is no kind check for a requested model.** Steps 1 and 2 grant any loaded or
  loadable model, whatever its kind. For example, `"llm_model": "<a TTS model id>"` gives
  `llm_source: "requested"`, and fails only at turn time, with an `llm` error.
- **A substitution is not retried.** The session stores the resolved id, so the next `config`
  requests the substitute, not the model originally asked for. To try model X again, send
  `"llm_model": "X"` (or the matching field) again.
- **The server never chooses an NPU text model for the LLM slot.** Realtime turns cannot run
  the NPU text pipeline, so steps 3 and 4 skip models placed on the NPU, and a
  `realtime_defaults.llm_model` on the NPU is ignored (with a warning in the server log). A
  client that names an NPU model in `llm_model` still gets it, and every turn then fails with
  an `llm` error.
- While a session is live, the three models its last `config` resolved to are protected from
  eviction. That protection ends when the session closes or a later `config` resolves to
  different models.

### `cancel`

```json
{"type": "cancel"}
```

No fields. It stops the turn that is running:

- LLM generation stops within about one token, and the TTS synthesis in progress is abandoned.
- The server sends [`done`](#done) for that turn.
- If `cancel` arrives during speech-to-text, the `transcript` event is still sent, followed by
  `done`.

Limits of `cancel`:

- It affects **only the turn in progress**. It has no effect when no turn is running, and it
  does not discard utterances already queued behind the current turn: each of them runs as a
  full turn.
- `text` events for tokens already generated can still arrive between `cancel` and `done`.
- A cancelled turn is not added to the history, and no `turn_history` event is sent, even
  if some reply text was already generated and sent as `text` events. This also holds when
  the cancel lands while the last sentences are being spoken. A client that wants to keep the
  partial reply can store it itself and send it back with `inject_history`.

The server never cancels a turn because the user started speaking. To interrupt a reply when
the user talks over it, the client must send `cancel`, for example on `speech_started`.

### `inject_history`

```json
{"type": "inject_history", "entries": [ /* history entries */ ]}
```

| Field | Type | Effect |
|---|---|---|
| `entries` | array of [history entries](#history-entry), required | Appended to the **retrieval pool** only. The rolling LLM context is not changed |

Processing:

- An empty array is ignored, and no event is sent.
- Otherwise:
  1. Entries without an `embedding` are embedded, if the session has an embedding model in use.
  2. The entries are appended to the pool. If the pool then holds more than 200 entries, the
     oldest are dropped.
  3. The server sends one `turn_history` event (with `turn_id: null`) for **every entry in the
     pool**, not only for the new ones.

Timing and capacity:

- The event is processed between turns. A batch sent during a turn waits until the turn ends.
- Up to 4 batches can wait. Further batches are dropped without notice.

### `reload_history`

```json
{"type": "reload_history", "entries": [ /* history entries */ ]}
```

| Field | Type | Effect |
|---|---|---|
| `entries` | array of [history entries](#history-entry), required | Replaces the **retrieval pool**: it is emptied, then filled with these entries (missing embeddings are computed as for `inject_history`, and the pool is capped at the newest 200). An empty array just empties the pool. The rolling LLM context is not changed |

- The server sends one [`history_reloaded`](#history_reloaded) event. It sends no
  `turn_history` events.
- Timing and the queue of 4 are as for `inject_history`.

### `request_model_load`

```json
{"type": "request_model_load", "model_id": "qwen3-8b"}
```

| Field | Type | Effect |
|---|---|---|
| `model_id` | string, required | Model to load in advance (to pre-warm before speaking) |

The server works out the model's slot from its kind: STT, TTS, embedding, or LLM for anything
else. It then applies the same rules as [Model resolution](#model-resolution), and answers with
[`model_loading`](#model_loading) events for `model_id`:

- **Already loaded:** `ready`.
- **A load was started:** `started`, followed later by `ready`, or by `failed` if the model is
  not ready within 90 s.
- **A load is already running** (from another session or a `config`): `started`, then as above.
- **The model is not loaded and cannot be loaded now** (a substitute would have been chosen,
  or the model is the slot's default and does not fit): `failed`, with one of the
  [reason codes](#model_selection) in `detail`.
- **An id the server does not know:** `failed` with `detail: "unknown_model"`. Nothing is
  loaded on its behalf.

What `request_model_load` does not do:

- It does not change the session's models, and it does not protect the model from eviction.
  Only a `config` that resolves to the model does that.
- Embedding models are handled like any other model: `ready` if loaded, or `started` and
  later `ready` if the model fits. But loading one does not make the session use it for
  retrieval; see [Embedding model](#embedding-model).

## Server events

| `type` | When |
|---|---|
| [`speech_started`](#speech_started) | VAD detected the start of speech |
| [`speech_stopped`](#speech_stopped) | VAD committed an utterance |
| [`transcript`](#transcript) | Speech-to-text result of a turn |
| [`text`](#text) | A piece of the reply text |
| [`done`](#done) | A turn ended (normally, cancelled, skipped or failed) |
| [`stopped`](#stopped) | A turn ended because of the barge-in phrase |
| [`turn_history`](#turn_history) | A history entry: a completed turn, or a pool entry echoed back |
| [`history_reloaded`](#history_reloaded) | `reload_history` completed |
| [`model_selection`](#model_selection) | Result of model resolution, after each `config` |
| [`model_loading`](#model_loading) | Progress of a background model load |
| [`error`](#error) | A turn or a frame failed |

### `speech_started`

`{"type": "speech_started"}`. No fields.

- Sent when VAD detects the start of speech, also while another turn is running.
- It does not cancel anything.

### `speech_stopped`

`{"type": "speech_stopped"}`. No fields.

- Sent when VAD ends an utterance (after 500 ms of silence, or at the 29.5 s cap).
- The utterance is then queued for a turn.
- If the queue is full, the utterance is discarded, and an `error` with `stage: "pipeline"`
  may follow; see [Limits](#limits).

### `transcript`

| Field | Type | Meaning |
|---|---|---|
| `text` | string | The transcript, trimmed of surrounding whitespace |
| `final` | boolean | Always `true`. There are no partial transcripts in this version |

- Sent once per answered turn, before any `text` event.
- It is not sent for a skipped turn (empty or short transcript), or for a barge-in phrase
  turn.

### `text`

| Field | Type | Meaning |
|---|---|---|
| `content` | string | The next piece of the reply. Concatenate the pieces in order to get the full reply |

- Pieces are fragments of the decoded output, not necessarily single tokens. With some model
  families the first piece is held back briefly and arrives combined.
- Reasoning ("thinking") is turned off for voice turns, and reasoning text is never sent.

### `done`

| Field | Type | Meaning |
|---|---|---|
| `turn_id` | integer | The id of the turn that ended |

`turn_id` identifies the turn:

- It starts at `0` on each connection and increases by one for every utterance that reaches
  the pipeline, whatever the outcome.
- Utterances discarded because the queue was full get no id.

`done` ends a turn that was answered, cancelled, skipped (empty or short transcript), or that
failed with an `admission`, `asr` or `llm` error.

One case does not end with `done`: a [barge-in phrase turn](#barge-in-phrase-turn), which ends
with `stopped`.

### `stopped`

| Field | Type | Meaning |
|---|---|---|
| `turn_id` | integer | The id of the turn |

Sent instead of `done` when the turn's transcript contained the barge-in phrase.

### `turn_history`

A history entry. Store it on the client to replay the conversation later.

| Field | Type | Meaning |
|---|---|---|
| `turn_id` | integer or `null` | The turn just completed, or `null` when an existing pool entry is echoed back (after `inject_history`, or when the server has embedded it: see [Embedding model](#embedding-model)) |
| `user` | string | The user's transcript |
| `assistant` | string | The reply text |
| `embedding` | array of numbers, optional | Embedding of `user + "\n" + assistant`. Omitted when no embedding model is in use, or embedding failed |
| `timestamp` | integer | Unix time in milliseconds. For a new turn, the server's clock when the turn ended. For an echoed entry, the entry's own value |
| `embedding_model` | string, optional | Model that produced `embedding` |
| `stt_model` | string, optional | STT model that produced `user` |
| `llm_model` | string, optional | LLM that produced `assistant` |

For a new turn, it is sent after the turn's `done`. Only turns with both a transcript and some
reply text produce it. A cancelled turn does not.

### `history_reloaded`

| Field | Type | Meaning |
|---|---|---|
| `count` | integer | Number of entries now in the retrieval pool |

### `model_selection`

Sent once after every `config`, after any `model_loading` `started` events from it.

| Field | Type | Meaning |
|---|---|---|
| `stt_model` | string | Resolved STT model id. Can be `""` (see [Model resolution](#model-resolution), step 5) |
| `stt_source` | string | How it was chosen (values below) |
| `stt_reason` | string, optional | Why it is not simply the requested model (values below) |
| `llm_model` | string | Resolved LLM id. Can be `""` |
| `llm_source` | string | As `stt_source` |
| `llm_reason` | string, optional | As `stt_reason` |
| `tts_model` | string | Resolved TTS model id. Always present. `""` when speech is off (`tts_source: "off"`) or no TTS model can serve the session (`tts_reason: "none_available"`) |
| `tts_source` | string | As `stt_source` |
| `tts_reason` | string, optional | As `stt_reason` |

`*_source` values:

| Value | Meaning |
|---|---|
| `requested` | The requested model was granted. It is loaded, or a background load was started (watch for `model_loading`) |
| `default` | The slot's configured default from `realtime_defaults` was chosen |
| `substituted` | Another loaded model was chosen. Also used when nothing could be chosen: the id is then the requested one, or `""` (for TTS always `""`, with `tts_reason: "none_available"`) |
| `off` | `tts_source` only: the client sent `"tts_model": ""`, so speech is off; `tts_model` is `""` |

`*_reason` values. The field is omitted when the requested model was granted, and when nothing
was requested and an already-loaded model was chosen:

| Value | Meaning |
|---|---|
| `would_evict_protected` | The requested model is not loaded, and loading it would evict a protected model (pinned, non-evictable, in its eviction grace window, or in use by a live realtime session). The same reason is given when the requested model is unknown to the server, when it does not fit even with nothing evictable, and when the server's dry-run check fails. A loaded model was chosen instead |
| `would_evict_busy` | The requested model, or the fallback default, cannot be loaded because the memory it needs is held by models with requests running. Nothing is loading for this slot. For TTS (since 0.9.1) only together with another loaded TTS model chosen instead; when there is none, the TTS reason is `none_available` |
| `not_resident_no_headroom` | No suitable model is loaded. With `source: "default"`, the default is being loaded in the background if a `model_loading` `started` event for it came just before; otherwise it does not fit and nothing is loading. With `source: "substituted"`, there is nothing to load. For TTS (since 0.9.1) only the first case: the default is loading; a TTS slot with nothing to load gets `none_available` |
| `none_available` | `tts_reason` only (since 0.9.1): no TTS model can serve the session, so `tts_model` is `""` and replies are text only. Nothing is loading for the slot. The next `config` resolves it again |

### `model_loading`

| Field | Type | Meaning |
|---|---|---|
| `model_id` | string | The model concerned |
| `stage` | string | `started`, `ready` or `failed` |
| `detail` | string, optional | For `failed`: a [reason code](#model_selection), `unknown_model` (reply to `request_model_load` for an unknown id), `"load did not complete within the watch window"` (the 90 s watch ran out), or `"model manager not initialised"` |

When it is sent:

- `started`, after a `config` (one per slot being loaded) or a `request_model_load`. The
  server then checks every 200 ms:
  - `ready` is sent when the model is loaded;
  - `failed` (detail `"load did not complete within the watch window"`) is sent if it is not
    loaded within 90 s. A load that fails earlier is reported only when those 90 s are up.
  - The session's TTS model, loaded after a `config`: when its load fails (the model is neither
    loaded nor loading any more), the session's TTS id is cleared to `""` at once (since 0.9.1),
    so later replies are text only, with no per-sentence warning or dropped count; the server
    logs one warning. No `model_selection` is sent for it; the next `config` resolves the slot
    again (send the id again to ask for the same model). The watch itself goes on: `failed`
    still comes when the 90 s are up, and if the model does get loaded within them (another
    load of it succeeded), `ready` is sent for it although this session no longer uses it —
    send a `config` naming it to use it.
- `ready` or `failed` immediately, in reply to a `request_model_load` (see that event).

A model can be reported several times in one session, for example if it is evicted and loaded
again.

`ready` sent in direct reply to `request_model_load` means the model is loaded at that moment.

### `error`

| Field | Type | Meaning |
|---|---|---|
| `stage` | string | Where it failed (values below) |
| `message` | string | Human-readable description. The wording is not stable; do not parse it |

| `stage` | Meaning | Turn outcome |
|---|---|---|
| `admission` | The turn could not get its device admission (the server's [`device_budgets`](CONFIG.md) caps) | `done` follows; nothing else was sent for the turn |
| `asr` | Speech-to-text failed: no STT model set (no `config` yet), the model is not loaded, or transcription failed | `done` follows |
| `llm` | The LLM failed: no LLM set, the model is not loaded or is being unloaded, the engine is at capacity or unavailable, the model is not a text or vision model, the prompt is too long, generation failed, or the generation stream ended without a finish (the reply was cut off, e.g. because the server could not deliver tokens as fast as they were produced) | `done` follows. `transcript` and possibly some `text` were already sent |
| `pipeline` | An utterance was discarded because the turn queue was full | Not tied to a turn. Sent once per backlog episode (see [Limits](#limits)) |
| `protocol` | A client text frame could not be parsed as an event | Nothing else happens; the session continues |

If the GPU fails in a way that requires the server to restart, `asr` and `llm` errors carry a
fixed "restarting / retry" style message instead of the raw engine error. If the LLM's engine
stalls during the turn (no progress within the server's stall deadlines, CONFIG.md
`stream_first_token_timeout_secs`), the `llm` error says the model stopped responding and was taken
out of service; the model is refused until the server restarts.

Text-to-speech failures, embedding failures, and dropped `inject_history` or `reload_history`
batches send **no** event.

## History and retrieval

### History entry

The same shape is used in `config.history`, `inject_history.entries`,
`reload_history.entries`, and in the `turn_history` event (where `turn_id` is added):

| Field | Type | Required | Meaning |
|---|---|---|---|
| `user` | string | yes | The user's words |
| `assistant` | string | yes | The reply |
| `timestamp` | integer | yes | Unix milliseconds. A good key for client-side storage |
| `embedding` | array of numbers | no | Embedding of `user + "\n" + assistant` |
| `embedding_model` | string | no | Model that produced `embedding`. Send it back unchanged |
| `stt_model` | string | no | Provenance only; never used by the server |
| `llm_model` | string | no | Provenance only; never used by the server |

### Two histories

A session keeps two separate histories. Both live only as long as the connection: the server
stores no history between connections. The client owns the stored history: it keeps the
`turn_history` entries it receives and sends them back (`config.history`, `inject_history`,
`reload_history`) when it reconnects. That is also why `inject_history` and `reload_history`
echo the whole pool as `turn_history`: the client gets every entry with its embedding.

**The rolling LLM context.** These are the last user/assistant exchanges, which are put into
every LLM prompt verbatim.

- Each completed turn is appended to it, and only the last **20** exchanges are kept.
- `config.history` replaces it with the last 20 entries of the supplied array.
- `inject_history` and `reload_history` do not touch it.

**The retrieval pool.** This holds past exchanges that are searched by similarity on each
turn. It is filled by:

- `config.history`: replaces the pool, keeping the newest 200 entries;
- `inject_history`: appends, capped at 200;
- `reload_history`: replaces, capped at 200;
- each completed turn: appended, capped at 200 (the oldest entry is dropped).

Entries supplied through `config.history` are in both histories.

### Retrieval on each turn

If the session has an embedding model in use and the pool is not empty:

1. The user's transcript (not the transcript plus a reply) is embedded.
2. Each pool entry with an `embedding` is scored by cosine similarity, under two conditions:
   - **Provenance gating:** an entry that has an `embedding_model` is scored only if it
     matches the session's embedding model exactly. An entry without `embedding_model` is
     scored only if its vector has the same length as the query's. All other entries score 0.
   - Entries without an `embedding` are never retrieved.
3. Entries scoring **0.5 or more** are kept, and the best **3** are used.
4. They are placed before the system prompt, as a block that starts with
   `[Relevant past context]` and lists each entry as `User: …` / `Assistant: …`.

If embedding the query fails, the turn proceeds without retrieved context.

### Embedding model

- `config.embed_model` only stores the id: it never starts a load. The model must be
  loaded by other means, for example by a preload, an `/v1/embeddings` request, or a
  [`request_model_load`](#request_model_load).
- The session starts with [`default_embed_model`](CONFIG.md) as its embedding model, if that
  is configured. A `config` with `embed_model` changes the session's setting.
- The model is **fixed once** and then kept for the life of the connection, because vectors
  from different models cannot be compared:
  - at connect, if the session's embedding model is loaded;
  - otherwise at the start of the first turn at which it is loaded.

  After that, a different `embed_model` in a later `config` has no effect. To change the
  embedding model, open a new connection and send the history with embeddings from the new
  model (or without embeddings, so the server computes them).
- The session does not keep the embedding model loaded or protected. If it is unloaded, turns
  go on without retrieval and their new entries get no embedding; once the model is loaded
  again, retrieval resumes.
- An entry **needs an embedding** when it has none, or when its `embedding_model` is not the
  session's embedding model (vectors from different models cannot be compared, so retrieval
  skips such an entry). An `embedding` with no `embedding_model` is used as it is.
- Entries that need one are embedded:
  - in `inject_history` and `reload_history` batches, when the session's embedding model is
    fixed and loaded;
  - for each completed turn;
  - in the whole pool, at connect and at the start of every turn. This covers
    `config.history` entries, batches sent before the embedding model was fixed, and
    entries from another model.

  Each entry embedded this way is echoed as a [`turn_history`](#turn_history) with
  `turn_id: null` and the new `embedding`, so the client can store it.

  The texts are embedded in batches of 32. The first turn after a large `config.history`
  therefore starts later by the time it takes to embed the pool (up to 200 entries). If a
  batch fails (the server logs a warning), the entries embedded before it are kept, and the
  pool back-fill is skipped for the next 5 turns.
- An `embedding` with no `embedding_model` and a length different from the session model's
  vectors is never retrieved, and is not re-embedded. Send `embedding_model` with every
  embedding.

## One turn, in order

In the sequences below, "→" is client to server and "←" is server to client. `speech_started`
and `speech_stopped` come from VAD and are sent as soon as VAD fires. When speech overlaps a
running turn, they can therefore appear between that turn's events.

### Normal turn

```
→ (binary audio frames…)
← speech_started
→ (more audio, then ≥ 500 ms of silence)
← speech_stopped
← transcript {text, final: true}
← text {content} …            (repeated)
← <binary audio: sentence 1>  (after the text that completes sentence 1)
← text {content} …
← <binary audio: sentence 2>
   …
← <binary audio: last sentence>
← done {turn_id}
← turn_history {turn_id, user, assistant, …}
```

With no usable TTS model, there are no binary frames; the rest is the same.

### Sentence cap reached (`max_sentences`)

As for a normal turn. After the last allowed sentence's audio, the server sends `done`
immediately, and stops generation. `turn_history` follows with the text generated so far.

### Cancelled with `cancel`

```
← speech_stopped
← transcript {…}              (sent even if cancel arrived during speech-to-text)
← text … / audio …            (whatever was already produced)
→ cancel
← text …                      (possibly a few pieces already generated)
← done {turn_id}
```

No `turn_history` follows: a cancelled turn is not added to the history.

### The user speaks during a reply

```
← text … / audio …            (turn N continues)
← speech_started              (from VAD; does not cancel turn N)
← speech_stopped              (utterance queued as turn N+1)
← … rest of turn N …
← done {turn_id: N}
← transcript {…}              (turn N+1 starts)
   …
```

The server does not interrupt turn N. A client that wants the user's speech to stop the
reply must send `cancel` itself, typically on `speech_started`, and stop its own audio
playback.

### Barge-in phrase turn

If `barge_in_phrase` is set, the server checks each transcript for it. Both texts are
normalised before the check: lower-cased, with punctuation removed and whitespace collapsed.
The phrase must appear as whole words: "stop" matches "Please stop." but not "Stopwatch" or
"it stopped". Words are separated by spaces, so in languages written without spaces between
words (Chinese, Japanese, Thai) the phrase matches only when it is the whole utterance. If the
transcript contains the phrase:

```
← speech_stopped
← <binary audio: a short fixed reply, e.g. "Calming down.">   (only if a TTS model is usable)
← stopped {turn_id}
```

There is no `transcript`, no `text` event, no `done` and no `turn_history`, and the LLM is not
called.

The check runs before the short-transcript filter below, so a phrase of one or two words,
such as "stop", matches when spoken on its own.

One limit applies: the check runs only when this utterance's turn starts. A stop phrase spoken
while a reply is playing is queued, and is acted on only after the current turn has ended, so it
does not stop that reply. To stop the reply itself, send `cancel`. (Planned for 0.9.2: check
utterances committed during a running turn at once, so the phrase interrupts the reply.)

### Empty or short transcript

If the transcript is empty, or has fewer than 3 words and does not contain the barge-in phrase (Whisper often produces short phantom
phrases from noise), the turn is skipped:

```
← speech_stopped
← done {turn_id}
```

There is no `transcript` event. Utterances of one or two words, such as "yes" or "no
thanks", are skipped this way.

### Errors

| Stage | Sequence |
|---|---|
| Device admission | `speech_stopped`, `error {stage: "admission"}`, `done` |
| Speech-to-text | `speech_stopped`, `error {stage: "asr"}`, `done` |
| LLM | `speech_stopped`, `transcript`, possibly some `text` and audio, `error {stage: "llm"}`, `done`. No `turn_history` |
| Text-to-speech | No event. The sentence's audio is missing; text and `done` arrive as normal |
| Turn queue full | `speech_stopped`, `error {stage: "pipeline"}`; no turn, no `done` |

**Does `done` always end a turn?** Almost always. It does not after a barge-in phrase turn,
which ends with `stopped`, and in the rare case described under [`done`](#done).

## Limits

| Limit | Value | What happens |
|---|---|---|
| Utterance length | 29.5 s (472 000 samples) | The utterance is committed at that point and the rest starts a new one |
| LLM output | `max_tokens`, default 120 | Generation stops; the reply may end mid-sentence |
| Sentences spoken | `max_sentences`, default unlimited | The turn ends with `done` after that many sentences were spoken (see below) |
| Rolling LLM context | 20 exchanges | The oldest exchange is dropped |
| Retrieval pool | 200 entries (supplied history) | The oldest entries are dropped; per-turn appends are not capped (see [Two histories](#two-histories)) |
| Retrieved memories | 3 per turn, similarity ≥ 0.5 | — |
| Queued utterances | 4 waiting behind the running turn | Further utterances are discarded; one `error` `pipeline` per backlog episode |
| Queued `inject_history` / `reload_history` batches | 4 of each | Further batches are dropped without notice |
| Idle session | 5 minutes | The session is closed with close code 1000 `idle_timeout`; see [How a session ends](#how-a-session-ends) |

Notes:

- **What a "sentence" is:** the reply text is split for speech only once at least 15
  characters have accumulated. The split falls at a blank line, or at `.`, `!` or `?`
  followed by a space, a newline, or the end of the text received so far. A `.` right after
  a digit at the end of the text received so far is held back, since it may be a decimal
  point.
- **`max_sentences`:**
  - It counts sentences whose audio was actually sent. With no usable TTS model, a value of 1
    or more never takes effect.
  - `0` does not turn speech off (send `"tts_model": ""` for that). The first sentence is synthesized and sent as usual, and
    then the turn ends. With no usable TTS model, the turn ends at the first sentence
    boundary.
- **`max_tokens` with a vision model** may be lowered further by the server, so that the reply
  fits the model's memory.
- **Backlog episode:** after a `pipeline` error, further discarded utterances are not reported
  until an utterance is accepted again.
- **Slow readers:** the server holds at most 64 outgoing messages for a connection. If the
  client stops reading, the turn pipeline waits. When that queue is full, the server also
  stops reading the client's frames, so client pings go unanswered. A client with a ping
  timeout (20 s by default in the Python `websockets` package) then drops the connection.

### Voice-flow pin

The server records a **voice-flow pin**: one shared STT/LLM/TTS model set, meant to let
several clients settle on the same models.

- It is set by the first turn on the server that completes with all three models set. Later
  turns do not change it while it exists. A session with an empty id (speech off, or a slot
  nothing could fill) never sets it.
- It is cleared 5 minutes after the last realtime session closes, or at once when the pinned
  LLM is unloaded.
- The pin is **advisory**:
  - The server does not apply it to sessions.
  - A client that wants the shared set reads it from
    [`GET /v1/admin/voice-pin`](INTERFACE.md) (admin-gated: see
    [Admin API status codes](INTERFACE.md#admin-api-status-codes)) and requests those ids
    in its `config`.
  - Separately, model resolution prefers loaded models that other live sessions already use;
    see [Model resolution](#model-resolution).

## How a session ends

A session ends in one of four ways.

**Idle timeout (5 minutes).**

- **What the timer measures:** the time the session spends waiting for its next committed
  utterance. That is the time between the end of one turn (or the start of the session) and
  the next `speech_stopped` that is accepted into the turn queue.
- **The timer is restarted** whenever that wait begins again:
  - at the end of every turn, whatever its outcome;
  - after every `inject_history` batch (non-empty) or `reload_history` the server processes.
- **The timer does not run** while a turn is being processed, however long the turn takes.
- **These do not reset the timer:** audio frames without detected speech, speech that never
  becomes a committed utterance, `config`, `cancel`, `request_model_load`, an empty
  `inject_history`, and WebSocket pings.

  A client that streams silence, or low-level noise below the VAD threshold, is therefore
  disconnected after 5 minutes.
- **The timeout is fixed** at 5 minutes; it is not a config setting.
- **On timeout** the server sends close code 1000 `idle_timeout` (see
  [Close codes](#close-codes)).

**Client close.** When the client sends a close frame or the connection drops:

- The server stops reading, and cancels the turn in progress.
- Utterances that were already queued are discarded: no turn is started after the close.
- The session is then removed.
- The server answers the client's close frame with the same frame (the WebSocket library
  echoes it) and sends no close frame of its own. A connection that just drops gets nothing.

**Admin force-close.** `DELETE /v1/admin/realtime/sessions/{id}` (see below):

- The turn in progress is cancelled, and its `done` may be sent.
- Queued utterances are discarded.
- The server sends close code 4000 `closed_by_admin`.

**Server shutdown** (SIGTERM, ctrl-C, a restart):

- The turn in progress is cancelled, as for an admin force-close, and queued utterances are
  discarded.
- The server sends close code 1001 `server_stopping`.
- The server waits up to 3 s for its realtime sessions to finish closing before it tears down
  the models and exits.

In every server-initiated case, the session is removed from the admin list. Its models stop
being protected from eviction as soon as it is removed. A pending `model_loading` wait ends
with the session: no `model_loading` event arrives after the close frame.

### Close codes

Every close the server begins sends a WebSocket close frame with a code and a short,
machine-readable reason. The server then waits up to 2 s for the client's close reply, and
drops the connection. Most client libraries reply on their own.

| Cause | Code | Reason |
|---|---|---|
| Idle timeout (5 minutes without a committed utterance) | 1000 | `idle_timeout` |
| Server shutdown or restart | 1001 | `server_stopping` |
| Admin force-close (`DELETE /v1/admin/realtime/sessions/{id}`) | 4000 | `closed_by_admin` |

- **1011 `internal_error` is never sent.** No server path ends a session because of an
  internal error: errors are reported as [`error`](#error) events and the session stays open.
- **A connection reset without a close frame** (1006 in most client libraries) therefore
  means the server process died, or the network failed. It is not an intentional close.
- **A client close** gets only the echo of the client's own frame (see above).

## Examples

### Minimal session (JSON transcript)

`→` client to server, `←` server to client. Binary frames are shown in angle brackets.

```
→ {"type":"config","stt_model":"whisper-base","llm_model":"qwen3-8b","tts_model":"speecht5-tts-ov","language":"en","system_prompt":"You are a concise voice assistant.","max_sentences":3}
← {"type":"model_selection","stt_model":"whisper-base","stt_source":"requested","llm_model":"qwen3-8b","llm_source":"requested","tts_model":"speecht5-tts-ov","tts_source":"requested"}
→ <binary: 16 kHz PCM16 audio, e.g. 20 ms = 640 bytes per frame>
← {"type":"speech_started"}
→ <binary: … the user says "what is the capital of France" …, then silence>
← {"type":"speech_stopped"}
← {"type":"transcript","text":"What is the capital of France?","final":true}
← {"type":"text","content":"The"}
← {"type":"text","content":" capital of France is Paris."}
← <binary: 16 kHz PCM16 audio of "The capital of France is Paris.">
← {"type":"done","turn_id":0}
← {"type":"turn_history","turn_id":0,"user":"What is the capital of France?","assistant":"The capital of France is Paris.","timestamp":1793952000000,"stt_model":"whisper-base","llm_model":"qwen3-8b"}
```

The model ids are examples; use ids registered on your server (`GET /v1/models`). If an
embedding model is in use, `turn_history` also carries `embedding` and `embedding_model`.

### Python client sketch

*Illustrative only:* no reconnect, no audio device handling, no error handling. It uses the
[`websockets`](https://pypi.org/project/websockets/) package (version 14 or later; older
versions name the header argument `extra_headers`).

```python
import asyncio, json, wave
import websockets

URL = "ws://127.0.0.1:8000/v1/realtime"
KEY = "sk-..."          # omit the header if the server has no keys configured
FRAME_BYTES = 640       # 20 ms of 16 kHz mono PCM16

async def main(wav_path: str) -> None:
    headers = {"Authorization": f"Bearer {KEY}"}
    async with websockets.connect(URL, additional_headers=headers, max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "config",
            "stt_model": "whisper-base",
            "llm_model": "qwen3-8b",
            "tts_model": "speecht5-tts-ov",
            "language": "en",
        }))

        async def send_audio() -> None:
            # The WAV must already be 16 kHz, mono, 16-bit.
            with wave.open(wav_path, "rb") as w:
                pcm = w.readframes(w.getnframes())
            pcm += b"\x00" * 16000 * 2          # 1 s of silence so VAD ends the utterance
            for i in range(0, len(pcm), FRAME_BYTES):
                await ws.send(pcm[i:i + FRAME_BYTES])
                await asyncio.sleep(0.02)       # real-time pacing

        sender = asyncio.create_task(send_audio())
        reply_audio = bytearray()
        # When the server closes the session (see Close codes), an idle timeout (1000) or
        # a server stop (1001) ends this loop quietly; an admin kill (4000) raises
        # websockets.ConnectionClosedError, whose .rcvd has the code and reason.
        async for msg in ws:
            if isinstance(msg, bytes):
                reply_audio += msg              # 16 kHz PCM16 mono, one sentence per frame
                continue
            event = json.loads(msg)
            kind = event["type"]
            if kind == "text":
                print(event["content"], end="", flush=True)
            elif kind == "transcript":
                print(f"\nuser: {event['text']}\nassistant: ", end="")
            elif kind == "error":
                print(f"\n[error {event['stage']}] {event['message']}")
            elif kind in ("done", "stopped"):
                print(f"\n[{kind} turn {event['turn_id']}]")
                if sender.done():
                    break
        with wave.open("reply.wav", "wb") as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
            out.writeframes(bytes(reply_audio))

asyncio.run(main("question_16k.wav"))
```

`max_size=None` lifts the library's default 1 MiB limit on incoming messages. That default
equals about 33 s of 16 kHz audio, and one long reply sentence with a large `max_tokens`
could exceed it.

## Admin and observability

### Admin endpoints

All of these are admin-gated. They need an admin key when `admin_api_keys` is configured,
and otherwise fall back to the inference key check. With no keys file at all, they answer
`503`. See [Admin API status codes](INTERFACE.md#admin-api-status-codes) in
[`INTERFACE.md`](INTERFACE.md).

| Method | Route | Returns |
|---|---|---|
| `GET` | `/v1/admin/realtime/sessions` | `{"sessions": [...]}`, one summary per live session: `id` (UUID), `conn_id`, `connected_at` (Unix ms), `stt_model`, `llm_model`, `tts_model`, `turn_count` |
| `GET` | `/v1/admin/realtime/sessions/{id}` | Full snapshot of one session: the fields above plus `tts_voice`, `language`, `system_prompt`, `barge_in_phrase`, `embed_model` and `history_entries` (the retrieval pool, embeddings included). `404 {"error": "session not found"}` for an unknown id |
| `DELETE` | `/v1/admin/realtime/sessions/{id}` | Force-closes the session: `204`, or `404 {"error": "session not found"}` |
| `GET` | `/v1/admin/voice-pin` | `{"pin": {stt_model, llm_model, tts_model} or null, "active_sessions", "ttl_secs"}` |

Notes on the snapshot:

- `turn_count` counts the turns that were added to the history.
- The models shown are the ones the last `config` resolved to.
- `max_tokens` and `max_sentences` are not shown.
- Unset fields of the snapshot are `null`, not omitted. Inside `history_entries`, absent
  optional fields (`embedding`, `embedding_model`, `stt_model`, `llm_model`) are omitted.

The same routes are listed in the endpoint tables of [`INTERFACE.md`](INTERFACE.md). The `rv`
command line wraps them as `rv sessions`, `rv session`, `rv session-kill` and `rv voice-pin`
(see [`README.md`](README.md)).

### Metrics

Realtime metrics are documented in the realtime section of [`METRICS.md`](METRICS.md):

- [`rustedvino_realtime_connections_total`](METRICS.md#realtime-voice)
- [`rustedvino_realtime_turns_total`](METRICS.md#realtime-voice) (label `result`)
- [`rustedvino_realtime_stt_duration_seconds`](METRICS.md#realtime-voice)
- [`rustedvino_realtime_llm_duration_seconds`](METRICS.md#realtime-voice)
- [`rustedvino_realtime_tts_sentences_total`](METRICS.md#realtime-voice) (label `result`)

[`rustedvino_rtf_ratio`](METRICS.md) also covers the STT and TTS work of realtime turns.

On the `result` label of `rustedvino_realtime_turns_total`:

- `barge_in` counts [barge-in phrase turns](#barge-in-phrase-turn).
- Skipped turns (an empty or short transcript) are not counted under any `result`.
