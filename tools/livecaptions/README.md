# livecaptions

Live captions from a microphone, a call, or a media file, using a RustedVINO server's own
`/v1/audio/transcriptions`. Optionally, each caption is also translated through
`/v1/chat/completions`. Everything runs on your server. No audio leaves the machines you
choose.

```
python3 tools/livecaptions/livecaptions.py --language pl
```

Output is one line per caption, stamped with the time since you started:

```
[00:59.5] Dzień dobry Państwu. To zapraszam do wysłuchania kolejnej rozmowy z cyklu rozmów z Vincentem Sewerskim.
[01:05.4] Dzisiaj to będzie rozmowa szpiegowsko-historyczna.
```

Needs Python 3.8+ and `ffmpeg` on the `PATH`. No Python packages.

## What to caption

- **Your microphone:** the default. Pass `--source <name>` for a different one.
- **A call, or anything else you hear:** pass the `.monitor` of the speaker the call plays
  through. It captions everyone on the call, whatever app the call runs in.
- **A file:** `--file talk.mp4` plays it at real speed, as if it were live. Add `--fast` to
  skip the waiting.

```
python3 tools/livecaptions/livecaptions.py --list-sources
python3 tools/livecaptions/livecaptions.py --source alsa_output.pci-0000_00_1f.3.analog-stereo.monitor --language en
```

Capture uses PulseAudio or PipeWire (through its PulseAudio layer), so it's Linux-first. An
unknown `--source` name is rejected: PipeWire would otherwise quietly fall back to the default
microphone.

**Captioning other people needs their consent.** If you caption a call, tell the others it's being
transcribed, as you would for a recording.

## Translation

```
python3 tools/livecaptions/livecaptions.py --language pl --translate-to English --llm-model <chat-model-id>
```

The original caption appears as soon as it's transcribed. Its translation follows on its own line
when the chat model finishes. Translation runs in a separate thread, so a slow translation never
holds up the next caption. Thinking is switched off (`enable_thinking: false`): translation is a
transformation job, and a reasoning model would otherwise spend its budget thinking. A
translation that still ran out of tokens is marked `[cut off]`.

`--srt out.srt` writes the captions to a subtitle file as they appear. With `--translate-to`,
the file holds the translations. Together with `--file --fast` that gives you translated
subtitles for a video.

## Which model

Model ids are local registration names, so there is no built-in default. livecaptions picks the
speech-to-text model in this order:

1. `--model ID`
2. the `LIVECAPTIONS_MODEL` environment variable
3. the only speech-to-text model *loaded* on `--target`, if there is exactly one. A model set
   to `load: "on_demand"` that isn't loaded yet isn't found this way; name it with `--model`

`--list-models` shows what `--target` offers. `--target` defaults to `http://localhost:11437`.
The translation model has no fallback: pass `--llm-model` or set `LIVECAPTIONS_LLM_MODEL`.

If the target requires auth, pass `--api-key`, set `LIVECAPTIONS_API_KEY`, or add an entry to
`tools/livecaptions/api_keys.json` (gitignored; copy `api_keys.example.json`). The order is
the flag, then the environment variable, then the key file.

**Always pass `--language`.** Whisper's language detection works from the audio it's given, and
a two-second caption is too little to go on.

## How it cuts speech into captions

Whisper transcribes up to 30 seconds at a time, and a caption is only useful if it arrives soon
after it's spoken. So livecaptions sends one request per utterance and finds the utterances
itself. ffmpeg delivers 16 kHz mono audio, and a voice activity detector in plain Python
watches its energy in 20 ms frames:

- **Speech starts** after 100 ms above the threshold. 300 ms before that are kept, so the first
  syllable isn't lost.
- **Speech ends** after 500 ms below it. These values are the same as the server's own
  realtime VAD.
- **Past 12 seconds** (`--soft-seconds`), a 300 ms pause is enough, so someone who never stops
  is cut at a breath.
- **At 25 seconds** (`--max-seconds`), the cut is forced.

The threshold adapts. `--threshold` (default 0.015) suits a microphone in a quiet room. A
podcast or a video is mastered much louder: in one we measured, even the quietest 5% of the
audio was above 0.015, so no pause was ever found and every caption ran the full 25 seconds.
The threshold therefore follows the recent noise floor: 2.5 × the quietest 10% of the last 10
seconds, never lower than `--threshold`.

Every utterance is sent with 300 ms of silence on each end. In testing, one chunk of clear
speech that started with a little room noise came back as a single word, every time. The same
chunk with silence in front was transcribed in full.

## What to expect

Measured with `whisper-large-v3-turbo` on an Intel Core Ultra (Lunar Lake) NPU, playing files at
real speed:

- **Speed:** captions appear about half a second after the speaker stops (median 0.47 to
  0.66 s, max 1.15 s). That includes the 500 ms of silence that ends an utterance.
- **Translation:** with an 8B chat model on the same laptop, translations followed 1.5 to 7
  seconds after the speech.
- **Clean speech** (an audiobook) transcribed nearly word for word.
- **Weak spots:**
  - fast-cut audio over music loses context, since each piece is short;
  - a phrase cut mid-way now and then gets an ending Whisper made up;
  - names are the usual trouble ("Aspinwall" came out as "Espinwall").

If your speakers pause rarely and you see invented endings, raise `--soft-seconds`. If captions
feel slow, lower it.

## Why Python, not Rust

RustedVINO is Rust, but livecaptions is a client: it should run cold, with no build step. Its whole
job is to read audio from ffmpeg, find the pauses, and make HTTP requests. The server's
`/v1/realtime` WebSocket also has a VAD and Whisper, but it always goes on to answer with its
LLM, so it isn't a captioning endpoint.
