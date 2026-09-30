# Third-party notices

RustedVINO itself is licensed under the Apache License 2.0 — see [`LICENSE`](LICENSE).

This file covers the third-party code that ships **inside the pre-built release
bundles** on the [Releases page](https://github.com/Jermalk/rustedvino-oe/releases).
A source checkout of this repository contains none of it; building from source
fetches dependencies from crates.io and links against an OpenVINO runtime you
install yourself.

---

## Mozilla Public License 2.0 — Symphonia

The release binary statically links **Symphonia** 0.5.5, used to decode uploaded
audio for the speech-to-text endpoints. Symphonia is licensed under the
**Mozilla Public License 2.0**.

Crates linked (all version 0.5.5):

```
symphonia                symphonia-codec-alac      symphonia-format-mkv
symphonia-bundle-flac    symphonia-codec-pcm       symphonia-format-ogg
symphonia-bundle-mp3     symphonia-codec-vorbis    symphonia-format-riff
symphonia-codec-aac      symphonia-core            symphonia-metadata
symphonia-codec-adpcm    symphonia-format-isomp4   symphonia-utils-xiph
```

- Upstream source: <https://github.com/pdeljanov/Symphonia>
- License text: <https://www.mozilla.org/en-US/MPL/2.0/>
- Exact versions as built: the `[[package]]` entries in [`Cargo.lock`](Cargo.lock)

The MPL-2.0 is a file-level copyleft. Symphonia's own source files remain under
the MPL-2.0 and are obtainable from the upstream repository above at the versions
listed; RustedVINO's own source stays under the Apache License 2.0, which the
MPL-2.0 explicitly permits for a larger work (MPL-2.0 §3.3).

---

## Apache License 2.0 — OpenVINO and OpenVINO GenAI

The release bundles ship Intel's **OpenVINO** and **OpenVINO GenAI** runtime
libraries (2026.2.1) so the server runs with no separate OpenVINO installation.
Both are licensed under the **Apache License 2.0**.

- <https://github.com/openvinotoolkit/openvino>
- <https://github.com/openvinotoolkit/openvino.genai>
- <https://github.com/openvinotoolkit/openvino_tokenizers>

---

## Not included — espeak-ng

Piper TTS voices need **espeak-ng** (GPL-3.0) to turn text into phonemes. RustedVINO
neither links nor ships it: the server runs the `espeak-ng` program you install yourself
(e.g. `apt install espeak-ng`) as a separate process and reads its output. Without it,
Piper voices refuse to load; every other feature works.

---

## Everything else

The remaining Rust dependencies are permissively licensed — predominantly
`MIT OR Apache-2.0`, with some MIT, Apache-2.0, BSD, ISC, Zlib, Unicode-3.0 and
public-domain-equivalent terms. Symphonia above is the only copyleft component.

The authoritative, per-crate list for any given build is that build's
[`Cargo.lock`](Cargo.lock); each crate's license is declared in its own
`Cargo.toml` and published on <https://crates.io>.
