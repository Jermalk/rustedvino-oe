# Third-party notices

RustedVINO itself is licensed under the Apache License 2.0 — see [`LICENSE`](LICENSE).

This file covers the third-party code that ships **inside the pre-built release
bundles** on the [Releases page](https://github.com/Jermalk/rustedvino-oe/releases).
A source checkout of this repository contains none of it; building from source
fetches dependencies from crates.io and links against an OpenVINO runtime you
install yourself.

Not every release publishes a bundle for every platform: the notes below on Windows-only
files (the `.dll` runtime, DirectML) apply only to a release that publishes a Windows bundle.

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
- Exact versions as built: the `[[package]]` entries in [`Cargo.lock`](Cargo.lock) of the release tag

The MPL-2.0 is a file-level copyleft. Symphonia's own source files remain under
the MPL-2.0 and are obtainable from the upstream repository above at the versions
listed; RustedVINO's own source stays under the Apache License 2.0, which the
MPL-2.0 explicitly permits for a larger work (MPL-2.0 §3.3).

---

## Apache License 2.0 — OpenVINO and OpenVINO GenAI

The release bundles ship Intel's **OpenVINO** and **OpenVINO GenAI** runtime
libraries (2026.4.1) so the server runs with no separate OpenVINO installation.
Both are licensed under the **Apache License 2.0**.

- <https://github.com/openvinotoolkit/openvino>
- <https://github.com/openvinotoolkit/openvino.genai>
- <https://github.com/openvinotoolkit/openvino_tokenizers>

The OpenVINO runtime libraries in the bundles (Linux: `lib/`; a Windows bundle, when published: the
`.dll` files beside `rustedvino.exe`) are the unmodified files of Intel's `openvino`, `openvino-genai` and
`openvino-tokenizers` 2026.4.1 Python wheels for that platform, and they bring their own
third-party components, among them **oneTBB** (Apache-2.0), **oneDNN** (Apache-2.0) and **hwloc**
(BSD-3-Clause), each with its copyright notice. Intel's licence files for all of them are in the bundle
under `licenses/openvino/` (look for the `third-party-programs` files).

A Windows bundle is compiled against Intel's Windows OpenVINO GenAI archive (headers and import
libraries), but ships none of its files: that archive's runtime is under Intel's separate
distribution licence, the wheels' identical runtime under Apache-2.0.

---

## MIT License — ONNX Runtime

The server binary of each release bundle statically links Microsoft's **ONNX Runtime** 1.24.2 (through the
`ort` crate), used for the Kokoro, Piper and VITS text-to-speech voices. ONNX Runtime is licensed under
the **MIT License**. Its licence and its third-party notices are in the bundle under
`licenses/onnxruntime/`.

- <https://github.com/microsoft/onnxruntime>

---

## Microsoft Software License Terms — DirectML (Windows bundle only)

A Windows bundle, when published, ships Microsoft's **DirectML** 1.15.4 (`DirectML.dll`, unmodified), which ONNX
Runtime loads on Windows. DirectML is not open source: it is distributed under the **Microsoft
Software License Terms for Microsoft DirectX Machine Learning (DirectML)**, which allow it to be
redistributed with applications that run on Windows. Those terms, the code licence and Microsoft's
third-party notices are in the bundle under `licenses/directml/`; they apply to `DirectML.dll` only.

- <https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4>

## Rust crates

The server and `rv` statically link the Rust crates listed in `Cargo.lock`. The licence files of every
one of them are collected in `licenses/rust-crates-licenses.txt` inside the bundle. Releases from 0.8.4 on
are built from the `Cargo.lock` of the tagged source tree, so the list matches the binary.

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
public-domain-equivalent terms. Symphonia above is the only copyleft component, and
DirectML (Windows bundle only) the only one that is not open source.

The authoritative, per-crate list for any given build is that build's
[`Cargo.lock`](Cargo.lock); each crate's license is declared in its own
`Cargo.toml` and published on <https://crates.io>.
