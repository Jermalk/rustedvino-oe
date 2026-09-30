// ============================================================
// src/pipelines/piper.rs — Piper VITS voices: config, phonemizer, ids
// ============================================================
// Piper voices (rhasspy/piper-voices) are VITS models in ONNX that take
// *phonemes* as input — produced by espeak-ng. espeak-ng is GPL-3.0, which is
// why the project never links it (`espeak-rs`, considered earlier, statically
// linked it). Here it runs as an **external program** the operator installs
// (`espeak-ng` on PATH): RustedVINO only reads its stdout, like any other
// command-line tool, and never ships or links it.
//
// The command-line espeak-ng is not byte-identical to the patched library Piper
// itself uses, so this module (1) mirrors Piper's own pipeline — clause by
// clause, terminator kept, `(lang)` flags removed, NFD codepoints as phonemes,
// `^ _ p1 _ p2 _ … $` ids — (2) applies the measured per-language differences
// (Vietnamese: expand digits ourselves, drop the tone digits `1`/`7` the CLI
// adds), and (3) proves it at load with a self-check against Piper's own
// output for a reference sentence. A mismatch refuses the load: an espeak-ng
// version that phonemizes differently would otherwise mispronounce silently.
// ============================================================

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

/// Piper's padding / beginning / end-of-sentence symbols (`piper/const.py`).
const PAD: &str = "_";
const BOS: &str = "^";
const EOS: &str = "$";

/// A Piper voice's `<name>.onnx.json` — only the fields synthesis needs.
#[derive(Debug, Clone, Deserialize)]
pub struct PiperConfig {
    /// Output sample rate.
    pub audio: PiperAudio,
    /// espeak-ng voice for phonemization (e.g. `"vi"`).
    pub espeak: PiperEspeak,
    /// Default synthesis scales.
    pub inference: PiperInference,
    /// Speakers in the model; >1 needs a `sid` input.
    #[serde(default)]
    pub num_speakers: u32,
    /// `"espeak"` for espeak-phonemized voices (the only kind supported).
    #[serde(default)]
    pub phoneme_type: Option<String>,
    /// Phoneme (one codepoint) → id(s).
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
}

/// `audio` block of a Piper config.
#[derive(Debug, Clone, Deserialize)]
pub struct PiperAudio {
    /// Hz.
    pub sample_rate: u32,
}

/// `espeak` block of a Piper config.
#[derive(Debug, Clone, Deserialize)]
pub struct PiperEspeak {
    /// espeak-ng voice name.
    pub voice: String,
}

/// `inference` block of a Piper config.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct PiperInference {
    /// VITS noise scale.
    pub noise_scale: f32,
    /// Duration scale (>1 = slower).
    pub length_scale: f32,
    /// Duration-predictor noise.
    pub noise_w: f32,
}

/// Find a Piper voice in `dir`: a `<name>.onnx.json` next to `<name>.onnx`,
/// with an espeak phoneme type. Returns `(onnx, config)` paths.
#[must_use]
pub fn find_piper_voice(dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".onnx.json"))
        .collect();
    entries.sort();
    entries.into_iter().find_map(|json| {
        let onnx = PathBuf::from(json.to_string_lossy().trim_end_matches(".json"));
        let cfg: PiperConfig = serde_json::from_str(&std::fs::read_to_string(&json).ok()?).ok()?;
        let espeak = cfg.phoneme_type.as_deref().is_none_or(|t| t == "espeak");
        (onnx.is_file() && espeak).then_some((onnx, json))
    })
}

/// Load a Piper config.
///
/// # Errors
/// Unreadable or malformed JSON.
pub fn load_config(path: &Path) -> anyhow::Result<PiperConfig> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing Piper config {}", path.display()))
}

// ── Phonemization ────────────────────────────────────────────────────────────

/// Runs espeak-ng on one clause and returns its IPA line(s). Injected so the
/// pipeline can be unit-tested without espeak-ng installed.
pub type EspeakFn = dyn Fn(&str, &str) -> anyhow::Result<String> + Send + Sync;

/// The real espeak-ng: `espeak-ng -v <voice> -q --ipa` with the clause on
/// stdin (never a shell — the text is data, not a command line).
///
/// # Errors
/// espeak-ng missing from PATH, or exiting non-zero.
pub fn espeak_cli(voice: &str, text: &str) -> anyhow::Result<String> {
    use std::io::Write;
    let mut child = std::process::Command::new("espeak-ng")
        .args(["-v", voice, "-q", "--ipa"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context(
            "running espeak-ng — Piper voices need it installed and on PATH \
             (e.g. `apt install espeak-ng`); it runs as a separate program, never linked",
        )?;
    child
        .stdin
        .take()
        .context("espeak-ng stdin")?
        .write_all(text.as_bytes())
        .context("writing to espeak-ng")?;
    let out = child.wait_with_output().context("waiting for espeak-ng")?;
    anyhow::ensure!(
        out.status.success(),
        "espeak-ng failed: {}",
        String::from_utf8_lossy(&out.stderr).trim()
    );
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Text → phonemes grouped by sentence (each phoneme one NFD codepoint, as a
/// string), mirroring `piper/phonemize_espeak.py`.
///
/// # Errors
/// espeak-ng errors.
pub fn phonemize(voice: &str, text: &str, espeak: &EspeakFn) -> anyhow::Result<Vec<Vec<String>>> {
    use unicode_normalization::UnicodeNormalization;

    let text = if voice == "vi" {
        expand_numbers_vi(text)
    } else {
        text.to_owned()
    };
    let mut sentences = Vec::new();
    let mut current = String::new();
    for (clause, terminator) in split_clauses(&text) {
        let mut ph = String::new();
        if !clause.trim().is_empty() {
            let raw = espeak(voice, clause.trim())?;
            ph = clean_ipa(voice, &raw);
        }
        if let Some(t) = terminator {
            ph.push(t);
            if matches!(t, ',' | ':' | ';') {
                ph.push(' ');
            }
        }
        current.push_str(&ph);
        if matches!(terminator, Some('.' | '!' | '?')) {
            sentences.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current);
    }
    Ok(sentences
        .into_iter()
        .map(|s| s.trim().nfd().map(String::from).collect())
        .filter(|s: &Vec<String>| !s.is_empty())
        .collect())
}

/// Split text into `(clause, terminator)` on `, . ; : ! ?`.
fn split_clauses(text: &str) -> Vec<(String, Option<char>)> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if matches!(ch, ',' | '.' | ';' | ':' | '!' | '?') {
            out.push((std::mem::take(&mut cur), Some(ch)));
        } else {
            cur.push(ch);
        }
    }
    if !cur.trim().is_empty() {
        out.push((cur, None));
    }
    out
}

/// Normalize the CLI's IPA toward what Piper's own espeak build emits: one
/// line, single spaces, `(lang)` switch flags removed, and — for Vietnamese —
/// the tone digits `1`/`7` the CLI adds (measured on the reference sentences).
fn clean_ipa(voice: &str, raw: &str) -> String {
    let mut s = String::with_capacity(raw.len());
    let mut depth = 0u32;
    for ch in raw.chars() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            c if depth == 0 && !(voice == "vi" && matches!(c, '1' | '7')) => s.push(c),
            _ => {}
        }
    }
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Phonemes → ids exactly as `piper/phoneme_ids.py`: `^ _` then each
/// phoneme's id(s) followed by `_`, then `$`. Unknown phonemes are skipped.
#[must_use]
pub fn phonemes_to_ids<S: std::hash::BuildHasher>(
    phonemes: &[String],
    map: &HashMap<String, Vec<i64>, S>,
) -> Vec<i64> {
    let get = |k: &str| map.get(k).cloned().unwrap_or_default();
    let pad = get(PAD);
    let mut ids = get(BOS);
    ids.extend(&pad);
    for p in phonemes {
        if let Some(v) = map.get(p) {
            ids.extend(v);
            ids.extend(&pad);
        }
    }
    ids.extend(get(EOS));
    ids
}

// ── Vietnamese numbers ───────────────────────────────────────────────────────

const VI_DIGITS: [&str; 10] = [
    "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín",
];

/// Replace each run of ASCII digits with Vietnamese words ("8 giờ 30" →
/// "tám giờ ba mươi"). Done here rather than by espeak-ng because the CLI's
/// own expansion keeps tone digits that plain words don't, which the tone-digit
/// cleanup would then strip wrongly. Runs longer than 12 digits are read digit
/// by digit (phone numbers, codes).
#[must_use]
pub fn expand_numbers_vi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut digits = String::new();
    let flush = |digits: &mut String, out: &mut String| {
        if digits.is_empty() {
            return;
        }
        let words = if digits.len() > 12 {
            digits
                .bytes()
                .map(|b| VI_DIGITS[usize::from(b - b'0')])
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            digits
                .parse::<u64>()
                .map_or_else(|_| digits.clone(), vi_number)
        };
        out.push_str(&words);
        digits.clear();
    };
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
        } else {
            flush(&mut digits, &mut out);
            out.push(ch);
        }
    }
    flush(&mut digits, &mut out);
    out
}

/// Read a number below 10¹² in Vietnamese (standard northern forms: `mươi`,
/// `mốt`, `lăm`, `linh`, `nghìn`, `triệu`, `tỷ`). Larger values are read digit
/// by digit by [`expand_numbers_vi`] before reaching here.
#[must_use]
pub fn vi_number(n: u64) -> String {
    const SCALES: [&str; 4] = ["", "nghìn", "triệu", "tỷ"];
    if n == 0 {
        return VI_DIGITS[0].to_owned();
    }
    let mut groups = Vec::new();
    let mut m = n;
    while m > 0 {
        groups.push(m % 1000);
        m /= 1000;
    }
    let top = groups.len() - 1;
    let mut words: Vec<String> = Vec::new();
    for (i, &g) in groups.iter().enumerate().rev() {
        if g == 0 {
            continue;
        }
        words.push(vi_triple(g, i < top));
        let scale = SCALES.get(i).copied().unwrap_or("");
        if !scale.is_empty() {
            words.push(scale.to_owned());
        }
    }
    words.join(" ")
}

/// Read 1–999; `inner` = not the leading group, so a missing hundreds digit is
/// read as "không trăm" (2026 → "hai nghìn không trăm hai mươi sáu").
fn vi_triple(n: u64, inner: bool) -> String {
    let digit = |d: u64| VI_DIGITS[usize::try_from(d).unwrap_or(0)];
    let (hundreds, tens, units) = (n / 100, (n / 10) % 10, n % 10);
    let mut words: Vec<&str> = Vec::new();
    if hundreds > 0 || inner {
        words.push(digit(hundreds));
        words.push("trăm");
    }
    match tens {
        0 if units > 0 && (hundreds > 0 || inner) => words.push("linh"),
        0 => {}
        1 => words.push("mười"),
        _ => {
            words.push(digit(tens));
            words.push("mươi");
        }
    }
    match units {
        0 => {}
        1 if tens >= 2 => words.push("mốt"),
        5 if tens >= 1 => words.push("lăm"),
        _ => words.push(digit(units)),
    }
    words.join(" ")
}

// ── Self-check ───────────────────────────────────────────────────────────────

/// Reference phonemization per espeak voice: `(sentence, Piper's own output)`,
/// captured with Piper's Python package (`PiperVoice.phonemize`) for the
/// `vi_VN` vais1000 voice on 2026-09-29.
const SELF_CHECK: &[(&str, &str, &str)] = &[(
    "vi",
    "Xin chào các bạn, hôm nay trời Hà Nội nắng đẹp và rất dễ chịu.",
    "sˈin tʃˈaː2w kˌaːɜc bˈaː6n, hˈom nˈaj tʃˈəː2j hˈaː2 nˈo6j nˈaɜŋ ɗˈɛ6p vˌaː2 zˈəɜt̪ zˈe5 tʃˈi6w.",
)];

/// Prove the installed espeak-ng + this module reproduce Piper's phonemes for
/// the voice's reference sentence. `Ok(false)` when no reference exists for
/// the voice (the caller warns); `Err` on a mismatch or a missing espeak-ng.
///
/// # Errors
/// espeak-ng missing/failing, or a phonemization mismatch (names both strings).
pub fn self_check(voice: &str, espeak: &EspeakFn) -> anyhow::Result<bool> {
    use unicode_normalization::UnicodeNormalization;
    let Some(&(_, sentence, expected)) = SELF_CHECK.iter().find(|(v, _, _)| *v == voice) else {
        return Ok(false);
    };
    let got: String = phonemize(voice, sentence, espeak)?.concat().concat();
    let want: String = expected.nfd().collect();
    anyhow::ensure!(
        got == want,
        "espeak-ng self-check failed for voice '{voice}': this espeak-ng build phonemizes the \
         reference sentence differently from the one the voice was trained with, so speech would \
         be mispronounced.\n  expected: {want}\n  got:      {got}"
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn vietnamese_numbers_read_correctly() {
        for (n, w) in [
            (0, "không"),
            (5, "năm"),
            (10, "mười"),
            (15, "mười lăm"),
            (21, "hai mươi mốt"),
            (25, "hai mươi lăm"),
            (30, "ba mươi"),
            (105, "một trăm linh năm"),
            (110, "một trăm mười"),
            (2026, "hai nghìn không trăm hai mươi sáu"),
            (1_000_000, "một triệu"),
            (3_000_005, "ba triệu không trăm linh năm"),
        ] {
            assert_eq!(vi_number(n), w, "{n}");
        }
        assert_eq!(expand_numbers_vi("lúc 8 giờ 30."), "lúc tám giờ ba mươi.");
    }

    /// The pipeline with a canned espeak: clause split, terminator + space,
    /// `(en)` flags and Vietnamese tone digits 1/7 removed, sentences split on
    /// `. ! ?`, NFD codepoints.
    #[test]
    fn phonemize_mirrors_piper_clause_handling() {
        let fake = |_: &str, clause: &str| -> anyhow::Result<String> {
            Ok(match clause {
                "Bạn có khỏe không" => " bˈaː6n kˈɔɜ xwˈɛ4 xˌo1ŋ\n".to_owned(),
                "Tôi khỏe" => "t̪ˈo1j xwˈɛ4".to_owned(),
                "cảm ơn" => "kˈaː4m (en)ˈəː7n(vi)".to_owned(),
                other => panic!("unexpected clause {other}"),
            })
        };
        let s = phonemize("vi", "Bạn có khỏe không? Tôi khỏe, cảm ơn!", &fake).unwrap();
        let joined: Vec<String> = s.iter().map(|p| p.concat()).collect();
        let nfd = |x: &str| {
            use unicode_normalization::UnicodeNormalization;
            x.nfd().collect::<String>()
        };
        assert_eq!(
            joined,
            vec![
                nfd("bˈaː6n kˈɔɜ xwˈɛ4 xˌoŋ?"),
                nfd("t̪ˈoj xwˈɛ4, kˈaː4m ˈəːn!")
            ]
        );
    }

    #[test]
    fn phonemes_to_ids_matches_piper_layout() {
        let map: HashMap<String, Vec<i64>> = [("_", 0), ("^", 1), ("$", 2), ("a", 14), ("b", 15)]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), vec![v]))
            .collect();
        let ph: Vec<String> = ["a", "?", "b"].iter().map(|s| (*s).to_owned()).collect();
        // "?" is unknown → skipped, like Piper.
        assert_eq!(phonemes_to_ids(&ph, &map), vec![1, 0, 14, 0, 15, 0, 2]);
    }

    #[test]
    fn self_check_passes_and_fails_on_canned_output() {
        let good = |_: &str, c: &str| -> anyhow::Result<String> {
            Ok(match c {
                "Xin chào các bạn" => "sˈi1n tʃˈaː2w kˌaːɜc bˈaː6n".to_owned(),
                _ => {
                    "hˈo1m nˈa1j tʃˈəː2j hˈaː2 nˈo6j nˈaɜŋ ɗˈɛ6p vˌaː2 zˈəɜt̪ zˈe5 tʃˈi6w".to_owned()
                }
            })
        };
        assert!(self_check("vi", &good).unwrap());
        let bad = |_: &str, _: &str| -> anyhow::Result<String> { Ok("xyz".to_owned()) };
        assert!(self_check("vi", &bad).is_err());
        assert!(
            !self_check("xx", &good).unwrap(),
            "no reference for this voice"
        );
    }

    /// Parity with Piper's own ids for every reference sentence, using the
    /// installed espeak-ng. Needs espeak-ng and the voice config, so it runs
    /// only when asked: `RV_PIPER_REF=<piper_reference.json> RV_PIPER_CFG=<voice.onnx.json>
    /// cargo test piper_parity -- --ignored`.
    #[test]
    #[ignore = "needs espeak-ng and a Piper voice; run with --ignored"]
    fn piper_parity_with_reference_ids() {
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("RV_PIPER_REF").unwrap()).unwrap(),
        )
        .unwrap();
        let cfg = load_config(Path::new(&std::env::var("RV_PIPER_CFG").unwrap())).unwrap();
        for case in reference.as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let want: Vec<Vec<i64>> = serde_json::from_value(case["ids"].clone()).unwrap();
            let got: Vec<Vec<i64>> = phonemize(&cfg.espeak.voice, text, &espeak_cli)
                .unwrap()
                .iter()
                .map(|p| phonemes_to_ids(p, &cfg.phoneme_id_map))
                .collect();
            assert_eq!(got, want, "{text}");
        }
    }
}
