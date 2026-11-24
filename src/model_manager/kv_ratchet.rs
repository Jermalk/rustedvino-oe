// ============================================================
// src/model_manager/kv_ratchet.rs — learned KV-capacity ceiling, version 2
// ============================================================
// The v1 ratchet (`template.rs`, `.rustedvino_kv_ratchet.json`) stores one number scaled
// by pool size. v2 stores the observation together with everything that determines the
// capacity it was observed under, and is used only when every part still matches: a
// ceiling learned under another precision, pool, prefix setting, engine version or
// device says nothing about this load. No scaling.
//
// The v1 file is never read here, never deleted (the model directory may be a shared
// read-only drive that older builds still write to), and noted once per model per process.
// A different file name means a 0.9.0-pre binary ignores v2 and a v2 build ignores v1.
// ============================================================

use std::path::Path;

/// File name of the version-2 ratchet, in the model's ratchet directory (see
/// `ModelManager::ratchet_dir`; before 2026-10-28 it was written into the model directory).
pub(crate) const RATCHET_V2_FILENAME: &str = ".rustedvino_kv_ratchet.v2.json";
/// File name of the version-1 ratchet (legacy gate; read and written by `template.rs` in the
/// model's ratchet directory; only noted here when found in the model library).
const RATCHET_V1_FILENAME: &str = ".rustedvino_kv_ratchet.json";
const RATCHET_VERSION: u32 = 2;

/// Everything that fixes the capacity an observation was made under.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct RatchetKey {
    /// KV blocks of the pool (`num_kv_blocks`).
    pub kv_pool_blocks: u64,
    /// The precision string the engine was built with (`i8`, `u4`, `f16`; `""` = plugin default).
    pub kv_precision: String,
    /// Prefix caching on or off.
    pub prefix_caching: bool,
    /// `max_num_seqs` of the scheduler.
    pub max_num_seqs: usize,
    /// `cb` or `vlm`.
    pub engine_path: String,
    /// Normalised `OpenVINO` version (`kv_profile::engine_pair`).
    pub openvino_version: String,
    /// Normalised `GenAI` version.
    pub genai_version: String,
    /// Device the engine ran on.
    pub device: String,
    /// The attached draft model (speculative decoding), `""` for none: `GenAI` splits the pool
    /// between main and draft, so a ceiling learned with one says nothing about another.
    #[serde(default)]
    pub draft_model: String,
}

impl RatchetKey {
    /// Build a key, normalising both engine versions the same way the capacity model does.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        kv_pool_blocks: u64,
        kv_precision: &str,
        prefix_caching: bool,
        max_num_seqs: usize,
        engine_path: &str,
        engine_versions: (&str, &str),
        device: &str,
        draft_model: &str,
    ) -> Self {
        let (openvino_version, genai_version) =
            super::kv_profile::engine_pair(engine_versions.0, engine_versions.1);
        Self {
            kv_pool_blocks,
            kv_precision: kv_precision.to_owned(),
            prefix_caching,
            max_num_seqs,
            engine_path: engine_path.to_owned(),
            openvino_version,
            genai_version,
            device: device.to_owned(),
            draft_model: draft_model.to_owned(),
        }
    }
}

/// One learned ceiling and the engine setup it was learned on.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct RatchetEntry {
    /// Prompt tokens observed at the moment the wedge tripped (already discounted by the
    /// caller's safety margin).
    observed_prompt_tokens: usize,
    #[serde(flatten)]
    key: RatchetKey,
}

/// The file: one entry per distinct key, so loading a model with another precision, pool or
/// stream count and going back does not forget what the first setup learned.
#[derive(serde::Serialize, serde::Deserialize)]
struct RatchetFile {
    version: u32,
    entries: Vec<RatchetEntry>,
}

/// The first single-entry layout of this file (pre-release builds), still read.
#[derive(serde::Serialize, serde::Deserialize)]
struct RatchetV2 {
    version: u32,
    observed_prompt_tokens: usize,
    #[serde(flatten)]
    key: RatchetKey,
}

/// Most entries kept per model directory; the oldest go first.
const MAX_RATCHET_ENTRIES: usize = 16;

/// The entries of the file in `model_dir` (empty when absent, unreadable, corrupt or another
/// version).
fn load_entries(model_dir: &Path) -> Vec<RatchetEntry> {
    let Ok(raw) = std::fs::read_to_string(model_dir.join(RATCHET_V2_FILENAME)) else {
        return Vec::new();
    };
    if let Ok(file) = serde_json::from_str::<RatchetFile>(&raw) {
        return if file.version == RATCHET_VERSION {
            file.entries
        } else {
            Vec::new()
        };
    }
    match serde_json::from_str::<RatchetV2>(&raw) {
        Ok(old) if old.version == RATCHET_VERSION => vec![RatchetEntry {
            observed_prompt_tokens: old.observed_prompt_tokens,
            key: old.key,
        }],
        _ => Vec::new(),
    }
}

/// The learned ceiling for `model_dir` applied to a pool described by `key`, or `None`.
///
/// Every part of the key except the pool size must match exactly. The pool size may be equal or
/// smaller than the one the ceiling was learned on: the recovery reload after a wedge can build
/// a slightly smaller pool (the learned weight footprint is charged from then on), and a ceiling
/// that was too loose for the bigger pool is scaled down by the ratio of blocks, as the v1
/// ratchet scaled by pool size. A larger pool gets nothing: the ceiling says nothing about it.
/// Absent, unreadable, corrupt, another version and zero tokens are `None` too. When several
/// entries apply, the strictest wins.
pub(crate) fn read_ratchet_v2(model_dir: &Path, key: &RatchetKey) -> Option<usize> {
    if key.kv_pool_blocks == 0 {
        return None;
    }
    load_entries(model_dir)
        .into_iter()
        .filter_map(|entry| {
            let same_but_pool = RatchetKey {
                kv_pool_blocks: entry.key.kv_pool_blocks,
                ..key.clone()
            };
            if entry.observed_prompt_tokens == 0
                || entry.key != same_but_pool
                || key.kv_pool_blocks > entry.key.kv_pool_blocks
            {
                return None;
            }
            let scaled = u128::try_from(entry.observed_prompt_tokens).ok()?
                * u128::from(key.kv_pool_blocks)
                / u128::from(entry.key.kv_pool_blocks);
            usize::try_from(scaled).ok().filter(|t| *t > 0)
        })
        .min()
}

/// Persist a learned ceiling (temp file + rename, so a crash never leaves a truncated file).
/// Best effort: logs and returns on any I/O error; the in-memory ratchet still protects this
/// process.
pub(crate) fn write_ratchet_v2(model_dir: &Path, observed_prompt_tokens: usize, key: &RatchetKey) {
    let mut entries = load_entries(model_dir);
    // A ceiling only ever tightens: two instances sharing a model directory, or one instance that
    // wedges twice, must not raise what the stricter observation learned for this exact key.
    if entries.iter().any(|e| {
        e.key == *key
            && e.observed_prompt_tokens > 0
            && e.observed_prompt_tokens <= observed_prompt_tokens
    }) {
        return;
    }
    entries.retain(|e| e.key != *key);
    entries.push(RatchetEntry {
        observed_prompt_tokens,
        key: key.clone(),
    });
    if entries.len() > MAX_RATCHET_ENTRIES {
        entries.drain(..entries.len() - MAX_RATCHET_ENTRIES);
    }
    let file = RatchetFile {
        version: RATCHET_VERSION,
        entries,
    };
    let Ok(serialized) = serde_json::to_string_pretty(&file) else {
        tracing::warn!(model_dir = %model_dir.display(), "kv capacity ratchet v2: could not serialize, skipping write");
        return;
    };
    let final_path = model_dir.join(RATCHET_V2_FILENAME);
    // Per-process temp name: two instances sharing a model directory must not clobber each
    // other's half-written file.
    let tmp_path = model_dir.join(format!("{RATCHET_V2_FILENAME}.{}.tmp", std::process::id()));
    if let Err((stage, err)) =
        write_durably(model_dir, &tmp_path, &final_path, serialized.as_bytes())
    {
        tracing::warn!(model_dir = %model_dir.display(), error = %err, stage, "kv capacity ratchet v2: write failed, skipping");
    }
}

/// Durable replace of `final_path` with `bytes` through `tmp_path` (both in `dir`), shared by the
/// v1 and v2 ratchet writers (R-10): write the temp file and `sync_all` it, rename it over the
/// final file, then sync `dir` so the rename survives a power loss. The directory sync is best
/// effort (some platforms and filesystems cannot open or sync a directory, Windows among them):
/// a failure logs at debug and the write still counts. On a temp-write, sync or rename error the
/// temp file is removed (best effort) and the error is returned with the stage that failed.
pub(crate) fn write_durably(
    dir: &Path,
    tmp_path: &Path,
    final_path: &Path,
    bytes: &[u8],
) -> Result<(), (&'static str, std::io::Error)> {
    let staged = write_synced(tmp_path, bytes)
        .map_err(|e| ("write temp file", e))
        .and_then(|()| std::fs::rename(tmp_path, final_path).map_err(|e| ("commit", e)));
    if staged.is_err() {
        let _ = std::fs::remove_file(tmp_path); // best effort: it may not exist
        return staged;
    }
    if let Err(err) = std::fs::File::open(dir).and_then(|d| d.sync_all()) {
        tracing::debug!(dir = %dir.display(), error = %err, "kv capacity ratchet: could not sync the directory after the rename");
    }
    Ok(())
}

/// Write `bytes` to `path` and `sync_all` it before returning (R-10): the rename that follows
/// must never publish a file whose contents a power loss could still drop.
fn write_synced(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;
    let mut file = std::fs::File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Note once per model directory per process that ratchet files found in the model library
/// (v1 or v2, written there by earlier builds) are ignored: learned ceilings now live in
/// `ratchet_dir`. Returns whether this call logged. The files are left in place (the library
/// may be shared with older builds, or read-only).
pub(crate) fn note_library_ratchets_ignored(model_dir: &Path, ratchet_dir: Option<&Path>) -> bool {
    static SEEN: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashSet<std::path::PathBuf>>,
    > = std::sync::OnceLock::new();
    let found: Vec<&str> = [RATCHET_V1_FILENAME, RATCHET_V2_FILENAME]
        .into_iter()
        .filter(|f| model_dir.join(f).is_file())
        .collect();
    if found.is_empty() {
        return false;
    }
    let first = SEEN
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(model_dir.to_path_buf());
    if first {
        tracing::info!(
            model_dir = %model_dir.display(),
            files = ?found,
            ratchet_dir = %ratchet_dir.map_or_else(|| "(none)".into(), |d| d.display().to_string()),
            "KV capacity ratchet files in the model library are ignored; learned limits now live \
             in the ratchet directory, a new one is learned on the next wedge (the old files are \
             left in place)"
        );
    }
    first
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    fn key() -> RatchetKey {
        RatchetKey {
            kv_pool_blocks: 12_705,
            kv_precision: "u4".into(),
            prefix_caching: false,
            max_num_seqs: 1,
            engine_path: "vlm".into(),
            openvino_version: "2026.4.1".into(),
            genai_version: "2026.4.1".into(),
            device: "GPU.1".into(),
            draft_model: String::new(),
        }
    }

    #[test]
    fn absent_file_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_ratchet_v2(dir.path(), &key()), None);
    }

    #[test]
    fn round_trip_with_the_same_key() {
        let dir = tempfile::tempdir().unwrap();
        write_ratchet_v2(dir.path(), 4_437, &key());
        assert_eq!(read_ratchet_v2(dir.path(), &key()), Some(4_437));
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "no temp file left behind");
    }

    #[test]
    fn any_one_field_different_gives_none() {
        let dir = tempfile::tempdir().unwrap();
        write_ratchet_v2(dir.path(), 4_437, &key());
        let variants: Vec<(&str, RatchetKey)> = vec![
            (
                "bigger pool",
                RatchetKey {
                    kv_pool_blocks: 12_706,
                    ..key()
                },
            ),
            (
                "precision",
                RatchetKey {
                    kv_precision: "i8".into(),
                    ..key()
                },
            ),
            (
                "prefix",
                RatchetKey {
                    prefix_caching: true,
                    ..key()
                },
            ),
            (
                "seqs",
                RatchetKey {
                    max_num_seqs: 4,
                    ..key()
                },
            ),
            (
                "path",
                RatchetKey {
                    engine_path: "cb".into(),
                    ..key()
                },
            ),
            (
                "openvino",
                RatchetKey {
                    openvino_version: "2026.5.0".into(),
                    ..key()
                },
            ),
            (
                "genai",
                RatchetKey {
                    genai_version: "2026.5.0".into(),
                    ..key()
                },
            ),
            (
                "device",
                RatchetKey {
                    device: "GPU.0".into(),
                    ..key()
                },
            ),
            (
                "draft",
                RatchetKey {
                    draft_model: "qwen3-0.6b".into(),
                    ..key()
                },
            ),
        ];
        for (what, k) in variants {
            assert_eq!(read_ratchet_v2(dir.path(), &k), None, "{what}");
        }
    }

    /// The recovery reload may build a smaller pool: the ceiling follows it down, never up.
    #[test]
    fn a_smaller_pool_gets_a_proportionally_smaller_ceiling() {
        let dir = tempfile::tempdir().unwrap();
        write_ratchet_v2(dir.path(), 4_000, &key());
        let half = RatchetKey {
            kv_pool_blocks: 12_705 / 2,
            ..key()
        };
        // 4,000 x 6,352 / 12,705 = 1,999.9 -> 1,999
        assert_eq!(read_ratchet_v2(dir.path(), &half), Some(1_999));
        let same = key();
        assert_eq!(read_ratchet_v2(dir.path(), &same), Some(4_000));
        let zero = RatchetKey {
            kv_pool_blocks: 0,
            ..key()
        };
        assert_eq!(read_ratchet_v2(dir.path(), &zero), None);
    }

    #[test]
    fn a_later_write_never_raises_the_ceiling_for_the_same_pool() {
        let dir = tempfile::tempdir().unwrap();
        write_ratchet_v2(dir.path(), 4_000, &key());
        write_ratchet_v2(dir.path(), 6_000, &key());
        assert_eq!(
            read_ratchet_v2(dir.path(), &key()),
            Some(4_000),
            "kept the stricter"
        );
        write_ratchet_v2(dir.path(), 3_000, &key());
        assert_eq!(
            read_ratchet_v2(dir.path(), &key()),
            Some(3_000),
            "tightened"
        );
        // Another pool (smaller) is a different entry; the first one is still there.
        let small = RatchetKey {
            kv_pool_blocks: 6_000,
            ..key()
        };
        write_ratchet_v2(dir.path(), 9_000, &small);
        // The bigger pool's 3,000 scaled to 6,000 blocks (1,416) is stricter than 9,000.
        assert_eq!(read_ratchet_v2(dir.path(), &small), Some(1_416));
        assert_eq!(read_ratchet_v2(dir.path(), &key()), Some(3_000));
    }

    /// R-10: the synced temp-file write leaves exactly the committed file behind, no `.tmp`.
    #[test]
    fn a_write_leaves_only_the_committed_file() {
        let dir = tempfile::tempdir().unwrap();
        write_ratchet_v2(dir.path(), 4_000, &key());
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec![RATCHET_V2_FILENAME.to_owned()]);
        assert_eq!(read_ratchet_v2(dir.path(), &key()), Some(4_000));
    }

    /// E1 F9: a failed commit (the final path is a directory, so the rename fails) removes the temp
    /// file and reports the stage; nothing is left behind.
    #[test]
    fn a_failed_commit_removes_the_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("final");
        std::fs::create_dir(&final_path).unwrap();
        std::fs::write(final_path.join("occupant"), b"x").unwrap();
        let tmp = dir.path().join("final.tmp");
        let Err((stage, _)) = write_durably(dir.path(), &tmp, &final_path, b"{}") else {
            panic!("renaming a file over a non-empty directory must fail");
        };
        assert_eq!(stage, "commit");
        assert!(!tmp.exists(), "temp file left behind");
        // A temp path in a missing directory fails at the first stage.
        let missing = dir.path().join("nope").join("t.tmp");
        let Err((stage, _)) = write_durably(dir.path(), &missing, &final_path, b"{}") else {
            panic!("a temp file in a missing directory cannot be written");
        };
        assert_eq!(stage, "write temp file");
    }

    /// Fable S-2: one entry per key. Another precision learned in between does not erase the first.
    #[test]
    fn a_different_key_does_not_overwrite_the_first_entry() {
        let dir = tempfile::tempdir().unwrap();
        let u4 = key();
        let f16 = RatchetKey {
            kv_precision: "f16".to_owned(),
            ..key()
        };
        write_ratchet_v2(dir.path(), 4_000, &u4);
        write_ratchet_v2(dir.path(), 2_000, &f16);
        assert_eq!(read_ratchet_v2(dir.path(), &u4), Some(4_000));
        assert_eq!(read_ratchet_v2(dir.path(), &f16), Some(2_000));
    }

    /// The single-entry layout written by earlier pre-release builds still reads.
    #[test]
    fn the_first_single_entry_layout_is_still_read() {
        let dir = tempfile::tempdir().unwrap();
        let old = RatchetV2 {
            version: RATCHET_VERSION,
            observed_prompt_tokens: 777,
            key: key(),
        };
        std::fs::write(
            dir.path().join(RATCHET_V2_FILENAME),
            serde_json::to_string(&old).unwrap(),
        )
        .unwrap();
        assert_eq!(read_ratchet_v2(dir.path(), &key()), Some(777));
    }

    #[test]
    fn zero_tokens_corrupt_and_other_versions_are_none() {
        let dir = tempfile::tempdir().unwrap();
        write_ratchet_v2(dir.path(), 0, &key());
        assert_eq!(read_ratchet_v2(dir.path(), &key()), None);
        std::fs::write(dir.path().join(RATCHET_V2_FILENAME), "{not json").unwrap();
        assert_eq!(read_ratchet_v2(dir.path(), &key()), None);
        let mut v3 = serde_json::to_value(RatchetV2 {
            version: 3,
            observed_prompt_tokens: 100,
            key: key(),
        })
        .unwrap();
        v3["version"] = 3.into();
        std::fs::write(dir.path().join(RATCHET_V2_FILENAME), v3.to_string()).unwrap();
        assert_eq!(read_ratchet_v2(dir.path(), &key()), None);
    }

    /// U-06: the old file is never read by v2, noted exactly once, and left in place; the
    /// v2 name differs so the two builds cannot read each other's files.
    #[test]
    fn library_ratchet_files_are_noted_once_and_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let v1 = dir.path().join(RATCHET_V1_FILENAME);
        std::fs::write(
            &v1,
            r#"{"observed_prompt_tokens":4437,"kv_cache_gb_at_learn":1.0}"#,
        )
        .unwrap();
        assert_eq!(read_ratchet_v2(dir.path(), &key()), None);
        assert!(
            note_library_ratchets_ignored(dir.path(), None),
            "first call logs"
        );
        assert!(
            !note_library_ratchets_ignored(dir.path(), None),
            "second call is quiet"
        );
        assert!(v1.is_file(), "never deleted");
        assert_ne!(RATCHET_V1_FILENAME, RATCHET_V2_FILENAME);
        let empty = tempfile::tempdir().unwrap();
        assert!(
            !note_library_ratchets_ignored(empty.path(), None),
            "nothing to note"
        );
    }
}
