// ============================================================
// src/cache_manifest.rs — per-model `OpenVINO` cache manifest
// ============================================================
// One JSON file per model_id under `manifest_root` (a sibling of
// `ov_cache_dir` itself — see `manifest_root`), recording expensive
// per-model content-identity metadata that would otherwise be recomputed
// from scratch on every load: `pipelines/image.rs`'s `model_hash` today,
// OV compile-cache blob attribution in a later pass
// (the project's internal engineering log).
//
// All I/O here is best-effort: a missing, corrupt, or unwritable manifest
// degrades to "no caching, compute as before" — it must never turn a
// working load into a failed one. Callers pass `Option<&Path>` for the
// manifest root; `None` (no `ov_cache_dir` configured, or no parent
// directory to root the manifest under) disables caching entirely.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A memoized `model_hash` computation for one model: which backbone file
/// was hashed, its size/mtime at hash time, and the resulting digest.
/// Cache hit requires an exact match on path + size + mtime — any of those
/// three changing (a checkpoint swap, a re-export, a different backbone
/// directory winning the `unet`/`transformer` search) invalidates the entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelHashEntry {
    /// Backbone path relative to the model directory (e.g.
    /// `"transformer/openvino_model.bin"`).
    pub backbone_relpath: String,
    pub size_bytes: u64,
    pub mtime_unix: i64,
    /// The `sha256:`-prefixed digest, exactly as `compute_model_hash` would
    /// return it.
    pub sha256: String,
}

/// OV compile-cache blob attribution for one model on one device: the file
/// names (not full paths — always joined with the live `ov_cache_dir` at
/// read time, since that directory can move) `OpenVINO`'s own cache created
/// or touched for this model, captured by diffing `ov_cache_dir`'s listing
/// around the `compile_model`/pipeline-construction call
/// (`lifecycle.rs`'s `fn load`).
///
/// Best-effort, not a guarantee: a load that crashed mid-diff, or ran before
/// this feature existed, has no entry here even though its blobs are on
/// disk. Capacity-bounded eviction (not built yet) must fall back to
/// file-mtime LRU for anything this manifest doesn't cover.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DeviceBlobs {
    pub blob_file_names: Vec<String>,
}

/// One model's manifest file.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ManifestFile {
    #[serde(default)]
    pub model_hash: Option<ModelHashEntry>,
    #[serde(default)]
    pub devices: std::collections::BTreeMap<String, DeviceBlobs>,
}

/// Computes the manifest root directory.
///
/// When `ov_cache_dir` is configured, the manifest lives as a `cache_manifest`
/// sibling of its parent — e.g. `ov_cache_dir` = `~/.cache/rustedvino/ov_cache`
/// → root = `~/.cache/rustedvino/cache_manifest` — so an operator who moved
/// the whole `rustedvino` cache tree elsewhere gets the manifest colocated
/// with it. Otherwise falls back to `$XDG_CACHE_HOME/rustedvino/cache_manifest`
/// or `~/.cache/rustedvino/cache_manifest` (same precedence
/// `device_inventory.rs::cache_path` already uses for `device_tiers.json`) —
/// **deliberately independent of `ov_cache_dir`**: `model_hash` memoization is
/// unrelated to `OpenVINO`'s own GPU blob cache, and machines exist that never
/// set `ov_cache_dir` yet still hit the multi-second hash recompute on every
/// load that motivated this cache.
///
/// Returns `None` only when neither `XDG_CACHE_HOME` nor `HOME` is set and
/// `ov_cache_dir` is also absent — no location to root the manifest under.
/// Never a panic.
#[must_use]
pub fn manifest_root(ov_cache_dir: Option<&str>) -> Option<PathBuf> {
    if let Some(dir) = ov_cache_dir.filter(|d| !d.is_empty())
        && let Some(parent) = Path::new(dir).parent()
    {
        return Some(parent.join("cache_manifest"));
    }
    default_cache_root().map(|base| base.join("cache_manifest"))
}

fn default_cache_root() -> Option<PathBuf> {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .map(|base| base.join("rustedvino"))
}

fn manifest_path(root: &Path, model_id: &str) -> PathBuf {
    root.join(format!("{model_id}.json"))
}

/// Reads `<root>/<model_id>.json`. `None` on any failure — missing file,
/// unreadable, unparseable, or a manifest written by a future schema
/// version this build doesn't understand. Never surfaces an error to the
/// caller; loading a model must not fail because its manifest is stale.
#[must_use]
pub fn read(root: &Path, model_id: &str) -> Option<ManifestFile> {
    let raw = std::fs::read_to_string(manifest_path(root, model_id)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Writes `<root>/<model_id>.json`, merging `entry` into whatever manifest
/// (if any) already exists for this model — a blob-attribution write later
/// (`devices`) must not clobber a `model_hash` entry written earlier and
/// vice versa.
///
/// Best-effort — logs and returns on any I/O error (unwritable root,
/// permissions, disk full) rather than failing the caller's load.
pub fn write_model_hash_entry(root: &Path, model_id: &str, entry: ModelHashEntry) {
    let mut file = read(root, model_id).unwrap_or_default();
    file.model_hash = Some(entry);
    commit(root, model_id, &file);
}

/// Records that `new_blob_file_names` were created/touched in `ov_cache_dir`
/// while loading `model_id` on `device` — merged (union, not replaced) into
/// whatever `devices[device]` this model's manifest already has, since a
/// later load that hits `OpenVINO`'s own cache (no new files) must not erase
/// blob names an earlier load already attributed. Best-effort, like
/// [`write_model_hash_entry`].
pub fn write_device_blobs_entry(
    root: &Path,
    model_id: &str,
    device: &str,
    new_blob_file_names: &[String],
) {
    let mut file = read(root, model_id).unwrap_or_default();
    let entry = file.devices.entry(device.to_owned()).or_default();
    for name in new_blob_file_names {
        if !entry.blob_file_names.contains(name) {
            entry.blob_file_names.push(name.clone());
        }
    }
    commit(root, model_id, &file);
}

/// A flat snapshot of file names directly inside `ov_cache_dir` — not
/// recursive; `OpenVINO`'s own cache directory is a flat population of
/// content-hash-named blob/kernel-cache files, no subdirectories
/// (the project's internal engineering log, 2026-07-28 finding). Best-effort:
/// an unreadable directory yields an empty snapshot rather than an error —
/// blob attribution degrading to "nothing captured" must never fail a load.
#[must_use]
pub fn snapshot_cache_dir(ov_cache_dir: &Path) -> std::collections::HashSet<String> {
    let Ok(entries) = std::fs::read_dir(ov_cache_dir) else {
        return std::collections::HashSet::new();
    };
    entries
        .filter_map(Result::ok)
        .filter(|e| e.path().is_file())
        .filter_map(|e| e.file_name().into_string().ok())
        .collect()
}

/// One file found directly inside `ov_cache_dir` (name, size, mtime) — the
/// prune pass (the project's internal engineering log) uses this to compute
/// total size and pick eviction order among candidates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheDirEntry {
    pub file_name: String,
    pub size_bytes: u64,
    pub mtime: std::time::SystemTime,
}

/// Lists every regular file directly inside `ov_cache_dir` with its size and
/// mtime — not recursive, same "OV's cache dir is flat" basis as
/// [`snapshot_cache_dir`]. Best-effort: an unreadable directory or a file
/// whose metadata can't be read yields an empty/partial list, never an
/// error — pruning must degrade to "did nothing this pass," not a crash.
#[must_use]
pub fn list_cache_dir_entries(ov_cache_dir: &Path) -> Vec<CacheDirEntry> {
    let Ok(entries) = std::fs::read_dir(ov_cache_dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            if !meta.is_file() {
                return None;
            }
            Some(CacheDirEntry {
                file_name: e.file_name().into_string().ok()?,
                size_bytes: meta.len(),
                mtime: meta.modified().ok()?,
            })
        })
        .collect()
}

/// Reverse index (blob file name → owning `model_id`), built by reading every
/// manifest file under `root` and flattening each model's `devices` map.
/// Best-effort: any manifest that fails to parse simply contributes nothing —
/// its blobs fall back to "unattributed" for the prune pass, same as a blob
/// from before this feature shipped.
#[must_use]
pub fn blob_owner_index(root: &Path) -> std::collections::HashMap<String, String> {
    let mut index = std::collections::HashMap::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return index;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let Some(model_id) = path
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|_| path.extension().is_some_and(|ext| ext == "json"))
        else {
            continue;
        };
        let Some(file) = read(root, model_id) else {
            continue;
        };
        for device_blobs in file.devices.values() {
            for blob_name in &device_blobs.blob_file_names {
                index.insert(blob_name.clone(), model_id.to_owned());
            }
        }
    }
    index
}

/// Serializes `file` and commits it to `<root>/<model_id>.json` via temp
/// file + rename within `root`, so a crash mid-write can never leave a
/// truncated file that poisons the next read. Best-effort — logs and returns
/// on any I/O error rather than failing the caller's load.
fn commit(root: &Path, model_id: &str, file: &ManifestFile) {
    if let Err(err) = std::fs::create_dir_all(root) {
        tracing::warn!(model_id = %model_id, error = %err, "cache_manifest: could not create manifest root, skipping write");
        return;
    }

    let Ok(serialized) = serde_json::to_string_pretty(file) else {
        tracing::warn!(model_id = %model_id, "cache_manifest: could not serialize manifest, skipping write");
        return;
    };

    let final_path = manifest_path(root, model_id);
    let tmp_path = root.join(format!("{model_id}.json.tmp"));
    if let Err(err) = std::fs::write(&tmp_path, serialized) {
        tracing::warn!(model_id = %model_id, error = %err, "cache_manifest: could not write temp manifest, skipping");
        return;
    }
    if let Err(err) = std::fs::rename(&tmp_path, &final_path) {
        tracing::warn!(model_id = %model_id, error = %err, "cache_manifest: could not commit manifest write");
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn sample_entry() -> ModelHashEntry {
        ModelHashEntry {
            backbone_relpath: "transformer/openvino_model.bin".to_owned(),
            size_bytes: 123,
            mtime_unix: 456,
            sha256: "sha256:deadbeef".to_owned(),
        }
    }

    /// `None`/empty `ov_cache_dir` falls back to the `~/.cache/rustedvino`
    /// default — deliberately does NOT disable the manifest, unlike
    /// `ov_cache_dir` itself. Boxes exist (this one, at the time this test was
    /// written) that never configure `ov_cache_dir` at all; `model_hash`
    /// memoization must still work for them, since OV's own GPU blob cache is
    /// an unrelated concern.
    #[test]
    fn manifest_root_falls_back_to_home_cache_when_ov_cache_dir_absent_or_empty() {
        for input in [None, Some("")] {
            let root = manifest_root(input).expect("fallback root computed");
            assert!(
                root.ends_with("rustedvino/cache_manifest"),
                "root: {root:?}"
            );
        }
    }

    /// A normal `ov_cache_dir` produces a `cache_manifest` sibling of its
    /// parent directory, not a subdirectory of `ov_cache_dir` itself (which
    /// `OpenVINO` treats as its own space) — and takes priority over the
    /// `~/.cache/rustedvino` default, so a box that relocated its whole cache
    /// tree gets the manifest colocated with it.
    #[test]
    fn manifest_root_is_sibling_of_ov_cache_dir_parent() {
        let root =
            manifest_root(Some("/home/user/.cache/rustedvino/ov_cache")).expect("root computed");
        assert_eq!(
            root,
            PathBuf::from("/home/user/.cache/rustedvino/cache_manifest")
        );
    }

    /// The filesystem root has no parent directory — falls back to the
    /// `~/.cache/rustedvino` default rather than panicking or disabling
    /// caching (the one real case `Path::parent()` returns `None` for).
    #[test]
    fn manifest_root_falls_back_when_ov_cache_dir_has_no_parent() {
        let root = manifest_root(Some("/")).expect("fallback root computed");
        assert!(
            root.ends_with("rustedvino/cache_manifest"),
            "root: {root:?}"
        );
    }

    /// Round trip: write then read returns the same entry.
    #[test]
    fn write_then_read_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_model_hash_entry(dir.path(), "flux-schnell-int4-ov", sample_entry());

        let file = read(dir.path(), "flux-schnell-int4-ov").expect("manifest read back");
        assert_eq!(file.model_hash, Some(sample_entry()));
        assert!(file.devices.is_empty(), "devices untouched by this write");
    }

    /// Reading a manifest that was never written returns `None`, not an
    /// error — the normal "first load of this model" case.
    #[test]
    fn read_returns_none_when_file_absent() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(read(dir.path(), "never-loaded"), None);
    }

    /// A corrupt manifest file degrades to a cache miss, not a crash or
    /// error surfaced to the caller.
    #[test]
    fn read_returns_none_when_file_corrupt() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("corrupt-model.json"), b"not json").expect("write garbage");
        assert_eq!(read(dir.path(), "corrupt-model"), None);
    }

    /// Writing a `model_hash` entry preserves any pre-existing `devices` data
    /// in the same file — a merge, not an overwrite.
    #[test]
    fn write_model_hash_entry_preserves_existing_devices() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_device_blobs_entry(
            dir.path(),
            "seeded-model",
            "GPU.1",
            &["abc.blob".to_owned()],
        );

        write_model_hash_entry(dir.path(), "seeded-model", sample_entry());

        let file = read(dir.path(), "seeded-model").expect("manifest read back");
        assert_eq!(file.model_hash, Some(sample_entry()));
        assert_eq!(file.devices.len(), 1, "devices entry survived the merge");
    }

    /// A fresh device entry records exactly the blob names it was given.
    #[test]
    fn write_device_blobs_entry_records_new_blobs() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_device_blobs_entry(
            dir.path(),
            "some-model",
            "GPU.1",
            &["a.blob".to_owned(), "b.blob".to_owned()],
        );

        let file = read(dir.path(), "some-model").expect("manifest read back");
        let entry = file.devices.get("GPU.1").expect("device entry present");
        assert_eq!(entry.blob_file_names, vec!["a.blob", "b.blob"]);
    }

    /// A second write for the same (model, device) unions in new names
    /// without dropping or duplicating ones already recorded — a later load
    /// that hits `OpenVINO`'s own cache (zero new files) must not erase an
    /// earlier load's attribution.
    #[test]
    fn write_device_blobs_entry_unions_across_writes() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_device_blobs_entry(dir.path(), "some-model", "GPU.1", &["a.blob".to_owned()]);
        write_device_blobs_entry(
            dir.path(),
            "some-model",
            "GPU.1",
            &["a.blob".to_owned(), "b.blob".to_owned()],
        );

        let file = read(dir.path(), "some-model").expect("manifest read back");
        let entry = file.devices.get("GPU.1").expect("device entry present");
        assert_eq!(
            entry.blob_file_names,
            vec!["a.blob", "b.blob"],
            "no duplicate a.blob, b.blob appended"
        );
    }

    /// Two different devices for the same model get independent entries —
    /// e.g. an embedding model on `GPU.0` and the LLM on `GPU.1`.
    #[test]
    fn write_device_blobs_entry_keeps_devices_independent() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_device_blobs_entry(dir.path(), "some-model", "GPU.0", &["a.blob".to_owned()]);
        write_device_blobs_entry(dir.path(), "some-model", "GPU.1", &["b.blob".to_owned()]);

        let file = read(dir.path(), "some-model").expect("manifest read back");
        assert_eq!(file.devices.len(), 2);
        assert_eq!(
            file.devices["GPU.0"].blob_file_names,
            vec!["a.blob".to_owned()]
        );
        assert_eq!(
            file.devices["GPU.1"].blob_file_names,
            vec!["b.blob".to_owned()]
        );
    }

    /// `snapshot_cache_dir` lists only regular files, not subdirectories,
    /// and degrades to an empty set (not an error) when the directory
    /// doesn't exist yet — the normal state before `ov_cache_dir`'s first
    /// write.
    #[test]
    fn snapshot_cache_dir_lists_files_not_subdirs_and_tolerates_missing_dir() {
        assert!(
            snapshot_cache_dir(&PathBuf::from("/nonexistent/path/for/this/test")).is_empty(),
            "missing directory yields an empty snapshot, not a panic"
        );

        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("blob-a.blob"), b"data").expect("write blob-a");
        std::fs::write(dir.path().join("blob-b.cl_cache"), b"data").expect("write blob-b");
        std::fs::create_dir(dir.path().join("a_subdir")).expect("mkdir subdir");

        let snapshot = snapshot_cache_dir(dir.path());
        assert_eq!(
            snapshot,
            ["blob-a.blob".to_owned(), "blob-b.cl_cache".to_owned()]
                .into_iter()
                .collect()
        );
    }

    /// `list_cache_dir_entries` reports size and mtime alongside the name,
    /// and degrades to empty (not an error) on a missing directory — the
    /// prune pass's total-size computation depends on this never panicking.
    #[test]
    fn list_cache_dir_entries_reports_size_and_tolerates_missing_dir() {
        assert!(
            list_cache_dir_entries(&PathBuf::from("/nonexistent/path/for/this/test")).is_empty()
        );

        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("a.blob"), b"12345").expect("write a.blob");

        let entries = list_cache_dir_entries(dir.path());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].file_name, "a.blob");
        assert_eq!(entries[0].size_bytes, 5);
    }

    /// `blob_owner_index` flattens every model's `devices` map across every
    /// manifest file under `root` into one blob-name → `model_id` lookup, and
    /// degrades to an empty index (not an error) on a missing/unreadable
    /// root — an unattributed blob is the correct fallback, never a crash.
    #[test]
    fn blob_owner_index_flattens_across_manifests_and_devices() {
        assert!(blob_owner_index(&PathBuf::from("/nonexistent/path/for/this/test")).is_empty());

        let dir = tempfile::tempdir().expect("tempdir");
        write_device_blobs_entry(dir.path(), "model-a", "GPU.1", &["a1.blob".to_owned()]);
        write_device_blobs_entry(dir.path(), "model-b", "GPU.0", &["b1.blob".to_owned()]);
        write_device_blobs_entry(dir.path(), "model-b", "GPU.1", &["b2.blob".to_owned()]);

        let index = blob_owner_index(dir.path());
        assert_eq!(index.get("a1.blob"), Some(&"model-a".to_owned()));
        assert_eq!(index.get("b1.blob"), Some(&"model-b".to_owned()));
        assert_eq!(index.get("b2.blob"), Some(&"model-b".to_owned()));
        assert_eq!(index.get("never-seen.blob"), None);
    }
}
