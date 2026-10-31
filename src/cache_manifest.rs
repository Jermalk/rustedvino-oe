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
/// Returns `None` only when none of `XDG_CACHE_HOME`, `HOME` and `LOCALAPPDATA` is set and
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

/// Where the learned KV-capacity ceilings (the ratchet) live: beside `ov_cache_dir` as
/// `kv_ratchet/` when that is set, else in the per-user cache — never in the model library,
/// which is shared between servers and builds and may be read-only.
#[must_use]
pub fn ratchet_root(ov_cache_dir: Option<&str>) -> Option<PathBuf> {
    if let Some(dir) = ov_cache_dir.filter(|d| !d.is_empty())
        && let Some(parent) = Path::new(dir).parent()
    {
        return Some(parent.join("kv_ratchet"));
    }
    default_cache_root().map(|base| base.join("kv_ratchet"))
}

fn default_cache_root() -> Option<PathBuf> {
    user_cache_base(
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
        std::env::var_os("LOCALAPPDATA").as_deref(),
    )
    .map(|base| base.join("rustedvino"))
}

/// The per-user cache directory the server's small caches live under:
/// `$XDG_CACHE_HOME`, else `~/.cache`, else `%LOCALAPPDATA%` (Windows sets
/// neither of the first two, so the device-tier cache and the model-hash
/// memo silently never worked there - Windows pass 2, D14). Pure, with the
/// environment passed in, so the order is testable on any OS; empty values
/// count as unset. Shared with `device_inventory`'s tier cache.
#[must_use]
pub(crate) fn user_cache_base(
    xdg_cache_home: Option<&std::ffi::OsStr>,
    home: Option<&std::ffi::OsStr>,
    local_app_data: Option<&std::ffi::OsStr>,
) -> Option<PathBuf> {
    let set = |v: Option<&std::ffi::OsStr>| v.filter(|s| !s.is_empty()).map(PathBuf::from);
    set(xdg_cache_home)
        .or_else(|| set(home).map(|h| h.join(".cache")))
        .or_else(|| set(local_app_data))
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

/// Names of the per-KV-precision blob-cache subdirectories (E-17). `OpenVINO`'s blob id does not
/// include `KV_CACHE_PRECISION` on 2026.4.x, so a blob compiled for one precision is silently
/// reused for another; keeping one directory per *passed* precision string avoids the clash.
pub const KV_DIR_PREFIX: &str = "kv-";

/// The cache subdirectory for the `KV_CACHE_PRECISION` string handed to the plugin.
///
/// The tag follows the **passed string**, never the precision the plugin would resolve: `""`
/// (nothing passed, plugin default) is its own directory, so a wrong guess about the default
/// can never put a blob in the wrong place. Lower-case only. No tag ends in `.bin` (`OpenVINO`
/// treats a `CACHE_DIR` ending in `.bin` as single-file storage).
///
/// Whoever starts passing another `KEY_/VALUE_CACHE_*` property to the plugin must add it to the
/// tag here; a test over `ov_bridge.cpp` fails when a new one appears.
///
/// # Errors
/// Any string other than `""`, `u8`, `i8`, `u4`, `i4`, `f16`, `f32`.
pub fn kv_cache_subdir(passed: &str) -> anyhow::Result<&'static str> {
    Ok(match passed {
        "" => "kv-auto",
        "u8" | "i8" => "kv-i8",
        "u4" | "i4" => "kv-u4",
        "f16" => "kv-f16",
        "f32" => "kv-f32",
        other => anyhow::bail!("no blob-cache directory for KV precision {other:?}"),
    })
}

/// `^kv-[a-z0-9]+$` - the only subdirectory names the cache walkers look into.
#[must_use]
pub fn is_kv_subdir(name: &str) -> bool {
    name.strip_prefix(KV_DIR_PREFIX).is_some_and(|tag| {
        !tag.is_empty()
            && tag
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
    })
}

/// The directory part of a manifest/walker file name: `Some("kv-u4")` for `"kv-u4/x.blob"`,
/// `None` for a bare name (a file directly in the cache root).
#[must_use]
pub fn blob_dir(name: &str) -> Option<&str> {
    name.split_once('/').map(|(dir, _)| dir)
}

/// Whether a manifest entry is a name this module could have produced: a bare file name or
/// `kv-<tag>/<file>`, never absolute, never `..`, never a backslash path. Anything else is
/// ignored by the owner index, so a hand-edited or corrupt manifest cannot point the prune
/// pass outside the cache directory.
fn is_valid_blob_name(name: &str) -> bool {
    let plain = |part: &str| {
        !part.is_empty() && part != "." && part != ".." && !part.contains(['\\', '/', ':'])
    };
    match name.split_once('/') {
        None => plain(name),
        Some((dir, file)) => is_kv_subdir(dir) && plain(file),
    }
}

/// Regular files directly inside `dir`, named `<prefix>/<file>` (or just `<file>` without a
/// prefix). Best-effort: an unreadable directory or a file whose metadata cannot be read is
/// skipped, never an error - blob attribution and pruning degrade to "did nothing".
fn list_dir_files(dir: &Path, prefix: Option<&str>) -> Vec<CacheDirEntry> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            if !meta.is_file() {
                return None;
            }
            let name = e.file_name().into_string().ok()?;
            Some(CacheDirEntry {
                file_name: prefix.map_or(name.clone(), |p| format!("{p}/{name}")),
                size_bytes: meta.len(),
                mtime: meta.modified().ok()?,
            })
        })
        .collect()
}

/// A snapshot of the file names directly inside `ov_cache_dir` - the root only, not the `kv-*`
/// subdirectories. Used for the kinds that cache in the root (NPU, whisper, image, ...); a
/// KV engine's directory is snapshotted with [`snapshot_cache_scope`]. Best-effort: an
/// unreadable directory yields an empty snapshot rather than an error - blob attribution
/// degrading to "nothing captured" must never fail a load.
#[must_use]
pub fn snapshot_cache_dir(ov_cache_dir: &Path) -> std::collections::HashSet<String> {
    snapshot_cache_scope(ov_cache_dir, None)
}

/// Like [`snapshot_cache_dir`] for the directory a load actually hands to the engine: the root
/// (`subdir = None`) or `root/<subdir>`. Names carry the `<subdir>/` prefix, the form the
/// manifest stores.
#[must_use]
pub fn snapshot_cache_scope(
    ov_cache_dir: &Path,
    subdir: Option<&str>,
) -> std::collections::HashSet<String> {
    let dir = subdir.map_or_else(|| ov_cache_dir.to_path_buf(), |s| ov_cache_dir.join(s));
    list_dir_files(&dir, subdir)
        .into_iter()
        .map(|e| e.file_name)
        .collect()
}

/// One file found in the cache (name, size, mtime) - the prune pass (the project's internal
/// engineering log) uses this to compute total size and pick eviction order among candidates.
/// `file_name` is relative to `ov_cache_dir`, components joined with `/` on every platform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheDirEntry {
    pub file_name: String,
    pub size_bytes: u64,
    pub mtime: std::time::SystemTime,
}

/// Lists every regular file in `ov_cache_dir` with its size and mtime: the files of the root
/// plus the files of its direct `kv-*` subdirectories (see [`is_kv_subdir`]). One level only,
/// symlinked directories are not followed, any other subdirectory is ignored. Shared by the
/// size gauge and the prune pass. Best-effort like [`snapshot_cache_dir`].
#[must_use]
pub fn list_cache_dir_entries(ov_cache_dir: &Path) -> Vec<CacheDirEntry> {
    let mut out = list_dir_files(ov_cache_dir, None);
    let Ok(entries) = std::fs::read_dir(ov_cache_dir) else {
        return out;
    };
    for entry in entries.filter_map(Result::ok) {
        // `DirEntry::file_type` does not follow symlinks, so a link named `kv-x` is skipped.
        if !entry.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        if is_kv_subdir(&name) {
            out.extend(list_dir_files(&entry.path(), Some(&name)));
        }
    }
    out
}

/// Creates `root/<subdir>` and proves it is writable, so `OpenVINO` (which throws on a
/// `CACHE_DIR` it cannot create) is never handed a directory that will fail.
///
/// # Errors
/// The directory cannot be created or written to (e.g. a regular file has its name).
pub fn ensure_writable_subdir(root: &Path, subdir: &str) -> std::io::Result<PathBuf> {
    let dir = root.join(subdir);
    std::fs::create_dir_all(&dir)?;
    let probe = dir.join(".rv-write-probe");
    std::fs::File::create(&probe)?;
    let _ = std::fs::remove_file(&probe);
    Ok(dir)
}

/// Every `(model_id, device)` that the manifests under `root` attribute each blob name to.
/// Names are as stored: bare for the cache root, `kv-<tag>/<file>` for a keyed directory.
/// Entries that are not valid names (see [`is_valid_blob_name`]) are dropped. Best-effort: a
/// manifest that fails to parse contributes nothing - its blobs are "unattributed".
#[must_use]
pub fn blob_owners(root: &Path) -> std::collections::HashMap<String, Vec<(String, String)>> {
    let mut index: std::collections::HashMap<String, Vec<(String, String)>> =
        std::collections::HashMap::new();
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
        for (device, device_blobs) in &file.devices {
            for blob_name in device_blobs
                .blob_file_names
                .iter()
                .filter(|n| is_valid_blob_name(n))
            {
                index
                    .entry(blob_name.clone())
                    .or_default()
                    .push((model_id.to_owned(), device.clone()));
            }
        }
    }
    index
}

/// Reverse index (blob name -> owning `model_id`), built from [`blob_owners`]. When several
/// models claim one name the last one read wins (as before; the prune pass uses
/// [`blob_owners`] and keeps every claimant).
#[must_use]
pub fn blob_owner_index(root: &Path) -> std::collections::HashMap<String, String> {
    blob_owners(root)
        .into_iter()
        .filter_map(|(name, owners)| owners.last().map(|(m, _)| (name, m.clone())))
        .collect()
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

    /// The ratchet sits beside `ov_cache_dir` when set, never in the model library.
    #[test]
    fn ratchet_root_is_beside_the_ov_cache_dir() {
        assert_eq!(
            ratchet_root(Some("/srv/rv/ovcache")),
            Some(PathBuf::from("/srv/rv/kv_ratchet"))
        );
        assert_eq!(
            ratchet_root(Some("")),
            default_cache_root().map(|b| b.join("kv_ratchet"))
        );
    }

    fn sample_entry() -> ModelHashEntry {
        ModelHashEntry {
            backbone_relpath: "transformer/openvino_model.bin".to_owned(),
            size_bytes: 123,
            mtime_unix: 456,
            sha256: "sha256:deadbeef".to_owned(),
        }
    }

    /// D14: Windows has no `HOME`/`XDG_CACHE_HOME`; `%LOCALAPPDATA%` is the
    /// last resort, and the Linux order is untouched.
    #[test]
    fn user_cache_base_order_is_xdg_then_home_then_local_app_data() {
        use std::ffi::OsStr;
        let o = |s: &'static str| Some(OsStr::new(s));
        assert_eq!(
            user_cache_base(o("/x"), o("/h"), o("/l")),
            Some(PathBuf::from("/x"))
        );
        assert_eq!(
            user_cache_base(None, o("/h"), o("/l")),
            Some(PathBuf::from("/h/.cache"))
        );
        assert_eq!(
            user_cache_base(None, None, o("/l")),
            Some(PathBuf::from("/l"))
        );
        assert_eq!(
            user_cache_base(o(""), o(""), o("/l")),
            Some(PathBuf::from("/l"))
        );
        assert_eq!(user_cache_base(None, None, None), None);
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

    /// The tag follows the passed string, case-sensitively, and no tag can be read as a
    /// single-file `*.bin` cache (U-C23).
    #[test]
    fn kv_cache_subdir_follows_the_passed_string() {
        for (passed, tag) in [
            ("", "kv-auto"),
            ("u8", "kv-i8"),
            ("i8", "kv-i8"),
            ("u4", "kv-u4"),
            ("i4", "kv-u4"),
            ("f16", "kv-f16"),
            ("f32", "kv-f32"),
        ] {
            let got = kv_cache_subdir(passed).unwrap();
            assert_eq!(got, tag, "passed {passed:?}");
            assert!(!got.to_ascii_lowercase().ends_with(".bin"));
            assert!(is_kv_subdir(got));
        }
        assert!(kv_cache_subdir("bogus").is_err());
        assert!(kv_cache_subdir("U8").is_err());
        assert!(
            kv_cache_subdir("auto").is_err(),
            "auto is normalised to \"\" before the bridge"
        );
    }

    /// Guard for the day another KV property is passed to the plugin (U-C24): the bridge sets
    /// the KV precision in exactly three places (the text engine, and the vision pipeline in its
    /// plain and per-role forms), all fed by `kv_precision_type`, which maps only the three
    /// strings the tag knows - no string-named setting, no separate key/value cache properties.
    #[test]
    fn bridge_passes_no_kv_property_the_cache_tag_does_not_cover() {
        let src = include_str!("../ov_bridge/ov_bridge.cpp");
        let code: Vec<&str> = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect();
        let code_text = code.join("\n");
        // The three places that set the property.
        assert_eq!(
            code_text.matches("ov::hint::kv_cache_precision(").count(),
            2
        );
        assert_eq!(
            code_text
                .matches("ov::hint::kv_cache_precision.name()")
                .count(),
            1
        );
        assert!(!code_text.contains("\"KV_CACHE_PRECISION\""));
        // The mapping: three strings, nothing else.
        let start = src.find("static bool kv_precision_type(").unwrap();
        let end = start + src[start..].find("\n}\n").unwrap();
        let mapping = &src[start..end];
        assert_eq!(mapping.matches("p == \"").count(), 3);
        for tag in ["\"i8\"", "\"u4\"", "\"f16\""] {
            assert!(mapping.contains(&format!("p == {tag}")), "{tag}");
        }
        // Everything that mentions the parameter is the mapping, the two creators, or the
        // record line; any other function reading it would be a new path to the plugin.
        let outside_mapping = src.replace(mapping, "");
        for line in outside_mapping
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
        {
            let l = line.to_ascii_lowercase();
            if !l.contains("kv_cache_precision") {
                continue;
            }
            assert!(
                l.contains("kv_precision_type(")
                    || l.contains("const char* kv_cache_precision")
                    || l.contains("kv_cache_precision ?")
                    || l.contains("kv_cache_precision, kv_form")
                    || l.contains("ov::hint::kv_cache_precision"),
                "unexpected KV property use: {line}"
            );
        }
        let lower = src.to_ascii_lowercase();
        for banned in [
            "key_cache_precision",
            "value_cache_precision",
            "key_cache_group_size",
            "value_cache_group_size",
        ] {
            assert!(!lower.contains(banned), "{banned}");
        }
    }

    /// Walker hygiene (U-14, U-C34): the root, plus the direct `kv-*` subdirectories; names
    /// joined with `/`; look-alike directories, deeper levels and symlinked directories ignored.
    #[test]
    fn walker_lists_root_and_kv_subdirs_only() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let put = |rel: &str, n: usize| {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, vec![0_u8; n]).unwrap();
        };
        put("a.blob", 1);
        put("kv-u4/b.blob", 2);
        put("kv-i8/c.blob", 3);
        put("kv-i8/c.cl_cache", 4);
        put("KV-U4/no.blob", 8);
        put("kv-u4.bak/no.blob", 8);
        put("kv-/no.blob", 8);
        put("other/no.blob", 8);
        put("kv-u4/sub/deep.blob", 8);
        #[cfg(unix)]
        std::os::unix::fs::symlink("/", root.join("kv-x")).unwrap();

        let mut names: Vec<String> = list_cache_dir_entries(root)
            .into_iter()
            .map(|e| e.file_name)
            .collect();
        names.sort();
        assert_eq!(
            names,
            ["a.blob", "kv-i8/c.blob", "kv-i8/c.cl_cache", "kv-u4/b.blob"]
        );
        // The root-only snapshot sees neither.
        assert_eq!(snapshot_cache_dir(root).len(), 1);
        assert!(list_cache_dir_entries(&root.join("missing")).is_empty());
    }

    /// The load's before/after diff covers only the directory the engine was given (U-C28).
    #[test]
    fn scope_snapshot_covers_only_the_engines_directory() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("kv-u4")).unwrap();
        let before = snapshot_cache_scope(root, Some("kv-u4"));
        assert!(before.is_empty());
        std::fs::write(root.join("kv-u4/x.blob"), b"1").unwrap();
        std::fs::write(root.join("kv-u4/k.cl_cache"), b"1").unwrap();
        std::fs::write(root.join("w.blob"), b"1").unwrap(); // an unrelated engine, in the root
        let mut new: Vec<String> = snapshot_cache_scope(root, Some("kv-u4"))
            .difference(&before)
            .cloned()
            .collect();
        new.sort();
        assert_eq!(new, ["kv-u4/k.cl_cache", "kv-u4/x.blob"]);
    }

    /// A directory that cannot be made yields an error, not a panic (U-21, U-C27).
    #[test]
    fn ensure_writable_subdir_reports_a_blocked_directory() {
        let dir = tempfile::tempdir().unwrap();
        let ok = ensure_writable_subdir(dir.path(), "kv-u4").unwrap();
        assert!(ok.is_dir());
        assert_eq!(
            std::fs::read_dir(&ok).unwrap().count(),
            0,
            "probe file removed"
        );
        std::fs::write(dir.path().join("kv-i8"), b"a file").unwrap();
        assert!(ensure_writable_subdir(dir.path(), "kv-i8").is_err());
    }

    /// The same basename in two directories has two owners (U-C29), and a manifest cannot name
    /// a file outside the cache (U-C32).
    #[test]
    fn owners_keep_same_basename_apart_and_drop_unsafe_names() {
        let dir = tempfile::tempdir().unwrap();
        write_device_blobs_entry(dir.path(), "a", "GPU.1", &["123.blob".to_owned()]);
        write_device_blobs_entry(dir.path(), "b", "GPU.1", &["kv-u4/123.blob".to_owned()]);
        write_device_blobs_entry(
            dir.path(),
            "evil",
            "GPU.1",
            &[
                "../../etc/passwd".to_owned(),
                "/tmp/x.blob".to_owned(),
                "kv-u4/../../y.blob".to_owned(),
                "C:\\x.blob".to_owned(),
                "kv-u4\\z.blob".to_owned(),
                "other/q.blob".to_owned(),
                "kv-u4/ok.blob".to_owned(),
            ],
        );
        let owners = blob_owners(dir.path());
        assert_eq!(owners["123.blob"], [("a".to_owned(), "GPU.1".to_owned())]);
        assert_eq!(
            owners["kv-u4/123.blob"],
            [("b".to_owned(), "GPU.1".to_owned())]
        );
        assert_eq!(
            owners.len(),
            3,
            "only 123.blob, kv-u4/123.blob and kv-u4/ok.blob: {owners:?}"
        );
        assert!(owners.contains_key("kv-u4/ok.blob"));
        assert_eq!(blob_dir("kv-u4/ok.blob"), Some("kv-u4"));
        assert_eq!(blob_dir("123.blob"), None);
    }
}
