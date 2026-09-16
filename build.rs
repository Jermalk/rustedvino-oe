// build.rs — compile the C++ OpenVINO GenAI bridge and link the runtime libs.
//
// CRASH COURSE — build scripts:
//   `build.rs` runs BEFORE the Rust compiler. Its printed `cargo:` directives
//   control how the crate is compiled and linked.
//
//   cc::Build compiles C/C++ source files into a static library that is
//   automatically linked into the final binary. We use it here to compile
//   ov_bridge.cpp with the flags required by the target platform's ABI.
//
// PLATFORM BRANCH (CHECK-00, dev/plans/windows-x86-compat.md):
//   `CARGO_CFG_TARGET_OS` describes the TARGET, not the host — so a
//   cross-compile from Linux to x86_64-pc-windows-msvc takes the Windows
//   branch. We cannot use `#[cfg(windows)]` here: build.rs itself always
//   runs on the HOST, and cfg! would describe the host, not the target.
//
// ABI NOTES
//   Linux: openvino-genai 2026.1 pip wheel uses _GLIBCXX_USE_CXX11_ABI=1 (the
//   new __cxx11 ABI). We must compile the bridge with the same flag.
//   Confirmed by: nm -D libopenvino_genai.so.2610 | grep __cxx11
//   Windows: no GLIBCXX (that is a libstdc++ concept); clang-cl targets the
//   MSVC ABI and uses the MSVC STL that cargo-xwin downloads. Exceptions
//   across the DLL boundary require /EHsc.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    emit_git_hash();

    // Target OS, not host OS — set by cargo for build scripts.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        build_windows();
    } else {
        build_linux();
    }
}

/// Set `RV_GIT_HASH` for `env!()` in `/health` — the short commit hash the
/// binary was built from. Falls back to `"unknown"` (never fails the build)
/// so a dist package built from a source tarball without a `.git` dir still
/// links. Rebuild is triggered by `.git/HEAD` and whatever ref it points at,
/// so switching branches or committing refreshes the hash on the next build.
fn emit_git_hash() {
    // An explicitly-provided RV_GIT_HASH wins. Container and tarball builds have
    // no usable git (no binary, or git refuses the mounted repo as
    // dubious-ownership under a different UID), and silently shipping a release
    // whose /health reports "unknown" leaves a bug reporter unable to say which
    // build they are running.
    let hash = env::var("RV_GIT_HASH")
        .ok()
        .map(|h| h.trim().to_owned())
        .filter(|h| !h.is_empty())
        .or_else(|| {
            std::process::Command::new("git")
                .args(["rev-parse", "--short=12", "HEAD"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=RV_GIT_HASH={hash}");

    // Without this, changing the passed-in hash would be silently cached.
    println!("cargo:rerun-if-env-changed=RV_GIT_HASH");
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD")
        && let Some(r#ref) = head.strip_prefix("ref: ")
    {
        println!("cargo:rerun-if-changed=.git/{}", r#ref.trim());
    }
}

/// Where the pip-installed `OpenVINO` lives: ask `python3` to import
/// `openvino_genai` and report its location.
///
/// Returns the `site-packages` directory containing `openvino_genai/`. Panics
/// with an actionable message rather than a link error further down — a wrong
/// `OpenVINO` install surfaces as an opaque `undefined symbol` at link time,
/// several steps removed from the actual cause.
fn detect_ov_site_packages() -> String {
    let probe = std::process::Command::new("python3")
        .args([
            "-c",
            "import openvino_genai, os; \
             print(os.path.dirname(os.path.dirname(openvino_genai.__file__)))",
        ])
        .output();

    if let Ok(out) = probe
        && out.status.success()
    {
        let path = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        if !path.is_empty() && std::path::Path::new(&path).is_dir() {
            return path;
        }
    }

    panic!(
        "could not locate an OpenVINO GenAI install.\n\
         \n\
         Install it:      pip install openvino openvino-genai openvino-tokenizers\n\
         Or point at one: set OV_GENAI_DIR, OV_DIR, OV_TOKENIZERS_DIR and OV_INCLUDE_DIR,\n\
         \x20                or build via scripts/rv-cargo.sh with RV_OV_VENV set to the\n\
         \x20                site-packages directory that contains openvino_genai/.\n\
         \n\
         (Detection runs `python3 -c \"import openvino_genai\"`; if your install is in a\n\
         virtualenv, activate it first or use RV_OV_VENV.)"
    )
}

/// Linux: link against the pip-wheel shared objects, rpath them in.
fn build_linux() {
    // ── locate the pip-installed OpenVINO ──────────────────────────────────
    // Asks the interpreter where `openvino_genai` actually is, which is the
    // same thing scripts/rv-cargo.sh does and works for anyone who ran the
    // documented `pip install openvino openvino-genai openvino-tokenizers`.
    // This used to be a hardcoded path to one machine's venv, which meant a
    // plain `cargo build` on any other machine panicked pointing at a
    // directory that had never existed there.
    //
    // Override any or all of OV_GENAI_DIR / OV_DIR / OV_TOKENIZERS_DIR /
    // OV_INCLUDE_DIR to skip detection (scripts/rv-cargo.sh derives all four
    // from RV_OV_VENV).
    // Lazy on purpose: when all four overrides are set (the scripts/rv-cargo.sh
    // path, and any CI that pins them), detection must not run at all — probing
    // for a `python3` that may not import openvino_genai would panic on a box
    // that was already correctly configured.
    let mut detected: Option<String> = None;
    let site_packages = |detected: &mut Option<String>| -> String {
        detected.get_or_insert_with(detect_ov_site_packages).clone()
    };

    let ov_genai_lib = env::var("OV_GENAI_DIR")
        .unwrap_or_else(|_| format!("{}/openvino_genai", site_packages(&mut detected)));
    let ov_base_lib = env::var("OV_DIR")
        .unwrap_or_else(|_| format!("{}/openvino/libs", site_packages(&mut detected)));
    // openvino_tokenizers is a separate pip package whose .so is loaded by
    // libopenvino_genai at runtime as an OV extension. Its path must be in
    // DT_RPATH so the extension lookup succeeds without LD_LIBRARY_PATH.
    let ov_tokenizers_lib = env::var("OV_TOKENIZERS_DIR")
        .unwrap_or_else(|_| format!("{}/openvino_tokenizers/lib", site_packages(&mut detected)));
    // Base OpenVINO headers. Default to the pip-install location; override with
    // OV_INCLUDE_DIR when the headers live elsewhere (e.g. a dedicated 2026.2
    // venv per the OV-migration plan). Backwards-compatible — default unchanged.
    let ov_include = env::var("OV_INCLUDE_DIR")
        .unwrap_or_else(|_| format!("{}/openvino/include", site_packages(&mut detected)));

    // ── compile the C++ bridge ─────────────────────────────────────────────
    bridge_cc()
        // Base OpenVINO headers (from pip install)
        .include(&ov_include)
        // CRITICAL: match the ABI of the pip wheel.
        // openvino-genai 2026.1 pip wheel uses ABI=1 (the __cxx11 ABI),
        // confirmed by `nm -D libopenvino_genai.so.2610 | grep __cxx11`.
        // ABI=1 is the GCC default since GCC 5; we set it explicitly to
        // document the constraint and prevent accidental override.
        .define("_GLIBCXX_USE_CXX11_ABI", "1")
        .compile("ov_bridge");

    // ── link the runtime shared libraries ─────────────────────────────────
    // The pip wheels ship ONLY versioned sonames (e.g. libopenvino_genai.so.2620),
    // but the linker resolves `-lopenvino_genai` to an UNVERSIONED libopenvino_genai.so.
    // We synthesise that dev symlink in OUT_DIR from OV_GENAI_DIR / OV_DIR (which
    // follow RV_OV_VENV) on every build — so it auto-tracks whichever OV install
    // (version / box) we link against. This replaces the old hand-maintained,
    // box-local ov_bridge/lib/ symlinks + scripts/setup-ov-links.sh (no more manual
    // repoint on a 2026.1↔2026.2 bump or a cross-box switch). The
    // rerun-if-env-changed=OV_GENAI_DIR/OV_DIR triggers below regenerate the links
    // when the venv changes.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap_or_else(|_| ".".into()));
    link_unversioned(&out_dir, &ov_genai_lib, "openvino_genai");
    link_unversioned(&out_dir, &ov_base_lib, "openvino");
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // Also search the pip-wheel dirs directly (versioned-soname resolution / fallback).
    println!("cargo:rustc-link-search=native={ov_genai_lib}");
    println!("cargo:rustc-link-search=native={ov_base_lib}");

    // ── link OpenVINO shared libraries ────────────────────────────────────
    // Both are direct deps of ov_bridge.o (LLMPipeline from genai; ov::Any from
    // the base lib), so --as-needed keeps both in the NEEDED list.
    println!("cargo:rustc-link-lib=dylib=openvino_genai");
    println!("cargo:rustc-link-lib=dylib=openvino");

    // ── runtime library path for the dynamic linker ────────────────────────
    // WHY --disable-new-dtags:
    //   Modern linkers default to DT_RUNPATH (new-dtags). DT_RUNPATH is only
    //   used for direct dependencies of the binary that has it — it does NOT
    //   propagate to transitive dependencies.
    //   DT_RPATH (old-dtags) IS inherited transitively: when libopenvino_genai
    //   loads libopenvino.so.2620 and libtbb.so.12, the dynamic linker falls
    //   back to the executable's DT_RPATH to find them. This also means all
    //   dlopen() calls from loaded libraries (OV plugin loading) search DT_RPATH.
    //   --disable-new-dtags produces DT_RPATH, solving the transitive lookup.
    //
    // RPATH order — searched in declaration order:
    //   1. $ORIGIN/lib  — dist package: all OV .so files bundled next to the binary.
    //                     $ORIGIN is an ELF linker token (not a shell variable),
    //                     resolved at runtime to the directory of the running binary.
    //   2. absolute dev paths — dev build fallback when lib/ is absent.
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{ov_genai_lib}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{ov_base_lib}");
    // tokenizers extension: loaded at runtime by libopenvino_genai via
    // the OV extension mechanism (not a direct NEEDED dep of our binary)
    println!("cargo:rustc-link-arg=-Wl,-rpath,{ov_tokenizers_lib}");

    // Ensure the tokenizer .so is findable at its default location — the
    // same directory as libopenvino_genai.so — so the binary works without
    // OPENVINO_TOKENIZERS_PATH_GENAI. pip installs tokenizers in a separate
    // package dir; a relative symlink bridges the gap once at build time.
    // WHY here and not in rv-cargo.sh: build.rs already resolves both dirs;
    // doing it here means `cargo build` is the single setup step on any box.
    let tok_so = format!("{ov_tokenizers_lib}/libopenvino_tokenizers.so");
    let link = format!("{ov_genai_lib}/libopenvino_tokenizers.so");
    if std::path::Path::new(&tok_so).exists() && !std::path::Path::new(&link).exists() {
        // Relative target keeps the symlink valid if the venv moves.
        let rel = "../openvino_tokenizers/lib/libopenvino_tokenizers.so";
        if let Err(e) = std::os::unix::fs::symlink(rel, &link) {
            println!(
                "cargo:warning=tokenizer symlink {link} → {rel}: {e} \
                      (set OPENVINO_TOKENIZERS_PATH_GENAI manually if needed)"
            );
        }
    }

    rerun_triggers();
    println!("cargo:rerun-if-env-changed=OV_GENAI_DIR");
    println!("cargo:rerun-if-env-changed=OV_DIR");
    println!("cargo:rerun-if-env-changed=OV_TOKENIZERS_DIR");
    println!("cargo:rerun-if-env-changed=OV_INCLUDE_DIR");
}

/// Create an unversioned dev symlink `lib<name>.so` in `out_dir`, pointing at the
/// versioned soname (`lib<name>.so.<NNNN>`) that the `OpenVINO` pip wheel ships in
/// `lib_dir`. The linker needs the unversioned name to resolve `-l<name>`; the
/// wheels only ship the versioned file. Regenerated on every build from the
/// env-driven `lib_dir`, so it tracks whatever OV install we link against —
/// replacing the box-local, hand-maintained `ov_bridge/lib/` symlinks.
///
/// Panics (fails the build) with an explicit message if the wheel dir is
/// unreadable, has no versioned soname, or the symlink cannot be written —
/// these are unrecoverable build-config errors, so a clear panic beats a later
/// cryptic linker failure.
fn link_unversioned(out_dir: &Path, lib_dir: &str, name: &str) {
    let prefix = format!("lib{name}.so."); // e.g. "libopenvino_genai.so."
    let entries = std::fs::read_dir(lib_dir)
        .unwrap_or_else(|e| panic!("cannot read OpenVINO lib dir {lib_dir}: {e}"));
    let versioned = entries
        .filter_map(Result::ok)
        .map(|e| e.file_name())
        .find(|n| n.to_string_lossy().starts_with(&prefix))
        .unwrap_or_else(|| {
            panic!("no {prefix}* (versioned soname) in {lib_dir} — not an OpenVINO install? Check RV_OV_VENV / OV_*_DIR.")
        });
    let target = PathBuf::from(lib_dir).join(&versioned);
    let link = out_dir.join(format!("lib{name}.so"));
    // Idempotent: a stale link from a previous build/env must be replaced.
    let _ = std::fs::remove_file(&link);
    std::os::unix::fs::symlink(&target, &link).unwrap_or_else(|e| {
        panic!(
            "cannot create {} -> {}: {e}",
            link.display(),
            target.display()
        )
    });
}

/// Windows (x86_64-pc-windows-msvc, cross-compiled via cargo-xwin + clang-cl):
/// link against the import libraries (.lib) from Intel's `OpenVINO` **`GenAI`**
/// Windows archive (the `GenAI` archive — the base runtime archive lacks
/// `openvino_genai.dll`). Version MUST match the Linux pip wheel generation
/// (2026.2 on this branch): the C++ API has no cross-version ABI guarantee.
/// Build-host prerequisite: clang >= 19 (the MSVC STL headers static-assert it;
/// `~/.local/bin/clang-cl` must resolve to clang-19, not 18).
///
/// `OV_WIN_DIR` points at the extracted archive root (the directory holding
/// `runtime/`). At runtime the DLLs are NOT resolved via these paths — they
/// ship in `runtime/` next to rustedvino.exe (exe-adjacent DLLs win the
/// Windows search order; there is no rpath on Windows).
fn build_windows() {
    // Required, not guessed. There is no conventional unpack location for the
    // GenAI Windows archive — the previous default pointed into one developer's
    // home directory, which is meaningless on anyone else's machine and fails
    // later as an opaque missing-include error rather than here as a clear one.
    let ov_win_dir = env::var("OV_WIN_DIR").unwrap_or_else(|_| {
        panic!(
            "OV_WIN_DIR is not set — cannot locate the OpenVINO GenAI Windows archive.\n\
             \n\
             Download openvino_genai_windows_<version>_x86_64.zip from\n\
             \x20 https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/\n\
             extract it, and set OV_WIN_DIR to the extracted directory (the one\n\
             containing runtime/include and runtime/lib)."
        )
    });

    let ov_include = format!("{ov_win_dir}/runtime/include");
    let ov_lib = format!("{ov_win_dir}/runtime/lib/intel64/Release");

    // ── compile the C++ bridge ─────────────────────────────────────────────
    let mut build = bridge_cc();
    build
        // Full OpenVINO + GenAI header set from the archive. ov_bridge/include
        // still takes precedence (added first in bridge_cc) — same shadowing
        // order as the Linux build.
        .include(&ov_include)
        // MSVC exception model: C++ exceptions unwind across the DLL boundary
        // (GenAI throws ov::Exception); without /EHsc clang-cl assumes
        // extern-C functions never throw and the process aborts instead.
        .flag_if_supported("/EHsc");
    build.compile("ov_bridge");

    // ── link the import libraries ──────────────────────────────────────────
    // `dylib=openvino_genai` resolves to openvino_genai.lib on MSVC targets —
    // an import lib that binds the symbol to openvino_genai.dll at load time.
    println!("cargo:rustc-link-search=native={ov_lib}");
    println!("cargo:rustc-link-lib=dylib=openvino_genai");
    println!("cargo:rustc-link-lib=dylib=openvino");
    // No rpath directives: Windows has no rpath. DLL discovery is exe-adjacent
    // (the runtime/ bundle) per dev/plans/windows-x86-compat.md §2/§3.2.

    rerun_triggers();
    println!("cargo:rerun-if-env-changed=OV_WIN_DIR");
}

/// Shared `cc::Build` base — everything platform-independent about the bridge.
fn bridge_cc() -> cc::Build {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("ov_bridge/ov_bridge.cpp")
        // GenAI headers (downloaded from openvinotoolkit/openvino.genai) —
        // added BEFORE the platform include dirs so they shadow consistently.
        .include(PathBuf::from("ov_bridge/include"))
        // Optimise even in debug builds — OV inference is always hot
        .opt_level(2);
    build
}

/// Rebuild triggers common to both platforms.
fn rerun_triggers() {
    println!("cargo:rerun-if-changed=ov_bridge/ov_bridge.cpp");
    println!("cargo:rerun-if-changed=ov_bridge/include/openvino/genai/llm_pipeline.hpp");
    println!(
        "cargo:rerun-if-changed=ov_bridge/include/openvino/genai/visual_language/pipeline.hpp"
    );
    println!("cargo:rerun-if-changed=ov_bridge/include/openvino/genai/chat_history.hpp");
    println!("cargo:rerun-if-changed=ov_bridge/include/openvino/genai/whisper_pipeline.hpp");
    println!(
        "cargo:rerun-if-changed=ov_bridge/include/openvino/genai/image_generation/text2image_pipeline.hpp"
    );
    println!("cargo:rerun-if-changed=build.rs");
}
