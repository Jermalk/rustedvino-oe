//! A standalone tokenizer loaded from a model directory, usable from any thread.
//!
//! The engine-owned tokenizers (`ov_cb`, `ov_vlm`) are not thread-safe, so `/tokenize` and
//! `/detokenize` have to go through the engine thread, which for a single-stream vision model
//! means waiting for the running generation to finish (D-13). This one carries its own mutex on
//! the C++ side, so a handler can call it directly from a blocking thread while the engine is busy.

use std::ffi::{CString, c_void};
use std::path::Path;

use anyhow::Context;

use crate::ov_pipeline::last_error;

unsafe extern "C" {
    fn ov_tok_create(model_path: *const std::ffi::c_char) -> *mut c_void;
    fn ov_tok_free(handle: *mut c_void);
    fn ov_tok_encode(
        handle: *mut c_void,
        text: *const u8,
        len: usize,
        out_ids: *mut i64,
        cap: usize,
        out_count: *mut usize,
    ) -> std::ffi::c_int;
    fn ov_tok_decode(
        handle: *mut c_void,
        ids: *const i64,
        count: usize,
        callback: Option<crate::ov_cb::OvTextCallback>,
        user_data: *mut c_void,
    ) -> std::ffi::c_int;
}

/// Owning handle to a mutex-guarded `ov::genai::Tokenizer`. Freed on drop.
pub struct OvTokenizer {
    handle: *mut c_void,
}

// SAFETY: the C++ `OvTokState` serialises every encode/decode on its own mutex and the handle is
// freed exactly once, in `Drop`, which runs only after the last `Arc` clone is gone. So sharing
// and sending the pointer between threads cannot race.
unsafe impl Send for OvTokenizer {}
// SAFETY: see `Send` above - all access is behind the C++ mutex.
unsafe impl Sync for OvTokenizer {}

impl std::fmt::Debug for OvTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OvTokenizer").finish_non_exhaustive()
    }
}

impl OvTokenizer {
    /// Load the tokenizer files (`openvino_tokenizer.xml`, ...) from `model_dir`.
    ///
    /// # Errors
    /// Returns an error if the path has an interior NUL or the tokenizer cannot be loaded.
    pub fn new(model_dir: &Path) -> anyhow::Result<Self> {
        let c_path = CString::new(model_dir.to_string_lossy().as_bytes())
            .context("model path contains interior NUL")?;
        // SAFETY: `c_path` outlives the call; the function returns a heap-allocated state or NULL.
        let handle = unsafe { ov_tok_create(c_path.as_ptr()) };
        if handle.is_null() {
            anyhow::bail!("ov_tok_create failed: {}", last_error());
        }
        Ok(Self { handle })
    }

    /// Encode `text` into token ids (two passes: count, then an exactly sized fill).
    ///
    /// # Errors
    /// Returns an error if the tokenizer fails.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        let mut count: usize = 0;
        // SAFETY: handle non-null (invariant of `new`); `text` is a valid `len`-byte buffer; no id
        // buffer requested (null/cap 0); `out_count` points to a live usize.
        let ret = unsafe {
            ov_tok_encode(
                self.handle,
                text.as_ptr(),
                text.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::addr_of_mut!(count),
            )
        };
        if ret != 0 {
            anyhow::bail!("ov_tok_encode (count) failed: {}", last_error());
        }
        let n = count;
        let mut ids = vec![0i64; n];
        if n == 0 {
            return Ok(ids);
        }
        // SAFETY: as above; `ids` has `n` slots and `cap == n`.
        let ret = unsafe {
            ov_tok_encode(
                self.handle,
                text.as_ptr(),
                text.len(),
                ids.as_mut_ptr(),
                n,
                std::ptr::addr_of_mut!(count),
            )
        };
        if ret != 0 {
            anyhow::bail!("ov_tok_encode failed: {}", last_error());
        }
        ids.truncate(count.min(n));
        Ok(ids)
    }

    /// Decode token ids back to text.
    ///
    /// # Errors
    /// Returns an error if an id is outside the vocabulary (the message starts with
    /// `invalid token id:` at that point) or the detokenizer fails.
    pub fn decode(&self, ids: &[i64]) -> anyhow::Result<String> {
        crate::ov_cb::decode_with(|callback, user_data| {
            // SAFETY: handle non-null; `ids`/`len` describe a valid slice (possibly empty, which
            // the C side allows); `callback`/`user_data` come from `decode_with`, which keeps the
            // context alive for the whole blocking call.
            unsafe {
                ov_tok_decode(
                    self.handle,
                    ids.as_ptr(),
                    ids.len(),
                    Some(callback),
                    user_data,
                )
            }
        })
        .map_err(|e| anyhow::anyhow!("ov_tok_decode failed: {e}"))
    }
}

impl Drop for OvTokenizer {
    fn drop(&mut self) {
        // SAFETY: `handle` is the unique owner, created by `ov_tok_create`; this is the only
        // free site.
        unsafe { ov_tok_free(self.handle) };
    }
}
