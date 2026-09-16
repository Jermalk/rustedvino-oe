// ============================================================
// src/streaming.rs — inference → SSE channel bridge
// ============================================================
// This module owns the data flow between the OV inference thread
// and the axum SSE handler. It is the core of the performance case:
//
//   Python today:
//     OV callback → call_soon_threadsafe(queue.put_nowait)
//     → asyncio wake-up → GIL grab → json.dumps → TCP write
//     ≈ 7 ms per token
//
//   Rust (this module):
//     OV callback → tx.blocking_send(token)     ← one channel write
//     → ReceiverStream wakes the axum handler    ← no GIL, no wake-up
//     → serde_json → TCP write
//     ≈ 0.5–1 ms per token (target)
//
// CRASH COURSE — tokio::sync::mpsc:
//   mpsc = "multi-producer, single-consumer".
//   Sender<T>: cloneable, can be moved across tasks/threads.
//   Receiver<T>: one owner only (not Clone).
//   channel(N): N = capacity before backpressure (send blocks).
//
//   Two flavours:
//     send().await      — async send; yields if channel full.
//     blocking_send()   — sync send for non-async contexts.
//                         Used by the OV callback (inside
//                         spawn_blocking, never inside async fn).
//
//   The OV callback fires on a C++ thread managed by the runtime.
//   blocking_send is the only safe choice there.
// ============================================================

use tokio::sync::mpsc;

use crate::ov_cb::FinishReason;

// ---- Channel types --------------------------------------------------

/// Capacity of the in-flight token buffer.
///
/// 256 tokens ≈ 1 KB of string data. Negligible RAM cost.
/// Large enough that the inference thread never blocks waiting
/// for the SSE handler to flush to the client.
pub const CHANNEL_CAPACITY: usize = 256;

/// Events flowing from the inference thread to the SSE handler.
///
/// The OV Streamer callback produces these; the axum chat handler
/// consumes them and converts to SSE frames.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// The exact prompt token count for this request, counted by the engine
    /// thread (which owns the tokenizer) right after the request is accepted.
    /// Sent once, before any `Token`. The non-streaming collector folds it into
    /// `usage.prompt_tokens`; the SSE path drops it (until `stream_options
    /// .include_usage`, Phase 3.7). Absent on the mock path (no tokenizer).
    PromptTokens(usize),
    /// A generated token (or sub-word piece) plus how many real tokens this
    /// delta represents. Plain decoding: always `1`. Speculative decoding
    /// (CB engine, draft attached): can be `>1` — one verification step can
    /// accept several draft tokens at once, landing in the same delta. Usage
    /// accounting must sum this field, not count `Token` events — counting
    /// events silently undercounts `usage.completion_tokens` under
    /// speculative decoding (found live 2026-07-19, see the project's internal engineering log).
    Token(String, usize),
    /// Normal end of generation. Carries why generation stopped so the
    /// `OpenAI` `finish_reason` field reflects EOS vs the token budget.
    Done(FinishReason),
    /// Inference failed. The SSE stream will include the error and close.
    // Constructed by the OV error path in Phase 1b; defined now so
    // stream_event_to_sse() handles it and tests can exercise it.
    #[allow(dead_code)]
    Error(String),
}

/// The sending half of the token channel.
///
/// Moved into the inference closure; call `blocking_send` from the
/// OV Streamer callback (which runs synchronously on the blocking thread).
pub type TokenSender = mpsc::Sender<StreamEvent>;

/// The receiving half of the token channel.
///
/// Wrapped with `ReceiverStream` in the chat handler to become a
/// `futures::Stream` that axum's `Sse` can drive.
pub type TokenReceiver = mpsc::Receiver<StreamEvent>;

/// Creates a (sender, receiver) pair for one inference request.
///
/// Call once per request; do not share across requests.
#[must_use]
pub fn stream_channel() -> (TokenSender, TokenReceiver) {
    mpsc::channel(CHANNEL_CAPACITY)
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    // CRASH COURSE — #[tokio::test]:
    //   Identical to #[test] but spins up a Tokio runtime for the
    //   test function. Required whenever the test uses .await.
    //   Without it: "async fn used as test without runtime" compile error.

    /// Tokens sent on the sender arrive on the receiver in order.
    ///
    /// This verifies the core channel plumbing before wiring in OV.
    /// If this test fails, nothing else will work.
    #[tokio::test]
    async fn channel_delivers_tokens_in_order() {
        let (tx, mut rx) = stream_channel();

        tx.send(StreamEvent::Token("Hello".into(), 1))
            .await
            .unwrap();
        tx.send(StreamEvent::Token(" world".into(), 1))
            .await
            .unwrap();
        tx.send(StreamEvent::Done(FinishReason::Stop))
            .await
            .unwrap();

        // CRASH COURSE — `if let Some(x) = option`:
        //   Pattern-match on Option<T>. If it's Some, bind x and
        //   execute the block. The None branch is silently skipped.
        //   Equivalent to Python's `if (x := option) is not None`.
        let first = rx.recv().await.unwrap();
        assert!(
            matches!(first, StreamEvent::Token(t, _) if t == "Hello"),
            "first event must be Token(\"Hello\")"
        );

        let second = rx.recv().await.unwrap();
        assert!(
            matches!(second, StreamEvent::Token(t, _) if t == " world"),
            "second event must be Token(\" world\")"
        );

        let third = rx.recv().await.unwrap();
        assert!(
            matches!(third, StreamEvent::Done(FinishReason::Stop)),
            "third event must be Done(Stop)"
        );
    }

    /// When the sender is dropped, the receiver returns None (channel closed).
    ///
    /// This is the normal shutdown path: the OV thread finishes and drops tx.
    /// The SSE handler's `ReceiverStream` terminates naturally — no explicit
    /// "close" signal needed. The channel closing IS the signal.
    #[tokio::test]
    async fn channel_closes_when_sender_dropped() {
        let (tx, mut rx) = stream_channel();

        tx.send(StreamEvent::Token("last".into(), 1)).await.unwrap();
        drop(tx); // sender gone — channel will drain then close

        let event = rx.recv().await.unwrap();
        assert!(matches!(event, StreamEvent::Token(t, _) if t == "last"));

        // After the sender drops, recv() returns None
        let after_close = rx.recv().await;
        assert!(
            after_close.is_none(),
            "recv() must return None after sender is dropped"
        );
    }

    /// `blocking_send` works from a `std::thread` (simulates the OV callback).
    ///
    /// The OV Streamer callback fires on a thread managed by the C++
    /// runtime — not a tokio task. `blocking_send` is the correct API there.
    /// This test proves it works before we wire in real OV.
    #[tokio::test]
    async fn blocking_send_from_std_thread_delivers_token() {
        let (tx, mut rx) = stream_channel();

        // Spawn a plain OS thread — no tokio runtime, no async
        let handle = std::thread::spawn(move || {
            // This is what the OV Streamer callback looks like
            tx.blocking_send(StreamEvent::Token("from_thread".into(), 1))
                .unwrap();
        });

        handle.join().unwrap();

        let event = rx.recv().await.unwrap();
        assert!(
            matches!(event, StreamEvent::Token(t, _) if t == "from_thread"),
            "token sent via blocking_send must arrive on the async receiver"
        );
    }
}
