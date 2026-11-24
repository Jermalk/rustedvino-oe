// ============================================================
// src/log_ring.rs — bounded in-memory tail of recent log lines
// ============================================================
// Lets an operator (or a UI like Pyramu) fetch recent server logs over HTTP
// without depending on how the process's stdout happens to be redirected —
// a file under `nohup`, a terminal, systemd/journald — none of which this
// process can discover or read back from. A `tracing_subscriber::fmt` writer
// duplicates every formatted line into a small ring buffer that the admin
// API serves directly.
// ============================================================

use std::collections::VecDeque;
use std::io;
use std::sync::{Arc, Mutex};

/// How many recent log lines to retain. Generous enough to cover a burst of
/// activity between polls without holding meaningfully more than a few
/// hundred KB (formatted lines run well under 1 KB each in practice).
const CAPACITY: usize = 2000;

/// Bounded, thread-safe ring buffer of recent formatted log lines.
///
/// Cheap to clone (an `Arc` internally) — one instance is created in `main`
/// and shared between the tracing subscriber (which pushes) and `AppState`
/// (which reads via [`LogRingBuffer::tail`]).
#[derive(Clone)]
pub struct LogRingBuffer {
    lines: Arc<Mutex<VecDeque<String>>>,
}

impl Default for LogRingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl LogRingBuffer {
    /// Creates an empty buffer with the default [`CAPACITY`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            lines: Arc::new(Mutex::new(VecDeque::with_capacity(CAPACITY))),
        }
    }

    /// Appends one line, evicting the oldest when at capacity.
    fn push_line(&self, line: &str) {
        // A poisoned lock (a prior panic while holding it) still holds a
        // usable `VecDeque` — recovering it is strictly better than losing
        // every subsequent log line to a second panic here.
        let mut guard = self
            .lines
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.len() >= CAPACITY {
            guard.pop_front();
        }
        guard.push_back(line.to_owned());
    }

    /// Returns up to the last `n` lines, oldest first (append-friendly order,
    /// like `tail -n`). `n` is silently clamped to the number of lines held.
    #[must_use]
    pub fn tail(&self, n: usize) -> Vec<String> {
        let guard = self
            .lines
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let skip = guard.len().saturating_sub(n);
        guard.iter().skip(skip).cloned().collect()
    }
}

/// A `tracing_subscriber::fmt` writer that duplicates every formatted event
/// to both stdout (unchanged behaviour) and the ring buffer. `W` is the
/// pass-through sink: stdout in the server, a test double in the tests.
pub struct TeeWriter<W = io::Stdout> {
    buffer: LogRingBuffer,
    sink: W,
}

impl<W: io::Write> io::Write for TeeWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // `fmt`'s writer receives one already-formatted event (including its
        // trailing newline) per `write` call — treat each call as one line.
        // Lossy on non-UTF8 bytes (should not occur; formatted log text is
        // always valid UTF-8) rather than dropping the line entirely.
        self.buffer
            .push_line(String::from_utf8_lossy(buf).trim_end_matches('\n'));
        // FM-8: the whole event goes to the sink here and the whole length is
        // reported. Returning a short count from a partial stdout write (a
        // console writing in bounded chunks) made the caller's `write_all`
        // call again with the rest, which was pushed as a second "line".
        self.sink.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.sink.flush()
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogRingBuffer {
    type Writer = TeeWriter;

    fn make_writer(&'a self) -> Self::Writer {
        TeeWriter {
            buffer: self.clone(),
            sink: io::stdout(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn empty_buffer_tails_nothing() {
        let buf = LogRingBuffer::new();
        assert!(buf.tail(10).is_empty());
    }

    #[test]
    fn tail_returns_oldest_first_up_to_n() {
        let buf = LogRingBuffer::new();
        for i in 0..5 {
            buf.push_line(&format!("line {i}"));
        }
        assert_eq!(
            buf.tail(3),
            vec![
                "line 2".to_owned(),
                "line 3".to_owned(),
                "line 4".to_owned()
            ]
        );
    }

    #[test]
    fn tail_larger_than_held_returns_everything() {
        let buf = LogRingBuffer::new();
        buf.push_line("only line");
        assert_eq!(buf.tail(100), vec!["only line".to_owned()]);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let buf = LogRingBuffer::new();
        for i in 0..(CAPACITY + 10) {
            buf.push_line(&format!("line {i}"));
        }
        let all = buf.tail(CAPACITY + 10);
        assert_eq!(all.len(), CAPACITY, "buffer must not exceed its capacity");
        assert_eq!(
            all.first(),
            Some(&"line 10".to_owned()),
            "oldest 10 lines evicted"
        );
        assert_eq!(all.last(), Some(&format!("line {}", CAPACITY + 9)));
    }

    #[test]
    fn writer_passes_bytes_through_and_records_the_line() {
        let buf = LogRingBuffer::new();
        let mut writer = <LogRingBuffer as tracing_subscriber::fmt::MakeWriter>::make_writer(&buf);
        let n = io::Write::write(&mut writer, b"hello world\n").unwrap();
        assert_eq!(n, 12);
        assert_eq!(buf.tail(1), vec!["hello world".to_owned()]);
    }

    /// Accepts at most 10 bytes per `write`, like a console that writes in
    /// bounded chunks.
    struct Chunked(Vec<u8>);

    impl io::Write for Chunked {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let n = buf.len().min(10);
            self.0.extend_from_slice(&buf[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    /// FM-8: a long event through a sink that takes 10 bytes per call is one
    /// ring line, and the sink still receives every byte once.
    #[test]
    fn long_event_through_a_partial_writer_is_one_line() {
        let buf = LogRingBuffer::new();
        let event = format!("{}\n", "x".repeat(5000));
        let mut writer = TeeWriter {
            buffer: buf.clone(),
            sink: Chunked(Vec::new()),
        };
        io::Write::write_all(&mut writer, event.as_bytes()).unwrap();
        assert_eq!(buf.tail(10), vec!["x".repeat(5000)]);
        assert_eq!(writer.sink.0, event.as_bytes());
    }
}
