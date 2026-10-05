//! Thin blocking HTTP client for the admin API. Port of `Api` in `scripts/rv`:
//! raw JSON/text/error, no formatting.

use std::time::Duration;

use serde_json::Value;

/// Timeout for read-only calls (`READ_TIMEOUT` in the Python script).
pub const READ_TIMEOUT: Duration = Duration::from_secs(5);
/// Timeout for load/unload/deregister, which can legitimately take 10-60 s+.
pub const MUTATE_TIMEOUT: Duration = Duration::from_secs(300);

/// A response body: JSON when it parses, otherwise the raw text.
#[derive(Debug, Clone, PartialEq)]
pub enum Payload {
    /// Parsed JSON.
    Json(Value),
    /// Not JSON (the log tail, `/metrics`).
    Text(String),
}

impl Payload {
    /// The JSON value, if this is JSON.
    #[must_use]
    pub fn json(&self) -> Option<&Value> {
        match self {
            Self::Json(v) => Some(v),
            Self::Text(_) => None,
        }
    }

    /// The raw text, if this is text.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(t) => Some(t),
            Self::Json(_) => None,
        }
    }

    /// As a JSON value for `--json` output: text becomes a JSON string.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match self {
            Self::Json(v) => v.clone(),
            Self::Text(t) => Value::String(t.clone()),
        }
    }
}

/// Outcome of one request, shaped like the Python `(code, body, err)` tuple.
#[derive(Debug, Clone, Default)]
pub struct Reply {
    /// HTTP status; `None` when no response arrived (connection error).
    pub code: Option<u16>,
    /// Body of a 2xx response.
    pub body: Option<Payload>,
    /// Error text: the body of a non-2xx response, or the transport error.
    pub err: Option<String>,
}

impl Reply {
    /// Whether the status was 200.
    #[must_use]
    pub fn is_200(&self) -> bool {
        self.code == Some(200)
    }

    /// The JSON body, if any.
    #[must_use]
    pub fn json(&self) -> Option<&Value> {
        self.body.as_ref().and_then(Payload::json)
    }
}

/// Admin API client: base URL plus an optional bearer key.
#[derive(Debug, Clone)]
pub struct Api {
    base: String,
    admin_key: Option<String>,
    client: reqwest::blocking::Client,
}

impl Api {
    /// Builds a client. Proxies are disabled: the target is always localhost.
    ///
    /// # Errors
    ///
    /// Fails only if the HTTP client cannot be constructed.
    pub fn new(base: String, admin_key: Option<String>) -> anyhow::Result<Self> {
        let client = reqwest::blocking::Client::builder().no_proxy().build()?;
        Ok(Self {
            base,
            admin_key,
            client,
        })
    }

    /// The same client without credentials (used to probe auth enforcement).
    #[must_use]
    pub fn keyless(&self) -> Self {
        Self {
            base: self.base.clone(),
            admin_key: None,
            client: self.client.clone(),
        }
    }

    fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        timeout: Duration,
        body: Option<&Value>,
    ) -> Reply {
        let mut req = self
            .client
            .request(method, format!("{}{path}", self.base))
            .timeout(timeout);
        if let Some(b) = body {
            req = req.json(b);
        }
        if let Some(k) = &self.admin_key {
            req = req.bearer_auth(k);
        }
        let resp = match req.send() {
            Ok(r) => r,
            Err(e) => {
                return Reply {
                    err: Some(describe_error(&e)),
                    ..Reply::default()
                };
            }
        };
        let status = resp.status();
        let code = Some(status.as_u16());
        let text = match resp.text() {
            Ok(t) => t,
            Err(e) => {
                return Reply {
                    code,
                    err: Some(describe_error(&e)),
                    ..Reply::default()
                };
            }
        };
        if status.is_success() {
            let body = serde_json::from_str::<Value>(&text)
                .map_or_else(|_| Payload::Text(text.clone()), Payload::Json);
            Reply {
                code,
                body: Some(body),
                err: None,
            }
        } else {
            let reason = status.canonical_reason().unwrap_or("error").to_owned();
            Reply {
                code,
                body: None,
                err: Some(if text.is_empty() { reason } else { text }),
            }
        }
    }

    /// `GET path` with the read timeout.
    #[must_use]
    pub fn get(&self, path: &str) -> Reply {
        self.request(reqwest::Method::GET, path, READ_TIMEOUT, None)
    }

    /// `DELETE path` with the mutate timeout.
    #[must_use]
    pub fn delete(&self, path: &str) -> Reply {
        self.request(reqwest::Method::DELETE, path, MUTATE_TIMEOUT, None)
    }

    /// `PATCH path` with a JSON body and the mutate timeout.
    #[must_use]
    pub fn patch(&self, path: &str, body: &Value) -> Reply {
        self.request(reqwest::Method::PATCH, path, MUTATE_TIMEOUT, Some(body))
    }

    /// `POST path` with an optional JSON body and an explicit timeout.
    #[must_use]
    pub fn post(&self, path: &str, body: Option<&Value>, timeout: Duration) -> Reply {
        self.request(reqwest::Method::POST, path, timeout, body)
    }
}

/// The root cause of a transport error, not reqwest's generic wrapper:
/// "Connection refused (os error 111)" is useful, "error sending request" is not.
fn describe_error(e: &reqwest::Error) -> String {
    let mut parts = vec![e.to_string()];
    let mut src: Option<&dyn std::error::Error> = std::error::Error::source(e);
    while let Some(s) = src {
        parts.push(s.to_string());
        src = s.source();
    }
    parts.pop().unwrap_or_default()
}

/// Percent-encodes a model id for use as one URL path segment.
#[must_use]
pub fn encode_segment(s: &str) -> String {
    use std::fmt::Write;
    // `.` and `..` are unreserved characters but URL parsers collapse them as
    // path segments, silently turning `/models/../check` into another route.
    if s == "." || s == ".." {
        return s.replace('.', "%2E");
    }
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::expect_used)] // test stub: a failed setup should fail the test loudly
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use super::*;

    /// Serves each canned `(status line, content type, body)` once, in order,
    /// on an ephemeral port, then returns that port.
    fn stub(replies: Vec<(&'static str, &'static str, &'static str)>) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        std::thread::spawn(move || {
            for (status, ctype, body) in replies {
                let Ok((mut sock, _)) = listener.accept() else {
                    return;
                };
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf);
                let _ = write!(
                    sock,
                    "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\
                     Connection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        port
    }

    fn api(port: u16) -> Api {
        Api::new(format!("http://127.0.0.1:{port}"), Some("k".into())).expect("client")
    }

    #[test]
    fn json_text_and_error_replies_are_shaped_like_the_python_tuple() {
        let port = stub(vec![
            ("200 OK", "application/json", r#"{"a":1}"#),
            ("200 OK", "text/plain", "line1\nline2"),
            (
                "401 Unauthorized",
                "application/json",
                r#"{"error":"nope"}"#,
            ),
            ("503 Service Unavailable", "text/plain", ""),
        ]);
        let a = api(port);
        let r = a.get("/x");
        assert_eq!(r.code, Some(200));
        assert_eq!(r.json().and_then(|v| v["a"].as_i64()), Some(1));
        let r = a.get("/x");
        assert_eq!(
            r.body.as_ref().and_then(Payload::text),
            Some("line1\nline2")
        );
        let r = a.get("/x");
        assert_eq!((r.code, r.body.is_none()), (Some(401), true));
        assert_eq!(r.err.as_deref(), Some(r#"{"error":"nope"}"#));
        let r = a.get("/x");
        assert_eq!(r.code, Some(503));
        assert_eq!(r.err.as_deref(), Some("Service Unavailable"));
    }

    #[test]
    fn connection_refused_is_a_transport_error_with_no_status() {
        // Bind then drop to learn a port that nothing listens on.
        let port = TcpListener::bind("127.0.0.1:0")
            .expect("bind")
            .local_addr()
            .expect("addr")
            .port();
        let r = api(port).get("/health");
        assert_eq!(r.code, None);
        assert!(r.err.is_some_and(|e| !e.is_empty()));
    }

    #[test]
    fn segment_encoding_keeps_model_ids_readable() {
        assert_eq!(encode_segment("qwen3-8b-int4-ov"), "qwen3-8b-int4-ov");
        assert_eq!(encode_segment("a b/c"), "a%20b%2Fc");
        assert_eq!(encode_segment(".."), "%2E%2E");
        assert_eq!(encode_segment("a..b"), "a..b");
        assert_eq!(
            encode_segment("LCM_Dreamshaper_v7.fp16"),
            "LCM_Dreamshaper_v7.fp16"
        );
    }

    #[test]
    fn payload_accessors() {
        let j = Payload::Json(serde_json::json!({"a": 1}));
        assert!(j.json().is_some() && j.text().is_none());
        let t = Payload::Text("x".into());
        assert_eq!(t.to_value(), Value::String("x".into()));
    }
}
