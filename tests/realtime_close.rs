//! 0.9.1 C5: every server-initiated close of a `/v1/realtime` WebSocket sends a
//! close frame with the code and reason from the close table; a client close
//! gets no server frame of its own (only tungstenite's automatic echo).
//!
//! Each test serves the real router on `127.0.0.1:0` and connects a real
//! WebSocket client. `AppState::default()` has no model manager, so a session
//! runs without models and no audio is ever sent.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::{Duration, Instant};

use futures_util::{SinkExt as _, StreamExt as _};
use rustedvino::app_state::AppState;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tower::ServiceExt as _;

type Client =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Long enough that idle never wins in the tests that are not about idle.
const LONG_IDLE: Duration = Duration::from_mins(10);

/// Serve the router for `state` on an ephemeral loopback port; returns the
/// router (for in-process admin requests on the same state) and the address.
async fn serve(state: AppState) -> (axum::Router, std::net::SocketAddr) {
    let app = rustedvino::create_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let served = app.clone();
    tokio::spawn(async move {
        axum::serve(listener, served).await.unwrap();
    });
    (app, addr)
}

async fn connect(addr: std::net::SocketAddr) -> Client {
    let (ws, _resp) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/realtime"))
        .await
        .unwrap();
    ws
}

/// Wait until the registry holds exactly `n` sessions (the entry is inserted
/// inside `on_upgrade`, after the client already saw the 101).
async fn wait_for_sessions(state: &AppState, n: usize, within: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < within {
        if state.realtime_sessions.read().unwrap().len() == n {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    false
}

fn only_session_id(state: &AppState) -> uuid::Uuid {
    let reg = state.realtime_sessions.read().unwrap();
    assert_eq!(reg.len(), 1);
    *reg.keys().next().unwrap()
}

/// Read until the server's close frame; return its code and reason. Keeps
/// reading after it, so tungstenite sends the client's close reply, and
/// asserts the stream then ends.
async fn read_close(ws: &mut Client) -> (u16, String) {
    let mut got = None;
    let deadline = Duration::from_secs(5);
    loop {
        match tokio::time::timeout(deadline, ws.next())
            .await
            .expect("server did not close the connection")
        {
            Some(Ok(Message::Close(frame))) => {
                let frame = frame.expect("close frame without a code");
                got = Some((u16::from(frame.code), frame.reason.as_str().to_owned()));
            }
            Some(Ok(_)) => {}
            Some(Err(e)) => panic!("connection error instead of a clean close: {e}"),
            None => break,
        }
    }
    got.expect("stream ended without a close frame")
}

#[tokio::test]
async fn idle_timeout_closes_with_1000_idle_timeout() {
    let state = AppState {
        realtime_idle_timeout: Duration::from_millis(300),
        ..AppState::default()
    };
    let (_app, addr) = serve(state.clone()).await;
    let mut ws = connect(addr).await;
    let (code, reason) = read_close(&mut ws).await;
    assert_eq!((code, reason.as_str()), (1000, "idle_timeout"));
    assert!(
        wait_for_sessions(&state, 0, Duration::from_secs(3)).await,
        "session not removed after the close handshake"
    );
}

#[tokio::test]
async fn server_stop_closes_with_1001_server_stopping() {
    let state = AppState {
        realtime_idle_timeout: LONG_IDLE,
        ..AppState::default()
    };
    let (_app, addr) = serve(state.clone()).await;
    let mut ws = connect(addr).await;
    assert!(wait_for_sessions(&state, 1, Duration::from_secs(3)).await);
    let started = Instant::now();
    // What `startup::serve` calls when the shutdown signal arrives.
    rustedvino::realtime_types::stop_realtime_sessions(&state.realtime_sessions);
    let (code, reason) = read_close(&mut ws).await;
    assert_eq!((code, reason.as_str()), (1001, "server_stopping"));
    assert!(wait_for_sessions(&state, 0, Duration::from_secs(3)).await);
    // Ended by the client's close reply, not by the 2 s wait running out.
    assert!(started.elapsed() < Duration::from_millis(1500));
}

#[tokio::test]
async fn admin_kill_closes_with_4000_closed_by_admin() {
    let state = AppState {
        realtime_idle_timeout: LONG_IDLE,
        ..AppState::default()
    };
    let (app, addr) = serve(state.clone()).await;
    let mut ws = connect(addr).await;
    assert!(wait_for_sessions(&state, 1, Duration::from_secs(3)).await);
    let id = only_session_id(&state);
    let started = Instant::now();
    // The documented route, on the same state (no keys configured = open admin).
    let resp = app
        .oneshot(
            axum::http::Request::delete(format!("/v1/admin/realtime/sessions/{id}"))
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), axum::http::StatusCode::NO_CONTENT);
    let (code, reason) = read_close(&mut ws).await;
    assert_eq!((code, reason.as_str()), (4000, "closed_by_admin"));
    assert!(wait_for_sessions(&state, 0, Duration::from_secs(3)).await);
    assert!(started.elapsed() < Duration::from_millis(1500));
}

#[tokio::test]
async fn client_close_gets_no_server_close_frame() {
    let state = AppState {
        realtime_idle_timeout: LONG_IDLE,
        ..AppState::default()
    };
    let (_app, addr) = serve(state.clone()).await;
    let mut ws = connect(addr).await;
    assert!(wait_for_sessions(&state, 1, Duration::from_secs(3)).await);
    ws.send(Message::Close(Some(CloseFrame {
        code: CloseCode::from(4100),
        reason: "client_bye".into(),
    })))
    .await
    .unwrap();
    let started = Instant::now();
    // The only close the client may see is the echo of its own frame.
    let (code, reason) = read_close(&mut ws).await;
    assert_eq!((code, reason.as_str()), (4100, "client_bye"));
    // The session ends at once: teardown does not sit out the 2 s close wait.
    assert!(wait_for_sessions(&state, 0, Duration::from_secs(3)).await);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "client close took {:?} to end the session",
        started.elapsed()
    );
}

#[tokio::test]
async fn network_drop_ends_the_session_without_waiting() {
    let state = AppState {
        realtime_idle_timeout: LONG_IDLE,
        ..AppState::default()
    };
    let (_app, addr) = serve(state.clone()).await;
    let ws = connect(addr).await;
    assert!(wait_for_sessions(&state, 1, Duration::from_secs(3)).await);
    let started = Instant::now();
    drop(ws); // TCP gone, no close handshake
    assert!(wait_for_sessions(&state, 0, Duration::from_secs(3)).await);
    assert!(started.elapsed() < Duration::from_secs(1));
}

/// Review F2: the post-drain step `startup::serve` runs (stop again + bounded
/// wait), against a live session: the client gets 1001 and the step returns
/// `true` well inside `REALTIME_STOP_WAIT` once the handshake is done.
#[tokio::test]
async fn serve_post_drain_step_closes_live_sessions_with_1001() {
    let state = AppState {
        realtime_idle_timeout: LONG_IDLE,
        ..AppState::default()
    };
    let (_app, addr) = serve(state.clone()).await;
    let mut ws = connect(addr).await;
    assert!(wait_for_sessions(&state, 1, Duration::from_secs(3)).await);
    let started = Instant::now();
    let step = {
        let registry = std::sync::Arc::clone(&state.realtime_sessions);
        tokio::spawn(async move { rustedvino::startup::close_realtime_sessions(&registry).await })
    };
    let (code, reason) = read_close(&mut ws).await;
    assert_eq!((code, reason.as_str()), (1001, "server_stopping"));
    assert!(
        step.await.unwrap(),
        "sessions still open after the stop wait"
    );
    assert!(started.elapsed() < rustedvino::realtime_types::REALTIME_STOP_WAIT);
    assert!(state.realtime_sessions.read().unwrap().is_empty());
}

/// Review F6: a session that upgrades after the stop signal starts stopped and
/// closes with 1001 at once, instead of sitting out the stop wait.
#[tokio::test]
async fn a_session_opened_after_the_stop_closes_with_1001_at_once() {
    let state = AppState {
        realtime_idle_timeout: LONG_IDLE,
        ..AppState::default()
    };
    let (_app, addr) = serve(state.clone()).await;
    rustedvino::realtime_types::stop_realtime_sessions(&state.realtime_sessions);
    let started = Instant::now();
    let mut ws = connect(addr).await;
    let (code, reason) = read_close(&mut ws).await;
    assert_eq!((code, reason.as_str()), (1001, "server_stopping"));
    assert!(wait_for_sessions(&state, 0, Duration::from_secs(3)).await);
    assert!(started.elapsed() < Duration::from_millis(1500));
}

/// Review F5: a client that floods the server and never reads parks the
/// receiver in `send_event` on a full outgoing queue. An admin kill must still
/// end the session within the close wait (it used to hang forever on the
/// receiver: a zombie registry entry).
#[tokio::test]
async fn a_client_that_stopped_reading_cannot_keep_a_killed_session_alive() {
    let state = AppState {
        realtime_idle_timeout: LONG_IDLE,
        ..AppState::default()
    };
    let (_app, addr) = serve(state.clone()).await;
    let mut ws = connect(addr).await;
    assert!(wait_for_sessions(&state, 1, Duration::from_secs(3)).await);
    // Each frame is an unknown event type; the `error` event the server sends
    // back echoes the name, so every reply is ~8 KB and the socket buffers and
    // the 64-message queue fill quickly. Send until the server stops reading.
    let frame = format!(r#"{{"type":"x{}"}}"#, "y".repeat(8 * 1024));
    let mut stalled = false;
    for _ in 0..20_000 {
        let send = ws.send(Message::Text(frame.clone().into()));
        if tokio::time::timeout(Duration::from_millis(500), send)
            .await
            .is_err()
        {
            stalled = true;
            break;
        }
    }
    assert!(
        stalled,
        "the server never stopped reading: the queue did not fill"
    );
    let started = Instant::now();
    let id = only_session_id(&state);
    state.realtime_sessions.read().unwrap()[&id].kill.cancel();
    assert!(
        wait_for_sessions(&state, 0, Duration::from_secs(5)).await,
        "killed session still registered: teardown hung on the receiver"
    );
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "{:?}",
        started.elapsed()
    );
}
