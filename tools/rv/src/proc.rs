//! Which process is the server? Port of `Ctx.process_state` in `scripts/rv`.
//!
//! The server now reports its own PID and uptime on the admin `server` block
//! (`GET /v1/admin/models`), which is authoritative and identical on every OS.
//! The OS probes below are the fallback for servers that do not report it, and
//! for "something holds the port but is not answering". They exist on Linux
//! only; elsewhere the fallback honestly says it cannot tell.

use std::path::Path;

/// How the process was identified; the caller reports it, because the weaker
/// sources are guesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The server told us (`pid` in the admin `server` block). Authoritative:
    /// this is the process that just answered the request.
    Server,
    /// The process holding the port (confirmed to be the right instance).
    Port,
    /// Named by the PID file, NOT confirmed against the port.
    PidFile,
    /// Found by scanning command lines; the PID file is stale or absent.
    Scan,
}

/// What `rv status` knows about the server process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcState {
    /// A server process was identified.
    pub alive: bool,
    /// The supervisor PID, or the sole process when not supervised.
    pub main_pid: Option<u32>,
    /// The worker PID, only when a supervisor sits above it.
    pub worker_pid: Option<u32>,
    /// How it was identified.
    pub source: Source,
    /// Seconds up, when known.
    pub uptime_secs: Option<u64>,
}

impl ProcState {
    /// "Nothing identified."
    #[must_use]
    pub const fn not_found() -> Self {
        Self {
            alive: false,
            main_pid: None,
            worker_pid: None,
            source: Source::Scan,
            uptime_secs: None,
        }
    }
}

/// Pulls `(pid, uptime_secs)` out of a `GET /v1/admin/models` body: the
/// `server.pid` / `server.uptime_secs` fields (valid-key callers only; absent on
/// servers older than 0.8.0). The server crate pins the same field names in
/// `server_info_with_auth_reports_pid_and_uptime`.
#[must_use]
pub fn server_report(admin_models: &serde_json::Value) -> Option<(u32, Option<u64>)> {
    let server = admin_models.get("server")?;
    let pid = u32::try_from(server.get("pid")?.as_u64()?).ok()?;
    Some((
        pid,
        server
            .get("uptime_secs")
            .and_then(serde_json::Value::as_u64),
    ))
}

/// Builds the state from what the server reported. Under a supervisor the
/// reported PID is the worker's and its parent is the supervisor (Linux can see
/// that; elsewhere the process is reported as a single process).
#[must_use]
pub fn from_server_report(pid: u32, uptime_secs: Option<u64>) -> ProcState {
    let parent = platform::parent_pid(pid).filter(|p| platform::is_server_pid(*p));
    ProcState {
        alive: true,
        main_pid: Some(parent.unwrap_or(pid)),
        worker_pid: parent.map(|_| pid),
        source: Source::Server,
        uptime_secs,
    }
}

/// OS-probe fallback: port holder, then the PID file, then a command-line scan.
#[must_use]
pub fn probe(port: u16, pid_file: &Path) -> ProcState {
    platform::probe(port, pid_file)
}

/// Whether `pid` was started with `RUSTEDVINO_CONFIG` equal to `config` (which
/// `rv start` always sets). Tells "my server, still loading models" apart from
/// another server on the same machine. `false` when it cannot be read.
#[must_use]
pub fn runs_config(pid: u32, config: &Path) -> bool {
    platform::runs_config(pid, config)
}

#[cfg(target_os = "linux")]
mod platform {
    use std::collections::HashSet;
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    use super::{ProcState, Source};

    /// argv[0] basenames of the server binary: a source build runs
    /// `rustedvino`, a release bundle runs `rustedvino-bin`.
    const SERVER_NAMES: [&str; 2] = ["rustedvino", "rustedvino-bin"];

    pub fn parent_pid(pid: u32) -> Option<u32> {
        let status = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        status
            .lines()
            .find_map(|l| l.strip_prefix("PPid:"))
            .and_then(|v| v.trim().parse().ok())
    }

    /// True when `pid` is a live process whose argv[0] basename is the server
    /// binary. Guards against PID reuse: a stale PID file can name a PID the
    /// kernel has since handed to something unrelated.
    pub fn is_server_pid(pid: u32) -> bool {
        let Ok(raw) = fs::read(format!("/proc/{pid}/cmdline")) else {
            return false;
        };
        let argv0 = raw.split(|b| *b == 0).next().unwrap_or(&[]);
        let argv0 = String::from_utf8_lossy(argv0);
        let base = argv0.rsplit('/').next().unwrap_or("");
        SERVER_NAMES.contains(&base)
    }

    pub fn runs_config(pid: u32, config: &Path) -> bool {
        let Ok(raw) = fs::read(format!("/proc/{pid}/environ")) else {
            return false;
        };
        raw.split(|b| *b == 0).any(|kv| {
            kv.strip_prefix(b"RUSTEDVINO_CONFIG=")
                .is_some_and(|v| Path::new(&*String::from_utf8_lossy(v)) == config)
        })
    }

    /// PID holding a LISTEN socket on `port`, via `/proc/net/tcp{,6}` then the
    /// `/proc/<pid>/fd` links. `None` means "unknown" (notably another user's
    /// socket: its fd directory is unreadable), never "not running".
    pub(super) fn pid_listening_on(port: u16) -> Option<u32> {
        let mut inodes: HashSet<String> = HashSet::new();
        for table in ["/proc/net/tcp", "/proc/net/tcp6"] {
            let Ok(text) = fs::read_to_string(table) else {
                continue;
            };
            for line in text.lines().skip(1) {
                let f: Vec<&str> = line.split_whitespace().collect();
                // st == 0A is TCP_LISTEN; local_address is HEXIP:HEXPORT.
                if f.len() < 10 || f[3] != "0A" {
                    continue;
                }
                let local_port = f[1]
                    .rsplit(':')
                    .next()
                    .and_then(|h| u16::from_str_radix(h, 16).ok());
                if local_port == Some(port) {
                    inodes.insert(f[9].to_owned());
                }
            }
        }
        if inodes.is_empty() {
            return None;
        }
        let targets: HashSet<String> = inodes.iter().map(|i| format!("socket:[{i}]")).collect();
        for entry in fs::read_dir("/proc").ok()?.flatten() {
            let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|n| n.parse::<u32>().ok())
            else {
                continue;
            };
            let Ok(fds) = fs::read_dir(format!("/proc/{pid}/fd")) else {
                continue;
            };
            for fd in fds.flatten() {
                if let Ok(link) = fs::read_link(fd.path())
                    && targets.contains(&link.to_string_lossy().into_owned())
                {
                    return Some(pid);
                }
            }
        }
        None
    }

    /// Top of the server process tree found by command line, for when the PID
    /// file is absent or stale (the server may have been started by systemd or
    /// by hand, neither of which writes it).
    fn discover_main_pid() -> Option<u32> {
        let mut pids: Vec<u32> = Vec::new();
        for entry in fs::read_dir("/proc").ok()?.flatten() {
            if let Some(pid) = entry
                .file_name()
                .to_str()
                .and_then(|n| n.parse::<u32>().ok())
                && is_server_pid(pid)
            {
                pids.push(pid);
            }
        }
        let known: HashSet<u32> = pids.iter().copied().collect();
        pids.iter()
            .copied()
            .find(|p| parent_pid(*p).is_some_and(|pp| !known.contains(&pp)))
            .or_else(|| pids.iter().copied().min())
    }

    fn children_of(pid: u32) -> Option<u32> {
        let mut kids: Vec<u32> = Vec::new();
        for entry in fs::read_dir("/proc").ok()?.flatten() {
            if let Some(c) = entry
                .file_name()
                .to_str()
                .and_then(|n| n.parse::<u32>().ok())
                && parent_pid(c) == Some(pid)
                && is_server_pid(c)
            {
                kids.push(c);
            }
        }
        kids.into_iter().min()
    }

    /// `ps -o etimes=` — the one remaining subprocess; /proc would need the
    /// kernel tick rate, which means `sysconf` and `unsafe`.
    fn uptime_secs(pid: u32) -> Option<u64> {
        let out = Command::new("ps")
            .args(["-o", "etimes=", "-p", &pid.to_string()])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }

    pub fn probe(port: u16, pid_file: &Path) -> ProcState {
        // The listening socket belongs to the worker; a supervisor is its parent.
        if let Some(port_pid) = pid_listening_on(port).filter(|p| is_server_pid(*p)) {
            let parent = parent_pid(port_pid).filter(|p| is_server_pid(*p));
            return ProcState {
                alive: true,
                main_pid: Some(parent.unwrap_or(port_pid)),
                worker_pid: parent.map(|_| port_pid),
                source: Source::Port,
                uptime_secs: uptime_secs(parent.unwrap_or(port_pid)),
            };
        }
        let from_file = fs::read_to_string(pid_file)
            .ok()
            .and_then(|t| t.trim().parse::<u32>().ok())
            .filter(|p| is_server_pid(*p));
        let (pid, source) = match from_file {
            Some(p) => (p, Source::PidFile),
            None => match discover_main_pid() {
                Some(p) => (p, Source::Scan),
                None => return ProcState::not_found(),
            },
        };
        ProcState {
            alive: true,
            main_pid: Some(pid),
            worker_pid: children_of(pid),
            source,
            uptime_secs: uptime_secs(pid),
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use std::path::Path;

    use super::ProcState;

    pub fn parent_pid(_pid: u32) -> Option<u32> {
        None
    }

    pub fn is_server_pid(_pid: u32) -> bool {
        false
    }

    pub fn runs_config(_pid: u32, _config: &Path) -> bool {
        false
    }

    /// No OS probe off Linux yet: a server that does not report its own PID is
    /// reported as "PID unknown" by the caller, not as "not running".
    pub fn probe(_port: u16, _pid_file: &Path) -> ProcState {
        ProcState::not_found()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)] // fixtures: a failed setup should fail the test loudly
mod tests {
    use super::*;

    #[test]
    fn server_report_without_a_supervisor_is_a_single_process() {
        // PID 1 is never the rustedvino binary, so nothing above it counts.
        let s = from_server_report(1, Some(42));
        assert!(s.alive);
        assert_eq!(s.main_pid, Some(1));
        assert_eq!(s.worker_pid, None);
        assert_eq!(s.source, Source::Server);
        assert_eq!(s.uptime_secs, Some(42));
    }

    #[test]
    fn server_report_reads_pid_and_uptime_and_tolerates_old_servers() {
        let j = serde_json::json!({"models": [], "server": {"pid": 4321, "uptime_secs": 99}});
        assert_eq!(server_report(&j), Some((4321, Some(99))));
        let no_uptime = serde_json::json!({"server": {"pid": 7}});
        assert_eq!(server_report(&no_uptime), Some((7, None)));
        // A pre-0.8.0 server, or a keyless one: no pid in the block.
        assert_eq!(
            server_report(&serde_json::json!({"server": {"host": "h"}})),
            None
        );
        assert_eq!(server_report(&serde_json::json!({"models": []})), None);
        assert_eq!(
            server_report(&serde_json::json!({"server": {"pid": -1}})),
            None
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_port_scan_finds_this_very_process() {
        // Bind a listener, then ask the /proc scan who holds the port.
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = l.local_addr().expect("addr").port();
        assert_eq!(platform::pid_listening_on(port), Some(std::process::id()));
        drop(l);
        assert_eq!(platform::pid_listening_on(port), None);
    }
}
