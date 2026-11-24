// ============================================================
// src/cli.rs — the server binary's command line (FM-1)
// ============================================================
// The binary takes a handful of flags and no values. Every argument is
// classified up front, before any config, signal thread or runtime: before
// this, an unknown argument (`--help`, `/?`, a mistyped `--list-device`,
// `--config x.json`) was ignored and the server started for real, with
// whatever config the search found. Hand-rolled (no parser crate): the
// set is small and fixed.
// ============================================================

use std::ffi::OsString;

/// Exit code for a command line the binary does not understand (`EX_USAGE`).
pub const EXIT_USAGE: u8 = 64;

/// Every argument the server binary accepts. `--gpu-probe` is internal (the supervisor's GPU
/// readiness probe child); the rest are listed in [`usage`].
pub const KNOWN_ARGS: [&str; 10] = [
    "--help",
    "-h",
    "--version",
    "-V",
    "--list-devices",
    "--device-info",
    "--print-default-config",
    "--print-default-model",
    "--supervise",
    "--gpu-probe",
];

/// What the command line asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CliAction {
    /// Only known flags (or none): carry on; `main` handles each flag as before.
    Proceed,
    /// `--help` / `-h`: print [`usage`] on stdout, exit 0.
    Help,
    /// An argument the binary does not know (shown lossily if it is not UTF-8): print it and
    /// [`usage`] on stderr, exit [`EXIT_USAGE`].
    Unknown(String),
}

/// Classifies the arguments after the program name. An unknown argument wins over `--help`
/// (the command line is wrong either way); a non-UTF-8 argument is unknown, so every argument
/// that reaches the rest of `main` is valid UTF-8 and `std::env::args()` cannot panic on it.
#[must_use]
pub fn classify_args(args: &[OsString]) -> CliAction {
    let mut help = false;
    for arg in args {
        match arg.to_str() {
            Some(a) if KNOWN_ARGS.contains(&a) => help |= a == "--help" || a == "-h",
            _ => return CliAction::Unknown(arg.to_string_lossy().into_owned()),
        }
    }
    if help {
        CliAction::Help
    } else {
        CliAction::Proceed
    }
}

/// The `--help` text: every public flag, the config search order and the exit codes.
#[must_use]
pub fn usage() -> String {
    format!(
        "rustedvino {version} - OpenAI-compatible inference server on OpenVINO

Usage: rustedvino [OPTION]

With no option, starts the server with the first config file found (below).

Options:
  -h, --help               Print this help and exit
  -V, --version            Print the version and exit
      --list-devices       Print the OpenVINO device names, one per line, and exit
      --device-info        Print each device's kind, tier and memory size, and exit
      --print-default-config
                           Print every global setting with its default (JSON) and exit
      --print-default-model
                           Print one models.<id> stanza with its defaults (JSON) and exit
      --supervise          Run the server as a child process and restart it after a
                           crash (Linux only)

Config file: there is no --config option. The server uses the first of:
  1. the file named by the RUSTEDVINO_CONFIG environment variable
     (an error if that file does not exist)
  2. /opt/rustedvino/config.json
  3. %LOCALAPPDATA%\\RustedVINO\\config.json (Windows only)
  4. config.json in the directory of the rustedvino executable
Example: RUSTEDVINO_CONFIG=$HOME/rustedvino/config.json rustedvino
(`rv start` sets RUSTEDVINO_CONFIG for the server it starts.)

Exit codes include: 0 success, 1 error, 64 unknown argument, 78 invalid config
(75 and 76 are restart signals for --supervise).
",
        version = env!("CARGO_PKG_VERSION"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn no_arguments_and_known_flags_proceed() {
        assert_eq!(classify_args(&[]), CliAction::Proceed);
        for a in KNOWN_ARGS
            .iter()
            .filter(|a| !matches!(**a, "--help" | "-h"))
        {
            assert_eq!(classify_args(&os(&[a])), CliAction::Proceed, "{a}");
        }
        assert_eq!(
            classify_args(&os(&["--supervise", "--list-devices"])),
            CliAction::Proceed
        );
    }

    #[test]
    fn help_and_h_ask_for_usage() {
        assert_eq!(classify_args(&os(&["--help"])), CliAction::Help);
        assert_eq!(classify_args(&os(&["-h"])), CliAction::Help);
        assert_eq!(classify_args(&os(&["--supervise", "-h"])), CliAction::Help);
    }

    /// FM-1: anything else is refused, `--config` included (there is no such flag; the config
    /// path comes from `RUSTEDVINO_CONFIG`), and wins over a `--help` beside it.
    #[test]
    fn unknown_arguments_are_refused() {
        for bad in [
            "--config",
            "config.json",
            "/?",
            "--list-device",
            "-v",
            "--HELP",
            "",
            "--help=1",
        ] {
            assert_eq!(
                classify_args(&os(&[bad])),
                CliAction::Unknown(bad.to_owned()),
                "{bad:?}"
            );
        }
        assert_eq!(
            classify_args(&os(&["--config", "x.json"])),
            CliAction::Unknown("--config".to_owned())
        );
        assert_eq!(
            classify_args(&os(&["--help", "--bogus"])),
            CliAction::Unknown("--bogus".to_owned())
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_non_utf8_argument_is_unknown_not_a_panic() {
        use std::os::unix::ffi::OsStringExt;
        let bad = OsString::from_vec(vec![b'-', b'-', 0xff]);
        assert!(matches!(classify_args(&[bad]), CliAction::Unknown(_)));
    }

    #[test]
    fn usage_names_every_public_flag_and_the_config_search() {
        let u = usage();
        for a in KNOWN_ARGS.iter().filter(|a| **a != "--gpu-probe") {
            assert!(u.contains(a), "{a}");
        }
        for want in [
            "RUSTEDVINO_CONFIG",
            "/opt/rustedvino/config.json",
            "LOCALAPPDATA",
            "64",
            "78",
        ] {
            assert!(u.contains(want), "{want}");
        }
    }
}
