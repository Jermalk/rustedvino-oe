//! Colour and glyphs. Port of the `green`/`red`/... helpers in `scripts/rv`.

use std::io::IsTerminal;

/// Whether to emit ANSI colour. Decided once, at startup.
#[derive(Debug, Clone, Copy)]
pub struct Style {
    color: bool,
}

impl Style {
    /// Colour only on a terminal, never when `NO_COLOR` is set. On Windows a
    /// legacy console prints the escapes literally, so colour additionally
    /// requires a terminal known to understand them (`WT_SESSION`, `TERM`).
    #[must_use]
    pub fn detect() -> Self {
        let mut color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
        if cfg!(windows) {
            color = color
                && (std::env::var_os("WT_SESSION").is_some() || std::env::var_os("TERM").is_some());
        }
        Self { color }
    }

    /// No colour; used by tests and `--json`.
    #[must_use]
    pub const fn plain() -> Self {
        Self { color: false }
    }

    /// Forced colour, for tests.
    #[must_use]
    pub const fn colored() -> Self {
        Self { color: true }
    }

    fn paint(self, code: &str, text: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    }

    /// Green text.
    #[must_use]
    pub fn green(self, t: &str) -> String {
        self.paint("32", t)
    }
    /// Red text.
    #[must_use]
    pub fn red(self, t: &str) -> String {
        self.paint("31", t)
    }
    /// Yellow text.
    #[must_use]
    pub fn yellow(self, t: &str) -> String {
        self.paint("33", t)
    }
    /// Dim text.
    #[must_use]
    pub fn dim(self, t: &str) -> String {
        self.paint("2", t)
    }
    /// Bold text.
    #[must_use]
    pub fn bold(self, t: &str) -> String {
        self.paint("1", t)
    }
    /// Green status dot.
    #[must_use]
    pub fn ok(self) -> String {
        self.green("●")
    }
    /// Red status dot.
    #[must_use]
    pub fn bad(self) -> String {
        self.red("●")
    }
    /// Yellow status dot.
    #[must_use]
    pub fn warn(self) -> String {
        self.yellow("●")
    }
}
