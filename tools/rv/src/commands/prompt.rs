//! Confirmation prompts. Port of `_confirm` in `scripts/rv`.

use std::io::{BufRead, IsTerminal, Write};

use crate::ansi::Style;

/// Whether an interactive answer means yes (`y`/`yes`, any case).
#[must_use]
pub fn answer_is_yes(answer: &str) -> bool {
    matches!(answer.trim().to_lowercase().as_str(), "y" | "yes")
}

/// Asks `prompt [y/N]` on the terminal. `assume_yes` (`-y/--yes`) skips it.
/// On a non-terminal without `-y` it refuses with a clear message — the Python
/// script died with an `EOFError` here, which read like a crash.
#[must_use]
pub fn confirm(style: Style, prompt: &str, assume_yes: bool) -> bool {
    if assume_yes {
        return true;
    }
    if !std::io::stdin().is_terminal() {
        eprintln!(
            "{}",
            style.red(&format!(
                "refusing without --yes (not an interactive terminal): {prompt}"
            ))
        );
        return false;
    }
    print!("{prompt} [y/N]: ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    if std::io::stdin().lock().read_line(&mut line).is_err() {
        return false;
    }
    answer_is_yes(&line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_y_and_yes_confirm() {
        for yes in ["y", "Y", "yes", "YES", " yes \n"] {
            assert!(answer_is_yes(yes), "{yes:?}");
        }
        for no in ["", "n", "no", "yep", "ye", "\n"] {
            assert!(!answer_is_yes(no), "{no:?}");
        }
    }

    #[test]
    fn assume_yes_skips_the_prompt() {
        assert!(confirm(Style::plain(), "anything", true));
    }
}
