//! Text normalization for speech synthesis: expands digit sequences and
//! substitutes symbols a TTS phonemizer drops or mispronounces (`&`, `%`,
//! `$`, `@`, `+`, `=`) into speakable words, before the text reaches the
//! engine. See the project's internal engineering log for the
//! problem this solves and the scope decisions below.
//!
//! English gets full cardinal-number expansion, thousands-comma grouping
//! (`"1,000"` → `"one thousand"`, not digit-by-digit), and short year-style
//! phrasing for a bare (non-comma, non-decimal) 4-digit number in a
//! plausible calendar range (`"1990"` → `"nineteen ninety"` rather than the
//! 6-word full cardinal `"one thousand nine hundred ninety"`) — not just a
//! nicety: a response with several dates read in full-cardinal form measurably
//! lengthens TTS synthesis time, since synthesis time scales with spoken
//! length. A comma-grouped number is exempt from year-style on purpose —
//! nobody writes the year 1990 as `"1,990"`, so a thousands-comma is a
//! reliable signal that the number is a quantity, not a date. Polish gets
//! best-effort cardinal expansion only (comma-grouping applies there too,
//! but no year-style shortening) — no case-agreement/declension (a much
//! bigger, open-ended linguistics problem, explicitly out of scope). Ranges
//! (`"2-3"`), full calendar dates (`"2026-08-25"`), and ordinals (`"1st"`)
//! are also out of scope: their digit runs are still expanded individually
//! (deterministic, tested), but the surrounding punctuation is not
//! specially interpreted.

/// Cardinal expansion above this many digits reads digit-by-digit instead
/// (a long ID/port/serial number becoming a run-on cardinal is worse than
/// reading it digit-by-digit). This module only implements
/// ones/teens/tens/hundreds/thousands scale words, so the cap is also a
/// natural ceiling for what `cardinal()` below can express.
const MAX_CARDINAL_DIGITS: usize = 6;

/// Normalize `text` for speech synthesis: symbol substitution and digit
/// expansion, language-aware. `lang` is the client-supplied language code
/// (e.g. `"en"`, `"pl-PL"`); only the first two lowercased characters are
/// matched, defaulting to English for `None` or anything unrecognized.
#[must_use]
pub fn normalize_for_speech(text: &str, lang: Option<&str>) -> String {
    let lang = Lang::from_code(lang);
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];

        // Currency prefix: '$' directly before a digit run — consume the
        // '$', expand the number, then append the currency word (not a
        // full currency-to-words formatter; see module doc on scope).
        // Year-style is never appropriate for a dollar amount, so it's
        // disabled here regardless of digit count.
        if c == '$' && chars.get(i + 1).is_some_and(char::is_ascii_digit) {
            let (words, next_i) = expand_number(&chars, i + 1, lang, false);
            out.push_str(&words);
            out.push(' ');
            out.push_str(lang.currency_word());
            i = next_i;
            continue;
        }

        // Negative sign: '-' directly before a digit, only when NOT itself
        // preceded by a digit — so "2-3" (a range, out of scope) is left
        // alone rather than misread as "two negative three". A negative
        // year isn't a real thing, so year-style is disabled here too.
        if c == '-'
            && chars.get(i + 1).is_some_and(char::is_ascii_digit)
            && (i == 0 || !chars[i - 1].is_ascii_digit())
        {
            let (words, next_i) = expand_number(&chars, i + 1, lang, false);
            out.push_str(lang.negative_word());
            out.push(' ');
            out.push_str(&words);
            i = next_i;
            continue;
        }

        if c.is_ascii_digit() {
            let (words, next_i) = expand_number(&chars, i, lang, true);
            out.push_str(&words);
            i = next_i;
            continue;
        }

        if let Some(word) = lang.symbol_word(c) {
            // Padded on both sides — a symbol adjacent to letters with no
            // whitespace ("R&D") would otherwise glue into one unreadable
            // word ("RandD"). The trailing whitespace-collapse below turns
            // any resulting double space back into one.
            out.push(' ');
            out.push_str(word);
            out.push(' ');
            i += 1;
            continue;
        }

        out.push(c);
        i += 1;
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Expand the digit run starting at `chars[start]` (already confirmed to be
/// an ASCII digit) into spoken words, folding in thousands-comma groups and
/// an optional decimal fraction (a `.` directly between two digit runs —
/// never a bare trailing `.`, which is left for the caller as ordinary
/// sentence punctuation). `allow_year_style` gates the short year-pairing
/// (English only, see module doc); pass `false` from any context where the
/// number can't be a bare calendar year (currency, a negative number).
/// Returns the expanded words and the index just past everything consumed.
fn expand_number(
    chars: &[char],
    start: usize,
    lang: Lang,
    allow_year_style: bool,
) -> (String, usize) {
    let mut j = start;
    while chars.get(j).is_some_and(char::is_ascii_digit) {
        j += 1;
    }
    let mut digits: String = chars[start..j].iter().collect();

    // Thousands-grouping commas ("1,000", "12,345,678"): a comma followed
    // by EXACTLY three digits (not more, not fewer) is almost always a
    // grouping separator, not a list comma ("1, 2, 3") — fold the group's
    // digits straight into the run and keep scanning for more groups.
    let mut had_comma_group = false;
    let mut end = j;
    while chars.get(end) == Some(&',')
        && chars.get(end + 1).is_some_and(char::is_ascii_digit)
        && chars.get(end + 2).is_some_and(char::is_ascii_digit)
        && chars.get(end + 3).is_some_and(char::is_ascii_digit)
        && !chars.get(end + 4).is_some_and(char::is_ascii_digit)
    {
        digits.push(chars[end + 1]);
        digits.push(chars[end + 2]);
        digits.push(chars[end + 3]);
        had_comma_group = true;
        end += 4;
    }
    let int_run = digits;

    let mut frac_run: Option<String> = None;
    if chars.get(end) == Some(&'.') && chars.get(end + 1).is_some_and(char::is_ascii_digit) {
        let fs = end + 1;
        let mut k = fs;
        while chars.get(k).is_some_and(char::is_ascii_digit) {
            k += 1;
        }
        frac_run = Some(chars[fs..k].iter().collect());
        end = k;
    }

    // A comma-grouped number is never year-style (see module doc); neither
    // is one with a decimal fraction or in a non-English session.
    let year_words =
        (allow_year_style && !had_comma_group && frac_run.is_none() && lang == Lang::En)
            .then(|| int_run.parse::<u32>().ok())
            .flatten()
            .and_then(year_en);

    let int_words = if let Some(y) = year_words {
        y
    } else if int_run.len() > MAX_CARDINAL_DIGITS || (int_run.len() > 1 && int_run.starts_with('0'))
    {
        // A leading zero (e.g. "007") or a run longer than this module's
        // cardinal scale (thousands) reads digit-by-digit, same as a phone
        // number or serial code would.
        digit_by_digit(&int_run, lang)
    } else {
        // Safe: int_run is 1..=MAX_CARDINAL_DIGITS ASCII digits (fits u32
        // many times over); `unwrap_or(0)` is unreachable defensive code,
        // not a real fallback path.
        lang.cardinal(int_run.parse().unwrap_or(0))
    };

    let mut words = int_words;
    if let Some(frac) = frac_run {
        words.push(' ');
        words.push_str(lang.decimal_point_word());
        words.push(' ');
        words.push_str(&digit_by_digit(&frac, lang));
    }
    (words, end)
}

/// Short year-style reading for a bare 4-digit number in a plausible
/// calendar range (`1100..=2099` — wide enough for the overwhelming
/// majority of real dates in conversation; outside it, the full cardinal
/// reading is used instead, which stays correct either way). `1990` →
/// `"nineteen ninety"`; round decades/centuries get "hundred"
/// (`1900` → `"nineteen hundred"`, `2000` → `"twenty hundred"` — a
/// deliberate, simple, consistent rule rather than special-casing exact
/// thousands boundaries, since English usage itself is inconsistent there).
/// English only — see module doc for why Polish doesn't get this.
fn year_en(n: u32) -> Option<String> {
    if !(1100..=2099).contains(&n) {
        return None;
    }
    let ab = n / 100;
    let cd = n % 100;
    Some(if cd == 0 {
        format!("{} hundred", two_digit_group_en(ab))
    } else {
        format!("{} {}", two_digit_group_en(ab), two_digit_group_en(cd))
    })
}

/// English cardinal expansion of a single `0..100` group — shared by
/// [`three_digit_en`]'s remainder and [`year_en`]'s two-digit halves.
fn two_digit_group_en(n: u32) -> String {
    if n < 20 {
        EN_ONES[n as usize].to_owned()
    } else {
        let tens = (n / 10) as usize;
        let ones = n % 10;
        if ones == 0 {
            EN_TENS[tens].to_owned()
        } else {
            format!("{} {}", EN_TENS[tens], EN_ONES[ones as usize])
        }
    }
}

/// Read a digit string one digit at a time (`"007"` → `"zero zero
/// seven"`), used for oversized/leading-zero integer runs and for every
/// fractional part.
fn digit_by_digit(run: &str, lang: Lang) -> String {
    run.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| lang.digit_word(d))
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Lang {
    En,
    Pl,
}

impl Lang {
    fn from_code(code: Option<&str>) -> Self {
        let lowered = code.map(str::to_ascii_lowercase);
        match lowered {
            Some(c) if c.starts_with("pl") => Self::Pl,
            _ => Self::En,
        }
    }

    fn digit_word(self, d: u32) -> &'static str {
        let words = match self {
            Self::En => EN_ONES,
            Self::Pl => PL_ONES,
        };
        words.get(d as usize).copied().unwrap_or("?")
    }

    /// Cardinal expansion of `n` (`0..=999_999`, enforced by
    /// `MAX_CARDINAL_DIGITS` at the call site).
    fn cardinal(self, n: u32) -> String {
        match self {
            Self::En => cardinal_en(n),
            Self::Pl => cardinal_pl(n),
        }
    }

    fn currency_word(self) -> &'static str {
        match self {
            Self::En => "dollars",
            Self::Pl => "dolarów",
        }
    }

    fn negative_word(self) -> &'static str {
        match self {
            Self::En => "negative",
            Self::Pl => "minus",
        }
    }

    fn decimal_point_word(self) -> &'static str {
        match self {
            Self::En => "point",
            Self::Pl => "przecinek",
        }
    }

    fn symbol_word(self, c: char) -> Option<&'static str> {
        match (self, c) {
            (Self::En, '&') => Some("and"),
            (Self::En, '%') => Some("percent"),
            (Self::En, '@') => Some("at"),
            (Self::En, '=') => Some("equals"),
            (Self::Pl, '&') => Some("i"),
            (Self::Pl, '%') => Some("procent"),
            (Self::Pl, '@') => Some("małpa"),
            (Self::Pl, '=') => Some("równa się"),
            // Same word in both languages.
            (Self::En | Self::Pl, '+') => Some("plus"),
            _ => None,
        }
    }
}

const EN_ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const EN_TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

/// English cardinal expansion, `0..=999_999`.
fn cardinal_en(n: u32) -> String {
    if n == 0 {
        return "zero".to_owned();
    }
    let thousands = n / 1000;
    let rem = n % 1000;
    let mut parts = Vec::new();
    if thousands > 0 {
        parts.push(format!("{} thousand", three_digit_en(thousands)));
    }
    if rem > 0 {
        parts.push(three_digit_en(rem));
    }
    parts.join(" ")
}

/// English cardinal expansion of a 0..1000 group (no "thousand" word).
fn three_digit_en(n: u32) -> String {
    let hundreds = n / 100;
    let rem = n % 100;
    let mut parts = Vec::new();
    if hundreds > 0 {
        parts.push(format!("{} hundred", EN_ONES[hundreds as usize]));
    }
    if rem > 0 {
        parts.push(two_digit_group_en(rem));
    }
    parts.join(" ")
}

const PL_ONES: [&str; 20] = [
    "zero",
    "jeden",
    "dwa",
    "trzy",
    "cztery",
    "pięć",
    "sześć",
    "siedem",
    "osiem",
    "dziewięć",
    "dziesięć",
    "jedenaście",
    "dwanaście",
    "trzynaście",
    "czternaście",
    "piętnaście",
    "szesnaście",
    "siedemnaście",
    "osiemnaście",
    "dziewiętnaście",
];
const PL_TENS: [&str; 10] = [
    "",
    "",
    "dwadzieścia",
    "trzydzieści",
    "czterdzieści",
    "pięćdziesiąt",
    "sześćdziesiąt",
    "siedemdziesiąt",
    "osiemdziesiąt",
    "dziewięćdziesiąt",
];
const PL_HUNDREDS: [&str; 10] = [
    "",
    "sto",
    "dwieście",
    "trzysta",
    "czterysta",
    "pięćset",
    "sześćset",
    "siedemset",
    "osiemset",
    "dziewięćset",
];

/// Polish best-effort cardinal expansion, `0..=999_999` — masculine/default
/// form only, no case-agreement or declension, and the thousands scale word
/// is always the singular "tysiąc" regardless of what correct Polish
/// grammar would require ("dwa tysiące", "pięć tysięcy", ...). This is a
/// deliberate scope limit (see module doc / the plan this implements), not
/// a bug: it is a real improvement over reading raw digits, not a claim of
/// grammatical correctness.
fn cardinal_pl(n: u32) -> String {
    if n == 0 {
        return "zero".to_owned();
    }
    let thousands = n / 1000;
    let rem = n % 1000;
    let mut parts = Vec::new();
    if thousands > 0 {
        if thousands == 1 {
            parts.push("tysiąc".to_owned());
        } else {
            parts.push(format!("{} tysiąc", three_digit_pl(thousands)));
        }
    }
    if rem > 0 {
        parts.push(three_digit_pl(rem));
    }
    parts.join(" ")
}

/// Polish cardinal expansion of a 0..1000 group (no "tysiąc" word).
fn three_digit_pl(n: u32) -> String {
    let hundreds = n / 100;
    let rem = n % 100;
    let mut parts = Vec::new();
    if hundreds > 0 {
        parts.push(PL_HUNDREDS[hundreds as usize].to_owned());
    }
    if rem > 0 {
        if rem < 20 {
            parts.push(PL_ONES[rem as usize].to_owned());
        } else {
            let tens = (rem / 10) as usize;
            let ones = rem % 10;
            if ones == 0 {
                parts.push(PL_TENS[tens].to_owned());
            } else {
                parts.push(format!("{} {}", PL_TENS[tens], PL_ONES[ones as usize]));
            }
        }
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    // ── English cardinals ──────────────────────────────────────────────────

    #[test]
    fn en_plain_cardinals() {
        assert_eq!(normalize_for_speech("42", Some("en")), "forty two");
        assert_eq!(normalize_for_speech("7", Some("en")), "seven");
        assert_eq!(normalize_for_speech("100", Some("en")), "one hundred");
        assert_eq!(normalize_for_speech("101", Some("en")), "one hundred one");
        assert_eq!(
            normalize_for_speech("123456", Some("en")),
            "one hundred twenty three thousand four hundred fifty six"
        );
        assert_eq!(normalize_for_speech("0", Some("en")), "zero");
    }

    #[test]
    fn en_decimal() {
        assert_eq!(
            normalize_for_speech("pi is 3.14 roughly", Some("en")),
            "pi is three point one four roughly"
        );
    }

    #[test]
    fn en_period_not_decimal_when_no_surrounding_digits() {
        // A bare sentence-ending period, not adjacent to digits on both
        // sides, must not be touched by number expansion at all.
        assert_eq!(normalize_for_speech("The end.", Some("en")), "The end.");
    }

    #[test]
    fn en_negative_number() {
        assert_eq!(
            normalize_for_speech("it was -5 degrees", Some("en")),
            "it was negative five degrees"
        );
    }

    #[test]
    fn en_leading_zero_reads_digit_by_digit() {
        assert_eq!(normalize_for_speech("007", Some("en")), "zero zero seven");
    }

    #[test]
    fn en_oversized_run_reads_digit_by_digit() {
        // 7 digits > MAX_CARDINAL_DIGITS.
        assert_eq!(
            normalize_for_speech("1234567", Some("en")),
            "one two three four five six seven"
        );
    }

    #[test]
    fn en_symbols() {
        assert_eq!(
            normalize_for_speech("cats & dogs", Some("en")),
            "cats and dogs"
        );
        assert_eq!(normalize_for_speech("50%", Some("en")), "fifty percent");
        assert_eq!(normalize_for_speech("$5", Some("en")), "five dollars");
        // No surrounding whitespace in the source ("R&D"-style) — the
        // padded substitution plus final whitespace collapse still keeps
        // every word separate and readable.
        assert_eq!(normalize_for_speech("me@home", Some("en")), "me at home");
        assert_eq!(normalize_for_speech("2+2", Some("en")), "two plus two");
        assert_eq!(normalize_for_speech("a=b", Some("en")), "a equals b");
    }

    #[test]
    fn en_currency_with_decimal() {
        assert_eq!(
            normalize_for_speech("$12.50", Some("en")),
            "twelve point five zero dollars"
        );
    }

    /// Ranges are explicitly out of scope (see module doc): each digit run
    /// is expanded independently and the hyphen is left as a literal
    /// character. Pinned here so this is a documented limitation, not a
    /// silently-discovered bug.
    #[test]
    fn en_range_hyphen_pinned_as_known_limitation() {
        assert_eq!(normalize_for_speech("2-3", Some("en")), "two-three");
    }

    /// Full calendar dates are explicitly out of scope: each digit run is
    /// still expanded independently with no date-aware phrasing (the "08"
    /// month, leading-zero, still reads digit-by-digit), but the "2026"
    /// year-group does get ordinary year-style treatment since it's just a
    /// bare 4-digit run in range as far as `expand_number` can tell.
    #[test]
    fn en_date_pinned_as_known_limitation() {
        assert_eq!(
            normalize_for_speech("2026-08-25", Some("en")),
            "twenty twenty six-zero eight-twenty five"
        );
    }

    // ── Comma-grouped thousands and year-style (follow-up) ───────────

    #[test]
    fn en_comma_grouped_thousands() {
        assert_eq!(normalize_for_speech("1,000", Some("en")), "one thousand");
        assert_eq!(
            normalize_for_speech("12,345", Some("en")),
            "twelve thousand three hundred forty five"
        );
        assert_eq!(
            normalize_for_speech("dating back over 1,000 years", Some("en")),
            "dating back over one thousand years"
        );
    }

    /// Beyond this module's thousands ceiling even after folding commas —
    /// stays a documented scope boundary (see `MAX_CARDINAL_DIGITS`), not a
    /// silent misread: a million-scale comma-grouped number reads
    /// digit-by-digit rather than mis-grouping or panicking.
    #[test]
    fn en_comma_grouped_beyond_cardinal_scale_reads_digit_by_digit() {
        assert_eq!(
            normalize_for_speech("1,000,000", Some("en")),
            "one zero zero zero zero zero zero"
        );
    }

    #[test]
    fn en_year_style_short_form() {
        assert_eq!(
            normalize_for_speech("in 1990", Some("en")),
            "in nineteen ninety"
        );
        assert_eq!(
            normalize_for_speech("in 2026", Some("en")),
            "in twenty twenty six"
        );
        assert_eq!(
            normalize_for_speech("in 1900", Some("en")),
            "in nineteen hundred"
        );
        assert_eq!(
            normalize_for_speech("in 2000", Some("en")),
            "in twenty hundred"
        );
    }

    /// Outside the plausible calendar range (`1100..=2099`), a bare 4-digit
    /// number still gets the full cardinal reading — e.g. "1,000 years" is
    /// a quantity, and even without the comma a value like 1050 is treated
    /// as a plain number, not forced into year-pairing.
    #[test]
    fn en_four_digit_outside_year_range_uses_full_cardinal() {
        assert_eq!(
            normalize_for_speech("1050", Some("en")),
            "one thousand fifty"
        );
    }

    /// A comma-grouped 4-digit number is never year-style, even though it
    /// happens to fall in the calendar range — the comma is a deliberate
    /// signal that this is a quantity (see module doc).
    #[test]
    fn en_comma_grouped_year_range_number_is_not_year_style() {
        assert_eq!(
            normalize_for_speech("1,990", Some("en")),
            "one thousand nine hundred ninety"
        );
    }

    /// A dollar amount is never year-style even if the digits fall in the
    /// calendar range.
    #[test]
    fn en_currency_in_year_range_is_not_year_style() {
        assert_eq!(
            normalize_for_speech("$1990", Some("en")),
            "one thousand nine hundred ninety dollars"
        );
    }

    #[test]
    fn default_language_is_english() {
        assert_eq!(normalize_for_speech("42", None), "forty two");
        assert_eq!(normalize_for_speech("42", Some("xx")), "forty two");
        assert_eq!(normalize_for_speech("42", Some("en-US")), "forty two");
    }

    // ── Polish (best-effort, not grammatically complete by design) ─────────

    #[test]
    fn pl_best_effort_cardinals() {
        assert_eq!(normalize_for_speech("5", Some("pl")), "pięć");
        assert_eq!(normalize_for_speech("21", Some("pl")), "dwadzieścia jeden");
        assert_eq!(normalize_for_speech("100", Some("pl")), "sto");
        // Known limitation: correct Polish is "dwa tysiące", not "dwa
        // tysiąc" — declension of the scale word is out of scope.
        assert_eq!(normalize_for_speech("2000", Some("pl")), "dwa tysiąc");
    }

    /// Year-style short-form is English-only (see module doc) — a bare
    /// 4-digit Polish year still gets the full best-effort cardinal, not
    /// pairing.
    #[test]
    fn pl_does_not_get_year_style() {
        assert_eq!(
            normalize_for_speech("1990", Some("pl")),
            "tysiąc dziewięćset dziewięćdziesiąt"
        );
    }

    #[test]
    fn pl_symbols() {
        assert_eq!(normalize_for_speech("koty & psy", Some("pl")), "koty i psy");
        assert_eq!(
            normalize_for_speech("50%", Some("pl")),
            "pięćdziesiąt procent"
        );
    }

    #[test]
    fn pl_language_code_variant_matches() {
        assert_eq!(normalize_for_speech("5", Some("pl-PL")), "pięć");
    }
}
