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
//! cardinals whose thousands scale word agrees with the count ("dwa tysiące",
//! "pięć tysięcy"), `zł` / `kg` / `km` agreeing with the number, and a bare
//! 4-digit year (1000..=2999) followed by "roku" / "r." read as an ordinal in
//! the right case ("w 1948 roku" locative, otherwise genitive). Everything
//! else about Polish stays best-effort: no declension of quantities by case or
//! gender ("dwa", never "dwie"), no year phrasing without "roku" (a bare
//! "w 2000 przypadkach" is a quantity). Ranges
//! (`"2-3"`), full calendar dates (`"2026-08-25"`), and ordinals (`"1st"`)
//! are also out of scope: their digit runs are still expanded individually
//! (deterministic, tested), but the surrounding punctuation is not
//! specially interpreted.
//!
//! Vietnamese reads numbers with Piper's own `vi_number`, so a `vi` session
//! hears Vietnamese words for digits, decimals, `%`, `&` and the like rather
//! than English ones. Vietnamese writes `.` for thousands and `,` for decimals;
//! this module keeps the English convention for both (out of scope, as for
//! Polish), so `"1.000"` reads as a decimal.

use crate::pipelines::piper::{VI_DIGITS, vi_number};

/// Cardinal expansion above this many digits reads digit-by-digit instead
/// (a long ID/port/serial number becoming a run-on cardinal is worse than
/// reading it digit-by-digit). This module only implements
/// ones/teens/tens/hundreds/thousands scale words, so the cap is also a
/// natural ceiling for what `cardinal()` below can express.
const MAX_CARDINAL_DIGITS: usize = 6;

/// True when `code` (e.g. `"pl"`, `"vi-VN"`) is a language this module has
/// number words for. Callers that must not guess (an explicit request for
/// some other language) use it to skip normalisation instead of falling back
/// to English words.
#[must_use]
pub fn supports_language(code: &str) -> bool {
    let lowered = code.to_ascii_lowercase();
    ["en", "pl", "vi"].iter().any(|l| lowered.starts_with(l))
}

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
            if lang == Lang::Pl {
                // A year before "roku" gets ordinal words in the right case.
                let mut run_end = i;
                while chars.get(run_end).is_some_and(char::is_ascii_digit) {
                    run_end += 1;
                }
                let fraction_follows = matches!(chars.get(run_end), Some('.' | ','))
                    && chars.get(run_end + 1).is_some_and(char::is_ascii_digit);
                if !fraction_follows && let Some((words, resume)) = pl_year_at(&chars, i, run_end) {
                    out.push_str(&words);
                    i = resume;
                    continue;
                }
            }
            let (words, next_i) = expand_number(&chars, i, lang, true);
            out.push_str(&words);
            // Polish counted units ("15 zł", "3kg") agree with the number.
            if lang == Lang::Pl {
                let run: String = chars[i..next_i].iter().collect();
                if let Some((unit, after)) = pl_unit_after(&chars, next_i, &run) {
                    out.push(' ');
                    out.push_str(unit);
                    i = after;
                    continue;
                }
            }
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
    let year_words = (allow_year_style
        && !had_comma_group
        && frac_run.is_none()
        && lang == Lang::En
        && !en_quantity_context(chars, start, j))
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

/// Words after which "1250" is money or a count, never a year.
const EN_CURRENCY_WORDS: [&str; 14] = [
    "dollar", "dollars", "euro", "euros", "pound", "pounds", "cent", "cents", "buck", "bucks",
    "usd", "eur", "gbp", "pln",
];

/// Words before a number that make it a year even when a plural noun follows
/// ("in 1990 tensions rose").
const EN_YEAR_PREPOSITIONS: [&str; 9] = [
    "in", "since", "until", "till", "by", "during", "circa", "before", "after",
];

/// Words ending in "s" that are not plural nouns ("1990 was a good year").
const EN_NON_PLURAL_S_WORDS: [&str; 12] = [
    "was", "is", "has", "as", "this", "his", "its", "us", "does", "yes", "thus", "always",
];

/// True when the 4-digit run `chars[start..end]` reads as a quantity rather
/// than a year: a currency word follows ("1250 dollars"), or a plural noun
/// follows ("1250 apples") and the word before is not a year preposition
/// ("in 1990 tensions rose" stays a year). Year-style ("twelve fifty") is a
/// reading for years only; for a quantity it is plainly wrong.
fn en_quantity_context(chars: &[char], start: usize, end: usize) -> bool {
    let (next, _) = next_word(chars, end);
    if EN_CURRENCY_WORDS.contains(&next.as_str()) {
        return true;
    }
    let plural = (next.len() > 3
        && next.ends_with('s')
        && !next.ends_with("ss")
        && !EN_NON_PLURAL_S_WORDS.contains(&next.as_str()))
        || matches!(next.as_str(), "people" | "children" | "men" | "women");
    plural && !EN_YEAR_PREPOSITIONS.contains(&prev_word(chars, start).as_str())
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
    Vi,
}

impl Lang {
    fn from_code(code: Option<&str>) -> Self {
        let lowered = code.map(str::to_ascii_lowercase);
        match lowered {
            Some(c) if c.starts_with("pl") => Self::Pl,
            Some(c) if c.starts_with("vi") => Self::Vi,
            _ => Self::En,
        }
    }

    fn digit_word(self, d: u32) -> &'static str {
        let words: &[&'static str] = match self {
            Self::En => &EN_ONES,
            Self::Pl => &PL_ONES,
            Self::Vi => &VI_DIGITS,
        };
        words.get(d as usize).copied().unwrap_or("?")
    }

    /// Cardinal expansion of `n` (`0..=999_999`, enforced by
    /// `MAX_CARDINAL_DIGITS` at the call site).
    fn cardinal(self, n: u32) -> String {
        match self {
            Self::En => cardinal_en(n),
            Self::Pl => cardinal_pl(n),
            Self::Vi => vi_number(u64::from(n)),
        }
    }

    fn currency_word(self) -> &'static str {
        match self {
            Self::En => "dollars",
            Self::Pl => "dolarów",
            Self::Vi => "đô la",
        }
    }

    fn negative_word(self) -> &'static str {
        match self {
            Self::En => "negative",
            Self::Pl => "minus",
            Self::Vi => "âm",
        }
    }

    fn decimal_point_word(self) -> &'static str {
        match self {
            Self::En => "point",
            Self::Pl => "przecinek",
            Self::Vi => "phẩy",
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
            (Self::Vi, '&') => Some("và"),
            (Self::Vi, '%') => Some("phần trăm"),
            (Self::Vi, '@') => Some("a còng"),
            (Self::Vi, '=') => Some("bằng"),
            (Self::Vi, '+') => Some("cộng"),
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

/// Polish cardinal expansion, `0..=999_999` — nominative, masculine/default
/// form only (no declension by case or gender: "dwa", never "dwie"), but the
/// thousands scale word agrees with its count ("tysiąc", "dwa tysiące",
/// "pięć tysięcy", "dwadzieścia dwa tysiące").
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
            parts.push(format!(
                "{} {}",
                three_digit_pl(thousands),
                pl_thousand_word(thousands)
            ));
        }
    }
    if rem > 0 {
        parts.push(three_digit_pl(rem));
    }
    parts.join(" ")
}

/// The scale word for `count` thousands (`count >= 2`): 2-4 (but not 12-14)
/// take "tysiące", everything else "tysięcy". A count of exactly 1 is
/// "tysiąc" and is handled by the caller.
fn pl_thousand_word(count: u32) -> &'static str {
    if matches!(count % 10, 2..=4) && !matches!(count % 100, 12..=14) {
        "tysiące"
    } else {
        "tysięcy"
    }
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

// ── Polish units and years ─────────────────────────────────────────────────

/// Abbreviation → [singular (1), plural (2-4), genitive plural (5+), genitive
/// singular (after a decimal fraction)].
const PL_UNITS: [(&str, [&str; 4]); 3] = [
    ("zł", ["złoty", "złote", "złotych", "złotego"]),
    ("kg", ["kilogram", "kilogramy", "kilogramów", "kilograma"]),
    ("km", ["kilometr", "kilometry", "kilometrów", "kilometra"]),
];

/// The form of a counted noun for the integer `n`: 0 = singular (1),
/// 1 = nominative plural (2-4 but not 12-14), 2 = genitive plural.
fn pl_count_form(n: u64) -> usize {
    if n == 1 {
        0
    } else if matches!(n % 10, 2..=4) && !matches!(n % 100, 12..=14) {
        1
    } else {
        2
    }
}

/// The word at or after `from` (skipping whitespace): its lowercase letters
/// and the index just past them.
fn next_word(chars: &[char], from: usize) -> (String, usize) {
    let mut k = from;
    while chars.get(k).is_some_and(|c| c.is_whitespace()) {
        k += 1;
    }
    let start = k;
    while chars.get(k).is_some_and(|c| c.is_alphabetic()) {
        k += 1;
    }
    (chars[start..k].iter().collect::<String>().to_lowercase(), k)
}

/// The word that ends just before `before` (skipping whitespace), lowercase.
fn prev_word(chars: &[char], before: usize) -> String {
    let mut k = before;
    while k > 0 && chars[k - 1].is_whitespace() {
        k -= 1;
    }
    let end = k;
    while k > 0 && chars[k - 1].is_alphabetic() {
        k -= 1;
    }
    chars[k..end].iter().collect::<String>().to_lowercase()
}

/// A unit abbreviation right after a Polish number (`15 zł`, `3kg`): the
/// unit word in the form the number requires and the index past the
/// abbreviation. `run` is the number's source text. Only a whole-word
/// abbreviation matches (`złoty` is not `zł`).
fn pl_unit_after(chars: &[char], end: usize, run: &str) -> Option<(&'static str, usize)> {
    let (word, after) = next_word(chars, end);
    let (_, forms) = PL_UNITS.iter().find(|(abbr, _)| *abbr == word)?;
    let form = if run.contains('.') {
        3
    } else {
        let digits: String = run.chars().filter(char::is_ascii_digit).collect();
        pl_count_form(digits.parse().unwrap_or(u64::MAX))
    };
    Some((forms[form], after))
}

/// Grammatical case of a Polish year: "w 1948 roku" is locative, a year
/// anywhere else before "roku" (od/do/z/przed ... roku) is genitive.
#[derive(Clone, Copy, PartialEq, Eq)]
enum YearCase {
    Locative,
    Genitive,
}

const ORD_ONES_PL: [&str; 20] = [
    "",
    "pierwszy",
    "drugi",
    "trzeci",
    "czwarty",
    "piąty",
    "szósty",
    "siódmy",
    "ósmy",
    "dziewiąty",
    "dziesiąty",
    "jedenasty",
    "dwunasty",
    "trzynasty",
    "czternasty",
    "piętnasty",
    "szesnasty",
    "siedemnasty",
    "osiemnasty",
    "dziewiętnasty",
];
const ORD_TENS_PL: [&str; 10] = [
    "",
    "",
    "dwudziesty",
    "trzydziesty",
    "czterdziesty",
    "pięćdziesiąty",
    "sześćdziesiąty",
    "siedemdziesiąty",
    "osiemdziesiąty",
    "dziewięćdziesiąty",
];
const ORD_HUNDREDS_PL: [&str; 10] = [
    "",
    "setny",
    "dwusetny",
    "trzysetny",
    "czterysetny",
    "pięćsetny",
    "sześćsetny",
    "siedemsetny",
    "osiemsetny",
    "dziewięćsetny",
];

/// Puts a nominative masculine ordinal ("czterdziesty", "trzeci") into the
/// case: locative "-ym"/"-im", genitive "-ego"/"-iego".
fn decline_ordinal_pl(nominative: &str, case: YearCase) -> String {
    let (stem, soft) = match nominative.strip_suffix('y') {
        Some(stem) => (stem, false),
        None => (nominative.strip_suffix('i').unwrap_or(nominative), true),
    };
    let ending = match (case, soft) {
        (YearCase::Locative, false) => "ym",
        (YearCase::Locative, true) => "im",
        (YearCase::Genitive, false) => "ego",
        (YearCase::Genitive, true) => "iego",
    };
    format!("{stem}{ending}")
}

/// A year `1000..=2999` as Polish words with the last component an ordinal
/// in `case`: 1948 → "tysiąc dziewięćset czterdziestym ósmym" (locative),
/// 2000 → "dwutysięcznym", 2024 → "dwa tysiące dwudziestym czwartym".
fn year_pl(n: u32, case: YearCase) -> Option<String> {
    if !(1000..=2999).contains(&n) {
        return None;
    }
    let thousands = n / 1000;
    let rest = n % 1000;
    let hundreds = (rest / 100) as usize;
    let tail = (rest % 100) as usize;
    if rest == 0 {
        let nominative = if thousands == 1 {
            "tysięczny"
        } else {
            "dwutysięczny"
        };
        return Some(decline_ordinal_pl(nominative, case));
    }
    let mut parts: Vec<String> = vec![if thousands == 1 {
        "tysiąc".to_owned()
    } else {
        "dwa tysiące".to_owned()
    }];
    if tail == 0 {
        parts.push(decline_ordinal_pl(ORD_HUNDREDS_PL[hundreds], case));
    } else {
        if hundreds > 0 {
            parts.push(PL_HUNDREDS[hundreds].to_owned());
        }
        if tail < 20 {
            parts.push(decline_ordinal_pl(ORD_ONES_PL[tail], case));
        } else {
            let (tens, ones) = (tail / 10, tail % 10);
            parts.push(decline_ordinal_pl(ORD_TENS_PL[tens], case));
            if ones > 0 {
                parts.push(decline_ordinal_pl(ORD_ONES_PL[ones], case));
            }
        }
    }
    Some(parts.join(" "))
}

/// Polish "year + roku" at the digit run `chars[start..end]`: a bare 4-digit
/// number `1000..=2999` followed by "roku" or "r." is a year. Returns the
/// year words and the index to resume at: past "r." (replaced by "roku" so
/// the abbreviation is not read letter by letter), or `end` for "roku",
/// which stays in the text. A bare "w 1948" without "roku" is NOT treated as
/// a year - "w 2000 przypadkach" is a quantity - so it keeps the plain
/// cardinal.
fn pl_year_at(chars: &[char], start: usize, end: usize) -> Option<(String, usize)> {
    if end - start != 4 || (start > 0 && matches!(chars[start - 1], '$' | '-')) {
        return None;
    }
    let value: u32 = chars[start..end].iter().collect::<String>().parse().ok()?;
    let (word, word_end) = next_word(chars, end);
    let (resume, suffix) = match word.as_str() {
        "roku" => (end, ""),
        "r" if chars.get(word_end) == Some(&'.') => {
            // Keep the sentence period when the abbreviation ends a sentence.
            let mut k = word_end + 1;
            while chars.get(k).is_some_and(|c| c.is_whitespace()) {
                k += 1;
            }
            let ends_sentence = chars.get(k).is_none_or(|c| c.is_uppercase());
            (word_end + 1, if ends_sentence { "roku." } else { "roku" })
        }
        _ => return None,
    };
    let case = if matches!(prev_word(chars, start).as_str(), "w" | "we" | "po") {
        YearCase::Locative
    } else {
        YearCase::Genitive
    };
    let mut words = year_pl(value, case)?;
    if !suffix.is_empty() {
        words.push(' ');
        words.push_str(suffix);
    }
    Some((words, resume))
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
    fn en_quantity_before_a_noun_is_not_a_year() {
        let en = |t: &str| normalize_for_speech(t, Some("en"));
        assert_eq!(
            en("We have 25 apples and 1250 dollars."),
            "We have twenty five apples and one thousand two hundred fifty dollars."
        );
        assert_eq!(
            en("It cost 1250 euros"),
            "It cost one thousand two hundred fifty euros"
        );
        assert_eq!(
            en("about 1500 people came"),
            "about one thousand five hundred people came"
        );
        assert_eq!(
            en("I paid 1250 bucks"),
            "I paid one thousand two hundred fifty bucks"
        );
    }

    #[test]
    fn en_year_before_a_plural_noun_stays_a_year_after_a_preposition() {
        let en = |t: &str| normalize_for_speech(t, Some("en"));
        assert_eq!(
            en("In 1990 tensions rose"),
            "In nineteen ninety tensions rose"
        );
        assert_eq!(
            en("Since 1850 workers organised"),
            "Since eighteen fifty workers organised"
        );
        assert_eq!(
            en("1990 was a good year"),
            "nineteen ninety was a good year"
        );
        assert_eq!(
            en("In 1990 Shannon published"),
            "In nineteen ninety Shannon published"
        );
    }

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
        assert_eq!(normalize_for_speech("2000", Some("pl")), "dwa tysiące");
    }

    #[test]
    fn pl_thousands_agree_with_the_count() {
        let pl = |t: &str| normalize_for_speech(t, Some("pl"));
        assert_eq!(pl("1000"), "tysiąc");
        assert_eq!(pl("3000"), "trzy tysiące");
        assert_eq!(pl("5000"), "pięć tysięcy");
        assert_eq!(pl("12000"), "dwanaście tysięcy");
        assert_eq!(pl("22000"), "dwadzieścia dwa tysiące");
        assert_eq!(pl("21000"), "dwadzieścia jeden tysięcy");
        assert_eq!(pl("112000"), "sto dwanaście tysięcy");
        assert_eq!(pl("1250"), "tysiąc dwieście pięćdziesiąt");
    }

    #[test]
    fn pl_percent_and_units_agree_with_the_number() {
        let pl = |t: &str| normalize_for_speech(t, Some("pl"));
        assert_eq!(
            pl("Cena wzrosła o 15%"),
            "Cena wzrosła o piętnaście procent"
        );
        assert_eq!(pl("1 zł"), "jeden złoty");
        assert_eq!(pl("3 zł"), "trzy złote");
        assert_eq!(pl("5 zł"), "pięć złotych");
        assert_eq!(pl("12 zł"), "dwanaście złotych");
        assert_eq!(pl("22zł"), "dwadzieścia dwa złote");
        assert_eq!(pl("2.5 kg"), "dwa przecinek pięć kilograma");
        assert_eq!(pl("10 km"), "dziesięć kilometrów");
        assert_eq!(pl("2 kg"), "dwa kilogramy");
        // Not an abbreviation: left alone.
        assert_eq!(pl("5 złotych"), "pięć złotych");
        assert_eq!(pl("5 zła"), "pięć zła");
    }

    #[test]
    fn pl_years_before_roku_are_ordinals_in_the_right_case() {
        let pl = |t: &str| normalize_for_speech(t, Some("pl"));
        assert_eq!(
            pl("W 1948 roku opublikowano."),
            "W tysiąc dziewięćset czterdziestym ósmym roku opublikowano."
        );
        assert_eq!(
            pl("Od 1948 roku."),
            "Od tysiąc dziewięćset czterdziestego ósmego roku."
        );
        assert_eq!(pl("w 2024 roku"), "w dwa tysiące dwudziestym czwartym roku");
        assert_eq!(pl("do 2000 roku"), "do dwutysięcznego roku");
        assert_eq!(pl("w 1900 roku"), "w tysiąc dziewięćsetnym roku");
        assert_eq!(pl("w 2011 roku"), "w dwa tysiące jedenastym roku");
        assert_eq!(
            pl("po 1999 roku"),
            "po tysiąc dziewięćset dziewięćdziesiątym dziewiątym roku"
        );
        assert_eq!(pl("w 2003 roku"), "w dwa tysiące trzecim roku");
        assert_eq!(pl("w 1000 roku"), "w tysięcznym roku");
    }

    #[test]
    fn pl_year_abbreviation_r_becomes_roku() {
        let pl = |t: &str| normalize_for_speech(t, Some("pl"));
        assert_eq!(
            pl("W 1948 r. wydano."),
            "W tysiąc dziewięćset czterdziestym ósmym roku wydano."
        );
        // Sentence-final: the period stays.
        assert_eq!(
            pl("To było w 1948 r."),
            "To było w tysiąc dziewięćset czterdziestym ósmym roku."
        );
        assert_eq!(
            pl("Do 1948 r. Potem."),
            "Do tysiąc dziewięćset czterdziestego ósmego roku. Potem."
        );
    }

    #[test]
    fn pl_four_digit_number_without_roku_stays_a_quantity() {
        let pl = |t: &str| normalize_for_speech(t, Some("pl"));
        assert_eq!(pl("w 2000 przypadkach"), "w dwa tysiące przypadkach");
        assert_eq!(
            pl("W 1948 Shannon"),
            "W tysiąc dziewięćset czterdzieści osiem Shannon"
        );
        // Out of the year range or not 4 digits: plain cardinal.
        assert_eq!(pl("w 3500 roku"), "w trzy tysiące pięćset roku");
        assert_eq!(
            pl("w 999 roku"),
            "w dziewięćset dziewięćdziesiąt dziewięć roku"
        );
        // A decimal is never a year.
        assert_eq!(
            pl("1948.5 roku"),
            "tysiąc dziewięćset czterdzieści osiem przecinek pięć roku"
        );
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

    // ── Vietnamese ──────────────────────────────────────────────────────────

    #[test]
    fn vi_numbers_are_vietnamese_words() {
        assert_eq!(
            normalize_for_speech("Bây giờ là 8 giờ 30.", Some("vi")),
            "Bây giờ là tám giờ ba mươi."
        );
        assert_eq!(
            normalize_for_speech("năm 2026", Some("vi")),
            "năm hai nghìn không trăm hai mươi sáu"
        );
    }

    #[test]
    fn vi_does_not_get_year_style() {
        assert_eq!(
            normalize_for_speech("1990", Some("vi")),
            "một nghìn chín trăm chín mươi"
        );
    }

    #[test]
    fn vi_symbols_decimals_and_negatives() {
        assert_eq!(
            normalize_for_speech("50%", Some("vi")),
            "năm mươi phần trăm"
        );
        assert_eq!(normalize_for_speech("3.5", Some("vi")), "ba phẩy năm");
        assert_eq!(normalize_for_speech("-5", Some("vi")), "âm năm");
        assert_eq!(normalize_for_speech("mèo & chó", Some("vi")), "mèo và chó");
    }

    #[test]
    fn vi_language_code_variant_matches() {
        assert_eq!(normalize_for_speech("5", Some("vi-VN")), "năm");
    }

    /// Normalised Vietnamese contains no digits, so Piper's own expansion
    /// (`expand_numbers_vi`) is a no-op on it.
    #[test]
    fn vi_output_has_no_digits_left_for_piper() {
        let out = normalize_for_speech("Có 12 con mèo, 3.5 kg, 007.", Some("vi"));
        assert!(!out.chars().any(|c| c.is_ascii_digit()), "{out}");
    }
}
