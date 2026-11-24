//! The order `rv` prints per-model tables in: by kind in a fixed order, then by model id
//! (case-insensitive, ties broken by the exact id so the order is deterministic). The server's
//! model map is a hash map, so its order is arbitrary and can change between calls. `--json`
//! output keeps the server's order; only the human tables use this.

use std::cmp::Ordering;

/// The server's kind labels (`ModelKind::label()`), in the order `rv` lists them.
pub const KIND_ORDER: [&str; 8] = [
    "text_gen",
    "vision",
    "embedding",
    "reranking",
    "stt",
    "tts",
    "image_gen",
    "decision",
];

/// The rank of `kind`: its index in [`KIND_ORDER`]; any other kind after those; a missing (or
/// empty) kind last.
#[must_use]
pub fn kind_rank(kind: Option<&str>) -> usize {
    match kind.filter(|k| !k.is_empty()) {
        Some(k) => KIND_ORDER
            .iter()
            .position(|x| *x == k)
            .unwrap_or(KIND_ORDER.len()),
        None => KIND_ORDER.len() + 1,
    }
}

/// Model ids alphabetically, case-insensitive; equal ignoring case, by the exact id.
#[must_use]
pub fn cmp_ids(a: &str, b: &str) -> Ordering {
    a.to_lowercase()
        .cmp(&b.to_lowercase())
        .then_with(|| a.cmp(b))
}

/// `(kind, id)` pairs in table order: [`kind_rank`], then an unlisted kind's name (so unknown
/// kinds group alphabetically), then [`cmp_ids`].
#[must_use]
pub fn cmp_models(a: (Option<&str>, &str), b: (Option<&str>, &str)) -> Ordering {
    let (ka, kb) = (kind_rank(a.0), kind_rank(b.0));
    ka.cmp(&kb)
        .then_with(|| {
            if ka == KIND_ORDER.len() {
                a.0.cmp(&b.0)
            } else {
                Ordering::Equal
            }
        })
        .then_with(|| cmp_ids(a.1, b.1))
}

/// Sorts `items` in table order, `key` giving each item's `(kind, id)`.
pub fn sort_models<T, F>(items: &mut [T], key: F)
where
    F: for<'a> Fn(&'a T) -> (Option<&'a str>, &'a str),
{
    items.sort_by(|a, b| cmp_models(key(a), key(b)));
}

/// Sorts admin-style model objects (`{"id": .., "kind": ..}`) in table order.
pub fn sort_values(items: &mut [&serde_json::Value]) {
    sort_models(items, |m| {
        (
            m.get("kind").and_then(serde_json::Value::as_str),
            m.get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(""),
        )
    });
}

#[cfg(test)]
mod tests {
    use super::{KIND_ORDER, cmp_ids, kind_rank, sort_models, sort_values};
    use serde_json::json;

    #[test]
    fn kinds_rank_in_the_fixed_order_then_unknown_then_missing() {
        let ranks: Vec<usize> = KIND_ORDER.iter().map(|k| kind_rank(Some(k))).collect();
        assert_eq!(ranks, (0..8).collect::<Vec<_>>());
        assert_eq!(kind_rank(Some("text_gen")), 0);
        assert_eq!(kind_rank(Some("decision")), 7);
        assert_eq!(kind_rank(Some("llm")), 8);
        assert_eq!(kind_rank(Some("")), 9);
        assert_eq!(kind_rank(None), 9);
    }

    #[test]
    fn ids_sort_case_insensitively_with_an_exact_tie_break() {
        let mut ids = vec!["qwen", "Bielik", "Qwen", "alpha", "bielik"];
        ids.sort_by(|a, b| cmp_ids(a, b));
        assert_eq!(ids, ["alpha", "Bielik", "bielik", "Qwen", "qwen"]);
    }

    #[test]
    fn models_sort_by_kind_then_id_with_unknown_kinds_grouped() {
        let mut items: Vec<(Option<&str>, &str)> = vec![
            (None, "nokind"),
            (Some("zeta_kind"), "a"),
            (Some("decision"), "d"),
            (Some("tts"), "t"),
            (Some("alpha_kind"), "z"),
            (Some("embedding"), "e"),
            (Some("text_gen"), "b"),
            (Some("image_gen"), "i"),
            (Some("stt"), "s"),
            (Some("text_gen"), "A"),
            (Some("reranking"), "r"),
            (Some("vision"), "v"),
        ];
        sort_models(&mut items, |x| *x);
        let ids: Vec<&str> = items.iter().map(|x| x.1).collect();
        assert_eq!(
            ids,
            [
                "A", "b", "v", "e", "r", "s", "t", "i", "d", "z", "a", "nokind"
            ]
        );
    }

    #[test]
    fn values_sort_reads_kind_and_id_fields() {
        let (a, b, c) = (
            json!({"id": "emb", "kind": "embedding"}),
            json!({"id": "qwen", "kind": "text_gen"}),
            json!({"id": "x"}),
        );
        let mut v = vec![&c, &a, &b];
        sort_values(&mut v);
        let ids: Vec<&str> = v.iter().filter_map(|m| m["id"].as_str()).collect();
        assert_eq!(ids, ["qwen", "emb", "x"]);
    }
}
