//! Stateful range constraints, tested without downloading a tokenizer or model.
use std::{collections::BTreeSet, sync::Arc};

use llguidance::{
    api::TopLevelGrammar,
    earley::SlicedBiasComputer,
    toktrie::{ApproximateTokEnv, InferenceCapabilities, TokEnv, TokRxInfo, TokTrie},
    Matcher, ParserFactory,
};

fn factory(env: &TokEnv) -> ParserFactory {
    let mut factory = ParserFactory::new(
        env,
        InferenceCapabilities::default(),
        &SlicedBiasComputer::general_slices(),
    )
    .unwrap();
    factory.quiet();
    factory
}

fn matcher(grammar: &str) -> (Matcher, u32) {
    let env = ApproximateTokEnv::single_byte_env();
    let parser = factory(&env)
        .create_parser(TopLevelGrammar::from_lark(grammar.to_string()))
        .unwrap();
    (Matcher::new(Ok(parser)), env.eos_token())
}

fn tokenizer_with_extra_tokens(extra: impl IntoIterator<Item = Vec<u8>>) -> TokEnv {
    let mut words = (0..=255).map(|b| vec![b]).collect::<Vec<_>>();
    words.extend(extra);
    let eos = words.len() as u32;
    words.push(b"\xff<eos>".to_vec());
    let info = TokRxInfo {
        vocab_size: words.len() as u32,
        tok_eos: eos,
        tok_bos: None,
        tok_pad: None,
        tok_unk: None,
        tok_end_of_turn: None,
    };
    Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)))
}

fn consume_bytes(matcher: &mut Matcher, input: &str) {
    for byte in input.bytes() {
        assert!(
            matcher
                .compute_mask_or_eos()
                .unwrap()
                .is_allowed(byte.into()),
            "byte {byte:?} rejected while consuming {input:?}"
        );
        matcher.consume_token(byte.into()).unwrap();
    }
}

fn accepts(base: &Matcher, eos: u32, input: &str) -> bool {
    let mut matcher = base.clone();
    for byte in input.bytes() {
        if !matcher
            .compute_mask_or_eos()
            .unwrap()
            .is_allowed(byte.into())
        {
            return false;
        }
        matcher.consume_token(byte.into()).unwrap();
    }
    matcher.compute_mask_or_eos().unwrap().is_allowed(eos)
}

fn examples(grammar: &str, valid: &[&str], invalid: &[&str]) {
    let (base, eos) = matcher(grammar);
    for input in valid {
        assert!(accepts(&base, eos, input), "rejected {input:?}");
    }
    for input in invalid {
        assert!(!accepts(&base, eos, input), "accepted {input:?}");
    }
}

#[test]
fn variable_width_ranges() {
    examples(
        r#"start: %int_ranges {"max":999}"#,
        &[
            "",
            "0",
            "6",
            "0-0",
            "1-3,4-5,6,7-8",
            "0-32,33,311-320,999",
            "123-456",
        ],
        &[
            "01",
            "-1-3",
            "3-2",
            "0-3,3-4",
            "3-4,1-2",
            "0-0,0-0",
            "0-1000",
            "1000-1000",
            "01-2",
            "0-01",
            "0-0,",
            "0-0, 1-1",
            "0-0x",
            ",0-0",
        ],
    );
}

#[test]
fn fixed_width_and_literal_utf8_separator() {
    examples(
        r#"start: %int_ranges {"min":1,"max":999,"width":3,"separator":" → "}"#,
        &["", "001", "001-001", "001-032 → 033 → 320-999"],
        &[
            "1",
            "01",
            "000",
            "1-1",
            "01-01",
            "0001-001",
            "000-001",
            "032-031",
            "001-032 → 032-033",
            "001-032,033-033",
            "001-032 →",
            "001-032 → 033-33",
        ],
    );
}

#[test]
fn incomplete_number_prefix_remains_viable() {
    let (mut m, eos) = matcher(r#"start: %int_ranges {"max":999}"#);
    consume_bytes(&mut m, "0-32,31");
    assert!(m.compute_ff_bytes().is_empty()); // 310..319 remain distinct choices.
    let mask = m.compute_mask_or_eos().unwrap();
    assert!(!mask.is_allowed(b'-'.into()));
    assert!(!mask.is_allowed(b','.into()));
    assert!(!mask.is_allowed(eos));
    assert!(mask.is_allowed(b'1'.into()));
    consume_bytes(&mut m, "1");
    let mask = m.compute_mask_or_eos().unwrap();
    assert!(mask.is_allowed(eos)); // 311 can finish a singleton or start a range.
    assert!(mask.is_allowed(b','.into()));
    assert!(mask.is_allowed(b'-'.into()));
    consume_bytes(&mut m, "-31");
    let mask = m.compute_mask_or_eos().unwrap();
    assert!(!mask.is_allowed(b'0'.into()));
    assert!(!mask.is_allowed(eos));
    assert!(mask.is_allowed(b'1'.into()));
    consume_bytes(&mut m, "1");
    assert!(m.compute_mask_or_eos().unwrap().is_allowed(eos));
}

#[test]
fn singletons_and_ranges_share_ordering() {
    examples(
        r#"start: %int_ranges {"max":3}"#,
        &["", "0", "0-0", "0,1-2,3", "0-1,2,3", "0,1,2,3"],
        &["0,0", "0-1,1", "0,0-1", "1-2,1", "3,3", "3-3,3"],
    );
    let (mut singleton, eos) = matcher(r#"start: %int_ranges {"max":0}"#);
    assert!(singleton.compute_mask_or_eos().unwrap().is_allowed(eos));
    assert!(singleton.compute_ff_bytes().is_empty());
    consume_bytes(&mut singleton, "0");
    let mask = singleton.compute_mask_or_eos().unwrap();
    assert!(mask.is_allowed(eos));
    assert!(mask.is_allowed(b'-'.into()));
    assert!(!mask.is_allowed(b','.into()));
    assert!(singleton.compute_ff_bytes().is_empty());
    consume_bytes(&mut singleton, "-");
    assert_eq!(singleton.compute_ff_bytes(), b"0");
    consume_bytes(&mut singleton, "0");
    assert!(singleton.compute_mask_or_eos().unwrap().is_allowed(eos));
}

#[test]
fn full_u64_domain() {
    examples(
        r#"start: %int_ranges {"min":18446744073709551614,"max":18446744073709551615}"#,
        &[
            "18446744073709551615",
            "18446744073709551614,18446744073709551615",
            "18446744073709551614-18446744073709551615",
            "18446744073709551614-18446744073709551614,18446744073709551615-18446744073709551615",
        ],
        &[
            "18446744073709551616",
            "18446744073709551615,0",
            "18446744073709551613-18446744073709551615",
            "18446744073709551615-18446744073709551614",
            "18446744073709551615-18446744073709551616",
            "18446744073709551615-18446744073709551615,0-0",
        ],
    );
    examples(
        r#"start: %int_ranges {"max":18446744073709551615,"width":20}"#,
        &[
            "00000000000000000000",
            "18446744073709551615",
            "00000000000000000000-18446744073709551615",
        ],
        &[
            "18446744073709551616",
            "00000000000000000000-18446744073709551616",
        ],
    );
}

#[test]
fn grammar_composition_resets_each_list() {
    examples(
        r#"
        start: "[" ranges "];[" ranges "]" | "none"
        ranges: %int_ranges {"max":99}
        "#,
        &["none", "[];[]", "[1-99];[0]", "[0,1-2];[0-0,1,2]"],
        &["[1-99];[0-0,0-0]", "[1-99];[100-100]"],
    );
    examples(
        r#"start: %int_ranges {"max":9} | %int_ranges {"min":100,"max":199}"#,
        &["", "0-9", "100-199"],
        &["0-199", "10-99", "100-200"],
    );

    // Native, special-token, and regular lookahead lexemes share one lexer.
    // Empty selections and empty tails must preserve their distinct boundaries.
    let env = ApproximateTokEnv::single_byte_env();
    let tool = env.tok_trie().get_special_token("<|tool|>").unwrap();
    let parser = factory(&env)
        .create_parser(TopLevelGrammar::from_lark(
            r#"
            start: ranges <|tool|> tail
            ranges: %int_ranges {"max":99}
            tail[suffix="!"]: /[a-z]*/
            "#
            .into(),
        ))
        .unwrap();
    let base = Matcher::new(Ok(parser));
    for (ranges, tail) in [("", "!"), ("6", "!"), ("1-9", "abc!")] {
        let mut m = base.clone();
        consume_bytes(&mut m, ranges);
        let mask = m.compute_mask_or_eos().unwrap();
        assert!(mask.is_allowed(tool));
        assert!(!mask.is_allowed(env.eos_token()));
        m.consume_token(tool).unwrap();
        consume_bytes(&mut m, tail);
        assert!(m.compute_mask_or_eos().unwrap().is_allowed(env.eos_token()));
    }
    let mut incomplete = base.clone();
    consume_bytes(&mut incomplete, "1-");
    assert!(!incomplete.compute_mask_or_eos().unwrap().is_allowed(tool));
}

#[test]
fn invalid_configuration_is_rejected() {
    let env = ApproximateTokEnv::single_byte_env();
    let factory = factory(&env);
    for config in [
        r#"{}"#,
        r#"{"max":-1}"#,
        r#"{"max":1.5}"#,
        r#"{"max":18446744073709551616}"#,
        r#"{"max":9,"unknown":true}"#,
        r#"{"min":10,"max":9}"#,
        r#"{"max":10,"width":1}"#,
        r#"{"max":9,"width":21}"#,
        r#"{"max":9,"separator":""}"#,
        r#"{"max":9,"separator":"1,"}"#,
        r#"{"max":9,"separator":"-"}"#,
        r#"{"max":9,"min_ranges":1}"#,
        r#"{"max":9,"max_ranges":1}"#,
    ] {
        let grammar = format!("start: %int_ranges {config}");
        assert!(
            factory
                .create_parser(TopLevelGrammar::from_lark(grammar))
                .is_err(),
            "accepted invalid configuration {config}"
        );
    }
    assert!(factory
        .create_parser(TopLevelGrammar::from_lark(
            "start: RANGES\nRANGES: %int_ranges {\"max\":9}".into()
        ))
        .is_err());
}

#[test]
fn masks_cover_multi_byte_tokens_and_restore_after_rollback() {
    let extra: &[&[u8]] = &[
        b"31",
        b"31-",
        b"311-311",
        b"33-33",
        b"320-320",
        b"32-32",
        b"311-310",
        b"311-311,312-312",
        b"311-311,311-311",
        b"33,34",
        b"33,33,",
        b"311,",
        b"31,",
    ];
    let env = tokenizer_with_extra_tokens(extra.iter().map(|word| word.to_vec()));
    let eos = env.eos_token();
    let parser = factory(&env)
        .create_parser(TopLevelGrammar::from_lark(
            r#"start: %int_ranges {"max":999}"#.into(),
        ))
        .unwrap();
    let mut m = Matcher::new(Ok(parser));
    consume_bytes(&mut m, "0-32,");
    let before = m.compute_mask_or_eos().unwrap();
    for (idx, allowed) in [
        true, false, true, true, true, false, false, true, false, true, false, true, false,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            before.is_allowed(256 + idx as u32),
            allowed,
            "token {:?}",
            extra[idx]
        );
    }
    let mut cloned = m.deep_clone();
    cloned.invalidate_bias_cache();
    assert_eq!(
        before.as_slice(),
        cloned.compute_mask_or_eos().unwrap().as_slice()
    );
    m.consume_token(258).unwrap(); // 311-311 in one token
    assert!(m.compute_mask_or_eos().unwrap().is_allowed(eos));
    consume_bytes(&mut m, ",");
    assert!(!m.compute_mask_or_eos().unwrap().is_allowed(259)); // 33-33 now goes backwards
    m.rollback(2).unwrap();
    assert_eq!(
        before.as_slice(),
        m.compute_mask_or_eos().unwrap().as_slice()
    );
    cloned.consume_token(265).unwrap(); // clone independently chooses singletons 33,34
    assert!(cloned.compute_mask_or_eos().unwrap().is_allowed(eos));
    consume_bytes(&mut cloned, ",");
    assert!(cloned.compute_mask_or_eos().unwrap().is_allowed(258));
}

/// Enumerate the small domain independently of the implementation and compare
/// every next-byte/EOS mask for mixed singletons and explicit ranges.
#[test]
fn small_domain_prefix_masks_match_complete_language() {
    fn enumerate(
        prefix: String,
        first: u32,
        max: u32,
        width: usize,
        sep: &str,
        words: &mut Vec<String>,
    ) {
        words.push(prefix.clone());
        let delimiter = if prefix.is_empty() { "" } else { sep };
        for start in first..=max {
            let singleton = format!("{prefix}{delimiter}{start:0width$}");
            enumerate(singleton, start + 1, max, width, sep, words);
            for end in start..=max {
                let range = format!("{prefix}{delimiter}{start:0width$}-{end:0width$}");
                enumerate(range, end + 1, max, width, sep, words);
            }
        }
    }
    // Crossing decimal lengths makes a prefix such as "1" viable for 10/11
    // without allowing it as a complete singleton or range endpoint.
    for (min, max, width, sep) in [
        (0, 3, 0, ","),
        (0, 3, 2, "; "),
        (8, 11, 0, ","),
        (8, 11, 2, "; "),
        (98, 101, 0, ","),
        (98, 101, 3, ","),
    ] {
        let mut words = vec![];
        enumerate(String::new(), min, max, width, sep, &mut words);
        let prefixes: BTreeSet<_> = words
            .iter()
            .flat_map(|word| (0..=word.len()).map(|n| word[..n].to_string()))
            .collect();
        let config = serde_json::json!({"min":min,"max":max,"width":width,"separator":sep});
        let (base, eos) = matcher(&format!("start: %int_ranges {config}"));
        for prefix in prefixes {
            let mut m = base.clone();
            consume_bytes(&mut m, &prefix);
            let mask = m.compute_mask_or_eos().unwrap();
            assert_eq!(
                mask.is_allowed(eos),
                words.contains(&prefix),
                "EOS at {prefix:?} in {config}"
            );
            for byte in 0..=255 {
                let expected = words.iter().any(|word| {
                    word.starts_with(&prefix) && word.as_bytes().get(prefix.len()) == Some(&byte)
                });
                assert_eq!(
                    mask.is_allowed(byte.into()),
                    expected,
                    "byte {byte:?} at {prefix:?} in {config}"
                );
            }
        }
    }
}

#[test]
fn long_selection_with_numeric_tokens() {
    // Three-digit numeric tokens resemble the tokens an integer-rich output
    // uses in practice. Mask traversal also explores numbers not selected by
    // this output, exercising lexer state growth under the default limits.
    let env = tokenizer_with_extra_tokens((0..1000).map(|n| format!("{n:03}").into_bytes()));
    for width in [0, 6] {
        let config = serde_json::json!({"max":100000,"width":width});
        let parser = factory(&env)
            .create_parser(TopLevelGrammar::from_lark(format!(
                "start: %int_ranges {config}"
            )))
            .unwrap();
        let mut m = Matcher::new(Ok(parser));
        let input = (0..1000)
            .map(|n| {
                if n % 2 == 0 {
                    format!("{0:0width$}", n * 97)
                } else {
                    format!("{0:0width$}-{1:0width$}", n * 97, n * 97 + 1)
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        for token in env.tokenize(&input) {
            let mask = m
                .compute_mask_or_eos()
                .unwrap_or_else(|err| panic!("{config}: {err}"));
            assert!(mask.is_allowed(token), "token {token} in {config}");
            m.consume_token(token).unwrap();
        }
        let mask = m.compute_mask_or_eos().unwrap();
        assert!(mask.is_allowed(env.eos_token()));
    }
}
