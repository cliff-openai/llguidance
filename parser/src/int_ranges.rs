//! Bounded, ordered selections of inclusive integer ranges.
//!
//! The recognizer is a lazily materialized DFA: its state is part of the lexer
//! state, rather than mutable session-global data. Speculation, rollback and
//! parser clones therefore use the same mechanism as ordinary regex lexemes.

use anyhow::{ensure, Result};
use derivre::{NextByte, RandomState};
use indexmap::IndexSet;
use serde::{Deserialize, Serialize};

/// Options for the `%int_ranges` Lark primitive.
///
/// The output consists of integers and `start-end` pairs with inclusive
/// endpoints. Each entry starts strictly after the previous entry ends.
/// See `docs/syntax.md#ordered-integer-ranges`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntRangesConfig {
    #[serde(default)]
    pub min: u64,
    pub max: u64,
    /// Zero selects canonical variable-width decimals; otherwise zero-pad to
    /// exactly this many digits (at most 20).
    #[serde(default)]
    pub width: u8,
    #[serde(default = "default_separator")]
    pub separator: String,
}

fn default_separator() -> String {
    ",".to_string()
}

impl IntRangesConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.min <= self.max, "%int_ranges: min must be <= max");
        ensure!(
            self.width <= 20,
            "%int_ranges: width must be between 0 and 20"
        );
        ensure!(
            self.width == 0 || self.max.to_string().len() <= self.width as usize,
            "%int_ranges: max does not fit in width"
        );
        ensure!(
            !self.separator.is_empty()
                && !self.separator.as_bytes()[0].is_ascii_digit()
                && self.separator.as_bytes()[0] != b'-',
            "%int_ranges: separator must be nonempty and must not start with a digit or '-'"
        );
        Ok(())
    }

    pub(crate) fn alphabet_bytes(&self) -> impl Iterator<Item = u8> + '_ {
        b"0123456789-"
            .iter()
            .chain(self.separator.as_bytes())
            .copied()
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Phase {
    Start,
    End,
    Separator(usize),
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct RangeState {
    phase: Phase,
    /// Minimum value of the number being read (previous endpoint + 1, or the
    /// current start). No full history of previously emitted ranges is needed.
    lower: u64,
    value: u64,
    digits: u8,
}

#[derive(Clone)]
pub(crate) struct IntRangesDfa {
    config: IntRangesConfig,
    // Insertion indices are stable state IDs; entries are never removed.
    states: IndexSet<RangeState, RandomState>,
}

impl IntRangesDfa {
    pub(crate) fn new(config: IntRangesConfig) -> Self {
        let initial = RangeState {
            phase: Phase::Start,
            lower: config.min,
            value: 0,
            digits: 0,
        };
        Self {
            config,
            states: IndexSet::from_iter([initial]),
        }
    }

    pub(crate) fn initial_state(&self) -> u32 {
        0
    }

    fn complete_number(&self, state: &RangeState) -> bool {
        state.digits > 0
            && (self.config.width == 0 || state.digits == self.config.width)
            && state.value >= state.lower
            && state.value <= self.config.max
    }

    fn viable_number(&self, state: &RangeState) -> bool {
        let lo = state.lower as u128;
        let hi = self.config.max as u128;
        let extra = if self.config.width == 0 {
            if state.value == 0 {
                0
            } else {
                20 - state.digits
            }
        } else {
            self.config.width - state.digits
        };
        let first = if self.config.width == 0 { 0 } else { extra };
        // u128 also covers all extensions of a 20-digit u64 prefix without
        // overflow, including prefixes close to u64::MAX.
        let mut scale = 10u128.pow(first as u32);
        for _ in first..=extra {
            let start = state.value as u128 * scale;
            if start <= hi && start + scale > lo {
                return true;
            }
            if start > hi {
                break;
            }
            scale *= 10;
        }
        false
    }

    fn step(&self, mut state: RangeState, byte: u8) -> Option<RangeState> {
        if let Phase::Separator(pos) = state.phase {
            if self.config.separator.as_bytes()[pos] != byte {
                return None;
            }
            state.phase = if pos + 1 == self.config.separator.len() {
                Phase::Start
            } else {
                Phase::Separator(pos + 1)
            };
            return Some(state);
        }
        if byte.is_ascii_digit() {
            if state.digits >= 20
                || (self.config.width > 0 && state.digits >= self.config.width)
                || (self.config.width == 0 && state.digits > 0 && state.value == 0)
            {
                return None;
            }
            state.value = state
                .value
                .checked_mul(10)?
                .checked_add((byte - b'0') as u64)?;
            state.digits += 1;
            return self.viable_number(&state).then_some(state);
        }
        if !self.complete_number(&state) {
            return None;
        }
        match state.phase {
            Phase::Start if byte == b'-' => {
                state.phase = Phase::End;
                state.lower = state.value;
            }
            Phase::Start | Phase::End if byte == self.config.separator.as_bytes()[0] => {
                if state.value == self.config.max {
                    return None;
                }
                state.phase = if self.config.separator.len() == 1 {
                    Phase::Start
                } else {
                    Phase::Separator(1)
                };
                state.lower = state.value + 1;
            }
            _ => return None,
        }
        state.value = 0;
        state.digits = 0;
        Some(state)
    }

    /// Merge states only when they have exactly the same possible suffixes.
    /// Without this, speculative token-trie walks retain copies of every
    /// numeric prefix for each previously selected endpoint.
    fn canonicalize(&self, state: &mut RangeState) {
        if state.digits == 0 {
            return;
        }
        if state.value >= state.lower {
            // Appending decimal digits cannot decrease a nonnegative value.
            state.lower = 0;
            return;
        }
        let lower = state.lower as u128;
        if self.config.width > 0 {
            let scale = 10u128.pow((self.config.width - state.digits) as u32);
            if state.value as u128 * scale >= lower {
                state.lower = 0;
            }
            return;
        }
        // All completions of a canonical decimal prefix p are the disjoint
        // intervals [p*10^k, (p+1)*10^k-1]. If lower splits one interval, its
        // exact value matters. Otherwise every threshold between the preceding
        // interval and the next interval describes the same suffix language.
        let mut scale = 1u128;
        let mut boundary = 0;
        for _ in state.digits..=20 {
            let start = state.value as u128 * scale;
            let end = start + scale - 1;
            if start >= lower {
                state.lower = boundary;
                return;
            }
            if end >= lower {
                return;
            }
            // end < lower <= u64::MAX, so this conversion cannot overflow.
            boundary = (end + 1) as u64;
            scale *= 10;
        }
    }

    pub(crate) fn transition(&mut self, state: u32, byte: u8) -> Option<u32> {
        let mut next = self.step(self.states[state as usize], byte)?;
        self.canonicalize(&mut next);
        let (id, _) = self.states.insert_full(next);
        Some(u32::try_from(id).expect("native lexer state limit"))
    }

    pub(crate) fn accepting(&self, state: u32) -> bool {
        let s = &self.states[state as usize];
        state == self.initial_state() || self.complete_number(s)
    }

    pub(crate) fn next_byte(&self, state: u32) -> NextByte {
        let s = self.states[state as usize];
        if let Phase::Separator(pos) = s.phase {
            return NextByte::ForcedByte(self.config.separator.as_bytes()[pos]);
        }
        let mut next = if self.accepting(state) {
            NextByte::ForcedEOI
        } else {
            NextByte::Dead
        };
        for byte in (b'0'..=b'9').chain([b'-', self.config.separator.as_bytes()[0]]) {
            if self.step(s, byte).is_some() {
                next = next | NextByte::ForcedByte(byte);
                if next.is_some_bytes() {
                    break;
                }
            }
        }
        next
    }

    pub(crate) fn num_bytes(&self) -> usize {
        // Approximate dense entries, cached hashes, and index-table overhead,
        // allowing for table slack; IndexSet does not expose each allocation.
        self.config.separator.capacity()
            + self.states.capacity()
                * (std::mem::size_of::<RangeState>() + 3 * std::mem::size_of::<usize>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare_numeric_suffixes(dfa: &IntRangesDfa, original: RangeState, canonical: RangeState) {
        assert_eq!(
            dfa.complete_number(&original),
            dfa.complete_number(&canonical)
        );
        for byte in b'0'..=b'9' {
            let a = dfa.step(original, byte);
            let b = dfa.step(canonical, byte);
            assert_eq!(
                a.is_some(),
                b.is_some(),
                "{original:?} vs {canonical:?}, byte {byte}"
            );
            if let (Some(a), Some(b)) = (a, b) {
                compare_numeric_suffixes(dfa, a, b);
            }
        }
        for byte in [b'-', b','] {
            assert_eq!(
                dfa.step(original, byte).is_some(),
                dfa.step(canonical, byte).is_some()
            );
        }
    }

    #[test]
    fn canonical_numeric_prefixes_preserve_suffixes() {
        // Include thresholds inside and between descendant intervals, e.g.
        // prefix 3 with lower 33, 40, 299, 300 and 310. Checking every remaining
        // digit suffix catches mistakes that acceptance-only tests would miss.
        for width in [0, 4] {
            let dfa = IntRangesDfa::new(IntRangesConfig {
                min: 0,
                max: 9999,
                width,
                separator: ",".into(),
            });
            for value in [1u64, 3, 4, 9, 10, 31, 32, 33, 99, 300, 350, 999] {
                for lower in [
                    0, 1, 9, 10, 19, 31, 32, 33, 40, 99, 100, 299, 300, 310, 311, 320, 999, 1000,
                    3500, 9999,
                ] {
                    for phase in [Phase::Start, Phase::End] {
                        let original = RangeState {
                            phase,
                            lower,
                            value,
                            digits: value.to_string().len() as u8,
                        };
                        if !dfa.viable_number(&original) {
                            continue;
                        }
                        let mut canonical = original;
                        dfa.canonicalize(&mut canonical);
                        compare_numeric_suffixes(&dfa, original, canonical);
                    }
                }
            }
        }
    }
}
