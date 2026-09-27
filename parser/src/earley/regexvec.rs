/// This file implements regex vectors.  To match tokens to lexemes, llguidance uses
/// a DFA whose nodes are regex vectors.  For more on this see
/// S. Owens, J. Reppy, and A. Turon.
/// Regular Expression Derivatives Reexamined".
/// Journal of Functional Programming 19(2):173-190, March 2009.
/// <https://www.khoury.northeastern.edu/home/turon/re-deriv.pdf> (retrieved 15 Nov 2024)
use anyhow::{bail, Result};
use derivre::raw::{DerivCache, ExprSet, NextByteCache, RelevanceCache, VecHashCons};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use toktrie::SimpleVob;

pub use derivre::{AlphabetInfo, ExprRef, NextByte, StateID};

use crate::{
    api::ParserLimits,
    int_ranges::{IntRangesConfig, IntRangesDfa},
};

use super::lexerspec::LexemeIdx;

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct LexerStats {
    pub num_regexps: usize,
    pub num_ast_nodes: usize,
    pub num_derived: usize,
    pub num_derivatives: usize,
    pub total_fuel_spent: usize,
    pub num_states: usize,
    pub num_transitions: usize,
    pub num_bytes: usize,
    pub alphabet_size: usize,
    pub error: bool,
}

#[derive(Clone)]
pub enum MatchingLexemes {
    None,
    One(LexemeIdx),
    Two([LexemeIdx; 2]),
    Many(Vec<LexemeIdx>),
}

impl MatchingLexemes {
    pub fn is_some(&self) -> bool {
        !matches!(self, MatchingLexemes::None)
    }

    pub fn is_none(&self) -> bool {
        !self.is_some()
    }

    pub fn first(&self) -> Option<LexemeIdx> {
        match self {
            MatchingLexemes::None => None,
            MatchingLexemes::One(idx) => Some(*idx),
            MatchingLexemes::Two([idx, _]) => Some(*idx),
            MatchingLexemes::Many(v) => v.first().copied(),
        }
    }

    pub fn contains(&self, idx: LexemeIdx) -> bool {
        match self {
            MatchingLexemes::None => false,
            MatchingLexemes::One(idx2) => *idx2 == idx,
            MatchingLexemes::Two([idx1, idx2]) => *idx1 == idx || *idx2 == idx,
            MatchingLexemes::Many(v) => v.contains(&idx),
        }
    }

    pub fn add(&mut self, idx: LexemeIdx) {
        match self {
            MatchingLexemes::None => *self = MatchingLexemes::One(idx),
            MatchingLexemes::One(idx2) => {
                *self = MatchingLexemes::Two([*idx2, idx]);
            }
            MatchingLexemes::Two([idx1, idx2]) => {
                *self = MatchingLexemes::Many(vec![*idx1, *idx2, idx]);
            }
            MatchingLexemes::Many(v) => {
                v.push(idx);
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            MatchingLexemes::None => 0,
            MatchingLexemes::One(_) => 1,
            MatchingLexemes::Two(_) => 2,
            MatchingLexemes::Many(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> &[LexemeIdx] {
        match self {
            MatchingLexemes::None => &[],
            MatchingLexemes::One(idx) => std::slice::from_ref(idx),
            MatchingLexemes::Two(v) => v,
            MatchingLexemes::Many(v) => v.as_slice(),
        }
    }
}

impl Debug for MatchingLexemes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchingLexemes::None => write!(f, "Lex:[]"),
            MatchingLexemes::One(idx) => write!(f, "Lex:[{}]", idx.as_usize()),
            MatchingLexemes::Two([idx1, idx2]) => {
                write!(f, "Lex:[{},{}]", idx1.as_usize(), idx2.as_usize())
            }
            MatchingLexemes::Many(v) => write!(
                f,
                "Lex:[{}]",
                v.iter()
                    .map(|idx| idx.as_usize().to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}

impl Display for LexerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "regexps: {} with {} nodes (+ {} derived via {} derivatives with total fuel {}), states: {}; transitions: {}; bytes: {}; alphabet size: {} {}",
            self.num_regexps,
            self.num_ast_nodes,
            self.num_derived,
            self.num_derivatives,
            self.total_fuel_spent,
            self.num_states,
            self.num_transitions,
            self.num_bytes,
            self.alphabet_size,
            if self.error { "ERROR" } else { "" }
        )
    }
}

impl Debug for LexerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

#[derive(Clone, Debug)]
pub struct LexemeSet {
    vob: SimpleVob,
}

impl LexemeSet {
    pub fn new(size: usize) -> Self {
        LexemeSet {
            vob: SimpleVob::alloc(size),
        }
    }

    pub fn len(&self) -> usize {
        self.vob.len()
    }

    pub fn from_vob(vob: &SimpleVob) -> Self {
        LexemeSet { vob: vob.clone() }
    }

    pub fn is_empty(&self) -> bool {
        self.vob.is_zero()
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = LexemeIdx> + '_ {
        self.vob.iter().map(|e| LexemeIdx::new(e as usize))
    }

    pub fn add(&mut self, idx: LexemeIdx) {
        self.vob.set(idx.as_usize(), true);
    }

    pub fn remove(&mut self, idx: LexemeIdx) {
        self.vob.set(idx.as_usize(), false);
    }

    pub fn first(&self) -> Option<LexemeIdx> {
        self.vob.first_bit_set().map(LexemeIdx::new)
    }

    pub fn contains(&self, idx: LexemeIdx) -> bool {
        self.vob.get(idx.as_usize())
    }

    pub fn clear(&mut self) {
        self.vob.set_all(false);
    }
}

#[derive(Clone)]
pub struct RegexVec {
    exprs: ExprSet,
    deriv: DerivCache,
    next_byte: NextByteCache,
    relevance: RelevanceCache,
    alpha: AlphabetInfo,
    lazy: LexemeSet,
    subsumable: LexemeSet,
    // The matcher kind determines whether each interned u32 state is a regex
    // derivative or a native state ID; no per-state discriminant is needed.
    matchers: Vec<LexemeMatcher>,
    native_fuel_spent: u64,
    special_token_rx: Option<ExprRef>,
    rx_sets: VecHashCons,
    state_table: Vec<StateID>,
    state_descs: Vec<StateDesc>,
    num_transitions: usize,
    num_ast_nodes: usize,
    max_states: usize,
    fuel: u64,
}

#[derive(Clone, Debug)]
pub struct StateDesc {
    pub state: StateID,
    pub greedy_accepting: MatchingLexemes,
    pub possible: LexemeSet,

    /// Index of lowest matching regex if any.
    /// Lazy regexes match as soon as they accept, while greedy only
    /// if they accept and force EOI.
    pub lazy_accepting: MatchingLexemes,
    pub lazy_hidden_len: u32,

    pub has_special_token: bool,

    possible_lookahead_len: Option<usize>,
    lookahead_len: Option<Option<usize>>,
    next_byte: Option<NextByte>,
}

// public implementation
impl RegexVec {
    pub fn alpha(&self) -> &AlphabetInfo {
        &self.alpha
    }

    pub fn lazy_regexes(&self) -> &LexemeSet {
        &self.lazy
    }

    /// Create and return the initial state of a DFA for this
    /// regex vector
    pub fn initial_state(&mut self, selected: &LexemeSet) -> StateID {
        let mut vec_desc = vec![];
        for idx in selected.iter() {
            match &self.matchers[idx.as_usize()] {
                LexemeMatcher::Regex(rx) => {
                    if *rx != ExprRef::NO_MATCH {
                        Self::push_state(&mut vec_desc, idx, rx.as_u32());
                    }
                }
                LexemeMatcher::IntRanges(native) => {
                    Self::push_state(&mut vec_desc, idx, native.initial_state());
                }
            }
        }
        self.insert_state(vec_desc)
    }

    #[inline(always)]
    pub fn state_desc(&self, state: StateID) -> &StateDesc {
        &self.state_descs[state.as_usize()]
    }

    pub fn possible_lookahead_len(&mut self, state: StateID) -> usize {
        let desc = &mut self.state_descs[state.as_usize()];
        if let Some(len) = desc.possible_lookahead_len {
            return len;
        }
        let mut max_len = 0;
        for (idx, e) in iter_state(&self.rx_sets, state) {
            max_len =
                max_len.max(self.matchers[idx.as_usize()].possible_lookahead_len(e, &self.exprs));
        }
        desc.possible_lookahead_len = Some(max_len);
        max_len
    }

    pub fn lookahead_len_for_state(&mut self, state: StateID) -> Option<usize> {
        let desc = &mut self.state_descs[state.as_usize()];
        if desc.greedy_accepting.is_none() {
            return None;
        }
        if let Some(len) = desc.lookahead_len {
            return len;
        }
        let mut res = None;
        let exprs = &self.exprs;
        for (idx2, e) in iter_state(&self.rx_sets, state) {
            let matcher = &self.matchers[idx2.as_usize()];
            if res.is_none() && matcher.accepting(e, exprs) {
                assert!(desc.greedy_accepting.contains(idx2));
                res = Some(match matcher {
                    LexemeMatcher::Regex(_) => exprs.lookahead_len(ExprRef::new(e)).unwrap_or(0),
                    LexemeMatcher::IntRanges(_) => 0,
                });
            }
        }
        desc.lookahead_len = Some(res);
        res
    }

    /// Given a transition (a from-state and a byte) of the DFA
    /// for this regex vector, return the to-state.  It is taken
    /// from the cache, if it is cached, and created otherwise.
    #[inline(always)]
    pub fn transition(&mut self, state: StateID, b: u8) -> StateID {
        self.transition_with_cancellation(state, b, &None)
    }

    #[inline(always)]
    pub(crate) fn transition_with_cancellation(
        &mut self,
        state: StateID,
        b: u8,
        cancellation: &Option<crate::CancellationHandle>,
    ) -> StateID {
        let idx = self.alpha.map_state(state, b);
        let new_state = self.state_table[idx];
        if new_state != StateID::MISSING {
            new_state
        } else {
            self.transition_inner(state, b, idx, cancellation)
        }
    }

    /// "Subsumption" is a feature implementing regex containment.
    /// subsume_possible() returns true if it's possible for this
    /// state, false otherwise.
    pub fn subsume_possible(&mut self, state: StateID) -> bool {
        if state.is_dead() || self.has_error() {
            return false;
        }
        for (idx, _) in iter_state(&self.rx_sets, state) {
            if self.lazy.contains(idx)
                || matches!(self.matchers[idx.as_usize()], LexemeMatcher::IntRanges(_))
            {
                return false;
            }
        }
        true
    }

    /// Part of the interface for "subsumption", a feature implementing
    /// regex containment.
    pub fn check_subsume(
        &mut self,
        state: StateID,
        lexeme_idx: LexemeIdx,
        budget: u64,
    ) -> Result<bool> {
        self.check_subsume_with_cancellation(state, lexeme_idx, budget, None)
    }

    pub(crate) fn check_subsume_with_cancellation(
        &mut self,
        state: StateID,
        lexeme_idx: LexemeIdx,
        mut budget: u64,
        cancellation: Option<&crate::CancellationHandle>,
    ) -> Result<bool> {
        let budget0 = budget;
        assert!(self.subsume_possible(state));
        let LexemeMatcher::Regex(small) = self.matchers[lexeme_idx.as_usize()] else {
            return Ok(false);
        };
        let mut res = false;
        for (idx, e) in iter_state(&self.rx_sets, state) {
            if let Some(cancellation) = cancellation {
                cancellation.check()?;
            }
            if !self.subsumable.contains(idx) {
                continue;
            }
            let c0 = self.exprs.cost();
            let cache_failures = budget > budget0 / 2;
            let is_contained = self
                .relevance
                .is_contained_in_prefixes(
                    &mut self.exprs,
                    &mut self.deriv,
                    small,
                    ExprRef::new(e),
                    budget,
                    cache_failures,
                )
                .unwrap_or(false);
            // println!("chk: {} in {} -> {}",
            //     self.exprs.expr_to_string(small),
            //     self.exprs.expr_to_string(e),
            //     is_contained
            // );
            if let Some(cancellation) = cancellation {
                cancellation.check()?;
            }
            if is_contained {
                res = true;
                break;
            }
            let cost = self.exprs.cost() - c0;
            budget = budget.saturating_sub(cost);
        }
        Ok(res)
    }

    /// Estimate the size of the regex tables in bytes.
    pub fn num_bytes(&self) -> usize {
        self.exprs.num_bytes()
            + self.deriv.num_bytes()
            + self.next_byte.num_bytes()
            + self.relevance.num_bytes()
            + self.state_descs.len() * 100
            + self.state_table.len() * std::mem::size_of::<StateID>()
            + self.rx_sets.num_bytes()
            + self.matchers.capacity() * std::mem::size_of::<LexemeMatcher>()
            + self
                .matchers
                .iter()
                .map(|matcher| match matcher {
                    LexemeMatcher::Regex(_) => 0,
                    LexemeMatcher::IntRanges(native) => {
                        std::mem::size_of::<IntRangesDfa>() + native.num_bytes()
                    }
                })
                .sum::<usize>()
    }

    // Find the lowest, or best, match in 'state'.  It is the first lazy regex.
    // If there is no lazy regex, and all greedy lexemes have reached the end of
    // the lexeme, then it is the first greedy lexeme.  If neither of these
    // criteria produce a choice for "best", 'None' is returned.
    fn lowest_match_inner(&mut self, desc: &mut StateDesc) {
        // 'all_eoi' is true if all greedy lexemes match, that is, if we are at
        // the end of lexeme for all of them.  End of lexeme is called
        // "end of input" or EOI for consistency with the regex package.
        // Initially, 'all_eoi' is true, vacuously.
        let mut all_eoi = true;

        // 'eoi_candidate' tracks the lowest (aka first or best) greedy match.
        // Initially, there is none.
        let mut eois = MatchingLexemes::None;

        let mut lazies = MatchingLexemes::None;

        let mut hidden_len = 0;

        // For every regex in this state
        for (idx, e) in iter_state(&self.rx_sets, desc.state) {
            // If this lexeme is not a match.  (If the derivative at this point is nullable,
            // there is a match, so if it is not nullable, there is no match.)
            // println!("idx: {:?} e: {:?} {:?}", idx, e,self.special_token_rx);
            let matcher = &self.matchers[idx.as_usize()];
            let accepting = matcher.accepting(e, &self.exprs);
            if !accepting {
                // No match, so not at end of lexeme
                all_eoi = false;
                continue;
            } else if matches!(matcher, LexemeMatcher::Regex(rx) if Some(*rx) == self.special_token_rx)
            {
                // the regex is /\xFF\[[0-9]+\]/ so it's guaranteed not to conflict with anything
                // else (starts with non-unicode byte); thus we ignore the rest of processing
                // when has_special_token is set, we just need to make sure lazy_accepting is non-empty,
                // the actual value is not important
                desc.lazy_accepting = MatchingLexemes::One(idx);
                desc.has_special_token = true;
                return;
            }

            // If this is the first lazy lexeme, we can cut things short.  The first
            // lazy lexeme is our lowest, or best, match.  We return it and are done.
            if self.lazy.contains(idx) {
                if lazies.is_none() {
                    all_eoi = false;
                    hidden_len = matcher.possible_lookahead_len(e, &self.exprs) as u32;
                }
                lazies.add(idx);
                continue;
            }

            // If all the greedy lexemes so far are matches.
            if all_eoi {
                // If this greedy lexeme is at end of lexeme ...
                let next = matcher.next_byte(e, &self.exprs, &mut self.next_byte);
                if next == NextByte::ForcedEOI {
                    // then, if we have not yet found a matching greedy lexeme, set
                    // this one to be our lowest match ...
                    eois.add(idx);
                } else {
                    // ... otherwise, if this greedy lexeme is not yet a match, then indicate
                    // that not all greedy lexemes are matches at this point.
                    all_eoi = false;
                }
            }
        }

        if lazies.is_some() {
            desc.lazy_accepting = lazies;
            desc.lazy_hidden_len = hidden_len;
        } else if all_eoi {
            desc.lazy_accepting = eois;
            // no hidden len
        }
    }

    /// Check if the there is only one transition out of state.
    /// This is an approximation - see docs for NextByte.
    pub fn next_byte(&mut self, state: StateID) -> NextByte {
        let desc = &mut self.state_descs[state.as_usize()];
        if let Some(next_byte) = desc.next_byte {
            return next_byte;
        }

        let mut next_byte = NextByte::Dead;
        for (idx, e) in iter_state(&self.rx_sets, state) {
            let next = self.matchers[idx.as_usize()].next_byte(e, &self.exprs, &mut self.next_byte);
            next_byte = next_byte | next;
            if next_byte.is_some_bytes() {
                break;
            }
        }

        desc.next_byte = Some(next_byte);
        next_byte
    }

    pub fn limit_state_to(&mut self, state: StateID, allowed_lexemes: &LexemeSet) -> StateID {
        let mut vec_desc = vec![];
        for (idx, e) in iter_state(&self.rx_sets, state) {
            if allowed_lexemes.contains(idx) {
                Self::push_state(&mut vec_desc, idx, e);
            }
        }
        self.insert_state(vec_desc)
    }

    pub fn total_fuel_spent(&self) -> u64 {
        self.exprs.cost() + self.native_fuel_spent
    }

    pub fn lexeme_weight(&mut self, lexeme_idx: LexemeIdx) -> u32 {
        match self.matchers[lexeme_idx.as_usize()] {
            LexemeMatcher::Regex(e) => self.exprs.get_weight(e),
            LexemeMatcher::IntRanges(_) => 1,
        }
    }

    pub fn set_max_states(&mut self, max_states: usize) {
        if !self.has_error() {
            self.max_states = max_states;
        }
    }

    // Each fuel point is on the order 100ns (though it varies).
    // So, for ~10ms limit, do a .set_fuel(100_000).
    pub fn set_fuel(&mut self, fuel: u64) {
        if !self.has_error() {
            self.fuel = fuel;
        }
    }

    pub fn get_fuel(&self) -> u64 {
        self.fuel
    }

    pub fn has_error(&self) -> bool {
        self.alpha.has_error()
    }

    pub fn get_error(&self) -> Option<String> {
        if self.has_error() {
            if self.fuel == 0 {
                Some("too many expressions constructed".to_string())
            } else if self.state_descs.len() >= self.max_states {
                Some(format!(
                    "too many states: {} >= {}",
                    self.state_descs.len(),
                    self.max_states
                ))
            } else {
                Some("unknown error".to_string())
            }
        } else {
            None
        }
    }

    pub fn stats(&self) -> LexerStats {
        LexerStats {
            num_regexps: self.matchers.len(),
            num_ast_nodes: self.num_ast_nodes,
            num_derived: self.exprs.len() - self.num_ast_nodes,
            num_derivatives: self.deriv.num_deriv,
            total_fuel_spent: self.total_fuel_spent() as usize,
            num_states: self.state_descs.len(),
            num_transitions: self.num_transitions,
            num_bytes: self.num_bytes(),
            alphabet_size: self.alpha.len(),
            error: self.has_error(),
        }
    }

    pub fn print_state_table(&self) {
        for (state, row) in self.state_table.chunks(self.alpha.len()).enumerate() {
            println!("state: {state}");
            for (b, &new_state) in row.iter().enumerate() {
                println!("  s{b:?} -> {new_state:?}");
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LexemeMatcherSpec {
    Regex(ExprRef),
    IntRanges(Box<IntRangesConfig>),
}

#[derive(Clone)]
enum LexemeMatcher {
    Regex(ExprRef),
    IntRanges(Box<IntRangesDfa>),
}

impl LexemeMatcher {
    fn accepting(&self, state: u32, exprs: &ExprSet) -> bool {
        match self {
            Self::Regex(_) => exprs.is_nullable(ExprRef::new(state)),
            Self::IntRanges(native) => native.accepting(state),
        }
    }

    fn possible_lookahead_len(&self, state: u32, exprs: &ExprSet) -> usize {
        match self {
            Self::Regex(_) => exprs.possible_lookahead_len(ExprRef::new(state)),
            Self::IntRanges(_) => 0,
        }
    }

    fn next_byte(&self, state: u32, exprs: &ExprSet, cache: &mut NextByteCache) -> NextByte {
        match self {
            Self::Regex(_) => cache.next_byte(exprs, ExprRef::new(state)),
            Self::IntRanges(native) => native.next_byte(state),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RxLexeme {
    pub matcher: LexemeMatcherSpec,
    pub lazy: bool,
    #[allow(dead_code)]
    pub priority: i32,
}

// private implementation
impl RegexVec {
    pub(crate) fn new_with_exprset(
        mut exprset: ExprSet,
        rx_lexemes: Vec<RxLexeme>,
        special_token_rx: Option<ExprRef>,
        limits: &mut ParserLimits,
    ) -> Result<Self> {
        // Native matchers have no regex root. Keep the regex roots in descriptor
        // order so alphabet compression can remap them without placeholder expressions.
        let mut roots: Vec<_> = rx_lexemes
            .iter()
            .filter_map(|r| match &r.matcher {
                LexemeMatcherSpec::Regex(rx) => Some(*rx),
                LexemeMatcherSpec::IntRanges(_) => None,
            })
            .collect();
        let num_regexps = roots.len();
        let spec_pos = special_token_rx.and_then(|rx| roots.iter().position(|r| *r == rx));
        // Regex-equivalent bytes need not be equivalent to a native recognizer:
        // each decimal digit changes its numeric state. Temporary singleton roots
        // prevent alphabet compression from merging those transition-cache keys.
        let mut native_bytes = [false; 256];
        for r in &rx_lexemes {
            if let LexemeMatcherSpec::IntRanges(config) = &r.matcher {
                for byte in config.alphabet_bytes() {
                    native_bytes[byte as usize] = true;
                }
            }
        }
        for (byte, required) in native_bytes.iter().enumerate() {
            if *required {
                roots.push(exprset.mk_byte(byte as u8));
            }
        }
        let (alpha, mut exprset, mut compressed_roots) =
            AlphabetInfo::from_exprset(exprset, &roots);
        compressed_roots.truncate(num_regexps);
        let special_token_rx = spec_pos.map(|pos| compressed_roots[pos]);
        let num_ast_nodes = exprset.len();
        let mut compressed_roots = compressed_roots.into_iter();

        let fuel0 = limits.initial_lexer_fuel;
        let mut relevance = RelevanceCache::new();
        let mut lazy = LexemeSet::new(rx_lexemes.len());
        let mut subsumable = LexemeSet::new(rx_lexemes.len());
        let mut matchers = Vec::with_capacity(rx_lexemes.len());
        for (idx, r) in rx_lexemes.into_iter().enumerate() {
            if r.lazy {
                lazy.add(LexemeIdx::new(idx));
            }
            let matcher = match r.matcher {
                LexemeMatcherSpec::Regex(_) => {
                    let mut rx = compressed_roots.next().unwrap();
                    let c0 = exprset.cost();
                    match relevance.is_non_empty_limited(
                        &mut exprset,
                        rx,
                        limits.initial_lexer_fuel,
                    ) {
                        Ok(true) => {}
                        Ok(false) => rx = ExprRef::NO_MATCH,
                        Err(_) => {
                            bail!(
                                "fuel exhausted when checking relevance of lexemes ({})",
                                fuel0
                            );
                        }
                    }
                    limits.initial_lexer_fuel = limits
                        .initial_lexer_fuel
                        .saturating_sub(exprset.cost() - c0);
                    if !r.lazy && exprset.attr_has_repeat(rx) {
                        subsumable.add(LexemeIdx::new(idx));
                    }
                    LexemeMatcher::Regex(rx)
                }
                LexemeMatcherSpec::IntRanges(config) => {
                    LexemeMatcher::IntRanges(Box::new(IntRangesDfa::new(*config)))
                }
            };
            matchers.push(matcher);
        }
        assert!(compressed_roots.next().is_none());

        let rx_sets = StateID::new_hash_cons();
        let mut r = RegexVec {
            deriv: DerivCache::new(),
            next_byte: NextByteCache::new(),
            special_token_rx,
            relevance,
            lazy,
            subsumable,
            exprs: exprset,
            alpha,
            matchers,
            native_fuel_spent: 0,
            rx_sets,
            state_table: vec![],
            state_descs: vec![],
            num_transitions: 0,
            num_ast_nodes,
            fuel: u64::MAX,
            max_states: usize::MAX,
        };

        assert!(r.lazy.len() == r.matchers.len());

        r.insert_state(vec![]);
        // also append state for the "MISSING"
        r.append_state(r.state_descs[0].clone());
        // in fact, transition from MISSING and DEAD should both lead to DEAD
        r.state_table.fill(StateID::DEAD);
        assert!(!r.alpha.is_empty());
        Ok(r)
    }

    fn append_state(&mut self, state_desc: StateDesc) {
        let mut new_states = vec![StateID::MISSING; self.alpha.len()];
        self.state_table.append(&mut new_states);
        self.state_descs.push(state_desc);
        if self.state_descs.len() >= self.max_states {
            self.alpha.enter_error_state();
        }
    }

    fn insert_state(&mut self, lst: Vec<u32>) -> StateID {
        // does this help?
        // if lst.len() == 0 {
        //     return StateID::DEAD;
        // }
        assert!(lst.len().is_multiple_of(2));
        let id = StateID::new(self.rx_sets.insert(&lst));
        if id.as_usize() >= self.state_descs.len() {
            let state_desc = self.compute_state_desc(id);
            self.append_state(state_desc);
        }
        if self.state_desc(id).lazy_accepting.is_some() {
            id._set_lowest_match()
        } else {
            id
        }
    }

    fn compute_state_desc(&mut self, state: StateID) -> StateDesc {
        let mut res = StateDesc {
            state,
            greedy_accepting: MatchingLexemes::None,
            possible: LexemeSet::new(self.matchers.len()),
            possible_lookahead_len: None,
            lookahead_len: None,
            next_byte: None,
            lazy_accepting: MatchingLexemes::None,
            lazy_hidden_len: 0,
            has_special_token: false,
        };
        for (idx, e) in iter_state(&self.rx_sets, state) {
            res.possible.add(idx);
            let accepting = self.matchers[idx.as_usize()].accepting(e, &self.exprs);
            if accepting {
                res.greedy_accepting.add(idx);
            }
        }

        if res.possible.is_empty() {
            assert!(state == StateID::DEAD);
        }

        self.lowest_match_inner(&mut res);

        // println!("state {:?} desc: {:?}", state, res);

        res
    }

    fn push_state(vec_desc: &mut Vec<u32>, idx: LexemeIdx, state: u32) {
        vec_desc.push(idx.as_usize() as u32);
        vec_desc.push(state);
    }

    /// Given a transition (from-state and byte), create the to-state.
    /// It is assumed the to-state does not exist.
    fn transition_inner(
        &mut self,
        state: StateID,
        b: u8,
        idx: usize,
        cancellation: &Option<crate::CancellationHandle>,
    ) -> StateID {
        assert!(state.is_valid());

        let mut vec_desc = vec![];

        // let d0 = self.deriv.num_deriv;
        let c0 = self.exprs.cost();
        let mut native_cost = 0;
        // let t0 = crate::Instant::now();
        // let mut state_size = 0;

        for (idx, e) in iter_state(&self.rx_sets, state) {
            if cancellation.as_ref().is_some_and(|c| c.is_cancelled()) {
                return StateID::DEAD;
            }
            let e = match &mut self.matchers[idx.as_usize()] {
                LexemeMatcher::Regex(_) => ExprRef::new(e),
                LexemeMatcher::IntRanges(native) => {
                    // Native transitions have bounded work (at most 20 decimal digits).
                    // Charge them to the same runtime budget as regex derivatives.
                    native_cost += 1;
                    self.native_fuel_spent += 1;
                    if self.fuel <= native_cost + self.exprs.cost() - c0 {
                        self.fuel = 0;
                        break;
                    }
                    if let Some(next) = native.transition(e, b) {
                        Self::push_state(&mut vec_desc, idx, next);
                    }
                    #[cfg(test)]
                    crate::cancellation::checkpoint("lexer");
                    continue;
                }
            };
            let d = self.deriv.derivative(&mut self.exprs, e, b);
            if cancellation.as_ref().is_some_and(|c| c.is_cancelled()) {
                return StateID::DEAD;
            }

            let fuel = self
                .fuel
                .saturating_sub(self.exprs.cost() - c0 + native_cost);
            let non_empty = self
                .relevance
                .is_non_empty_limited(&mut self.exprs, d, fuel);
            if cancellation.as_ref().is_some_and(|c| c.is_cancelled()) {
                return StateID::DEAD;
            }
            let d = match non_empty {
                Ok(true) => d,
                Ok(false) => ExprRef::NO_MATCH,
                Err(_) => {
                    self.fuel = 0; // just in case
                    break;
                }
            };

            #[cfg(test)]
            crate::cancellation::checkpoint("lexer");
            // state_size += 1;
            if d != ExprRef::NO_MATCH {
                Self::push_state(&mut vec_desc, idx, d.as_u32());
            }
        }

        // Do not publish an incomplete transition or change shared error state.
        if cancellation.as_ref().is_some_and(|c| c.is_cancelled()) {
            return StateID::DEAD;
        }

        // let num_deriv = self.deriv.num_deriv - d0;
        let cost = self.exprs.cost() - c0 + native_cost;
        self.fuel = self.fuel.saturating_sub(cost);
        if self.fuel == 0 {
            self.alpha.enter_error_state();
        }
        // if false && cost > 40 {
        //     eprintln!(
        //         "cost: {:?} {} {} size={}",
        //         t0.elapsed() / (cost as u32),
        //         num_deriv,
        //         cost,
        //         state_size
        //     );

        //     // for (idx, e) in iter_state(&self.rx_sets, state) {
        //     //     eprintln!("expr{}: {}", idx, self.exprs.expr_to_string(e));
        //     // }
        // }
        let new_state = self.insert_state(vec_desc);
        self.num_transitions += 1;
        self.state_table[idx] = new_state;
        new_state
    }
}

impl Debug for RegexVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RegexVec({})", self.stats())
    }
}

fn iter_state(
    rx_sets: &VecHashCons,
    state: StateID,
) -> impl Iterator<Item = (LexemeIdx, u32)> + '_ {
    let lst = rx_sets.get(state.as_u32());
    (0..lst.len())
        .step_by(2)
        .map(move |idx| (LexemeIdx::new(lst[idx] as usize), lst[idx + 1]))
}

// #[test]
// fn test_fuel() {
//     let mut rx = RegexVec::new_single("a(bc+|b[eh])g|.h").unwrap();
//     println!("{:?}", rx);
//     rx.set_fuel(200);
//     match_(&mut rx, "abcg");
//     assert!(!rx.has_error());
//     let mut rx = RegexVec::new_single("a(bc+|b[eh])g|.h").unwrap();
//     println!("{:?}", rx);
//     rx.set_fuel(20);
//     no_match(&mut rx, "abcg");
//     assert!(rx.has_error());
// }
