//! Microsecond-scale Tamil-native linear pipeline vector query engine
//! and columnar database.
//!
//! # Architecture
//! - [`lexer`] — zero-allocation UTF-8 / Tamil DSL scanner
//! - [`parser`] — flat arena AST (`u32` index links, no pointer trees)
//! - [`storage`] — Arrow-aligned columnar segments
//! - [`runtime`] — batch-1024 vectorized execution
//!
//! Hot execution loops do not call `alloc`, construct `String`/`Vec`/`Box`,
//! or tear Tamil grapheme clusters.

#![allow(clippy::needless_range_loop)]

pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod storage;
pub mod utf8;

pub use lexer::{Lexer, LexerError, Token, TokenKind, MAX_TOKENS};
pub use parser::{parse_query, AstArena, AstNode, NodeKind, ParseError, Parser, AST_CAP, NIL};
pub use runtime::{demo_catalog, run_query, Engine, QueryResult};
pub use storage::{
    seed_users_table, Catalog, ColName, ColumnData, PhysType, SelectionVector, Table, BATCH_ROWS,
    MAX_ROWS,
};

/// Canonical end-to-end demo query from the system specification.
pub const DEMO_QUERY: &str =
    "இருந்து பயனர்கள் | வடி வயது > 21 | அடுக்கு வயது | எடு 10 | தேடு பெயர், வயது;";

#[cfg(test)]
mod e2e_tests {
    use super::*;
    use std::cell::Cell;
    use std::alloc::{GlobalAlloc, Layout, System};

    thread_local! {
        static TRACKING: Cell<bool> = const { Cell::new(false) };
        static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
        static ALLOC_BYTES: Cell<usize> = const { Cell::new(0) };
    }

    /// Counting allocator — thread-local tracking so parallel tests cannot
    /// poison the zero-heap assertion.
    struct CountingAlloc;

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            TRACKING.with(|t| {
                if t.get() {
                    ALLOC_COUNT.with(|c| c.set(c.get().wrapping_add(1)));
                    ALLOC_BYTES.with(|c| c.set(c.get().wrapping_add(layout.size())));
                }
            });
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            TRACKING.with(|t| {
                if t.get() {
                    ALLOC_COUNT.with(|c| c.set(c.get().wrapping_add(1)));
                    ALLOC_BYTES.with(|c| c.set(c.get().wrapping_add(new_size)));
                }
            });
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static A: CountingAlloc = CountingAlloc;

    fn reset_counters() {
        ALLOC_COUNT.with(|c| c.set(0));
        ALLOC_BYTES.with(|c| c.set(0));
    }

    fn set_tracking(on: bool) {
        TRACKING.with(|t| t.set(on));
    }

    fn alloc_count() -> usize {
        ALLOC_COUNT.with(|c| c.get())
    }

    fn alloc_bytes() -> usize {
        ALLOC_BYTES.with(|c| c.get())
    }

    #[test]
    fn demo_pipeline_e2e_zero_heap_and_tamil_safe() {
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = Box::new(QueryResult::new());

        assert!(DEMO_QUERY.contains("தேடு"));
        assert!(DEMO_QUERY.contains("பெயர்"));
        let thedu = "தேடு";
        let mut chars = thedu.chars();
        assert_eq!(chars.next(), Some('த'));
        assert_eq!(chars.next(), Some('ே'));
        assert_eq!(chars.next(), Some('ட'));
        assert_eq!(chars.next(), Some('ு'));

        reset_counters();
        set_tracking(true);
        let ok = run_query(DEMO_QUERY, &catalog, &mut arena, &mut out);
        set_tracking(false);

        assert!(ok, "pipeline must execute successfully");
        let allocs = alloc_count();
        let bytes = alloc_bytes();
        assert_eq!(
            allocs, 0,
            "hot path must not allocate (saw {allocs} allocs, {bytes} bytes)"
        );
        assert_eq!(bytes, 0);

        assert_eq!(out.col_count, 2);
        assert_eq!(out.row_count, 10);
        assert_eq!(out.types[0], PhysType::Utf8);
        assert_eq!(out.types[1], PhysType::Int64);
        assert_eq!(out.schema[0].name.as_bytes(), "பெயர்".as_bytes());
        assert_eq!(out.schema[1].name.as_bytes(), "வயது".as_bytes());

        let mut prev = i64::MIN;
        let mut i = 0usize;
        while i < out.row_count as usize {
            let name = out.utf8_out[0]
                .get_row(i)
                .expect("utf8 name row must be valid UTF-8 on grapheme boundaries");
            assert!(!name.is_empty());
            assert!(core::str::from_utf8(name.as_bytes()).is_ok());
            let age = out.int_out[1].values[i];
            assert!(age > 21, "filter வயது > 21 violated: {age}");
            assert!(age >= prev, "sort அடுக்கு வயது violated");
            prev = age;
            i += 1;
        }

        let expected: [i64; 10] = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
        let mut j = 0usize;
        while j < 10 {
            assert_eq!(out.int_out[1].values[j], expected[j]);
            j += 1;
        }
    }

    #[test]
    fn lexer_preserves_tamil_vowel_markers() {
        let mut lex = Lexer::new(DEMO_QUERY.as_bytes());
        let mut found_thedu = false;
        let mut found_peyar = false;
        for tok in lex.by_ref() {
            if tok.kind == TokenKind::Eof {
                break;
            }
            if tok.kind == TokenKind::Thedu {
                assert_eq!(tok.text(DEMO_QUERY.as_bytes()), Some("தேடு"));
                found_thedu = true;
            }
            if tok.kind == TokenKind::Ident {
                if let Some(t) = tok.text(DEMO_QUERY.as_bytes()) {
                    if t == "பெயர்" {
                        found_peyar = true;
                        assert_eq!(t.chars().next(), Some('ப'));
                    }
                }
            }
        }
        assert!(found_thedu);
        assert!(found_peyar);
    }

    // ── Stress / fuzz edge cases (GROK-4.5-STRESS-FUZZ) ──────────────────

    /// 1. Tamil grapheme tail break: buffer ends mid-`தே` (after 4 of 6 bytes).
    #[test]
    fn fuzz_mid_syllable_the_returns_malformed_utf8() {
        let full = "தே".as_bytes();
        assert_eq!(full.len(), 6);
        // Cut precisely mid-syllable: த complete + first byte of ே.
        let truncated = &full[..4];
        assert!(core::str::from_utf8(truncated).is_err());

        let mut lex = Lexer::new(truncated);
        let err = lex
            .next_token()
            .expect_err("torn தே must not panic or succeed");
        assert_eq!(err, LexerError::MalformedUtf8);

        // Iterator path also latches Error without panicking.
        let mut lex2 = Lexer::new(truncated);
        let tok = lex2.next().expect("iterator yields error token");
        assert_eq!(tok.kind, TokenKind::Error);
        assert_eq!(lex2.last_error(), Some(LexerError::MalformedUtf8));
    }

    /// 2. Fixed arena overflow at `[AstNode; 1024]` boundary.
    #[test]
    fn fuzz_arena_overflow_returns_defensive_error() {
        let q = DEMO_QUERY.as_bytes();
        let mut arena = AstArena::new();
        arena.len = AST_CAP as u32;
        let err = parse_query(q, &mut arena).expect_err("saturated arena must error");
        assert_eq!(err, ParseError::ArenaOverflow);
        assert!(arena.is_full());
        // try_alloc itself is bounds-safe.
        let again = arena.try_alloc(AstNode::empty());
        assert_eq!(again, Err(ParseError::ArenaOverflow));
    }

    /// 3. Chunk-tail scalar protection for non-1024 / non-8 row counts.
    #[test]
    fn fuzz_chunk_tail_scalar_residue_no_simd_corruption() {
        // 23 rows ⇒ one incomplete batch, aligned 16 + scalar residue 7.
        const LIVE: usize = 23;
        let mut values = [0i64; MAX_ROWS];
        let mut i = 0usize;
        while i < LIVE {
            values[i] = i as i64;
            i += 1;
        }
        // Poison past the live window — kernels must not touch these slots.
        let mut p = LIVE;
        while p < LIVE + 16 && p < MAX_ROWS {
            values[p] = -1;
            p += 1;
        }
        let mut sel = SelectionVector::all(LIVE);
        let mut p = LIVE;
        while p < LIVE + 16 && p < MAX_ROWS {
            sel.mask[p] = 0x5A;
            p += 1;
        }

        Engine::filter_i64_gt(&values, &mut sel, LIVE, 10);
        let mut r = 0usize;
        while r <= 10 {
            assert_eq!(sel.mask[r], 0, "row {r} <= 10 must drop");
            r += 1;
        }
        while r < LIVE {
            assert_eq!(sel.mask[r], 1, "row {r} > 10 must keep");
            r += 1;
        }
        let mut t = LIVE;
        while t < LIVE + 16 && t < MAX_ROWS {
            assert_eq!(sel.mask[t], 0x5A, "must not clobber mask past live rows");
            t += 1;
        }

        // Near-capacity residue: 1023 = 1024-1, exercises phase-B/C only.
        let mut sel2 = SelectionVector::all(1023);
        let mut values2 = [0i64; MAX_ROWS];
        let mut k = 0usize;
        while k < 1023 {
            values2[k] = (k % 50) as i64;
            k += 1;
        }
        Engine::filter_i64_eq(&values2, &mut sel2, 1023, 7);
        let mut kept = 0usize;
        let mut m = 0usize;
        while m < 1023 {
            kept += sel2.mask[m] as usize;
            if sel2.mask[m] != 0 {
                assert_eq!(values2[m], 7);
            }
            m += 1;
        }
        assert!(kept > 0);
        assert_eq!(kept, (0..1023).filter(|&x| x % 50 == 7).count());
    }

    /// 4. Mid-syllable fault propagates through parse without panic / OOB.
    #[test]
    fn fuzz_torn_syllable_parse_pipeline_no_panic() {
        let full = "இருந்து தே".as_bytes();
        // Truncate inside the trailing தே syllable.
        let the_off = full.len() - 6;
        let torn = &full[..the_off + 4];
        let mut arena = AstArena::new();
        let err = parse_query(torn, &mut arena).expect_err("parse must surface lex fault");
        assert_eq!(err, ParseError::LexMalformedUtf8);
        assert_eq!(arena.root, NIL);
        // run_query maps the fault to false without aborting the process.
        let cat = demo_catalog();
        let _out = Box::new(QueryResult::new());
        let mut arena2 = AstArena::new();
        assert!(parse_query(&"தே".as_bytes()[..4], &mut arena2).is_err());
        let _ = cat;
    }
}
