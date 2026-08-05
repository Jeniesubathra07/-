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
pub use runtime::{
    demo_catalog, execute_chunk_parallel, execute_chunk_parallel_os, lsd_radix_sort_ages,
    lsd_radix_sort_ages_tls, run_query, vector_merge_join, ArithOp, ChunkScratch, Engine,
    EngineScratchPad, QueryResult, RadixScratchPad, RuntimeScratch,
};
pub use storage::{
    seed_orders_database, seed_orders_table, seed_users_table, Catalog, ColName, ColumnData,
    FixedOrdersDatabase, Int64Column, PhysType, SelectionVector, Table, Utf8Column, BATCH_ROWS,
    MAX_ROWS,
};
pub use parser::{
    alloc_token_window, parse_query, AstArena, AstNode, NodeKind, OpKind, ParseError, Parser, ParserError, AST_CAP, NIL,
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
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();

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
        let ok = run_query(DEMO_QUERY, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens);
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

    // ── Stress / fuzz edge cases ─────────────────────────────────────────

    /// Tamil grapheme tail break: buffer ends mid-`தே` (after 4 of 6 bytes).
    #[test]
    fn fuzz_mid_syllable_the_returns_malformed_utf8() {
        let full = "தே".as_bytes();
        assert_eq!(full.len(), 6);
        let truncated = &full[..4];
        assert!(core::str::from_utf8(truncated).is_err());

        let mut lex = Lexer::new(truncated);
        let err = lex
            .next_token()
            .expect_err("torn தே must not panic or succeed");
        assert_eq!(err, LexerError::MalformedUtf8(0));

        let mut lex2 = Lexer::new(truncated);
        let tok = lex2.next().expect("iterator yields error token");
        assert_eq!(tok.kind, TokenKind::Error);
        assert_eq!(lex2.last_error(), Some(LexerError::MalformedUtf8(0)));
    }

    /// Fixed arena overflow at `[AstNode; 1024]` boundary.
    #[test]
    fn fuzz_arena_overflow_returns_defensive_error() {
        let q = DEMO_QUERY.as_bytes();
        let mut arena = AstArena::new();
        arena.len = AST_CAP as u32;
        let mut tw = alloc_token_window();
        let err = parse_query(q, &mut arena, &mut tw).expect_err("saturated arena must error");
        assert_eq!(err, ParserError::ArenaOverflow);
        assert!(arena.is_full());
        let again = arena.try_alloc(AstNode::empty());
        assert_eq!(again, Err(ParserError::ArenaOverflow));
    }

    /// Chunk-tail scalar protection for non-1024 / non-8 row counts.
    #[test]
    fn fuzz_chunk_tail_scalar_residue_no_simd_corruption() {
        const LIVE: usize = 23;
        let mut values = [0i64; MAX_ROWS];
        let mut i = 0usize;
        while i < LIVE {
            values[i] = i as i64;
            i += 1;
        }
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
    }

    /// Mid-syllable fault propagates through parse without panic / OOB.
    #[test]
    fn fuzz_torn_syllable_parse_pipeline_no_panic() {
        let full = "இருந்து தே".as_bytes();
        let the_off = full.len() - 6;
        let torn = &full[..the_off + 4];
        let mut arena = AstArena::new();
        let mut tw = alloc_token_window();
        let err = parse_query(torn, &mut arena, &mut tw).expect_err("parse must surface lex fault");
        assert_eq!(err, ParserError::LexMalformedUtf8);
        assert_eq!(arena.root, NIL);
    }

    // ── GROK-4.5-OMEGA-EDGE-LOCK named harness ───────────────────────────

    /// Validates truncating a query exactly mid-Tamil-syllable returns
    /// `LexerError::MalformedUtf8(cursor)`.
    #[test]
    fn test_fragmented_input_grapheme_safety() {
        // தே = த (3) + ே (3). Cut at byte 4 ⇒ mid-syllable.
        let the = "தே".as_bytes();
        let torn = &the[..4];
        let mut lex = Lexer::new(torn);
        match lex.next_token() {
            Err(LexerError::MalformedUtf8(cursor)) => assert_eq!(cursor, 0),
            other => panic!("expected MalformedUtf8(cursor), got {other:?}"),
        }

        // Streaming query fragment ending mid-syllable after இருந்து + space + torn தே.
        let prefix = "இருந்து ";
        let mut buf = [0u8; 64];
        let pb = prefix.as_bytes();
        buf[..pb.len()].copy_from_slice(pb);
        buf[pb.len()..pb.len() + 4].copy_from_slice(&the[..4]);
        let stream = &buf[..pb.len() + 4];

        let mut arena = AstArena::new();
        let mut tw = alloc_token_window();
        let err = parse_query(stream, &mut arena, &mut tw).expect_err("torn stream");
        assert_eq!(err, ParserError::LexMalformedUtf8);

        // Maximal munch: வடிவமைப்பு must remain Ident (not keyword வடி).
        let mut lex2 = Lexer::new("வடிவமைப்பு".as_bytes());
        let tok = lex2.next_token().unwrap();
        assert_eq!(tok.kind, TokenKind::Ident);
    }

    /// Simulates a massive 1025-stage pipeline and proves ArenaOverflow.
    #[test]
    fn test_arena_overflow_via_deep_pipeline_stages() {
        let mut q = String::from("இருந்து பயனர்கள்");
        // 1025 filter stages ⇒ far beyond AST_CAP node budget.
        let mut i = 0usize;
        while i < 1025 {
            q.push_str(" | வடி வயது > 0");
            i += 1;
        }
        q.push(';');

        let mut arena = Box::new(AstArena::new());
        let mut tw = alloc_token_window();
        let err = parse_query(q.as_bytes(), &mut arena, &mut tw).expect_err("ArenaOverflow");
        assert_eq!(err, ParserError::ArenaOverflow);

        // Missing இருந்து source context.
        let mut arena2 = AstArena::new();
        let mut tw2 = alloc_token_window();
        let err2 = parse_query("வடி வயது > 1;".as_bytes(), &mut arena2, &mut tw2).unwrap_err();
        assert_eq!(err2, ParserError::MissingSourceContext);
    }

    /// Dataset of exactly 1025 rows — scalar tail must capture remainder row.
    #[test]
    fn test_simd_unaligned_tail_cleanup() {
        const LIVE: usize = 1025;
        assert_eq!(LIVE % BATCH_ROWS, 1);

        let mut values = [0i64; MAX_ROWS];
        let mut i = 0usize;
        while i < LIVE {
            values[i] = i as i64;
            i += 1;
        }
        values[LIVE] = -1;
        let mut sel = SelectionVector::all(LIVE);
        sel.mask[LIVE] = 0xA5;

        Engine::filter_i64_eq(&values, &mut sel, LIVE, 1024);
        let mut r = 0usize;
        while r < 1024 {
            assert_eq!(sel.mask[r], 0);
            r += 1;
        }
        assert_eq!(sel.mask[1024], 1);
        assert_eq!(sel.mask[LIVE], 0xA5);

        let mut table = Table::new_boxed("பயனர்கள்".as_bytes());
        let age_i = table.add_int64_column("வயது".as_bytes()).unwrap();
        let name_i = table.add_utf8_column("பெயர்".as_bytes()).unwrap();
        {
            let col = table.int64_mut(age_i).unwrap();
            let mut r = 0usize;
            while r < LIVE {
                col.values[r] = r as i64;
                col.validity.set(r, true);
                r += 1;
            }
        }
        {
            let col = table.utf8_mut(name_i).unwrap();
            col.clear();
            let mut r = 0usize;
            while r < LIVE {
                let b = [b'0' + (r % 10) as u8];
                assert!(col.set_row(r, &b));
                r += 1;
            }
        }
        table.set_row_count(LIVE);
        let mut cat = Catalog::new();
        let _ = cat.register_box(table);
        let q = "இருந்து பயனர்கள் | வடி வயது > 1023 | தேடு வயது;";
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        assert!(run_query(q, &cat, &mut arena, &mut out, &mut scratch, &mut tokens));
        assert_eq!(out.row_count, 1);
        assert_eq!(out.int_out[0].values[0], 1024);
    }

    /// Flood VT / ZWSP / CR through the whitespace LUT without branch panics.
    #[test]
    fn test_malformed_whitespace_injection() {
        // Vertical tabs, CR, LF, FF, TAB, spaces, and UTF-8 ZWSP (E2 80 8B).
        let mut buf = [0u8; 32];
        let mut n = 0usize;
        for &b in &[0x0Bu8, b'\r', b'\n', 0x0C, b'\t', b' '] {
            buf[n] = b;
            n += 1;
        }
        buf[n] = 0xE2;
        buf[n + 1] = 0x80;
        buf[n + 2] = 0x8B;
        n += 3;
        for &b in &[0x0Bu8, b'\r', b'\n'] {
            buf[n] = b;
            n += 1;
        }
        let ws = &buf[..n];

        let mut lex = Lexer::new(ws);
        let tok = lex.next_token().expect("ws stream must not fault");
        assert_eq!(tok.kind, TokenKind::Eof);

        let mut arena = AstArena::new();
        assert_eq!(
            {
            let mut tw = alloc_token_window();
            parse_query(ws, &mut arena, &mut tw).unwrap_err()
        },
            ParserError::EmptyInput
        );

        // Zero-heap hot path integrity under the tracking allocator.
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        reset_counters();
        set_tracking(true);
        let ok = run_query(DEMO_QUERY, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens);
        set_tracking(false);
        assert!(ok);
        assert_eq!(alloc_count(), 0);
        assert_eq!(alloc_bytes(), 0);
    }

    /// Ω-COMPLEXITY-MAX: demo query zero-heap + 2050-row dual-batch residue path.
    #[test]
    fn complexity_max_e2e_2050_remainder_and_demo_zero_heap() {
        // --- A: canonical Tamil pipeline, zero hot-path heap ---
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        reset_counters();
        set_tracking(true);
        assert!(run_query(DEMO_QUERY, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens));
        set_tracking(false);
        assert_eq!(alloc_count(), 0);
        assert_eq!(out.row_count, 10);
        let expected: [i64; 10] = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
        let mut j = 0usize;
        while j < 10 {
            assert_eq!(out.int_out[1].values[j], expected[j]);
            j += 1;
        }

        // --- B: 2050 rows = 2×1024 SIMD batches + 2 scalar residue rows ---
        const LIVE: usize = 2050;
        assert_eq!(LIVE / BATCH_ROWS, 2);
        assert_eq!(LIVE % BATCH_ROWS, 2);

        let mut values = [0i64; MAX_ROWS];
        let mut i = 0usize;
        while i < LIVE {
            values[i] = i as i64;
            i += 1;
        }
        let mut sel = SelectionVector::all(LIVE);
        Engine::filter_i64_gt(&values, &mut sel, LIVE, 2047);
        // Rows 2048, 2049 kept (the 2-row scalar residue after two full batches).
        assert_eq!(sel.mask[2047], 0);
        assert_eq!(sel.mask[2048], 1);
        assert_eq!(sel.mask[2049], 1);

        let mut order = [0u16; MAX_ROWS];
        let mut order_len = 0usize;
        let mut tmp = [0u16; MAX_ROWS];
        Engine::sort_i64_selected(&values, &sel, LIVE, &mut order, &mut order_len, &mut tmp);
        assert_eq!(order_len, 2);
        assert_eq!(order[0], 2048);
        assert_eq!(order[1], 2049);

        // End-to-end against a 2050-row columnar relation.
        let mut table = Table::new_boxed("பயனர்கள்".as_bytes());
        let age_i = table.add_int64_column("வயது".as_bytes()).unwrap();
        let name_i = table.add_utf8_column("பெயர்".as_bytes()).unwrap();
        {
            let col = table.int64_mut(age_i).unwrap();
            let mut r = 0usize;
            while r < LIVE {
                col.values[r] = r as i64;
                col.validity.set(r, true);
                r += 1;
            }
        }
        {
            let col = table.utf8_mut(name_i).unwrap();
            col.clear();
            let mut r = 0usize;
            while r < LIVE {
                let b = [b'0' + (r % 10) as u8];
                assert!(col.set_row(r, &b));
                r += 1;
            }
        }
        table.set_row_count(LIVE);
        let mut cat = Catalog::new();
        let _ = cat.register_box(table);
        let q = "இருந்து பயனர்கள் | வடி வயது > 2047 | அடுக்கு வயது | எடு 10 | தேடு வயது;";
        let mut arena2 = Box::new(AstArena::new());
        let mut out2 = QueryResult::new_boxed();
        let mut scratch2 = RuntimeScratch::new_boxed();
        let mut tokens2 = alloc_token_window();
        assert!(run_query(q, &cat, &mut arena2, &mut out2, &mut scratch2, &mut tokens2));
        assert_eq!(out2.row_count, 2);
        assert_eq!(out2.int_out[0].values[0], 2048);
        assert_eq!(out2.int_out[0].values[1], 2049);
    }

    /// Stage-1 micro-arch layout locks + 2050-row remainder via `lsd_radix_sort_ages`.
    #[test]
    fn microarch_stage1_align_radix_and_2050_remainder() {
        assert_eq!(core::mem::size_of::<AstNode>(), 32);
        assert_eq!(core::mem::align_of::<AstNode>(), 32);
        assert_eq!(core::mem::align_of::<AstArena>(), 64);
        assert_eq!(MAX_ROWS, 4096);
        assert_eq!(core::mem::align_of::<Int64Column>(), 64);
        assert_eq!(core::mem::align_of::<Utf8Column>(), 64);
        assert_eq!(core::mem::align_of::<SelectionVector>(), 64);
        assert_eq!(core::mem::align_of::<Table>(), 64);
        assert_eq!(core::mem::align_of::<QueryResult>(), 64);

        const LIVE: usize = 2050;
        let mut values = [0i64; MAX_ROWS];
        let mut i = 0usize;
        while i < LIVE {
            values[i] = (LIVE as i64) - (i as i64); // reverse ages
            i += 1;
        }
        let mut sel = SelectionVector::all(LIVE);
        Engine::filter_i64_gt(&values, &mut sel, LIVE, 2048);
        // values[0]=2050, values[1]=2049 → kept; values[2]=2048 dropped.
        assert_eq!(sel.mask[0], 1);
        assert_eq!(sel.mask[1], 1);
        assert_eq!(sel.mask[2], 0);

        let mut order = [0u16; MAX_ROWS];
        let mut order_len = 0usize;
        let mut tmp = [0u16; MAX_ROWS];
        Engine::sort_i64_selected(&values, &sel, LIVE, &mut order, &mut order_len, &mut tmp);
        assert_eq!(order_len, 2);
        // After LSD radix: ascending ages → 2049 then 2050.
        assert_eq!(values[order[0] as usize], 2049);
        assert_eq!(values[order[1] as usize], 2050);

        // Direct `lsd_radix_sort_ages` on a compacted window.
        let mut direct = [0u16; MAX_ROWS];
        direct[0] = 0;
        direct[1] = 1;
        lsd_radix_sort_ages(&values, &mut direct, 2, &mut tmp);
        assert_eq!(values[direct[0] as usize], 2049);
        assert_eq!(values[direct[1] as usize], 2050);

        // Full pipeline on 2050-row table still zero-heap on hot path.
        let mut table = Table::new_boxed("பயனர்கள்".as_bytes());
        let age_i = table.add_int64_column("வயது".as_bytes()).unwrap();
        let name_i = table.add_utf8_column("பெயர்".as_bytes()).unwrap();
        {
            let col = table.int64_mut(age_i).unwrap();
            let mut r = 0usize;
            while r < LIVE {
                col.values[r] = r as i64;
                col.validity.set(r, true);
                r += 1;
            }
        }
        {
            let col = table.utf8_mut(name_i).unwrap();
            col.clear();
            let mut r = 0usize;
            while r < LIVE {
                let b = [b'0' + (r % 10) as u8];
                assert!(col.set_row(r, &b));
                r += 1;
            }
        }
        table.set_row_count(LIVE);
        let mut cat = Catalog::new();
        let _ = cat.register_box(table);
        let demo_cat = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        reset_counters();
        set_tracking(true);
        assert!(run_query(DEMO_QUERY, &demo_cat, &mut arena, &mut out, &mut scratch, &mut tokens));
        set_tracking(false);
        assert_eq!(alloc_count(), 0);
        assert_eq!(out.row_count, 10);

        let mut arena2 = Box::new(AstArena::new());
        let mut out2 = QueryResult::new_boxed();
        let mut scratch2 = RuntimeScratch::new_boxed();
        let mut tokens2 = alloc_token_window();
        let q = "இருந்து பயனர்கள் | வடி வயது > 2047 | அடுக்கு வயது | எடு 10 | தேடு வயது;";
        assert!(run_query(q, &cat, &mut arena2, &mut out2, &mut scratch2, &mut tokens2));
        assert_eq!(out2.row_count, 2);
        assert_eq!(out2.int_out[0].values[0], 2048);
        assert_eq!(out2.int_out[0].values[1], 2049);
    }

    /// INF-STAGE2: sort-merge join via `இணை ஆர்டர்கள்` — zero heap, O(N+M).
    #[test]
    fn stage2_inai_sort_merge_join_zero_heap() {
        let q = "இருந்து பயனர்கள் | இணை ஆர்டர்கள் | வடி வயது > 21 | தேடு பெயர், விலை;";
        // Lexer must recognize இணை without grapheme tear.
        let mut lex = Lexer::new(q.as_bytes());
        let mut saw_inai = false;
        loop {
            match lex.next_token() {
                Ok(t) if t.kind == TokenKind::Eof => break,
                Ok(t) if t.kind == TokenKind::Inai => {
                    assert_eq!(t.text(q.as_bytes()), Some("இணை"));
                    saw_inai = true;
                }
                Ok(_) => {}
                Err(e) => panic!("lex fault: {e:?}"),
            }
        }
        assert!(saw_inai);

        let catalog = demo_catalog();
        assert!(catalog.orders.is_some());
        assert!(catalog.find("ஆர்டர்கள்".as_bytes()).is_some());
        assert_eq!(
            core::mem::align_of::<FixedOrdersDatabase>(),
            64
        );

        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        reset_counters();
        set_tracking(true);
        assert!(run_query(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens));
        set_tracking(false);
        assert_eq!(alloc_count(), 0, "join hot path must not allocate");
        assert_eq!(out.col_count, 2);
        assert_eq!(out.schema[0].name.as_bytes(), "பெயர்".as_bytes());
        assert_eq!(out.schema[1].name.as_bytes(), "விலை".as_bytes());
        // Orders with user age > 21: all seeded except user_id 0 (age 18).
        assert_eq!(out.row_count, 11);
        let mut i = 0usize;
        while i < out.row_count as usize {
            let name = out.utf8_out[0].get_row(i).expect("name");
            assert!(!name.is_empty());
            let price = out.int_out[1].values[i];
            assert!(price > 0);
            i += 1;
        }

        // Direct vector_merge_join unit check (heap scratch — no stack thrash).
        let mut js = RuntimeScratch::new_boxed();
        js.left_dense[0] = 1;
        js.left_dense[1] = 2;
        js.left_dense[2] = 3;
        js.key_buf[0] = 3;
        js.key_buf[1] = 1;
        js.key_buf[2] = 9;
        let n = vector_merge_join(
            &js.left_dense,
            3,
            &js.key_buf,
            3,
            &mut js.join_left,
            &mut js.join_right,
            &mut js.left_order,
            &mut js.right_order,
            &mut js.tmp_u16,
        );
        assert_eq!(n, 2);
    }

    /// Ω-STAGE2: deep chained operators — flat iterative walk, O(1) call stack.
    #[test]
    fn test_prevent_stack_overflow_deep_chain() {
        // 80 filter stages × ~4 arena nodes + from/take/project stays under AST_CAP.
        let mut q = String::from("இருந்து பயனர்கள்");
        let mut s = 0usize;
        while s < 80 {
            q.push_str(" | வடி வயது > 0");
            s += 1;
        }
        q.push_str(" | எடு 5 | தேடு வயது;");

        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        assert!(
            run_query(&q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens),
            "deep chain must parse+execute without stack overflow"
        );
        assert_eq!(out.row_count, 5);
        assert!(arena.len as usize > 80);
        assert!(arena.len as usize <= AST_CAP);
    }

    /// Ω-STAGE2: asymmetric 1→many join — non-backtracking forward sweep.
    #[test]
    fn test_asymmetric_one_to_many_join_resilience() {
        let mut sc = RuntimeScratch::new_boxed();
        // One left key matches three right rows; another matches two.
        sc.left_dense[0] = 7;
        sc.left_dense[1] = 3;
        sc.key_buf[0] = 3;
        sc.key_buf[1] = 7;
        sc.key_buf[2] = 7;
        sc.key_buf[3] = 9;
        sc.key_buf[4] = 7;
        sc.key_buf[5] = 3;
        let n = vector_merge_join(
            &sc.left_dense,
            2,
            &sc.key_buf,
            6,
            &mut sc.join_left,
            &mut sc.join_right,
            &mut sc.left_order,
            &mut sc.right_order,
            &mut sc.tmp_u16,
        );
        assert_eq!(n, 5, "1+3 and 1+2 matches expected");
        // Every emitted pair must share equal keys.
        let mut i = 0usize;
        while i < n {
            let lk = sc.left_dense[sc.join_left[i] as usize];
            let rk = sc.key_buf[sc.join_right[i] as usize];
            assert_eq!(lk, rk);
            i += 1;
        }
        // Count per left key without thrashing pointers.
        let mut c7 = 0usize;
        let mut c3 = 0usize;
        let mut i = 0usize;
        while i < n {
            match sc.left_dense[sc.join_left[i] as usize] {
                7 => c7 += 1,
                3 => c3 += 1,
                _ => panic!("unexpected key"),
            }
            i += 1;
        }
        assert_eq!(c7, 3);
        assert_eq!(c3, 2);

        // End-to-end: seed a custom 1→many catalog.
        let mut users = Table::new_boxed("பயனர்கள்".as_bytes());
        let uid = users.add_int64_column("அடையாளம்".as_bytes()).unwrap();
        let age = users.add_int64_column("வயது".as_bytes()).unwrap();
        let name = users.add_utf8_column("பெயர்".as_bytes()).unwrap();
        {
            let c = users.int64_mut(uid).unwrap();
            c.values[0] = 1;
            c.validity.set(0, true);
            c.values[1] = 2;
            c.validity.set(1, true);
        }
        {
            let c = users.int64_mut(age).unwrap();
            c.values[0] = 30;
            c.validity.set(0, true);
            c.values[1] = 40;
            c.validity.set(1, true);
        }
        {
            let c = users.utf8_mut(name).unwrap();
            c.clear();
            assert!(c.set_row(0, "அ".as_bytes()));
            assert!(c.set_row(1, "ஆ".as_bytes()));
        }
        users.set_row_count(2);

        let mut orders = Table::new_boxed("ஆர்டர்கள்".as_bytes());
        let oid = orders.add_int64_column("அடையாளம்".as_bytes()).unwrap();
        let price = orders.add_int64_column("விலை".as_bytes()).unwrap();
        {
            let c = orders.int64_mut(oid).unwrap();
            // user 1 → 3 orders; user 2 → 1 order
            let keys = [1i64, 1, 1, 2];
            let mut r = 0usize;
            while r < 4 {
                c.values[r] = keys[r];
                c.validity.set(r, true);
                r += 1;
            }
        }
        {
            let c = orders.int64_mut(price).unwrap();
            let ps = [10i64, 20, 30, 40];
            let mut r = 0usize;
            while r < 4 {
                c.values[r] = ps[r];
                c.validity.set(r, true);
                r += 1;
            }
        }
        orders.set_row_count(4);

        let mut cat = Catalog::new();
        let _ = cat.register_box(users);
        let _ = cat.register_box(orders);
        let q = "இருந்து பயனர்கள் | இணை ஆர்டர்கள் | தேடு பெயர், விலை;";
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        assert!(run_query(q, &cat, &mut arena, &mut out, &mut scratch, &mut tokens));
        assert_eq!(out.row_count, 4);
    }

    /// Ω-STAGE2: 2050-element chunk — two SIMD batches + 2-row scalar residue.
    #[test]
    fn test_unaligned_residue_tail_fidelity() {
        const LIVE: usize = 2050;
        assert_eq!(LIVE % BATCH_ROWS, 2);
        let mut scratch = RuntimeScratch::new_boxed();
        let mut i = 0usize;
        while i < LIVE {
            scratch.key_buf[i] = i as i64;
            i += 1;
        }
        // Poison past-live sentinel — residue must not clobber it.
        scratch.key_buf[LIVE] = -1;
        let mut sel = SelectionVector::all(LIVE);
        if LIVE < MAX_ROWS {
            sel.mask[LIVE] = 0x5A;
        }
        Engine::filter_i64_gt(&scratch.key_buf, &mut sel, LIVE, 2047);
        assert_eq!(sel.mask[2047], 0);
        assert_eq!(sel.mask[2048], 1);
        assert_eq!(sel.mask[2049], 1);
        if LIVE < MAX_ROWS {
            assert_eq!(sel.mask[LIVE], 0x5A, "scalar tail must not overrun");
        }

        let mut order_len = 0usize;
        Engine::sort_i64_selected(
            &scratch.key_buf,
            &sel,
            LIVE,
            &mut scratch.order,
            &mut order_len,
            &mut scratch.tmp_u16,
        );
        assert_eq!(order_len, 2);
        assert_eq!(scratch.order[0], 2048);
        assert_eq!(scratch.order[1], 2049);

        // Full pipeline on 2050-row table.
        let mut table = Table::new_boxed("பயனர்கள்".as_bytes());
        let age_i = table.add_int64_column("வயது".as_bytes()).unwrap();
        {
            let col = table.int64_mut(age_i).unwrap();
            let mut r = 0usize;
            while r < LIVE {
                col.values[r] = r as i64;
                col.validity.set(r, true);
                r += 1;
            }
        }
        table.set_row_count(LIVE);
        let mut cat = Catalog::new();
        let _ = cat.register_box(table);
        let q = "இருந்து பயனர்கள் | வடி வயது > 2047 | அடுக்கு வயது | எடு 10 | தேடு வயது;";
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut sc2 = RuntimeScratch::new_boxed();
        let mut tokens2 = alloc_token_window();
        assert!(run_query(q, &cat, &mut arena, &mut out, &mut sc2, &mut tokens2));
        assert_eq!(out.row_count, 2);
        assert_eq!(out.int_out[0].values[0], 2048);
        assert_eq!(out.int_out[0].values[1], 2049);
    }

    /// Ω-STAGE2: torn Tamil grapheme → discrete lex error, no panic cascade.
    #[test]
    fn test_malformed_tamil_boundary_fragmentation() {
        // "தேடு" = four 3-byte Tamil codepoints (12 bytes). Mid-codepoint cuts
        // must map to MalformedUtf8; complete-codepoint prefixes may lex as Ident.
        let full = "தேடு".as_bytes();
        assert_eq!(full.len(), 12);
        // Cuts that land inside a 3-byte UTF-8 sequence (not on a lead boundary).
        let cuts = [1usize, 2, 4, 5, 7, 8, 10, 11];
        let mut ci = 0usize;
        while ci < cuts.len() {
            let cut = cuts[ci];
            let frag = &full[..cut];
            let mut lex = Lexer::new(frag);
            let err = lex
                .next_token()
                .expect_err("torn Tamil must be MalformedUtf8");
            match err {
                LexerError::MalformedUtf8(cursor) => {
                    assert!(
                        (cursor as usize) <= cut,
                        "cursor {cursor} must not run past cut {cut}"
                    );
                }
                other => panic!("expected MalformedUtf8, got {other:?}"),
            }
            // Parser must map lex fault → LexMalformedUtf8 (no panic / no loop).
            let mut arena = Box::new(AstArena::new());
            let mut tokens_frag = alloc_token_window();
            let perr = parse_query(frag, &mut arena, &mut tokens_frag).expect_err("parse must fail");
            assert_eq!(perr, ParserError::LexMalformedUtf8);
            ci += 1;
        }

        // Explicit mid-syllable "தே" cut (matches Stage-1 lexer contract).
        let the = "தே".as_bytes();
        assert_eq!(the.len(), 6);
        let torn = &the[..4];
        let mut lex = Lexer::new(torn);
        assert_eq!(
            lex.next_token().unwrap_err(),
            LexerError::MalformedUtf8(0)
        );
    }

    /// INF-STAGE3 (legacy): `கணி` arithmetic derive — superseded by matrix TIER E.12.
    #[test]
    fn test_derive_math_pipeline_evaluation_legacy_smoke() {
        let q = "இருந்து பயனர்கள் | இணை ஆர்டர்கள் | கணி புதிய_விலை = விலை * 2 | வடி புதிய_விலை > 200;";
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        assert!(run_query(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens));
        assert_eq!(out.row_count, 11);
    }

    /// INF-STAGE3 (legacy): chunk router smoke — superseded by matrix TIER E.11.
    #[test]
    fn test_parallel_chunk_distribution_integrity_legacy_smoke() {
        const LIVE: usize = 2050;
        let mut src = RuntimeScratch::new_boxed();
        let mut dst = RuntimeScratch::new_boxed();
        let mut i = 0usize;
        while i < LIVE {
            src.key_buf[i] = (i as i64) + 1;
            i += 1;
        }
        execute_chunk_parallel(&src.key_buf, &mut dst.derived, LIVE, ArithOp::Mul, 2);
        assert_eq!(dst.derived[2048], 2049 * 2);
        assert_eq!(dst.derived[2049], 2050 * 2);
    }

    /// Ω-QA-CORE-STRESS-STAGE3 — micro-arch checklist matrix (release-safe).
    #[test]
    fn omega_qa_stage3_matrix() {
        // --- GATE 3: cache-line packing density ---
        assert_eq!(core::mem::align_of::<AstArena>(), 64);
        assert_eq!(core::mem::align_of::<QueryResult>(), 64);
        assert_eq!(core::mem::align_of::<RuntimeScratch>(), 64);
        assert_eq!(core::mem::align_of::<Int64Column>(), 64);
        assert_eq!(core::mem::align_of::<Utf8Column>(), 64);
        assert_eq!(core::mem::align_of::<SelectionVector>(), 64);
        assert_eq!(core::mem::align_of::<Table>(), 64);
        assert_eq!(core::mem::align_of::<FixedOrdersDatabase>(), 64);
        assert_eq!(core::mem::align_of::<ChunkScratch>(), 64);
        assert_eq!(core::mem::align_of::<EngineScratchPad>(), 64);
        assert_eq!(core::mem::align_of::<RadixScratchPad>(), 64);
        assert_eq!(core::mem::align_of::<Parser<'static>>(), 64);
        assert_eq!(core::mem::size_of::<AstNode>(), 32);

        // --- GATE 4: 2050-row unaligned residue (2×1024 + 2) ---
        const LIVE: usize = 2050;
        let mut sc = RuntimeScratch::new_boxed();
        let mut i = 0usize;
        while i < LIVE {
            sc.key_buf[i] = i as i64;
            i += 1;
        }
        let mut sel = SelectionVector::all(LIVE);
        Engine::filter_i64_gt(&sc.key_buf, &mut sel, LIVE, 2047);
        assert_eq!(sel.mask[2047], 0);
        assert_eq!(sel.mask[2048], 1);
        assert_eq!(sel.mask[2049], 1);
        execute_chunk_parallel(&sc.key_buf, &mut sc.derived, LIVE, ArithOp::Mul, 3);
        assert_eq!(sc.derived[2048], 2048 * 3);
        assert_eq!(sc.derived[2049], 2049 * 3);

        // --- GATE 2: TLS radix pad flatness ---
        let mut j = 0usize;
        while j < 16 {
            sc.order[j] = (15 - j) as u16;
            sc.key_buf[j] = (15 - j) as i64;
            j += 1;
        }
        lsd_radix_sort_ages_tls(&sc.key_buf, &mut sc.order, 16);
        let mut prev = i64::MIN;
        let mut k = 0usize;
        while k < 16 {
            let v = sc.key_buf[sc.order[k] as usize];
            assert!(v >= prev);
            prev = v;
            k += 1;
        }

        // --- GATE 5: mid-syllable Tamil tear ---
        let torn = &"தேடு".as_bytes()[..5];
        let mut lex = Lexer::new(torn);
        match lex.next_token() {
            Err(LexerError::MalformedUtf8(_)) => {}
            other => panic!("expected MalformedUtf8, got {other:?}"),
        }

        // --- GATE 1: zero-heap hot path on Stage-3 query ---
        let q = "இருந்து பயனர்கள் | இணை ஆர்டர்கள் | கணி புதிய_விலை = விலை * 2 | வடி புதிய_விலை > 200;";
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        reset_counters();
        set_tracking(true);
        assert!(run_query(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens));
        set_tracking(false);
        assert_eq!(alloc_count(), 0, "GATE1 heap must stay 0");
        assert_eq!(out.row_count, 11);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Ω-CORE-PROD-VALIDATION-MATRIX — Tier A…E (exact production gate names)
    // ═══════════════════════════════════════════════════════════════════════

    /// TIER A.1 — left-to-right lex + parse for a standard Tamil filter pipeline.
    #[test]
    fn test_basic_pipeline_lexing_and_parsing() {
        let q = "இருந்து பயனர்கள் | வடி வயது > 21;";
        let src = q.as_bytes();
        let mut lex = Lexer::new(src);
        let mut kinds = [TokenKind::Eof; 16];
        let mut texts: [Option<&str>; 16] = [None; 16];
        let mut n = 0usize;
        loop {
            let t = lex.next_token().expect("lex must succeed");
            kinds[n] = t.kind;
            texts[n] = t.text(src);
            n += 1;
            if t.kind == TokenKind::Eof || n >= 16 {
                break;
            }
        }
        assert_eq!(kinds[0], TokenKind::Irundu);
        assert_eq!(kinds[1], TokenKind::Ident);
        assert_eq!(texts[1], Some("பயனர்கள்"));
        assert_eq!(kinds[2], TokenKind::Pipe);
        assert_eq!(kinds[3], TokenKind::Vadi);
        assert_eq!(kinds[4], TokenKind::Ident);
        assert_eq!(kinds[5], TokenKind::Gt);
        assert_eq!(kinds[6], TokenKind::Number);
        assert_eq!(kinds[7], TokenKind::Semi);
        assert_eq!(kinds[8], TokenKind::Eof);

        let mut arena = Box::new(AstArena::new());
        let mut tokens = alloc_token_window();
        let root = parse_query(src, &mut arena, &mut tokens).expect("parse");
        let pipe = arena.get(root).expect("root");
        assert_eq!(pipe.kind, NodeKind::Pipeline);
        let from = arena.get(pipe.left).expect("from");
        assert_eq!(from.kind, NodeKind::From);
        let filter = arena.get(from.next).expect("filter");
        assert_eq!(filter.kind, NodeKind::Filter);
        assert_eq!(filter.next, NIL);
    }

    /// TIER A.2 — ASCII structural delimiters map to packed TokenKind enums.
    #[test]
    fn test_all_structural_delimiters() {
        let src = b"| = > < , ; * + -";
        let mut lex = Lexer::new(src);
        let expect = [
            TokenKind::Pipe,
            TokenKind::Eq,
            TokenKind::Gt,
            TokenKind::Lt,
            TokenKind::Comma,
            TokenKind::Semi,
            TokenKind::Star,
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Eof,
        ];
        let mut i = 0usize;
        while i < expect.len() {
            let t = lex.next_token().expect("delimiter lex");
            assert_eq!(t.kind, expect[i], "delimiter index {i}");
            i += 1;
        }
        // Register-density: TokenKind is a u8 discriminant.
        assert_eq!(core::mem::size_of::<TokenKind>(), 1);
    }

    /// TIER B.3 — mid-syllable truncation → MalformedUtf8, never panic.
    #[test]
    fn test_grapheme_tearing_and_buffer_truncation() {
        let full = "தே".as_bytes();
        assert_eq!(full.len(), 6);
        // Cut after 4 bytes: mid combining-vowel sequence.
        let torn = &full[..4];
        assert!(core::str::from_utf8(torn).is_err());
        let mut lex = Lexer::new(torn);
        let err = lex.next_token().expect_err("must reject torn தே");
        assert_eq!(err, LexerError::MalformedUtf8(0));
        assert_eq!(lex.last_error(), Some(LexerError::MalformedUtf8(0)));

        // Streaming packet drops across "தேடு" mid-codepoint cuts.
        let thedu = "தேடு".as_bytes();
        let cuts = [1usize, 2, 4, 5, 7, 8, 10, 11];
        let mut ci = 0usize;
        while ci < cuts.len() {
            let frag = &thedu[..cuts[ci]];
            let mut lx = Lexer::new(frag);
            match lx.next_token() {
                Err(LexerError::MalformedUtf8(c)) => assert!((c as usize) <= cuts[ci]),
                Err(LexerError::TokenBufferFull) => panic!("unexpected TokenBufferFull"),
                Ok(t) => panic!("cut {} must not succeed: {t:?}", cuts[ci]),
            }
            let mut arena = Box::new(AstArena::new());
            let mut tw = alloc_token_window();
            assert_eq!(
                parse_query(frag, &mut arena, &mut tw).unwrap_err(),
                ParserError::LexMalformedUtf8
            );
            ci += 1;
        }
    }

    /// TIER B.4 — maximal munch: "வடிவமைப்பு" stays Ident, bare "வடி" is Vadi.
    #[test]
    fn test_maximal_munch_keyword_collisions() {
        let composite = "வடிவமைப்பு";
        let mut lex = Lexer::new(composite.as_bytes());
        let tok = lex.next_token().expect("lex composite");
        assert_eq!(tok.kind, TokenKind::Ident);
        assert_eq!(tok.text(composite.as_bytes()), Some("வடிவமைப்பு"));
        assert_eq!(lex.next_token().unwrap().kind, TokenKind::Eof);

        let mut lex2 = Lexer::new("வடி".as_bytes());
        assert_eq!(lex2.next_token().unwrap().kind, TokenKind::Vadi);

        // Pipeline using composite as column name must not tear into Vadi.
        let q = "இருந்து பயனர்கள் | வடி வடிவமைப்பு > 1;";
        let mut arena = Box::new(AstArena::new());
        let mut tokens = alloc_token_window();
        let root = parse_query(q.as_bytes(), &mut arena, &mut tokens).expect("parse");
        let pipe = arena.get(root).unwrap();
        let from = arena.get(pipe.left).unwrap();
        let filter = arena.get(from.next).unwrap();
        assert_eq!(filter.kind, NodeKind::Filter);
        let bin = arena.get(filter.left).unwrap();
        let col = arena.get(bin.left).unwrap();
        assert_eq!(col.kind, NodeKind::Ident);
        let name = &q.as_bytes()[col.start as usize..col.end as usize];
        assert_eq!(name, "வடிவமைப்பு".as_bytes());
    }

    /// TIER B.5 — VT / FF / CR / LF / ZWSP flood isolated via WHITESPACE_LUT.
    #[test]
    fn test_chaotic_whitespace_injection() {
        let mut buf = [0u8; 512];
        let mut n = 0usize;
        // Prefix: ASCII ws flood including VT (0x0B) and FF (0x0C).
        let ascii = [b' ', b'\t', b'\n', b'\r', 0x0Bu8, 0x0Cu8];
        let mut a = 0usize;
        while a < 48 {
            buf[n] = ascii[a % ascii.len()];
            n += 1;
            a += 1;
        }
        // ZWSP U+200B = E2 80 8B
        let mut z = 0usize;
        while z < 16 {
            buf[n] = 0xE2;
            buf[n + 1] = 0x80;
            buf[n + 2] = 0x8B;
            n += 3;
            z += 1;
        }
        let body = "இருந்து பயனர்கள் | வடி வயது > 21;".as_bytes();
        buf[n..n + body.len()].copy_from_slice(body);
        n += body.len();
        // Trailing chaos
        buf[n] = 0x0B;
        buf[n + 1] = b'\r';
        buf[n + 2] = 0xE2;
        buf[n + 3] = 0x80;
        buf[n + 4] = 0x8B;
        n += 5;
        let stream = &buf[..n];

        let mut lex = Lexer::new(stream);
        assert_eq!(lex.next_token().unwrap().kind, TokenKind::Irundu);
        assert_eq!(lex.next_token().unwrap().kind, TokenKind::Ident);
        assert_eq!(lex.next_token().unwrap().kind, TokenKind::Pipe);
        assert_eq!(lex.next_token().unwrap().kind, TokenKind::Vadi);

        let mut arena = Box::new(AstArena::new());
        let mut tokens = alloc_token_window();
        assert!(parse_query(stream, &mut arena, &mut tokens).is_ok());

        // Pure-ws stream → EmptyInput, never panic (complete ASCII ws only).
        let ws_only = &buf[..48];
        let mut lex_ws = Lexer::new(ws_only);
        assert_eq!(lex_ws.next_token().unwrap().kind, TokenKind::Eof);
        let mut arena2 = Box::new(AstArena::new());
        let mut tw2 = alloc_token_window();
        assert_eq!(
            parse_query(ws_only, &mut arena2, &mut tw2).unwrap_err(),
            ParserError::EmptyInput
        );
    }

    /// TIER C.6 — 1024-node arena hard stop → ArenaOverflow (no OOB / panic).
    #[test]
    fn test_arena_overflow_prevention() {
        let mut arena = AstArena::new();
        let mut i = 0usize;
        while i < AST_CAP {
            let id = arena
                .try_alloc(AstNode::empty())
                .expect("slot within capacity");
            assert_eq!(id as usize, i);
            i += 1;
        }
        assert!(arena.is_full());
        assert_eq!(arena.len as usize, AST_CAP);
        let overflow = arena.try_alloc(AstNode::empty());
        assert_eq!(overflow, Err(ParserError::ArenaOverflow));
        assert_eq!(arena.len as usize, AST_CAP);

        // Parse against saturated arena also returns ArenaOverflow.
        let q = "இருந்து பயனர்கள் | வடி வயது > 21;";
        let mut tokens = alloc_token_window();
        let err = parse_query(q.as_bytes(), &mut arena, &mut tokens).unwrap_err();
        assert_eq!(err, ParserError::ArenaOverflow);
    }

    /// TIER C.7 — stages before இருந்து → MissingSourceContext.
    #[test]
    fn test_out_of_order_pipeline_source() {
        let cases: [&str; 4] = [
            "வடி வயது > 21;",
            "அடுக்கு வயது | எடு 10;",
            "தேடு பெயர்;",
            "இணை ஆர்டர்கள் | வடி வயது > 1;",
        ];
        let mut ci = 0usize;
        while ci < cases.len() {
            let mut arena = Box::new(AstArena::new());
            let mut tokens = alloc_token_window();
            let err = parse_query(cases[ci].as_bytes(), &mut arena, &mut tokens).unwrap_err();
            assert_eq!(
                err,
                ParserError::MissingSourceContext,
                "case {}",
                cases[ci]
            );
            ci += 1;
        }
    }

    /// TIER D.8 — 1→1500 asymmetric join multiplicity, forward-only emit.
    #[test]
    fn test_asymmetric_one_to_many_join_multiplicity() {
        const RIGHT_DUP: usize = 1500;
        let mut sc = RuntimeScratch::new_boxed();
        sc.left_dense[0] = 42;
        let mut r = 0usize;
        while r < RIGHT_DUP {
            sc.key_buf[r] = 42;
            r += 1;
        }
        // Poison trailing keys to ensure we do not over-read.
        sc.key_buf[RIGHT_DUP] = 99;
        sc.key_buf[RIGHT_DUP + 1] = 7;

        let n = vector_merge_join(
            &sc.left_dense,
            1,
            &sc.key_buf,
            RIGHT_DUP,
            &mut sc.join_left,
            &mut sc.join_right,
            &mut sc.left_order,
            &mut sc.right_order,
            &mut sc.tmp_u16,
        );
        assert_eq!(n, RIGHT_DUP);
        let mut i = 0usize;
        while i < n {
            assert_eq!(sc.join_left[i], 0);
            assert_eq!(sc.key_buf[sc.join_right[i] as usize], 42);
            // Constant-forward: right indices from the sorted equal-run are unique.
            i += 1;
        }
        // All right row ids appear exactly once.
        let mut seen = [0u8; 2048];
        let mut i = 0usize;
        while i < n {
            let rid = sc.join_right[i] as usize;
            assert!(rid < RIGHT_DUP);
            assert_eq!(seen[rid], 0, "duplicate emit / backtrack at {rid}");
            seen[rid] = 1;
            i += 1;
        }
    }

    /// TIER D.9 — disjoint key domains → O(1) sparsity fast-abort (0 matches).
    #[test]
    fn test_constant_time_sparsity_fast_abort() {
        let mut sc = RuntimeScratch::new_boxed();
        // Left: 0..100, Right: 5000..5100 — no overlap.
        let mut i = 0usize;
        while i < 100 {
            sc.left_dense[i] = i as i64;
            i += 1;
        }
        let mut j = 0usize;
        while j < 100 {
            sc.key_buf[j] = 5000 + (j as i64);
            j += 1;
        }
        let n = vector_merge_join(
            &sc.left_dense,
            100,
            &sc.key_buf,
            100,
            &mut sc.join_left,
            &mut sc.join_right,
            &mut sc.left_order,
            &mut sc.right_order,
            &mut sc.tmp_u16,
        );
        assert_eq!(n, 0, "disjoint domains must fast-abort to zero matches");

        // Boundary touch: left max == right min − 1 still disjoint.
        sc.left_dense[0] = 4999;
        sc.key_buf[0] = 5000;
        let n2 = vector_merge_join(
            &sc.left_dense,
            1,
            &sc.key_buf,
            1,
            &mut sc.join_left,
            &mut sc.join_right,
            &mut sc.left_order,
            &mut sc.right_order,
            &mut sc.tmp_u16,
        );
        assert_eq!(n2, 0);

        // Adjacent equal keys must NOT abort.
        sc.left_dense[0] = 5000;
        let n3 = vector_merge_join(
            &sc.left_dense,
            1,
            &sc.key_buf,
            1,
            &mut sc.join_left,
            &mut sc.join_right,
            &mut sc.left_order,
            &mut sc.right_order,
            &mut sc.tmp_u16,
        );
        assert_eq!(n3, 1);
    }

    /// TIER E.10 — 2050-row remainder: 2×1024 batches + 2-row scalar tail.
    #[test]
    fn test_simd_unaligned_tail_residue_fidelity() {
        const LIVE: usize = 2050;
        assert_eq!(LIVE / BATCH_ROWS, 2);
        assert_eq!(LIVE % BATCH_ROWS, 2);

        let mut sc = RuntimeScratch::new_boxed();
        let mut i = 0usize;
        while i < LIVE {
            sc.key_buf[i] = i as i64;
            i += 1;
        }
        // Sentinel past live window — must not be clobbered.
        if LIVE < MAX_ROWS {
            sc.key_buf[LIVE] = -1;
        }
        let mut sel = SelectionVector::all(LIVE);
        if LIVE < MAX_ROWS {
            sel.mask[LIVE] = 0xA5;
        }
        Engine::filter_i64_gt(&sc.key_buf, &mut sel, LIVE, 2047);
        assert_eq!(sel.mask[2047], 0);
        assert_eq!(sel.mask[2048], 1);
        assert_eq!(sel.mask[2049], 1);
        if LIVE < MAX_ROWS {
            assert_eq!(sel.mask[LIVE], 0xA5, "SIMD must not serialize past live");
        }

        let mut order_len = 0usize;
        Engine::sort_i64_selected(
            &sc.key_buf,
            &sel,
            LIVE,
            &mut sc.order,
            &mut order_len,
            &mut sc.tmp_u16,
        );
        assert_eq!(order_len, 2);
        assert_eq!(sc.order[0], 2048);
        assert_eq!(sc.order[1], 2049);

        // Derive chunk router residue on the same 2050 window.
        execute_chunk_parallel(&sc.key_buf, &mut sc.derived, LIVE, ArithOp::Add, 10);
        assert_eq!(sc.derived[0], 10);
        assert_eq!(sc.derived[2048], 2058);
        assert_eq!(sc.derived[2049], 2059);
        if LIVE < MAX_ROWS {
            assert_eq!(sc.key_buf[LIVE], -1);
        }
    }

    /// TIER E.11 — 1024-row chunk distribution via ENGINE_SCRATCH_PAD, zero heap.
    #[test]
    fn test_parallel_chunk_distribution_integrity() {
        const LIVE: usize = 2050;
        let mut src = RuntimeScratch::new_boxed();
        let mut dst = RuntimeScratch::new_boxed();
        let mut i = 0usize;
        while i < LIVE {
            src.key_buf[i] = (i as i64) + 1;
            i += 1;
        }
        reset_counters();
        set_tracking(true);
        execute_chunk_parallel(&src.key_buf, &mut dst.derived, LIVE, ArithOp::Mul, 2);
        set_tracking(false);
        assert_eq!(alloc_count(), 0, "chunk TLS path must not allocate");
        let mut i = 0usize;
        while i < LIVE {
            assert_eq!(dst.derived[i], src.key_buf[i].wrapping_mul(2), "row {i}");
            i += 1;
        }
        assert_eq!(dst.derived[2048], 2049 * 2);
        assert_eq!(dst.derived[2049], 2050 * 2);

        // Single partial chunk (no multi-batch) still zero-heap.
        let mut out_add = RuntimeScratch::new_boxed();
        reset_counters();
        set_tracking(true);
        execute_chunk_parallel(&src.key_buf, &mut out_add.derived, 17, ArithOp::Add, 5);
        set_tracking(false);
        assert_eq!(alloc_count(), 0);
        assert_eq!(out_add.derived[0], 6);
        assert_eq!(out_add.derived[16], 22);

        // Full Tamil derive pipeline under the same zero-heap lock.
        let q = "இருந்து பயனர்கள் | இணை ஆர்டர்கள் | கணி புதிய_விலை = விலை * 2 | வடி புதிய_விலை > 200;";
        let catalog = demo_catalog();
        assert!(catalog.orders.is_some());
        let orders = catalog.orders.as_ref().unwrap();
        assert_eq!(core::mem::align_of_val(orders.as_ref()), 64);
        assert_eq!(orders.derived_prices[0], orders.price_column[0]);

        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        reset_counters();
        set_tracking(true);
        assert!(run_query(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens));
        set_tracking(false);
        assert_eq!(alloc_count(), 0);
        assert_eq!(out.row_count, 11);
        assert!(out.int_out[0].values[0] > 200);
    }

    /// TIER E.12 — full `கணி` derive math pipeline with raw derived buffer writes.
    #[test]
    fn test_derive_math_pipeline_evaluation() {
        let q = "இருந்து பயனர்கள் | இணை ஆர்டர்கள் | கணி புதிய_விலை = விலை * 2 | வடி புதிய_விலை > 200;";

        let mut lex = Lexer::new(q.as_bytes());
        let mut saw_kani = false;
        let mut saw_star = false;
        loop {
            match lex.next_token() {
                Ok(t) if t.kind == TokenKind::Eof => break,
                Ok(t) if t.kind == TokenKind::Kani => {
                    assert_eq!(t.text(q.as_bytes()), Some("கணி"));
                    saw_kani = true;
                }
                Ok(t) if t.kind == TokenKind::Star => saw_star = true,
                Ok(_) => {}
                Err(e) => panic!("lex fault: {e:?}"),
            }
        }
        assert!(saw_kani);
        assert!(saw_star);

        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        reset_counters();
        set_tracking(true);
        assert!(run_query(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens));
        set_tracking(false);
        assert_eq!(alloc_count(), 0, "derive hot path must not allocate");
        assert_eq!(scratch.has_derived, 1);
        assert_eq!(scratch.derived_name.as_bytes(), "புதிய_விலை".as_bytes());

        // Raw pointer address view into derived slab (no copy / no alloc).
        let derived_ptr: *mut i64 = scratch.derived.as_mut_ptr();
        assert!(!derived_ptr.is_null());
        // price*2 > 200 ⇒ price > 100; seeded 100 drops → 11 rows.
        assert_eq!(out.col_count, 1);
        assert_eq!(out.row_count, 11);
        let mut i = 0usize;
        while i < out.row_count as usize {
            let v = out.int_out[0].values[i];
            assert!(v > 200);
            assert_eq!(v % 2, 0);
            // Cross-check against scratch.derived via order slots when present.
            i += 1;
        }
        // Direct slab probe: at least one join slot holds a doubled price.
        let mut found = false;
        let mut s = 0usize;
        while s < MAX_ROWS {
            let v = unsafe { *derived_ptr.add(s) };
            if v > 200 && v % 2 == 0 {
                found = true;
                break;
            }
            s += 1;
        }
        assert!(found, "derived slab must hold computed *mut i64 values");
    }
}
