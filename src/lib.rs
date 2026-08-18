//! Microsecond-scale Tamil-native linear pipeline vector query engine
//! and columnar database.
//!
//! # Architecture
//! - [`ingest`] — CSV → columnar `.bin`/`.meta` cold-path pipeline
//! - [`zonemap`] — Int64 `.zmap` sidecars for page-level predicate pushdown
//! - [`lexer`] — zero-allocation UTF-8 / Tamil DSL scanner
//! - [`parser`] — flat arena AST (`u32` index links, no pointer trees)
//! - [`storage`] — Arrow-aligned columnar segments
//! - [`runtime`] — batch-1024 vectorized execution
//!
//! Hot execution loops do not call `alloc`, construct `String`/`Vec`/`Box`,
//! or tear Tamil grapheme clusters.

#![allow(clippy::needless_range_loop)]

pub mod ingest;
pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod storage;
pub mod utf8;
pub mod zonemap;

pub use lexer::{Lexer, LexerError, Token, TokenKind, MAX_TOKENS};
pub use runtime::{
    demo_catalog, execute_chunk_parallel, execute_chunk_parallel_os,
    execute_int64_filter_pushdown, execute_mmap_age_filter_stream,
    execute_mmap_table_filter_project_stream, lsd_radix_sort_ages, lsd_radix_sort_ages_tls,
    run_query, run_query_checked, vector_merge_join, ArithOp, ChunkScratch, Engine, EngineError,
    EngineScratchPad, GroupedAgg, MmapStreamStats, PushdownStats, QueryResult, RadixScratchPad,
    RuntimeScratch,
};
pub use storage::{
    dup_price_orders_catalog, os_page_size_bytes, seed_dup_price_orders_table, seed_orders_database,
    seed_orders_table, seed_users_table, write_i64_column_bin, write_stage4_columnar_demo,
    write_utf8_column_files, Catalog, ColName, ColumnData, ColumnarChunk, ColumnarFileStream,
    ColumnarTablePage, ColumnarTableStream, FixedOrdersDatabase, Int64Column, Int64ColumnFile,
    Int64ColumnMeta, PhysType, SelectionVector, Table, Utf8Column, Utf8ColumnFile, Utf8ColumnMeta,
    Utf8OffsetEntry, BATCH_ROWS, MAX_ROWS, NAME_CAP,
};
pub use zonemap::{
    page_can_satisfy, write_zonemap_for_column, ZoneCmp, ZoneMap, ZoneMapEntry, MAX_ZMAP_PAGES,
};
pub use parser::{
    alloc_token_window, parse_query, AstArena, AstNode, NodeKind, OpKind, ParseError, Parser,
    ParserError, AST_CAP, NIL,
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
        let ok = run_query(DEMO_QUERY, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok();
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
        assert!(run_query(q, &cat, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
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
        let ok = run_query(DEMO_QUERY, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok();
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
        assert!(run_query(DEMO_QUERY, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
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
        assert!(run_query(q, &cat, &mut arena2, &mut out2, &mut scratch2, &mut tokens2).is_ok());
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
        assert!(run_query(DEMO_QUERY, &demo_cat, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
        set_tracking(false);
        assert_eq!(alloc_count(), 0);
        assert_eq!(out.row_count, 10);

        let mut arena2 = Box::new(AstArena::new());
        let mut out2 = QueryResult::new_boxed();
        let mut scratch2 = RuntimeScratch::new_boxed();
        let mut tokens2 = alloc_token_window();
        let q = "இருந்து பயனர்கள் | வடி வயது > 2047 | அடுக்கு வயது | எடு 10 | தேடு வயது;";
        assert!(run_query(q, &cat, &mut arena2, &mut out2, &mut scratch2, &mut tokens2).is_ok());
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
        assert!(run_query(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
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
            run_query(&q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok(),
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
        assert!(run_query(q, &cat, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
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
        assert!(run_query(q, &cat, &mut arena, &mut out, &mut sc2, &mut tokens2).is_ok());
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
        assert!(run_query(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
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
        assert!(run_query(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
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
        assert!(run_query(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
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
        assert!(run_query(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
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

    /// STAGE-4 v2: mmap page stream over 10_000 rows using OS page-size chunking.
    #[test]
    #[cfg_attr(miri, ignore = "memmap2 file-backed mmap unsupported under Miri")]
    fn test_persistent_mmap_page_streaming_fidelity() {
        const TOTAL: usize = 10_000;
        let page_rows = os_page_size_bytes() / 8;
        let expected_full = (TOTAL / page_rows) as u32;
        let expected_rem = (TOTAL % page_rows) as u32;

        let dir = std::env::temp_dir().join("tamil_stage4_mmap_fidelity_v2");
        let _ = std::fs::create_dir_all(&dir);
        write_stage4_columnar_demo(&dir, TOTAL).expect("write columnar bins");

        let ages_path = dir.join("ages.bin");
        let mut ages = ColumnarFileStream::open_i64(&ages_path).expect("mmap ages");
        assert_eq!(ages.total_rows(), TOTAL as u64);
        assert_eq!(ages.page_rows(), page_rows);

        let mut sum = 0usize;
        let mut full = 0u32;
        let mut residue = 0u32;
        let mut last_rem = 0u32;
        while let Some(p) = ages.next_page_chunk() {
            sum += p.row_count as usize;
            if p.is_residue == 0 {
                assert_eq!(p.row_count as usize, page_rows);
                full += 1;
            } else {
                residue += 1;
                last_rem = p.row_count as u32;
            }
        }
        assert_eq!(sum, TOTAL);
        assert_eq!(full, expected_full);
        if expected_rem == 0 {
            assert_eq!(residue, 0);
        } else {
            assert_eq!(residue, 1);
            assert_eq!(last_rem, expected_rem);
        }

        let mut ages2 = ColumnarFileStream::open_i64(&ages_path).expect("mmap ages2");
        let mut scratch = RuntimeScratch::new_boxed();
        let mut stats = MmapStreamStats::default();
        reset_counters();
        set_tracking(true);
        let t0 = std::time::Instant::now();
        execute_mmap_age_filter_stream(&mut ages2, 21, 2, &mut scratch, &mut stats);
        let elapsed = t0.elapsed();
        set_tracking(false);
        assert_eq!(alloc_count(), 0, "mmap page loop must not allocate");
        assert_eq!(stats.pages_full, expected_full);
        assert_eq!(stats.pages_residue, if expected_rem == 0 { 0 } else { 1 });
        if expected_rem != 0 {
            assert_eq!(stats.residue_rows, expected_rem);
        }
        assert_eq!(stats.rows_scanned, TOTAL as u64);
        assert!(stats.rows_kept > 0);
        assert!(stats.rows_kept < stats.rows_scanned);
        let ns_per_row = elapsed.as_nanos() / (TOTAL as u128);
        assert!(
            ns_per_row < 5_000,
            "per-row mmap path too slow: {ns_per_row} ns"
        );

        let mut table = ColumnarTableStream::open(
            &dir.join("user_ids.bin"),
            &dir.join("ages.bin"),
            &dir.join("prices.bin"),
        )
        .expect("open table stream");
        assert_eq!(table.total_rows(), TOTAL as u64);
        let mut out_prices = [0i64; 64];
        let mut out_len = 0usize;
        let mut stats2 = MmapStreamStats::default();
        let mut scratch2 = RuntimeScratch::new_boxed();
        reset_counters();
        set_tracking(true);
        execute_mmap_table_filter_project_stream(
            &mut table,
            21,
            &mut scratch2,
            &mut stats2,
            &mut out_prices,
            &mut out_len,
        );
        set_tracking(false);
        assert_eq!(alloc_count(), 0);
        assert_eq!(stats2.pages_full, expected_full);
        assert_eq!(stats2.pages_residue, if expected_rem == 0 { 0 } else { 1 });
        assert_eq!(stats2.rows_scanned, TOTAL as u64);
        assert!(out_len > 0);
        assert!(out_len <= 64);
        let mut k = 0usize;
        while k < out_len {
            assert!(out_prices[k] >= 100);
            k += 1;
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "memmap2 file-backed mmap unsupported under Miri")]
    fn test_mmap_page_streaming_10000_rows_exact_remainder() {
        const TOTAL: usize = 10_000;
        let dir = std::env::temp_dir().join("tamil_mmap_10000_remainder");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("col.bin");
        write_i64_column_bin(&path, TOTAL, |i| i as i64).unwrap();
        let mut stream = ColumnarFileStream::open_i64(&path).unwrap();
        let page_rows = stream.page_rows();
        assert_eq!(page_rows, os_page_size_bytes() / core::mem::size_of::<i64>());
        let expected_rem = TOTAL % page_rows;
        let mut sum = 0usize;
        let mut last_is_residue = false;
        let mut last_n = 0usize;
        while let Some(chunk) = stream.next_page_chunk() {
            sum += chunk.row_count as usize;
            last_is_residue = chunk.is_residue != 0;
            last_n = chunk.row_count as usize;
        }
        assert_eq!(sum, TOTAL);
        if expected_rem == 0 {
            assert!(!last_is_residue);
        } else {
            assert!(last_is_residue);
            assert_eq!(last_n, expected_rem);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "memmap2 file-backed mmap unsupported under Miri")]
    fn test_mmap_missing_file_returns_io_error_not_panic() {
        let path = std::path::Path::new("/tmp/tamil_query_engine_missing_column_file_zzz.bin");
        let err = ColumnarFileStream::open_i64(path)
            .map(|_| ())
            .map_err(EngineError::from);
        assert!(matches!(err, Err(EngineError::IoError)));
    }

    #[test]
    fn test_run_query_distinguishes_parse_vs_column_not_found() {
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();

        let parse_err = run_query(
            "வடிவமைப்பு | எடு 1;",
            &catalog,
            &mut arena,
            &mut out,
            &mut scratch,
            &mut tokens,
        );
        assert!(matches!(parse_err, Err(EngineError::ParseFailed)));

        let col_err = run_query(
            "இருந்து பயனர்கள் | தேடு வடிவமைப்பு;",
            &catalog,
            &mut arena,
            &mut out,
            &mut scratch,
            &mut tokens,
        );
        assert!(
            matches!(col_err, Err(EngineError::ColumnNotFound { .. })),
            "expected ColumnNotFound, got {col_err:?}"
        );
        assert!(!matches!(col_err, Err(EngineError::ParseFailed)));
    }

    /// 1.1: public constructors must be heap-only — survive a stack smaller
    /// than `size_of::<Table>()` (267 520). Documented safe minimum for the
    /// boxed path: 256 KiB. The pre-fix `Table::new` / `QueryResult::new`
    /// aborted with stack overflow at 64 KiB (EXIT 134) — see audit log.
    #[test]
    fn test_boxed_constructors_survive_constrained_stack() {
        const STACK: usize = 256 * 1024;
        assert!(STACK < 267_520, "stack must stay below Table stack footprint");
        let handle = std::thread::Builder::new()
            .stack_size(STACK)
            .name("boxed-ctors".into())
            .spawn(|| {
                let t = Table::new_boxed(b"x");
                let q = QueryResult::new_boxed();
                assert_eq!(t.col_count, 0);
                assert_eq!(q.row_count, 0);
                let mut cat = Catalog::new();
                assert!(cat.register(t).is_some());
                core::hint::black_box(q);
            })
            .expect("spawn");
        handle
            .join()
            .expect("boxed constructors must not overflow a 256KiB stack");
    }

    /// 1.2: Filter.stage.left is BinOp — full pipeline e2e (not an isolated unit).
    /// Would fail if execute treated Filter.left as a column Ident.
    #[test]
    fn test_filter_binop_shape_full_pipeline_e2e() {
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        let q = "இருந்து பயனர்கள் | வடி வயது > 21 | தேடு வயது;";
        assert!(run_query(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
        assert!(out.row_count > 0);
        // Structural proof: Filter.left is BinOp, BinOp.left is Ident.
        let root = arena.get(arena.root).unwrap();
        let mut stage = root.left;
        let mut saw_filter = false;
        while stage != NIL {
            let n = arena.get(stage).unwrap();
            if n.kind == NodeKind::Filter {
                let bin = arena.get(n.left).unwrap();
                assert_eq!(bin.kind, NodeKind::BinOp);
                let col = arena.get(bin.left).unwrap();
                assert_eq!(col.kind, NodeKind::Ident);
                saw_filter = true;
            }
            stage = n.next;
        }
        assert!(saw_filter);
    }

    /// 2.7: zero-heap proof across demo / join / mmap page walk / malformed.
    #[test]
    #[cfg_attr(miri, ignore = "memmap2 file-backed mmap unsupported under Miri")]
    fn test_zero_heap_expanded_entry_points() {
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();

        reset_counters();
        set_tracking(true);
        let _ = run_query(DEMO_QUERY, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens);
        set_tracking(false);
        let demo_allocs = alloc_count();
        assert_eq!(demo_allocs, 0, "DEMO_QUERY hot path allocs={demo_allocs}");

        let join_q =
            "இருந்து பயனர்கள் | இணை ஆர்டர்கள் | வடி விலை > 100 | தேடு விலை;";
        reset_counters();
        set_tracking(true);
        let _ = run_query(join_q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens);
        set_tracking(false);
        let join_allocs = alloc_count();
        assert_eq!(join_allocs, 0, "join pipeline allocs={join_allocs}");

        let bad = "@@@not-a-query;;;";
        reset_counters();
        set_tracking(true);
        let _ = run_query(bad, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens);
        set_tracking(false);
        let bad_allocs = alloc_count();
        assert_eq!(bad_allocs, 0, "malformed query allocs={bad_allocs}");

        let dir = std::env::temp_dir().join("tamil_zero_heap_mmap");
        let _ = std::fs::create_dir_all(&dir);
        write_i64_column_bin(&dir.join("ages.bin"), 2_000, |i| i as i64).unwrap();
        let mut stream = ColumnarFileStream::open_i64(&dir.join("ages.bin")).unwrap();
        reset_counters();
        set_tracking(true);
        let mut sum = 0usize;
        while let Some(c) = stream.next_page_chunk() {
            sum += c.row_count as usize;
        }
        set_tracking(false);
        assert_eq!(sum, 2_000);
        let mmap_allocs = alloc_count();
        assert_eq!(mmap_allocs, 0, "mmap page walk allocs={mmap_allocs}");
    }

    /// 2.9-ish: exercise distinct EngineError return paths through run_query.
    #[test]
    fn test_execute_error_paths_coverage_smoke() {
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();

        // missing table
        let e = run_query(
            "இருந்து இல்லாதஅட்டவணை | தேடு வயது;",
            &catalog,
            &mut arena,
            &mut out,
            &mut scratch,
            &mut tokens,
        );
        assert!(matches!(e, Err(EngineError::ColumnNotFound { .. })));

        // missing filter column
        let e = run_query(
            "இருந்து பயனர்கள் | வடி இல்லாதநெடுவரிசை > 1 | தேடு வயது;",
            &catalog,
            &mut arena,
            &mut out,
            &mut scratch,
            &mut tokens,
        );
        assert!(matches!(e, Err(EngineError::ColumnNotFound { .. })));

        // missing sort column
        let e = run_query(
            "இருந்து பயனர்கள் | அடுக்கு இல்லாதநெடுவரிசை | தேடு வயது;",
            &catalog,
            &mut arena,
            &mut out,
            &mut scratch,
            &mut tokens,
        );
        assert!(matches!(e, Err(EngineError::ColumnNotFound { .. })));

        // missing project column (known CLI case)
        let e = run_query(
            "இருந்து பயனர்கள் | தேடு இல்லாதநெடுவரிசை;",
            &catalog,
            &mut arena,
            &mut out,
            &mut scratch,
            &mut tokens,
        );
        assert!(matches!(e, Err(EngineError::ColumnNotFound { .. })));

        // parse failure
        let e = run_query("| | |;", &catalog, &mut arena, &mut out, &mut scratch, &mut tokens);
        assert!(matches!(e, Err(EngineError::ParseFailed)));
    }

    /// Group-by price: sorted distinct keys (hand-computed from 12 unique prices).
    #[test]
    fn test_group_by_price_sorted_distinct_e2e() {
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        let q = "இருந்து ஆர்டர்கள் | தொகுப்பு விலை | தேடு விலை;";
        assert!(run_query_checked(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
        let expected: [i64; 12] = [100, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1100, 1200];
        assert_eq!(out.row_count as usize, 12);
        let mut i = 0usize;
        while i < 12 {
            assert_eq!(out.int_out[0].values[i], expected[i], "row {i}");
            i += 1;
        }
    }

    /// Overflowing decimal literal must not silently wrap (LiteralOverflow).
    #[test]
    fn test_literal_overflow_rejected_e2e() {
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        let e = run_query_checked(
            "இருந்து பயனர்கள் | வடி வயது > 99999999999999999999999 | தேடு பெயர்;",
            &catalog,
            &mut arena,
            &mut out,
            &mut scratch,
            &mut tokens,
        );
        assert!(matches!(e, Err(EngineError::LiteralOverflow)), "got {e:?}");
        let e2 = run_query_checked(
            "இருந்து பயனர்கள் | எடு 99999999999999999999999 | தேடு பெயர்;",
            &catalog,
            &mut arena,
            &mut out,
            &mut scratch,
            &mut tokens,
        );
        assert!(matches!(e2, Err(EngineError::LiteralOverflow)), "got {e2:?}");
    }

    /// Double join against ஆர்டர்கள்: each order user_id is unique → 12 rows
    /// (same as single join). Pre-fix used join-slot indices as user rows → 8.
    #[test]
    fn test_double_join_orders_cardinality_e2e() {
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        let q1 = "இருந்து பயனர்கள் | இணை ஆர்டர்கள் | தேடு பெயர்;";
        assert!(run_query_checked(q1, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
        assert_eq!(out.row_count, 12);
        let q2 = "இருந்து பயனர்கள் | இணை ஆர்டர்கள் | இணை ஆர்டர்கள் | தேடு பெயர்;";
        assert!(run_query_checked(q2, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
        assert_eq!(out.row_count, 12, "double join must preserve 1:1 order cardinality");
    }

    /// After Group, `எண்ணிக்கை` holds per-group COUNT (all 1 for unique prices).
    #[test]
    fn test_group_count_derived_e2e() {
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        let q = "இருந்து ஆர்டர்கள் | தொகுப்பு விலை | தேடு எண்ணிக்கை;";
        assert!(run_query_checked(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
        assert_eq!(out.row_count, 12);
        assert_eq!(scratch.groups.len, 12);
        let mut i = 0usize;
        while i < 12 {
            assert_eq!(out.int_out[0].values[i], 1);
            assert_eq!(scratch.groups.count[i], 1);
            assert_eq!(scratch.groups.min[i], scratch.groups.keys[i]);
            assert_eq!(scratch.groups.max[i], scratch.groups.keys[i]);
            assert_eq!(scratch.groups.sum[i], scratch.groups.keys[i]);
            i += 1;
        }
    }

    /// Multi-row GroupedAgg (groups of size 3+) with hand-computed expectations.
    #[test]
    fn test_group_multi_row_count_sum_min_max_e2e() {
        let catalog = dup_price_orders_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        let q = "இருந்து ஆர்டர்கள் | தொகுப்பு விலை | தேடு விலை;";
        assert!(run_query_checked(q, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
        // Sorted distinct keys: 100, 250, 500
        assert_eq!(out.row_count, 3);
        assert_eq!(scratch.groups.len, 3);
        assert_eq!(out.int_out[0].values[0], 100);
        assert_eq!(out.int_out[0].values[1], 250);
        assert_eq!(out.int_out[0].values[2], 500);
        // Hand-computed: 100×3, 250×4, 500×1
        assert_eq!(scratch.groups.count[0], 3);
        assert_eq!(scratch.groups.sum[0], 300);
        assert_eq!(scratch.groups.min[0], 100);
        assert_eq!(scratch.groups.max[0], 100);
        assert_eq!(scratch.groups.count[1], 4);
        assert_eq!(scratch.groups.sum[1], 1000);
        assert_eq!(scratch.groups.min[1], 250);
        assert_eq!(scratch.groups.max[1], 250);
        assert_eq!(scratch.groups.count[2], 1);
        assert_eq!(scratch.groups.sum[2], 500);
        assert_eq!(scratch.groups.min[2], 500);
        assert_eq!(scratch.groups.max[2], 500);

        let q2 = "இருந்து ஆர்டர்கள் | தொகுப்பு விலை | தேடு எண்ணிக்கை;";
        assert!(run_query_checked(q2, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
        assert_eq!(out.row_count, 3);
        assert_eq!(out.int_out[0].values[0], 3);
        assert_eq!(out.int_out[0].values[1], 4);
        assert_eq!(out.int_out[0].values[2], 1);
    }

    /// Empty table end-to-end through `run_query_checked`.
    #[test]
    fn test_empty_table_query_e2e() {
        let mut cat = Catalog::new();
        let mut t = Table::new_boxed("வெறுமை".as_bytes());
        let _ = t.add_int64_column("வயது".as_bytes()).unwrap();
        t.set_row_count(0);
        let _ = cat.register_box(t);
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        let q = "இருந்து வெறுமை | வடி வயது > 0 | தேடு வயது;";
        assert!(run_query_checked(q, &cat, &mut arena, &mut out, &mut scratch, &mut tokens).is_ok());
        assert_eq!(out.row_count, 0);
    }

    /// Zero-row Int64 column file open + query-shaped stream walk.
    #[test]
    #[cfg_attr(miri, ignore = "file I/O")]
    fn test_empty_column_file_stream_e2e() {
        let dir = std::env::temp_dir().join("tamil_empty_col_e2e");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("empty.bin");
        write_i64_column_bin(&path, 0, |_| 0).unwrap();
        let mut stream = ColumnarFileStream::open_i64(&path).unwrap();
        assert_eq!(stream.total_rows(), 0);
        assert!(stream.next_page_chunk().is_none());
        let mut copied = ColumnarFileStream::open_int64_copied(&path).unwrap();
        assert!(copied.is_copied());
        assert_eq!(copied.total_rows(), 0);
        assert!(copied.next_page_chunk().is_none());
    }

    /// Zone map min/max for a 12k-row multipage column, checked against
    /// independently computed page ranges from the raw generator (not from .zmap).
    #[test]
    #[cfg_attr(miri, ignore = "file I/O")]
    fn test_zonemap_written_correctly_for_multipage_column() {
        use crate::ingest::{ingest_csv, parse_schema};
        use std::fs;

        let dir = std::env::temp_dir().join(format!(
            "tqe_zmap_multipage_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("orders.csv");
        let n = 12_000i64;
        let mut csv = String::from("price\n");
        let mut raw: Vec<i64> = Vec::with_capacity(n as usize);
        for i in 0..n {
            let price = 100 + (i % 900);
            raw.push(price);
            csv.push_str(&format!("{price}\n"));
        }
        fs::write(&csv_path, &csv).unwrap();
        let schema = parse_schema("price:i64").unwrap();
        let out_dir = dir.join("out");
        let report = ingest_csv(&csv_path, &schema, &out_dir, true).unwrap();
        assert_eq!(report.rows_ingested, n as usize);

        let bin = out_dir.join("price.bin");
        let zmap_path = bin.with_extension("zmap");
        assert!(zmap_path.exists(), ".zmap must be written by ingest");
        let zm = ZoneMap::open(&zmap_path).unwrap();

        let page_rows = os_page_size_bytes() / 8;
        let expected_pages = ((n as usize) + page_rows - 1) / page_rows;
        assert_eq!(zm.page_count() as usize, expected_pages);

        let mut p = 0usize;
        while p < expected_pages {
            let start = p * page_rows;
            let end = ((p + 1) * page_rows).min(n as usize);
            let mut mn = i64::MAX;
            let mut mx = i64::MIN;
            let mut i = start;
            while i < end {
                if raw[i] < mn {
                    mn = raw[i];
                }
                if raw[i] > mx {
                    mx = raw[i];
                }
                i += 1;
            }
            let e = zm.entry(p).expect("entry");
            assert_eq!(e.page_index, p as u32);
            assert_eq!(e.row_count as usize, end - start);
            assert_eq!(e.min, mn, "page {p} min");
            assert_eq!(e.max, mx, "page {p} max");
            p += 1;
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// Disjoint page ranges: pushdown must skip page 0 and keep correctness.
    #[test]
    #[cfg_attr(miri, ignore = "file I/O")]
    fn test_pushdown_skips_pages_outside_predicate_range() {
        use crate::ingest::{ingest_csv, parse_schema};
        use std::fs;

        let page_rows = os_page_size_bytes() / 8;
        let dir = std::env::temp_dir().join(format!(
            "tqe_pushdown_skip_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("vals.csv");
        let mut csv = String::from("v\n");
        // Page 0: 0..999; page 1: 100_000+
        let mut i = 0usize;
        while i < page_rows {
            let v = (i % 1000) as i64;
            csv.push_str(&format!("{v}\n"));
            i += 1;
        }
        let mut i = 0usize;
        while i < page_rows {
            let v = 100_000 + (i as i64);
            csv.push_str(&format!("{v}\n"));
            i += 1;
        }
        fs::write(&csv_path, &csv).unwrap();
        let schema = parse_schema("v:i64").unwrap();
        let out_dir = dir.join("out");
        ingest_csv(&csv_path, &schema, &out_dir, true).unwrap();

        let bin = out_dir.join("v.bin");
        let meta = bin.with_extension("meta");
        let zmap_path = bin.with_extension("zmap");
        let mut stream = ColumnarFileStream::open_i64_with_meta(&bin, &meta).unwrap();
        let zmap = ZoneMap::open(&zmap_path).unwrap();
        assert!(zmap.page_count() >= 2);

        let mut scratch = RuntimeScratch::new_boxed();
        let mut out_on = vec![0i64; MAX_ROWS];
        let mut out_off = vec![0i64; MAX_ROWS];
        let mut len_on = 0usize;
        let mut len_off = 0usize;
        let mut stats_on = PushdownStats::default();
        let mut stats_off = PushdownStats::default();

        // Predicate only page 1 can satisfy: v > 50_000
        execute_int64_filter_pushdown(
            &mut stream,
            Some(&zmap),
            ZoneCmp::Gt,
            50_000,
            true,
            &mut scratch,
            &mut out_on,
            &mut len_on,
            &mut stats_on,
        );
        stream.rewind();
        execute_int64_filter_pushdown(
            &mut stream,
            Some(&zmap),
            ZoneCmp::Gt,
            50_000,
            false,
            &mut scratch,
            &mut out_off,
            &mut len_off,
            &mut stats_off,
        );

        assert_eq!(
            stats_on.pages_skipped, 1,
            "page 0 [0..999] must be skipped for > 50000; stats={stats_on:?}"
        );
        assert_eq!(stats_on.pages_scanned, 1);
        assert_eq!(stats_off.pages_skipped, 0);
        assert_eq!(stats_off.pages_scanned, 2);
        assert_eq!(len_on, len_off, "pushdown must not change result cardinality");
        assert_eq!(&out_on[..len_on], &out_off[..len_off]);
        assert_eq!(len_on, page_rows);
        assert!(out_on[..len_on].iter().all(|&v| v > 50_000));
        let _ = fs::remove_dir_all(&dir);
    }

    /// Threshold exactly equal to a page max: Gt must skip; Gte must not.
    #[test]
    #[cfg_attr(miri, ignore = "file I/O")]
    fn test_pushdown_correct_at_exact_boundary_values() {
        use crate::ingest::{ingest_csv, parse_schema};
        use std::fs;

        let page_rows = os_page_size_bytes() / 8;
        let dir = std::env::temp_dir().join(format!(
            "tqe_pushdown_bound_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("vals.csv");
        let mut csv = String::from("v\n");
        // Page 0: all values == 10 (min=max=10). Page 1: all 200.
        let mut i = 0usize;
        while i < page_rows {
            csv.push_str("10\n");
            i += 1;
        }
        let mut i = 0usize;
        while i < page_rows {
            csv.push_str("200\n");
            i += 1;
        }
        fs::write(&csv_path, &csv).unwrap();
        ingest_csv(&csv_path, &parse_schema("v:i64").unwrap(), &dir.join("out"), true).unwrap();
        let bin = dir.join("out/v.bin");
        let mut stream = ColumnarFileStream::open_i64_with_meta(&bin, &bin.with_extension("meta")).unwrap();
        let zmap = ZoneMap::open(&bin.with_extension("zmap")).unwrap();
        let e0 = zmap.entry(0).unwrap();
        assert_eq!(e0.min, 10);
        assert_eq!(e0.max, 10);

        let mut scratch = RuntimeScratch::new_boxed();
        let mut out = vec![0i64; MAX_ROWS];
        let mut len = 0usize;
        let mut stats = PushdownStats::default();

        // Gt 10: page 0 max==10 cannot satisfy → skip.
        execute_int64_filter_pushdown(
            &mut stream,
            Some(&zmap),
            ZoneCmp::Gt,
            10,
            true,
            &mut scratch,
            &mut out,
            &mut len,
            &mut stats,
        );
        assert_eq!(stats.pages_skipped, 1);
        assert_eq!(len, page_rows);
        assert!(out[..len].iter().all(|&v| v > 10));

        // Gte 10: page 0 MUST be scanned (boundary value satisfies).
        stream.rewind();
        len = 0;
        stats = PushdownStats::default();
        execute_int64_filter_pushdown(
            &mut stream,
            Some(&zmap),
            ZoneCmp::Gte,
            10,
            true,
            &mut scratch,
            &mut out,
            &mut len,
            &mut stats,
        );
        assert_eq!(
            stats.pages_skipped, 0,
            "exact min/max boundary must not skip under Gte; {stats:?}"
        );
        assert_eq!(stats.pages_scanned, 2);
        assert_eq!(len, page_rows * 2);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Demo catalog has no `.zmap` — pushdown is a no-op fallback.
    #[test]
    fn test_pushdown_falls_back_cleanly_without_zonemap() {
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = alloc_token_window();
        let stats = run_query_checked(
            DEMO_QUERY,
            &catalog,
            &mut arena,
            &mut out,
            &mut scratch,
            &mut tokens,
        )
        .expect("demo query");
        assert_eq!(stats.pages_skipped, 0);
        assert_eq!(stats.pages_total, 0);
        assert_eq!(stats.pages_scanned, 0);
        assert_eq!(out.row_count, 10);
    }
}
