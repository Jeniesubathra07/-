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

pub use lexer::{Lexer, Token, TokenKind, MAX_TOKENS};
pub use parser::{parse_query, AstArena, AstNode, NodeKind, Parser, AST_CAP, NIL};
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
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Counting allocator used only in this test module to prove the query
    /// hot path performs zero heap allocations.
    struct CountingAlloc;

    static TRACKING: AtomicBool = AtomicBool::new(false);
    static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
    static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if TRACKING.load(Ordering::SeqCst) {
                ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
                ALLOC_BYTES.fetch_add(layout.size(), Ordering::SeqCst);
            }
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            if TRACKING.load(Ordering::SeqCst) {
                DEALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
            }
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            if TRACKING.load(Ordering::SeqCst) {
                ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
                ALLOC_BYTES.fetch_add(new_size, Ordering::SeqCst);
            }
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static A: CountingAlloc = CountingAlloc;

    fn reset_counters() {
        ALLOC_COUNT.store(0, Ordering::SeqCst);
        ALLOC_BYTES.store(0, Ordering::SeqCst);
        DEALLOC_COUNT.store(0, Ordering::SeqCst);
    }

    #[test]
    fn demo_pipeline_e2e_zero_heap_and_tamil_safe() {
        // Cold path: build catalog / pre-size arenas outside the measured window.
        let catalog = demo_catalog();
        let mut arena = Box::new(AstArena::new());
        let mut out = Box::new(QueryResult::new());

        // Verify vowel-marker integrity on the source before execution.
        assert!(DEMO_QUERY.contains("தேடு"));
        assert!(DEMO_QUERY.contains("பெயர்"));
        let thedu = "தேடு";
        // 'தே' = 'த' + 'ே' — two Unicode scalars, one grapheme visually
        let mut chars = thedu.chars();
        assert_eq!(chars.next(), Some('த'));
        assert_eq!(chars.next(), Some('ே'));
        assert_eq!(chars.next(), Some('ட'));
        assert_eq!(chars.next(), Some('ு'));

        reset_counters();
        TRACKING.store(true, Ordering::SeqCst);
        let ok = run_query(DEMO_QUERY, &catalog, &mut arena, &mut out);
        TRACKING.store(false, Ordering::SeqCst);

        assert!(ok, "pipeline must execute successfully");
        let allocs = ALLOC_COUNT.load(Ordering::SeqCst);
        let bytes = ALLOC_BYTES.load(Ordering::SeqCst);
        assert_eq!(
            allocs, 0,
            "hot path must not allocate (saw {allocs} allocs, {bytes} bytes)"
        );
        assert_eq!(bytes, 0);

        // Columnar projection: பெயர், வயது — 10 rows, ages > 21, sorted.
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
            // Round-trip through from_utf8 already done in get_row.
            assert!(core::str::from_utf8(name.as_bytes()).is_ok());
            let age = out.int_out[1].values[i];
            assert!(age > 21, "filter வயது > 21 violated: {age}");
            assert!(age >= prev, "sort அடுக்கு வயது violated");
            prev = age;
            i += 1;
        }

        // Expected sorted ages > 21 from seed data (ascending), take 10:
        // 22,23,24,25,26,27,28,29,30,31
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
                        // Ensure we did not tear ெ from ப
                        assert_eq!(t.chars().next(), Some('ப'));
                    }
                }
            }
        }
        assert!(found_thedu);
        assert!(found_peyar);
    }
}
