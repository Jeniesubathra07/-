# Ω-FINAL-CONVERGENCE Audit Report

## Baseline re-verification (Section 1)

| Claim | Live status |
|-------|-------------|
| 1.1 Stack ctors removed | **CONFIRMED** — only `*_boxed`; no `Default` for `QueryResult` |
| 1.2 `validate_columns` BinOp | **N/A** — symbol not in tree; Filter walks use `BinOp` + `expect_ident_bytes` |
| 1.3 Group silent no-op | **WAS STILL BROKEN** → fixed this pass ([NEW]) |
| 1.4 `scan_number` wrap | **WAS STILL BROKEN** → fixed this pass ([NEW]) |

## Defects

1. **[NEW] Group/`தொகுப்பு` silent no-op** — returned unsorted raw rows, EXIT 0.  
   BEFORE: `450,800,…,100`. AFTER: sorted distinct `100…1200`.  
   Fix: sort-then-scan collapse via `lsd_radix_sort_ages` (no `HashMap`).

2. **[NEW] `scan_number` wrapping_mul/add** — 23-digit literal looked like empty filter / no-op take.  
   BEFORE: EXIT 0 empty/`all rows`. AFTER: `LiteralOverflow` EXIT 1.  
   Fix: `checked_*` + `Parser::expect_number` rejecting `i64::MIN` sentinel.

3. **[NEW] Double-join wrong cardinality** — used join-slot index as user row → 8 rows.  
   BEFORE: 8. AFTER: 12 (matches hand 1:1 order keys).  
   Fix: when `joined`, left keys from `join_left[slot]`.

4. **[BASELINE-CONFIRMED] Stack ctor / SIGBUS docs / Ident guards** — still in place.

## `overflow-checks` decision

**KEEP `overflow-checks = true`** in `[profile.release]`.

Measured `stage3_bench` (2000 iters, release):

| Metric | ON | OFF |
|--------|-----|-----|
| demo_pipeline_e2e | 1472 ns | 1844 ns |
| derive_kani_join_filter | 2492 ns | 2486 ns |

Deltas are noise-level for the demo path; untrusted boundaries also use `checked_*`.

## Derive / Aggregate

- **`கணி` Derive**: already implemented (not deferred).
- **`சுருக்கு` Aggregate**: SUM over groups (COUNT stored by Group); group-key==measure → exact SUM.

## Unsafe inventory

`rg '\bunsafe\b' src/` → **39** matches. Mmap i64 views: page-aligned base, `row_width==8`, length checked; SIGBUS is a separate POSIX hazard (documented; subprocess EXIT 135).

## `Engine::execute` Err coverage (manual; no tarpaulin)

| Path | Covered by |
|------|------------|
| ParseFailed (bad AST / missing From) | `test_run_query_distinguishes_*`, `test_execute_error_paths_*` |
| ColumnNotFound table/column/project/filter/sort | `test_execute_error_paths_*`, distinguishes test |
| LiteralOverflow | `test_literal_overflow_rejected_e2e` |
| Group/join success paths | `test_group_by_price_*`, `test_double_join_*` |
| IoError / PageCorrupt | mmap missing-file test; PageCorrupt via streaming guard |

`validate_columns`: **not present** — N/A.

## NOT VERIFIED

- Full Miri/ASAN re-run this pass (prior pass had clean ASAN/Miri; nightly available but time-boxed)
- libFuzzer (libc++ headers historically missing); pure-Rust fuzz driver remains
- Windows / big-endian / real disk-full / concurrent writers
- Signal-handler SIGBUS recovery (documented unmitigated; EXIT 135 observed)
