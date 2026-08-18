# Ω-PHASE-2 — Zone-map predicate pushdown (final report)

## Baseline

Phase 0/1 intact. Representative `cargo test --release --lib` after Phase 2:

```
test result: ok. 86 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

(Prior Phase 1 baseline was 77; Phase 2 adds zonemap + pushdown coverage → **86**.)

`overflow-checks = true` remains set in `[profile.release]`.

## Zone map file format (as implemented)

Per Int64 column: `<column>.zmap` beside `<column>.bin` / `<column>.meta`.

On-disk record size **24 bytes**, little-endian, one record per OS-derived
page chunk (not `MAX_ROWS`):

| Offset | Field        | Type |
|--------|--------------|------|
| 0      | `page_index` | `u32` |
| 4      | `row_count`  | `u32` |
| 8      | `min`        | `i64` |
| 16     | `max`        | `i64` |

In-memory: `ZonePage` is `#[repr(C, align(64))]`. `ZoneMap` loads into a
fixed `[ZonePage; MAX_ZONE_PAGES]` (`MAX_ZONE_PAGES = 1024`) — zero heap at
query-open after the cold `File::read_exact` loop.

`write_zonemap_for_column(bin_path, zmap_path)` streams the published `.bin`
via `ColumnarFileStream::open_int64_copied` (does not re-read CSV). Meta path
is not a separate argument — row count comes from the `.bin` (+ companion
`.meta` when present via the normal open path).

`tqe_ingest` auto-writes `.zmap` for every Int64 column after successful
publish; Utf8 columns are skipped (no stub).

## Pushdown

- `ZonePredicate::{Gt, Lt, Eq}` — matches lexer tokens (`>`, `<`, `=`). No
  `>=` / `<=` / `!=` in the grammar.
- `execute_mmap_i64_filter_stream_pushdown` consults the map once-loaded;
  impossible pages are not decoded into scratch pads.
- Missing `.zmap` → full scan; `PushdownStats.pages_skipped == 0`.
- `run_query_checked` returns `PushdownStats::in_memory_fallback()` for the
  demo catalog (no on-disk pages).

## Tests proving skip + correctness

| Protocol name | Implemented as |
|---------------|----------------|
| multipage min/max vs raw data | `zonemap::tests::zonemap_written_correctly_for_multipage_column` (+ e2e twin) |
| skip + identical result set | `runtime::tests::pushdown_skips_pages_outside_predicate_range_and_stays_correct` |
| boundary min/max | `runtime::tests::pushdown_correct_at_exact_boundary_threshold` + `zonemap::tests::predicate_boundary_values_are_correct_not_off_by_one` |
| demo no-zmap fallback | `test_pushdown_falls_back_cleanly_without_zonemap` (e2e + runtime) |

## Benchmark (real numbers)

Command:

```
cargo run --release --bin zonemap_bench
```

Output (200 000 rows, 20 iters, release, this environment):

```
clustered (monotonic id column): full_scan=166418ns (pages_scanned=391/391) pushdown=2827ns (pages_scanned=5/391, skipped=386) speedup=58.867x
random (uniform 0..1_000_000): full_scan=163868ns (pages_scanned=391/391) pushdown=165321ns (pages_scanned=391/391, skipped=0) speedup=0.991x
```

**Finding:** On clustered/monotonic data, pushdown is ~59×. On uniform random
data with a mid-domain filter, **0 pages skipped** and wall-clock is
noise-level slower (~0.99×) — the zone-map consult cost with no skips. That
is reported honestly, not hidden.

## Deferred (out of scope this phase)

| Item | Why deferred |
|------|----------------|
| Utf8 zone maps | No numeric min/max; would need different stats |
| `>=` / `<=` / `!=` | Not in `TokenKind` / lexer today |
| Multi-column composite pushdown | Single-column filters only; premature until multi-predicate plans exist |
| Cost-based optimizer / join reorder | Premature until multi-table workloads at scale |

## PR

https://github.com/Jeniesubathra07/-/pull/8 — branch `cursor/phase2-zonemap-redesign-96c3`
