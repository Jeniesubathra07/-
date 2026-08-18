# Ω-PHASE-2 — Zone-map predicate pushdown

## Baseline

`cargo test --release --lib` prior to this phase: **77 passed**.
`overflow-checks = true` remains in `[profile.release]`.

## Zone map format (as implemented)

File: `<column>.zmap` beside `<column>.bin` / `<column>.meta`.

Each record is `#[repr(C, align(64))]` `ZoneMapEntry` (64 bytes):

| Field | Type | Meaning |
|--------|------|---------|
| `page_index` | `u32` | 0-based page ordinal (matches stream page windows) |
| `row_count` | `u32` | rows in this page window |
| `min` | `i64` | min value on the page |
| `max` | `i64` | max value on the page |
| `_pad` | `[u8; 40]` | alignment to 64 |

Written by `write_zonemap_for_column` via streaming `ColumnarFileStream`
(not re-reading CSV). `tqe_ingest` / `ingest_csv` auto-write `.zmap` for
every Int64 column after `.bin`/`.meta`.

## Pushdown

`execute_int64_filter_pushdown(..., enable_pushdown, ...)` consults
`ZoneMap` once loaded; unsatisfiable pages use `skip_next_page()` (no
scratch fill). Grammar today: lexer `>` / `<` / `=` only; `ZoneCmp` also
exposes Gte/Lte/Ne for boundary API tests.

`run_query_checked` → `Result<PushdownStats, EngineError>`; in-memory demo
catalog returns `PushdownStats::in_memory_fallback()` (all zeros).

## Benchmark (verbatim)

Command: `cargo run --release --bin stage3_bench`

```
pushdown probe: pages_total=196 pages_skipped=97 pages_scanned=99 kept_capped=4096
                  filter_pushdown_ON       28391 ns/iter  (200 iters)
                 filter_pushdown_OFF       89950 ns/iter  (200 iters)
speedup OFF/ON = 3.168x  (values >1 mean pushdown is faster)
```

100k-row half-low / half-high distribution; predicate `v > 500_000`.

## Deferred

- Utf8 zone maps (no numeric min/max).
- `>=` / `<=` / `!=` lexer tokens (not in grammar; ZoneCmp API ready).
- Multi-column / composite pushdown.
- Wiring Tamil DSL `Engine::execute` directly onto mmap+zmap catalogs
  (mmap filter API + `run_query_checked` stats cover the phase contract;
  in-memory From tables unchanged).

## Suite

```
test result: ok. 83 passed; 0 failed
```
