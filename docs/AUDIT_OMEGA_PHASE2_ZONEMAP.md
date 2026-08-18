# Ω-PHASE-2 (redesign) — Zone-map predicate pushdown

## Format (as implemented)

On-disk `.zmap`: packed 24-byte LE records (`ZONE_PAGE_BYTES`):
`page_index:u32`, `row_count:u32`, `min:i64`, `max:i64`.

In-memory: `ZonePage` is `#[repr(C, align(64))]`; `ZoneMap` holds
`[ZonePage; MAX_ZONE_PAGES]` (1024) loaded once via `ZoneMap::load`
(returns `Ok(None)` if missing).

`write_zonemap_for_column(bin, zmap)` streams via `open_int64_copied`.
CLI `tqe_ingest` writes `.zmap` after Int64 ingest (non-fatal on failure).

## Pushdown API

`execute_mmap_i64_filter_stream_pushdown(stream, Option<&ZoneMap>, ZonePredicate, …)`
with `PushdownStats { pages_total, pages_skipped, pages_scanned, rows_scanned, rows_kept }`.

Grammar scope: `ZonePredicate::{Gt,Lt,Eq}` only.

## Benchmark command

```
cargo run --release --bin zonemap_bench
```

Reports clustered (monotonic) vs random (uniform) distributions honestly.
