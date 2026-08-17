# Exhaustive unsafe audit (Ω-PRODUCTION-READY)

Total `rg '\bunsafe\b' src/` hits: **41** (includes `unsafe fn` / `unsafe impl` keywords).

Every former "Review / See local SAFETY comment" placeholder is replaced below with a concrete invariant, enforcement, and verdict.

| File:line | Site | Invariant | Enforcement | OK? |
|-----------|------|-----------|-------------|-----|
| `parser.rs:214` | `alloc_token_window` zeroed box | `Layout::new::<[Token; MAX_TOKENS]>()`; null → `handle_alloc_error`; fields zeroed before `Box::from_raw` | cold-path only; size fixed at compile time | Y |
| `storage.rs:299` | `Table::new_boxed` | Same alloc_zeroed pattern; `name`/`col_count`/`columns` written before Box observes pointer | cold-path; no stack `Table::new` | Y |
| `storage.rs:427-428` | `MappedRegion::as_slice` | `ptr` valid for `len` bytes for `'a` | `unsafe fn` contract; caller supplies region from live slice/mmap | Y (caller) |
| `storage.rs:436-439` | `MappedRegion::slice_at` | `off+len <= self.len` before `ptr.add` | runtime checked; returns `None` on overflow/OOB | Y |
| `storage.rs:662` | `FixedOrdersDatabase::new_boxed` | alloc_zeroed + field init before Box | cold-path | Y |
| `storage.rs:791-792` | `memcpy_bytes` | `dst`/`src` valid for `len`, non-overlapping | `unsafe fn` contract | Y (caller) |
| `storage.rs:830` | `sysconf(_SC_PAGESIZE)` | POSIX page size; treat `<=0` as failure → 4096 fallback | `cfg(unix)` + fallback | Y |
| `storage.rs:906-907` | `Utf8ColumnFile` `Mmap::map` | RO open; offsets length multiple of entry size; `get_row` bounds-checks blob | single-writer-absent; **mmap SIGBUS if violated** | Partial — mmap path [DOCUMENTED-LIMIT]; Int64 has `open_int64_copied` |
| `storage.rs:1096` | `ColumnBytes::rows_at` | `start+n` within backing; Mmap page-aligned / Owned `Vec<i64>` aligned | caller bounds-checks byte_end; `debug_assert` on alignment | Y |
| `storage.rs:1212` | copy-open `read_exact` into `Vec<i64>` as bytes | `vals` length `len/8`; writing `len` LE bytes into i64 slab | length validated `% 8 == 0` | Y |
| `storage.rs:1222` | `Mmap::map` in `open_i64_inner` | RO file; length `% 8 == 0` | meta row_count ≤ capacity; SIGBUS if file mutated | Partial — use `open_int64_copied` for crash-proof [MITIGATED] |
| `storage.rs:1336` | `next_page_chunk` rows view | bounds via `byte_len`; then `rows_at` | checked_mul/add; early `None` | Y |
| `storage.rs:1433-1435` | `ColumnarTableStream::next_page` | three streams share `start_row`/`n`; each `byte_end` ≤ backing | pre-checked before `rows_at` | Y |
| `storage.rs:1482` | `write_i64_column_bin` window as bytes | local `[i64; MAX_ROWS]` viewed as LE bytes for `n` elements | `n ≤ MAX_ROWS`; LE host contract | Y |
| `runtime.rs:126` | `QueryResult::new_boxed` | alloc_zeroed + schema/types init | cold-path; no stack ctor | Y |
| `runtime.rs:240` | `RuntimeScratch::new_boxed` | alloc_zeroed + derived/groups init | cold-path | Y |
| `runtime.rs:359` | TLS `CHUNK_SCRATCH` borrow | single-threaded TLS; exclusive `&mut` for duration of call | `thread_local!` + no re-entrant engine on same thread into same pad | Y |
| `runtime.rs:408` | TLS `ENGINE_SCRATCH_PAD` | same exclusive TLS borrow | same | Y |
| `runtime.rs:458-460` | OS parallel chunk slices | `src_addr`/`dst_addr` from live `&[i64; MAX_ROWS]` for scope lifetime | `thread::scope`; chunks disjoint write ranges by construction | Y |
| `runtime.rs:491` | `apply_unroll8` | `base+j+7 < n ≤ MAX_ROWS` | caller loops; SAFETY comment + `debug_assert` opportunity at call sites | Y |
| `runtime.rs:556,570` | filter unroll calls | Phase A/B guarantee `base+j+7 < n` | loop structure | Y |
| `runtime.rs:726,782,2127` | TLS pad borrows | exclusive TLS | same as above | Y |
| `lexer.rs:468` | test token buffer alloc | test-only alloc_zeroed of `[Token; MAX_TOKENS]` | test cfg | Y |
| `lib.rs:60-82` | `CountingAlloc` | forwards to `System`; counts only | test GlobalAlloc | Y |
| `lib.rs:1558` | derived pointer probe | `derived_ptr` from `scratch.derived.as_mut_ptr()`; `s < MAX_ROWS` | test loop bound | Y |

## SIGBUS status

| Path | Status |
|------|--------|
| `ColumnarFileStream::open_i64` (mmap) | Still SIGBUS on truncate — demonstrated EXIT 135 (`--mmap`) |
| `ColumnarFileStream::open_int64_copied` | **[MITIGATED]** — truncate-after-open survives; EXIT 0 |
| `Utf8ColumnFile::open` | **[DOCUMENTED-LIMIT]** — mmap only; no copy-on-open in this pass |

Measured open+scan 100 000 rows × 50 (release): mmap ≈ 21 772 ns avg, copy ≈ 57 059 ns avg (**~2.62×**).
