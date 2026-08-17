# Ω-STAGE-X Exhaustive Correctness Audit

Executed on x86_64 Linux in this environment. Claims without pasted command
output are marked **NOT VERIFIED**.

## Defects found

1. **[KNOWN] Stack-overflow public constructors** — `Table::new` /
   `QueryResult::new` / `Default for QueryResult` / by-value `Catalog::register(Table)`.
   - BEFORE (64 KiB thread, public `Table::new` + `QueryResult::new`):
     `thread 'stack-probe' has overflowed its stack` … `EXIT:134`
   - AFTER: constructors removed; only `*_boxed`; `register(Box<Table>)`;
     external `Table::new` → `error[E0599]`; boxed path on 256 KiB stack →
     `JOIN_OK` / `test_boxed_constructors_survive_constrained_stack` ok.
2. **[DISCOVERED] AST-shape: Project `ColumnList` head** — parser retags the
   first projection Ident as `NodeKind::ColumnList` while keeping its span.
   A new Ident-only guard rejected the demo pipeline (`Err(ParseFailed)`).
   Fixed via `expect_projection_name_bytes` accepting `Ident | ColumnList`.
3. **[KNOWN class] Filter `BinOp` vs Ident** — audited; Filter path already
   walked `BinOp.left`; added explicit `BinOp` / `Ident` guards + full-pipeline
   e2e `test_filter_binop_shape_full_pipeline_e2e`.
4. **[DISCOVERED] Silent release wraparound policy gap** —
   `overflow-checks = true` added to `[profile.release]`; Utf8
   `set_row` / ingest blob length use `checked_add` / `try_from`.
5. **[DISCOVERED] mmap SIGBUS vs “defensive Result” philosophy** — documented
   as hard precondition; subprocess truncate-while-mapped delivered
   **SIGBUS** (`returncode -7`, harness `EXIT:135`). Not convertible to
   `Result` without a signal handler (out of scope; no new deps).

## Tooling results (executed)

| Tool | Result |
|------|--------|
| `cargo test --lib` | **54 passed** |
| `cargo test --release --lib` | **54 passed** |
| `cargo +nightly miri test --lib` (`MIRIFLAGS=-Zmiri-disable-isolation`) | **47 passed, 7 ignored** (file-backed mmap unsupported by Miri) |
| ASAN (`cargo +nightly test -Zbuild-std --target x86_64-unknown-linux-gnu --lib`) | **54 passed**, exit 0 |
| UBSAN (`-Zsanitizer=undefined`) | **NOT AVAILABLE** on this rustc (sanitizer list has no `undefined`) |
| `cargo fuzz` / libFuzzer | **NOT VERIFIED** — libc++ headers missing (`cassert` / `cstdint`) |
| Pure-Rust `audit_fuzz_driver 60` | 60s+60s, **no crash**; query iters=468051575; columnar iters=494146 |
| `cargo llvm-cov test --lib --summary-only` | TOTAL lines **83.67%**; runtime.rs **73.55%** |
| SIGBUS harness | truncate-after-mmap → **SIGBUS** |

## AST Ident sites audited

| Site | Status |
|------|--------|
| `From` / `Join` relation | Fixed: `expect_ident_bytes` |
| `Filter` predicate | Safe after `BinOp` assert + Ident on `BinOp.left` |
| `Sort` column | Fixed: `expect_ident_bytes` |
| `Project` list | Fixed: `ColumnList \| Ident` |
| `Derive` target | Fixed: Ident kind check |
| `apply_filter` left | Fixed: Ident kind check |
| `resolve_operand_dense` | Already matched on `NodeKind::Ident` |

`run_query_checked` / `validate_columns` — **not present** in this tree; N/A.

## Unsafe inventory

`rg '\bunsafe\b' src/` → **39** matches across storage/runtime/lib/parser/lexer
(includes `unsafe fn`, `unsafe impl`, and blocks). Hot-path mmap i64
reinterpret documents page-alignment ≥ 8; verified under Miri only for
non-mmap paths.

## Dependencies

- Shipped `[dependencies]`: still only **`memmap2 = "0.9"`** (pulls `libc` transitively).
- Tool-only: nightly miri, cargo-fuzz (unusable here), cargo-llvm-cov, fuzz/
  targets + `audit_fuzz_driver` / `sigbus_mmap_hazard` bins.

## NOT VERIFIED

- Windows / big-endian / non-Linux page-size APIs
- Classic UBSAN (`-Zsanitizer=undefined` unsupported)
- libFuzzer / AFL (missing C++ stdlib headers)
- Concurrent writer corruption; disk-full; network-FS drop mid-mmap
- 10-minute libFuzzer campaigns (substituted 60s pure-Rust mutation each)
- Whether ASAN observes SIGBUS truncate (signal kills process first)
