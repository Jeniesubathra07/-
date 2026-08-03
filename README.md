# Tamil Query Engine

A hyper-optimized, microsecond-scale, zero-allocation database engine and query language written natively in Rust for Tamil DSL processing. Implements a left-to-right non-backtracking linear pipeline, a data-oriented flat AST arena parser, and an Apache Arrow-aligned columnar storage kernel utilizing hardware-level branchless SIMD execution.

## Pipeline DSL

```text
இருந்து பயனர்கள் | வடி வயது > 21 | அடுக்கு வயது | எடு 10 | தேடு பெயர், வயது;
```

| Keyword | Meaning |
|---------|---------|
| இருந்து | From |
| வடி | Filter |
| கணி | Derive |
| அடுக்கு | Sort |
| எடு | Take |
| தொகுப்பு | Group |
| சுருக்கு | Aggregate |
| இணை | Join |
| எங்கே | Conditional |
| தேடு | Select / Project |

## Modules

- `src/lexer.rs` — byte-slice lexer with Tamil keyword tables and branchless number parse
- `src/parser.rs` — flat `[AstNode; 1024]` arena (`u32` child links)
- `src/storage.rs` — columnar Int64 / Utf8 segments (Arrow-style offsets + slab)
- `src/runtime.rs` — batch-1024 vectorized filter / sort / take / project
- `src/utf8.rs` — grapheme-safe Tamil UTF-8 boundary helpers

## Build / Test

```bash
cargo test
cargo build --release
```
