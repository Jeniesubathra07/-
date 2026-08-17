# Ω-PRODUCTION-READY — Final Hardening Report

## Baseline re-check (one representative run)

`cargo test --release --lib` → **67 passed** after this pass (was 58).
Representative still-green paths: group-by unique prices, literal overflow,
double-join cardinality, mmap page streaming, boxed constructors.

## Defect / gap list

1. **[MITIGATED]** SIGBUS on Int64 mmap truncate — added
   `ColumnarFileStream::open_int64_copied()` (owned `Vec<i64>`).
   - BEFORE (`--mmap`): EXIT **135** Bus error after truncate.
   - AFTER (default/`--copied`): EXIT **0**, `rows=4096 sum=14336`.
   - Tradeoff (100k rows × 50): mmap ≈ 21772 ns, copy ≈ 57059 ns (**2.62×**).

2. **[DOCUMENTED-LIMIT]** Utf8ColumnFile remains mmap-only (SIGBUS contract).
   Int64 production path can opt into copy-on-open.

3. **[NEW]** GroupedAgg projection bug — after `தொகுப்பு`, `derived` is dense by
   group index but project used source `order[slot]`. Size-1 unique-key fixtures
   masked it (`derived[slot]==1` for all slots). Multi-row fixture exposed
   `எண்ணிக்கை` row 2 = 0 instead of 1. Fixed by indexing `derived[oi]` when
   `groups.len == order_len`.

4. **[NEW-TEST]** `seed_dup_price_orders_table` / `test_group_multi_row_count_sum_min_max_e2e`
   — hand-computed 100×3 / 250×4 / 500×1.

5. **[NEW-TEST]** Empty table E2E; empty column file; corrupt `.meta`; concurrent
   mmap readers; Utf8 grapheme spanning page boundary; permission-denied write
   (no published `.bin`/`.meta`); `/dev/full` ENOSPC pattern.

6. **[CI-ADDED]** `.github/workflows/ci.yml` — stable build/test, nightly Miri,
   `audit_fuzz_driver 300` on main.

7. **[DOCUMENTED]** `docs/UNSAFE_AUDIT.md` — all former "Review" placeholders
   replaced with concrete invariants (41 `unsafe` hits).

## Final suite (verbatim)

```
test result: ok. 67 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
```

## What CI verifies that this sandbox could not

- **`cargo +nightly miri test`** on GitHub `ubuntu-latest` (nightly/Miri historically
  blocked or flaky in the agent egress sandbox).
- Repeatable **fuzz smoke** on every push to `main`.
- Sanitizers / libFuzzer remain out of scope here (no libc++ / no ASan job yet);
  Miri is the automated UB stand-in wired by this workflow.
