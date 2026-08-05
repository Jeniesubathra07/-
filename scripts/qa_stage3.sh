#!/usr/bin/env bash
# Ω-QA-CORE-STRESS-STAGE3 — hardware-deterministic diagnostic runner
# Usage: bash scripts/qa_stage3.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== [1] RELEASE COMPILATION + STRICT TESTS ==="
cargo test --release --lib
cargo test --release --bin tamil_query_engine -- --test-threads=1 2>/dev/null || true

echo "=== [2] ZERO-HEAP HOT-PATH (counting allocator unit tests) ==="
cargo test --release --lib test_derive_math_pipeline_evaluation -- --exact --nocapture
cargo test --release --lib test_parallel_chunk_distribution_integrity -- --exact --nocapture
cargo test --release --lib demo_pipeline_e2e_zero_heap_and_tamil_safe -- --exact --nocapture
cargo test --release --lib omega_qa_stage3_matrix -- --exact --nocapture

echo "=== [3] MICRO-LATENCY BENCHMARKS (ns Instant harness) ==="
cargo run --release --bin stage3_bench
echo "(Criterion omitted: clap_lex edition2024 incompatible with rustc 1.83)"

echo "=== [4] ALLOCATION PROFILING (optional tools) ==="
if command -v valgrind >/dev/null 2>&1; then
  echo "--- massif (heap snapshots) ---"
  cargo build --release
  valgrind --tool=massif --massif-out-file=target/massif.stage3.out \
    ./target/release/tamil_query_engine \
    'இருந்து பயனர்கள் | இணை ஆர்டர்கள் | கணி புதிய_விலை = விலை * 2 | வடி புதிய_விலை > 200;'
  ms_print target/massif.stage3.out | head -n 80
else
  echo "valgrind not installed — skip massif (install: apt install valgrind)"
fi

if command -v heaptrack >/dev/null 2>&1; then
  echo "--- heaptrack ---"
  heaptrack -o target/heaptrack.stage3 ./target/release/tamil_query_engine \
    'இருந்து பயனர்கள் | வடி வயது > 21 | அடுக்கு வயது | எடு 10 | தேடு பெயர், வயது;'
else
  echo "heaptrack not installed — skip"
fi

echo "=== [5] LAYOUT / ALIGN LOCKS ==="
cargo test --release --lib omega_qa_stage3_matrix -- --exact --nocapture

echo "=== QA STAGE3 COMPLETE ==="
