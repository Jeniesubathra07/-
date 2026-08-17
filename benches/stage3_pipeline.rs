//! Stage-3 micro-latency bench harness (nanosecond `Instant` timers).
//!
//! Zero Criterion dependency (Cargo 1.83 / edition2024 gate).
//! Run: `cargo run --release --bin stage3_bench`

use std::time::Instant;
use tamil_query_engine::{
    alloc_token_window, demo_catalog, execute_chunk_parallel, execute_chunk_parallel_os,
    lsd_radix_sort_ages, run_query, ArithOp, AstArena, QueryResult, RuntimeScratch, BATCH_ROWS,
    DEMO_QUERY, MAX_ROWS,
};

const DERIVE_QUERY: &str =
    "இருந்து பயனர்கள் | இணை ஆர்டர்கள் | கணி புதிய_விலை = விலை * 2 | வடி புதிய_விலை > 200;";
const ITERS: u32 = 2_000;

#[inline(never)]
fn bench_ns(name: &str, iters: u32, mut body: impl FnMut()) {
    // Warmup
    let mut w = 0u32;
    while w < 64 {
        body();
        w += 1;
    }
    let t0 = Instant::now();
    let mut i = 0u32;
    while i < iters {
        body();
        i += 1;
    }
    let elapsed = t0.elapsed();
    let ns = elapsed.as_nanos() / u128::from(iters);
    println!("{name:>36}  {ns:>10} ns/iter  ({iters} iters)");
}

fn main() {
    println!("=== Ω-QA Stage-3 micro-latency (release Instant) ===");

    let catalog = demo_catalog();
    let mut arena = Box::new(AstArena::new());
    let mut out = QueryResult::new_boxed();
    let mut scratch = RuntimeScratch::new_boxed();
    let mut tokens = alloc_token_window();

    bench_ns("demo_pipeline_e2e", ITERS, || {
        let ok = run_query(
            DEMO_QUERY,
            &catalog,
            &mut arena,
            &mut out,
            &mut scratch,
            &mut tokens,
        );
        core::hint::black_box(ok.is_ok());
        core::hint::black_box(out.row_count);
    });

    bench_ns("derive_kani_join_filter", ITERS, || {
        let ok = run_query(
            DERIVE_QUERY,
            &catalog,
            &mut arena,
            &mut out,
            &mut scratch,
            &mut tokens,
        );
        core::hint::black_box(ok.is_ok());
        core::hint::black_box(out.row_count);
    });

    let mut src = RuntimeScratch::new_boxed();
    let mut dst = RuntimeScratch::new_boxed();
    let mut i = 0usize;
    while i < MAX_ROWS {
        src.key_buf[i] = i as i64;
        i += 1;
    }

    for &n in &[64usize, BATCH_ROWS, 2050, MAX_ROWS] {
        let label = format!("chunk_tls_mul2_n={n}");
        bench_ns(&label, ITERS, || {
            execute_chunk_parallel(&src.key_buf, &mut dst.derived, n, ArithOp::Mul, 2);
            core::hint::black_box(dst.derived[0]);
        });
    }

    // OS-thread path (may allocate JoinHandles — bench only).
    bench_ns("chunk_os_mul2_n=2050", 200, || {
        execute_chunk_parallel_os(&src.key_buf, &mut dst.derived, 2050, ArithOp::Mul, 2);
        core::hint::black_box(dst.derived[2049]);
    });

    let mut j = 0usize;
    while j < 2050 {
        src.key_buf[j] = (2050 - j) as i64;
        src.order[j] = j as u16;
        j += 1;
    }
    bench_ns("lsd_radix_sort_ages_2050", ITERS, || {
        let mut k = 0usize;
        while k < 2050 {
            src.order[k] = k as u16;
            k += 1;
        }
        lsd_radix_sort_ages(&src.key_buf, &mut src.order, 2050, &mut src.tmp_u16);
        core::hint::black_box(src.order[0]);
    });

    println!("=== done ===");
}
