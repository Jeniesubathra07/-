//! Stage-3 / Phase-2 micro-latency bench harness (nanosecond `Instant` timers).
//!
//! Zero Criterion dependency (Cargo 1.83 / edition2024 gate).
//! Run: `cargo run --release --bin stage3_bench`

use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tamil_query_engine::{
    alloc_token_window, demo_catalog, execute_chunk_parallel, execute_chunk_parallel_os,
    execute_int64_filter_pushdown, ingest, lsd_radix_sort_ages, run_query, ArithOp, AstArena,
    ColumnarFileStream, PushdownStats, QueryResult, RuntimeScratch, ZoneCmp, ZoneMap, BATCH_ROWS,
    DEMO_QUERY, MAX_ROWS,
};

const DERIVE_QUERY: &str =
    "இருந்து பயனர்கள் | இணை ஆர்டர்கள் | கணி புதிய_விலை = விலை * 2 | வடி புதிய_விலை > 200;";
const ITERS: u32 = 2_000;
const PUSHDOWN_ITERS: u32 = 200;
const PUSHDOWN_ROWS: usize = 100_000;

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

fn prepare_pushdown_dataset() -> PathBuf {
    let dir = std::env::temp_dir().join("tqe_phase2_bench_pushdown");
    let _ = fs::create_dir_all(&dir);
    let csv_path = dir.join("vals.csv");
    let out_dir = dir.join("out");
    // Half the rows are low (0..999), half are high (1_000_000+) so a
    // predicate `v > 500_000` skips ~half the pages.
    let mut csv = String::from("v\n");
    let half = PUSHDOWN_ROWS / 2;
    let mut i = 0usize;
    while i < half {
        csv.push_str(&format!("{}\n", (i % 1000) as i64));
        i += 1;
    }
    let mut i = 0usize;
    while i < half {
        csv.push_str(&format!("{}\n", 1_000_000i64 + i as i64));
        i += 1;
    }
    fs::write(&csv_path, &csv).expect("write csv");
    let schema = ingest::parse_schema("v:i64").expect("schema");
    ingest::ingest_csv(&csv_path, &schema, &out_dir, true).expect("ingest");
    out_dir
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

    println!("=== Phase-2 zonemap pushdown (100k rows, release Instant) ===");
    let out_dir = prepare_pushdown_dataset();
    let bin = out_dir.join("v.bin");
    let meta = bin.with_extension("meta");
    let zmap_path = bin.with_extension("zmap");
    let mut stream = ColumnarFileStream::open_i64_with_meta(&bin, &meta).expect("open");
    let zmap = ZoneMap::open(&zmap_path).expect("zmap");
    let mut scratch2 = RuntimeScratch::new_boxed();
    let mut out_vals = vec![0i64; MAX_ROWS];
    let mut out_len = 0usize;
    let mut stats = PushdownStats::default();

    execute_int64_filter_pushdown(
        &mut stream,
        Some(&zmap),
        ZoneCmp::Gt,
        500_000,
        true,
        &mut scratch2,
        &mut out_vals,
        &mut out_len,
        &mut stats,
    );
    println!(
        "pushdown probe: pages_total={} pages_skipped={} pages_scanned={} kept_capped={}",
        stats.pages_total, stats.pages_skipped, stats.pages_scanned, out_len
    );

    let mut ns_on = 0u128;
    let mut ns_off = 0u128;
    for _ in 0..8 {
        stream.rewind();
        out_len = 0;
        execute_int64_filter_pushdown(
            &mut stream,
            Some(&zmap),
            ZoneCmp::Gt,
            500_000,
            true,
            &mut scratch2,
            &mut out_vals,
            &mut out_len,
            &mut stats,
        );
    }
    for _ in 0..PUSHDOWN_ITERS {
        stream.rewind();
        out_len = 0;
        let t0 = Instant::now();
        execute_int64_filter_pushdown(
            &mut stream,
            Some(&zmap),
            ZoneCmp::Gt,
            500_000,
            true,
            &mut scratch2,
            &mut out_vals,
            &mut out_len,
            &mut stats,
        );
        ns_on += t0.elapsed().as_nanos();
        core::hint::black_box(out_len);
    }
    for _ in 0..PUSHDOWN_ITERS {
        stream.rewind();
        out_len = 0;
        let t0 = Instant::now();
        execute_int64_filter_pushdown(
            &mut stream,
            Some(&zmap),
            ZoneCmp::Gt,
            500_000,
            false,
            &mut scratch2,
            &mut out_vals,
            &mut out_len,
            &mut stats,
        );
        ns_off += t0.elapsed().as_nanos();
        core::hint::black_box(out_len);
    }
    let avg_on = ns_on / u128::from(PUSHDOWN_ITERS);
    let avg_off = ns_off / u128::from(PUSHDOWN_ITERS);
    let ratio = if avg_on == 0 {
        0.0
    } else {
        avg_off as f64 / avg_on as f64
    };
    println!(
        "{:>36}  {:>10} ns/iter  ({} iters)",
        "filter_pushdown_ON", avg_on, PUSHDOWN_ITERS
    );
    println!(
        "{:>36}  {:>10} ns/iter  ({} iters)",
        "filter_pushdown_OFF", avg_off, PUSHDOWN_ITERS
    );
    println!("speedup OFF/ON = {ratio:.3}x  (values >1 mean pushdown is faster)");

    println!("=== done ===");
}
