//! Real wall-clock comparison: full scan vs. zone-map pushdown, on two
//! honestly different data distributions. Zone-map pushdown's benefit is
//! NOT distribution-independent — a page whose values are uniformly
//! spread across the whole domain will almost always overlap any
//! reasonable filter threshold's range, so there's little to skip. This
//! binary measures both cases rather than reporting only the favorable
//! one.

use std::time::Instant;
use tamil_query_engine::runtime::{
    execute_mmap_i64_filter_stream_pushdown, PushdownStats, RuntimeScratch,
};
use tamil_query_engine::storage::{write_i64_column_bin, ColumnarFileStream};
use tamil_query_engine::zonemap::{write_zonemap_for_column, ZoneMap, ZonePredicate};

fn bench_one(
    label: &str,
    bin_path: &std::path::Path,
    zmap_path: &std::path::Path,
    predicate: ZonePredicate,
    iters: u32,
) {
    let zm = ZoneMap::load(zmap_path).unwrap();

    let mut scratch = RuntimeScratch::new_boxed();
    let mut stats = PushdownStats::default();

    // Warm the OS page cache identically for both runs before timing.
    {
        let mut s = ColumnarFileStream::open_int64_copied(bin_path).unwrap();
        let mut warm_stats = PushdownStats::default();
        execute_mmap_i64_filter_stream_pushdown(
            &mut s,
            None,
            predicate,
            &mut scratch,
            &mut warm_stats,
        );
    }

    // Full scan (no zone map).
    let mut stream = ColumnarFileStream::open_int64_copied(bin_path).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        execute_mmap_i64_filter_stream_pushdown(
            &mut stream,
            None,
            predicate,
            &mut scratch,
            &mut stats,
        );
    }
    let full_scan_ns = t0.elapsed().as_nanos() / iters as u128;
    let full_scan_stats = stats;

    // Pushdown-enabled.
    let mut stream2 = ColumnarFileStream::open_int64_copied(bin_path).unwrap();
    let t1 = Instant::now();
    for _ in 0..iters {
        execute_mmap_i64_filter_stream_pushdown(
            &mut stream2,
            zm.as_ref(),
            predicate,
            &mut scratch,
            &mut stats,
        );
    }
    let pushdown_ns = t1.elapsed().as_nanos() / iters as u128;
    let pushdown_stats = stats;

    let speedup = full_scan_ns as f64 / pushdown_ns.max(1) as f64;
    println!(
        "{label}: full_scan={full_scan_ns}ns (pages_scanned={}/{}) pushdown={pushdown_ns}ns \
         (pages_scanned={}/{}, skipped={}) speedup={speedup:.3}x",
        full_scan_stats.pages_scanned,
        full_scan_stats.pages_total,
        pushdown_stats.pages_scanned,
        pushdown_stats.pages_total,
        pushdown_stats.pages_skipped,
    );
}

fn main() {
    let dir = std::env::temp_dir().join(format!("tqe_zonemap_bench_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let n = 200_000usize;
    let iters = 20u32;

    // Case 1: CLUSTERED data — values increase monotonically with row
    // index. Pushdown should skip most pages for a selective filter.
    {
        let bin_path = dir.join("clustered.bin");
        let zmap_path = dir.join("clustered.zmap");
        write_i64_column_bin(&bin_path, n, |i| i as i64).unwrap();
        write_zonemap_for_column(&bin_path, &zmap_path).unwrap();
        bench_one(
            "clustered (monotonic id column)",
            &bin_path,
            &zmap_path,
            ZonePredicate::Gt((n as i64) - (n as i64) / 100),
            iters,
        );
    }

    // Case 2: RANDOM data — values uniformly spread 0..1_000_000.
    // Honest unflattering case: little to skip.
    {
        let bin_path = dir.join("random.bin");
        let zmap_path = dir.join("random.zmap");
        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 1_000_000) as i64
        };
        let values: Vec<i64> = (0..n).map(|_| next()).collect();
        write_i64_column_bin(&bin_path, n, |i| values[i]).unwrap();
        write_zonemap_for_column(&bin_path, &zmap_path).unwrap();
        bench_one(
            "random (uniform 0..1_000_000)",
            &bin_path,
            &zmap_path,
            ZonePredicate::Gt(500_000),
            iters,
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}
