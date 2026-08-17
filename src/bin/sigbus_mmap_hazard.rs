//! SIGBUS hazard demonstration + copy-on-open mitigation proof.
//!
//! Usage:
//!   cargo run --release --bin sigbus_mmap_hazard            # copied (default)
//!   cargo run --release --bin sigbus_mmap_hazard -- --mmap  # may die SIGBUS
//!   cargo run --release --bin sigbus_mmap_hazard -- --bench # print mmap vs copy timings
//!
//! Default mode opens via [`ColumnarFileStream::open_int64_copied`], truncates the
//! backing file, then iterates — must exit 0 with a clean sum (no SIGBUS).
//! `--mmap` retains the historical fatal-signal demonstration.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::process;
use std::hint::black_box;
use std::time::Instant;
use tamil_query_engine::{write_i64_column_bin, ColumnarFileStream};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("--copied");
    match mode {
        "--mmap" => run_mmap_hazard(),
        "--bench" => run_bench(),
        _ => run_copied_safe(),
    }
}

fn run_copied_safe() {
    let dir = std::env::temp_dir().join("tamil_sigbus_copied");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("col.bin");
    write_i64_column_bin(&path, 4096, |i| i as i64).expect("write");

    let mut stream = ColumnarFileStream::open_int64_copied(&path).expect("copy-open");
    assert!(stream.is_copied(), "must be owned backing");

    let mut f = OpenOptions::new().write(true).open(&path).expect("reopen");
    f.set_len(0).expect("truncate");
    f.flush().ok();
    drop(f);

    eprintln!("sigbus_mmap_hazard: copied mode — iterating after truncate…");
    let mut sum = 0i64;
    let mut rows = 0usize;
    while let Some(chunk) = stream.next_page_chunk() {
        rows += chunk.row_count as usize;
        if !chunk.rows.is_empty() {
            sum = sum.wrapping_add(chunk.rows[0]);
        }
    }
    eprintln!("sigbus_mmap_hazard: OK (no SIGBUS); rows={rows} sum={sum}");
    process::exit(0);
}

fn run_mmap_hazard() {
    let dir = std::env::temp_dir().join("tamil_sigbus_hazard");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("col.bin");
    write_i64_column_bin(&path, 4096, |i| i as i64).expect("write");

    let mut stream = ColumnarFileStream::open_i64(&path).expect("mmap");
    let mut f = OpenOptions::new().write(true).open(&path).expect("reopen");
    f.set_len(0).expect("truncate");
    f.flush().ok();
    drop(f);

    eprintln!("sigbus_mmap_hazard: mmap mode — about to iterate after truncate…");
    let mut sum = 0i64;
    while let Some(chunk) = stream.next_page_chunk() {
        if !chunk.rows.is_empty() {
            sum = sum.wrapping_add(chunk.rows[0]);
        }
    }
    eprintln!("sigbus_mmap_hazard: no SIGBUS observed; sum={sum}");
    process::exit(0);
}

fn run_bench() {
    const N: usize = 100_000;
    const ITERS: u32 = 50;
    let dir = std::env::temp_dir().join("tamil_sigbus_bench");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("col.bin");
    write_i64_column_bin(&path, N, |i| i as i64).expect("write");

    let mut mmap_ns = 0u128;
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let mut s = ColumnarFileStream::open_i64(&path).expect("mmap");
        let mut sum = 0i64;
        while let Some(c) = s.next_page_chunk() {
            if !c.rows.is_empty() {
                sum = sum.wrapping_add(c.rows[0]);
            }
        }
        black_box(sum);
        mmap_ns += t0.elapsed().as_nanos();
    }

    let mut copy_ns = 0u128;
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let mut s = ColumnarFileStream::open_int64_copied(&path).expect("copy");
        let mut sum = 0i64;
        while let Some(c) = s.next_page_chunk() {
            if !c.rows.is_empty() {
                sum = sum.wrapping_add(c.rows[0]);
            }
        }
        black_box(sum);
        copy_ns += t0.elapsed().as_nanos();
    }

    println!(
        "open+scan {} rows × {}: mmap_avg_ns={} copy_avg_ns={} copy/mmap={:.2}x",
        N,
        ITERS,
        mmap_ns / ITERS as u128,
        copy_ns / ITERS as u128,
        (copy_ns as f64) / (mmap_ns as f64)
    );
}
