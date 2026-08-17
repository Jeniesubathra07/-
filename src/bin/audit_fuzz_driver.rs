//! Pure-Rust mutation fuzz driver (no libfuzzer / C++ runtime required).
//!
//! Used when `cargo fuzz` cannot link libFuzzer (`cassert` / libc++ missing).
//! Run:
//!   cargo run --release --bin audit_fuzz_driver -- 120
//! Arg = wall-clock seconds per target (query + columnar).

use std::env;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tamil_query_engine::{
    alloc_token_window, demo_catalog, run_query, write_i64_column_bin, AstArena,
    ColumnarFileStream, QueryResult, RuntimeScratch, Utf8ColumnFile,
};

fn mutate(seed: &mut u64, buf: &mut Vec<u8>) {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let r = *seed;
    match (r >> 8) % 5 {
        0 => {
            if !buf.is_empty() {
                let i = (r as usize) % buf.len();
                buf[i] = (r >> 16) as u8;
            }
        }
        1 => buf.push((r >> 24) as u8),
        2 => {
            if !buf.is_empty() {
                buf.pop();
            }
        }
        3 => {
            let n = ((r >> 16) % 17) as usize;
            buf.extend(std::iter::repeat((r >> 8) as u8).take(n));
        }
        _ => {
            buf.clear();
            buf.extend_from_slice("இருந்து பயனர்கள் | வடி வயது > 21 | தேடு வயது;".as_bytes());
            let i = (r as usize) % (buf.len().max(1));
            if !buf.is_empty() {
                buf[i] ^= 0x20;
            }
        }
    }
    if buf.len() > 4096 {
        buf.truncate(4096);
    }
}

fn fuzz_queries(seconds: u64) -> u64 {
    let catalog = demo_catalog();
    let mut arena = Box::new(AstArena::new());
    let mut out = QueryResult::new_boxed();
    let mut scratch = RuntimeScratch::new_boxed();
    let mut tokens = alloc_token_window();
    let mut buf = "இருந்து பயனர்கள் | வடி வயது > 21 | தேடு வயது;".as_bytes().to_vec();
    let mut seed = 0xC0FFEEu64;
    let mut iters = 0u64;
    let end = Instant::now() + Duration::from_secs(seconds);
    while Instant::now() < end {
        mutate(&mut seed, &mut buf);
        if let Ok(src) = std::str::from_utf8(&buf) {
            let _ = run_query(src, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens);
        }
        iters += 1;
    }
    iters
}

fn fuzz_columnar(seconds: u64) -> u64 {
    let dir = std::env::temp_dir().join("tamil_audit_fuzz_col");
    let _ = std::fs::create_dir_all(&dir);
    let mut seed = 0xDEAD_BEEFu64;
    let mut buf = vec![0u8; 64];
    let mut iters = 0u64;
    let end = Instant::now() + Duration::from_secs(seconds);
    while Instant::now() < end {
        mutate(&mut seed, &mut buf);
        let bin = dir.join("c.bin");
        let meta = dir.join("c.meta");
        let mut aligned = buf.clone();
        while aligned.len() % 8 != 0 {
            aligned.push(0);
        }
        if let Ok(mut f) = std::fs::File::create(&bin) {
            let _ = f.write_all(&aligned);
        }
        let rows = (aligned.len() / 8) as u64;
        if let Ok(mut f) = std::fs::File::create(&meta) {
            let _ = f.write_all(&rows.to_le_bytes());
        }
        if let Ok(mut s) = ColumnarFileStream::open_i64_with_meta(&bin, &meta) {
            let mut n = 0usize;
            while let Some(c) = s.next_page_chunk() {
                n = n.wrapping_add(c.row_count as usize);
                if n > 100_000 {
                    break;
                }
            }
        }
        // Utf8 corrupt pair
        let off = dir.join("o.offsets");
        let blob = dir.join("o.blob");
        let um = dir.join("o.meta");
        if let Ok(mut f) = std::fs::File::create(&off) {
            let _ = f.write_all(&buf);
        }
        if let Ok(mut f) = std::fs::File::create(&blob) {
            let _ = f.write_all(&aligned);
        }
        let blob_len = aligned.len() as u64;
        let row_guess = (buf.len() / 8) as u64;
        if let Ok(mut f) = std::fs::File::create(&um) {
            let _ = f.write_all(&row_guess.to_le_bytes());
            let _ = f.write_all(&blob_len.to_le_bytes());
        }
        if let Ok(file) = Utf8ColumnFile::open(&off, &blob, Some(&um)) {
            let mut i = 0usize;
            while i < file.total_rows().min(64) {
                let _ = file.get_row(i);
                i += 1;
            }
        }
        iters += 1;
    }
    let _ = write_i64_column_bin(&dir.join("ok.bin"), 16, |i| i as i64);
    let _ = PathBuf::from(&dir);
    iters
}

fn main() {
    let secs: u64 = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    eprintln!("audit_fuzz_driver: {secs}s per target");
    let t0 = Instant::now();
    let q = fuzz_queries(secs);
    let t1 = Instant::now();
    let c = fuzz_columnar(secs);
    let t2 = Instant::now();
    println!(
        "fuzz_run_query iters={q} wall_ms={}",
        (t1 - t0).as_millis()
    );
    println!(
        "fuzz_columnar_files iters={c} wall_ms={}",
        (t2 - t1).as_millis()
    );
    println!("total_wall_ms={}", (t2 - t0).as_millis());
}
