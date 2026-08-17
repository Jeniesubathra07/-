//! Subprocess harness: mmap a column file, truncate it, then touch mapped memory.
//!
//! Documents the SIGBUS / fatal signal hazard when the single-writer-absent
//! snapshot precondition is violated. Run:
//!   cargo run --release --bin sigbus_mmap_hazard
//! Expectation: process aborts with SIGBUS (or equivalent fatal signal),
//! demonstrating that this is NOT a recoverable Rust Result::Err.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::process;
use tamil_query_engine::{write_i64_column_bin, ColumnarFileStream};

fn main() {
    let dir = std::env::temp_dir().join("tamil_sigbus_hazard");
    let _ = fs::create_dir_all(&dir);
    let path = dir.join("col.bin");
    write_i64_column_bin(&path, 4096, |i| i as i64).expect("write");

    let mut stream = ColumnarFileStream::open_i64(&path).expect("mmap");
    // Truncate while mapping is live — violates the documented precondition.
    let mut f = OpenOptions::new().write(true).open(&path).expect("reopen");
    f.set_len(0).expect("truncate");
    f.flush().ok();
    drop(f);

    // Touch mapped pages — POSIX may deliver SIGBUS here.
    eprintln!("sigbus_mmap_hazard: about to iterate after truncate…");
    let mut sum = 0i64;
    while let Some(chunk) = stream.next_page_chunk() {
        if !chunk.rows.is_empty() {
            sum = sum.wrapping_add(chunk.rows[0]);
        }
    }
    // If we reach here, the OS did not SIGBUS (some FS/kernels may COW).
    eprintln!("sigbus_mmap_hazard: no SIGBUS observed; sum={sum}");
    process::exit(0);
}
