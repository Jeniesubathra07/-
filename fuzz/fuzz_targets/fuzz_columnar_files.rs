#![no_main]
use libfuzzer_sys::fuzz_target;
use std::io::Write;
use tamil_query_engine::{ColumnarFileStream, Utf8ColumnFile};

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }
    let dir = std::env::temp_dir().join(format!("fuzz_col_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let bin = dir.join("c.bin");
    let meta = dir.join("c.meta");
    let off = dir.join("c.offsets");
    let blob = dir.join("c.blob");
    let umeta = dir.join("c.umeta");

    // Split input: int64 bytes + utf8 offsets/blob noise.
    let mid = data.len() / 2;
    let (a, b) = data.split_at(mid);
    // Force length multiple of 8 for open validation path coverage.
    let mut a_aligned = a.to_vec();
    while a_aligned.len() % 8 != 0 {
        a_aligned.push(0);
    }
    if let Ok(mut f) = std::fs::File::create(&bin) {
        let _ = f.write_all(&a_aligned);
    }
    let rows = (a_aligned.len() / 8) as u64;
    if let Ok(mut f) = std::fs::File::create(&meta) {
        let _ = f.write_all(&rows.to_le_bytes());
    }
    let _ = ColumnarFileStream::open_i64_with_meta(&bin, &meta).map(|mut s| {
        let mut n = 0usize;
        while let Some(c) = s.next_page_chunk() {
            n = n.wrapping_add(c.row_count as usize);
            if n > 1_000_000 {
                break;
            }
        }
    });

    // Corrupt utf8 pair files.
    if let Ok(mut f) = std::fs::File::create(&off) {
        let _ = f.write_all(b);
    }
    if let Ok(mut f) = std::fs::File::create(&blob) {
        let _ = f.write_all(a);
    }
    let blob_len = a.len() as u64;
    let row_guess = (b.len() / 8) as u64;
    if let Ok(mut f) = std::fs::File::create(&umeta) {
        let _ = f.write_all(&row_guess.to_le_bytes());
        let _ = f.write_all(&blob_len.to_le_bytes());
    }
    if let Ok(file) = Utf8ColumnFile::open(&off, &blob, Some(&umeta)) {
        let mut i = 0usize;
        while i < file.total_rows().min(256) {
            let _ = file.get_row(i);
            i += 1;
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
});
