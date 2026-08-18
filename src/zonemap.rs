//! Phase 2: Int64 zone maps (`.zmap`) for page-level predicate pushdown.
//!
//! One sidecar file per Int64 column: `<column>.zmap` next to `.bin`/`.meta`.
//! Each record is one OS-page window of the columnar stream (`page_rows =
//! os_page_size / 8`). Zone maps are derived from already-published `.bin`
//! files via [`ColumnarFileStream`] — never by re-reading the source CSV.
//!
//! Hot-path lookups index a small mmap (or owned copy) loaded once at open;
//! they allocate no heap. Utf8 columns are intentionally out of scope.

use crate::storage::{os_page_size_bytes, ColumnarFileStream};
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

/// Maximum zone-map pages addressable in one file (covers multi-megarow
/// Int64 columns at typical 512 rows/OS-page). Cold-path open rejects more.
pub const MAX_ZMAP_PAGES: usize = 8192;

/// One page's value range — fixed layout, cache-line aligned.
///
/// On-disk and in-memory representation are identical (little-endian host).
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ZoneMapEntry {
    pub page_index: u32,
    pub row_count: u32,
    pub min: i64,
    pub max: i64,
    pub _pad: [u8; 40],
}

const _: () = assert!(core::mem::size_of::<ZoneMapEntry>() == 64);
const _: () = assert!(core::mem::align_of::<ZoneMapEntry>() == 64);

impl ZoneMapEntry {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            page_index: 0,
            row_count: 0,
            min: 0,
            max: 0,
            _pad: [0; 40],
        }
    }
}

/// Comparison predicates supported for zone-map pushdown.
///
/// Matches the Tamil DSL filter grammar today: lexer emits only `>`, `<`,
/// `=` (`TokenKind::{Gt,Lt,Eq}`). `Gte`/`Lte`/`Ne` are included for the
/// zone-map API so boundary tests can exercise exact min/max semantics
/// without inventing lexer tokens the parser does not produce.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ZoneCmp {
    Gt = 0,
    Lt = 1,
    Eq = 2,
    Gte = 3,
    Lte = 4,
    Ne = 5,
}

/// Return `true` if any value in `[min, max]` could satisfy `op lit`.
///
/// When this returns `false`, the page may be skipped without decoding.
#[inline(always)]
pub fn page_can_satisfy(min: i64, max: i64, op: ZoneCmp, lit: i64) -> bool {
    match op {
        ZoneCmp::Gt => max > lit,
        ZoneCmp::Lt => min < lit,
        ZoneCmp::Eq => lit >= min && lit <= max,
        ZoneCmp::Gte => max >= lit,
        ZoneCmp::Lte => min <= lit,
        // Skip only when every value on the page equals `lit`.
        ZoneCmp::Ne => !(min == max && min == lit),
    }
}

impl ZoneMapEntry {
    #[inline(always)]
    pub fn can_satisfy(&self, op: ZoneCmp, lit: i64) -> bool {
        if self.row_count == 0 {
            return false;
        }
        page_can_satisfy(self.min, self.max, op, lit)
    }
}

/// Loaded `.zmap` — cold-path open; hot path is [`ZoneMap::entry`] index.
pub struct ZoneMap {
    bytes: ZoneMapBytes,
    page_count: u32,
}

enum ZoneMapBytes {
    Mmap(Mmap),
    Owned(Vec<ZoneMapEntry>),
}

impl ZoneMap {
    /// Cold-path: mmap a `.zmap` written by [`write_zonemap_for_column`].
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        let ent = core::mem::size_of::<ZoneMapEntry>();
        if len % ent != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zonemap length not multiple of ZoneMapEntry",
            ));
        }
        let n = len / ent;
        if n > MAX_ZMAP_PAGES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zonemap exceeds MAX_ZMAP_PAGES",
            ));
        }
        // SAFETY: read-only open; length validated. SIGBUS if file truncated
        // while mapped (same contract as ColumnarFileStream mmap opens).
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self {
            bytes: ZoneMapBytes::Mmap(mmap),
            page_count: n as u32,
        })
    }

    /// Crash-proof cold open: copy entries into an owned `Vec` (no SIGBUS).
    pub fn open_copied(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        let ent = core::mem::size_of::<ZoneMapEntry>();
        if len % ent != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zonemap length not multiple of ZoneMapEntry",
            ));
        }
        let n = len / ent;
        if n > MAX_ZMAP_PAGES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "zonemap exceeds MAX_ZMAP_PAGES",
            ));
        }
        let mut entries = vec![ZoneMapEntry::empty(); n];
        let dst = unsafe {
            core::slice::from_raw_parts_mut(entries.as_mut_ptr() as *mut u8, len)
        };
        let mut f = file;
        use std::io::Read;
        f.read_exact(dst)?;
        Ok(Self {
            bytes: ZoneMapBytes::Owned(entries),
            page_count: n as u32,
        })
    }

    #[inline(always)]
    pub fn page_count(&self) -> u32 {
        self.page_count
    }

    #[inline(always)]
    pub fn entry(&self, i: usize) -> Option<&ZoneMapEntry> {
        if i >= self.page_count as usize {
            return None;
        }
        match &self.bytes {
            ZoneMapBytes::Owned(v) => Some(&v[i]),
            ZoneMapBytes::Mmap(m) => {
                let off = i.wrapping_mul(64);
                let end = off.wrapping_add(64);
                if end > m.len() {
                    return None;
                }
                // SAFETY: `off` is multiple of 64; mmap base page-aligned;
                // length covers one ZoneMapEntry.
                let p = m[off..end].as_ptr() as *const ZoneMapEntry;
                Some(unsafe { &*p })
            }
        }
    }
}

/// Stream an Int64 `.bin` (+ `.meta`) and write a matching `.zmap`.
///
/// Uses [`ColumnarFileStream`] page geometry so zone pages align with
/// query-time `next_page_chunk` windows. Publishes via tmp-then-rename.
pub fn write_zonemap_for_column(
    bin_path: &Path,
    meta_path: &Path,
    zmap_path: &Path,
) -> io::Result<()> {
    let mut stream = ColumnarFileStream::open_i64_with_meta(bin_path, meta_path)?;
    let tmp = zmap_path.with_extension("zmap.tmp");
    let result = (|| -> io::Result<u32> {
        let mut file = File::create(&tmp)?;
        let mut page_index: u32 = 0;
        while let Some(chunk) = stream.next_page_chunk() {
            let n = chunk.row_count as usize;
            if n == 0 {
                continue;
            }
            if page_index as usize >= MAX_ZMAP_PAGES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "column has more pages than MAX_ZMAP_PAGES",
                ));
            }
            let mut mn = i64::MAX;
            let mut mx = i64::MIN;
            let mut i = 0usize;
            while i < n {
                let v = chunk.rows[i];
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
                i += 1;
            }
            let entry = ZoneMapEntry {
                page_index,
                row_count: n as u32,
                min: mn,
                max: mx,
                _pad: [0; 40],
            };
            // SAFETY: ZoneMapEntry is POD; write exact 64 LE bytes.
            let bytes = unsafe {
                core::slice::from_raw_parts(
                    (&entry as *const ZoneMapEntry) as *const u8,
                    core::mem::size_of::<ZoneMapEntry>(),
                )
            };
            file.write_all(bytes)?;
            page_index = page_index.wrapping_add(1);
        }
        file.flush()?;
        Ok(page_index)
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, zmap_path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    let _ = os_page_size_bytes(); // keep page geometry coupled to storage
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::write_i64_column_bin;

    #[test]
    fn page_can_satisfy_boundaries() {
        // max == lit → Gt must NOT satisfy; Gte must.
        assert!(!page_can_satisfy(1, 10, ZoneCmp::Gt, 10));
        assert!(page_can_satisfy(1, 10, ZoneCmp::Gte, 10));
        // min == lit → Lt must NOT; Lte must.
        assert!(!page_can_satisfy(1, 10, ZoneCmp::Lt, 1));
        assert!(page_can_satisfy(1, 10, ZoneCmp::Lte, 1));
        assert!(page_can_satisfy(1, 10, ZoneCmp::Eq, 1));
        assert!(page_can_satisfy(1, 10, ZoneCmp::Eq, 10));
        assert!(!page_can_satisfy(1, 10, ZoneCmp::Eq, 11));
    }

    #[test]
    #[cfg_attr(miri, ignore = "file I/O")]
    fn write_and_open_zonemap_smoke() {
        let dir = std::env::temp_dir().join("tqe_zmap_smoke");
        let _ = std::fs::create_dir_all(&dir);
        let bin = dir.join("c.bin");
        write_i64_column_bin(&bin, 100, |i| i as i64).unwrap();
        let meta = bin.with_extension("meta");
        let zmap = bin.with_extension("zmap");
        write_zonemap_for_column(&bin, &meta, &zmap).unwrap();
        let zm = ZoneMap::open(&zmap).unwrap();
        assert!(zm.page_count() >= 1);
        let e0 = zm.entry(0).unwrap();
        assert_eq!(e0.page_index, 0);
        assert!(e0.row_count > 0);
        assert!(e0.min <= e0.max);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
