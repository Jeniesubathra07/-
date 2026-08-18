//! Phase 2 of the production roadmap: zone-map (min/max page statistics)
//! predicate pushdown.
//!
//! Phase 1's ingest pipeline proved the engine can stream data past
//! `MAX_ROWS` for the first time (12,000-row synthetic dataset, 3 pages).
//! Every `வடி` filter still scans every mapped page in full — there was
//! no mechanism to skip a page based on its value range. This module adds
//! that mechanism for Int64 columns.
//!
//! Scope, stated explicitly: only comparison predicates on Int64 columns
//! are covered, because those are the only predicate kind `வடி` actually
//! supports today (`TokenKind::{Gt, Lt, Eq}` — confirmed against the
//! lexer, not assumed; there is no `>=`/`<=`/`!=` token in this grammar).
//! Utf8 columns have no zone map — there is no meaningful numeric min/max
//! for text, and building one is out of scope for this phase.

use crate::storage::ColumnarFileStream;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

/// One page's min/max/row-count record. `#[repr(C, align(64))]` to match
/// this crate's existing convention for on-disk fixed-layout structs
/// (e.g. `Int64ColumnMeta`), and so a whole page-worth of records is
/// cheap to memcpy into a fixed on-stack array at query-open time with no
/// heap allocation.
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ZonePage {
    pub page_index: u32,
    pub row_count: u32,
    pub min: i64,
    pub max: i64,
}

const ZONE_PAGE_BYTES: usize = 24; // 4 + 4 + 8 + 8, packed on disk (no padding written)

/// Maximum pages a zone map can describe while still being readable into
/// a fixed on-stack array with zero heap allocation at query time. Page
/// granularity is the real OS page size (bytes-per-page / 8 for Int64 —
/// 512 rows/page on a 4096-byte-page system, NOT `MAX_ROWS`; these are
/// distinct concepts by original Stage 4 design), so 1024 pages covers
/// roughly 512K rows per column on a typical system — comfortably beyond
/// Phase 1's proven 12,000-row dataset (which itself spans 24 such pages,
/// not 3 — confirmed by running the test, not assumed). A column with
/// more pages than this simply cannot use pushdown (falls back to full
/// scan, exactly like a column with no `.zmap` at all) rather than
/// silently truncating its zone map.
pub const MAX_ZONE_PAGES: usize = 1024;

/// Zone-map read/write failure modes, named and specific rather than a
/// bare `io::Error` — mirrors `IngestError`'s discipline.
#[derive(Debug)]
pub enum ZoneMapError {
    Io(io::Error),
    /// `.zmap` file length is not a whole multiple of the fixed record
    /// size — cannot be a valid zone map for this format.
    Truncated { file_len: u64 },
    /// The zone map describes more pages than [`MAX_ZONE_PAGES`] can
    /// hold in the fixed on-stack load buffer.
    TooManyPages { pages: usize },
}

impl From<io::Error> for ZoneMapError {
    fn from(e: io::Error) -> Self {
        ZoneMapError::Io(e)
    }
}

/// Compute and write a `.zmap` sidecar for an already-published Int64
/// column file, by streaming it page-by-page through the real
/// [`ColumnarFileStream`] API — not by re-reading the original source
/// CSV, keeping this decoupled from `ingest.rs`. Cold path: freely uses
/// `Vec` to accumulate the (small, bounded by page count) records before
/// one bulk write.
pub fn write_zonemap_for_column(bin_path: &Path, zmap_path: &Path) -> Result<usize, ZoneMapError> {
    let mut stream = ColumnarFileStream::open_int64_copied(bin_path)?;
    let mut records: Vec<ZonePage> = Vec::new();

    loop {
        let page = match stream.next_page_chunk() {
            Some(p) => p,
            None => break,
        };
        let n = page.row_count as usize;
        if n == 0 {
            continue;
        }
        // First value seeds min/max; checked scan over the rest. Using
        // the same "never silently produce a wrong answer from
        // untrusted/edge-case data" discipline as the lexer's number
        // parsing and ingest's numeric validation — min/max computation
        // over i64::MIN/i64::MAX-adjacent values cannot overflow (min/max
        // comparison, not arithmetic), so there is no wraparound risk
        // here, but the accumulation is still written out explicitly
        // rather than relying on an iterator adapter, to keep this
        // auditable at a glance.
        let mut min = page.rows[0];
        let mut max = page.rows[0];
        let mut i = 1usize;
        while i < n {
            let v = page.rows[i];
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            i += 1;
        }
        records.push(ZonePage {
            page_index: page.page_index,
            row_count: page.row_count as u32,
            min,
            max,
        });
    }

    if records.len() > MAX_ZONE_PAGES {
        return Err(ZoneMapError::TooManyPages {
            pages: records.len(),
        });
    }

    let tmp_path = zmap_path.with_extension("zmap.tmp");
    {
        let mut f = File::create(&tmp_path)?;
        for r in &records {
            let mut buf = [0u8; ZONE_PAGE_BYTES];
            buf[0..4].copy_from_slice(&r.page_index.to_le_bytes());
            buf[4..8].copy_from_slice(&r.row_count.to_le_bytes());
            buf[8..16].copy_from_slice(&r.min.to_le_bytes());
            buf[16..24].copy_from_slice(&r.max.to_le_bytes());
            f.write_all(&buf)?;
        }
        f.flush()?;
    }
    std::fs::rename(&tmp_path, zmap_path)?;
    Ok(records.len())
}

/// A zone map loaded into a fixed-capacity on-stack array — zero heap
/// allocation, matching this crate's hot-path discipline. Query code
/// loads this once per filtered column at query start (cold path), then
/// consults it per page during the hot streaming loop with no further
/// I/O or allocation.
#[repr(C, align(64))]
#[derive(Copy, Clone)]
pub struct ZoneMap {
    pages: [ZonePage; MAX_ZONE_PAGES],
    len: usize,
}

impl ZoneMap {
    /// Load a `.zmap` file fully into a fixed on-stack array. Returns
    /// `Ok(None)` (not an error) when `zmap_path` does not exist — the
    /// documented, tested fallback for columns published before Phase 2,
    /// or Utf8 columns, which never have a zone map. Any OTHER failure
    /// (truncated file, unreadable, malformed) is a real
    /// [`ZoneMapError`], since a `.zmap` file that exists but is corrupt
    /// must not be silently ignored the same way a missing one is —
    /// silently falling back on a corrupt-but-present file would hide
    /// real data corruption behind what looks like a routine cache miss.
    pub fn load(zmap_path: &Path) -> Result<Option<Self>, ZoneMapError> {
        let mut f = match File::open(zmap_path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let file_len = f.metadata()?.len();
        if file_len as usize % ZONE_PAGE_BYTES != 0 {
            return Err(ZoneMapError::Truncated { file_len });
        }
        let n = file_len as usize / ZONE_PAGE_BYTES;
        if n > MAX_ZONE_PAGES {
            return Err(ZoneMapError::TooManyPages { pages: n });
        }

        let mut zm = ZoneMap {
            pages: [ZonePage {
                page_index: 0,
                row_count: 0,
                min: 0,
                max: 0,
            }; MAX_ZONE_PAGES],
            len: n,
        };
        let mut buf = [0u8; ZONE_PAGE_BYTES];
        let mut i = 0usize;
        while i < n {
            f.read_exact(&mut buf)?;
            zm.pages[i] = ZonePage {
                page_index: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
                row_count: u32::from_le_bytes(buf[4..8].try_into().unwrap()),
                min: i64::from_le_bytes(buf[8..16].try_into().unwrap()),
                max: i64::from_le_bytes(buf[16..24].try_into().unwrap()),
            };
            i += 1;
        }
        Ok(Some(zm))
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn page(&self, i: usize) -> Option<&ZonePage> {
        if i < self.len {
            Some(&self.pages[i])
        } else {
            None
        }
    }
}

/// The three comparison predicates `வடி` actually supports today
/// (confirmed against `TokenKind`/`CmpOp` in `runtime.rs` — there is no
/// `>=`/`<=`/`!=` token in this grammar, so this is not an
/// under-implementation, it's the true scope).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ZonePredicate {
    Gt(i64),
    Lt(i64),
    Eq(i64),
}

impl ZonePredicate {
    /// Returns `true` if this page's `[min, max]` range could contain a
    /// value satisfying the predicate — i.e. the page must be scanned.
    /// Returns `false` only when it is IMPOSSIBLE for any value in
    /// `[min, max]` to satisfy the predicate — i.e. the page can be
    /// skipped outright. Deliberately conservative: a page is only ever
    /// skipped when correctness is provable from the range alone, never
    /// as a heuristic.
    #[inline(always)]
    pub fn page_may_match(&self, page: &ZonePage) -> bool {
        match *self {
            // v > lit is satisfiable somewhere in [min,max] unless the
            // entire range is <= lit. Boundary: max == lit does NOT
            // satisfy v > lit, so max == lit still means "cannot match" —
            // correctly excluded by `page.max > lit`.
            ZonePredicate::Gt(lit) => page.max > lit,
            // Symmetric: v < lit is satisfiable unless the entire range
            // is >= lit. min == lit does NOT satisfy v < lit.
            ZonePredicate::Lt(lit) => page.min < lit,
            // v == lit is satisfiable only if lit falls within [min,max]
            // inclusive — both boundary values are valid equality matches.
            ZonePredicate::Eq(lit) => lit >= page.min && lit <= page.max,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::write_i64_column_bin;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("tqe_zonemap_{pid}_{nanos}_{name}"));
        p
    }

    #[test]
    fn zonemap_written_correctly_for_multipage_column() {
        // Same style as Phase 1's 12,000-row hand-computed-sum test:
        // large enough to span multiple OS pages, with per-page min/max
        // independently computed from the raw generated values.
        let n = 12_000usize;
        let dir = tmp_path("multipage");
        std::fs::create_dir_all(&dir).unwrap();
        let bin_path = dir.join("val.bin");
        let zmap_path = dir.join("val.zmap");

        let values: Vec<i64> = (0..n as i64)
            .map(|i| ((i * 2654435761i64) % 1_000_000).abs())
            .collect();
        write_i64_column_bin(&bin_path, n, |i| values[i]).unwrap();

        let page_count = write_zonemap_for_column(&bin_path, &zmap_path).unwrap();

        let zm = ZoneMap::load(&zmap_path).unwrap().expect("zmap must exist");
        assert_eq!(zm.len(), page_count);

        let probe = ColumnarFileStream::open_int64_copied(&bin_path).unwrap();
        let page_rows = probe.page_rows();
        let expected_pages = (n + page_rows - 1) / page_rows;
        assert_eq!(
            page_count, expected_pages,
            "page count must match real OS-page-derived chunking"
        );

        let mut offset = 0usize;
        for i in 0..zm.len() {
            let zp = zm.page(i).unwrap();
            let take = (n - offset).min(page_rows);
            let slice = &values[offset..offset + take];
            let expected_min = *slice.iter().min().unwrap();
            let expected_max = *slice.iter().max().unwrap();
            assert_eq!(zp.min, expected_min, "page {i} min mismatch");
            assert_eq!(zp.max, expected_max, "page {i} max mismatch");
            assert_eq!(zp.row_count as usize, take, "page {i} row_count mismatch");
            offset += take;
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn zonemap_load_missing_file_is_none_not_error() {
        let path = tmp_path("missing.zmap");
        let result = ZoneMap::load(&path).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn zonemap_load_truncated_file_is_a_real_error() {
        let path = tmp_path("truncated.zmap");
        std::fs::write(&path, [0u8; 7]).unwrap(); // not a multiple of 24
        let result = ZoneMap::load(&path);
        assert!(matches!(result, Err(ZoneMapError::Truncated { file_len: 7 })));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn predicate_boundary_values_are_correct_not_off_by_one() {
        let page = ZonePage {
            page_index: 0,
            row_count: 10,
            min: 100,
            max: 200,
        };
        assert!(!ZonePredicate::Gt(200).page_may_match(&page));
        assert!(ZonePredicate::Gt(199).page_may_match(&page));
        assert!(!ZonePredicate::Lt(100).page_may_match(&page));
        assert!(ZonePredicate::Lt(101).page_may_match(&page));
        assert!(ZonePredicate::Eq(100).page_may_match(&page));
        assert!(ZonePredicate::Eq(200).page_may_match(&page));
        assert!(!ZonePredicate::Eq(99).page_may_match(&page));
        assert!(!ZonePredicate::Eq(201).page_may_match(&page));
    }
}
