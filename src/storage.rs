//! Columnar storage kernel (Apache Arrow–aligned layout).
//!
//! Values live in packed, contiguous column buffers. Variable-width UTF-8
//! fields use an offsets array + a single data slab — never per-row `String`s.
//! Hot-path access is pointer arithmetic over pre-mapped / pre-sized regions.
//!
//! Catalog registration may box tables once (cold path). Query execution loops
//! never call `alloc`, `Vec`, `String`, or `clone`.

use core::ptr;

/// Vectorized batch width used by the runtime engine.
pub const BATCH_ROWS: usize = 1024;

/// Maximum columns in a table schema.
pub const MAX_COLUMNS: usize = 8;

/// Maximum rows in an in-core table segment (supports multi-batch + scalar tail).
/// Sized for ≥2050-row integration scans while staying 64-byte / 1024-batch aligned.
pub const MAX_ROWS: usize = 4096;

/// Maximum bytes in the shared UTF-8 data slab for Utf8 columns.
pub const UTF8_SLAB_CAP: usize = 8192;

/// Maximum length of a column / relation name (bytes).
pub const NAME_CAP: usize = 64;

/// Physical column type tag.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PhysType {
    Null = 0,
    Int64 = 1,
    Utf8 = 2,
    Bool = 3,
}

/// Fixed-size column name stored inline (no heap).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ColName {
    pub bytes: [u8; NAME_CAP],
    pub len: u16,
    pub _pad: [u8; 2],
}

impl ColName {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            bytes: [0; NAME_CAP],
            len: 0,
            _pad: [0; 2],
        }
    }

    #[inline(always)]
    pub fn from_bytes(src: &[u8]) -> Self {
        let mut out = Self::empty();
        let n = src.len().min(NAME_CAP);
        out.bytes[..n].copy_from_slice(&src[..n]);
        out.len = n as u16;
        out
    }

    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len as usize]
    }

    #[inline(always)]
    pub fn eq_bytes(&self, other: &[u8]) -> bool {
        self.as_bytes() == other
    }
}

/// Schema entry for one column.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ColumnMeta {
    pub name: ColName,
    pub phys: PhysType,
    pub _pad: [u8; 3],
    /// Byte offset into the table's typed payload region for this column.
    pub data_off: u32,
    /// For Utf8: offset into the offsets array region.
    pub offsets_off: u32,
}

impl ColumnMeta {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            name: ColName::empty(),
            phys: PhysType::Null,
            _pad: [0; 3],
            data_off: 0,
            offsets_off: 0,
        }
    }
}

/// Validity bitmap: 1 bit per row, packed into u64 limbs (Arrow-compatible).
#[repr(C, align(64))]
#[derive(Copy, Clone)]
pub struct ValidityBitmap {
    pub limbs: [u64; MAX_ROWS / 64],
}

impl ValidityBitmap {
    #[inline(always)]
    pub const fn all_valid() -> Self {
        Self {
            limbs: [u64::MAX; MAX_ROWS / 64],
        }
    }

    #[inline(always)]
    pub const fn none_valid() -> Self {
        Self {
            limbs: [0u64; MAX_ROWS / 64],
        }
    }

    #[inline(always)]
    pub fn set(&mut self, row: usize, valid: bool) {
        let limb = row / 64;
        let bit = row % 64;
        let mask = 1u64 << bit;
        let cur = self.limbs[limb];
        let cleared = cur & !mask;
        let with = cleared | (mask & (0u64.wrapping_sub(valid as u64)));
        self.limbs[limb] = with;
    }

    #[inline(always)]
    pub fn get(&self, row: usize) -> bool {
        let limb = row / 64;
        let bit = row % 64;
        ((self.limbs[limb] >> bit) & 1) != 0
    }
}

/// Int64 physical column buffer (contiguous, 64-byte aligned).
#[repr(C, align(64))]
pub struct Int64Column {
    pub values: [i64; MAX_ROWS],
    pub validity: ValidityBitmap,
}

impl Int64Column {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            values: [0; MAX_ROWS],
            validity: ValidityBitmap::all_valid(),
        }
    }
}

/// Bool physical column buffer packed as bytes (0/1) for SIMD-friendly scans.
#[repr(C, align(64))]
pub struct BoolColumn {
    pub values: [u8; MAX_ROWS],
    pub validity: ValidityBitmap,
}

impl BoolColumn {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            values: [0; MAX_ROWS],
            validity: ValidityBitmap::all_valid(),
        }
    }
}

/// Utf8 physical column: Arrow-style offsets[row_count+1] + data slab.
#[repr(C, align(64))]
pub struct Utf8Column {
    /// offsets[i] .. offsets[i+1] indexes into `data`.
    pub offsets: [u32; MAX_ROWS + 1],
    pub data: [u8; UTF8_SLAB_CAP],
    pub data_len: u32,
    pub validity: ValidityBitmap,
}

impl Utf8Column {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            offsets: [0; MAX_ROWS + 1],
            data: [0; UTF8_SLAB_CAP],
            data_len: 0,
            validity: ValidityBitmap::all_valid(),
        }
    }

    /// Reset slab write cursor without releasing backing storage.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.data_len = 0;
        self.offsets[0] = 0;
        self.validity = ValidityBitmap::none_valid();
    }

    /// Append a UTF-8 value for `row`. Returns false if the slab is exhausted.
    #[inline(always)]
    pub fn set_row(&mut self, row: usize, bytes: &[u8]) -> bool {
        if row >= MAX_ROWS {
            return false;
        }
        let start = self.data_len as usize;
        let end = start + bytes.len();
        if end > UTF8_SLAB_CAP {
            return false;
        }
        self.data[start..end].copy_from_slice(bytes);
        self.offsets[row] = start as u32;
        self.offsets[row + 1] = end as u32;
        self.data_len = end as u32;
        self.validity.set(row, true);
        true
    }

    /// Borrow the UTF-8 bytes for `row`, validated as Unicode on grapheme-safe bounds.
    #[inline(always)]
    pub fn get_row<'a>(&'a self, row: usize) -> Option<&'a str> {
        if row >= MAX_ROWS || !self.validity.get(row) {
            return None;
        }
        let start = self.offsets[row] as usize;
        let end = self.offsets[row + 1] as usize;
        if end < start || end > self.data_len as usize {
            return None;
        }
        core::str::from_utf8(&self.data[start..end]).ok()
    }
}

/// Column payload slot — tagged union of physical buffers.
#[repr(C)]
pub enum ColumnData {
    Int64(Int64Column),
    Utf8(Utf8Column),
    Bool(BoolColumn),
    Null,
}

impl ColumnData {
    #[inline(always)]
    pub fn phys_type(&self) -> PhysType {
        match self {
            ColumnData::Int64(_) => PhysType::Int64,
            ColumnData::Utf8(_) => PhysType::Utf8,
            ColumnData::Bool(_) => PhysType::Bool,
            ColumnData::Null => PhysType::Null,
        }
    }
}

/// In-core columnar table segment. All column buffers are inline.
/// Base address locked to a 64-byte cache line for TLB-friendly scans.
#[repr(C, align(64))]
pub struct Table {
    pub name: ColName,
    pub col_meta: [ColumnMeta; MAX_COLUMNS],
    pub col_count: u16,
    pub row_count: u16,
    pub _pad: [u8; 4],
    pub columns: [ColumnData; MAX_COLUMNS],
}

const _: () = assert!(MAX_ROWS == 4096);
const _: () = assert!(MAX_ROWS % 64 == 0);
const _: () = assert!(MAX_ROWS % BATCH_ROWS == 0);
const _: () = assert!(core::mem::align_of::<Int64Column>() == 64);
const _: () = assert!(core::mem::align_of::<Utf8Column>() == 64);

impl Table {
    pub fn new(name: &[u8]) -> Self {
        Self {
            name: ColName::from_bytes(name),
            col_meta: [ColumnMeta::empty(); MAX_COLUMNS],
            col_count: 0,
            row_count: 0,
            _pad: [0; 4],
            columns: [
                ColumnData::Null,
                ColumnData::Null,
                ColumnData::Null,
                ColumnData::Null,
                ColumnData::Null,
                ColumnData::Null,
                ColumnData::Null,
                ColumnData::Null,
            ],
        }
    }

    /// Cold-path heap construction — avoids placing a multi-hundred-KB `Table`
    /// temporary on the caller's stack frame.
    pub fn new_boxed(name: &[u8]) -> Box<Self> {
        use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
        unsafe {
            let layout = Layout::new::<Self>();
            let ptr = alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            (*ptr).name = ColName::from_bytes(name);
            (*ptr).col_count = 0;
            (*ptr).row_count = 0;
            let mut i = 0usize;
            while i < MAX_COLUMNS {
                core::ptr::write(&mut (*ptr).columns[i], ColumnData::Null);
                (*ptr).col_meta[i] = ColumnMeta::empty();
                i += 1;
            }
            Box::from_raw(ptr)
        }
    }

    #[inline(always)]
    pub fn add_int64_column(&mut self, name: &[u8]) -> Option<usize> {
        let i = self.col_count as usize;
        if i >= MAX_COLUMNS {
            return None;
        }
        self.col_meta[i] = ColumnMeta {
            name: ColName::from_bytes(name),
            phys: PhysType::Int64,
            _pad: [0; 3],
            data_off: 0,
            offsets_off: 0,
        };
        self.columns[i] = ColumnData::Int64(Int64Column::new());
        self.col_count = self.col_count.wrapping_add(1);
        Some(i)
    }

    #[inline(always)]
    pub fn add_utf8_column(&mut self, name: &[u8]) -> Option<usize> {
        let i = self.col_count as usize;
        if i >= MAX_COLUMNS {
            return None;
        }
        self.col_meta[i] = ColumnMeta {
            name: ColName::from_bytes(name),
            phys: PhysType::Utf8,
            _pad: [0; 3],
            data_off: 0,
            offsets_off: 0,
        };
        self.columns[i] = ColumnData::Utf8(Utf8Column::new());
        self.col_count = self.col_count.wrapping_add(1);
        Some(i)
    }

    #[inline(always)]
    pub fn find_column(&self, name: &[u8]) -> Option<usize> {
        let n = self.col_count as usize;
        let mut i = 0usize;
        while i < n {
            if self.col_meta[i].name.eq_bytes(name) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    #[inline(always)]
    pub fn set_row_count(&mut self, rows: usize) {
        self.row_count = rows.min(MAX_ROWS) as u16;
    }

    #[inline(always)]
    pub fn int64_mut(&mut self, col: usize) -> Option<&mut Int64Column> {
        match &mut self.columns[col] {
            ColumnData::Int64(c) => Some(c),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn int64(&self, col: usize) -> Option<&Int64Column> {
        match &self.columns[col] {
            ColumnData::Int64(c) => Some(c),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn utf8_mut(&mut self, col: usize) -> Option<&mut Utf8Column> {
        match &mut self.columns[col] {
            ColumnData::Utf8(c) => Some(c),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn utf8(&self, col: usize) -> Option<&Utf8Column> {
        match &self.columns[col] {
            ColumnData::Utf8(c) => Some(c),
            _ => None,
        }
    }
}

/// Memory-map style view over an external contiguous byte region.
/// Does not own the memory; caller guarantees lifetime / alignment.
#[repr(C)]
pub struct MappedRegion<'a> {
    pub ptr: *const u8,
    pub len: usize,
    pub _marker: core::marker::PhantomData<&'a [u8]>,
}

impl<'a> MappedRegion<'a> {
    #[inline(always)]
    pub fn from_slice(slice: &'a [u8]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            _marker: core::marker::PhantomData,
        }
    }

    /// # Safety
    /// `ptr` must remain valid for `len` bytes for `'a`.
    #[inline(always)]
    pub unsafe fn as_slice(&self) -> &'a [u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Pointer-arithmetic subview without allocation.
    ///
    /// # Safety
    /// Parent region must outlive the returned view.
    #[inline(always)]
    pub unsafe fn slice_at(&self, off: usize, len: usize) -> Option<MappedRegion<'a>> {
        if off.checked_add(len).map(|e| e <= self.len).unwrap_or(false) {
            Some(MappedRegion {
                ptr: unsafe { self.ptr.add(off) },
                len,
                _marker: core::marker::PhantomData,
            })
        } else {
            None
        }
    }
}

/// Catalog of named in-core tables (fixed slots).
/// Tables are boxed once at registration (cold path / mmap substitute for
/// large columnar segments); query loops only borrow `&Table`.
pub const MAX_TABLES: usize = 8;

#[repr(C)]
pub struct Catalog {
    pub tables: [Option<Box<Table>>; MAX_TABLES],
    pub len: u16,
    pub _pad: [u8; 6],
}

impl Catalog {
    pub fn new() -> Self {
        Self {
            tables: [None, None, None, None, None, None, None, None],
            len: 0,
            _pad: [0; 6],
        }
    }

    pub fn register(&mut self, table: Table) -> Option<usize> {
        self.register_box(Box::new(table))
    }

    pub fn register_box(&mut self, table: Box<Table>) -> Option<usize> {
        let i = self.len as usize;
        if i >= MAX_TABLES {
            return None;
        }
        self.tables[i] = Some(table);
        self.len = self.len.wrapping_add(1);
        Some(i)
    }

    pub fn find(&self, name: &[u8]) -> Option<&Table> {
        let n = self.len as usize;
        let mut i = 0usize;
        while i < n {
            if let Some(t) = &self.tables[i] {
                if t.name.eq_bytes(name) {
                    return Some(t.as_ref());
                }
            }
            i += 1;
        }
        None
    }

    pub fn find_mut(&mut self, name: &[u8]) -> Option<&mut Table> {
        let n = self.len as usize;
        let mut i = 0usize;
        while i < n {
            let matches = self.tables[i]
                .as_ref()
                .map(|t| t.name.eq_bytes(name))
                .unwrap_or(false);
            if matches {
                return self.tables[i].as_deref_mut();
            }
            i += 1;
        }
        None
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy row selection mask for MAX_ROWS rows.
#[repr(C, align(64))]
pub struct SelectionVector {
    /// 1 = keep, 0 = drop — byte mask for SIMD compares.
    pub mask: [u8; MAX_ROWS],
    pub len: u16,
    pub _pad: [u8; 6],
}

impl SelectionVector {
    #[inline(always)]
    pub const fn all(rows: usize) -> Self {
        Self {
            mask: [1u8; MAX_ROWS],
            len: rows as u16,
            _pad: [0; 6],
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        let mut i = 0usize;
        while i < MAX_ROWS {
            self.mask[i] = 0;
            i += 1;
        }
        self.len = 0;
    }

    #[inline(always)]
    pub fn count_selected(&self) -> usize {
        let n = self.len as usize;
        let mut c = 0usize;
        let mut i = 0usize;
        while i < n {
            c += self.mask[i] as usize;
            i += 1;
        }
        c
    }
}

const _: () = assert!(core::mem::align_of::<SelectionVector>() == 64);
const _: () = assert!(core::mem::align_of::<Table>() == 64);

/// Build the demo `பயனர்கள்` (users) table used by the integration harness.
pub fn seed_users_table() -> Box<Table> {
    let mut t = Table::new_boxed("பயனர்கள்".as_bytes());
    let name_i = t.add_utf8_column("பெயர்".as_bytes()).unwrap();
    let age_i = t.add_int64_column("வயது".as_bytes()).unwrap();

    let names: [&[u8]; 16] = [
        "அருண்".as_bytes(),
        "பிரியா".as_bytes(),
        "கண்ணன்".as_bytes(),
        "லட்சுமி".as_bytes(),
        "முருகன்".as_bytes(),
        "கவிதா".as_bytes(),
        "ராஜ்".as_bytes(),
        "மீனா".as_bytes(),
        "சுரேஷ்".as_bytes(),
        "தீபா".as_bytes(),
        "விஜய்".as_bytes(),
        "அனிதா".as_bytes(),
        "கோபால்".as_bytes(),
        "சுமதி".as_bytes(),
        "கார்த்திக்".as_bytes(),
        "நந்தினி".as_bytes(),
    ];
    let ages: [i64; 16] = [
        18, 22, 19, 25, 30, 21, 27, 24, 17, 35, 28, 20, 23, 26, 31, 29,
    ];

    {
        let utf8 = t.utf8_mut(name_i).unwrap();
        utf8.clear();
        let mut r = 0usize;
        while r < 16 {
            assert!(utf8.set_row(r, names[r]));
            r += 1;
        }
    }
    {
        let i64c = t.int64_mut(age_i).unwrap();
        let mut r = 0usize;
        while r < 16 {
            i64c.values[r] = ages[r];
            i64c.validity.set(r, true);
            r += 1;
        }
    }
    t.set_row_count(16);
    t
}

/// Raw byte copy helper that never allocates.
///
/// # Safety
/// `dst` / `src` must be valid for `len` bytes and non-overlapping.
#[inline(always)]
pub unsafe fn memcpy_bytes(dst: *mut u8, src: *const u8, len: usize) {
    unsafe { ptr::copy_nonoverlapping(src, dst, len) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn users_table_roundtrip() {
        let t = seed_users_table();
        assert_eq!(t.row_count, 16);
        let name = t.find_column("பெயர்".as_bytes()).unwrap();
        let age = t.find_column("வயது".as_bytes()).unwrap();
        assert_eq!(t.utf8(name).unwrap().get_row(1), Some("பிரியா"));
        assert_eq!(t.int64(age).unwrap().values[1], 22);
        assert_eq!(t.col_meta[name].name.as_bytes(), "பெயர்".as_bytes());
    }
}
