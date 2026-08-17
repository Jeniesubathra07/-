//! Columnar storage kernel (Apache Arrow–aligned layout).
//!
//! Values live in packed, contiguous column buffers. Variable-width UTF-8
//! fields use an offsets array + a single data slab — never per-row `String`s.
//! Hot-path access is pointer arithmetic over pre-mapped / pre-sized regions.
//!
//! Catalog registration may box tables once (cold path). Query execution loops
//! never call `alloc`, `Vec`, `String`, or `clone`.
//!
//! # Platform contract (Stage-4 persistence)
//! On-disk Int64 payloads are **little-endian** `i64` bytes. `os_page_size_bytes`
//! uses POSIX `sysconf(_SC_PAGESIZE)` under `#[cfg(unix)]` with a 4096 fallback.
//! Windows page-size / big-endian hosts are **not** verified in this environment
//! (x86_64 Linux only was executed). Mmap readers assume a single-writer-absent
//! snapshot — see SIGBUS hazard docs on every public `open*` method.

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
        let end = match start.checked_add(bytes.len()) {
            Some(e) => e,
            None => return false,
        };
        if end > UTF8_SLAB_CAP {
            return false;
        }
        // Truncating casts are safe: end <= UTF8_SLAB_CAP fits in u32.
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
    /// Cold-path heap construction — the only supported public constructor.
    ///
    /// `size_of::<Table>() == 267_520`. There is intentionally no stack `new()`
    /// / `Default` path: constructing or moving `Table` by value overflows
    /// constrained stacks (verified: 64 KiB thread → stack overflow / EXIT 134).
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
    pub orders: Option<Box<FixedOrdersDatabase>>,
    pub len: u16,
    pub _pad: [u8; 6],
}

impl Catalog {
    pub fn new() -> Self {
        Self {
            tables: [None, None, None, None, None, None, None, None],
            orders: None,
            len: 0,
            _pad: [0; 6],
        }
    }

    /// Register a heap-allocated table. Takes `Box<Table>` so callers never
    /// materialize a 267 KB `Table` by-value on the stack.
    pub fn register(&mut self, table: Box<Table>) -> Option<usize> {
        self.register_box(table)
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

    pub fn set_orders(&mut self, orders: Box<FixedOrdersDatabase>) {
        self.orders = Some(orders);
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
    let id_i = t.add_int64_column("அடையாளம்".as_bytes()).unwrap();
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
        let ids = t.int64_mut(id_i).unwrap();
        let mut r = 0usize;
        while r < 16 {
            ids.values[r] = r as i64;
            ids.validity.set(r, true);
            r += 1;
        }
    }
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

/// Secondary packed orders relation for sort-merge joins (`ஆர்டர்கள்`).
#[repr(C, align(64))]
pub struct FixedOrdersDatabase {
    pub user_id_column: [i64; MAX_ROWS],
    pub price_column: [i64; MAX_ROWS],
    /// Derived price slot (`கணி` / Stage-3 arithmetic output mirror).
    pub derived_prices: [i64; MAX_ROWS],
    pub row_count: u16,
    pub _pad: [u8; 6],
}

const _: () = assert!(core::mem::align_of::<FixedOrdersDatabase>() == 64);

impl FixedOrdersDatabase {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            user_id_column: [0; MAX_ROWS],
            price_column: [0; MAX_ROWS],
            derived_prices: [0; MAX_ROWS],
            row_count: 0,
            _pad: [0; 6],
        }
    }

    /// Cold-path heap construction (orders buffers are multi-hundred KB).
    pub fn new_boxed() -> Box<Self> {
        use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
        unsafe {
            let layout = Layout::new::<Self>();
            let ptr = alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            (*ptr).row_count = 0;
            Box::from_raw(ptr)
        }
    }
}

/// Seed `ஆர்டர்கள்` — user_id keys align with பயனர்கள்.அடையாளம்.
pub fn seed_orders_database() -> Box<FixedOrdersDatabase> {
    let mut o = FixedOrdersDatabase::new_boxed();
    // 12 orders spanning ages above/below the வயது > 21 filter boundary.
    // user_id → row in users; price in விலை units.
    let pairs: [(i64, i64); 12] = [
        (1, 450),  // பிரியா age 22
        (3, 800),  // லட்சுமி age 25
        (4, 1200), // முருகன் age 30
        (6, 350),  // ராஜ் age 27
        (7, 600),  // மீனா age 24
        (9, 900),  // தீபா age 35
        (10, 500), // விஜய் age 28
        (12, 700), // கோபால் age 23
        (13, 550), // சுமதி age 26
        (14, 1100),// கார்த்திக் age 31
        (15, 400), // நந்தினி age 29
        (0, 100),  // அருண் age 18 (filtered out by வயது > 21)
    ];
    let n = pairs.len();
    let mut i = 0usize;
    while i < n {
        o.user_id_column[i] = pairs[i].0;
        o.price_column[i] = pairs[i].1;
        // Pre-seed derived_prices as identity; runtime `கணி` overwrites per query.
        o.derived_prices[i] = pairs[i].1;
        i += 1;
    }
    o.row_count = n as u16;
    o
}

/// Also expose orders as a columnar `Table` named `ஆர்டர்கள்` for catalog lookup.
pub fn seed_orders_table() -> Box<Table> {
    let mut t = Table::new_boxed("ஆர்டர்கள்".as_bytes());
    let uid = t.add_int64_column("அடையாளம்".as_bytes()).unwrap();
    let price = t.add_int64_column("விலை".as_bytes()).unwrap();
    let orders = seed_orders_database();
    let n = orders.row_count as usize;
    {
        let c = t.int64_mut(uid).unwrap();
        let mut i = 0usize;
        while i < n {
            c.values[i] = orders.user_id_column[i];
            c.validity.set(i, true);
            i += 1;
        }
    }
    {
        let c = t.int64_mut(price).unwrap();
        let mut i = 0usize;
        while i < n {
            c.values[i] = orders.price_column[i];
            c.validity.set(i, true);
            i += 1;
        }
    }
    t.set_row_count(n);
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

// ═══════════════════════════════════════════════════════════════════════════
// Stage-4 — zero-allocation columnar disk persistence (mmap page stream)
// ═══════════════════════════════════════════════════════════════════════════
//
// READ-ONLY single-writer-absent snapshots: ingest happens once via a cold-path
// tool; query-time mmap readers assume no concurrent writers. Concurrent-writer
// safety is intentionally out of scope for this stage.
//
// memmap2 maps pages via the OS virtual-memory subsystem — not `std::alloc`.
// Accessing mapped bytes is a load from an OS mapping, not a `Vec`/`Box`/`String`
// allocation, so hot-path page walks preserve the no-heap invariant.

use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

/// POSIX `_SC_PAGESIZE` (Linux). Declared locally so we do not add a `libc` crate.
#[cfg(unix)]
const _SC_PAGESIZE: i32 = 30;

#[cfg(unix)]
extern "C" {
    fn sysconf(name: i32) -> i64;
}

/// Real OS page size in bytes (`sysconf(_SC_PAGESIZE)`).
///
/// Page chunking for fixed-width columns derives
/// `page_rows = page_size_bytes / row_width_bytes` — never a hardcoded row count.
#[inline(always)]
pub fn os_page_size_bytes() -> usize {
    #[cfg(unix)]
    {
        // SAFETY: sysconf(_SC_PAGESIZE) is defined on POSIX and returns > 0.
        let p = unsafe { sysconf(_SC_PAGESIZE) };
        if p > 0 {
            return p as usize;
        }
    }
    4096
}

/// Ingest-time `.meta` companion for an Int64 `.bin` column (written once).
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Int64ColumnMeta {
    pub row_count: u64,
}

/// On-disk Int64 column: raw `[i64]` bytes (no header), page-aligned file length.
///
/// Row count is derived as `file_len / 8` at open and validated against `.meta`.
#[repr(C)]
pub struct Int64ColumnFile {
    pub stream: ColumnarFileStream,
    pub meta: Int64ColumnMeta,
}

/// One Utf8 index entry: offset+length into the sibling `.blob` (no fixed padding).
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Utf8OffsetEntry {
    pub offset: u32,
    pub length: u32,
}

/// Ingest-time `.meta` for a Utf8 column (`.offsets` + `.blob`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Utf8ColumnMeta {
    pub row_count: u64,
    pub blob_len: u64,
}

/// On-disk Utf8 column: page-aligned `.offsets` (u32 pairs) + raw `.blob` bytes.
///
/// The blob has no page-alignment constraint — it is addressed by offset, not chunked.
pub struct Utf8ColumnFile {
    offsets_mmap: Mmap,
    blob_mmap: Mmap,
    pub meta: Utf8ColumnMeta,
    total_rows: usize,
}

impl Utf8ColumnFile {
    /// Cold-path open of `.offsets` + `.blob` (+ optional `.meta` validation).
    ///
    /// # Hard precondition (SIGBUS hazard)
    /// Both mappings are READ-ONLY single-writer-absent snapshots. Truncating,
    /// deleting, or replacing either file while this value is live can deliver
    /// `SIGBUS` on the next access — not a Rust `Result::Err`.
    pub fn open(offsets_path: &Path, blob_path: &Path, meta_path: Option<&Path>) -> io::Result<Self> {
        let off_file = File::open(offsets_path)?;
        let blob_file = File::open(blob_path)?;
        let off_len = off_file.metadata()?.len() as usize;
        let entry_size = core::mem::size_of::<Utf8OffsetEntry>();
        if off_len % entry_size != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "utf8 .offsets length not multiple of 8",
            ));
        }
        let capacity_rows = off_len / entry_size;
        // SAFETY: read-only files; lengths validated.
        let offsets_mmap = unsafe { Mmap::map(&off_file)? };
        let blob_mmap = unsafe { Mmap::map(&blob_file)? };
        let blob_len = blob_mmap.len() as u64;
        let (meta, total_rows) = if let Some(mp) = meta_path {
            let m = read_utf8_meta(mp)?;
            let rc = m.row_count as usize;
            // Page-alignment padding may extend `.offsets` past logical rows.
            if rc > capacity_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "utf8 .meta row_count exceeds .offsets capacity",
                ));
            }
            if m.blob_len != blob_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "utf8 .meta blob_len mismatch",
                ));
            }
            (m, rc)
        } else {
            (
                Utf8ColumnMeta {
                    row_count: capacity_rows as u64,
                    blob_len,
                },
                capacity_rows,
            )
        };
        Ok(Self {
            offsets_mmap,
            blob_mmap,
            meta,
            total_rows,
        })
    }

    #[inline(always)]
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// Zero-copy UTF-8 view for row `i` (offset+length into `.blob`).
    #[inline(always)]
    pub fn get_row(&self, i: usize) -> Option<&str> {
        if i >= self.total_rows {
            return None;
        }
        let base = i.wrapping_mul(core::mem::size_of::<Utf8OffsetEntry>());
        if base + 8 > self.offsets_mmap.len() {
            return None;
        }
        let off = u32::from_le_bytes([
            self.offsets_mmap[base],
            self.offsets_mmap[base + 1],
            self.offsets_mmap[base + 2],
            self.offsets_mmap[base + 3],
        ]) as usize;
        let len = u32::from_le_bytes([
            self.offsets_mmap[base + 4],
            self.offsets_mmap[base + 5],
            self.offsets_mmap[base + 6],
            self.offsets_mmap[base + 7],
        ]) as usize;
        let end = off.checked_add(len)?;
        if end > self.blob_mmap.len() {
            return None;
        }
        core::str::from_utf8(&self.blob_mmap[off..end]).ok()
    }
}

fn read_int64_meta(path: &Path) -> io::Result<Int64ColumnMeta> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "int64 .meta too short",
        ));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[..8]);
    Ok(Int64ColumnMeta {
        row_count: u64::from_le_bytes(buf),
    })
}

fn write_int64_meta(path: &Path, meta: &Int64ColumnMeta) -> io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(&meta.row_count.to_le_bytes())?;
    f.flush()?;
    Ok(())
}

fn read_utf8_meta(path: &Path) -> io::Result<Utf8ColumnMeta> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "utf8 .meta too short",
        ));
    }
    let mut a = [0u8; 8];
    let mut b = [0u8; 8];
    a.copy_from_slice(&bytes[..8]);
    b.copy_from_slice(&bytes[8..16]);
    Ok(Utf8ColumnMeta {
        row_count: u64::from_le_bytes(a),
        blob_len: u64::from_le_bytes(b),
    })
}

fn write_utf8_meta(path: &Path, meta: &Utf8ColumnMeta) -> io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(&meta.row_count.to_le_bytes())?;
    f.write_all(&meta.blob_len.to_le_bytes())?;
    f.flush()?;
    Ok(())
}

/// Pad `file` so its length is a multiple of `page_size` (ingest cold path).
fn pad_to_page(file: &mut File, page_size: usize) -> io::Result<()> {
    let len = file.metadata()?.len() as usize;
    let rem = len % page_size;
    if rem == 0 {
        return Ok(());
    }
    let mut left = page_size - rem;
    let chunk = [0u8; 512];
    while left > 0 {
        let n = left.min(chunk.len());
        file.write_all(&chunk[..n])?;
        left -= n;
    }
    Ok(())
}

/// One memory-mapped page window over a fixed-width Int64 column file.
///
/// `rows` is a direct view into the mmap — **no copy, no heap**.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ColumnarChunk<'a> {
    pub rows: &'a [i64],
    pub row_count: u16,
    pub page_index: u32,
    /// 1 when this is the final partial chunk (`row_count < page_rows`).
    pub is_residue: u8,
    pub _pad: [u8; 1],
}

impl<'a> ColumnarChunk<'a> {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            rows: &[],
            row_count: 0,
            page_index: 0,
            is_residue: 0,
            _pad: [0; 1],
        }
    }
}

/// Zero-copy Int64 columnar file stream via OS virtual memory map.
///
/// # Page derivation
/// `page_rows = os_page_size_bytes() / row_width` for fixed-width columns
/// (`row_width == 8` for `i64`). Chunking follows the real OS page size from
/// `sysconf(_SC_PAGESIZE)`, not a hardcoded row count such as 4096.
///
/// Cold path: `open` / ingest. Hot path: [`ColumnarFileStream::next_page_chunk`]
/// returns a borrowed slice over the mmap (lifetime tied to `&self` / `&mut self`).
pub struct ColumnarFileStream {
    /// Owns the OS mapping for this column's lifetime.
    pub(crate) mmap: Mmap,
    pub row_width: usize,
    /// `= sysconf(_SC_PAGESIZE) / row_width`.
    pub page_rows: usize,
    pub total_rows: usize,
    pub cursor_row: usize,
}

impl ColumnarFileStream {
    /// Cold-path: open + mmap a raw Int64 `.bin` (optional companion `.meta`).
    ///
    /// # Hard precondition (SIGBUS hazard)
    /// The returned mapping is READ-ONLY and assumes a **single-writer-absent
    /// snapshot**: the backing file must not be truncated, deleted, or replaced
    /// for the lifetime of this stream. Violating that precondition can deliver
    /// `SIGBUS` on access — a process-level signal, **not** a Rust `Result::Err`.
    pub fn open_i64(path: &Path) -> io::Result<Self> {
        let meta_path = path.with_extension("meta");
        let meta = match File::open(&meta_path) {
            Ok(mut f) => {
                let mut buf = [0u8; 8];
                use std::io::Read;
                f.read_exact(&mut buf)?;
                Some(Int64ColumnMeta {
                    row_count: u64::from_le_bytes(buf),
                })
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        Self::open_i64_inner(path, meta)
    }

    /// Cold-path open with explicit `.meta` validation (row_count must match).
    ///
    /// # Hard precondition (SIGBUS hazard)
    /// Same as [`ColumnarFileStream::open_i64`]: do not mutate the backing file
    /// for the lifetime of the returned mapping.
    pub fn open_i64_with_meta(bin_path: &Path, meta_path: &Path) -> io::Result<Self> {
        let meta = read_int64_meta(meta_path)?;
        Self::open_i64_inner(bin_path, Some(meta))
    }

    fn open_i64_inner(path: &Path, meta: Option<Int64ColumnMeta>) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        let row_width = core::mem::size_of::<i64>();
        if len % row_width != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "i64 column file length not multiple of 8",
            ));
        }
        // SAFETY: file is opened read-only; length validated.
        let mmap = unsafe { Mmap::map(&file)? };
        // Logical row count from payload bytes (ignore page-alignment padding
        // beyond the last full i64 when a `.meta` row_count is present).
        let file_rows = len / row_width;
        let total_rows = if let Some(m) = meta {
            let rc = m.row_count as usize;
            if rc > file_rows {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "i64 .meta row_count exceeds file capacity",
                ));
            }
            rc
        } else {
            file_rows
        };
        let page_size = os_page_size_bytes();
        let page_rows = if row_width == 0 {
            1
        } else {
            (page_size / row_width).max(1)
        };
        Ok(Self {
            mmap,
            row_width,
            page_rows,
            total_rows,
            cursor_row: 0,
        })
    }

    /// Open an [`Int64ColumnFile`] (stream + validated meta).
    pub fn open_int64_column(bin_path: &Path, meta_path: &Path) -> io::Result<Int64ColumnFile> {
        let meta = read_int64_meta(meta_path)?;
        let stream = Self::open_i64_inner(bin_path, Some(meta))?;
        Ok(Int64ColumnFile { stream, meta })
    }

    #[inline(always)]
    pub fn total_rows(&self) -> u64 {
        self.total_rows as u64
    }

    #[inline(always)]
    pub fn cursor_row(&self) -> u64 {
        self.cursor_row as u64
    }

    #[inline(always)]
    pub fn page_rows(&self) -> usize {
        self.page_rows
    }

    #[inline(always)]
    pub fn pages_emitted(&self) -> u32 {
        if self.page_rows == 0 {
            return 0;
        }
        (self.cursor_row / self.page_rows) as u32
    }

    /// Rewind to the first page (no remapping / no alloc).
    #[inline(always)]
    pub fn rewind(&mut self) {
        self.cursor_row = 0;
    }

    /// Advance `cursor_row` by `page_rows` (or the remainder) and return a
    /// zero-copy borrowed slice view over the mmap.
    ///
    /// Emits a final partial chunk when `total_rows % page_rows != 0`.
    #[inline(always)]
    pub fn next_page_chunk(&mut self) -> Option<ColumnarChunk<'_>> {
        if self.cursor_row >= self.total_rows {
            return None;
        }
        let remaining = self.total_rows - self.cursor_row;
        let n = remaining.min(self.page_rows);
        let byte_off = match self.cursor_row.checked_mul(self.row_width) {
            Some(o) => o,
            None => return None,
        };
        let byte_len = match n.checked_mul(self.row_width) {
            Some(l) => l,
            None => return None,
        };
        let end = match byte_off.checked_add(byte_len) {
            Some(e) => e,
            None => return None,
        };
        if end > self.mmap.len() {
            return None;
        }
        let bytes = &self.mmap[byte_off..end];
        // SAFETY: (1) `n * row_width` bytes with `row_width == 8`; (2) mmap base
        // is OS-page-aligned and page size is a multiple of 8 on POSIX, so
        // `byte_off` (multiple of 8) yields an 8-byte-aligned `i64` pointer;
        // (3) length covers exactly `n` i64 elements within the mapped region.
        // Verified under Miri with isolation disabled on x86_64 Linux.
        debug_assert_eq!(bytes.as_ptr() as usize % core::mem::align_of::<i64>(), 0);
        let rows: &[i64] =
            unsafe { core::slice::from_raw_parts(bytes.as_ptr() as *const i64, n) };
        let page_index = if self.page_rows == 0 {
            0
        } else {
            (self.cursor_row / self.page_rows) as u32
        };
        let is_residue = (n < self.page_rows) as u8;
        self.cursor_row = self.cursor_row.wrapping_add(n);
        Some(ColumnarChunk {
            rows,
            row_count: n as u16,
            page_index,
            is_residue,
            _pad: [0; 1],
        })
    }
}

/// Multi-column mmap table: independent `.bin` files per physical column.
///
/// Hot path advances all streams in lockstep via [`ColumnarTableStream::next_page`].
pub struct ColumnarTableStream {
    pub user_ids: ColumnarFileStream,
    pub ages: ColumnarFileStream,
    pub prices: ColumnarFileStream,
}

/// One lockstep multi-column page (zero-copy views).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ColumnarTablePage<'a> {
    pub user_ids: &'a [i64],
    pub ages: &'a [i64],
    pub prices: &'a [i64],
    pub row_count: u16,
    pub page_index: u32,
    pub is_residue: u8,
    pub _pad: [u8; 1],
}

impl ColumnarTableStream {
    /// Cold-path open of three Int64 column files (must share identical row counts).
    pub fn open(user_ids: &Path, ages: &Path, prices: &Path) -> io::Result<Self> {
        let user_ids = ColumnarFileStream::open_i64(user_ids)?;
        let ages = ColumnarFileStream::open_i64(ages)?;
        let prices = ColumnarFileStream::open_i64(prices)?;
        if user_ids.total_rows != ages.total_rows || ages.total_rows != prices.total_rows {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "columnar .bin row counts diverge",
            ));
        }
        if user_ids.page_rows != ages.page_rows || ages.page_rows != prices.page_rows {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "columnar page_rows diverge",
            ));
        }
        Ok(Self {
            user_ids,
            ages,
            prices,
        })
    }

    #[inline(always)]
    pub fn total_rows(&self) -> u64 {
        self.ages.total_rows as u64
    }

    #[inline(always)]
    pub fn rewind(&mut self) {
        self.user_ids.rewind();
        self.ages.rewind();
        self.prices.rewind();
    }

    /// Lockstep page advance — all columns share the same OS-page row window.
    #[inline(always)]
    pub fn next_page(&mut self) -> Option<ColumnarTablePage<'_>> {
        if self.ages.cursor_row >= self.ages.total_rows {
            return None;
        }
        let start_row = self.ages.cursor_row;
        let page_rows = self.ages.page_rows;
        let remaining = self.ages.total_rows.saturating_sub(start_row);
        let n = remaining.min(page_rows);
        let byte_off = start_row.wrapping_mul(8);
        let byte_end = byte_off.wrapping_add(n.wrapping_mul(8));
        if byte_end > self.ages.mmap.len()
            || byte_end > self.user_ids.mmap.len()
            || byte_end > self.prices.mmap.len()
        {
            return None;
        }
        let ages = unsafe {
            core::slice::from_raw_parts(
                self.ages.mmap[byte_off..byte_end].as_ptr() as *const i64,
                n,
            )
        };
        let user_ids = unsafe {
            core::slice::from_raw_parts(
                self.user_ids.mmap[byte_off..byte_end].as_ptr() as *const i64,
                n,
            )
        };
        let prices = unsafe {
            core::slice::from_raw_parts(
                self.prices.mmap[byte_off..byte_end].as_ptr() as *const i64,
                n,
            )
        };
        let page_index = if page_rows == 0 {
            0
        } else {
            (start_row / page_rows) as u32
        };
        let is_residue = (n < page_rows) as u8;
        let next_cursor = start_row.wrapping_add(n);
        self.ages.cursor_row = next_cursor;
        self.user_ids.cursor_row = next_cursor;
        self.prices.cursor_row = next_cursor;
        Some(ColumnarTablePage {
            user_ids,
            ages,
            prices,
            row_count: n as u16,
            page_index,
            is_residue,
            _pad: [0; 1],
        })
    }
}

/// Cold-path: write a packed little-endian Int64 `.bin` + companion `.meta`.
///
/// File payload is padded to the OS page size. Writes in stack windows of
/// [`MAX_ROWS`] — never materializes the full column as `Vec` of values
/// (padding uses a small page-sized buffer only at EOF).
pub fn write_i64_column_bin<F>(path: &Path, total_rows: usize, mut fill: F) -> io::Result<()>
where
    F: FnMut(usize) -> i64,
{
    let mut file = File::create(path)?;
    let mut window = [0i64; MAX_ROWS];
    let mut written = 0usize;
    while written < total_rows {
        let n = (total_rows - written).min(MAX_ROWS);
        let mut i = 0usize;
        while i < n {
            window[i] = fill(written + i);
            i += 1;
        }
        let bytes = unsafe {
            core::slice::from_raw_parts(
                window.as_ptr() as *const u8,
                n.wrapping_mul(core::mem::size_of::<i64>()),
            )
        };
        file.write_all(bytes)?;
        written += n;
    }
    let page_size = os_page_size_bytes();
    pad_to_page(&mut file, page_size)?;
    file.flush()?;
    let meta_path = path.with_extension("meta");
    write_int64_meta(
        &meta_path,
        &Int64ColumnMeta {
            row_count: total_rows as u64,
        },
    )?;
    Ok(())
}

/// Cold-path: write Utf8 `.offsets` (page-aligned) + `.blob` + `.meta`.
pub fn write_utf8_column_files(
    offsets_path: &Path,
    blob_path: &Path,
    meta_path: &Path,
    rows: &[&[u8]],
) -> io::Result<()> {
    let mut off_file = File::create(offsets_path)?;
    let mut blob_file = File::create(blob_path)?;
    let mut blob_len: u32 = 0;
    let mut i = 0usize;
    while i < rows.len() {
        let s = rows[i];
        let len_u32 = match u32::try_from(s.len()) {
            Ok(l) => l,
            Err(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "utf8 row longer than u32::MAX",
                ))
            }
        };
        let entry = Utf8OffsetEntry {
            offset: blob_len,
            length: len_u32,
        };
        off_file.write_all(&entry.offset.to_le_bytes())?;
        off_file.write_all(&entry.length.to_le_bytes())?;
        blob_file.write_all(s)?;
        blob_len = match blob_len.checked_add(len_u32) {
            Some(v) => v,
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "utf8 blob length overflow",
                ))
            }
        };
        i += 1;
    }
    pad_to_page(&mut off_file, os_page_size_bytes())?;
    off_file.flush()?;
    blob_file.flush()?;
    write_utf8_meta(
        meta_path,
        &Utf8ColumnMeta {
            row_count: rows.len() as u64,
            blob_len: blob_len as u64,
        },
    )?;
    Ok(())
}

/// Cold-path: materialize Stage-4 demo columnar files under `dir`.
///
/// Layout: `user_ids.bin`, `ages.bin`, `prices.bin` (+ `.meta`) — `total_rows` each.
pub fn write_stage4_columnar_demo(dir: &Path, total_rows: usize) -> io::Result<()> {
    write_i64_column_bin(&dir.join("user_ids.bin"), total_rows, |i| (i % 16) as i64)?;
    write_i64_column_bin(&dir.join("ages.bin"), total_rows, |i| (18 + (i % 40)) as i64)?;
    write_i64_column_bin(&dir.join("prices.bin"), total_rows, |i| {
        100 + (i as i64).wrapping_mul(3)
    })?;
    Ok(())
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

    #[test]
    #[cfg_attr(miri, ignore = "memmap2 file-backed mmap unsupported under Miri")]
    fn test_mmap_page_streaming_10000_rows_exact_remainder() {
        const TOTAL: usize = 10_000;
        let dir = std::env::temp_dir().join("tamil_mmap_page_stream_v2");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ages.bin");
        write_i64_column_bin(&path, TOTAL, |i| i as i64).unwrap();
        let mut stream = ColumnarFileStream::open_i64(&path).unwrap();
        let page_rows = stream.page_rows();
        assert_eq!(page_rows, os_page_size_bytes() / 8);
        assert_eq!(stream.total_rows(), TOTAL as u64);
        let expected_full = TOTAL / page_rows;
        let expected_rem = TOTAL % page_rows;
        let mut sum = 0usize;
        let mut full = 0usize;
        let mut last_n = 0usize;
        let mut last_residue = 0u8;
        while let Some(chunk) = stream.next_page_chunk() {
            sum += chunk.row_count as usize;
            if chunk.is_residue == 0 {
                assert_eq!(chunk.row_count as usize, page_rows);
                full += 1;
            } else {
                last_n = chunk.row_count as usize;
                last_residue = 1;
            }
        }
        assert_eq!(sum, TOTAL);
        assert_eq!(full, expected_full);
        if expected_rem == 0 {
            assert_eq!(last_residue, 0);
        } else {
            assert_eq!(last_residue, 1);
            assert_eq!(last_n, expected_rem);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "memmap2 file-backed mmap unsupported under Miri")]
    fn columnar_file_stream_pages_10000() {
        // Compatibility wrapper: same geometry as the named Stage-4 v2 test.
        const TOTAL: usize = 10_000;
        let dir = std::env::temp_dir().join("tamil_mmap_unit_v2");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("ages.bin");
        write_i64_column_bin(&path, TOTAL, |i| i as i64).unwrap();
        let mut stream = ColumnarFileStream::open_i64(&path).unwrap();
        let page_rows = stream.page_rows();
        let rem = TOTAL % page_rows;
        let mut sum = 0usize;
        while let Some(p) = stream.next_page_chunk() {
            sum += p.row_count as usize;
            if p.is_residue != 0 {
                assert_eq!(p.row_count as usize, rem);
            } else {
                assert_eq!(p.row_count as usize, page_rows);
            }
        }
        assert_eq!(sum, TOTAL);
    }

    #[test]
    #[cfg_attr(miri, ignore = "memmap2 file-backed mmap unsupported under Miri")]
    fn utf8_column_file_offset_blob_roundtrip() {
        let dir = std::env::temp_dir().join("tamil_utf8_col_v2");
        let _ = std::fs::create_dir_all(&dir);
        let offsets = dir.join("names.offsets");
        let blob = dir.join("names.blob");
        let meta = dir.join("names.meta");
        let rows: [&[u8]; 3] = ["அருண்".as_bytes(), "பிரியா".as_bytes(), "கண்ணன்".as_bytes()];
        write_utf8_column_files(&offsets, &blob, &meta, &rows).unwrap();
        let f = Utf8ColumnFile::open(&offsets, &blob, Some(&meta)).unwrap();
        assert_eq!(f.total_rows(), 3);
        assert_eq!(f.get_row(0), Some("அருண்"));
        assert_eq!(f.get_row(1), Some("பிரியா"));
        assert_eq!(f.get_row(2), Some("கண்ணன்"));
    }
}
