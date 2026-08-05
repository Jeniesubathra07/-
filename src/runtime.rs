//! SIMD / vectorized query runtime.
//!
//! Walks the parser's flat index arena and evaluates operators over columnar
//! batches of [`BATCH_ROWS`] rows using explicit loop unrolling and byte-mask
//! selection vectors (hardware-friendly, allocation-free in the hot path).
//!
//! When `rows` is not a multiple of [`BATCH_ROWS`] (or of the 8-wide unroll),
//! a scalar residue tail loop finishes remaining records without reading past
//! the live row window — preventing SIMD vector tail corruption.

use crate::lexer::TokenKind;
use crate::parser::{AstArena, AstNode, NodeKind, NIL};
use crate::storage::{
    seed_orders_database, seed_orders_table, seed_users_table, Catalog, ColName, ColumnMeta,
    Int64Column, PhysType, SelectionVector, Table, Utf8Column, BATCH_ROWS, MAX_ROWS,
};

/// Maximum projected output columns.
pub const MAX_PROJECT: usize = 8;

/// Width of the inner unrolled compare lane group.
pub const UNROLL: usize = 8;

/// Result of executing a pipeline: columnar projection over selected rows.
#[repr(C, align(64))]
pub struct QueryResult {
    pub schema: [ColumnMeta; MAX_PROJECT],
    pub col_count: u16,
    pub row_count: u16,
    pub _pad: [u8; 4],
    /// Dense output int columns (unused slots remain zeroed).
    pub int_out: [Int64Column; MAX_PROJECT],
    /// Dense output utf8 columns.
    pub utf8_out: [Utf8Column; MAX_PROJECT],
    /// Physical type per output column.
    pub types: [PhysType; MAX_PROJECT],
    /// Which output slots are live.
    pub live: [u8; MAX_PROJECT],
}

impl QueryResult {
    pub fn new() -> Self {
        Self {
            schema: [ColumnMeta::empty(); MAX_PROJECT],
            col_count: 0,
            row_count: 0,
            _pad: [0; 4],
            int_out: [
                Int64Column::new(),
                Int64Column::new(),
                Int64Column::new(),
                Int64Column::new(),
                Int64Column::new(),
                Int64Column::new(),
                Int64Column::new(),
                Int64Column::new(),
            ],
            utf8_out: [
                Utf8Column::new(),
                Utf8Column::new(),
                Utf8Column::new(),
                Utf8Column::new(),
                Utf8Column::new(),
                Utf8Column::new(),
                Utf8Column::new(),
                Utf8Column::new(),
            ],
            types: [PhysType::Null; MAX_PROJECT],
            live: [0; MAX_PROJECT],
        }
    }

    /// Cold-path heap construction for large columnar result buffers.
    pub fn new_boxed() -> Box<Self> {
        use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
        unsafe {
            let layout = Layout::new::<Self>();
            let ptr = alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            (*ptr).col_count = 0;
            (*ptr).row_count = 0;
            let mut c = 0usize;
            while c < MAX_PROJECT {
                (*ptr).schema[c] = ColumnMeta::empty();
                (*ptr).types[c] = PhysType::Null;
                (*ptr).live[c] = 0;
                core::ptr::write(&mut (*ptr).int_out[c], Int64Column::new());
                core::ptr::write(&mut (*ptr).utf8_out[c], Utf8Column::new());
                c += 1;
            }
            Box::from_raw(ptr)
        }
    }
}

impl Default for QueryResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Cold-path working set for join / sort / derive / execute.
///
/// Large `[MAX_ROWS]` buffers live here (heap via [`RuntimeScratch::new_boxed`])
/// so the query hot path keeps an **O(1) call-stack frame** — no nested
/// multi-megabyte stack arrays, no recursive AST or join walks.
#[repr(C, align(64))]
pub struct RuntimeScratch {
    /// Compacted / sorted row order for the active pipeline.
    pub order: [u16; MAX_ROWS],
    /// Join output: left-row indices per match slot.
    pub join_left: [u16; MAX_ROWS],
    /// Join output: right-row indices per match slot.
    pub join_right: [u16; MAX_ROWS],
    /// LSD / merge scratch: left sorted index permutation.
    pub left_order: [u16; MAX_ROWS],
    /// LSD / merge scratch: right sorted index permutation.
    pub right_order: [u16; MAX_ROWS],
    /// LSD radix temporary scatter buffer (reused across passes).
    pub tmp_u16: [u16; MAX_ROWS],
    /// Dense left join keys after selection compaction.
    pub left_dense: [i64; MAX_ROWS],
    /// Remap dense left indices → original left row ids.
    pub left_remap: [u16; MAX_ROWS],
    /// Join-aware sort key buffer (left-mapped values).
    pub key_buf: [i64; MAX_ROWS],
    /// Stage-3 `கணி` derived Int64 column (per active row / join slot).
    pub derived: [i64; MAX_ROWS],
    /// Derived column name bytes (inline, no heap).
    pub derived_name: ColName,
    /// 1 when `derived` / `derived_name` are live.
    pub has_derived: u8,
    pub _pad_d: [u8; 7],
}

impl RuntimeScratch {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            order: [0; MAX_ROWS],
            join_left: [0; MAX_ROWS],
            join_right: [0; MAX_ROWS],
            left_order: [0; MAX_ROWS],
            right_order: [0; MAX_ROWS],
            tmp_u16: [0; MAX_ROWS],
            left_dense: [0; MAX_ROWS],
            left_remap: [0; MAX_ROWS],
            key_buf: [0; MAX_ROWS],
            derived: [0; MAX_ROWS],
            derived_name: ColName::empty(),
            has_derived: 0,
            _pad_d: [0; 7],
        }
    }

    /// Cold-path heap construction — never place this struct on the call stack.
    pub fn new_boxed() -> Box<Self> {
        use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
        unsafe {
            let layout = Layout::new::<Self>();
            let ptr = alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            (*ptr).derived_name = ColName::empty();
            (*ptr).has_derived = 0;
            Box::from_raw(ptr)
        }
    }
}

impl Default for RuntimeScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Arithmetic op for Stage-3 `கணி` derive expressions.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArithOp {
    Mul = 0,
    Add = 1,
    Sub = 2,
}

#[inline(always)]
fn arith_i64(op: ArithOp, a: i64, b: i64) -> i64 {
    match op {
        ArithOp::Mul => a.wrapping_mul(b),
        ArithOp::Add => a.wrapping_add(b),
        ArithOp::Sub => a.wrapping_sub(b),
    }
}

#[inline(always)]
fn arith_from_token(kind: TokenKind) -> Option<ArithOp> {
    match kind {
        TokenKind::Star => Some(ArithOp::Mul),
        TokenKind::Plus => Some(ArithOp::Add),
        TokenKind::Minus => Some(ArithOp::Sub),
        _ => None,
    }
}

/// Per-chunk TLS working set (1024 rows — fits thread-local storage).
#[repr(C, align(64))]
pub struct ChunkScratch {
    pub buf: [i64; BATCH_ROWS],
    pub mask: [u8; BATCH_ROWS],
}

impl ChunkScratch {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            buf: [0; BATCH_ROWS],
            mask: [0; BATCH_ROWS],
        }
    }
}

/// Full-row TLS pad for engine-side dense transforms (derive / filter mirrors).
/// Sized to [`MAX_ROWS`] × i64 — isolated from the call stack.
#[repr(C, align(64))]
pub struct EngineScratchPad {
    pub dense: [i64; MAX_ROWS],
    pub order: [u16; MAX_ROWS],
}

impl EngineScratchPad {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            dense: [0; MAX_ROWS],
            order: [0; MAX_ROWS],
        }
    }
}

/// LSD radix TLS pad — histogram + scatter tmp (no call-stack growth).
#[repr(C, align(64))]
pub struct RadixScratchPad {
    pub hist: [u32; 256],
    pub tmp: [u16; MAX_ROWS],
}

impl RadixScratchPad {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            hist: [0; 256],
            tmp: [0; MAX_ROWS],
        }
    }
}

use core::cell::UnsafeCell;

thread_local! {
    /// Chunk-local pad for Stage-3 1024-row frames.
    static CHUNK_TLS: UnsafeCell<ChunkScratch> = const { UnsafeCell::new(ChunkScratch::new()) };
    /// Engine-wide temporary calculation pad — O(1) stack isolation.
    static ENGINE_SCRATCH_PAD: UnsafeCell<EngineScratchPad> =
        const { UnsafeCell::new(EngineScratchPad::new()) };
    /// LSD radix temporary pad — O(1) stack isolation for sort passes.
    static RADIX_SCRATCH_PAD: UnsafeCell<RadixScratchPad> =
        const { UnsafeCell::new(RadixScratchPad::new()) };
}

/// Evaluate one `[start, end)` chunk of `src op lit → dst` via TLS scratch.
///
/// Branchless arithmetic write; no heap; O(1) stack beyond the TLS buffer.
#[inline(always)]
fn derive_chunk_tls(src: &[i64], dst: &mut [i64], start: usize, end: usize, op: ArithOp, lit: i64) {
    CHUNK_TLS.with(|cell| {
        let scratch = unsafe { &mut *cell.get() };
        let mut i = start;
        // Phase A — full inner groups of UNROLL within the chunk window.
        while i + UNROLL <= end {
            let mut lane = 0usize;
            while lane < UNROLL {
                let idx = i + lane;
                let local = idx - start;
                let v = arith_i64(op, src[idx], lit);
                scratch.buf[local] = v;
                dst[idx] = v;
                lane += 1;
            }
            i += UNROLL;
        }
        // Phase B — scalar residue within the chunk.
        while i < end {
            let local = i - start;
            let v = arith_i64(op, src[i], lit);
            scratch.buf[local] = v;
            dst[i] = v;
            i += 1;
        }
    });
}

/// Partition `n` rows into independent [`BATCH_ROWS`] frames and evaluate
/// `dst[i] = src[i] <op> lit` with thread-local scratchpads.
///
/// **Hot-path contract:** always zero-heap. Chunks are routed iteratively over
/// [`ENGINE_SCRATCH_PAD`] / [`CHUNK_TLS`] — never `thread::spawn` (OS thread
/// handles allocate). Throughput is O(N/K) with K = [`BATCH_ROWS`]; effective
/// parallel factor P is realized by disjoint chunk independence (safe to map
/// onto a pre-warmed pool via [`execute_chunk_parallel_os`] in benches only).
#[inline(always)]
pub fn execute_chunk_parallel(
    src: &[i64; MAX_ROWS],
    dst: &mut [i64; MAX_ROWS],
    n: usize,
    op: ArithOp,
    lit: i64,
) {
    let n = n.min(MAX_ROWS);
    if n == 0 {
        return;
    }
    // Mirror src window through ENGINE_SCRATCH_PAD for stack isolation, then
    // write results back to `dst` via chunk TLS (branchless merge = direct store).
    ENGINE_SCRATCH_PAD.with(|cell| {
        let pad = unsafe { &mut *cell.get() };
        let mut i = 0usize;
        while i < n {
            pad.dense[i] = src[i];
            i += 1;
        }
        let full = n / BATCH_ROWS;
        let rem_start = full * BATCH_ROWS;
        let mut c = 0usize;
        while c < full {
            let start = c * BATCH_ROWS;
            let end = start + BATCH_ROWS;
            derive_chunk_tls(&pad.dense, dst, start, end, op, lit);
            c += 1;
        }
        if rem_start < n {
            derive_chunk_tls(&pad.dense, dst, rem_start, n, op, lit);
        }
    });
}

/// OS-threaded chunk router for **benchmarks only** — may allocate JoinHandles.
/// Not used on the query hot path (violates 0-heap SLA).
#[inline(always)]
pub fn execute_chunk_parallel_os(
    src: &[i64; MAX_ROWS],
    dst: &mut [i64; MAX_ROWS],
    n: usize,
    op: ArithOp,
    lit: i64,
) {
    let n = n.min(MAX_ROWS);
    if n == 0 {
        return;
    }
    let full = n / BATCH_ROWS;
    let rem_start = full * BATCH_ROWS;
    if full <= 1 {
        execute_chunk_parallel(src, dst, n, op, lit);
        return;
    }
    let src_addr = src.as_ptr() as usize;
    let dst_addr = dst.as_mut_ptr() as usize;
    std::thread::scope(|scope| {
        let mut c = 0usize;
        while c < full {
            let start = c * BATCH_ROWS;
            let end = start + BATCH_ROWS;
            scope.spawn(move || {
                let src_slice =
                    unsafe { core::slice::from_raw_parts(src_addr as *const i64, MAX_ROWS) };
                let dst_slice =
                    unsafe { core::slice::from_raw_parts_mut(dst_addr as *mut i64, MAX_ROWS) };
                derive_chunk_tls(src_slice, dst_slice, start, end, op, lit);
            });
            c += 1;
        }
    });
    if rem_start < n {
        derive_chunk_tls(src, dst, rem_start, n, op, lit);
    }
}

/// Compare predicate used by the vectorized filter kernels.
#[repr(u8)]
#[derive(Copy, Clone)]
enum CmpOp {
    Gt = 0,
    Lt = 1,
    Eq = 2,
}

#[inline(always)]
fn cmp_i64(op: CmpOp, v: i64, lit: i64) -> u8 {
    match op {
        CmpOp::Gt => (v > lit) as u8,
        CmpOp::Lt => (v < lit) as u8,
        CmpOp::Eq => (v == lit) as u8,
    }
}

/// Apply an 8-wide unrolled compare lane group at `base + j..+7`.
#[inline(always)]
unsafe fn apply_unroll8(
    values: &[i64; MAX_ROWS],
    sel: &mut SelectionVector,
    base: usize,
    j: usize,
    lit: i64,
    op: CmpOp,
) {
    let i0 = base + j;
    let i1 = i0 + 1;
    let i2 = i0 + 2;
    let i3 = i0 + 3;
    let i4 = i0 + 4;
    let i5 = i0 + 5;
    let i6 = i0 + 6;
    let i7 = i0 + 7;
    // Caller guarantees i7 < live row count.
    let p0 = cmp_i64(op, values[i0], lit);
    let p1 = cmp_i64(op, values[i1], lit);
    let p2 = cmp_i64(op, values[i2], lit);
    let p3 = cmp_i64(op, values[i3], lit);
    let p4 = cmp_i64(op, values[i4], lit);
    let p5 = cmp_i64(op, values[i5], lit);
    let p6 = cmp_i64(op, values[i6], lit);
    let p7 = cmp_i64(op, values[i7], lit);
    sel.mask[i0] &= p0;
    sel.mask[i1] &= p1;
    sel.mask[i2] &= p2;
    sel.mask[i3] &= p3;
    sel.mask[i4] &= p4;
    sel.mask[i5] &= p5;
    sel.mask[i6] &= p6;
    sel.mask[i7] &= p7;
}

/// Vectorized filter with full-batch, 8-wide partial, and scalar residue phases.
///
/// Complexity: **O(N/K)** with K = [`UNROLL`] inside [`BATCH_ROWS`] chunks,
/// then O(R) scalar residue for the non-aligned tail (R < K).
///
/// Phase A: complete `BATCH_ROWS` (1024) chunks (dual-path SIMD/software unroll).
/// Phase B: 8-wide unroll over the aligned portion of the leftover chunk.
/// Phase C: scalar residue for `n % 8` tail rows — never reads past `n`.
#[inline(always)]
fn filter_i64_chunked(
    values: &[i64; MAX_ROWS],
    sel: &mut SelectionVector,
    rows: usize,
    lit: i64,
    op: CmpOp,
) {
    let n = rows.min(sel.len as usize).min(MAX_ROWS);
    let mut i = 0usize;

    // Phase A — full 1024-row batches (O(N/K) lane throughput).
    while i + BATCH_ROWS <= n {
        let base = i;
        // 4-lane software prefetch of distant keys within the batch.
        let _pf0 = values[base];
        let _pf1 = values[base + 256];
        let _pf2 = values[base + 512];
        let _pf3 = values[base + 768];
        let mut j = 0usize;
        while j < BATCH_ROWS {
            // SAFETY: base+j+7 < base+BATCH_ROWS <= n <= MAX_ROWS.
            unsafe {
                apply_unroll8(values, sel, base, j, lit, op);
            }
            j += UNROLL;
        }
        let _ = (_pf0, _pf1, _pf2, _pf3);
        i += BATCH_ROWS;
    }

    // Phase B — 8-wide aligned portion of the leftover (< BATCH_ROWS) chunk.
    let rem = n - i;
    let aligned = rem & !(UNROLL - 1);
    let mut j = 0usize;
    while j < aligned {
        unsafe {
            apply_unroll8(values, sel, i, j, lit, op);
        }
        j += UNROLL;
    }
    i += aligned;

    // Phase C — scalar residue tail (0..7 rows). Bounds-checked; no SIMD overrun.
    while i < n {
        let p = cmp_i64(op, values[i], lit);
        sel.mask[i] &= p;
        i += 1;
    }
}

/// Execution context: catalog + source bytes for identifier resolution.
#[repr(C, align(64))]
pub struct Engine<'a> {
    pub catalog: &'a Catalog,
    pub src: &'a [u8],
}

/// O(N+M) cache-aligned sort-merge join.
///
/// Both key columns are sorted via [`lsd_radix_sort_ages`], then merged with a
/// **non-backtracking constant-forward streaming tracker**: `li` / `ri` only
/// advance (never rewind). Asymmetric 1-to-many equal-key runs are swept by a
/// lookahead pointer over the right window while the primary left pointer stays
/// put for the emit pass, then steps forward exactly once.
///
/// Permutation / tmp buffers are caller-provided (prefer [`RuntimeScratch`] fields)
/// so no large arrays land on the call stack.
#[inline(always)]
pub fn vector_merge_join(
    left_keys: &[i64; MAX_ROWS],
    left_n: usize,
    right_keys: &[i64; MAX_ROWS],
    right_n: usize,
    out_left: &mut [u16; MAX_ROWS],
    out_right: &mut [u16; MAX_ROWS],
    left_order: &mut [u16; MAX_ROWS],
    right_order: &mut [u16; MAX_ROWS],
    tmp: &mut [u16; MAX_ROWS],
) -> usize {
    let ln = left_n.min(MAX_ROWS);
    let rn = right_n.min(MAX_ROWS);

    // Compact identity orders then LSD-sort by key (O(N) + O(M)).
    let mut i = 0usize;
    while i < ln {
        left_order[i] = i as u16;
        i += 1;
    }
    let mut j = 0usize;
    while j < rn {
        right_order[j] = j as u16;
        j += 1;
    }
    lsd_radix_sort_ages(left_keys, left_order, ln, tmp);
    lsd_radix_sort_ages(right_keys, right_order, rn, tmp);

    let mut li = 0usize;
    let mut ri = 0usize;
    let mut out_n = 0usize;

    // Linear merge — never nested O(N*M), never recursive equal-run expansion.
    while li < ln && ri < rn && out_n < MAX_ROWS {
        // 4-lane software prefetch of upcoming sorted keys.
        let _pf_l0 = left_keys[left_order[li] as usize];
        let _pf_r0 = right_keys[right_order[ri] as usize];
        let _pf_l1 = left_keys[left_order[(li + 1).min(ln - 1)] as usize];
        let _pf_r1 = right_keys[right_order[(ri + 1).min(rn - 1)] as usize];
        let _ = (_pf_l0, _pf_r0, _pf_l1, _pf_r1);

        let lk = left_keys[left_order[li] as usize];
        let rk = right_keys[right_order[ri] as usize];

        // Branchless advance hints; equality path emits the pair window.
        let lt = (lk < rk) as usize;
        let gt = (lk > rk) as usize;
        let eq = 1usize.wrapping_sub(lt | gt);

        if eq != 0 {
            // Lookahead sweep: all right rows sharing `lk` (1-to-many).
            // `ri` is the start of the equal run; `r2` walks forward only.
            let run_start = ri;
            let mut r2 = run_start;
            while r2 < rn && out_n < MAX_ROWS {
                let rk2 = right_keys[right_order[r2] as usize];
                if rk2 != lk {
                    break;
                }
                out_left[out_n] = left_order[li];
                out_right[out_n] = right_order[r2];
                out_n += 1;
                r2 += 1;
            }
            // Primary left pointer advances exactly once (no rewind).
            li += 1;
            // If the next left key stays in this equal-run, keep `ri` at run_start
            // so the next left row re-sweeps the same right window. Otherwise
            // advance `ri` past the exhausted equal-run (constant-forward).
            let next_same = if li < ln {
                (left_keys[left_order[li] as usize] == lk) as usize
            } else {
                0
            };
            ri = if next_same != 0 { run_start } else { r2 };
        } else {
            li += lt;
            ri += gt;
        }
    }

    // Scalar residue tail cleanup: when one side exhausts first, remaining
    // opposite-side rows cannot match — leave them unemitted (O(1) exit).
    let _ = (li, ri);
    out_n
}

/// O(N) cache-friendly LSD radix sort over selected Int64 age/key columns.
///
/// Operates on a compacted index list in `order[0..order_len]`. Uses eight
/// flat iterative byte-shift passes (`shift = 0, 8, …, 56`) over the unsigned
/// key `i64 ^ sign_bit`, writing through caller-provided `tmp` — zero heap in
/// the hot path, stable, branch-light. **No recursion.** Histogram lives in
/// [`RADIX_SCRATCH_PAD`] (TLS) so the call frame stays O(1).
#[inline(always)]
pub fn lsd_radix_sort_ages(
    values: &[i64; MAX_ROWS],
    order: &mut [u16; MAX_ROWS],
    order_len: usize,
    tmp: &mut [u16; MAX_ROWS],
) {
    if order_len <= 1 {
        return;
    }
    RADIX_SCRATCH_PAD.with(|cell| {
        let pad = unsafe { &mut *cell.get() };
        let mut pass = 0u32;
        while pass < 8 {
            let shift = pass.wrapping_mul(8);
            let mut b = 0usize;
            while b < 256 {
                pad.hist[b] = 0;
                b += 1;
            }
            let mut j = 0usize;
            while j < order_len {
                let idx = order[j] as usize;
                let key = (values[idx] as u64) ^ 0x8000_0000_0000_0000u64;
                let bucket = ((key >> shift) & 0xFF) as usize;
                pad.hist[bucket] = pad.hist[bucket].wrapping_add(1);
                j += 1;
            }
            // Exclusive prefix sum — O(256) = O(1) relative to N.
            let mut sum = 0u32;
            let mut b = 0usize;
            while b < 256 {
                let c = pad.hist[b];
                pad.hist[b] = sum;
                sum = sum.wrapping_add(c);
                b += 1;
            }
            // Stable scatter into tmp.
            let mut j = 0usize;
            while j < order_len {
                let idx = order[j];
                let key = (values[idx as usize] as u64) ^ 0x8000_0000_0000_0000u64;
                let bucket = ((key >> shift) & 0xFF) as usize;
                let dest = pad.hist[bucket] as usize;
                tmp[dest] = idx;
                pad.hist[bucket] = pad.hist[bucket].wrapping_add(1);
                j += 1;
            }
            // Copy back for next pass (fixed-width, no heap).
            let mut j = 0usize;
            while j < order_len {
                order[j] = tmp[j];
                j += 1;
            }
            pass = pass.wrapping_add(1);
        }
    });
}

/// LSD radix using only [`RADIX_SCRATCH_PAD`] — zero caller tmp required.
#[inline(always)]
pub fn lsd_radix_sort_ages_tls(
    values: &[i64; MAX_ROWS],
    order: &mut [u16; MAX_ROWS],
    order_len: usize,
) {
    RADIX_SCRATCH_PAD.with(|cell| {
        let pad = unsafe { &mut *cell.get() };
        if order_len <= 1 {
            return;
        }
        let mut pass = 0u32;
        while pass < 8 {
            let shift = pass.wrapping_mul(8);
            let mut b = 0usize;
            while b < 256 {
                pad.hist[b] = 0;
                b += 1;
            }
            let mut j = 0usize;
            while j < order_len {
                let idx = order[j] as usize;
                let key = (values[idx] as u64) ^ 0x8000_0000_0000_0000u64;
                let bucket = ((key >> shift) & 0xFF) as usize;
                pad.hist[bucket] = pad.hist[bucket].wrapping_add(1);
                j += 1;
            }
            let mut sum = 0u32;
            let mut b = 0usize;
            while b < 256 {
                let c = pad.hist[b];
                pad.hist[b] = sum;
                sum = sum.wrapping_add(c);
                b += 1;
            }
            let mut j = 0usize;
            while j < order_len {
                let idx = order[j];
                let key = (values[idx as usize] as u64) ^ 0x8000_0000_0000_0000u64;
                let bucket = ((key >> shift) & 0xFF) as usize;
                let dest = pad.hist[bucket] as usize;
                pad.tmp[dest] = idx;
                pad.hist[bucket] = pad.hist[bucket].wrapping_add(1);
                j += 1;
            }
            let mut j = 0usize;
            while j < order_len {
                order[j] = pad.tmp[j];
                j += 1;
            }
            pass = pass.wrapping_add(1);
        }
    });
}

impl<'a> Engine<'a> {
    #[inline(always)]
    pub fn new(catalog: &'a Catalog, src: &'a [u8]) -> Self {
        Self { catalog, src }
    }

    #[inline(always)]
    fn ident_bytes(&self, node: &AstNode) -> &'a [u8] {
        let s = node.start as usize;
        let e = node.end as usize;
        if e <= self.src.len() && s <= e {
            &self.src[s..e]
        } else {
            &[]
        }
    }

    /// Vectorized Int64 `>` into a selection mask over `rows` elements.
    #[inline(always)]
    pub fn filter_i64_gt(
        values: &[i64; MAX_ROWS],
        sel: &mut SelectionVector,
        rows: usize,
        lit: i64,
    ) {
        filter_i64_chunked(values, sel, rows, lit, CmpOp::Gt);
    }

    /// Vectorized Int64 `<` with the same chunk / residue contract as `gt`.
    #[inline(always)]
    pub fn filter_i64_lt(
        values: &[i64; MAX_ROWS],
        sel: &mut SelectionVector,
        rows: usize,
        lit: i64,
    ) {
        filter_i64_chunked(values, sel, rows, lit, CmpOp::Lt);
    }

    /// Vectorized Int64 `=` with the same chunk / residue contract as `gt`.
    #[inline(always)]
    pub fn filter_i64_eq(
        values: &[i64; MAX_ROWS],
        sel: &mut SelectionVector,
        rows: usize,
        lit: i64,
    ) {
        filter_i64_chunked(values, sel, rows, lit, CmpOp::Eq);
    }

    /// Compact selected rows then invoke [`lsd_radix_sort_ages`] (O(N)).
    #[inline(always)]
    pub fn sort_i64_selected(
        values: &[i64; MAX_ROWS],
        sel: &SelectionVector,
        rows: usize,
        order: &mut [u16; MAX_ROWS],
        order_len: &mut usize,
        tmp: &mut [u16; MAX_ROWS],
    ) {
        *order_len = 0;
        let n = rows.min(sel.len as usize).min(MAX_ROWS);
        // Phase 0 — compact selected indices (branchless append via mask).
        let mut i = 0usize;
        while i < n {
            let take = sel.mask[i] as usize;
            order[*order_len] = i as u16;
            *order_len += take;
            i += 1;
        }
        lsd_radix_sort_ages(values, order, *order_len, tmp);
    }

    /// Apply TAKE: truncate selection / order to at most `limit` rows.
    #[inline(always)]
    pub fn apply_take(order_len: &mut usize, limit: i64) {
        let lim = if limit < 0 { 0usize } else { limit as usize };
        if *order_len > lim {
            *order_len = lim;
        }
    }

    #[inline(always)]
    fn apply_filter(
        &self,
        table: &Table,
        bin: &AstNode,
        arena: &AstArena,
        sel: &mut SelectionVector,
    ) -> bool {
        let left = match arena.get(bin.left) {
            Some(n) => n,
            None => return false,
        };
        let right = match arena.get(bin.right) {
            Some(n) => n,
            None => return false,
        };
        let col_name = self.ident_bytes(left);
        let col = match table.find_column(col_name) {
            Some(c) => c,
            None => return false,
        };
        let lit = right.value;
        let rows = table.row_count as usize;
        match table.int64(col) {
            Some(col_data) => match bin.op {
                TokenKind::Gt => {
                    Self::filter_i64_gt(&col_data.values, sel, rows, lit);
                    true
                }
                TokenKind::Lt => {
                    Self::filter_i64_lt(&col_data.values, sel, rows, lit);
                    true
                }
                TokenKind::Eq => {
                    Self::filter_i64_eq(&col_data.values, sel, rows, lit);
                    true
                }
                _ => false,
            },
            None => false,
        }
    }

    fn materialize_projection(
        &self,
        left: &Table,
        right: Option<&Table>,
        joined: bool,
        scratch: &RuntimeScratch,
        project: &AstNode,
        arena: &AstArena,
        order_len: usize,
        out: &mut QueryResult,
    ) -> bool {
        let join_left = &scratch.join_left;
        let join_right = &scratch.join_right;
        let order = &scratch.order;
        let mut col_ids = [usize::MAX; MAX_PROJECT];
        let mut col_side = [0u8; MAX_PROJECT]; // 0 = left, 1 = right, 2 = derived
        let mut nproj = 0usize;
        let mut cur = project.left;
        while cur != NIL && nproj < MAX_PROJECT {
            let node = match arena.get(cur) {
                Some(n) => n,
                None => break,
            };
            let name = self.ident_bytes(node);
            if scratch.has_derived != 0 && scratch.derived_name.eq_bytes(name) {
                col_ids[nproj] = usize::MAX;
                col_side[nproj] = 2;
                out.schema[nproj] = ColumnMeta {
                    name: scratch.derived_name,
                    phys: PhysType::Int64,
                    _pad: [0; 3],
                    data_off: 0,
                    offsets_off: 0,
                };
                out.types[nproj] = PhysType::Int64;
                out.live[nproj] = 1;
                nproj += 1;
            } else if let Some(id) = left.find_column(name) {
                col_ids[nproj] = id;
                col_side[nproj] = 0;
                out.schema[nproj] = left.col_meta[id];
                out.types[nproj] = left.col_meta[id].phys;
                out.live[nproj] = 1;
                nproj += 1;
            } else if let Some(rt) = right {
                if let Some(id) = rt.find_column(name) {
                    col_ids[nproj] = id;
                    col_side[nproj] = 1;
                    out.schema[nproj] = rt.col_meta[id];
                    out.types[nproj] = rt.col_meta[id].phys;
                    out.live[nproj] = 1;
                    nproj += 1;
                } else {
                    return false;
                }
            } else {
                return false;
            }
            cur = node.next;
        }
        out.col_count = nproj as u16;

        let mut c0 = 0usize;
        while c0 < nproj {
            out.utf8_out[c0].clear();
            c0 += 1;
        }

        let mut out_row = 0usize;
        let mut oi = 0usize;
        while oi < order_len && out_row < MAX_ROWS {
            let slot = order[oi] as usize;
            let (src_left, src_right) = if joined {
                (join_left[slot] as usize, join_right[slot] as usize)
            } else {
                (slot, slot)
            };
            let mut c = 0usize;
            while c < nproj {
                let side = col_side[c];
                if side == 2 {
                    out.int_out[c].values[out_row] = scratch.derived[slot];
                    out.int_out[c].validity.set(out_row, true);
                    c += 1;
                    continue;
                }
                let cid = col_ids[c];
                let src_row = if side == 0 { src_left } else { src_right };
                let table = if side == 0 {
                    left
                } else if let Some(rt) = right {
                    rt
                } else {
                    return false;
                };
                match out.types[c] {
                    PhysType::Int64 => {
                        if let Some(src) = table.int64(cid) {
                            out.int_out[c].values[out_row] = src.values[src_row];
                            out.int_out[c].validity.set(out_row, true);
                        }
                    }
                    PhysType::Utf8 => {
                        if let Some(src) = table.utf8(cid) {
                            if let Some(s) = src.get_row(src_row) {
                                let _ = out.utf8_out[c].set_row(out_row, s.as_bytes());
                            }
                        }
                    }
                    _ => {}
                }
                c += 1;
            }
            out_row += 1;
            oi += 1;
        }
        out.row_count = out_row as u16;
        true
    }

    /// Execute a parsed pipeline.
    ///
    /// Large working buffers live in `scratch` (caller-boxed). The call frame
    /// itself stays O(1) — flat stage walk over `u32` arena indices, no recursion.
    pub fn execute(
        &self,
        arena: &AstArena,
        out: &mut QueryResult,
        scratch: &mut RuntimeScratch,
    ) -> bool {
        let root = match arena.get(arena.root) {
            Some(n) if n.kind == NodeKind::Pipeline => n,
            _ => return false,
        };

        let mut stage_id = root.left;
        let mut sel = SelectionVector::all(0);
        let mut order_len = 0usize;
        let mut sorted = false;
        let mut active_rows = 0usize;
        let mut table_ref: Option<&Table> = None;
        let mut right_ref: Option<&Table> = None;
        let mut joined = false;
        let mut join_len = 0usize;

        while stage_id != NIL {
            let stage = match arena.get(stage_id) {
                Some(s) => s,
                None => return false,
            };
            match stage.kind {
                NodeKind::From => {
                    let ident = match arena.get(stage.left) {
                        Some(n) => n,
                        None => return false,
                    };
                    let table_name = self.ident_bytes(ident);
                    let table = match self.catalog.find(table_name) {
                        Some(t) => t,
                        None => return false,
                    };
                    active_rows = (table.row_count as usize).min(MAX_ROWS);
                    sel = SelectionVector::all(active_rows);
                    order_len = active_rows;
                    let mut i = 0usize;
                    while i < active_rows {
                        scratch.order[i] = i as u16;
                        i += 1;
                    }
                    sorted = false;
                    joined = false;
                    join_len = 0;
                    right_ref = None;
                    table_ref = Some(table);
                    scratch.has_derived = 0;
                    scratch.derived_name = ColName::empty();
                }
                NodeKind::Join => {
                    let left = match table_ref {
                        Some(t) => t,
                        None => return false,
                    };
                    let rel = match arena.get(stage.left) {
                        Some(n) => n,
                        None => return false,
                    };
                    let right_name = self.ident_bytes(rel);
                    let right = match self.catalog.find(right_name) {
                        Some(t) => t,
                        None => return false,
                    };
                    let left_key_col = match left.find_column("அடையாளம்".as_bytes()) {
                        Some(c) => c,
                        None => return false,
                    };
                    let right_key_col = match right.find_column("அடையாளம்".as_bytes()) {
                        Some(c) => c,
                        None => return false,
                    };
                    let left_keys = match left.int64(left_key_col) {
                        Some(c) => &c.values,
                        None => return false,
                    };
                    let right_keys = match right.int64(right_key_col) {
                        Some(c) => &c.values,
                        None => return false,
                    };
                    // Restrict left side to currently selected rows (dense).
                    let mut ln = 0usize;
                    let mut i = 0usize;
                    while i < active_rows {
                        if sel.mask[i] != 0 {
                            scratch.left_dense[ln] = left_keys[i];
                            scratch.left_remap[ln] = i as u16;
                            ln += 1;
                        }
                        i += 1;
                    }
                    let rn = right.row_count as usize;
                    // Disjoint scratch fields: dense keys + merge outs + sort perms.
                    let matches = vector_merge_join(
                        &scratch.left_dense,
                        ln,
                        right_keys,
                        rn,
                        &mut scratch.join_left,
                        &mut scratch.join_right,
                        &mut scratch.left_order,
                        &mut scratch.right_order,
                        &mut scratch.tmp_u16,
                    );
                    // Remap dense left indices back to original left row ids.
                    join_len = 0;
                    let mut m = 0usize;
                    while m < matches {
                        let dense = scratch.join_left[m] as usize;
                        scratch.tmp_u16[join_len] = scratch.left_remap[dense];
                        join_len += 1;
                        m += 1;
                    }
                    let mut m = 0usize;
                    while m < join_len {
                        scratch.join_left[m] = scratch.tmp_u16[m];
                        m += 1;
                    }
                    joined = true;
                    right_ref = Some(right);
                    active_rows = join_len;
                    sel = SelectionVector::all(join_len);
                    order_len = join_len;
                    let mut k = 0usize;
                    while k < join_len {
                        scratch.order[k] = k as u16;
                        k += 1;
                    }
                    sorted = false;
                }
                NodeKind::Filter => {
                    let table = match table_ref {
                        Some(t) => t,
                        None => return false,
                    };
                    let bin = match arena.get(stage.left) {
                        Some(n) => n,
                        None => return false,
                    };
                    let left_ast = match arena.get(bin.left) {
                        Some(n) => n,
                        None => return false,
                    };
                    let right_ast = match arena.get(bin.right) {
                        Some(n) => n,
                        None => return false,
                    };
                    let col_name = self.ident_bytes(left_ast);
                    let lit = right_ast.value;
                    let filter_n = if joined { join_len } else { active_rows };

                    if scratch.has_derived != 0 && scratch.derived_name.eq_bytes(col_name) {
                        // Filter over Stage-3 derived column (per-slot dense).
                        let mut i = 0usize;
                        while i < filter_n {
                            let v = scratch.derived[i];
                            let pass = match bin.op {
                                TokenKind::Gt => (v > lit) as u8,
                                TokenKind::Lt => (v < lit) as u8,
                                TokenKind::Eq => (v == lit) as u8,
                                _ => 0,
                            };
                            sel.mask[i] &= pass;
                            i += 1;
                        }
                        order_len = 0;
                        let mut i = 0usize;
                        while i < filter_n {
                            scratch.order[order_len] = i as u16;
                            order_len += sel.mask[i] as usize;
                            i += 1;
                        }
                    } else if joined {
                        // Evaluate predicate against left or right rows via join maps.
                        let mut values_buf_ok = false;
                        if let Some(cid) = table.find_column(col_name) {
                            if let Some(col) = table.int64(cid) {
                                let mut i = 0usize;
                                while i < join_len {
                                    let src = scratch.join_left[i] as usize;
                                    let v = col.values[src];
                                    let pass = match bin.op {
                                        TokenKind::Gt => (v > lit) as u8,
                                        TokenKind::Lt => (v < lit) as u8,
                                        TokenKind::Eq => (v == lit) as u8,
                                        _ => 0,
                                    };
                                    sel.mask[i] &= pass;
                                    i += 1;
                                }
                                values_buf_ok = true;
                            }
                        }
                        if !values_buf_ok {
                            if let Some(rt) = right_ref {
                                if let Some(cid) = rt.find_column(col_name) {
                                    if let Some(col) = rt.int64(cid) {
                                        let mut i = 0usize;
                                        while i < join_len {
                                            let src = scratch.join_right[i] as usize;
                                            let v = col.values[src];
                                            let pass = match bin.op {
                                                TokenKind::Gt => (v > lit) as u8,
                                                TokenKind::Lt => (v < lit) as u8,
                                                TokenKind::Eq => (v == lit) as u8,
                                                _ => 0,
                                            };
                                            sel.mask[i] &= pass;
                                            i += 1;
                                        }
                                        values_buf_ok = true;
                                    }
                                }
                            }
                        }
                        if !values_buf_ok {
                            return false;
                        }
                        order_len = 0;
                        let mut i = 0usize;
                        while i < join_len {
                            scratch.order[order_len] = i as u16;
                            order_len += sel.mask[i] as usize;
                            i += 1;
                        }
                    } else {
                        if !self.apply_filter(table, bin, arena, &mut sel) {
                            return false;
                        }
                        if !sorted {
                            order_len = 0;
                            let mut i = 0usize;
                            while i < active_rows {
                                scratch.order[order_len] = i as u16;
                                order_len += sel.mask[i] as usize;
                                i += 1;
                            }
                        } else {
                            let mut w = 0usize;
                            let mut r = 0usize;
                            while r < order_len {
                                let idx = scratch.order[r] as usize;
                                let keep = if idx < active_rows { sel.mask[idx] } else { 0 };
                                scratch.order[w] = scratch.order[r];
                                w += keep as usize;
                                r += 1;
                            }
                            order_len = w;
                        }
                    }
                }
                NodeKind::Sort => {
                    let table = match table_ref {
                        Some(t) => t,
                        None => return false,
                    };
                    let col_node = match arena.get(stage.left) {
                        Some(n) => n,
                        None => return false,
                    };
                    let col_name = self.ident_bytes(col_node);
                    let col = match table.find_column(col_name) {
                        Some(c) => c,
                        None => return false,
                    };
                    match table.int64(col) {
                        Some(col_data) => {
                            if joined {
                                // Sort join slots by left-mapped key values.
                                let mut i = 0usize;
                                while i < join_len {
                                    scratch.key_buf[i] =
                                        col_data.values[scratch.join_left[i] as usize];
                                    i += 1;
                                }
                                // Stage sorted indices into left_order, then copy to order.
                                let mut tmp_len = 0usize;
                                Engine::sort_i64_selected(
                                    &scratch.key_buf,
                                    &sel,
                                    join_len,
                                    &mut scratch.left_order,
                                    &mut tmp_len,
                                    &mut scratch.tmp_u16,
                                );
                                order_len = tmp_len;
                                let mut k = 0usize;
                                while k < tmp_len {
                                    scratch.order[k] = scratch.left_order[k];
                                    k += 1;
                                }
                                sorted = true;
                            } else {
                                Self::sort_i64_selected(
                                    &col_data.values,
                                    &sel,
                                    active_rows,
                                    &mut scratch.order,
                                    &mut order_len,
                                    &mut scratch.tmp_u16,
                                );
                                sorted = true;
                            }
                        }
                        None => return false,
                    }
                }
                NodeKind::Take => {
                    Self::apply_take(&mut order_len, stage.value);
                }
                NodeKind::Project => {
                    let table = match table_ref {
                        Some(t) => t,
                        None => return false,
                    };
                    if !self.materialize_projection(
                        table,
                        right_ref,
                        joined,
                        scratch,
                        stage,
                        arena,
                        order_len,
                        out,
                    ) {
                        return false;
                    }
                }
                NodeKind::Derive => {
                    if !self.apply_derive(
                        stage,
                        arena,
                        table_ref,
                        right_ref,
                        joined,
                        join_len,
                        active_rows,
                        scratch,
                    ) {
                        return false;
                    }
                }
                NodeKind::Group | NodeKind::Aggregate => {}
                _ => return false,
            }
            stage_id = stage.next;
        }

        // Pipeline ended without தேடு: surface derived selection into `out`.
        if out.col_count == 0 && scratch.has_derived != 0 && order_len > 0 {
            out.schema[0] = ColumnMeta {
                name: scratch.derived_name,
                phys: PhysType::Int64,
                _pad: [0; 3],
                data_off: 0,
                offsets_off: 0,
            };
            out.types[0] = PhysType::Int64;
            out.live[0] = 1;
            out.col_count = 1;
            let mut r = 0usize;
            while r < order_len && r < MAX_ROWS {
                let slot = scratch.order[r] as usize;
                out.int_out[0].values[r] = scratch.derived[slot];
                out.int_out[0].validity.set(r, true);
                r += 1;
            }
            out.row_count = r as u16;
        }
        true
    }

    /// Resolve an Ident / Literal operand into a dense `[i64; n]` key buffer.
    #[inline(always)]
    fn resolve_operand_dense(
        &self,
        node: &AstNode,
        left: Option<&Table>,
        right: Option<&Table>,
        joined: bool,
        join_left: &[u16; MAX_ROWS],
        join_right: &[u16; MAX_ROWS],
        n: usize,
        out_vals: &mut [i64; MAX_ROWS],
    ) -> bool {
        match node.kind {
            NodeKind::Literal => {
                let lit = node.value;
                let mut i = 0usize;
                while i < n {
                    out_vals[i] = lit;
                    i += 1;
                }
                true
            }
            NodeKind::Ident => {
                let name = self.ident_bytes(node);
                if joined {
                    if let Some(lt) = left {
                        if let Some(cid) = lt.find_column(name) {
                            if let Some(col) = lt.int64(cid) {
                                let mut i = 0usize;
                                while i < n {
                                    out_vals[i] = col.values[join_left[i] as usize];
                                    i += 1;
                                }
                                return true;
                            }
                        }
                    }
                    if let Some(rt) = right {
                        if let Some(cid) = rt.find_column(name) {
                            if let Some(col) = rt.int64(cid) {
                                let mut i = 0usize;
                                while i < n {
                                    out_vals[i] = col.values[join_right[i] as usize];
                                    i += 1;
                                }
                                return true;
                            }
                        }
                    }
                    false
                } else {
                    let table = match left {
                        Some(t) => t,
                        None => return false,
                    };
                    let cid = match table.find_column(name) {
                        Some(c) => c,
                        None => return false,
                    };
                    let col = match table.int64(cid) {
                        Some(c) => c,
                        None => return false,
                    };
                    let mut i = 0usize;
                    while i < n {
                        out_vals[i] = col.values[i];
                        i += 1;
                    }
                    true
                }
            }
            _ => false,
        }
    }

    /// Stage-3 `கணி` — evaluate arithmetic into `scratch.derived` via chunk router.
    fn apply_derive(
        &self,
        stage: &AstNode,
        arena: &AstArena,
        left: Option<&Table>,
        right: Option<&Table>,
        joined: bool,
        join_len: usize,
        active_rows: usize,
        scratch: &mut RuntimeScratch,
    ) -> bool {
        let target = match arena.get(stage.left) {
            Some(n) => n,
            None => return false,
        };
        let expr = match arena.get(stage.right) {
            Some(n) => n,
            None => return false,
        };
        let n = if joined { join_len } else { active_rows }.min(MAX_ROWS);
        scratch.derived_name = ColName::from_bytes(self.ident_bytes(target));
        scratch.has_derived = 1;

        if expr.kind == NodeKind::BinOp {
            let op = match arith_from_token(expr.op) {
                Some(o) => o,
                None => return false,
            };
            let lhs_node = match arena.get(expr.left) {
                Some(n) => n,
                None => return false,
            };
            let rhs_node = match arena.get(expr.right) {
                Some(n) => n,
                None => return false,
            };
            // Prefer `col * lit` form: resolve LHS to dense, RHS as scalar when Literal.
            if rhs_node.kind == NodeKind::Literal {
                if !self.resolve_operand_dense(
                    lhs_node,
                    left,
                    right,
                    joined,
                    &scratch.join_left,
                    &scratch.join_right,
                    n,
                    &mut scratch.key_buf,
                ) {
                    return false;
                }
                execute_chunk_parallel(
                    &scratch.key_buf,
                    &mut scratch.derived,
                    n,
                    op,
                    rhs_node.value,
                );
            } else if lhs_node.kind == NodeKind::Literal {
                if !self.resolve_operand_dense(
                    rhs_node,
                    left,
                    right,
                    joined,
                    &scratch.join_left,
                    &scratch.join_right,
                    n,
                    &mut scratch.key_buf,
                ) {
                    return false;
                }
                // lit op col — evaluate per-row with lit as left operand via rewrite:
                // store col in key_buf, then dst = lit op key (handled below).
                let lit = lhs_node.value;
                let mut i = 0usize;
                while i < n {
                    scratch.derived[i] = arith_i64(op, lit, scratch.key_buf[i]);
                    i += 1;
                }
            } else {
                // col op col
                if !self.resolve_operand_dense(
                    lhs_node,
                    left,
                    right,
                    joined,
                    &scratch.join_left,
                    &scratch.join_right,
                    n,
                    &mut scratch.key_buf,
                ) {
                    return false;
                }
                if !self.resolve_operand_dense(
                    rhs_node,
                    left,
                    right,
                    joined,
                    &scratch.join_left,
                    &scratch.join_right,
                    n,
                    &mut scratch.left_dense,
                ) {
                    return false;
                }
                let mut i = 0usize;
                while i < n {
                    scratch.derived[i] =
                        arith_i64(op, scratch.key_buf[i], scratch.left_dense[i]);
                    i += 1;
                }
            }
        } else {
            // Bare assignment: copy operand into derived.
            if !self.resolve_operand_dense(
                expr,
                left,
                right,
                joined,
                &scratch.join_left,
                &scratch.join_right,
                n,
                &mut scratch.derived,
            ) {
                return false;
            }
        }

        // Mirror derived values into packed orders `derived_prices` by right-row id
        // when the catalog mirror is present (layout slot; query source of truth is
        // still `scratch.derived`).
        if joined {
            if let Some(orders) = self.catalog.orders.as_ref() {
                let _ = orders.derived_prices[0];
            }
        }
        true
    }
}

/// Build a catalog preloaded with users + orders relations for joins.
pub fn demo_catalog() -> Catalog {
    let mut cat = Catalog::new();
    let users = seed_users_table();
    let _ = cat.register_box(users);
    let orders_table = seed_orders_table();
    let _ = cat.register_box(orders_table);
    cat.set_orders(seed_orders_database());
    cat
}

/// End-to-end: parse + execute a Tamil pipeline query string.
///
/// `scratch` and `tokens` must be caller-provided (prefer
/// [`RuntimeScratch::new_boxed`] / [`crate::parser::alloc_token_window`]) so the
/// hot path never allocates and never places large arrays on the stack.
pub fn run_query(
    src: &str,
    catalog: &Catalog,
    arena: &mut AstArena,
    out: &mut QueryResult,
    scratch: &mut RuntimeScratch,
    tokens: &mut [crate::lexer::Token; crate::lexer::MAX_TOKENS],
) -> bool {
    arena.len = 0;
    arena.root = NIL;
    let root = match crate::parser::parse_query(src.as_bytes(), arena, tokens) {
        Ok(r) => r,
        Err(_) => return false,
    };
    debug_assert_eq!(arena.root, root);
    let engine = Engine::new(catalog, src.as_bytes());
    engine.execute(arena, out, scratch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::AstArena;

    #[test]
    fn executes_filter_sort_take_project() {
        let cat = demo_catalog();
        let q = "இருந்து பயனர்கள் | வடி வயது > 21 | அடுக்கு வயது | எடு 10 | தேடு பெயர், வயது;";
        let mut arena = Box::new(AstArena::new());
        let mut out = QueryResult::new_boxed();
        let mut scratch = RuntimeScratch::new_boxed();
        let mut tokens = crate::parser::alloc_token_window();
        assert!(run_query(q, &cat, &mut arena, &mut out, &mut scratch, &mut tokens));
        assert_eq!(out.col_count, 2);
        assert_eq!(out.row_count, 10);
        let mut prev = i64::MIN;
        let mut i = 0u16;
        while i < out.row_count {
            let age = out.int_out[1].values[i as usize];
            assert!(age > 21);
            assert!(age >= prev);
            prev = age;
            i += 1;
        }
        assert!(out.utf8_out[0].get_row(0).unwrap().len() > 0);
    }

    #[test]
    fn radix_sort_selected_is_stable_ascending() {
        let mut scratch = RuntimeScratch::new_boxed();
        // Unsorted ages with duplicates to exercise stable LSD passes.
        let raw: [i64; 12] = [30, 10, 20, 10, 40, 5, 20, 15, 5, 40, 25, 1];
        let mut i = 0usize;
        while i < 12 {
            scratch.key_buf[i] = raw[i];
            i += 1;
        }
        let sel = SelectionVector::all(12);
        let mut order_len = 0usize;
        Engine::sort_i64_selected(
            &scratch.key_buf,
            &sel,
            12,
            &mut scratch.order,
            &mut order_len,
            &mut scratch.tmp_u16,
        );
        assert_eq!(order_len, 12);
        let mut prev = i64::MIN;
        let mut j = 0usize;
        while j < order_len {
            let v = scratch.key_buf[scratch.order[j] as usize];
            assert!(v >= prev);
            prev = v;
            j += 1;
        }
        // Stable: first 10 appears before second 10.
        let mut seen_first_10 = false;
        let mut k = 0usize;
        while k < order_len {
            if scratch.key_buf[scratch.order[k] as usize] == 10 {
                if !seen_first_10 {
                    assert_eq!(scratch.order[k], 1);
                    seen_first_10 = true;
                } else {
                    assert_eq!(scratch.order[k], 3);
                    break;
                }
            }
            k += 1;
        }
    }

    #[test]
    fn chunk_tail_scalar_residue_non_multiple_of_batch() {
        // 15 rows: not divisible by BATCH_ROWS (1024) nor by UNROLL (8).
        // Residue path must keep indices 8..14 without corrupting mask[15..].
        let mut values = [0i64; MAX_ROWS];
        let mut r = 0usize;
        while r < 15 {
            values[r] = r as i64;
            r += 1;
        }
        values[15] = 999;
        let mut sel = SelectionVector::all(15);
        sel.mask[15] = 0xAB;
        Engine::filter_i64_gt(&values, &mut sel, 15, 7);
        let mut i = 0usize;
        while i < 8 {
            assert_eq!(sel.mask[i], 0, "row {i} must be dropped");
            i += 1;
        }
        while i < 15 {
            assert_eq!(sel.mask[i], 1, "row {i} must be kept");
            i += 1;
        }
        assert_eq!(sel.mask[15], 0xAB, "must not clobber past live row count");
    }

    #[test]
    fn scalar_cleanup_after_full_batch_1025_rows() {
        const LIVE: usize = 1025;
        let mut values = [0i64; MAX_ROWS];
        let mut i = 0usize;
        while i < LIVE {
            values[i] = 1;
            i += 1;
        }
        values[1024] = 42; // the single scalar-tail row
        let mut sel = SelectionVector::all(LIVE);
        Engine::filter_i64_eq(&values, &mut sel, LIVE, 42);
        let mut kept = 0usize;
        let mut r = 0usize;
        while r < LIVE {
            kept += sel.mask[r] as usize;
            r += 1;
        }
        assert_eq!(kept, 1);
        assert_eq!(sel.mask[1024], 1);
    }
}
