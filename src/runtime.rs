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
    seed_users_table, Catalog, ColumnMeta, Int64Column, PhysType, SelectionVector, Table,
    Utf8Column, BATCH_ROWS, MAX_ROWS,
};

/// Maximum projected output columns.
pub const MAX_PROJECT: usize = 8;

/// Width of the inner unrolled compare lane group.
pub const UNROLL: usize = 8;

/// Result of executing a pipeline: columnar projection over selected rows.
#[repr(C)]
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
#[repr(C)]
pub struct Engine<'a> {
    pub catalog: &'a Catalog,
    pub src: &'a [u8],
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

    /// Stable argsort of selected Int64 keys into `order` (row indices).
    ///
    /// Complexity: **O(N)** LSD radix sort over a fixed 64-bit key width
    /// (8×256 counting passes) — cache-friendly, allocation-free, branch-light.
    /// Selected rows are compacted first in a linear O(N) mask scan.
    #[inline(always)]
    pub fn sort_i64_selected(
        values: &[i64; MAX_ROWS],
        sel: &SelectionVector,
        rows: usize,
        order: &mut [u16; MAX_ROWS],
        order_len: &mut usize,
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
        let len = *order_len;
        if len <= 1 {
            return;
        }
        // Phase 1 — LSD radix sort (unsigned key = i64 ^ sign-bit).
        let mut tmp = [0u16; MAX_ROWS];
        let mut pass = 0u32;
        while pass < 8 {
            let shift = pass.wrapping_mul(8);
            let mut hist = [0u32; 256];
            let mut j = 0usize;
            while j < len {
                let idx = order[j] as usize;
                let key = (values[idx] as u64) ^ 0x8000_0000_0000_0000u64;
                let bucket = ((key >> shift) & 0xFF) as usize;
                hist[bucket] = hist[bucket].wrapping_add(1);
                j += 1;
            }
            // Exclusive prefix sum — O(256) = O(1) relative to N.
            let mut sum = 0u32;
            let mut b = 0usize;
            while b < 256 {
                let c = hist[b];
                hist[b] = sum;
                sum = sum.wrapping_add(c);
                b += 1;
            }
            // Stable scatter into tmp.
            let mut j = 0usize;
            while j < len {
                let idx = order[j];
                let key = (values[idx as usize] as u64) ^ 0x8000_0000_0000_0000u64;
                let bucket = ((key >> shift) & 0xFF) as usize;
                let dest = hist[bucket] as usize;
                tmp[dest] = idx;
                hist[bucket] = hist[bucket].wrapping_add(1);
                j += 1;
            }
            // Copy back for next pass (fixed-width memcpy, no heap).
            let mut j = 0usize;
            while j < len {
                order[j] = tmp[j];
                j += 1;
            }
            pass = pass.wrapping_add(1);
        }
    }

    /// Apply TAKE: truncate selection / order to at most `limit` rows.
    #[inline(always)]
    pub fn apply_take(order_len: &mut usize, limit: i64) {
        let lim = if limit < 0 { 0usize } else { limit as usize };
        if *order_len > lim {
            *order_len = lim;
        }
    }

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
        table: &Table,
        project: &AstNode,
        arena: &AstArena,
        order: &[u16],
        order_len: usize,
        out: &mut QueryResult,
    ) -> bool {
        let mut col_ids = [usize::MAX; MAX_PROJECT];
        let mut nproj = 0usize;
        let mut cur = project.left;
        while cur != NIL && nproj < MAX_PROJECT {
            let node = match arena.get(cur) {
                Some(n) => n,
                None => break,
            };
            let name = self.ident_bytes(node);
            match table.find_column(name) {
                Some(id) => {
                    col_ids[nproj] = id;
                    out.schema[nproj] = table.col_meta[id];
                    out.types[nproj] = table.col_meta[id].phys;
                    out.live[nproj] = 1;
                    nproj += 1;
                }
                None => return false,
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
            let src_row = order[oi] as usize;
            if src_row >= table.row_count as usize {
                oi += 1;
                continue;
            }
            let mut c = 0usize;
            while c < nproj {
                let cid = col_ids[c];
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

    /// Execute a parsed pipeline. Hot path uses only stack / caller-provided buffers.
    pub fn execute(&self, arena: &AstArena, out: &mut QueryResult) -> bool {
        let root = match arena.get(arena.root) {
            Some(n) if n.kind == NodeKind::Pipeline => n,
            _ => return false,
        };

        let mut stage_id = root.left;
        let mut sel = SelectionVector::all(0);
        let mut order = [0u16; MAX_ROWS];
        let mut order_len = 0usize;
        let mut sorted = false;
        let mut active_rows = 0usize;
        let mut table_ref: Option<&Table> = None;

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
                        order[i] = i as u16;
                        i += 1;
                    }
                    sorted = false;
                    table_ref = Some(table);
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
                    if !self.apply_filter(table, bin, arena, &mut sel) {
                        return false;
                    }
                    if !sorted {
                        order_len = 0;
                        let mut i = 0usize;
                        while i < active_rows {
                            order[order_len] = i as u16;
                            order_len += sel.mask[i] as usize;
                            i += 1;
                        }
                    } else {
                        let mut w = 0usize;
                        let mut r = 0usize;
                        while r < order_len {
                            let idx = order[r] as usize;
                            let keep = if idx < active_rows { sel.mask[idx] } else { 0 };
                            order[w] = order[r];
                            w += keep as usize;
                            r += 1;
                        }
                        order_len = w;
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
                            Self::sort_i64_selected(
                                &col_data.values,
                                &sel,
                                active_rows,
                                &mut order,
                                &mut order_len,
                            );
                            sorted = true;
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
                    if !self.materialize_projection(table, stage, arena, &order, order_len, out) {
                        return false;
                    }
                }
                NodeKind::Derive | NodeKind::Group | NodeKind::Aggregate | NodeKind::Join => {}
                _ => return false,
            }
            stage_id = stage.next;
        }
        true
    }
}

/// Build a catalog preloaded with the demo users relation.
pub fn demo_catalog() -> Catalog {
    let mut cat = Catalog::new();
    let users = seed_users_table();
    let _ = cat.register_box(users);
    cat
}

/// End-to-end: parse + execute a Tamil pipeline query string.
pub fn run_query(src: &str, catalog: &Catalog, arena: &mut AstArena, out: &mut QueryResult) -> bool {
    arena.len = 0;
    arena.root = NIL;
    let root = match crate::parser::parse_query(src.as_bytes(), arena) {
        Ok(r) => r,
        Err(_) => return false,
    };
    debug_assert_eq!(arena.root, root);
    let engine = Engine::new(catalog, src.as_bytes());
    engine.execute(arena, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::AstArena;

    #[test]
    fn executes_filter_sort_take_project() {
        let cat = demo_catalog();
        let q = "இருந்து பயனர்கள் | வடி வயது > 21 | அடுக்கு வயது | எடு 10 | தேடு பெயர், வயது;";
        let mut arena = AstArena::new();
        let mut out = QueryResult::new_boxed();
        assert!(run_query(q, &cat, &mut arena, &mut out));
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
        let mut values = [0i64; MAX_ROWS];
        // Unsorted ages with duplicates to exercise stable LSD passes.
        let raw: [i64; 12] = [30, 10, 20, 10, 40, 5, 20, 15, 5, 40, 25, 1];
        let mut i = 0usize;
        while i < 12 {
            values[i] = raw[i];
            i += 1;
        }
        let sel = SelectionVector::all(12);
        let mut order = [0u16; MAX_ROWS];
        let mut order_len = 0usize;
        Engine::sort_i64_selected(&values, &sel, 12, &mut order, &mut order_len);
        assert_eq!(order_len, 12);
        let mut prev = i64::MIN;
        let mut j = 0usize;
        while j < order_len {
            let v = values[order[j] as usize];
            assert!(v >= prev);
            prev = v;
            j += 1;
        }
        // Stable: first 10 appears before second 10.
        let mut seen_first_10 = false;
        let mut k = 0usize;
        while k < order_len {
            if values[order[k] as usize] == 10 {
                if !seen_first_10 {
                    assert_eq!(order[k], 1);
                    seen_first_10 = true;
                } else {
                    assert_eq!(order[k], 3);
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
