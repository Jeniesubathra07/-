//! SIMD / vectorized query runtime.
//!
//! Walks the parser's flat index arena and evaluates operators over columnar
//! batches of [`BATCH_ROWS`] rows using explicit loop unrolling and byte-mask
//! selection vectors (hardware-friendly, allocation-free in the hot path).

use crate::lexer::TokenKind;
use crate::parser::{AstArena, AstNode, NodeKind, NIL};
use crate::storage::{
    seed_users_table, Catalog, ColumnMeta, Int64Column, PhysType, SelectionVector, Table,
    Utf8Column, BATCH_ROWS, MAX_ROWS,
};

/// Maximum projected output columns.
pub const MAX_PROJECT: usize = 8;

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
}

impl Default for QueryResult {
    fn default() -> Self {
        Self::new()
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

    /// Vectorized Int64 compare into a selection mask over `rows` elements.
    /// Processes in chunks of 1024 with explicit 8-wide unroll inside.
    #[inline(always)]
    pub fn filter_i64_gt(
        values: &[i64; MAX_ROWS],
        sel: &mut SelectionVector,
        rows: usize,
        lit: i64,
    ) {
        let n = rows.min(sel.len as usize).min(MAX_ROWS);
        let mut i = 0usize;
        // Chunk by BATCH_ROWS
        while i + BATCH_ROWS <= n {
            let base = i;
            let mut j = 0usize;
            while j < BATCH_ROWS {
                // 8-wide unroll
                let j0 = j;
                let j1 = j + 1;
                let j2 = j + 2;
                let j3 = j + 3;
                let j4 = j + 4;
                let j5 = j + 5;
                let j6 = j + 6;
                let j7 = j + 7;
                let m0 = sel.mask[base + j0];
                let m1 = sel.mask[base + j1];
                let m2 = sel.mask[base + j2];
                let m3 = sel.mask[base + j3];
                let m4 = sel.mask[base + j4];
                let m5 = sel.mask[base + j5];
                let m6 = sel.mask[base + j6];
                let m7 = sel.mask[base + j7];
                // Branchless predicate: (v > lit) as u8
                let p0 = (values[base + j0] > lit) as u8;
                let p1 = (values[base + j1] > lit) as u8;
                let p2 = (values[base + j2] > lit) as u8;
                let p3 = (values[base + j3] > lit) as u8;
                let p4 = (values[base + j4] > lit) as u8;
                let p5 = (values[base + j5] > lit) as u8;
                let p6 = (values[base + j6] > lit) as u8;
                let p7 = (values[base + j7] > lit) as u8;
                sel.mask[base + j0] = m0 & p0;
                sel.mask[base + j1] = m1 & p1;
                sel.mask[base + j2] = m2 & p2;
                sel.mask[base + j3] = m3 & p3;
                sel.mask[base + j4] = m4 & p4;
                sel.mask[base + j5] = m5 & p5;
                sel.mask[base + j6] = m6 & p6;
                sel.mask[base + j7] = m7 & p7;
                j += 8;
            }
            i += BATCH_ROWS;
        }
        while i < n {
            let p = (values[i] > lit) as u8;
            sel.mask[i] &= p;
            i += 1;
        }
    }

    #[inline(always)]
    pub fn filter_i64_lt(
        values: &[i64; MAX_ROWS],
        sel: &mut SelectionVector,
        rows: usize,
        lit: i64,
    ) {
        let n = rows.min(sel.len as usize).min(MAX_ROWS);
        let mut i = 0usize;
        while i < n {
            let p = (values[i] < lit) as u8;
            sel.mask[i] &= p;
            i += 1;
        }
    }

    #[inline(always)]
    pub fn filter_i64_eq(
        values: &[i64; MAX_ROWS],
        sel: &mut SelectionVector,
        rows: usize,
        lit: i64,
    ) {
        let n = rows.min(sel.len as usize).min(MAX_ROWS);
        let mut i = 0usize;
        while i < n {
            let p = (values[i] == lit) as u8;
            sel.mask[i] &= p;
            i += 1;
        }
    }

    /// Stable argsort of selected Int64 keys into `order` (row indices).
    /// Uses insertion sort — optimal for TAKE-bounded micro batches, zero alloc.
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
        let mut i = 0usize;
        while i < n {
            let take = sel.mask[i];
            // Branchless append when selected
            order[*order_len] = i as u16;
            *order_len += take as usize;
            i += 1;
        }
        // Insertion sort on the compacted index list
        let mut a = 1usize;
        while a < *order_len {
            let key_idx = order[a];
            let key_val = values[key_idx as usize];
            let mut b = a;
            while b > 0 {
                let prev = order[b - 1];
                let prev_val = values[prev as usize];
                if prev_val <= key_val {
                    break;
                }
                order[b] = prev;
                b -= 1;
            }
            order[b] = key_idx;
            a += 1;
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
        // Collect projected column names from ColumnList chain.
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

        // Reset output slabs (no heap — reuse inline buffers).
        let mut c0 = 0usize;
        while c0 < nproj {
            out.utf8_out[c0].clear();
            c0 += 1;
        }

        let mut out_row = 0usize;
        let mut oi = 0usize;
        while oi < order_len && out_row < MAX_ROWS {
            let src_row = order[oi] as usize;
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
                    active_rows = table.row_count as usize;
                    sel = SelectionVector::all(active_rows);
                    // Compact identity order
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
                    // Rebuild order from selection if not yet custom-sorted
                    if !sorted {
                        order_len = 0;
                        let mut i = 0usize;
                        while i < active_rows {
                            order[order_len] = i as u16;
                            order_len += sel.mask[i] as usize;
                            i += 1;
                        }
                    } else {
                        // Filter existing order list
                        let mut w = 0usize;
                        let mut r = 0usize;
                        while r < order_len {
                            let idx = order[r] as usize;
                            let keep = sel.mask[idx];
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
                NodeKind::Derive | NodeKind::Group | NodeKind::Aggregate | NodeKind::Join => {
                    // Supported in AST; demo pipeline does not exercise these stages.
                    // Keep as recognized no-op extension points without heap traffic.
                }
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
    let _ = cat.register(users);
    cat
}

/// End-to-end: parse + execute a Tamil pipeline query string.
pub fn run_query(src: &str, catalog: &Catalog, arena: &mut AstArena, out: &mut QueryResult) -> bool {
    arena.len = 0;
    arena.root = NIL;
    let root = match crate::parser::parse_query(src.as_bytes(), arena) {
        Some(r) => r,
        None => return false,
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
        let mut out = QueryResult::new();
        assert!(run_query(q, &cat, &mut arena, &mut out));
        assert_eq!(out.col_count, 2);
        assert_eq!(out.row_count, 10);
        // Ages must be sorted ascending and all > 21
        let mut prev = i64::MIN;
        let mut i = 0u16;
        while i < out.row_count {
            let age = out.int_out[1].values[i as usize];
            assert!(age > 21);
            assert!(age >= prev);
            prev = age;
            i += 1;
        }
        // First projected column is பெயர் (utf8), non-empty
        assert!(out.utf8_out[0].get_row(0).unwrap().len() > 0);
    }
}
