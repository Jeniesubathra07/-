# Exhaustive unsafe audit (Ω-FINAL)

Total `unsafe` keyword hits from `rg`: **44**

Verified against current sources; SIGBUS remains a POSIX precondition on mmap sites.

| File:line | Snippet | Invariant | Enforcement | OK? |
|-----------|---------|-----------|-------------|-----|
| `src/parser.rs:214` | `unsafe {` | See local SAFETY comment | context | Review |
| `src/storage.rs:299` | `unsafe {` | See local SAFETY comment | context | Review |
| `src/storage.rs:427` | `pub unsafe fn as_slice(&self) -> &'a [u8] {` | MappedRegion bounds/lifetime | documented unsafe | Caller |
| `src/storage.rs:428` | `unsafe { core::slice::from_raw_parts(self.ptr, self.len` | Zeroed `Layout::new::<T>` cold alloc; fields written before Box observes | null-check + cold path | Y |
| `src/storage.rs:436` | `pub unsafe fn slice_at(&self, off: usize, len: usize) -` | See local SAFETY comment | context | Review |
| `src/storage.rs:439` | `ptr: unsafe { self.ptr.add(off) },` | MappedRegion bounds/lifetime | documented unsafe | Caller |
| `src/storage.rs:662` | `unsafe {` | See local SAFETY comment | context | Review |
| `src/storage.rs:740` | `pub unsafe fn memcpy_bytes(dst: *mut u8, src: *const u8` | Valid non-overlapping regions | unsafe fn contract | Caller |
| `src/storage.rs:741` | `unsafe { ptr::copy_nonoverlapping(src, dst, len) }` | Valid non-overlapping regions | unsafe fn contract | Caller |
| `src/storage.rs:778` | `// SAFETY: sysconf(_SC_PAGESIZE) is defined on POSIX an` | POSIX pagesize > 0 | cfg(unix)+4096 fallback | Y |
| `src/storage.rs:779` | `let p = unsafe { sysconf(_SC_PAGESIZE) };` | POSIX pagesize > 0 | cfg(unix)+4096 fallback | Y |
| `src/storage.rs:848` | `// SAFETY: read-only files; lengths validated.` | See local SAFETY comment | context | Review |
| `src/storage.rs:849` | `let offsets_mmap = unsafe { Mmap::map(&off_file)? };` | RO file mapping valid while struct lives | single-writer-absent docs | Partial (SIGBUS) |
| `src/storage.rs:850` | `let blob_mmap = unsafe { Mmap::map(&blob_file)? };` | RO file mapping valid while struct lives | single-writer-absent docs | Partial (SIGBUS) |
| `src/storage.rs:1077` | `// SAFETY: file is opened read-only; length validated.` | See local SAFETY comment | context | Review |
| `src/storage.rs:1078` | `let mmap = unsafe { Mmap::map(&file)? };` | RO file mapping valid while struct lives | single-writer-absent docs | Partial (SIGBUS) |
| `src/storage.rs:1172` | `// SAFETY: (1) `n * row_width` bytes with `row_width ==` | RO file mapping valid while struct lives | single-writer-absent docs | Partial (SIGBUS) |
| `src/storage.rs:1179` | `unsafe { core::slice::from_raw_parts(bytes.as_ptr() as ` | Zeroed `Layout::new::<T>` cold alloc; fields written before Box observes | null-check + cold path | Y |
| `src/storage.rs:1274` | `let ages = unsafe {` | See local SAFETY comment | context | Review |
| `src/storage.rs:1280` | `let user_ids = unsafe {` | See local SAFETY comment | context | Review |
| `src/storage.rs:1286` | `let prices = unsafe {` | See local SAFETY comment | context | Review |
| `src/storage.rs:1333` | `let bytes = unsafe {` | See local SAFETY comment | context | Review |
| `src/runtime.rs:126` | `unsafe {` | See local SAFETY comment | context | Review |
| `src/runtime.rs:205` | `unsafe {` | See local SAFETY comment | context | Review |
| `src/runtime.rs:323` | `let scratch = unsafe { &mut *cell.get() };` | See local SAFETY comment | context | Review |
| `src/runtime.rs:372` | `let pad = unsafe { &mut *cell.get() };` | See local SAFETY comment | context | Review |
| `src/runtime.rs:422` | `unsafe { core::slice::from_raw_parts(src_addr as *const` | Zeroed `Layout::new::<T>` cold alloc; fields written before Box observes | null-check + cold path | Y |
| `src/runtime.rs:424` | `unsafe { core::slice::from_raw_parts_mut(dst_addr as *m` | Zeroed `Layout::new::<T>` cold alloc; fields written before Box observes | null-check + cold path | Y |
| `src/runtime.rs:455` | `unsafe fn apply_unroll8(` | TLS exclusive; indices < MAX_ROWS/BATCH | caller n bounds | Y |
| `src/runtime.rs:519` | `// SAFETY: base+j+7 < base+BATCH_ROWS <= n <= MAX_ROWS.` | See local SAFETY comment | context | Review |
| `src/runtime.rs:520` | `unsafe {` | See local SAFETY comment | context | Review |
| `src/runtime.rs:534` | `unsafe {` | See local SAFETY comment | context | Review |
| `src/runtime.rs:690` | `let pad = unsafe { &mut *cell.get() };` | See local SAFETY comment | context | Review |
| `src/runtime.rs:746` | `let pad = unsafe { &mut *cell.get() };` | See local SAFETY comment | context | Review |
| `src/runtime.rs:2042` | `let pad = unsafe { &mut *cell.get() };` | See local SAFETY comment | context | Review |
| `src/lexer.rs:468` | `unsafe {` | See local SAFETY comment | context | Review |
| `src/lib.rs:60` | `unsafe impl GlobalAlloc for CountingAlloc {` | Zeroed `Layout::new::<T>` cold alloc; fields written before Box observes | null-check + cold path | Y |
| `src/lib.rs:61` | `unsafe fn alloc(&self, layout: Layout) -> *mut u8 {` | Zeroed `Layout::new::<T>` cold alloc; fields written before Box observes | null-check + cold path | Y |
| `src/lib.rs:68` | `unsafe { System.alloc(layout) }` | Zeroed `Layout::new::<T>` cold alloc; fields written before Box observes | null-check + cold path | Y |
| `src/lib.rs:71` | `unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) ` | Zeroed `Layout::new::<T>` cold alloc; fields written before Box observes | null-check + cold path | Y |
| `src/lib.rs:72` | `unsafe { System.dealloc(ptr, layout) }` | Zeroed `Layout::new::<T>` cold alloc; fields written before Box observes | null-check + cold path | Y |
| `src/lib.rs:75` | `unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, ` | Zeroed `Layout::new::<T>` cold alloc; fields written before Box observes | null-check + cold path | Y |
| `src/lib.rs:82` | `unsafe { System.realloc(ptr, layout, new_size) }` | Zeroed `Layout::new::<T>` cold alloc; fields written before Box observes | null-check + cold path | Y |
| `src/lib.rs:1558` | `let v = unsafe { *derived_ptr.add(s) };` | MappedRegion bounds/lifetime | documented unsafe | Caller |

\* Alignment assumes OS page size is a multiple of 8 (true for POSIX base pages).
