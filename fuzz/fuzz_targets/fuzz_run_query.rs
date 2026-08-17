#![no_main]
use libfuzzer_sys::fuzz_target;
use tamil_query_engine::{
    alloc_token_window, demo_catalog, run_query, AstArena, QueryResult, RuntimeScratch,
};

fuzz_target!(|data: &[u8]| {
    // Cap input — lexer/parser already bound tokens; keep fuzzer throughput high.
    if data.len() > 4096 {
        return;
    }
    let Ok(src) = std::str::from_utf8(data) else {
        return;
    };
    let catalog = demo_catalog();
    let mut arena = Box::new(AstArena::new());
    let mut out = QueryResult::new_boxed();
    let mut scratch = RuntimeScratch::new_boxed();
    let mut tokens = alloc_token_window();
    let _ = run_query(src, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens);
});
