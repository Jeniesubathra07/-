//! CLI entry point for the Tamil-native vector query engine.

use std::env;
use std::process;
use tamil_query_engine::{
    alloc_token_window, demo_catalog, run_query, AstArena, EngineError, PhysType, QueryResult,
    RuntimeScratch, DEMO_QUERY,
};

fn main() {
    let mut args = env::args().skip(1);
    let query = match args.next() {
        Some(q) => q,
        None => DEMO_QUERY.to_string(),
    };

    let catalog = demo_catalog();
    let mut arena = Box::new(AstArena::new());
    let mut out = QueryResult::new_boxed();
    let mut scratch = RuntimeScratch::new_boxed();
    let mut tokens = alloc_token_window();

    if let Err(e) = run_query(&query, &catalog, &mut arena, &mut out, &mut scratch, &mut tokens) {
        match e {
            EngineError::ParseFailed => eprintln!("error: ParseFailed"),
            EngineError::ColumnNotFound { table, column } => {
                let t = core::str::from_utf8(trim_z(&table)).unwrap_or("?");
                let c = core::str::from_utf8(trim_z(&column)).unwrap_or("?");
                eprintln!("error: ColumnNotFound {{ table: {t}, column: {c} }}");
            }
            EngineError::IoError => eprintln!("error: IoError"),
            EngineError::PageCorrupt { page_index } => {
                eprintln!("error: PageCorrupt {{ page_index: {page_index} }}");
            }
            EngineError::LiteralOverflow => eprintln!("error: LiteralOverflow"),
            EngineError::NotImplemented { stage } => {
                eprintln!("error: NotImplemented {{ stage: {stage} }}");
            }
        }
        eprintln!("query: {query}");
        process::exit(1);
    }

    // Header
    let mut c = 0usize;
    while c < out.col_count as usize {
        if c > 0 {
            print!("\t");
        }
        let name = core::str::from_utf8(out.schema[c].name.as_bytes()).unwrap_or("?");
        print!("{name}");
        c += 1;
    }
    println!();

    // Rows
    let mut r = 0usize;
    while r < out.row_count as usize {
        let mut c = 0usize;
        while c < out.col_count as usize {
            if c > 0 {
                print!("\t");
            }
            match out.types[c] {
                PhysType::Int64 => print!("{}", out.int_out[c].values[r]),
                PhysType::Utf8 => {
                    let s = out.utf8_out[c].get_row(r).unwrap_or("");
                    print!("{s}");
                }
                PhysType::Bool => {
                    print!("?");
                }
                PhysType::Null => print!("NULL"),
            }
            c += 1;
        }
        println!();
        r += 1;
    }
}

fn trim_z(buf: &[u8]) -> &[u8] {
    let mut n = 0usize;
    while n < buf.len() && buf[n] != 0 {
        n += 1;
    }
    &buf[..n]
}
