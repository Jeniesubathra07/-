//! CLI entry point for the Tamil-native vector query engine.

use std::env;
use std::process;
use tamil_query_engine::{demo_catalog, run_query, AstArena, PhysType, QueryResult, DEMO_QUERY};

fn main() {
    let mut args = env::args().skip(1);
    let query = match args.next() {
        Some(q) => q,
        None => DEMO_QUERY.to_string(),
    };

    let catalog = demo_catalog();
    let mut arena = Box::new(AstArena::new());
    let mut out = Box::new(QueryResult::new());

    if !run_query(&query, &catalog, &mut arena, &mut out) {
        eprintln!("error: failed to parse or execute query");
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
                    // Bool columns are not projected by the demo path; keep CLI complete.
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
