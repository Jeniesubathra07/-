//! `tqe_ingest` — CLI front-end for `tamil_query_engine::ingest`.
//!
//! Usage:
//!   tqe_ingest <csv_path> --schema "id:i64,name:utf8,age:i64" --out <dir> [--no-header]
//!
//! Exit codes: 0 on success, 1 on any ingest/schema/IO error (with a
//! specific, actionable message on stderr — never a bare panic).

use std::env;
use std::path::PathBuf;
use std::process;
use tamil_query_engine::ingest::{ingest_csv, parse_schema};

fn print_usage_and_exit(msg: Option<&str>) -> ! {
    if let Some(m) = msg {
        eprintln!("error: {m}");
    }
    eprintln!(
        "usage: tqe_ingest <csv_path> --schema \"col:type,col:type,...\" --out <dir> [--no-header]\n\n\
         types: i64 (also: int, int64), utf8 (also: text, string)\n\n\
         example:\n\
         \x20 tqe_ingest orders.csv --schema \"user_id:i64,price:i64\" --out ./mydb"
    );
    process::exit(1);
}

fn main() {
    let mut args = env::args().skip(1);
    let mut csv_path: Option<PathBuf> = None;
    let mut schema_spec: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut has_header = true;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--schema" => {
                schema_spec = Some(match args.next() {
                    Some(v) => v,
                    None => print_usage_and_exit(Some("--schema requires a value")),
                });
            }
            "--out" => {
                out_dir = Some(match args.next() {
                    Some(v) => PathBuf::from(v),
                    None => print_usage_and_exit(Some("--out requires a value")),
                });
            }
            "--no-header" => has_header = false,
            "-h" | "--help" => print_usage_and_exit(None),
            other if csv_path.is_none() && !other.starts_with("--") => {
                csv_path = Some(PathBuf::from(other));
            }
            other => print_usage_and_exit(Some(&format!("unrecognized argument '{other}'"))),
        }
    }

    let csv_path = csv_path.unwrap_or_else(|| print_usage_and_exit(Some("missing <csv_path>")));
    let schema_spec =
        schema_spec.unwrap_or_else(|| print_usage_and_exit(Some("missing --schema")));
    let out_dir = out_dir.unwrap_or_else(|| print_usage_and_exit(Some("missing --out")));

    let schema = match parse_schema(&schema_spec) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    match ingest_csv(&csv_path, &schema, &out_dir, has_header) {
        Ok(report) => {
            eprintln!(
                "ingested {} rows into {} column file(s) under {}",
                report.rows_ingested,
                report.columns_written.len(),
                out_dir.display()
            );
        }
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    }
}
