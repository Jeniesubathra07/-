//! Phase 1 of the public-release roadmap: a real CSV → columnar ingest
//! pipeline.
//!
//! Everything else in this crate reads *pre-made* `.bin`/`.meta`/
//! `.offsets`/`.blob` files (or the two hardcoded 12–16 row demo tables).
//! There was previously no path from "I have a CSV of real data" to
//! "it's queryable" — this module is that path.
//!
//! Design decisions, stated explicitly:
//!
//! - VALIDATION HAPPENS AT INGEST TIME, NOT QUERY TIME. A malformed row,
//!   a non-numeric value in an i64 column, or a numeric literal that
//!   overflows `i64` is rejected here, with the exact line number and
//!   column, before any bytes are written to disk. This mirrors the same
//!   discipline already applied to `Lexer::scan_number` (checked
//!   arithmetic, explicit rejection, no silent wraparound) — ingest is
//!   the other place untrusted external data enters this engine, and it
//!   gets the same treatment.
//!
//! - NO NEW DEPENDENCY. `Cargo.toml` deliberately states `memmap2` as the
//!   "sole dependency". Rather than pull in a CSV-parsing crate, this
//!   module hand-rolls a minimal RFC4180-subset parser: comma-separated
//!   fields, optional double-quote-delimited fields with `""` as an
//!   escaped quote. This covers the overwhelming majority of real-world
//!   CSV exports (Excel, Sheets, Postgres `COPY ... CSV`) without pulling
//!   in a dependency for a task this small.
//!
//! - INGEST IS A COLD PATH. Unlike `runtime.rs`'s query execution loops,
//!   this module freely uses `String`/`Vec` — the zero-heap invariant in
//!   this crate has only ever applied to hot query execution, never to
//!   one-time ingest, and the existing `write_i64_column_bin` /
//!   `write_utf8_column_files` cold-path writers already establish this
//!   precedent.
//!
//! - ATOMIC PUBLISH, PER COLUMN. Each column file is written through the
//!   existing tmp-then-rename discipline in `write_i64_column_bin`, so a
//!   process killed mid-ingest never leaves a column file that looks
//!   valid but contains truncated/partial data. Cross-column atomicity
//!   (all-or-nothing across every column in one ingest run) is NOT
//!   provided — callers that need all-or-nothing across columns should
//!   ingest into a temp directory and rename it on success. See the
//!   documentation on `ingest_csv` for exactly what is and isn't
//!   guaranteed, rather than implying a stronger guarantee than exists.

use crate::storage::{write_i64_column_bin, write_utf8_column_files};
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// The two physical column types this engine supports end to end today.
/// (Matches `PhysType::{Int64, Utf8}` in `runtime.rs` — kept as an
/// independent enum here rather than reusing `PhysType` directly, since
/// ingest is a schema-description concern and shouldn't need to import
/// execution-layer types.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Int64,
    Utf8,
}

impl fmt::Display for ColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColumnType::Int64 => write!(f, "i64"),
            ColumnType::Utf8 => write!(f, "utf8"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ColumnSpec {
    pub name: String,
    pub ty: ColumnType,
}

/// A validated, ordered column schema for one ingest run.
#[derive(Debug, Clone)]
pub struct IngestSchema {
    pub columns: Vec<ColumnSpec>,
}

/// Every failure mode is named and carries enough context (line number,
/// column name, offending value) to actually locate and fix the source
/// data — a bare `io::Error` or `"parse failed"` is not acceptable for a
/// pipeline whose entire job is validating untrusted external input.
#[derive(Debug)]
pub enum IngestError {
    Io(io::Error),
    /// The `--schema` spec string itself was malformed.
    BadSchemaSpec { reason: String },
    /// The schema declared zero columns.
    EmptySchema,
    /// Two columns in the schema share the same name.
    DuplicateColumnName(String),
    /// The CSV's header row (if present) doesn't match the declared
    /// schema names, in order.
    HeaderMismatch {
        expected: Vec<String>,
        found: Vec<String>,
    },
    /// A data row had a different number of fields than the schema.
    ColumnCountMismatch {
        line: usize,
        expected: usize,
        found: usize,
    },
    /// An i64-typed field could not be parsed as an integer, or exceeded
    /// `i64` range. Mirrors `ParserError::NumberLiteralOverflow`'s
    /// "reject, never silently wrap" policy at the ingest boundary.
    NumericParse {
        line: usize,
        column: String,
        value: String,
    },
    /// A raw CSV field was not valid UTF-8.
    InvalidUtf8 { line: usize, column: String },
    /// A quoted CSV field was never closed before end of file.
    UnterminatedQuote { line: usize },
}

impl fmt::Display for IngestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IngestError::Io(e) => write!(f, "I/O error: {e}"),
            IngestError::BadSchemaSpec { reason } => {
                write!(f, "invalid --schema spec: {reason}")
            }
            IngestError::EmptySchema => write!(f, "schema declares zero columns"),
            IngestError::DuplicateColumnName(name) => {
                write!(f, "duplicate column name in schema: '{name}'")
            }
            IngestError::HeaderMismatch { expected, found } => write!(
                f,
                "CSV header does not match schema: expected {expected:?}, found {found:?}"
            ),
            IngestError::ColumnCountMismatch {
                line,
                expected,
                found,
            } => write!(
                f,
                "line {line}: expected {expected} columns, found {found}"
            ),
            IngestError::NumericParse {
                line,
                column,
                value,
            } => write!(
                f,
                "line {line}, column '{column}': '{value}' is not a valid i64 (non-numeric or out of range)"
            ),
            IngestError::InvalidUtf8 { line, column } => {
                write!(f, "line {line}, column '{column}': field is not valid UTF-8")
            }
            IngestError::UnterminatedQuote { line } => {
                write!(f, "line {line}: unterminated quoted field")
            }
        }
    }
}

impl std::error::Error for IngestError {}

impl From<io::Error> for IngestError {
    fn from(e: io::Error) -> Self {
        IngestError::Io(e)
    }
}

/// Parse a schema spec of the form `"name:type,name:type,..."`, e.g.
/// `"id:i64,name:utf8,age:i64"`. Rejects empty schemas and duplicate
/// column names up front, before any CSV row is read.
pub fn parse_schema(spec: &str) -> Result<IngestSchema, IngestError> {
    let mut columns = Vec::new();
    for (idx, field) in spec.split(',').enumerate() {
        let field = field.trim();
        if field.is_empty() {
            if idx == 0 && spec.trim().is_empty() {
                return Err(IngestError::EmptySchema);
            }
            return Err(IngestError::BadSchemaSpec {
                reason: format!("empty column spec at position {idx}"),
            });
        }
        let mut parts = field.splitn(2, ':');
        let name = parts
            .next()
            .ok_or_else(|| IngestError::BadSchemaSpec {
                reason: format!("missing name in '{field}'"),
            })?
            .trim()
            .to_string();
        let ty_str = parts
            .next()
            .ok_or_else(|| IngestError::BadSchemaSpec {
                reason: format!("missing ':type' in '{field}' (expected e.g. 'name:i64')"),
            })?
            .trim();
        let ty = match ty_str {
            "i64" | "int64" | "int" => ColumnType::Int64,
            "utf8" | "text" | "string" => ColumnType::Utf8,
            other => {
                return Err(IngestError::BadSchemaSpec {
                    reason: format!(
                        "unknown type '{other}' for column '{name}' (expected 'i64' or 'utf8')"
                    ),
                })
            }
        };
        if name.is_empty() {
            return Err(IngestError::BadSchemaSpec {
                reason: format!("empty column name in '{field}'"),
            });
        }
        columns.push(ColumnSpec { name, ty });
    }
    if columns.is_empty() {
        return Err(IngestError::EmptySchema);
    }
    for i in 0..columns.len() {
        for j in (i + 1)..columns.len() {
            if columns[i].name == columns[j].name {
                return Err(IngestError::DuplicateColumnName(columns[i].name.clone()));
            }
        }
    }
    Ok(IngestSchema { columns })
}

/// Minimal RFC4180-subset line splitter: comma-separated fields, with
/// optional double-quote-delimited fields supporting `""` as an escaped
/// literal quote. Does NOT handle embedded newlines inside a quoted
/// field (each CSV row must be exactly one text line) — real-world
/// exports from Excel/Sheets/`COPY ... CSV` overwhelmingly satisfy this;
/// multi-line quoted fields are out of scope rather than silently
/// mishandled (an unterminated quote is rejected explicitly, not
/// guessed at).
fn split_csv_line(line: &str, line_no: usize) -> Result<Vec<String>, IngestError> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut chars = line.chars().peekable();
    let mut in_quotes = false;
    let mut field_was_quoted = false;

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push(c);
            }
        } else if c == '"' && cur.is_empty() && !field_was_quoted {
            in_quotes = true;
            field_was_quoted = true;
        } else if c == ',' {
            fields.push(std::mem::take(&mut cur));
            field_was_quoted = false;
        } else {
            cur.push(c);
        }
    }
    if in_quotes {
        return Err(IngestError::UnterminatedQuote { line: line_no });
    }
    fields.push(cur);
    Ok(fields)
}

/// Result of a successful ingest run.
#[derive(Debug)]
pub struct IngestReport {
    pub rows_ingested: usize,
    pub columns_written: Vec<PathBuf>,
}

/// Ingest a CSV file into page-aligned columnar files under `out_dir`,
/// one `.bin`/`.meta` (Int64) or `.offsets`/`.blob`/`.meta` (Utf8) triple
/// per schema column, named after the column.
///
/// `has_header`: if `true`, the first line is checked against
/// `schema.columns`' names (in order) and consumed rather than ingested
/// as data — a mismatch is rejected with [`IngestError::HeaderMismatch`]
/// rather than being silently ingested as a bogus first row.
///
/// ATOMICITY: each column file is published via the existing
/// tmp-then-rename discipline in [`write_i64_column_bin`] /
/// [`write_utf8_column_files`], so a crash mid-write of ONE column never
/// leaves that column's on-disk file looking valid while actually
/// truncated. There is NO cross-column atomicity: if ingest fails while
/// writing the third of five columns, the first two columns' files are
/// already fully and validly published to `out_dir`. Callers that need
/// all-or-nothing semantics across an entire multi-column ingest should
/// ingest into a fresh temporary directory and atomically rename that
/// whole directory into place only on full success — this function
/// deliberately does not impose that policy itself, since some callers
/// legitimately want incremental per-column publication.
pub fn ingest_csv(
    csv_path: &Path,
    schema: &IngestSchema,
    out_dir: &Path,
    has_header: bool,
) -> Result<IngestReport, IngestError> {
    let raw = fs::read(csv_path)?;
    let text = String::from_utf8(raw).map_err(|_| IngestError::InvalidUtf8 {
        line: 0,
        column: "<file>".to_string(),
    })?;

    fs::create_dir_all(out_dir)?;

    let n_cols = schema.columns.len();
    let mut int_cols: Vec<Vec<i64>> = schema
        .columns
        .iter()
        .map(|c| {
            if c.ty == ColumnType::Int64 {
                Vec::new()
            } else {
                Vec::new()
            }
        })
        .collect();
    let mut utf8_cols: Vec<Vec<Vec<u8>>> = (0..n_cols).map(|_| Vec::new()).collect();

    let mut lines = text.lines().enumerate();
    if has_header {
        if let Some((_, header_line)) = lines.next() {
            let header_fields = split_csv_line(header_line, 1)?;
            let expected: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
            if header_fields != expected {
                return Err(IngestError::HeaderMismatch {
                    expected,
                    found: header_fields,
                });
            }
        }
    }

    let mut rows_ingested = 0usize;
    for (idx, line) in lines {
        let line_no = idx + 1; // 1-based, human-facing
        if line.is_empty() {
            continue; // tolerate trailing/blank lines, do not miscount them as data
        }
        let fields = split_csv_line(line, line_no)?;
        if fields.len() != n_cols {
            return Err(IngestError::ColumnCountMismatch {
                line: line_no,
                expected: n_cols,
                found: fields.len(),
            });
        }
        for (c, field) in fields.into_iter().enumerate() {
            match schema.columns[c].ty {
                ColumnType::Int64 => {
                    let trimmed = field.trim();
                    let v: i64 = trimmed.parse().map_err(|_| IngestError::NumericParse {
                        line: line_no,
                        column: schema.columns[c].name.clone(),
                        value: field.clone(),
                    })?;
                    int_cols[c].push(v);
                }
                ColumnType::Utf8 => {
                    utf8_cols[c].push(field.into_bytes());
                }
            }
        }
        rows_ingested += 1;
    }

    let mut columns_written = Vec::with_capacity(n_cols);
    for (c, spec) in schema.columns.iter().enumerate() {
        match spec.ty {
            ColumnType::Int64 => {
                let bin_path = out_dir.join(format!("{}.bin", spec.name));
                let data = &int_cols[c];
                write_i64_column_bin(&bin_path, data.len(), |i| data[i])?;
                columns_written.push(bin_path);
            }
            ColumnType::Utf8 => {
                let offsets_path = out_dir.join(format!("{}.offsets", spec.name));
                let blob_path = out_dir.join(format!("{}.blob", spec.name));
                let meta_path = out_dir.join(format!("{}.meta", spec.name));
                let refs: Vec<&[u8]> = utf8_cols[c].iter().map(|v| v.as_slice()).collect();
                write_utf8_column_files(&offsets_path, &blob_path, &meta_path, &refs)?;
                columns_written.push(offsets_path);
            }
        }
    }

    Ok(IngestReport {
        rows_ingested,
        columns_written,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{ColumnarFileStream, Utf8ColumnFile};

    fn tmp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("tqe_ingest_{pid}_{nanos}_{name}"));
        p
    }

    fn write_csv(path: &Path, content: &str) {
        fs::write(path, content).unwrap();
    }

    #[test]
    fn parse_schema_basic() {
        let s = parse_schema("id:i64,name:utf8,age:int64").unwrap();
        assert_eq!(s.columns.len(), 3);
        assert_eq!(s.columns[0].name, "id");
        assert_eq!(s.columns[0].ty, ColumnType::Int64);
        assert_eq!(s.columns[1].ty, ColumnType::Utf8);
        assert_eq!(s.columns[2].ty, ColumnType::Int64);
    }

    #[test]
    fn parse_schema_rejects_empty_and_duplicates() {
        assert!(matches!(parse_schema(""), Err(IngestError::EmptySchema)));
        assert!(matches!(
            parse_schema("id:i64,id:utf8"),
            Err(IngestError::DuplicateColumnName(n)) if n == "id"
        ));
        assert!(matches!(
            parse_schema("id:banana"),
            Err(IngestError::BadSchemaSpec { .. })
        ));
    }

    #[test]
    #[cfg_attr(miri, ignore = "file I/O / mmap unsupported under Miri isolation")]
    fn ingest_round_trip_small_csv() {
        let dir = tmp_dir("roundtrip");
        let csv_path = dir.join("in.csv");
        fs::create_dir_all(&dir).unwrap();
        write_csv(
            &csv_path,
            "id,name,age\n1,அருண்,18\n2,பிரியா,22\n3,கண்ணன்,19\n",
        );
        let schema = parse_schema("id:i64,name:utf8,age:i64").unwrap();
        let out_dir = dir.join("out");
        let report = ingest_csv(&csv_path, &schema, &out_dir, true).unwrap();
        assert_eq!(report.rows_ingested, 3);
        assert_eq!(report.columns_written.len(), 3);

        let mut ids = ColumnarFileStream::open_int64_column(
            &out_dir.join("id.bin"),
            &out_dir.join("id.meta"),
        )
        .unwrap();
        assert_eq!(ids.stream.total_rows(), 3);
        let chunk = ids.stream.next_page_chunk().unwrap();
        assert_eq!(chunk.rows, &[1, 2, 3]);

        let names = Utf8ColumnFile::open(
            &out_dir.join("name.offsets"),
            &out_dir.join("name.blob"),
            Some(&out_dir.join("name.meta")),
        )
        .unwrap();
        assert_eq!(names.get_row(0).unwrap(), "அருண்");
        assert_eq!(names.get_row(1).unwrap(), "பிரியா");
        assert_eq!(names.get_row(2).unwrap(), "கண்ணன்");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg_attr(miri, ignore = "file I/O")]
    fn ingest_rejects_column_count_mismatch_with_line_number() {
        let dir = tmp_dir("badcount");
        let csv_path = dir.join("in.csv");
        fs::create_dir_all(&dir).unwrap();
        write_csv(&csv_path, "id,age\n1,18\n2\n"); // line 3 (data line 2) missing a field
        let schema = parse_schema("id:i64,age:i64").unwrap();
        let err = ingest_csv(&csv_path, &schema, &dir.join("out"), true).unwrap_err();
        match err {
            IngestError::ColumnCountMismatch {
                line,
                expected,
                found,
            } => {
                assert_eq!(line, 3);
                assert_eq!(expected, 2);
                assert_eq!(found, 1);
            }
            other => panic!("expected ColumnCountMismatch, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg_attr(miri, ignore = "file I/O")]
    fn ingest_rejects_non_numeric_i64_field() {
        let dir = tmp_dir("badnum");
        let csv_path = dir.join("in.csv");
        fs::create_dir_all(&dir).unwrap();
        write_csv(&csv_path, "id,age\n1,not_a_number\n");
        let schema = parse_schema("id:i64,age:i64").unwrap();
        let err = ingest_csv(&csv_path, &schema, &dir.join("out"), true).unwrap_err();
        match err {
            IngestError::NumericParse {
                line,
                column,
                value,
            } => {
                assert_eq!(line, 2);
                assert_eq!(column, "age");
                assert_eq!(value, "not_a_number");
            }
            other => panic!("expected NumericParse, got {other:?}"),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg_attr(miri, ignore = "file I/O")]
    fn ingest_rejects_i64_overflow_field() {
        // Same "reject, never silently wrap" discipline as
        // Lexer::scan_number / Parser::expect_number, applied at the
        // ingest boundary: str::parse::<i64> already uses checked
        // arithmetic internally and returns Err on overflow rather than
        // wrapping, so this is naturally consistent with that fix.
        let dir = tmp_dir("overflow");
        let csv_path = dir.join("in.csv");
        fs::create_dir_all(&dir).unwrap();
        write_csv(&csv_path, "id,age\n1,99999999999999999999999\n");
        let schema = parse_schema("id:i64,age:i64").unwrap();
        let err = ingest_csv(&csv_path, &schema, &dir.join("out"), true).unwrap_err();
        assert!(matches!(err, IngestError::NumericParse { line: 2, .. }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg_attr(miri, ignore = "file I/O")]
    fn ingest_rejects_header_mismatch() {
        let dir = tmp_dir("badheader");
        let csv_path = dir.join("in.csv");
        fs::create_dir_all(&dir).unwrap();
        write_csv(&csv_path, "id,wrong_name\n1,2\n");
        let schema = parse_schema("id:i64,age:i64").unwrap();
        let err = ingest_csv(&csv_path, &schema, &dir.join("out"), true).unwrap_err();
        assert!(matches!(err, IngestError::HeaderMismatch { .. }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg_attr(miri, ignore = "file I/O")]
    fn ingest_handles_quoted_fields_with_embedded_comma_and_escaped_quote() {
        let dir = tmp_dir("quoted");
        let csv_path = dir.join("in.csv");
        fs::create_dir_all(&dir).unwrap();
        write_csv(
            &csv_path,
            "id,note\n1,\"hello, world\"\n2,\"she said \"\"hi\"\"\"\n",
        );
        let schema = parse_schema("id:i64,note:utf8").unwrap();
        let out_dir = dir.join("out");
        ingest_csv(&csv_path, &schema, &out_dir, true).unwrap();
        let notes = Utf8ColumnFile::open(
            &out_dir.join("note.offsets"),
            &out_dir.join("note.blob"),
            Some(&out_dir.join("note.meta")),
        )
        .unwrap();
        assert_eq!(notes.get_row(0).unwrap(), "hello, world");
        assert_eq!(notes.get_row(1).unwrap(), "she said \"hi\"");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg_attr(miri, ignore = "file I/O")]
    fn ingest_rejects_unterminated_quote() {
        let dir = tmp_dir("unterminated");
        let csv_path = dir.join("in.csv");
        fs::create_dir_all(&dir).unwrap();
        write_csv(&csv_path, "id,note\n1,\"never closed\n");
        let schema = parse_schema("id:i64,note:utf8").unwrap();
        let err = ingest_csv(&csv_path, &schema, &dir.join("out"), true).unwrap_err();
        assert!(matches!(err, IngestError::UnterminatedQuote { line: 2 }));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg_attr(miri, ignore = "file I/O / mmap")]
    fn ingest_large_synthetic_dataset_end_to_end_with_hand_computed_sum() {
        // The real end-to-end proof this module exists for: not the
        // 12-16 row demo tables, and not a hand-typed fixture, but a
        // generated dataset large enough (12,000 rows) to span multiple
        // mmap pages and exceed MAX_ROWS, run through the real ingest
        // path, then re-opened and scanned via the real
        // ColumnarFileStream chunked API, with the result checked
        // against an independently hand-computed expected sum.
        let dir = tmp_dir("large");
        fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("orders.csv");

        let n = 12_000i64;
        let mut csv = String::from("user_id,price\n");
        let mut expected_sum: i64 = 0;
        for i in 0..n {
            let user_id = i % 500;
            let price = 100 + (i % 900);
            expected_sum += price;
            csv.push_str(&format!("{user_id},{price}\n"));
        }
        write_csv(&csv_path, &csv);

        let schema = parse_schema("user_id:i64,price:i64").unwrap();
        let out_dir = dir.join("out");
        let report = ingest_csv(&csv_path, &schema, &out_dir, true).unwrap();
        assert_eq!(report.rows_ingested, n as usize);

        let mut prices = ColumnarFileStream::open_int64_column(
            &out_dir.join("price.bin"),
            &out_dir.join("price.meta"),
        )
        .unwrap();
        assert_eq!(prices.stream.total_rows(), n as u64);

        let mut actual_sum: i64 = 0;
        let mut rows_seen = 0usize;
        while let Some(chunk) = prices.stream.next_page_chunk() {
            for &v in chunk.rows {
                actual_sum += v;
            }
            rows_seen += chunk.rows.len();
        }
        assert_eq!(
            rows_seen, n as usize,
            "every row must be streamed exactly once"
        );
        assert_eq!(
            actual_sum, expected_sum,
            "ingested data must survive round-trip exactly"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
