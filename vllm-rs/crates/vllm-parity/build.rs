use serde::Serialize;
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

#[derive(Debug, serde::Deserialize, Serialize)]
struct Row {
    section: String,
    feature: String,
    python: String,
    rust: String,
    rust_mlx: String,
    notes: String,
    deprecated: String,
}

#[derive(Serialize)]
struct SectionSummary {
    yes: u32,
    partial: u32,
    no: u32,
    total: u32,
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let csv_path = Path::new(&manifest_dir).join("parity.csv");
    println!("cargo:rerun-if-changed={}", csv_path.display());

    let mut rdr = csv::Reader::from_path(&csv_path).expect("failed to open parity.csv");
    let mut rows: Vec<Row> = Vec::new();
    for result in rdr.deserialize() {
        let row: Row = result.expect("failed to parse CSV row");
        rows.push(row);
    }

    // Compute per-section summaries (exclude deprecated and wontfix)
    let mut summaries: BTreeMap<String, SectionSummary> = BTreeMap::new();
    for row in &rows {
        let dep = row.deprecated.trim().to_ascii_lowercase();
        if dep == "yes" || dep == "wontfix" {
            continue;
        }
        let entry = summaries
            .entry(row.section.clone())
            .or_insert(SectionSummary {
                yes: 0,
                partial: 0,
                no: 0,
                total: 0,
            });
        entry.total += 1;
        match row.rust.trim() {
            "yes" => entry.yes += 1,
            "partial" => entry.partial += 1,
            _ => entry.no += 1,
        }
    }

    let out_dir = env::var("OUT_DIR").unwrap();

    let rows_json = serde_json::to_string(&rows).unwrap();
    fs::write(Path::new(&out_dir).join("parity_rows.json"), rows_json).unwrap();

    let summaries_json = serde_json::to_string(&summaries).unwrap();
    fs::write(
        Path::new(&out_dir).join("parity_summaries.json"),
        summaries_json,
    )
    .unwrap();
}
