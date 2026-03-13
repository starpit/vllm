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

/// Load a CSV file and return its contents as a JSON array string.
/// Each row becomes a JSON object with header keys. Numeric values are
/// parsed as f64; everything else stays a string.
fn load_csv_as_json(path: &Path) -> String {
    if !path.exists() {
        return "[]".to_string();
    }
    let mut rdr = csv::Reader::from_path(path).expect("failed to open bench CSV");
    let headers: Vec<String> = rdr
        .headers()
        .unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut rows = Vec::new();
    for result in rdr.records() {
        let record = result.expect("failed to parse bench CSV row");
        let mut map = serde_json::Map::new();
        for (i, h) in headers.iter().enumerate() {
            let val = record.get(i).unwrap_or("");
            if let Ok(n) = val.parse::<f64>() {
                map.insert(h.clone(), serde_json::json!(n));
            } else {
                map.insert(h.clone(), serde_json::json!(val));
            }
        }
        rows.push(serde_json::Value::Object(map));
    }
    serde_json::to_string(&rows).unwrap()
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let csv_path = Path::new(&manifest_dir).join("parity.csv");
    let bench_dir = Path::new(&manifest_dir).join("bench_data");
    println!("cargo:rerun-if-changed={}", csv_path.display());
    println!("cargo:rerun-if-changed={}", bench_dir.display());

    // --- Parity CSV ---
    let mut rdr = csv::Reader::from_path(&csv_path).expect("failed to open parity.csv");
    let mut rows: Vec<Row> = Vec::new();
    for result in rdr.deserialize() {
        let row: Row = result.expect("failed to parse CSV row");
        rows.push(row);
    }

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

    // --- Bench data CSVs (per-model directory structure) ---
    // bench_data/cuda/L40s/{latency,throughput,serve}/ModelName.csv
    // Each CSV has an `engine` column ("python" or "rust"); rows from both are combined.
    let l40s_dir = bench_dir.join("cuda").join("L40s");
    println!("cargo:rerun-if-changed={}", l40s_dir.display());

    let mut bench_obj = serde_json::Map::new();
    for (key, subpath) in &[
        ("latency", "latency"),
        ("throughput", "throughput"),
        ("serve", "serve"),
        ("startup_warm", "startup/warm"),
    ] {
        let type_dir = l40s_dir.join(subpath);
        let mut type_obj = serde_json::Map::new();
        if let Ok(rd) = fs::read_dir(&type_dir) {
            let mut paths: Vec<_> = rd
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "csv"))
                .collect();
            paths.sort();
            for path in paths {
                println!("cargo:rerun-if-changed={}", path.display());
                let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                let json_str = load_csv_as_json(&path);
                let json_val: serde_json::Value = serde_json::from_str(&json_str).unwrap();
                type_obj.insert(stem, json_val);
            }
        }
        bench_obj.insert(key.to_string(), serde_json::Value::Object(type_obj));
    }
    let bench_json = serde_json::to_string(&serde_json::Value::Object(bench_obj)).unwrap();
    fs::write(Path::new(&out_dir).join("bench_data.json"), bench_json).unwrap();
}
