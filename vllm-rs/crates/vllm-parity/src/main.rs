use std::env;
use std::fs;

const TEMPLATE: &str = include_str!("../templates/index.html");
const PARITY_ROWS: &str = include_str!(concat!(env!("OUT_DIR"), "/parity_rows.json"));
const PARITY_SUMMARIES: &str = include_str!(concat!(env!("OUT_DIR"), "/parity_summaries.json"));
const BENCH_DATA: &str = include_str!(concat!(env!("OUT_DIR"), "/bench_data.json"));

fn main() {
    let html = TEMPLATE
        .replace("/*__PARITY_ROWS__*/", PARITY_ROWS)
        .replace("/*__PARITY_SUMMARIES__*/", PARITY_SUMMARIES)
        .replace("/*__BENCH_DATA__*/", BENCH_DATA);

    let args: Vec<String> = env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "-o" || a == "--output")
        && let Some(path) = args.get(pos + 1)
    {
        if let Some(parent) = std::path::Path::new(path).parent() {
            fs::create_dir_all(parent).ok();
        }
        fs::write(path, &html).expect("failed to write output file");
        eprintln!("Wrote {}", path);
        return;
    }

    print!("{html}");
}
