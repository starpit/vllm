// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! ORCA (Open Request Cost Aggregation) load-reporting headers.
//!
//! Smart HTTP load balancers (e.g., Envoy) use ORCA to route traffic away
//! from overloaded backends. When a client sends an
//! `endpoint-load-metrics-format` request header, the server attaches an
//! `endpoint-load-metrics` response header with current load stats.

use crate::metrics::VllmMetrics;

/// The request header that clients send to opt-in to ORCA reporting.
pub const ORCA_REQUEST_HEADER: &str = "endpoint-load-metrics-format";

/// The response header attached when the client opts in.
pub const ORCA_RESPONSE_HEADER: &str = "endpoint-load-metrics";

/// Build the ORCA response header value for the given format.
///
/// Returns `Some((header_name, header_value))` for "TEXT" or "JSON" formats,
/// or `None` for unsupported formats.
pub fn orca_header(format: &str) -> Option<(String, String)> {
    let m = VllmMetrics::global();
    let kv = m.kv_cache_usage_perc.get();
    let waiting = m.num_requests_waiting.get();

    let value = match format.to_ascii_uppercase().as_str() {
        "TEXT" => format!(
            "named_metrics.kv_cache_usage_perc={kv},named_metrics.num_requests_waiting={waiting}"
        ),
        "JSON" => format!(
            "{{\"named_metrics\":{{\"kv_cache_usage_perc\":{kv},\"num_requests_waiting\":{waiting}}}}}"
        ),
        _ => return None,
    };

    Some((ORCA_RESPONSE_HEADER.to_string(), value))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orca_text_format() {
        // Reset gauges to known values.
        let m = VllmMetrics::global();
        m.kv_cache_usage_perc.set(0.0);
        m.num_requests_waiting.set(0.0);

        let result = orca_header("TEXT");
        assert!(result.is_some());
        let (name, value) = result.unwrap();
        assert_eq!(name, ORCA_RESPONSE_HEADER);
        assert!(value.contains("kv_cache_usage_perc="));
        assert!(value.contains("num_requests_waiting="));
    }

    #[test]
    fn test_orca_json_format() {
        let m = VllmMetrics::global();
        m.kv_cache_usage_perc.set(0.0);
        m.num_requests_waiting.set(0.0);

        let result = orca_header("JSON");
        assert!(result.is_some());
        let (name, value) = result.unwrap();
        assert_eq!(name, ORCA_RESPONSE_HEADER);
        // Should be valid JSON.
        let parsed: serde_json::Value = serde_json::from_str(&value).unwrap();
        assert!(parsed["named_metrics"]["kv_cache_usage_perc"].is_number());
        assert!(parsed["named_metrics"]["num_requests_waiting"].is_number());
    }

    #[test]
    fn test_orca_case_insensitive() {
        let result = orca_header("text");
        assert!(result.is_some());

        let result = orca_header("json");
        assert!(result.is_some());

        let result = orca_header("Json");
        assert!(result.is_some());
    }

    #[test]
    fn test_orca_unsupported_format() {
        let result = orca_header("BINARY");
        assert!(result.is_none());

        let result = orca_header("");
        assert!(result.is_none());
    }

    #[test]
    fn test_orca_with_nonzero_values() {
        let m = VllmMetrics::global();
        m.kv_cache_usage_perc.set(0.75);
        m.num_requests_waiting.set(3.0);

        let (_, value) = orca_header("TEXT").unwrap();
        assert!(value.contains("kv_cache_usage_perc=0.75"));
        assert!(value.contains("num_requests_waiting=3"));

        let (_, value) = orca_header("JSON").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&value).unwrap();
        assert_eq!(parsed["named_metrics"]["kv_cache_usage_perc"], 0.75);
        assert_eq!(parsed["named_metrics"]["num_requests_waiting"], 3.0);

        // Reset.
        m.kv_cache_usage_perc.set(0.0);
        m.num_requests_waiting.set(0.0);
    }
}
