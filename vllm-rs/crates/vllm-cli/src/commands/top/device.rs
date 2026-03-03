// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Device metrics sampling (GPU utilization, memory, temperature, power).

#[cfg(target_os = "macos")]
use super::ioreprt::IOReportSampler;

#[derive(Debug, Clone, Default)]
pub struct DeviceMetrics {
    pub gpu_util_pct: f64,
    pub gpu_mem_used_mb: f64,
    pub gpu_mem_total_mb: f64,
    pub gpu_temp_c: f64,
    pub gpu_power_w: f64,
    pub gpu_power_limit_w: f64,
    pub gpu_clock_mhz: f64,
}

pub trait DeviceSampler: Send {
    /// Sample metrics for all devices. Returns one entry per GPU/device.
    fn sample(&mut self) -> Vec<DeviceMetrics>;
}

// ---------------------------------------------------------------------------
// nvidia-smi sampler (Linux/CUDA) — returns one DeviceMetrics per GPU
// ---------------------------------------------------------------------------

#[cfg(not(target_os = "macos"))]
pub struct NvidiaSampler;

#[cfg(not(target_os = "macos"))]
impl NvidiaSampler {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(not(target_os = "macos"))]
impl DeviceSampler for NvidiaSampler {
    fn sample(&mut self) -> Vec<DeviceMetrics> {
        let output = match std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.current.sm",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            Ok(o) if o.status.success() => o,
            _ => return Vec::new(),
        };
        let text = String::from_utf8_lossy(&output.stdout);
        text.lines()
            .filter_map(|line| {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() < 7 {
                    return None;
                }
                Some(DeviceMetrics {
                    gpu_util_pct: parts[0].parse().unwrap_or(0.0),
                    gpu_mem_used_mb: parts[1].parse().unwrap_or(0.0),
                    gpu_mem_total_mb: parts[2].parse().unwrap_or(0.0),
                    gpu_temp_c: parts[3].parse().unwrap_or(0.0),
                    gpu_power_w: parts[4].parse().unwrap_or(0.0),
                    gpu_power_limit_w: parts[5].parse().unwrap_or(0.0),
                    gpu_clock_mhz: parts[6].parse().unwrap_or(0.0),
                })
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// macOS sampler (Apple Silicon — IOReport for GPU + sysctl for memory)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
pub struct MacosSampler {
    total_mem_mb: f64,
    io_sampler: Option<IOReportSampler>,
}

#[cfg(target_os = "macos")]
impl MacosSampler {
    pub fn new() -> Self {
        let total_mem_mb = {
            let mut size: u64 = 0;
            let mut len = std::mem::size_of::<u64>();
            let name = c"hw.memsize";
            unsafe {
                libc::sysctl(
                    name.as_ptr() as *mut _,
                    2,
                    &mut size as *mut u64 as *mut _,
                    &mut len,
                    std::ptr::null_mut(),
                    0,
                );
            }
            size as f64 / (1024.0 * 1024.0)
        };
        let io_sampler = IOReportSampler::new();
        Self {
            total_mem_mb,
            io_sampler,
        }
    }

    fn mem_used_mb(&self) -> f64 {
        let output = match std::process::Command::new("vm_stat").output() {
            Ok(o) if o.status.success() => o,
            _ => return 0.0,
        };
        let text = String::from_utf8_lossy(&output.stdout);
        let page_size: f64 = 16384.0; // Apple Silicon default
        let mut pages_active = 0u64;
        let mut pages_wired = 0u64;
        let mut pages_compressed = 0u64;
        for line in text.lines() {
            if let Some(val) = parse_vmstat_line(line, "Pages active:") {
                pages_active = val;
            } else if let Some(val) = parse_vmstat_line(line, "Pages wired down:") {
                pages_wired = val;
            } else if let Some(val) = parse_vmstat_line(line, "Pages occupied by compressor:") {
                pages_compressed = val;
            }
        }
        (pages_active + pages_wired + pages_compressed) as f64 * page_size / (1024.0 * 1024.0)
    }
}

#[cfg(target_os = "macos")]
impl DeviceSampler for MacosSampler {
    fn sample(&mut self) -> Vec<DeviceMetrics> {
        let mem_used = self.mem_used_mb();
        let (gpu_util, gpu_freq, gpu_power) = match &mut self.io_sampler {
            Some(s) => s.sample(),
            None => (0.0, 0.0, 0.0),
        };

        vec![DeviceMetrics {
            gpu_util_pct: gpu_util,
            gpu_mem_used_mb: mem_used,
            gpu_mem_total_mb: self.total_mem_mb,
            gpu_temp_c: 0.0, // requires SMC (follow-up)
            gpu_power_w: gpu_power,
            gpu_power_limit_w: 0.0,
            gpu_clock_mhz: gpu_freq,
        }]
    }
}

#[cfg(target_os = "macos")]
fn parse_vmstat_line(line: &str, prefix: &str) -> Option<u64> {
    let stripped = line.strip_prefix(prefix)?;
    stripped.trim().trim_end_matches('.').parse().ok()
}

// ---------------------------------------------------------------------------
// Auto-detect best sampler for this platform
// ---------------------------------------------------------------------------

pub fn create_sampler() -> Box<dyn DeviceSampler> {
    #[cfg(target_os = "macos")]
    {
        Box::new(MacosSampler::new())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let has_nvidia_smi = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if has_nvidia_smi {
            Box::new(NvidiaSampler::new())
        } else {
            Box::new(NoopSampler)
        }
    }
}

#[cfg(not(target_os = "macos"))]
struct NoopSampler;
#[cfg(not(target_os = "macos"))]
impl DeviceSampler for NoopSampler {
    fn sample(&mut self) -> Vec<DeviceMetrics> {
        Vec::new()
    }
}
