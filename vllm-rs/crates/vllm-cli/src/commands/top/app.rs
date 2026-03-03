// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! TUI application — ratatui render loop and state.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, RenderDirection, Sparkline};

use super::client::StatsResponse;
use super::device::DeviceMetrics;
use super::heat_spark::{Gradient, HeatSparkline};

/// Max sparkline history entries.
const MAX_SPARKLINE: usize = 128;

/// GPU colors for multi-GPU sparklines.
const GPU_COLORS: &[Color] = &[
    Color::Green,
    Color::Cyan,
    Color::Yellow,
    Color::Magenta,
    Color::Blue,
    Color::Red,
    Color::LightGreen,
    Color::LightCyan,
];

/// Events fed into the TUI render loop.
pub enum AppEvent {
    Stats(StatsResponse),
    Device(Vec<DeviceMetrics>),
    Quit,
    ToggleColor,
    IncInterval,
    DecInterval,
}

/// Per-GPU sparkline + latest metrics.
struct GpuState {
    util_spark: Vec<u64>,
    latest: DeviceMetrics,
}

impl GpuState {
    fn new() -> Self {
        Self {
            util_spark: Vec::new(),
            latest: DeviceMetrics::default(),
        }
    }
}

pub struct App {
    pub model_name: String,
    pub version: String,
    pub interval_ms: u64,
    pub color: bool,

    // Latest raw counters (for rate computation).
    prev_stats: Option<StatsResponse>,
    prev_time: Option<std::time::Instant>,

    // Computed rates.
    req_per_sec: f64,
    prompt_tok_per_sec: f64,
    output_tok_per_sec: f64,

    // Sparkline histories (newest first — prepend, render right-to-left).
    req_spark: Vec<u64>,
    prompt_tok_spark: Vec<u64>,
    output_tok_spark: Vec<u64>,
    kv_spark: Vec<u64>,

    // Latest gauges.
    requests_active: i64,
    num_running: f64,
    num_waiting: f64,
    kv_cache_usage: f64,
    gpu_blocks_used: i64,
    gpu_blocks_total: i64,

    // Histogram averages + sparkline history (ms, scaled to u64 for sparkline).
    avg_ttft_ms: f64,
    avg_itl_ms: f64,
    avg_latency_ms: f64,
    ttft_spark: Vec<u64>,
    itl_spark: Vec<u64>,
    latency_spark: Vec<u64>,

    // Per-GPU state (indexed by GPU ordinal).
    gpus: Vec<GpuState>,
}

impl App {
    pub fn new(model_name: String, version: String, interval_ms: u64) -> Self {
        Self {
            model_name,
            version,
            interval_ms,
            color: true,
            prev_stats: None,
            prev_time: None,
            req_per_sec: 0.0,
            prompt_tok_per_sec: 0.0,
            output_tok_per_sec: 0.0,
            req_spark: Vec::new(),
            prompt_tok_spark: Vec::new(),
            output_tok_spark: Vec::new(),
            kv_spark: Vec::new(),
            requests_active: 0,
            num_running: 0.0,
            num_waiting: 0.0,
            kv_cache_usage: 0.0,
            gpu_blocks_used: 0,
            gpu_blocks_total: 0,
            avg_ttft_ms: 0.0,
            avg_itl_ms: 0.0,
            avg_latency_ms: 0.0,
            ttft_spark: Vec::new(),
            itl_spark: Vec::new(),
            latency_spark: Vec::new(),
            gpus: Vec::new(),
        }
    }

    pub fn on_stats(&mut self, s: StatsResponse) {
        let now = std::time::Instant::now();

        // Compute rates from counter deltas.
        if let (Some(prev), Some(prev_t)) = (&self.prev_stats, self.prev_time) {
            let dt = now.duration_since(prev_t).as_secs_f64().max(0.001);
            self.req_per_sec = (s.requests_total.saturating_sub(prev.requests_total)) as f64 / dt;
            self.prompt_tok_per_sec =
                (s.prompt_tokens_total
                    .saturating_sub(prev.prompt_tokens_total)) as f64
                    / dt;
            self.output_tok_per_sec =
                (s.output_tokens_total
                    .saturating_sub(prev.output_tokens_total)) as f64
                    / dt;

            // Histogram deltas → rolling average.
            let ttft_d_sum = s.ttft_sum - prev.ttft_sum;
            let ttft_d_count = s.ttft_count.saturating_sub(prev.ttft_count);
            if ttft_d_count > 0 {
                self.avg_ttft_ms = (ttft_d_sum / ttft_d_count as f64) * 1000.0;
            }
            let itl_d_sum = s.itl_sum - prev.itl_sum;
            let itl_d_count = s.itl_count.saturating_sub(prev.itl_count);
            if itl_d_count > 0 {
                self.avg_itl_ms = (itl_d_sum / itl_d_count as f64) * 1000.0;
            }
            let lat_d_sum = s.latency_sum - prev.latency_sum;
            let lat_d_count = s.latency_count.saturating_sub(prev.latency_count);
            if lat_d_count > 0 {
                self.avg_latency_ms = (lat_d_sum / lat_d_count as f64) * 1000.0;
            }
        }

        // Update gauges.
        self.requests_active = s.requests_active;
        self.num_running = s.num_requests_running;
        self.num_waiting = s.num_requests_waiting;
        self.kv_cache_usage = s.kv_cache_usage;
        self.gpu_blocks_used = s.gpu_cache_blocks_used;
        self.gpu_blocks_total = s.gpu_cache_blocks_total;

        if self.model_name.is_empty() {
            self.model_name.clone_from(&s.model_name);
        }

        // Push sparkline data (newest first for right-to-left rendering).
        push_spark(&mut self.req_spark, self.req_per_sec as u64);
        push_spark(&mut self.prompt_tok_spark, self.prompt_tok_per_sec as u64);
        push_spark(&mut self.output_tok_spark, self.output_tok_per_sec as u64);
        push_spark(
            &mut self.kv_spark,
            (self.kv_cache_usage * 100.0).round() as u64,
        );
        // Latency in tenths-of-ms for sparkline resolution (avoid rounding to 0).
        push_spark(&mut self.ttft_spark, (self.avg_ttft_ms * 10.0) as u64);
        push_spark(&mut self.itl_spark, (self.avg_itl_ms * 10.0) as u64);
        push_spark(&mut self.latency_spark, (self.avg_latency_ms * 10.0) as u64);

        self.prev_stats = Some(s);
        self.prev_time = Some(now);
    }

    pub fn on_device(&mut self, devices: Vec<DeviceMetrics>) {
        // Grow/shrink to match GPU count.
        while self.gpus.len() < devices.len() {
            self.gpus.push(GpuState::new());
        }
        self.gpus.truncate(devices.len());

        for (gpu, metrics) in self.gpus.iter_mut().zip(devices) {
            push_spark(&mut gpu.util_spark, metrics.gpu_util_pct.round() as u64);
            gpu.latest = metrics;
        }
    }

    pub fn render(&self, f: &mut Frame) {
        let size = f.area();

        // Title bar.
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(0)])
            .split(size);

        let title = Line::from(vec![
            Span::styled(
                " vllm top ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!(
                " {} | v{} | {}ms",
                self.model_name, self.version, self.interval_ms
            )),
            Span::styled(
                "  q:quit  c:color  +/-:interval ",
                Style::default().fg(Color::DarkGray),
            ),
        ]);
        f.render_widget(Paragraph::new(title), chunks[0]);

        // Left: Throughput + Latency stacked. Right: Device (full height).
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[1]);
        let left_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(cols[0]);

        self.render_throughput(f, left_rows[0]);
        self.render_kv_cache(f, left_rows[1]);
        self.render_device(f, cols[1]);
    }

    fn render_throughput(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" Throughput & Latency ")
            .borders(Borders::ALL)
            .border_style(self.border_style());
        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(1),
                Constraint::Min(3),
            ])
            .split(inner);

        let summary = Paragraph::new(vec![
            Line::from(format!(
                "  req/s: {:.1}    prompt tok/s: {:.0}    output tok/s: {:.0}",
                self.req_per_sec, self.prompt_tok_per_sec, self.output_tok_per_sec
            )),
            Line::from(format!(
                "  active: {}    running: {:.0}    waiting: {:.0}    TTFT: {:.1}ms    ITL: {:.1}ms",
                self.requests_active,
                self.num_running,
                self.num_waiting,
                self.avg_ttft_ms,
                self.avg_itl_ms
            )),
        ]);
        f.render_widget(summary, chunks[0]);

        let label = Paragraph::new(Line::from(Span::styled(
            "  output tok/s",
            Style::default().fg(Color::DarkGray),
        )));
        f.render_widget(label, chunks[1]);

        if self.color {
            f.render_widget(
                HeatSparkline::new(&self.output_tok_spark).gradient(Gradient::YlGnBu),
                chunks[2],
            );
        } else {
            let spark = Sparkline::default()
                .direction(RenderDirection::RightToLeft)
                .data(&self.output_tok_spark)
                .style(Style::default().fg(Color::White));
            f.render_widget(spark, chunks[2]);
        }
    }

    fn render_kv_cache(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(" KV Cache ")
            .borders(Borders::ALL)
            .border_style(self.border_style());
        let inner = block.inner(area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Length(3),
                Constraint::Length(1),
                Constraint::Min(3),
            ])
            .split(inner);

        let pct = (self.kv_cache_usage * 100.0).min(100.0);
        let info = Paragraph::new(vec![Line::from(format!(
            "  blocks: {} / {}    usage: {:.1}%",
            self.gpu_blocks_used, self.gpu_blocks_total, pct
        ))]);
        f.render_widget(info, chunks[0]);

        let gauge_color = if pct > 90.0 {
            Color::Red
        } else if pct > 70.0 {
            Color::Yellow
        } else {
            Color::Green
        };
        let gauge = Gauge::default()
            .block(Block::default())
            .gauge_style(
                Style::default()
                    .fg(if self.color {
                        gauge_color
                    } else {
                        Color::White
                    })
                    .bg(Color::DarkGray),
            )
            .percent(pct.round() as u16)
            .label(format!("{:.1}%", pct));
        f.render_widget(gauge, chunks[1]);

        let label = Paragraph::new(Line::from(Span::styled(
            "  cache usage %",
            Style::default().fg(Color::DarkGray),
        )));
        f.render_widget(label, chunks[2]);

        if self.color {
            f.render_widget(
                HeatSparkline::new(&self.kv_spark)
                    .max(100)
                    .gradient(Gradient::Heat),
                chunks[3],
            );
        } else {
            let spark = Sparkline::default()
                .max(100)
                .direction(RenderDirection::RightToLeft)
                .data(&self.kv_spark)
                .style(Style::default().fg(Color::White));
            f.render_widget(spark, chunks[3]);
        }
    }

    fn render_device(&self, f: &mut Frame, area: Rect) {
        let n = self.gpus.len();
        let title = if n > 1 {
            format!(" Device ({n} GPUs) ")
        } else {
            " Device ".to_string()
        };
        let block = Block::default()
            .title(title)
            .borders(Borders::ALL)
            .border_style(self.border_style());
        let inner = block.inner(area);
        f.render_widget(block, area);

        if self.gpus.is_empty() {
            let lines = vec![
                Line::raw(""),
                Line::from(Span::styled(
                    "  No device metrics available",
                    Style::default().fg(Color::DarkGray),
                )),
            ];
            f.render_widget(Paragraph::new(lines), inner);
            return;
        }

        // Layout: summary lines at top, then one sparkline row per GPU.
        let summary_height = if n == 1 { 2 } else { (n as u16).min(4) };
        let constraints = vec![
            Constraint::Length(summary_height),
            Constraint::Length(1), // label
            Constraint::Min(3),    // sparklines
        ];
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(inner);

        // Summary: one line per GPU with key stats.
        let mut lines = Vec::new();
        if n == 1 {
            let d = &self.gpus[0].latest;
            let mem_pct = if d.gpu_mem_total_mb > 0.0 {
                d.gpu_mem_used_mb / d.gpu_mem_total_mb * 100.0
            } else {
                0.0
            };
            lines.push(Line::from(format!(
                "  GPU util:  {:>5.1}%    Mem: {:.0}/{:.0} MB ({:.1}%)",
                d.gpu_util_pct, d.gpu_mem_used_mb, d.gpu_mem_total_mb, mem_pct
            )));
            let power_str = if d.gpu_power_limit_w > 0.0 {
                format!("{:.1}/{:.0}W", d.gpu_power_w, d.gpu_power_limit_w)
            } else {
                format!("{:.1}W", d.gpu_power_w)
            };
            let temp_str = if d.gpu_temp_c > 0.0 {
                format!("{:.0}C", d.gpu_temp_c)
            } else {
                String::new()
            };
            let clock_str = if d.gpu_clock_mhz > 0.0 {
                format!("    Clock: {:.0} MHz", d.gpu_clock_mhz)
            } else {
                String::new()
            };
            lines.push(Line::from(format!(
                "  Power: {power_str}    {temp_str}{clock_str}"
            )));
        } else {
            for (i, gpu) in self.gpus.iter().enumerate() {
                let d = &gpu.latest;
                let mem_pct = if d.gpu_mem_total_mb > 0.0 {
                    d.gpu_mem_used_mb / d.gpu_mem_total_mb * 100.0
                } else {
                    0.0
                };
                let color = gpu_color(i, self.color);
                lines.push(Line::from(vec![
                    Span::styled(format!("  GPU {i}"), Style::default().fg(color)),
                    Span::raw(format!(
                        "  util:{:>5.1}%  mem:{:.0}/{:.0}MB({:.0}%)  {:.0}C {:.1}W",
                        d.gpu_util_pct,
                        d.gpu_mem_used_mb,
                        d.gpu_mem_total_mb,
                        mem_pct,
                        d.gpu_temp_c,
                        d.gpu_power_w,
                    )),
                ]));
            }
        }
        f.render_widget(Paragraph::new(lines), chunks[0]);

        // Label.
        let label_text = if n == 1 {
            "  GPU utilization %"
        } else {
            "  GPU utilization % (per GPU)"
        };
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                label_text,
                Style::default().fg(Color::DarkGray),
            ))),
            chunks[1],
        );

        // Sparklines: split vertically, one row per GPU.
        let spark_area = chunks[2];
        if n == 1 {
            if self.color {
                f.render_widget(
                    HeatSparkline::new(&self.gpus[0].util_spark)
                        .max(100)
                        .gradient(Gradient::Heat),
                    spark_area,
                );
            } else {
                let spark = Sparkline::default()
                    .max(100)
                    .direction(RenderDirection::RightToLeft)
                    .data(&self.gpus[0].util_spark)
                    .style(Style::default().fg(Color::White));
                f.render_widget(spark, spark_area);
            }
        } else {
            let per_gpu: Vec<Constraint> = (0..n).map(|_| Constraint::Min(1)).collect();
            let rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints(per_gpu)
                .split(spark_area);
            for (i, gpu) in self.gpus.iter().enumerate() {
                if i >= rows.len() {
                    break;
                }
                if self.color {
                    f.render_widget(
                        HeatSparkline::new(&gpu.util_spark)
                            .max(100)
                            .gradient(Gradient::Heat),
                        rows[i],
                    );
                } else {
                    let spark = Sparkline::default()
                        .max(100)
                        .direction(RenderDirection::RightToLeft)
                        .data(&gpu.util_spark)
                        .style(Style::default().fg(Color::White));
                    f.render_widget(spark, rows[i]);
                }
            }
        }
    }

    fn border_style(&self) -> Style {
        if self.color {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::White)
        }
    }
}

fn gpu_color(idx: usize, color: bool) -> Color {
    if color {
        GPU_COLORS[idx % GPU_COLORS.len()]
    } else {
        Color::White
    }
}

/// Prepend a value (newest first) and truncate to MAX_SPARKLINE.
fn push_spark(buf: &mut Vec<u64>, val: u64) {
    buf.insert(0, val);
    buf.truncate(MAX_SPARKLINE);
}
