// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! A sparkline widget with a vertical color gradient: each cell is colored
//! by its Y position within the bar.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;

const BAR_CHARS: [&str; 9] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

/// Color gradient for the sparkline.
pub enum Gradient {
    /// Blue → Cyan → Green → Yellow → Red (for percentage metrics).
    Heat,
    /// YlGnBu: yellow → green → blue (ColorBrewer sequential).
    YlGnBu,
}

/// Gradient sparkline. Data is newest-first (right-to-left).
/// Each filled cell is colored by its vertical position in the area.
pub struct HeatSparkline<'a> {
    data: &'a [u64],
    max: Option<u64>,
    gradient: Gradient,
}

impl<'a> HeatSparkline<'a> {
    pub fn new(data: &'a [u64]) -> Self {
        Self {
            data,
            max: None,
            gradient: Gradient::Heat,
        }
    }

    pub fn max(mut self, max: u64) -> Self {
        self.max = Some(max);
        self
    }

    pub fn gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = gradient;
        self
    }
}

impl Widget for HeatSparkline<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let max = self
            .max
            .unwrap_or_else(|| self.data.iter().copied().max().unwrap_or(1))
            .max(1);
        let w = area.width as usize;
        let h = area.height as u64;
        let total_eighths = h * 8;

        for col in 0..w {
            let x = area.right().saturating_sub(1 + col as u16);
            if x < area.left() {
                break;
            }

            let val = if col < self.data.len() {
                self.data[col]
            } else {
                0
            };

            let mut bar_eighths = ((val as f64 / max as f64) * total_eighths as f64).round() as u64;

            for row in 0..area.height {
                let y = area.bottom().saturating_sub(1 + row);
                if y < area.top() {
                    break;
                }

                let (symbol, is_filled) = if bar_eighths >= 8 {
                    bar_eighths -= 8;
                    (BAR_CHARS[8], true)
                } else if bar_eighths > 0 {
                    let idx = bar_eighths as usize;
                    bar_eighths = 0;
                    (BAR_CHARS[idx], true)
                } else {
                    (BAR_CHARS[0], false)
                };

                if is_filled {
                    let y_ratio = if h > 1 {
                        row as f64 / (h - 1) as f64
                    } else {
                        val as f64 / max as f64
                    };
                    let color = gradient_color(&self.gradient, y_ratio);
                    buf.cell_mut((x, y))
                        .unwrap()
                        .set_symbol(symbol)
                        .set_style(Style::default().fg(color));
                } else {
                    buf.cell_mut((x, y)).unwrap().set_symbol(symbol);
                }
            }
        }
    }
}

fn gradient_color(gradient: &Gradient, t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    match gradient {
        Gradient::Heat => heat(t),
        Gradient::YlGnBu => {
            const STOPS: [(u8, u8, u8); 9] = [
                (8, 29, 88),     // #081d58
                (37, 52, 148),   // #253494
                (34, 94, 168),   // #225ea8
                (29, 145, 192),  // #1d91c0
                (65, 182, 196),  // #41b6c4
                (127, 205, 187), // #7fcdbb
                (199, 233, 180), // #c7e9b4
                (237, 248, 177), // #edf8b1
                (255, 255, 217), // #ffffd9
            ];
            let scaled = t * (STOPS.len() - 1) as f64;
            let idx = (scaled as usize).min(STOPS.len() - 2);
            let frac = scaled - idx as f64;
            lerp_rgb(STOPS[idx], STOPS[idx + 1], frac)
        }
    }
}

/// RdBu diverging colormap (ColorBrewer): blue → white → red.
fn heat(t: f64) -> Color {
    // 11-stop RdBu diverging (ColorBrewer): blue → white → red.
    const STOPS: [(u8, u8, u8); 11] = [
        (5, 48, 97),     // #053061
        (33, 102, 172),  // #2166ac
        (67, 147, 195),  // #4393c3
        (146, 197, 222), // #92c5de
        (209, 229, 240), // #d1e5f0
        (247, 247, 247), // #f7f7f7
        (253, 219, 199), // #fddbc7
        (244, 165, 130), // #f4a582
        (214, 96, 77),   // #d6604d
        (178, 24, 43),   // #b2182b
        (103, 0, 31),    // #67001f
    ];
    let t = t.clamp(0.0, 1.0);
    let scaled = t * (STOPS.len() - 1) as f64;
    let idx = (scaled as usize).min(STOPS.len() - 2);
    let frac = scaled - idx as f64;
    lerp_rgb(STOPS[idx], STOPS[idx + 1], frac)
}

fn lerp_rgb(from: (u8, u8, u8), to: (u8, u8, u8), t: f64) -> Color {
    let r = from.0 as f64 + (to.0 as f64 - from.0 as f64) * t;
    let g = from.1 as f64 + (to.1 as f64 - from.1 as f64) * t;
    let b = from.2 as f64 + (to.2 as f64 - from.2 as f64) * t;
    Color::Rgb(r as u8, g as u8, b as u8)
}
