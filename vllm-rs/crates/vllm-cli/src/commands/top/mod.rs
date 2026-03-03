// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! `vllm top` — live TUI dashboard for monitoring a running vLLM server.

mod app;
mod client;
mod device;
mod heat_spark;
#[cfg(target_os = "macos")]
mod ioreprt;

use std::io;
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::ExecutableCommand;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::args::TopArgs;
use app::{App, AppEvent};
use client::StatsClient;

pub async fn run_top(args: TopArgs) -> Result<()> {
    let client = StatsClient::new(&args.host, args.port);

    // Verify server is reachable.
    eprintln!("Connecting to http://{}:{}...", args.host, args.port);
    client
        .check_health()
        .await
        .context("cannot reach vllm server — is it running?")?;

    // Fetch initial stats to populate model name.
    let initial = client
        .fetch_stats()
        .await
        .context("server is up but /stats not available — run with --enable-metrics")?;

    let model_name = initial.model_name.clone();
    let version = initial.version.clone();

    // Subscribe to SSE stream for live stats.
    let mut sse_rx = client.subscribe_stats_live(args.interval);

    // Set up terminal.
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut app = App::new(model_name, version, args.interval);
    app.on_stats(initial);

    // Channel for events from background threads → render loop.
    let (tx, rx) = mpsc::channel::<AppEvent>();

    // SSE → AppEvent bridge: forward async SSE events to the sync channel.
    let sse_tx = tx.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build tokio runtime for SSE bridge");
        rt.block_on(async move {
            while let Some(stats) = sse_rx.recv().await {
                if sse_tx.send(AppEvent::Stats(stats)).is_err() {
                    break;
                }
            }
        });
    });

    // Device sampler thread.
    let device_tx = tx.clone();
    let device_interval = args.interval;
    std::thread::spawn(move || {
        let mut sampler = device::create_sampler();
        loop {
            let devices = sampler.sample();
            if device_tx.send(AppEvent::Device(devices)).is_err() {
                break;
            }
            std::thread::sleep(Duration::from_millis(device_interval));
        }
    });

    // Input handler thread.
    let input_tx = tx;
    std::thread::spawn(move || {
        loop {
            if event::poll(Duration::from_millis(100)).unwrap_or(false)
                && let Ok(Event::Key(key)) = event::read()
            {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                // Ctrl+C → quit (not toggle color).
                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    let _ = input_tx.send(AppEvent::Quit);
                    break;
                }
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        let _ = input_tx.send(AppEvent::Quit);
                        break;
                    }
                    KeyCode::Char('c') => {
                        let _ = input_tx.send(AppEvent::ToggleColor);
                    }
                    KeyCode::Char('+') | KeyCode::Char('=') => {
                        let _ = input_tx.send(AppEvent::IncInterval);
                    }
                    KeyCode::Char('-') => {
                        let _ = input_tx.send(AppEvent::DecInterval);
                    }
                    _ => {}
                }
            }
        }
    });

    // Render loop.
    loop {
        terminal.draw(|f| app.render(f))?;

        // Process all pending events (non-blocking drain), then block for next.
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(evt) => {
                if process_event(&mut app, evt) {
                    break;
                }
                // Drain any additional queued events.
                while let Ok(evt) = rx.try_recv() {
                    if process_event(&mut app, evt) {
                        break;
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Restore terminal.
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}

/// Returns true if the app should quit.
fn process_event(app: &mut App, evt: AppEvent) -> bool {
    match evt {
        AppEvent::Stats(s) => app.on_stats(s),
        AppEvent::Device(devices) => app.on_device(devices),
        AppEvent::Quit => return true,
        AppEvent::ToggleColor => app.color = !app.color,
        AppEvent::IncInterval => {
            app.interval_ms = (app.interval_ms + 500).min(10_000);
        }
        AppEvent::DecInterval => {
            app.interval_ms = app.interval_ms.saturating_sub(500).max(200);
        }
    }
    false
}
