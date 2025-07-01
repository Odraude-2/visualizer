use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode};
use ratatui::{
    backend::CrosstermBackend,
    layout::Rect,
    style::{Color, Style},
    widgets::{Block, Bar, BarChart, BarGroup},
    Terminal,
};
use rustfft::{num_complex::Complex, FftPlanner};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::stdout;
use std::sync::{Arc, Mutex};
use std::time::Duration;

// --- Constants for configuration ---
const SMOOTHING_FACTOR: f32 = 0.995; // Keep 99.5% of the previous peak
const GRAVITY_ACCELERATION: f32 = 0.05; // Speed at which bars fall
const SCALING_FACTOR: f32 = 0.25; // Overall height of the bars
const MIN_BAR_HEIGHT: f32 = 0.01; // Minimum height for a bar to be visible (as a fraction of max_height)
const MAX_BARS: u16 = 10; // Maximum number of bars to render

#[derive(Debug, Deserialize, Serialize)]
struct AppConfig {
    colors: ColorsConfig,
}

#[derive(Debug, Deserialize, Serialize)]
struct ColorsConfig {
    low: (u8, u8, u8),
    medium: (u8, u8, u8),
    high: (u8, u8, u8),
    peak: (u8, u8, u8),
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            colors: ColorsConfig {
                low: (0, 0, 255),    // Blue
                medium: (0, 255, 0), // Green
                high: (255, 255, 0), // Yellow
                peak: (255, 0, 0),   // Red
            },
        }
    }
}

impl AppConfig {
    fn load() -> Self {
        match fs::read_to_string("config.toml") {
            Ok(content) => match toml::from_str(&content) {
                Ok(config) => config,
                Err(e) => {
                    eprintln!("Error parsing config.toml: {}. Using default config.", e);
                    Self::default()
                }
            },
            Err(e) => {
                eprintln!("Error reading config.toml: {}. Creating default config.", e);
                let default_config = Self::default();
                if let Err(e) = fs::write(
                    "config.toml",
                    toml::to_string_pretty(&default_config).expect("Failed to serialize default config"),
                ) {
                    eprintln!("Error writing default config.toml: {}", e);
                }
                default_config
            }
        }
    }
}

/// Represents the state of the visualizer bars
struct VisualizerState {
    peaks: Vec<f32>,
}

impl VisualizerState {
    fn new(bar_count: usize) -> Self {
        Self { peaks: vec![0.0; bar_count] }
    }

    /// Update the peaks with new data, applying smoothing and gravity
    fn update(&mut self, new_magnitudes: &[f32]) {
        for (i, &new_val) in new_magnitudes.iter().enumerate() {
            if new_val > self.peaks[i] {
                self.peaks[i] = new_val;
            } else {
                // Apply gravity
                self.peaks[i] -= GRAVITY_ACCELERATION;
                // Apply smoothing
                self.peaks[i] *= SMOOTHING_FACTOR;
                if self.peaks[i] < 0.0 {
                    self.peaks[i] = 0.0;
                }
            }
        }
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let config = AppConfig::load();

    // --- Audio Capture Setup ---
    let audio_buffer = Arc::new(Mutex::new(Vec::new()));
    let (stream, sample_rate) = setup_audio_capture(audio_buffer.clone())?;
    stream.play()?;

    // --- Terminal Setup ---
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    crossterm::terminal::enable_raw_mode()?;
    terminal.clear()?;

    let mut visualizer_state = VisualizerState::new(0);
    let mut fft_processor = FftProcessor::new();

    // --- Main Loop ---
    loop {
        if crossterm::event::poll(Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }

        let fft_resolution = audio_buffer.lock().unwrap().len();
        if fft_resolution == 0 {
            continue;
        }
        let magnitudes = fft_processor.process_audio_data(&audio_buffer, fft_resolution);

        terminal.draw(|f| {
            let size = f.size();
            if size.width == 0 || size.height == 0 {
                return;
            }

            // Calculate dynamic bar width and gap
            let min_bar_width_chars = 2; // Minimum width for a bar (e.g., "██")
            let min_gap_chars = 1;       // Minimum gap between bars (e.g., " ")

            // Calculate the maximum number of bars that can fit
            let max_bars_that_fit = size.width / (min_bar_width_chars + min_gap_chars);
            let num_bars = (max_bars_that_fit as u16).min(MAX_BARS);

            // If no bars can be displayed, return
            if num_bars == 0 {
                return;
            }

            // Calculate the actual bar width and gap to fill the screen
            let total_width_needed = num_bars * min_bar_width_chars + (num_bars - 1) * min_gap_chars;
            let remaining_width = size.width - total_width_needed;

            let bar_width = min_bar_width_chars + (remaining_width / num_bars);
            let bar_gap = min_gap_chars; // Keep minimum gap, distribute extra width to bars

            if visualizer_state.peaks.len() != num_bars as usize {
                visualizer_state = VisualizerState::new(num_bars as usize);
            }

            let processed_magnitudes = distribute_magnitudes(&magnitudes, num_bars as usize, sample_rate, fft_resolution);
            visualizer_state.update(&processed_magnitudes);

            draw_bars(f, &visualizer_state.peaks, size, bar_width, bar_gap, &config.colors);
        })?;
    }

    // --- Cleanup ---
    crossterm::terminal::disable_raw_mode()?;
    terminal.show_cursor()?;
    Ok(())
}

/// Sets up cpal to capture audio from the default input device
fn setup_audio_capture(audio_buffer: Arc<Mutex<Vec<f32>>>) -> std::result::Result<(cpal::Stream, u32), Box<dyn std::error::Error>> {
    let host = cpal::default_host();

    // Try to find a loopback device (e.g., "Stereo Mix")
    let mut selected_device = None;
    for device in host.input_devices()? {
        if let Ok(name) = device.name() {
            if name.to_lowercase().contains("stereo mix") || name.to_lowercase().contains("what u hear") {
                eprintln!("Found loopback device: {}", name);
                selected_device = Some(device);
                break;
            }
        }
    }

    let device = selected_device.unwrap_or_else(|| {
        eprintln!("No 'Stereo Mix' or 'What U Hear' device found. Falling back to default input device (microphone).");
        host.default_input_device().expect("No default input device available")
    });

    let config: cpal::SupportedStreamConfig = device.default_input_config()?.into();
    let sample_rate = config.sample_rate().0;

    let stream = device.build_input_stream(
        &config.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = audio_buffer.lock().unwrap();
            buffer.clear();
            buffer.extend_from_slice(data);
        },
        |err| eprintln!("An error occurred on stream: {}", err),
        None,
    )?;
    Ok((stream, sample_rate))
}

/// Helper struct to manage FFT resources
struct FftProcessor {
    planner: FftPlanner<f32>,
    fft: Option<Arc<dyn rustfft::Fft<f32>>>,
    current_fft_resolution: usize,
}

impl FftProcessor {
    fn new() -> Self {
        FftProcessor {
            planner: FftPlanner::new(),
            fft: None,
            current_fft_resolution: 0,
        }
    }

    /// Processes the raw audio buffer using FFT, reusing the FFT instance if possible
    fn process_audio_data(&mut self, audio_buffer: &Arc<Mutex<Vec<f32>>>, fft_resolution: usize) -> Vec<f32> {
        let audio_data = audio_buffer.lock().unwrap();
        if audio_data.is_empty() {
            return Vec::new();
        }

        // Re-plan FFT only if resolution changes
        if self.current_fft_resolution != fft_resolution || self.fft.is_none() {
            self.fft = Some(self.planner.plan_fft_forward(fft_resolution));
            self.current_fft_resolution = fft_resolution;
        }

        let fft = self.fft.as_ref().unwrap();

        let mut buffer: Vec<Complex<f32>> = audio_data
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .collect();
        buffer.resize(fft_resolution, Complex::new(0.0, 0.0));

        fft.process(&mut buffer);

        let magnitudes: Vec<f32> = buffer[0..fft_resolution / 2]
            .iter()
            .map(|c| c.norm())
            .collect();

        magnitudes
    }
}

/// Distributes FFT magnitudes across the available terminal columns (bars) using a logarithmic scale
fn distribute_magnitudes(magnitudes: &[f32], num_bars: usize, sample_rate: u32, fft_resolution: usize) -> Vec<f32> {
    if num_bars == 0 || magnitudes.is_empty() {
        return vec![0.0; num_bars];
    }

    let mut distributed = vec![0.0; num_bars];

    let min_freq: f32 = 20.0; // Hz
    let max_freq: f32 = (sample_rate as f32 / 2.0) * 0.9; // Max frequency is Nyquist, but limit to 90% for stability

    let log_min_freq = min_freq.log10();
    let log_max_freq = max_freq.log10();
    let log_range = log_max_freq - log_min_freq;

    for i in 0..num_bars {
        let bar_start_log_freq = log_min_freq + (i as f32 / num_bars as f32) * log_range;
        let bar_end_log_freq = log_min_freq + ((i + 1) as f32 / num_bars as f32) * log_range;

        let bar_start_freq = 10.0_f32.powf(bar_start_log_freq);
        let bar_end_freq = 10.0_f32.powf(bar_end_log_freq);

        // Convert frequencies to FFT bin indices
        let start_bin = (bar_start_freq * fft_resolution as f32 / sample_rate as f32) as usize;
        let end_bin = (bar_end_freq * fft_resolution as f32 / sample_rate as f32) as usize;

        let start_bin = start_bin.min(magnitudes.len() - 1);
        let end_bin = end_bin.min(magnitudes.len() - 1);

        let range = start_bin..end_bin;

        let avg: f32 = if range.is_empty() || start_bin >= end_bin {
            0.0
        } else {
            magnitudes[range].iter().sum::<f32>() / (end_bin - start_bin) as f32
        };
        distributed[i] = avg;
    }

    distributed
}

/// Draws the final bars on the terminal UI using a BarChart widget
fn draw_bars(f: &mut ratatui::Frame, peaks: &[f32], area: Rect, bar_width: u16, bar_gap: u16, colors: &ColorsConfig) {
    let max_height = area.height as f32;

    let bars: Vec<Bar> = peaks
        .iter()
        .map(|&peak| {
            let height = ((peak * SCALING_FACTOR * max_height).max(MIN_BAR_HEIGHT * max_height)).min(max_height) as u64;
            let color = match height as f32 / max_height {
                h if h > 0.75 => Color::Rgb(colors.peak.0, colors.peak.1, colors.peak.2),
                h if h > 0.5 => Color::Rgb(colors.high.0, colors.high.1, colors.high.2),
                h if h > 0.25 => Color::Rgb(colors.medium.0, colors.medium.1, colors.medium.2),
                _ => Color::Rgb(colors.low.0, colors.low.1, colors.low.2),
            };
            Bar::default()
                .value(height)
                .style(Style::default().fg(color))
                .value_style(Style::default().bg(color))
        })
        .collect();

    let barchart = BarChart::default()
        .block(Block::default())
        .bar_width(bar_width)
        .bar_gap(bar_gap)
        .data(BarGroup::default().bars(&bars));

    f.render_widget(barchart, area);
}
