//! Plays a high-fps frame sequence to a host in real time through `Client`, then checks what came back against the
//! whole engine run locally and against the real in-between frames. A debug line every second.
//!
//!   cargo run --release --example bench -- --model <dir> --frames <dir> [--step 4] [--mult 2] [--count 61]
//!       [--fps 60] [--loops 10] [--server ws://127.0.0.1:9300] [--token <secret>] [--report out/report.json]
//!
//! The source is every `step`-th frame of the sequence; phase j / mult of an interval is compared with the sequence
//! frame j * step / mult after its first frame. `--loops` plays the source that many times (steady-state rates and
//! latency); quality is measured on the last pass.
use anyhow::{ensure, Context, Result};
use serde_json::json;
use slipstream_client::{Client, Options};
use slipstream_core::{decode_image, psnr};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn quantiles(v: &[f64]) -> serde_json::Value {
    if v.is_empty() {
        return json!(null);
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    json!({ "p50": q(0.5), "p95": q(0.95), "max": s[s.len() - 1], "n": s.len() })
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: u32| get(k).map_or(Ok(d), |v| v.parse::<u32>()).context(k.to_string());
    let model = PathBuf::from(get("--model").context("--model <dir>")?);
    let dir = PathBuf::from(get("--frames").context("--frames <dir>")?);
    let server = get("--server").unwrap_or_else(|| "ws://127.0.0.1:9300".into());
    let (step, mult, count, fps) = (num("--step", 4)?, num("--mult", 2)?, num("--count", 61)?, num("--fps", 60)?);
    let loops = num("--loops", 1)?.max(1);
    ensure!(step % mult == 0, "--step must be a multiple of --mult (the real in-betweens must exist)");
    let report = PathBuf::from(get("--report").unwrap_or_else(|| "out/report.json".into()));

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    files.sort();
    ensure!(files.len() > ((count - 1) * step) as usize, "{} frames in {}", files.len(), dir.display());
    let load = |p: &Path| -> Result<(u32, u32, Vec<u8>)> { decode_image(&std::fs::read(p)?) };
    let src: Vec<Vec<u8>> =
        (0..count as usize).map(|i| load(&files[i * step as usize]).map(|f| f.2)).collect::<Result<_>>()?;
    let (w, h, _) = load(&files[0])?;

    let mut opts = Options::new(model, w, h).fps(fps as f32).ring(count as usize + 1);
    if let Some(t) = get("--token") {
        opts = opts.token(t);
    }
    let t_connect = Instant::now();
    let mut c = Client::connect(&server, opts)?;
    eprintln!("host {} ready in {:.0} ms", c.host_gpu(), t_connect.elapsed().as_secs_f64() * 1e3);

    let total = (loops * count) as u64;
    let interval = Duration::from_secs_f64(1.0 / fps as f64);
    let start = Instant::now();
    let mut push_ms = Vec::new();
    let (mut last, mut last_up, mut last_down, mut last_n) = (Instant::now(), 0u64, 0u64, (0u64, 0u64));
    for i in 0..total {
        while start.elapsed() < interval * i as u32 {
            c.poll()?;
            std::thread::sleep(Duration::from_micros(300));
        }
        let t0 = Instant::now();
        c.push(&src[(i % count as u64) as usize])?;
        push_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        c.poll()?;
        if last.elapsed() >= Duration::from_secs(1) {
            let s = last.elapsed().as_secs_f64();
            let st = &c.stats;
            let (dn, dm) = (st.frames - last_n.0, st.maps - last_n.1);
            eprintln!(
                "[client {:5.1}s] up {:5.1} ips {:5.1} KB/ipf {:5.2} Mbit/s | down {:5.1} ips {:5.1} KB/ipf {:5.2} Mbit/s | latency {:4.0} ms | host {:4.1} ms",
                start.elapsed().as_secs_f64(),
                dn as f64 / s,
                (st.bytes_up - last_up) as f64 / dn.max(1) as f64 / 1024.0,
                (st.bytes_up - last_up) as f64 * 8.0 / s / 1e6,
                dm as f64 / s,
                (st.bytes_down - last_down) as f64 / dm.max(1) as f64 / 1024.0,
                (st.bytes_down - last_down) as f64 * 8.0 / s / 1e6,
                st.latency_ms,
                st.host_ms
            );
            (last, last_up, last_down, last_n) = (Instant::now(), st.bytes_up, st.bytes_down, (st.frames, st.maps));
        }
    }
    let t_end = Instant::now();
    while c.stats.maps + 1 < total && t_end.elapsed() < Duration::from_secs(60) {
        c.wait(Duration::from_millis(100))?;
    }
    ensure!(c.stats.maps + 1 >= total, "{} of {} pairs came back", c.stats.maps, total - 1);
    let stream_s = start.elapsed().as_secs_f64();
    let latency: Vec<f64> = c.stats.recent_latency_ms.iter().rev().take(count as usize).cloned().collect();

    eprintln!("comparing the last pass");
    let base = (loops - 1) as u64 * count as u64;
    let (mut p_ss, mut p_ref, mut p_par, mut render_ms) = (vec![], vec![], vec![], vec![]);
    for i in 1..count as u64 {
        let id = base + i;
        for j in 1..mult {
            let t = j as f32 / mult as f32;
            let truth = load(&files[((i - 1) as u32 * step + j * step / mult) as usize])?.2;
            let t0 = Instant::now();
            let ss = c.render(id, t)?;
            render_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            let reference = c.reference(id, t)?;
            p_ss.push(psnr(&ss, &truth));
            p_ref.push(psnr(&reference, &truth));
            p_par.push(psnr(&ss, &reference));
        }
    }
    let st = &c.stats;
    let out = json!({
        "frame": [w, h], "fps": fps, "mult": mult, "loops": loops, "host_gpu": c.host_gpu(),
        "latency_ms": quantiles(&latency), "stream_s": stream_s,
        "up_kb_per_frame": st.bytes_up as f64 / st.frames as f64 / 1024.0,
        "down_kb_per_pair": st.bytes_down as f64 / st.maps as f64 / 1024.0,
        "up_mbit_s": st.bytes_up as f64 * 8.0 / stream_s / 1e6, "down_mbit_s": st.bytes_down as f64 * 8.0 / stream_s / 1e6,
        "client_push_ms": quantiles(&push_ms), "client_render_ms_incl_readback": mean(&render_ms),
        "psnr_slipstream_vs_truth": mean(&p_ss), "psnr_local_vs_truth": mean(&p_ref), "psnr_slipstream_vs_local": mean(&p_par),
    });
    if let Some(d) = report.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(&report, serde_json::to_string_pretty(&out)?)?;
    println!("{}", serde_json::to_string_pretty(&out)?);
    c.close();
    Ok(())
}
