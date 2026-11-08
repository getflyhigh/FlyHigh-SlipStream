//! SlipStream host capacity: N clients at once, each playing the same instructions (a pre-encoded H.264 stream of a
//! clip's network inputs, looped) at the given frame rate, for a while. Reports per client how many maps came back
//! per second and how late. Needs no GPU: the instructions are made once on the CPU (`rung::downscale` + openh264).
//!
//!   cargo run --release --example load -- --frames <dir> [--step 10] [--count 49] [--fps 24] [--kbps 1500]
//!       [--clients 8] [--seconds 20] [--server ws://127.0.0.1:9300] [--token <secret>]
use anyhow::{ensure, Context, Result};
use openh264::encoder::{
    BitRate, Complexity, Encoder, EncoderConfig, FrameRate, IntraFramePeriod, RateControlMode, UsageType,
};
use openh264::formats::{RgbSliceU8, YUVBuffer};
use openh264::OpenH264API;
use slipstream_client::Link;
use slipstream_core::proto::Hello;
use slipstream_core::{decode_image, rung};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct ClientResult {
    maps: u32,
    lat: Vec<f64>,
    error: Option<String>,
    secs: f64,
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: u32| get(k).map_or(Ok(d), |v| v.parse::<u32>()).context(k.to_string());
    let dir = PathBuf::from(get("--frames").context("--frames <dir>")?);
    let (step, count, fps, kbps) = (num("--step", 10)?, num("--count", 49)?, num("--fps", 24)?, num("--kbps", 1500)?);
    let (clients, seconds) = (num("--clients", 8)?, num("--seconds", 20)?);
    let server = get("--server").unwrap_or_else(|| "ws://127.0.0.1:9300".into());
    let token = get("--token").unwrap_or_default();

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    files.sort();
    ensure!(files.len() >= ((count - 1) * step + 1) as usize, "not enough frames in {}", dir.display());
    let (w, h, _) = decode_image(&std::fs::read(&files[0])?)?;
    let hello =
        Hello { protocol: slipstream_core::proto::PROTOCOL, w, h, k: 1, codec: "j50.40".into(), token: token.clone() };
    let probe = Link::connect(&server, &hello)?;
    let (rungs, work) = (probe.ready.rung, probe.ready.work);
    probe.close();

    eprintln!("encoding {count} frames ({w}x{h} -> {}x{}) at {kbps} kbit/s", rungs.0, rungs.1);
    let cfg = EncoderConfig::new()
        .bitrate(BitRate::from_bps(kbps * 1000))
        .max_frame_rate(FrameRate::from_hz(fps as f32))
        .usage_type(UsageType::CameraVideoRealTime)
        .rate_control_mode(RateControlMode::Bitrate)
        .skip_frames(false)
        .intra_frame_period(IntraFramePeriod::from_num_frames(0))
        .complexity(Complexity::Low);
    let mut enc = Encoder::with_api_config(OpenH264API::from_source(), cfg)?;
    let mut aus = Vec::new();
    for i in 0..count as usize {
        let (_, _, f) = decode_image(&std::fs::read(&files[i * step as usize])?)?;
        let rgb = rung::to_rgb8(&rung::downscale(&f, w, h, work, rungs), rungs.0, rungs.1);
        let yuv = YUVBuffer::from_rgb8_source(RgbSliceU8::new(&rgb, (rungs.0 as usize, rungs.1 as usize)));
        aus.push(enc.encode(&yuv)?.to_vec());
    }
    // the first access unit is the only IDR (with SPS / PPS): looping back to it resets the host's decoder cleanly
    let aus = Arc::new(aus);
    let kb = aus.iter().map(|a| a.len()).sum::<usize>() as f64 / aus.len() as f64 / 1024.0;
    eprintln!("{clients} clients x {fps} fps for {seconds} s, {kb:.1} KB per frame");

    let threads: Vec<_> = (0..clients)
        .map(|c| {
            let (aus, hello, server) = (aus.clone(), hello.clone(), server.clone());
            std::thread::spawn(move || -> ClientResult {
                let mut r = ClientResult { maps: 0, lat: Vec::new(), error: None, secs: 0.0 };
                let mut link = match Link::connect(&server, &hello) {
                    Ok(l) => l,
                    Err(e) => {
                        r.error = Some(format!("{e:#}"));
                        return r;
                    }
                };
                // staggered starts, like users arriving: the host's sessions are not all built in the same instant
                std::thread::sleep(Duration::from_millis(150 * c as u64));
                let clock = Instant::now();
                let now = || clock.elapsed().as_secs_f64() * 1e3;
                let interval = 1000.0 / fps as f64;
                let total = (seconds * fps) as usize;
                let mut run = || -> Result<()> {
                    for i in 0..total {
                        while now() < i as f64 * interval {
                            match link.poll()? {
                                Some((hd, _)) => {
                                    r.maps += 1;
                                    r.lat.push(now() - hd.time_ms);
                                }
                                None => std::thread::sleep(Duration::from_micros(500)),
                            }
                        }
                        link.send_frame(i as u32, now(), &aus[i % aus.len()])?;
                    }
                    let end = now();
                    while (r.maps as usize) < total - 1 && now() - end < 10000.0 {
                        match link.poll()? {
                            Some((hd, _)) => {
                                r.maps += 1;
                                r.lat.push(now() - hd.time_ms);
                            }
                            None => std::thread::sleep(Duration::from_micros(500)),
                        }
                    }
                    Ok(())
                };
                if let Err(e) = run() {
                    r.error = Some(format!("{e:#}"));
                }
                r.secs = now() / 1e3;
                link.close();
                r
            })
        })
        .collect();
    let results: Vec<ClientResult> = threads.into_iter().map(|t| t.join().expect("client thread")).collect();
    let mut all_lat = Vec::new();
    let mut ok = 0;
    for (c, r) in results.iter().enumerate() {
        let mut l = r.lat.clone();
        l.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| if l.is_empty() { 0.0 } else { l[((l.len() - 1) as f64 * p).round() as usize] };
        // the second half of the run: steady state, after every client's session is up
        let tail: Vec<f64> = r.lat[r.lat.len() / 2..].to_vec();
        let tail_p50 = {
            let mut t = tail.clone();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if t.is_empty() {
                0.0
            } else {
                t[t.len() / 2]
            }
        };
        if r.error.is_none() && r.maps + 1 >= seconds * fps {
            ok += 1;
        }
        all_lat.extend(tail);
        eprintln!(
            "client {c:2}: {} maps, latency p50 {:6.1} p95 {:6.1} max {:6.1} ms, second half p50 {:6.1} ms{}",
            r.maps,
            q(0.5),
            q(0.95),
            q(1.0),
            tail_p50,
            r.error.as_ref().map(|e| format!(" | {e}")).unwrap_or_default()
        );
        let _ = r.secs;
    }
    all_lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| if all_lat.is_empty() { 0.0 } else { all_lat[((all_lat.len() - 1) as f64 * p).round() as usize] };
    println!(
        "{clients} clients x {fps} fps: {ok} complete | steady latency p50 {:.1} p95 {:.1} p99 {:.1} max {:.1} ms",
        q(0.5),
        q(0.95),
        q(0.99),
        q(1.0)
    );
    Ok(())
}
