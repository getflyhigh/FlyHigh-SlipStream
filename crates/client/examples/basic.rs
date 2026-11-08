//! The smallest client: doubles the frame rate of an image sequence through a host and writes the result.
//!
//!   cargo run --release --example basic -- <model dir> <frames dir> <out dir> [ws://127.0.0.1:9300]
use anyhow::{Context, Result};
use slipstream_client::{Client, Options};
use slipstream_core::decode_image;
use std::time::Duration;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, frames, out] = [0, 1, 2].map(|i| args.get(i).cloned());
    let (model, frames, out) = (model.context("model dir")?, frames.context("frames dir")?, out.context("out dir")?);
    let url = args.get(3).cloned().unwrap_or_else(|| "ws://127.0.0.1:9300".into());

    let mut files: Vec<_> = std::fs::read_dir(&frames)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    files.sort();
    let (w, h, _) = decode_image(&std::fs::read(&files[0])?)?;
    std::fs::create_dir_all(&out)?;
    let mut c = Client::connect(&url, Options::new(model, w, h))?;
    let mut n = 0;
    let mut save = |rgba: &[u8]| -> Result<()> {
        image::save_buffer(format!("{out}/{n:05}.png"), rgba, w, h, image::ExtendedColorType::Rgba8)?;
        n += 1;
        Ok(())
    };
    for f in &files {
        let rgba = decode_image(&std::fs::read(f)?)?.2;
        let id = c.push(&rgba)?;
        if id > 0 {
            // a file sequence can simply wait for each pair; a player would keep pushing and render what is ready
            while !c.is_ready(id) {
                c.wait(Duration::from_secs(5))?;
            }
            save(&c.render(id, 0.5)?)?;
        }
        save(&rgba)?;
    }
    println!("{} frames from {}, host {}", n, files.len(), c.host_gpu());
    c.close();
    Ok(())
}
