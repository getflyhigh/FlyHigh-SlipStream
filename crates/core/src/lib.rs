//! FlyHigh SlipStream: what the client (`slipstream-client`) and the host (`slipstream-server`) share: the wire format
//! (`proto`), the map codecs (`fields`), the engine session both ends build (`motion`), a CPU copy of the engine's
//! downscale (`rung`) and the debug counters (`meter`).
pub mod fields;
pub mod meter;
pub mod motion;
pub mod proto;
pub mod rung;

/// PSNR over RGB of two RGBA8 frames of the same size.
pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mut se = 0f64;
    let mut n = 0u64;
    for (pa, pb) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0.iter()) {
        for c in 0..3 {
            let d = pa[c] as f64 - pb[c] as f64;
            se += d * d;
        }
        n += 3;
    }
    let mse = se / n.max(1) as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

/// An image file (PNG, JPEG) as RGBA8: width, height, pixels.
pub fn decode_image(bytes: &[u8]) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let img = image::load_from_memory(bytes)?.to_rgba8();
    Ok((img.width(), img.height(), img.into_raw()))
}
