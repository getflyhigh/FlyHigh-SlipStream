//! The network's input for one source frame, made on the client exactly as the engine's fast render makes it on the
//! GPU (shaders/tex_down.wgsl): the antialiased triangle downscale of the 32-aligned, edge-replicated frame. The client
//! uploads only this (a 1080p frame becomes 864 x 480), so the full-resolution frames never leave the machine.

/// `[3][oh][ow]` BGR planes in 0..1 from an RGBA8 frame `w` x `h`; `work` = the frame's 32-aligned size.
pub fn downscale(rgba: &[u8], w: u32, h: u32, work: (u32, u32), out: (u32, u32)) -> Vec<f32> {
    let (iw, ih) = work;
    let (ow, oh) = out;
    let sh = ih as f32 / oh as f32;
    let sw = iw as f32 / ow as f32;
    let (fsh, fsw) = (sh.max(1.0), sw.max(1.0));
    let plane = (ow * oh) as usize;
    let mut v = vec![0f32; 3 * plane];
    let taps = |o: u32, s: f32, fs: f32, lim: u32| -> Vec<(i32, f32)> {
        let c = (o as f32 + 0.5) * s;
        let lo = ((c - fs + 0.5).floor() as i32).max(0);
        let hi = ((c + fs + 0.5).floor() as i32).min(lim as i32);
        (lo..hi).map(|i| (i, (1.0 - ((i as f32 + 0.5 - c) / fs).abs()).max(0.0))).collect()
    };
    let xs: Vec<Vec<(i32, f32)>> = (0..ow).map(|x| taps(x, sw, fsw, iw)).collect();
    for oy in 0..oh {
        let ys = taps(oy, sh, fsh, ih);
        for ox in 0..ow {
            let mut acc = [0f32; 3];
            let mut norm = 0f32;
            for &(y, wy) in &ys {
                let row = (y.min(h as i32 - 1) as usize) * w as usize;
                for &(x, wx) in &xs[ox as usize] {
                    let wgt = wx * wy;
                    let p = (row + x.min(w as i32 - 1) as usize) * 4;
                    acc[0] += rgba[p] as f32 / 255.0 * wgt;
                    acc[1] += rgba[p + 1] as f32 / 255.0 * wgt;
                    acc[2] += rgba[p + 2] as f32 / 255.0 * wgt;
                    norm += wgt;
                }
            }
            let o = (oy * ow + ox) as usize;
            v[o] = acc[2] / norm;
            v[plane + o] = acc[1] / norm;
            v[2 * plane + o] = acc[0] / norm;
        }
    }
    v
}

/// The planes as an interleaved RGB8 image (for JPEG / PNG upload).
pub fn to_rgb8(planes: &[f32], ow: u32, oh: u32) -> Vec<u8> {
    let plane = (ow * oh) as usize;
    let q = |x: f32| (x * 255.0).round().clamp(0.0, 255.0) as u8;
    (0..plane).flat_map(|o| [q(planes[2 * plane + o]), q(planes[plane + o]), q(planes[o])]).collect()
}

/// RGB8 (`channels` = 3) or RGBA8 (4) pixels back to the network's f16 BGR planes.
pub fn rgb8_to_f16(px: &[u8], channels: usize, ow: u32, oh: u32) -> Vec<u16> {
    let plane = (ow * oh) as usize;
    let mut v = vec![0u16; 3 * plane];
    for o in 0..plane {
        let p = &px[o * channels..];
        for (c, src) in [2usize, 1, 0].into_iter().enumerate() {
            v[c * plane + o] = half::f16::from_f32(p[src] as f32 / 255.0).to_bits();
        }
    }
    v
}

pub fn f32_to_f16(planes: &[f32]) -> Vec<u16> {
    planes.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect()
}
