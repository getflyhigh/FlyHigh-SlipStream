//! The motion maps the render reads (`render_tex.wgsl`): flow to A and B (4 channels), blend weights (3) and the
//! residual (3), all at a quarter of the network size, f16. `Layout` says where they sit in the engine's buffers;
//! both ends build the same session, so the byte ranges match and the client writes them back in place.
//!
//! Codecs: `f16z` = the f16 bytes, zstd (exact). `q<F>.<O>` (e.g. `q10.8`) = flow quantised to F bits and the other
//! maps to O bits over each plane's own range, then a left-neighbour delta per row and zstd (the maps are smooth, the
//! deltas small). `j<F>.<O>` (e.g. `j90.80`) = every plane mapped to 8 bits over its own range, the flow planes stacked
//! into one grayscale JPEG of quality F and the others into one of quality O (smooth maps cost little in a DCT).
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    /// (byte offset, byte length) in the flow, blend-weight and residual buffers
    pub ranges: [(u64, u64); 3],
    pub hq: u32,
    pub wq: u32,
}
impl Layout {
    pub fn plane_bytes(&self) -> usize {
        (self.hq * self.wq * 2) as usize
    }
    pub fn raw_bytes(&self) -> usize {
        self.ranges.iter().map(|r| r.1 as usize).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Codec {
    F16z,
    Quant { flow_bits: u32, other_bits: u32 },
    Jpeg { flow_q: u8, other_q: u8 },
}
impl Codec {
    pub fn parse(s: &str) -> Result<Codec> {
        if s == "f16z" {
            return Ok(Codec::F16z);
        }
        if let Some(j) = s.strip_prefix('j') {
            let (f, o) = j.split_once('.').context("codec: j<F>.<O>")?;
            let (flow_q, other_q) = (f.parse()?, o.parse()?);
            ensure!((1..=100).contains(&flow_q) && (1..=100).contains(&other_q), "codec: JPEG quality 1..100");
            return Ok(Codec::Jpeg { flow_q, other_q });
        }
        let q = s.strip_prefix('q').context("codec: f16z, q<F>.<O> or j<F>.<O>")?;
        let (f, o) = q.split_once('.').context("codec: q<F>.<O>")?;
        let (flow_bits, other_bits) = (f.parse()?, o.parse()?);
        ensure!((2..=15).contains(&flow_bits) && (2..=15).contains(&other_bits), "codec bits 2..15");
        Ok(Codec::Quant { flow_bits, other_bits })
    }
}

const ZLEVEL: i32 = 3;

fn zigzag(d: i32) -> u32 {
    ((d << 1) ^ (d >> 31)) as u32
}
fn unzigzag(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

fn f16s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>().0.iter().map(|x| half::f16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32()).collect()
}

fn pack_jpeg(lay: &Layout, parts: &[Vec<u8>; 3], flow_q: u8, other_q: u8) -> Result<Vec<u8>> {
    let mut head: Vec<u8> = Vec::new();
    let mut imgs: [Vec<u8>; 2] = [Vec::new(), Vec::new()];
    for (i, part) in parts.iter().enumerate() {
        for plane in part.chunks_exact(lay.plane_bytes()) {
            let v = f16s(plane);
            let lo = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            head.extend_from_slice(&lo.to_le_bytes());
            head.extend_from_slice(&hi.to_le_bytes());
            let span = (hi - lo).max(1e-12);
            imgs[(i > 0) as usize].extend(v.iter().map(|&x| ((x - lo) / span * 255.0).round() as u8));
        }
    }
    let mut out = (head.len() as u32).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    for (img, q) in imgs.iter().zip([flow_q, other_q]) {
        let mut j = Vec::new();
        let rows = (img.len() / lay.wq as usize) as u32;
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut j, q).encode(
            img,
            lay.wq,
            rows,
            image::ExtendedColorType::L8,
        )?;
        out.extend_from_slice(&(j.len() as u32).to_le_bytes());
        out.extend_from_slice(&j);
    }
    Ok(out)
}

fn unpack_jpeg(lay: &Layout, data: &[u8]) -> Result<Vec<u8>> {
    let hl = u32::from_le_bytes(data[..4].try_into()?) as usize;
    let head = &data[4..4 + hl];
    let mut at = 4 + hl;
    let mut px: Vec<u8> = Vec::new();
    for _ in 0..2 {
        let n = u32::from_le_bytes(data[at..at + 4].try_into()?) as usize;
        px.extend(image::load_from_memory(&data[at + 4..at + 4 + n])?.to_luma8().into_raw());
        at += 4 + n;
    }
    let plane = (lay.hq * lay.wq) as usize;
    ensure!(px.len() == lay.raw_bytes() / 2, "fields: {} map pixels, layout {}", px.len(), lay.raw_bytes() / 2);
    let mut all = Vec::with_capacity(lay.raw_bytes());
    for (p, chunk) in px.chunks_exact(plane).enumerate() {
        let lo = f32::from_le_bytes(head[8 * p..8 * p + 4].try_into()?);
        let hi = f32::from_le_bytes(head[8 * p + 4..8 * p + 8].try_into()?);
        let span = (hi - lo).max(1e-12);
        for &q in chunk {
            all.extend_from_slice(&half::f16::from_f32(lo + q as f32 / 255.0 * span).to_bits().to_le_bytes());
        }
    }
    Ok(all)
}

pub fn pack(codec: Codec, lay: &Layout, parts: &[Vec<u8>; 3]) -> Result<Vec<u8>> {
    let (flow_bits, other_bits) = match codec {
        Codec::F16z => return Ok(zstd::encode_all(&parts.concat()[..], ZLEVEL)?),
        Codec::Jpeg { flow_q, other_q } => return pack_jpeg(lay, parts, flow_q, other_q),
        Codec::Quant { flow_bits, other_bits } => (flow_bits, other_bits),
    };
    let mut head: Vec<u8> = Vec::new();
    let mut body: Vec<u8> = Vec::new();
    let wq = lay.wq as usize;
    for (i, part) in parts.iter().enumerate() {
        let bits = if i == 0 { flow_bits } else { other_bits };
        let levels = ((1u32 << bits) - 1) as f32;
        for plane in part.chunks_exact(lay.plane_bytes()) {
            let v = f16s(plane);
            let lo = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            head.extend_from_slice(&lo.to_le_bytes());
            head.extend_from_slice(&hi.to_le_bytes());
            let span = (hi - lo).max(1e-12);
            let qv: Vec<i32> = v.iter().map(|&x| ((x - lo) / span * levels).round() as i32).collect();
            for (j, &q) in qv.iter().enumerate() {
                let z = zigzag(if j % wq == 0 { q } else { q - qv[j - 1] });
                if bits <= 7 {
                    body.push(z as u8);
                } else {
                    body.extend_from_slice(&(z as u16).to_le_bytes());
                }
            }
        }
    }
    let mut out = (head.len() as u32).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    out.extend_from_slice(&zstd::encode_all(&body[..], ZLEVEL)?);
    Ok(out)
}

pub fn unpack(codec: Codec, lay: &Layout, data: &[u8]) -> Result<[Vec<u8>; 3]> {
    let lens = lay.ranges.map(|r| r.1 as usize);
    let split = |all: Vec<u8>| -> Result<[Vec<u8>; 3]> {
        ensure!(all.len() == lay.raw_bytes(), "fields: {} bytes, layout {}", all.len(), lay.raw_bytes());
        let (a, rest) = all.split_at(lens[0]);
        let (b, c) = rest.split_at(lens[1]);
        Ok([a.to_vec(), b.to_vec(), c.to_vec()])
    };
    let (flow_bits, other_bits) = match codec {
        Codec::F16z => return split(zstd::decode_all(data)?),
        Codec::Jpeg { .. } => return split(unpack_jpeg(lay, data)?),
        Codec::Quant { flow_bits, other_bits } => (flow_bits, other_bits),
    };
    ensure!(data.len() >= 4, "fields: short");
    let hl = u32::from_le_bytes(data[..4].try_into().unwrap()) as usize;
    let head = &data[4..4 + hl];
    let body = zstd::decode_all(&data[4 + hl..])?;
    let plane = (lay.hq * lay.wq) as usize;
    let wq = lay.wq as usize;
    let (mut hp, mut bp) = (0usize, 0usize);
    let mut all = Vec::with_capacity(lay.raw_bytes());
    for (i, &len) in lens.iter().enumerate() {
        let bits = if i == 0 { flow_bits } else { other_bits };
        let levels = ((1u32 << bits) - 1) as f32;
        for _ in 0..len / lay.plane_bytes() {
            let lo = f32::from_le_bytes(head[hp..hp + 4].try_into().unwrap());
            let hi = f32::from_le_bytes(head[hp + 4..hp + 8].try_into().unwrap());
            hp += 8;
            let span = (hi - lo).max(1e-12);
            let mut prev = 0i32;
            for j in 0..plane {
                let z = if bits <= 7 {
                    bp += 1;
                    body[bp - 1] as u32
                } else {
                    bp += 2;
                    u16::from_le_bytes([body[bp - 2], body[bp - 1]]) as u32
                };
                let d = unzigzag(z);
                let q = if j % wq == 0 { d } else { prev + d };
                prev = q;
                let x = lo + q as f32 / levels * span;
                all.extend_from_slice(&half::f16::from_f32(x).to_bits().to_le_bytes());
            }
        }
    }
    if bp != body.len() {
        bail!("fields: {} of {} body bytes used", bp, body.len());
    }
    split(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quant_roundtrip_is_close_and_f16z_exact() {
        let lay = Layout { ranges: [(0, 4 * 2 * 6 * 8), (0, 3 * 2 * 6 * 8), (0, 3 * 2 * 6 * 8)], hq: 6, wq: 8 };
        let mk = |n: usize, s: f32| -> Vec<u8> {
            (0..n / 2).flat_map(|i| half::f16::from_f32((i as f32 * 0.37).sin() * s).to_bits().to_le_bytes()).collect()
        };
        let parts =
            [mk(lay.ranges[0].1 as usize, 30.0), mk(lay.ranges[1].1 as usize, 2.0), mk(lay.ranges[2].1 as usize, 0.1)];
        let exact = unpack(Codec::F16z, &lay, &pack(Codec::F16z, &lay, &parts).unwrap()).unwrap();
        assert_eq!(exact, parts);
        let c = Codec::parse("q10.8").unwrap();
        let back = unpack(c, &lay, &pack(c, &lay, &parts).unwrap()).unwrap();
        let f = |b: &[u8]| -> Vec<f32> {
            b.as_chunks::<2>()
                .0
                .iter()
                .map(|x| half::f16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
                .collect()
        };
        for (p, q) in parts.iter().zip(back.iter()) {
            let e = f(p).iter().zip(f(q).iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(e < 0.06, "max error {e}");
        }
    }
}
