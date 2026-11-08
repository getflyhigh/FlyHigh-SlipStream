enable f16;
// The network input (f16 B, G, R planes, w x h) as an I420 picture for the H.264 encoder, BT.601 limited range with
// the coefficients of openh264 (formats/rgb2yuv.rs), so the host's decoder inverts what the client's GPU made. One
// thread per 8 x 2 block: two rows of 8 luma bytes and 4 + 4 chroma bytes are whole u32 words, no thread shares one.
struct P { w: u32, h: u32 };
@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> p: P;

fn rgb(x: u32, y: u32) -> vec3<f32> {
    let plane = p.w * p.h; let o = y * p.w + x;
    return vec3<f32>(f32(src[2u * plane + o]), f32(src[plane + o]), f32(src[o])) * 255.0;
}
fn luma(c: vec3<f32>) -> u32 {
    return u32(clamp(round(16.0 + 0.2578125 * c.r + 0.50390625 * c.g + 0.09765625 * c.b), 0.0, 255.0));
}
fn pack4(a: u32, b: u32, c: u32, d: u32) -> u32 { return a | (b << 8u) | (c << 16u) | (d << 24u); }

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x0 = gid.x * 8u; let y0 = gid.y * 2u;
    if (x0 >= p.w || y0 >= p.h) { return; }
    for (var r = 0u; r < 2u; r++) {
        var ys: array<u32, 8>;
        for (var i = 0u; i < 8u; i++) { ys[i] = luma(rgb(x0 + i, y0 + r)); }
        let base = ((y0 + r) * p.w + x0) / 4u;
        dst[base] = pack4(ys[0], ys[1], ys[2], ys[3]);
        dst[base + 1u] = pack4(ys[4], ys[5], ys[6], ys[7]);
    }
    var u: array<u32, 4>;
    var v: array<u32, 4>;
    for (var i = 0u; i < 4u; i++) {
        let xa = x0 + 2u * i;
        let c = (rgb(xa, y0) + rgb(xa + 1u, y0) + rgb(xa, y0 + 1u) + rgb(xa + 1u, y0 + 1u)) * 0.25;
        u[i] = u32(clamp(round(128.0 - 0.1484375 * c.r - 0.2890625 * c.g + 0.4375 * c.b), 0.0, 255.0));
        v[i] = u32(clamp(round(128.0 + 0.4375 * c.r - 0.3671875 * c.g - 0.0703125 * c.b), 0.0, 255.0));
    }
    let ysize = p.w * p.h; let csize = ysize / 4u;
    let co = (y0 / 2u) * (p.w / 2u) + x0 / 2u;
    dst[(ysize + co) / 4u] = pack4(u[0], u[1], u[2], u[3]);
    dst[(ysize + csize + co) / 4u] = pack4(v[0], v[1], v[2], v[3]);
}
