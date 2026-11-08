//! Wire format (WebSocket). Text messages are JSON; binary messages are a 24-byte `Header` and a payload.
//!
//! 1. client -> host, text: `Hello` (frame size, map codec, token).
//! 2. host -> client, text: `ServerMsg::Ready` (what the client needs to render) or `ServerMsg::Error`.
//! 3. client -> host, binary, kind 1, once per source frame: the frame's network input (see `rung`) as one H.264
//!    access unit, Annex B; the stream continues from message to message. `seq` numbers the frames, `time_ms` is the
//!    client's clock (echoed back).
//! 4. host -> client, binary, kind 2, once per frame after the first: the motion maps of the pair (`seq` - 1, `seq`),
//!    packed by `fields`.
use serde::{Deserialize, Serialize};

pub const PROTOCOL: u32 = 1;
pub const KIND_FRAME: u8 = 1;
pub const KIND_FIELDS: u8 = 2;
pub const HEADER: usize = 24;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Hello {
    #[serde(default = "protocol")]
    pub protocol: u32,
    /// the client's frame size (more than 960 px wide: below that the engine runs the network at the frame size)
    pub w: u32,
    pub h: u32,
    /// fine-flow iterations on the host, 1..3
    #[serde(default = "one")]
    pub k: u32,
    /// `fields::Codec` spec of the maps the host sends ("j50.40", "q10.8", "f16z")
    #[serde(default = "default_codec")]
    pub codec: String,
    /// the host's shared secret, when it has one
    #[serde(default)]
    pub token: String,
}
fn protocol() -> u32 {
    PROTOCOL
}
fn one() -> u32 {
    1
}
fn default_codec() -> String {
    "j50.40".into()
}

/// The host's answer to `Hello`.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ServerMsg {
    Ready(Ready),
    Error { message: String },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Ready {
    pub gpu: String,
    /// the network input size the client encodes (`rung`)
    pub rung: (u32, u32),
    /// the frame's 32-aligned working size the network input is cut from
    pub work: (u32, u32),
    pub layout: crate::fields::Layout,
    /// what a client without the engine needs to run the render itself (`render_tex.wgsl`)
    #[serde(default)]
    pub render: RenderInfo,
}

/// The render's constants for this session (`render_tex.wgsl` `Par`): the maps' scales and the mode word. The maps
/// arrive packed from their first plane, so a client keeps them at f0 = v0 = 0.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq)]
pub struct RenderInfo {
    /// flow (network pixels at the quarter grid) to frame pixels
    pub fsx: f32,
    pub fsy: f32,
    /// blend weight scale
    pub vs: f32,
    /// 1 = bicubic source taps (the model's render), 0 = bilinear
    pub cubic: u32,
    /// render mode word: bit 0 decisive blend, bit 1 still test
    pub flags: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Header {
    pub kind: u8,
    pub flags: u8,
    pub seq: u32,
    pub time_ms: f64,
    pub server_ms: f32,
    pub phase: f32,
}
impl Header {
    pub fn write(&self, payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::with_capacity(HEADER + payload.len());
        v.extend_from_slice(&[self.kind, self.flags, 0, 0]);
        v.extend_from_slice(&self.seq.to_le_bytes());
        v.extend_from_slice(&self.time_ms.to_le_bytes());
        v.extend_from_slice(&self.server_ms.to_le_bytes());
        v.extend_from_slice(&self.phase.to_le_bytes());
        v.extend_from_slice(payload);
        v
    }
    pub fn read(m: &[u8]) -> anyhow::Result<(Header, &[u8])> {
        anyhow::ensure!(m.len() >= HEADER, "short message ({} bytes)", m.len());
        let f4 = |o: usize| [m[o], m[o + 1], m[o + 2], m[o + 3]];
        let h = Header {
            kind: m[0],
            flags: m[1],
            seq: u32::from_le_bytes(f4(4)),
            time_ms: f64::from_le_bytes(m[8..16].try_into().unwrap()),
            server_ms: f32::from_le_bytes(f4(16)),
            phase: f32::from_le_bytes(f4(20)),
        };
        Ok((h, &m[HEADER..]))
    }
}
