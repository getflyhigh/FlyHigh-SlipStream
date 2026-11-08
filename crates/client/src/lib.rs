//! The SlipStream client: sends a host instructions for each frame and renders in-betweens from the instructions it
//! gets back. Only the render runs here; the network runs on the host.
//!
//! ```no_run
//! use slipstream_client::{Client, Options};
//! # fn frames() -> Vec<Vec<u8>> { vec![] }
//! let mut c = Client::connect("ws://127.0.0.1:9300", Options::new("models", 1920, 1080).fps(60.0))?;
//! for rgba in frames() {
//!     let id = c.push(&rgba)?;                // this frame's instructions go to the host
//!     for ready in c.poll()? {                 // maps that came back: pairs (ready - 1, ready)
//!         let mid = c.render(ready, 0.5)?;     // the in-between, RGBA8
//!     }
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! Frames are RGBA8 (`width * height * 4` bytes) in display order. Maps come back one round trip after a frame is
//! pushed; a player shows frames that much late, or pushes frames ahead of playback when it has them (a video file).
//!
//! Lower level: `Instructions` (the H.264 encoder of the network input), `Link` (the WebSocket),
//! `slipstream_core::motion::Motion` (the engine session).
use anyhow::{bail, ensure, Context, Result};
use openh264::encoder::{
    BitRate, Complexity, Encoder, EncoderConfig, FrameRate, IntraFramePeriod, RateControlMode, UsageType,
};
use openh264::formats::YUVBuffer;
use openh264::OpenH264API;
use slipstream_core::fields::{unpack, Codec, Layout};
use slipstream_core::proto::{Header, Hello, Ready, ServerMsg, KIND_FIELDS, KIND_FRAME, PROTOCOL};
use std::collections::{HashMap, VecDeque};
use std::io::ErrorKind;
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tungstenite::{Message, WebSocket};

pub use slipstream_core::motion::Motion;

#[derive(Clone, Debug)]
pub struct Options {
    /// the FlyHighFG model directory (flyhigh-engine's `models/`); the client runs only its render
    pub model: PathBuf,
    /// frame size, more than 960 px wide
    pub width: u32,
    pub height: u32,
    /// source frame rate (the instruction encoder's rate control)
    pub fps: f32,
    /// instruction stream rate in kbit/s; 0 = 50 kbit per frame (3000 at 60 fps)
    pub kbps: u32,
    /// the host's shared secret
    pub token: String,
    /// how the host packs the maps (`slipstream_core::fields::Codec`)
    pub codec: String,
    /// recent frames kept on the GPU: a pair renders while both its frames are among the last `ring` pushed
    pub ring: usize,
    /// fine-flow iterations on the host, 1..3 (more costs the host, not the client)
    pub k: u32,
}
impl Options {
    pub fn new(model: impl Into<PathBuf>, width: u32, height: u32) -> Options {
        Options {
            model: model.into(),
            width,
            height,
            fps: 30.0,
            kbps: 0,
            token: String::new(),
            codec: "j50.40".into(),
            ring: 8,
            k: 1,
        }
    }
    pub fn fps(mut self, fps: f32) -> Self {
        self.fps = fps;
        self
    }
    pub fn kbps(mut self, kbps: u32) -> Self {
        self.kbps = kbps;
        self
    }
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = token.into();
        self
    }
    pub fn codec(mut self, codec: impl Into<String>) -> Self {
        self.codec = codec.into();
        self
    }
    pub fn ring(mut self, frames: usize) -> Self {
        self.ring = frames.max(2);
        self
    }
    pub fn k(mut self, k: u32) -> Self {
        self.k = k;
        self
    }
}

/// Traffic and timing so far.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub frames: u64,
    pub maps: u64,
    pub bytes_up: u64,
    pub bytes_down: u64,
    /// push to maps back, the last pair
    pub latency_ms: f64,
    /// the host's time for the last pair
    pub host_ms: f64,
    /// push to maps back, the last 1000 pairs
    pub recent_latency_ms: VecDeque<f64>,
}

pub struct Client {
    m: Motion,
    enc: Instructions,
    link: Link,
    codec: Codec,
    lay: Layout,
    ring: usize,
    next: u64,
    maps: HashMap<u64, [Vec<u8>; 3]>,
    order: VecDeque<u64>,
    loaded: Option<u64>,
    sent: HashMap<u64, Instant>,
    host: String,
    pub stats: Stats,
}

impl Client {
    /// `url` = ws://host:port. Builds the local render session, then asks the host for one of the same shape.
    pub fn connect(url: &str, opts: Options) -> Result<Client> {
        let m = Motion::new(&opts.model, opts.width, opts.height, opts.k, opts.ring.max(2))?;
        let hello = Hello {
            protocol: PROTOCOL,
            w: opts.width,
            h: opts.height,
            k: opts.k,
            codec: opts.codec.clone(),
            token: opts.token.clone(),
        };
        let link = Link::connect(url, &hello)?;
        let lay = m.layout();
        ensure!(
            link.ready.layout == lay && link.ready.rung == m.rung(),
            "the host's session differs from this client's (model or engine version): {:?} against {:?}",
            link.ready.layout,
            lay
        );
        let kbps = if opts.kbps > 0 { opts.kbps } else { (opts.fps * 50.0).round().clamp(500.0, 8000.0) as u32 };
        let enc = Instructions::new(m.rung(), opts.fps, kbps)?;
        Ok(Client {
            host: link.ready.gpu.clone(),
            codec: Codec::parse(&opts.codec)?,
            lay,
            ring: opts.ring.max(2),
            m,
            enc,
            link,
            next: 0,
            maps: HashMap::new(),
            order: VecDeque::new(),
            loaded: None,
            sent: HashMap::new(),
            stats: Stats::default(),
        })
    }

    /// The host's GPU, as it reports it.
    pub fn host_gpu(&self) -> &str {
        &self.host
    }

    /// The next source frame (RGBA8, display order): kept on the GPU, its instructions sent. Returns its id (0, 1, ...).
    pub fn push(&mut self, rgba: &[u8]) -> Result<u64> {
        let id = self.next;
        let slot = (id % self.ring as u64) as usize;
        self.m.upload(slot, rgba)?;
        let au = self.enc.encode(self.m.rung_i420(slot)?)?;
        self.link.send_frame(id as u32, 0.0, &au)?;
        self.sent.insert(id, Instant::now());
        self.stats.frames += 1;
        self.stats.bytes_up += au.len() as u64;
        self.next += 1;
        Ok(id)
    }

    /// Takes in the maps that have arrived. Returns the ids now ready to render: `id` = the pair (`id` - 1, `id`).
    pub fn poll(&mut self) -> Result<Vec<u64>> {
        let mut ready = Vec::new();
        while let Some((h, payload)) = self.link.poll()? {
            if h.kind != KIND_FIELDS {
                continue;
            }
            let id = self.widen(h.seq);
            self.stats.maps += 1;
            self.stats.bytes_down += payload.len() as u64;
            self.stats.host_ms = h.server_ms as f64;
            if let Some(t) = self.sent.remove(&id) {
                self.stats.latency_ms = t.elapsed().as_secs_f64() * 1e3;
                self.stats.recent_latency_ms.push_back(self.stats.latency_ms);
                if self.stats.recent_latency_ms.len() > 1000 {
                    self.stats.recent_latency_ms.pop_front();
                }
            }
            self.maps.insert(id, unpack(self.codec, &self.lay, &payload)?);
            self.order.push_back(id);
            while self.order.len() > self.ring {
                if let Some(old) = self.order.pop_front() {
                    self.maps.remove(&old);
                }
            }
            ready.push(id);
        }
        self.sent.retain(|&k, _| k + 4 * self.ring as u64 > self.next);
        Ok(ready)
    }

    /// Like `poll`, waiting up to `timeout` for at least one pair.
    pub fn wait(&mut self, timeout: Duration) -> Result<Vec<u64>> {
        let t0 = Instant::now();
        loop {
            let r = self.poll()?;
            if !r.is_empty() || t0.elapsed() >= timeout {
                return Ok(r);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Whether the pair (`id` - 1, `id`) can render: its maps are in and both frames still on the GPU.
    pub fn is_ready(&self, id: u64) -> bool {
        id >= 1 && id < self.next && self.next - (id - 1) <= self.ring as u64 && self.maps.contains_key(&id)
    }

    /// The in-between of frames `id` - 1 and `id` at phase `t` (0 = frame `id` - 1, 1 = frame `id`), RGBA8.
    pub fn render(&mut self, id: u64, t: f32) -> Result<Vec<u8>> {
        self.render_gpu(id, t)?;
        self.m.read_out()
    }

    /// The same, left on the GPU in `motion().output_texture()` (rgba8unorm, frame size).
    pub fn render_gpu(&mut self, id: u64, t: f32) -> Result<()> {
        ensure!(
            self.is_ready(id),
            "pair {id} is not ready (maps not in yet, or its frames left the ring of {})",
            self.ring
        );
        if self.loaded != Some(id) {
            self.m.write_fields(&self.maps[&id])?;
            self.loaded = Some(id);
        }
        let ring = self.ring as u64;
        self.m.render(((id - 1) % ring) as usize, (id % ring) as usize, t.clamp(0.0, 1.0));
        Ok(())
    }

    /// The whole engine run locally on the same pair (no host): what `render` should match. For comparisons.
    pub fn reference(&mut self, id: u64, t: f32) -> Result<Vec<u8>> {
        ensure!(id >= 1 && id < self.next && self.next - (id - 1) <= self.ring as u64, "pair {id} left the ring");
        let ring = self.ring as u64;
        self.m.local(((id - 1) % ring) as usize, (id % ring) as usize, t.clamp(0.0, 1.0))?;
        self.loaded = None;
        self.m.read_out()
    }

    /// The engine session underneath (its device, queue and textures).
    pub fn motion(&self) -> &Motion {
        &self.m
    }

    pub fn close(self) {
        self.link.close();
    }

    // the wire carries ids as u32
    fn widen(&self, seq: u32) -> u64 {
        let hi = self.next & !0xffff_ffff;
        let id = hi | seq as u64;
        if id >= self.next && hi > 0 {
            id - (1 << 32)
        } else {
            id
        }
    }
}

/// The network input of each frame as one H.264 access unit (openh264). The stream is continuous: after the first
/// keyframe most frames are a few KB of motion vectors and corrections.
pub struct Instructions {
    enc: Encoder,
    w: u32,
    h: u32,
}
impl Instructions {
    /// `rung` = the network input size (`Ready::rung`), `fps` = the source frame rate, `kbps` = the stream's rate.
    pub fn new(rung: (u32, u32), fps: f32, kbps: u32) -> Result<Instructions> {
        let cfg = EncoderConfig::new()
            .bitrate(BitRate::from_bps(kbps * 1000))
            .max_frame_rate(FrameRate::from_hz(fps))
            .usage_type(UsageType::CameraVideoRealTime)
            .rate_control_mode(RateControlMode::Bitrate)
            // every frame is needed: a skipped frame would leave the host without that pair
            .skip_frames(false)
            .intra_frame_period(IntraFramePeriod::from_num_frames(0))
            .complexity(Complexity::Low)
            .num_threads(4);
        let enc = Encoder::with_api_config(OpenH264API::from_source(), cfg)?;
        Ok(Instructions { enc, w: rung.0, h: rung.1 })
    }
    /// One frame from its network input as I420 (`Motion::rung_i420`).
    pub fn encode(&mut self, i420: Vec<u8>) -> Result<Vec<u8>> {
        let yuv = YUVBuffer::from_vec(i420, self.w as usize, self.h as usize);
        let out = self.enc.encode(&yuv)?.to_vec();
        if out.is_empty() {
            bail!("H.264 encoder dropped a frame");
        }
        Ok(out)
    }
}

/// The WebSocket to a host, non-blocking on the receiving side so a real-time sender never waits for it.
pub struct Link {
    ws: WebSocket<TcpStream>,
    pub ready: Ready,
}
impl Link {
    /// `url` = ws://host:port
    pub fn connect(url: &str, hello: &Hello) -> Result<Link> {
        let addr = url.strip_prefix("ws://").context("url: ws://host:port")?.trim_end_matches('/');
        let tcp = TcpStream::connect(addr).with_context(|| format!("connect {addr}"))?;
        tcp.set_nodelay(true)?;
        let (mut ws, _) = tungstenite::client(url, tcp).map_err(|e| anyhow::anyhow!("handshake: {e}"))?;
        ws.send(Message::text(serde_json::to_string(hello)?))?;
        let ready = match ws.read()? {
            Message::Text(t) => match serde_json::from_str::<ServerMsg>(&t)? {
                ServerMsg::Ready(r) => r,
                ServerMsg::Error { message } => bail!("host: {message}"),
            },
            m => bail!("expected the host's answer, got {m:?}"),
        };
        ws.get_mut().set_read_timeout(Some(Duration::from_millis(1)))?;
        Ok(Link { ws, ready })
    }
    pub fn send_frame(&mut self, seq: u32, time_ms: f64, payload: &[u8]) -> Result<()> {
        let hd = Header { kind: KIND_FRAME, seq, time_ms, ..Default::default() };
        self.ws.send(Message::binary(hd.write(payload)))?;
        Ok(())
    }
    /// The next binary message from the host, if one has arrived.
    pub fn poll(&mut self) -> Result<Option<(Header, Vec<u8>)>> {
        loop {
            match self.ws.read() {
                Ok(Message::Binary(b)) => {
                    let (h, p) = Header::read(&b)?;
                    return Ok(Some((h, p.to_vec())));
                }
                Ok(Message::Text(t)) => {
                    if let Ok(ServerMsg::Error { message }) = serde_json::from_str::<ServerMsg>(&t) {
                        bail!("host: {message}");
                    }
                }
                Ok(Message::Close(_)) => bail!("host closed the connection"),
                Ok(_) => {}
                Err(tungstenite::Error::Io(e)) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    return Ok(None)
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
    pub fn close(mut self) {
        self.ws.close(None).ok();
        for _ in 0..100 {
            if self.ws.read().is_err() {
                break;
            }
        }
    }
}
