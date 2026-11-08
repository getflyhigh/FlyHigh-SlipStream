//! The SlipStream host: any machine with a GPU runs it, any client may connect. A client sends instructions (an H.264
//! stream of each frame's network input), the host runs the FlyHigh network on every pair and sends instructions back
//! (the motion maps, `slipstream_core::fields`). Every client gets its own thread and engine session.
//!
//! ```no_run
//! use slipstream_server::{Host, HostOptions};
//! Host::new(HostOptions::new("models").listen("0.0.0.0:9300").token("secret")).run()?;
//! # Ok::<(), anyhow::Error>(())
//! ```
use anyhow::{bail, Context, Result};
use openh264::decoder::Decoder;
use slipstream_core::fields::{pack, Codec};
use slipstream_core::meter::{Avg, Every, Rate};
use slipstream_core::motion::Motion;
use slipstream_core::proto::{Header, Hello, Ready, ServerMsg, KIND_FIELDS, KIND_FRAME, PROTOCOL};
use slipstream_core::rung;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tungstenite::{Message, WebSocket};

#[derive(Clone, Debug)]
pub struct HostOptions {
    /// the FlyHighFG model directory (flyhigh-engine's `models/`)
    pub model: PathBuf,
    /// address to listen on; 127.0.0.1 takes local clients (and SSH tunnels) only
    pub listen: String,
    /// shared secret a client must send; empty = none
    pub token: String,
    /// clients served at once; one more gets "host busy"
    pub max_clients: usize,
    /// a debug line per client per second (instructions per second, sizes, times)
    pub debug: bool,
}
impl HostOptions {
    pub fn new(model: impl Into<PathBuf>) -> HostOptions {
        HostOptions {
            model: model.into(),
            listen: "127.0.0.1:9300".into(),
            token: String::new(),
            max_clients: 4,
            debug: true,
        }
    }
    pub fn listen(mut self, addr: impl Into<String>) -> Self {
        self.listen = addr.into();
        self
    }
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = token.into();
        self
    }
    pub fn max_clients(mut self, n: usize) -> Self {
        self.max_clients = n;
        self
    }
    pub fn debug(mut self, on: bool) -> Self {
        self.debug = on;
        self
    }
}

pub struct Host {
    opts: HostOptions,
    clients: AtomicUsize,
}

impl Host {
    pub fn new(opts: HostOptions) -> Arc<Host> {
        Arc::new(Host { opts, clients: AtomicUsize::new(0) })
    }

    pub fn options(&self) -> &HostOptions {
        &self.opts
    }

    /// Clients being served now.
    pub fn clients(&self) -> usize {
        self.clients.load(Ordering::SeqCst)
    }

    /// Listens and serves until the process ends (one thread per client).
    pub fn run(self: &Arc<Self>) -> Result<()> {
        let o = &self.opts;
        let listener = TcpListener::bind(&o.listen).with_context(|| format!("listen on {}", o.listen))?;
        eprintln!(
            "slipstream host on {}, model {}, up to {} clients{}",
            o.listen,
            o.model.display(),
            o.max_clients,
            if o.token.is_empty() { ", no token" } else { ", token required" }
        );
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let host = self.clone();
            std::thread::spawn(move || host.handle(stream));
        }
        Ok(())
    }

    /// Serves one connection to its end (what `run` does on each client's thread).
    pub fn handle(&self, stream: TcpStream) {
        stream.set_nodelay(true).ok();
        let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
        let n = self.clients.fetch_add(1, Ordering::SeqCst);
        let r = if n >= self.opts.max_clients {
            refuse(stream, &format!("host busy: {} clients connected", self.opts.max_clients))
        } else {
            self.serve(stream, &peer)
        };
        self.clients.fetch_sub(1, Ordering::SeqCst);
        match r {
            Ok(s) => eprintln!("{peer}: {s}"),
            Err(e) => eprintln!("{peer}: {e:#}"),
        }
    }

    fn serve(&self, stream: TcpStream, peer: &str) -> Result<String> {
        let mut ws: WebSocket<TcpStream> =
            tungstenite::accept(stream).map_err(|e| anyhow::anyhow!("handshake: {e}"))?;
        let hello: Hello = match ws.read()? {
            Message::Text(t) => serde_json::from_str(&t)?,
            m => bail!("expected hello, got {m:?}"),
        };
        let deny = |ws: WebSocket<TcpStream>, why: String| -> Result<String> {
            let mut ws = ws;
            send_text(&mut ws, &ServerMsg::Error { message: why.clone() })?;
            goodbye(ws);
            Ok(format!("refused ({why})"))
        };
        if !self.opts.token.is_empty() && hello.token != self.opts.token {
            return deny(ws, "wrong token".into());
        }
        if hello.protocol != PROTOCOL {
            return deny(ws, format!("protocol {} (this host speaks {PROTOCOL})", hello.protocol));
        }
        eprintln!("{peer}: {}x{} k{} {}", hello.w, hello.h, hello.k, hello.codec);
        let setup = (|| -> Result<(Motion, Codec)> {
            Ok((Motion::new(&self.opts.model, hello.w, hello.h, hello.k, 0)?, Codec::parse(&hello.codec)?))
        })();
        let (m, codec) = match setup {
            Ok(v) => v,
            Err(e) => return deny(ws, format!("{e:#}")),
        };
        let lay = m.layout();
        let ready = Ready { gpu: m.gpu_name(), rung: m.rung(), work: m.work(), layout: lay, render: m.render_info() };
        send_text(&mut ws, &ServerMsg::Ready(ready))?;
        let (rw, rh) = m.rung();
        let mut decoder = Decoder::new()?;
        let mut rgb = Vec::new();
        let mut prev: Option<Vec<u16>> = None;
        let (mut n, mut busy_ms) = (0u64, 0f64);
        let (mut rin, mut rout, mut t_dec, mut t_net, mut tick) =
            (Rate::default(), Rate::default(), Avg::default(), Avg::default(), Every::second());
        loop {
            if self.opts.debug {
                if let Some(secs) = tick.due() {
                    eprintln!(
                        "[host {peer} {:5.0}s] in {} | out {} | decode {:5.2} ms | network+pack {:5.2} ms",
                        tick.uptime(),
                        rin.take(secs),
                        rout.take(secs),
                        t_dec.take(),
                        t_net.take()
                    );
                }
            }
            let msg = match ws.read() {
                Ok(Message::Binary(b)) => b,
                Ok(Message::Close(_)) => break,
                Ok(_) => continue,
                Err(tungstenite::Error::ConnectionClosed)
                | Err(tungstenite::Error::AlreadyClosed)
                | Err(tungstenite::Error::Protocol(_)) => break,
                Err(e) => return Err(e.into()),
            };
            let t0 = Instant::now();
            let (h, payload) = Header::read(&msg)?;
            if h.kind != KIND_FRAME {
                continue;
            }
            n += 1;
            rin.add(payload.len());
            let yuv = decoder.decode(payload)?.context("H.264: the access unit gave no picture")?;
            let (w, hh) = openh264::formats::YUVSource::dimensions(&yuv);
            if (w as u32, hh as u32) != (rw, rh) {
                bail!("H.264 picture {w}x{hh}, the network input is {rw}x{rh}");
            }
            rgb.resize(w * hh * 3, 0);
            yuv.write_rgb8(&mut rgb);
            let cur = rung::rgb8_to_f16(&rgb, 3, rw, rh);
            t_dec.add(t0.elapsed().as_secs_f64() * 1e3);
            if let Some(p) = &prev {
                let t1 = Instant::now();
                m.pair_inputs(p, &cur);
                let packed = pack(codec, &lay, &m.read_fields()?)?;
                t_net.add(t1.elapsed().as_secs_f64() * 1e3);
                rout.add(packed.len());
                let server_ms = t0.elapsed().as_secs_f32() * 1e3;
                let hd = Header { kind: KIND_FIELDS, seq: h.seq, time_ms: h.time_ms, server_ms, phase: 0.5, flags: 0 };
                ws.send(Message::binary(hd.write(&packed)))?;
            }
            prev = Some(cur);
            busy_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
        Ok(format!("done, {n} frames, {:.1} ms per frame on the host", busy_ms / n.max(1) as f64))
    }
}

fn send_text(ws: &mut WebSocket<TcpStream>, m: &ServerMsg) -> Result<()> {
    ws.send(Message::text(serde_json::to_string(m)?))?;
    Ok(())
}

fn refuse(stream: TcpStream, why: &str) -> Result<String> {
    let mut ws = tungstenite::accept(stream).map_err(|e| anyhow::anyhow!("handshake: {e}"))?;
    ws.read().ok();
    send_text(&mut ws, &ServerMsg::Error { message: why.into() })?;
    goodbye(ws);
    Ok(format!("refused ({why})"))
}

// A socket dropped with the client's data unread is reset, and the client loses the error sent just before: read
// until the client has seen the close.
fn goodbye(mut ws: WebSocket<TcpStream>) {
    ws.close(None).ok();
    ws.get_mut().set_read_timeout(Some(Duration::from_secs(2))).ok();
    while ws.read().is_ok() {}
}
