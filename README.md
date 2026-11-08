<div align="center">

![FlyHigh SlipStream](assets/banner.svg)

**Frame generation split between a client and a GPU host.<br>Only instructions cross the wire, never a frame.**

[![Status: experimental](https://img.shields.io/badge/status-experimental-f59e0b?style=flat-square)](docs/RESEARCH.md)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-3b82f6?style=flat-square)](LICENSE)
[![Weights: CC BY-NC 4.0](https://img.shields.io/badge/weights-CC%20BY--NC%204.0-7c3aed?style=flat-square)](https://github.com/getflyhigh/flyhigh-engine/blob/main/models/WEIGHTS_LICENSE.md)
[![Built on FlyHigh](https://img.shields.io/badge/built%20on-FlyHigh%20engine-19c37d?style=flat-square)](https://github.com/getflyhigh/flyhigh-engine)

[**Quick start**](#quick-start) / [Host](#host) / [Client](#client) / [How it works](#how-it-works) /
[Research notes](docs/RESEARCH.md) / [Security](#security)

</div>

---

> **Experimental.** No releases yet and no stability promise: the API, the protocol and the codecs change as the
> measurements say.

<div align="center">

![The client sends the host an H.264 stream of each frame's network input; the host sends back the motion maps; the client renders every in-between](assets/flow.svg)

<sub>Grey: the source frames. Blue: generated on the client from the host's maps. No frame crosses the wire.</sub>

</div>

SlipStream is two Rust libraries on top of [flyhigh-engine](https://github.com/getflyhigh/flyhigh-engine):

- **`slipstream-server`**: the host. Any machine with a GPU runs it; any number of clients connect and get the
  FlyHigh network run for them.
- **`slipstream-client`**: the client. It sends the host a few KB of instructions per frame and renders the
  in-between frames from the few KB the host sends back. The network never runs on the client.

## Quick start

```
git clone https://github.com/getflyhigh/FlyHigh-SlipStream && cd FlyHigh-SlipStream
git clone --depth 1 --branch v0.3.1 https://github.com/getflyhigh/flyhigh-engine   # the model: flyhigh-engine/models
cargo run --release -p slipstream-server -- --model flyhigh-engine/models          # on the GPU machine
cargo run --release --example basic -- flyhigh-engine/models <frames dir> <out dir> ws://<host>:9300
```

`basic` doubles the frame rate of a directory of PNG / JPEG frames through the host.

## Host

```rust
use slipstream_server::{Host, HostOptions};

Host::new(HostOptions::new("flyhigh-engine/models").listen("0.0.0.0:9300").token("secret").max_clients(8)).run()?;
```

Or the binary: `slipstream-server --model <dir> [--listen 127.0.0.1:9300] [--token <secret>] [--max-clients 4]
[--quiet]`. Each client gets its own thread and engine session; a wrong token or a full host gets an error message.
One RTX 5090 serves about 30 clients at 24 -> 120 fps.

## Client

```rust
use slipstream_client::{Client, Options};

let mut c = Client::connect("ws://host:9300", Options::new("flyhigh-engine/models", 1920, 1080).fps(60.0))?;
for frame in frames {                       // RGBA8, width * height * 4 bytes, in display order
    let id = c.push(&frame)?;               // its instructions go to the host
    for ready in c.poll()? {                // pairs whose maps came back: (ready - 1, ready)
        let mid = c.render(ready, 0.5)?;    // the in-between at t = 0.5, RGBA8; any t in 0..1, as many as you like
    }
}
```

- `render_gpu(id, t)` leaves the frame on the GPU (`motion().output_texture()`), for hosts that present it
  themselves; `motion()` exposes the wgpu device and queue.
- `wait(timeout)` is `poll` that blocks until a pair is ready.
- `stats` holds traffic and latency; `reference(id, t)` runs the whole engine locally on the same pair, for
  comparisons.
- Options: `fps`, `kbps` (instruction stream rate, default 50 kbit per frame), `token`, `ring` (recent frames kept on
  the GPU, default 8), `k` (network iterations on the host, 1..3), `codec` (how the maps are packed).

Maps come back one round trip after a frame is pushed. A player either shows its frames that much late or, when it
has frames before it shows them (a file, a stream buffer), pushes them early: then nothing is late at all.

Lower level, for other transports or clients in other languages: `Instructions` (the H.264 encoder), `Link` (the
WebSocket), `slipstream_core::proto` (the wire format), `slipstream_core::fields` (the map codecs) and
`slipstream_core::motion::Motion` (the engine session both ends use).

### Examples

| | |
|---|---|
| `basic` | a frame directory at twice the frame rate |
| `bench` | real-time play of a high-fps sequence: debug line every second, quality against the local engine and the real in-betweens |
| `load` | N clients at once against one host: what it carries |

```
[client   5.0s] up  60.0 ips   6.0 KB/ipf  2.97 Mbit/s | down  60.0 ips  12.8 KB/ipf  6.31 Mbit/s | latency  279 ms | host  8.1 ms
```

ips = instructions per second, ipf = KB per instruction (one frame's).

## How it works

- The client makes each frame's network input on its GPU with the engine's own downscale (864 x 480 for 1080p),
  converts it to I420 and encodes it as one H.264 access unit. A video bitstream is a list of instructions to a
  decoder (motion vectors and corrections), so most frames take about 6 KB.
- The host decodes it, runs the network on each pair and sends back the maps the render reads: flow to both frames,
  blend weights and residual at a quarter of the network size, packed into about 13 KB.
- The client renders each in-between at full resolution from its own frames and those maps. The network runs once
  per pair and every phase is a render, so 2x, 5x and 8x cost the same traffic.

Measured from a home PC (RTX 4060 Ti) to an RTX 5090 about 250 ms away, CS2 footage at 1080p:

| | 60 fps source | 24 fps source |
|---|---:|---:|
| client to host | 3.0 Mbit/s | 1.4 Mbit/s |
| host to client | 6.3 Mbit/s | 2.7 Mbit/s |
| latency (the round trip) | 279 ms | 282 ms |
| picture against the whole engine run locally | -0.05 dB | equal |
| client time per frame | 4.6 ms | 5.8 ms |

Sending finished frames instead would take about 50 times the traffic as JPEG, and by estimate 5 to 8 times as an
H.264 stream. All runs, the codec comparisons, host capacity and a browser client built on the same protocol:
[docs/RESEARCH.md](docs/RESEARCH.md).

## Requirements

- Stable Rust (builds with 1.99).
- A GPU with `shader-f16` on both ends (Vulkan, DX12 or Metal through wgpu).
- The FlyHighFG model: the `models/` directory of flyhigh-engine v0.3.1. Both ends must use the same model.
- Frames more than 960 px wide (below that the engine runs the network at the frame size, which this split does not
  cover).

## Security

The host decodes H.264 from whoever connects and spends its GPU on them. The WebSocket is plain `ws://`, so the
token travels in clear text. Do not open a host to the internet as it is: keep it on `127.0.0.1` behind an SSH
tunnel, or put it behind a TLS proxy (`wss://`) with a token, and set `max_clients`.

## Layout

- `crates/core`: what both ends share: `proto` (the wire format), `fields` (map layout and codecs), `motion` (the
  engine session), `rung` (a CPU copy of the engine's downscale), `meter` (the debug lines).
- `crates/server`: `Host`, `HostOptions` and the `slipstream-server` binary.
- `crates/client`: `Client`, `Options`, `Instructions`, `Link`, and `examples/`.

## License

Code: Apache-2.0. The model weights are not part of this repository; they ship with flyhigh-engine under their own
license (CC-BY-NC-4.0).
