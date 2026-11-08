# SlipStream research notes

## Question

Can frame generation run on someone else's GPU and still reach the user, at the same quality as the local engine,
over a home connection? If it can, the user's own GPU only has to render and present, and the host can afford more
than a real-time budget allows.

## Design: a client and a host that exchange instructions

There are two ways to split the work:

1. **Frames** (baseline, measured in the first round, no longer in the code). The client uploads full frames. The host generates the in-betweens and
   sends them back. Traffic grows with the multiplier: at 4x, three full frames come back per source frame.
2. **Instructions** (SlipStream). Neither side sends a frame.
   - Client to host: the network input of each frame (the 480-line rung: 864 x 480 for 1080p) as an H.264 stream.
     A video bitstream is itself a list of instructions to a decoder (motion vectors and corrections against the
     previous picture), so most frames take a few KB. The client makes it on its GPU with the engine's own downscale
     (`tex_down.wgsl`) and an RGB-to-I420 pass (`i420.wgsl`, the coefficients of the openh264 decoder the host uses),
     then encodes it.
   - Host to client: the maps the render reads. Flow to both frames (4 channels), blend weights (3) and residual (3),
     216 x 120 each for 1080p, 506 KB raw per pair, packed to about 13 KB (`j50.40`: each plane at 8 bits in a
     grayscale JPEG).
   - The client renders every in-between at full resolution from its own frames and those maps. With the engine's
     linear t, the t-dependent part runs once per pair at t = 0.5 and every phase is a render that scales the maps,
     so one set of maps serves 2x, 5x and 8x alike.

What makes the split clean:

- The network never sees full resolution: on the fast render path it reads the antialiased 480-line downscale, so
  sending exactly that loses nothing the network uses.
- The render reads only the three maps and the two source frames (`render_tex.wgsl`). Both ends build the same
  engine session, so the maps keep the same byte layout. The client writes them into its own session's buffers and
  runs only the render step. Everything goes through the public `Session` / `SessionExt` API of flyhigh-engine
  v0.3.1; the engine itself is unchanged.

The host takes many clients: one thread and one engine session per client, an optional shared token, and a client
limit. A refused client gets the reason as an error message.

## Setup (2026-11-08)

- Host: RTX 5090, Seoul. Client: RTX 4060 Ti, a home PC. Route: one SSH tunnel. Measured with scp: about
  22 Mbit/s each way, about 250 ms round trip.
- Content: a CS2 first-person recording at 240 fps (960 x 540, upscaled to 1920 x 1080 so the engine takes its
  1080p path: network at 864 x 480, render at full size). Source = every 4th frame (60 fps); phase 1/2 is compared
  with the real frame between them. For 24 -> 120 fps: every 10th frame, phases 1/5 to 4/5.
- Model: FlyHighFG v6p (the flyhigh-engine v0.3.1 default), k = 1 unless stated.
- "vs truth" = PSNR of the generated frame against the real one. "vs local" = PSNR against the whole engine run on
  the client, i.e. how much the transport changed the picture. Above about 40 dB the difference is not visible.

## Results: client and host with H.264 instructions

Steady state over a 10 s stream (`--loops`). Latency = from the client starting a frame's instructions to holding the
maps of the pair that frame closes, on the client's own clock.

| run | up | down | latency p50 / p95 | client per frame | vs truth (local engine) |
|---|---:|---:|---:|---:|---|
| 60 fps, 2x | 6.1 KB, 3.0 Mbit/s | 12.8 KB, 6.3 Mbit/s | 279 / 282 ms | 4.6 ms | 28.075 (28.128) dB |
| 24 fps, 5x (24 -> 120), 1.5 Mbit/s stream | 7.6 KB, 1.4 Mbit/s | 13.7 KB, 2.7 Mbit/s | 282 / 285 ms | 5.8 ms | 24.395 (24.390) dB |
| two clients at once, 60 fps each | 3.0 Mbit/s each | 6.3 Mbit/s each | 278 / 282 and 278 / 288 ms | 4.7 to 5.3 ms | 28.076 / 28.075 dB |

- **Same picture as the local engine**: −0.05 dB at 60 fps, equal at 24 -> 120.
- **Latency is the round trip and nothing else**: 279 ms against a 250 ms ping, flat from the 3rd second on. The
  first 1 to 2 s run longer (up to about 0.7 s) while TCP opens its window on a 250 ms path.
- **Client cost** (RTX 4060 Ti + CPU): downscale and I420 on the GPU 1.5 ms, H.264 encode (openh264, low
  complexity, 4 threads) 3.1 ms, render 0.8 ms per pair. The whole engine locally takes about 5.4 ms of GPU per
  pair; for a browser client the encoder would be the hardware one behind WebCodecs.
- **Host cost** (RTX 5090): 4.5 ms to decode the H.264 frame and convert it (CPU), 3.6 ms for the network, readback
  and map packing: about 120 frames a second per client thread.
- **Traffic per hour**: about 4.2 GB at 60 fps, about 1.8 GB at 24 -> 120.

Against sending frames, for 24 -> 120 fps:

| | traffic | per hour |
|---|---:|---:|
| instructions (measured) | 1.4 + 2.7 = 4.1 Mbit/s | ~1.8 GB |
| frames as JPEG (scaled from the measured 30 fps 4x run) | ~45 + ~170 Mbit/s | ~95 GB |
| frames as an H.264 stream (estimate) | ~5-8 + ~15-25 Mbit/s | ~9-15 GB |

Frames back would also have to survive a video codec (a loss of their own), and the host would encode 1080p at
120 fps for every client.

Playback runs late by the latency plus one source interval (a pair needs its second frame): about 0.3 s here. That
is fine for video, after a short sync step when SlipStream switches on, and useless for games. Even a host 20 ms away
adds more than a frame of input lag compared with the local engine.

## Results: a browser client on the same protocol

A Chrome extension (Framegen SlipStream, a separate project) implements the client in the browser: WebGPU for the
engine's downscale and render shaders, WebCodecs for the H.264 instructions, the same WebSocket protocol. Measured in
Playwright Chromium 158, RTX 4060 Ti, the same 5090 host and tunnel. Test page: the CS2 clip as a
1080p H.264 file, 60 fps or 24 fps, 40 to 80 s long. Display 240 Hz. Numbers from the extension's HUD, steady state.

| source | output | delay (sync) | up | down | latency p50 / p95 | late maps |
|---|---|---:|---:|---:|---:|---:|
| 60 fps | 240 fps (every display frame generated) | 436 ms | 3.4 Mbit/s, 7 KB/ipf | 6.4-7.2 Mbit/s, 13-15 KB/ipf | 279 / 285 ms | 0 |
| 24 fps | 240 fps | 422 ms | 1.3-1.6 Mbit/s | 2.5-3.1 Mbit/s | 281 / 284 ms | 0 |

- The browser's WebCodecs H.264 stream decodes on the host's openh264 as it is (constrained baseline, Annex B).
- Capture to sent instructions takes 4 to 5 ms. The client never runs the network.
- Sync took 1.5 to 3 s. The first seconds of a fresh connection queue, so sync waits until the spread of the last
  second's latencies is under 60 ms + 15 % of their median, then fixes D.
- From an https page the extension reached `ws://127.0.0.1` without a prompt in this Chromium build.
- Not tested here: YouTube itself (in a fresh automated profile it asks to sign in), audio sync by ear, other GPUs
  and browsers.

## Lookahead: hiding the round trip for video

A video page already holds its future: a player buffers tens of seconds ahead. Taking the frames from there, a second
or so before they play, sends the instructions early enough that the maps are back before the video reaches them.
The overlay then follows the video's own clock, so the viewer sees no delay and the sound is untouched.

Two sources of future frames, both in the browser client:
- plain files: a hidden shadow player plays the same file `lead` seconds ahead;
- Media Source players: the page's appends to its video SourceBuffer, copied in the page's world, demuxed (fragmented
  MP4, WebM) and decoded ahead with WebCodecs. The decoder's timestamps are matched to the player's media time once
  from 16 x 9 thumbnails (an edit list or a composition delay would shift them; on the test files the offset came
  out 0).

Same setup as above (60 fps source, 240 Hz display, the Seoul host, about 280 ms away):

| source | delay seen | output | late maps | live after start | after a seek |
|---|---:|---:|---:|---:|---:|
| plain file (shadow player) | none | 240 fps | 0 | 2.1 s | 1.6 s |
| Media Source, H.264 High with B-frames (fMP4) | none | 240 fps | 0 | 2.7 s | not tested |
| Media Source, VP9 (WebM) | none | 240 fps | 0 | 2.6 s | 1.0 s |
| Media Source, AV1 (fMP4) | none | 240 fps | 0 | 2.4 s | not tested |

At 60 fps on a 240 Hz display, about 190 of the 240 presents a second are generated in-betweens and about 50 land
on a source frame. The lead starts at 0.8 s, grows while the first seconds of a connection queue (TCP opening its
window), then shrinks back to p95 + 0.35 s, about 0.8 s here. Pause and play need no new sync.

What it costs: the video is decoded twice (by the page and ahead for the instructions), and a page keeps a copy of
the newest video appends (48 MB per buffer). Games have no future frames: for them the latency stays the round trip.

## Host capacity (24 -> 120 fps clients on one RTX 5090)

The `load` example on the host machine itself, so the link is not measured: N clients play the same pre-encoded
instructions at 24 fps for 20 s.

| clients | host latency p50 / p95 | |
|---:|---:|---|
| 4 | 10 / 12 ms | |
| 12 | 9 / 11 ms | |
| 24 | 10 / 25 ms | 4 GB of VRAM |
| 32 | 14 / 111 ms | at the limit |
| 40 | 201 / 986 ms | over |
| 48 | 2711 / 4389 ms | over |

About 30 clients at 24 -> 120 per RTX 5090 (about 720 pairs a second), about 12 at a 60 fps source. The GPU is the
limit, and every client has its own device and session today. A shared device with batched pairs is the obvious next
step. At 30 clients the host moves about 125 Mbit/s.

## Results: how much the transport costs (first round)

60 fps source, 2x, JPEG or offline H.264 uploads (the client streamed images then). Local engine: **28.128 dB** vs
truth. Repeating the nearest source frame: 22.59 dB.

| upload / maps | up, KB per frame | down, KB per pair | vs truth, dB | vs local, dB |
|---|---:|---:|---:|---:|
| f16 network input, `f16z` maps (exact path, 30 pairs) | 1866 | 470 | = local | 69.5 |
| JPEG q90 / `f16z` | 117 | 468 | 28.127 | 55.4 |
| JPEG q90 / `q10.8` | 117 | 189 | 28.127 | 55.2 |
| JPEG q75 / `q6.4` | 68 | 54 | 28.094 | 46.5 |
| JPEG q75 / `j75.60` | 68 | 19 | 28.106 | 47.3 |
| JPEG q75 / `j50.40` | 68 | 13 | 28.092 | 45.3 |
| JPEG q30 / `j75.60` | 33 | 19 | 28.101 | 44.3 |
| JPEG q15 / `j75.60` | 22 | 20 | 28.084 | 41.3 |
| H.264 (x264) 6 Mbit/s / `j50.40`, offline round trip | 10.9 | 13 | 28.095 | 43.4 |
| H.264 (x264) 3 Mbit/s / `j50.40`, offline round trip | 5.4 | 13 | 28.098 | 41.4 |
| H.264 (x264) 1.5 Mbit/s / `j50.40`, offline round trip | 2.7 | 13 | 27.973 | 37.2 |
| frames: JPEG q75 up, JPEG q75 in-betweens down | 249 | 234 | 28.036 | 40.5 |

The network barely reacts to a compressed input, and the maps survive 8-bit grayscale JPEG. The exact path (f16 in,
f16 maps out) reproduces the local engine (69.5 dB: CPU against GPU rounding of the downscale), which checks the
plumbing.

### Spending the host's headroom

The client's cost is the render whatever the network costs, so the host can run more than a real-time budget allows.
Same model: k = 2 gives +0.08 dB and k = 3 gives +0.10 dB (28.212 and 28.230 against 28.128) at no client cost. The
bigger lever is a model trained for the host, several times the size of v6p, which the local engine could not run in
real time.

## Next

1. **Browser client and lookahead: done** (above). Next: YouTube by hand, scene cuts (the host could flag them), a
   CORS reload for cross-origin videos like Framegen's, `wss://` (or WebTransport) for remote hosts.
2. **Host-side decode on the GPU** (NVDEC, or I420 straight into the network input): 4.5 ms of CPU per frame today.
3. **Maps as video.** The maps are smooth and change slowly from pair to pair; a video codec (HEVC / AV1, 10 bits
   for the flow) should cut the 13 KB per pair several times.
4. **A host-sized model** (above), judged with the same harness against the engine and LSFG 3.1.

## Caveats

- One clip (CS2, upscaled from 540p) and one route. Natural video and anime next, plus a host closer to the user.
- PSNR only. The flick contrast metric and the side-by-side videos (the engine's acceptance tools) still need to run
  on SlipStream output, although at 40 dB or more against the local engine nothing visible is expected to change.
- The browser client is tested on local test pages only (plain files and Media Source with H.264, VP9 and AV1),
  not on YouTube itself.

Raw reports: `results/2026-11-08/*.json` (bench output, one file per run).
