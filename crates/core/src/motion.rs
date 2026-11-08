//! One engine session at the client's frame size, used both ways: the server runs the network on uploaded network
//! inputs and reads the motion maps out; the client writes received maps in and runs only the render. Both build the
//! same session (fast render, linear t: the t-dependent half runs once per pair at t = 0.5, every phase is a render),
//! so the maps' byte layout is the same at both ends.
use crate::fields::Layout;
use anyhow::{ensure, Context, Result};
use flyhigh_engine::{clone_step, map_read, Engine, EngineOptions, Model, Session, SessionOptions, Tensor};
use std::path::Path;
use wgpu::util::DeviceExt;

pub struct Motion {
    i420: std::cell::OnceCell<wgpu::ComputePipeline>,
    pub engine: Engine,
    pub model: Model,
    pub s: Session,
    pub w: u32,
    pub h: u32,
    out: usize,
}

impl Motion {
    /// `inputs` = frame slots on the GPU (the host needs none, a client keeps a ring of recent frames)
    pub fn new(model_dir: &Path, w: u32, h: u32, k: u32, inputs: usize) -> Result<Motion> {
        let engine = Engine::new(&EngineOptions::default())?;
        let model = Model::from_dir(model_dir)?;
        Motion::with(engine, model, w, h, k, inputs)
    }
    pub fn with(engine: Engine, model: Model, w: u32, h: u32, k: u32, inputs: usize) -> Result<Motion> {
        ensure!(!engine.gpu().f32_mode, "SlipStream needs shader-f16 (the maps travel as f16)");
        let mut o = SessionOptions::new(w, h);
        o.k = k;
        o.fast_render = true;
        o.linear_t = true;
        let mut s = Session::new(&engine, &model, &o)?;
        ensure!(
            s.ext().is_fast(),
            "{w}x{h}: the network runs at the frame size here, nothing to split (needs > 960 px)"
        );
        for _ in 0..inputs.max(2) {
            s.add_input();
        }
        let out = s.add_output();
        Ok(Motion { i420: Default::default(), engine, model, s, w, h, out })
    }
    fn t(&self, k: &str) -> &Tensor {
        &self.s.ext().env[k]
    }
    pub fn gpu_name(&self) -> String {
        self.engine.gpu().name.clone()
    }
    /// The network input size (32-aligned), what motion-mode clients upload.
    pub fn rung(&self) -> (u32, u32) {
        let a = self.t("fast:ai");
        (a.w, a.h)
    }
    pub fn work(&self) -> (u32, u32) {
        (self.s.ext().wp, self.s.ext().hp)
    }
    pub fn render_info(&self) -> crate::proto::RenderInfo {
        let ext = self.s.ext();
        let (f, v, ai) = (self.t("fast:ftq"), self.t("fast:vtq"), self.t("fast:ai"));
        crate::proto::RenderInfo {
            fsx: f.scale * ext.wp as f32 / ai.w as f32,
            fsy: f.scale * ext.hp as f32 / ai.h as f32,
            vs: v.scale,
            cubic: ext.net.cubic as u32,
            flags: ext.net.render_flags(),
        }
    }
    pub fn layout(&self) -> Layout {
        let (f, v, r) = (self.t("fast:ftq"), self.t("fast:vtq"), self.t("fast:rt"));
        let plane = (f.h * f.w * 2) as u64;
        // the render reads flow channels f0..f0+4, weights v0..v0+3, residual 0..3
        let ranges =
            [(f.c0 as u64 * plane, 4 * plane), (v.c0 as u64 * plane, 3 * plane), (r.c0 as u64 * plane, 3 * plane)];
        Layout { ranges, hq: f.h, wq: f.w }
    }
    fn bufs(&self) -> [&wgpu::Buffer; 3] {
        [&*self.t("fast:ftq").buf, &*self.t("fast:vtq").buf, &*self.t("fast:rt").buf]
    }

    /// Server: the network on two uploaded network inputs (f16 BGR planes at `rung`), maps left in the session.
    pub fn pair_inputs(&self, a: &[u16], b: &[u16]) {
        let gpu = self.engine.gpu();
        gpu.queue.write_buffer(&self.t("fast:ai").buf, 0, bytemuck::cast_slice(a));
        gpu.queue.write_buffer(&self.t("fast:bi").buf, 0, bytemuck::cast_slice(b));
        let ext = self.s.ext();
        gpu.replay(&ext.plan_pair.iter().map(clone_step).collect::<Vec<_>>());
        ext.mid_maps_half(gpu);
    }

    /// Client: the network input of the frame in input slot `slot`, made on the GPU by the engine's own downscale
    /// (the first step of its pair stage), as f16 BGR planes at `rung`.
    pub fn rung_gpu(&self, slot: usize) -> Result<Vec<u16>> {
        let gpu = self.engine.gpu();
        let v = (slot, self.s.input_view(slot));
        let steps = self.s.ext().pair_steps_views(gpu, v, v);
        gpu.replay(&steps[..1]);
        let ai = self.t("fast:ai");
        let bytes = self.read_range(&ai.buf, 0, (3 * ai.w * ai.h * 2) as u64)?;
        Ok(bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes([b[0], b[1]])).collect())
    }

    /// Client: the network input of the frame in input slot `slot` as an I420 picture at `rung` (i420.wgsl), what
    /// the H.264 encoder takes: the colour conversion stays on the GPU and a quarter of the bytes come back.
    pub fn rung_i420(&self, slot: usize) -> Result<Vec<u8>> {
        let gpu = self.engine.gpu();
        let v = (slot, self.s.input_view(slot));
        let steps = self.s.ext().pair_steps_views(gpu, v, v);
        gpu.replay(&steps[..1]);
        let ai = self.t("fast:ai");
        let (w, h) = (ai.w, ai.h);
        ensure!(w % 8 == 0 && h % 2 == 0, "network input {w}x{h}: I420 here needs a width in 8s and an even height");
        let pipe = self.i420.get_or_init(|| {
            let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ss_i420"),
                source: wgpu::ShaderSource::Wgsl(include_str!("i420.wgsl").into()),
            });
            gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ss_i420"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let size = (w * h * 3 / 2) as u64;
        let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ss_i420_out"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ss_i420_p"),
            contents: bytemuck::cast_slice(&[w, h]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ai.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
        });
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((w / 8).div_ceil(8), (h / 2).div_ceil(8), 1);
        }
        gpu.queue.submit(Some(enc.finish()));
        self.read_range(&out, 0, size)
    }

    fn read_range(&self, buf: &wgpu::Buffer, off: u64, len: u64) -> Result<Vec<u8>> {
        let gpu = self.engine.gpu();
        let stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ss_read"),
            size: len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buf, off, &stage, 0, len);
        gpu.queue.submit(Some(enc.finish()));
        let fut = map_read(&stage);
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        pollster::block_on(fut).map_err(|e| anyhow::anyhow!("map_async: {e:?}"))?;
        let v = stage.slice(..).get_mapped_range().map_err(|e| anyhow::anyhow!("mapped range: {e:?}"))?.to_vec();
        stage.unmap();
        Ok(v)
    }

    pub fn read_fields(&self) -> Result<[Vec<u8>; 3]> {
        let lay = self.layout();
        let gpu = self.engine.gpu();
        let total: u64 = lay.ranges.iter().map(|r| r.1).sum();
        let stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ss_fields"),
            size: total,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut at = 0;
        for (buf, &(off, len)) in self.bufs().iter().zip(lay.ranges.iter()) {
            enc.copy_buffer_to_buffer(buf, off, &stage, at, len);
            at += len;
        }
        gpu.queue.submit(Some(enc.finish()));
        let fut = map_read(&stage);
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        pollster::block_on(fut).map_err(|e| anyhow::anyhow!("map_async: {e:?}"))?;
        let all = stage.slice(..).get_mapped_range().map_err(|e| anyhow::anyhow!("mapped range: {e:?}"))?.to_vec();
        stage.unmap();
        let (l0, l1) = (lay.ranges[0].1 as usize, lay.ranges[1].1 as usize);
        Ok([all[..l0].to_vec(), all[l0..l0 + l1].to_vec(), all[l0 + l1..].to_vec()])
    }

    pub fn write_fields(&self, parts: &[Vec<u8>; 3]) -> Result<()> {
        let lay = self.layout();
        let gpu = self.engine.gpu();
        for ((buf, &(off, len)), p) in self.bufs().iter().zip(lay.ranges.iter()).zip(parts.iter()) {
            ensure!(p.len() as u64 == len, "fields part {} bytes, layout {len}", p.len());
            gpu.queue.write_buffer(buf, off, p);
        }
        Ok(())
    }

    pub fn upload(&self, slot: usize, rgba: &[u8]) -> Result<()> {
        self.s.write_rgba(slot, rgba)
    }
    /// The whole engine locally (the reference): the network on input slots (a, b), then phase `t`.
    pub fn local(&self, a: usize, b: usize, t: f32) -> Result<()> {
        self.s.pair(a, b)?;
        self.s.mid(t, self.out)
    }
    /// Only the render of phase `t` from the maps in the session and the frames in input slots (a, b).
    pub fn render(&self, a: usize, b: usize, t: f32) {
        let ext = self.s.ext();
        let gpu = self.engine.gpu();
        let step = ext.render_tex_step(
            gpu,
            (a, self.s.input_view(a)),
            (b, self.s.input_view(b)),
            (self.out, self.s.output_view(self.out)),
            Some(t),
        );
        gpu.replay(&[step]);
    }
    /// The render's output (rgba8unorm, frame size).
    pub fn output_texture(&self) -> &wgpu::Texture {
        self.s.output_texture(self.out)
    }
    pub fn read_out(&self) -> Result<Vec<u8>> {
        self.s.read_rgba(self.out)?.block(self.engine.gpu()).context("read output")
    }
    pub fn wait(&self) {
        self.engine.gpu().device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    }
}
