//! Live debug counters, printed once a second by the client and the host: instructions per second (ips), the size of
//! one frame's instructions (ipf, KB), the bit rate, and mean times.
use std::time::Instant;

#[derive(Default)]
pub struct Rate {
    n: u32,
    bytes: u64,
}
impl Rate {
    pub fn add(&mut self, bytes: usize) {
        self.n += 1;
        self.bytes += bytes as u64;
    }
    /// "60 ips 5.2 KB/ipf 2.49 Mbit/s" over `secs`, then the counters restart
    pub fn take(&mut self, secs: f64) -> String {
        let ips = self.n as f64 / secs;
        let ipf = if self.n > 0 { self.bytes as f64 / self.n as f64 / 1024.0 } else { 0.0 };
        let mbit = self.bytes as f64 * 8.0 / secs / 1e6;
        *self = Rate::default();
        format!("{ips:5.1} ips {ipf:6.1} KB/ipf {mbit:6.2} Mbit/s")
    }
}

#[derive(Default)]
pub struct Avg {
    sum: f64,
    n: u32,
    pub last: f64,
}
impl Avg {
    pub fn add(&mut self, v: f64) {
        self.sum += v;
        self.n += 1;
        self.last = v;
    }
    /// the mean since the last take, then the counters restart (`last` stays)
    pub fn take(&mut self) -> f64 {
        let m = if self.n > 0 { self.sum / self.n as f64 } else { 0.0 };
        self.sum = 0.0;
        self.n = 0;
        m
    }
}

pub struct Every {
    start: Instant,
    last: Instant,
    period: f64,
}
impl Every {
    pub fn second() -> Every {
        let now = Instant::now();
        Every { start: now, last: now, period: 1.0 }
    }
    /// Some(seconds since the previous tick) once a period has passed
    pub fn due(&mut self) -> Option<f64> {
        let s = self.last.elapsed().as_secs_f64();
        if s < self.period {
            return None;
        }
        self.last = Instant::now();
        Some(s)
    }
    pub fn uptime(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}
