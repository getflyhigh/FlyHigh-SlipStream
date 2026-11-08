//!   slipstream-server --model <dir> [--listen 127.0.0.1:9300] [--token <secret>] [--max-clients 4] [--quiet]
use anyhow::{Context, Result};
use slipstream_server::{Host, HostOptions};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let mut o = HostOptions::new(get("--model").context("--model <dir>")?).debug(!args.iter().any(|a| a == "--quiet"));
    if let Some(v) = get("--listen") {
        o = o.listen(v);
    }
    if let Some(v) = get("--token") {
        o = o.token(v);
    }
    if let Some(v) = get("--max-clients") {
        o = o.max_clients(v.parse().context("--max-clients")?);
    }
    Host::new(o).run()
}
