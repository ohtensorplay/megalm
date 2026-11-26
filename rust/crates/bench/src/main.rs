use clap::Parser;
use megalm_bench::{run_serve, Cli, Command};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => run_serve(*args).await,
        Command::Report(args) => megalm_bench::report::render_command(&args),
    }
}
