use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use rmcp::{transport::stdio, ServiceExt};
use tracing_subscriber::EnvFilter;

use mcp_distill::server::DistillServer;
use mcp_distill::storage::Store;

#[derive(Parser)]
#[command(
    name = "mcp-distill",
    version,
    about = "Record LLM interactions for small-model distillation"
)]
struct Cli {
    /// Storage root (default: $MCP_DISTILL_ROOT or ~/.mcp-distill).
    #[arg(long, global = true)]
    root: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the MCP server over stdio (default).
    Serve,
    /// Print known sessions as JSON.
    List,
    /// Print quick stats as JSON.
    Stats,
    /// Show the resolved storage root.
    Where,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let root = cli.root.unwrap_or_else(Store::default_root);

    match cli.cmd.unwrap_or(Cmd::Serve) {
        Cmd::Serve => {
            tracing::info!(root = %root.display(), "starting mcp-distill server");
            let server = DistillServer::new(root)?;
            let service = server.serve(stdio()).await?;
            service.waiting().await?;
        }
        Cmd::List => {
            let store = Store::new(root)?;
            println!("{}", serde_json::to_string_pretty(&store.list_sessions()?)?);
        }
        Cmd::Stats => {
            let store = Store::new(root)?;
            let sessions = store.list_sessions()?;
            println!(
                "{{\"sessions\": {}, \"root\": \"{}\"}}",
                sessions.len(),
                store.root.display()
            );
        }
        Cmd::Where => {
            println!("{}", root.display());
        }
    }
    Ok(())
}
