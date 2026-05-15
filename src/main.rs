use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use rmcp::{transport::stdio, ServiceExt};
use tracing_subscriber::EnvFilter;

use mcp_distill::install::{self, Action, Client, InstallSpec};
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
    /// Wire mcp-distill into an agent CLI's MCP server config.
    Install(InstallArgs),
    /// Remove mcp-distill from an agent CLI's MCP server config.
    Uninstall(UninstallArgs),
}

#[derive(clap::Args)]
struct InstallArgs {
    /// Target client: `codex` or `claude`.
    client: String,
    /// Server name to register (default: distill).
    #[arg(long, default_value = "distill")]
    name: String,
    /// Path to the mcp-distill binary (default: the currently-running binary).
    #[arg(long)]
    binary: Option<PathBuf>,
    /// Storage root the installed server should use (writes MCP_DISTILL_ROOT).
    #[arg(long)]
    store_root: Option<PathBuf>,
    /// Persist provider-native `raw` payloads (writes MCP_DISTILL_KEEP_RAW=1).
    #[arg(long)]
    keep_raw: bool,
    /// Compression: `none` or `zstd` (writes MCP_DISTILL_COMPRESSION).
    #[arg(long)]
    compression: Option<String>,
    /// (codex only) Per-server tool-approval mode written into codex config:
    /// `auto` | `prompt` | `approve`. Default `approve` so codex auto-approves
    /// our recording calls (otherwise `codex exec` cancels them with no human).
    /// Ignored for claude — Claude Code handles approvals at runtime in its UI.
    #[arg(long, default_value = "approve")]
    approval: String,
    /// Overwrite an existing entry that differs from the new one.
    #[arg(long)]
    force: bool,
}

#[derive(clap::Args)]
struct UninstallArgs {
    /// Target client: `codex` or `claude`.
    client: String,
    /// Server name to remove (default: distill).
    #[arg(long, default_value = "distill")]
    name: String,
}

fn current_binary() -> Result<PathBuf> {
    std::env::current_exe().map_err(|e| anyhow!("could not resolve current executable: {e}"))
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
        Cmd::Install(args) => {
            let client = Client::parse(&args.client)?;
            let binary = match args.binary {
                Some(p) => p,
                None => current_binary()?,
            };
            let approval_mode = if matches!(client, Client::Codex) {
                match args.approval.as_str() {
                    "" | "none" | "off" => None,
                    other => Some(other.to_string()),
                }
            } else {
                None
            };
            let spec = InstallSpec {
                client,
                server_name: args.name,
                binary,
                store_root: args.store_root,
                keep_raw: args.keep_raw,
                compression: args.compression,
                approval_mode,
                force: args.force,
            };
            let report = install::install(&spec)?;
            let verb = match report.action {
                Action::Created => "installed",
                Action::Updated => "updated",
                Action::Unchanged => "unchanged",
                Action::Removed | Action::NotPresent => unreachable!(),
            };
            println!(
                "{verb} mcp_servers.{} in {}",
                spec.server_name,
                report.config_path.display()
            );
        }
        Cmd::Uninstall(args) => {
            let client = Client::parse(&args.client)?;
            let report = install::uninstall(client, &args.name)?;
            let verb = match report.action {
                Action::Removed => "removed",
                Action::NotPresent => "not present",
                _ => unreachable!(),
            };
            println!(
                "{verb}: mcp_servers.{} in {}",
                args.name,
                report.config_path.display()
            );
        }
    }
    Ok(())
}
