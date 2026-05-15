//! Wire `mcp-distill` into agent CLIs as an MCP server.
//!
//! Supports:
//! - **codex**: edits `~/.codex/config.toml` (override via `$CODEX_HOME`)
//!   in place via `toml_edit`, preserving other settings/comments.
//! - **claude** (Claude Code): shells out to `claude mcp add -s user`,
//!   which writes user-scope MCP config (~/.claude.json) in whatever
//!   format the installed CLI version expects.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use toml_edit::{value, Array, DocumentMut, Item, Table};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Client {
    Codex,
    Claude,
}

impl Client {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "codex" => Ok(Client::Codex),
            "claude" | "claude-code" | "claudecode" => Ok(Client::Claude),
            other => Err(anyhow!(
                "unknown client {other:?} (supported: codex, claude)"
            )),
        }
    }

    pub fn config_path(self) -> PathBuf {
        match self {
            Client::Codex => {
                if let Ok(home) = std::env::var("CODEX_HOME") {
                    return PathBuf::from(home).join("config.toml");
                }
                if let Some(home) = directories::BaseDirs::new() {
                    return home.home_dir().join(".codex").join("config.toml");
                }
                PathBuf::from(".codex/config.toml")
            }
            Client::Claude => {
                // Best-guess for *reporting* — actual writes go through `claude mcp`.
                if let Some(home) = directories::BaseDirs::new() {
                    return home.home_dir().join(".claude.json");
                }
                PathBuf::from(".claude.json")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct InstallSpec {
    pub client: Client,
    pub server_name: String,
    pub binary: PathBuf,
    pub store_root: Option<PathBuf>,
    pub keep_raw: bool,
    pub compression: Option<String>,
    pub force: bool,
    /// codex per-server approval mode: "auto" | "prompt" | "approve".
    /// `approve` makes codex run distill's tool calls without prompting —
    /// without this, `codex exec` cancels every MCP tool call.
    pub approval_mode: Option<String>,
}

#[derive(Debug)]
pub struct InstallReport {
    pub config_path: PathBuf,
    pub action: Action,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    Created,
    Updated,
    Unchanged,
    Removed,
    NotPresent,
}

fn load_or_new(path: &Path) -> Result<DocumentMut> {
    if !path.exists() {
        return Ok(DocumentMut::new());
    }
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    text.parse::<DocumentMut>()
        .with_context(|| format!("parse {}", path.display()))
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn build_server_table(spec: &InstallSpec) -> Table {
    let mut t = Table::new();
    t.set_implicit(false);
    t["command"] = value(spec.binary.to_string_lossy().to_string());

    let mut args = Array::new();
    args.push("serve");
    t["args"] = value(args);

    let envs = env_pairs(spec);
    if !envs.is_empty() {
        let mut env_tbl = Table::new();
        env_tbl.set_implicit(false);
        for (k, v) in envs {
            env_tbl[k] = value(v);
        }
        t["env"] = Item::Table(env_tbl);
    }
    if let Some(mode) = &spec.approval_mode {
        t["default_tools_approval_mode"] = value(mode.clone());
    }
    t
}

fn server_tables_equal(a: &Table, b: &Table) -> bool {
    a.to_string() == b.to_string()
}

/// Build the env-var pairs we want exported into the spawned server.
fn env_pairs(spec: &InstallSpec) -> Vec<(&'static str, String)> {
    let mut envs: Vec<(&'static str, String)> = Vec::new();
    if let Some(root) = &spec.store_root {
        envs.push(("MCP_DISTILL_ROOT", root.to_string_lossy().to_string()));
    }
    if spec.keep_raw {
        envs.push(("MCP_DISTILL_KEEP_RAW", "1".to_string()));
    }
    if let Some(comp) = &spec.compression {
        envs.push(("MCP_DISTILL_COMPRESSION", comp.clone()));
    }
    envs
}

pub fn install(spec: &InstallSpec) -> Result<InstallReport> {
    match spec.client {
        Client::Codex => install_codex(spec),
        Client::Claude => install_claude(spec),
    }
}

fn install_codex(spec: &InstallSpec) -> Result<InstallReport> {
    let path = spec.client.config_path();
    ensure_parent(&path)?;
    let mut doc = load_or_new(&path)?;

    // Ensure top-level [mcp_servers] table exists.
    if !doc.contains_key("mcp_servers") {
        let mut t = Table::new();
        t.set_implicit(true);
        doc["mcp_servers"] = Item::Table(t);
    }
    let servers = doc["mcp_servers"]
        .as_table_mut()
        .ok_or_else(|| anyhow!("`mcp_servers` exists but is not a table"))?;

    let new_table = build_server_table(spec);
    let action = match servers.get(&spec.server_name) {
        Some(Item::Table(existing)) => {
            if server_tables_equal(existing, &new_table) {
                Action::Unchanged
            } else if spec.force {
                Action::Updated
            } else {
                return Err(anyhow!(
                    "[mcp_servers.{}] already exists and differs from the new entry; \
                     pass --force to overwrite",
                    spec.server_name,
                ));
            }
        }
        Some(_) => {
            return Err(anyhow!(
                "[mcp_servers.{}] exists but is not a table",
                spec.server_name,
            ));
        }
        None => Action::Created,
    };

    if action != Action::Unchanged {
        servers.insert(&spec.server_name, Item::Table(new_table));
    }

    fs::write(&path, doc.to_string()).with_context(|| format!("write {}", path.display()))?;
    Ok(InstallReport {
        config_path: path,
        action,
    })
}

pub fn uninstall(client: Client, server_name: &str) -> Result<InstallReport> {
    match client {
        Client::Codex => uninstall_codex(client, server_name),
        Client::Claude => uninstall_claude(server_name),
    }
}

fn uninstall_codex(client: Client, server_name: &str) -> Result<InstallReport> {
    let path = client.config_path();
    if !path.exists() {
        return Ok(InstallReport {
            config_path: path,
            action: Action::NotPresent,
        });
    }
    let mut doc = load_or_new(&path)?;
    let action = match doc.get_mut("mcp_servers").and_then(|i| i.as_table_mut()) {
        Some(servers) if servers.contains_key(server_name) => {
            servers.remove(server_name);
            Action::Removed
        }
        _ => Action::NotPresent,
    };
    if action == Action::Removed {
        fs::write(&path, doc.to_string()).with_context(|| format!("write {}", path.display()))?;
    }
    Ok(InstallReport {
        config_path: path,
        action,
    })
}

// --- claude (Claude Code) ----------------------------------------------------

fn require_claude_cli() -> Result<()> {
    Command::new("claude")
        .arg("--version")
        .output()
        .map_err(|e| {
            anyhow!(
                "could not invoke `claude` CLI: {e}. \
             Install Claude Code (https://claude.com/claude-code) and ensure `claude` is on PATH."
            )
        })?;
    Ok(())
}

/// Whether `claude mcp` already has an entry with this name (any scope).
fn claude_has_server(name: &str) -> Result<bool> {
    let out = Command::new("claude")
        .args(["mcp", "get", name])
        .output()
        .with_context(|| "spawn `claude mcp get`")?;
    Ok(out.status.success())
}

fn claude_remove(name: &str) -> Result<()> {
    // Try user scope first; fall back to default scope. Either succeeding is fine.
    let _ = Command::new("claude")
        .args(["mcp", "remove", "-s", "user", name])
        .status();
    let _ = Command::new("claude")
        .args(["mcp", "remove", name])
        .status();
    Ok(())
}

/// Build the argv we pass to `claude mcp add`. Public for unit testing.
pub fn claude_install_argv(spec: &InstallSpec) -> Vec<String> {
    let mut argv: Vec<String> = vec!["mcp".into(), "add".into(), "-s".into(), "user".into()];
    // `claude mcp add` declares `-e` as variadic, so a chain of `-e K=V` will
    // greedily swallow the next positional (the server name) as another env
    // value. The `--env=K=V` form is a single token and avoids this.
    for (k, v) in env_pairs(spec) {
        argv.push(format!("--env={k}={v}"));
    }
    argv.push(spec.server_name.clone());
    argv.push("--".into());
    argv.push(spec.binary.to_string_lossy().to_string());
    argv.push("serve".into());
    argv
}

fn install_claude(spec: &InstallSpec) -> Result<InstallReport> {
    require_claude_cli()?;
    let existed = claude_has_server(&spec.server_name)?;
    if existed && !spec.force {
        return Err(anyhow!(
            "claude already has an MCP server named {:?}; pass --force to overwrite",
            spec.server_name,
        ));
    }
    if existed {
        claude_remove(&spec.server_name)?;
    }
    let argv = claude_install_argv(spec);
    let out = Command::new("claude")
        .args(&argv)
        .output()
        .with_context(|| "spawn `claude mcp add`")?;
    if !out.status.success() {
        return Err(anyhow!(
            "`claude mcp add` failed (status={:?}): {}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim(),
        ));
    }
    let action = if existed {
        Action::Updated
    } else {
        Action::Created
    };
    Ok(InstallReport {
        config_path: Client::Claude.config_path(),
        action,
    })
}

fn uninstall_claude(server_name: &str) -> Result<InstallReport> {
    require_claude_cli()?;
    let existed = claude_has_server(server_name)?;
    if !existed {
        return Ok(InstallReport {
            config_path: Client::Claude.config_path(),
            action: Action::NotPresent,
        });
    }
    let out = Command::new("claude")
        .args(["mcp", "remove", "-s", "user", server_name])
        .output()
        .with_context(|| "spawn `claude mcp remove`")?;
    if !out.status.success() {
        // Try without explicit scope — older claude versions or different default scope.
        let out2 = Command::new("claude")
            .args(["mcp", "remove", server_name])
            .output()
            .with_context(|| "spawn `claude mcp remove` (fallback)")?;
        if !out2.status.success() {
            return Err(anyhow!(
                "`claude mcp remove` failed: {}",
                String::from_utf8_lossy(&out2.stderr).trim(),
            ));
        }
    }
    Ok(InstallReport {
        config_path: Client::Claude.config_path(),
        action: Action::Removed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use tempfile::TempDir;

    // CODEX_HOME is process-global, so install tests can't run in parallel.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn make_spec(server_name: &str) -> InstallSpec {
        InstallSpec {
            client: Client::Codex,
            server_name: server_name.into(),
            binary: PathBuf::from("/usr/local/bin/mcp-distill"),
            store_root: Some(PathBuf::from("/tmp/distill")),
            keep_raw: false,
            compression: Some("zstd".into()),
            force: false,
            approval_mode: Some("approve".into()),
        }
    }

    /// Returns (tempdir, env-lock guard). The guard must outlive the test body
    /// so concurrent tests don't trample each other's CODEX_HOME.
    fn with_codex_home<F: FnOnce(&Path)>(f: F) -> (TempDir, parking_lot::MutexGuard<'static, ()>) {
        let guard = ENV_LOCK.lock();
        let tmp = TempDir::new().unwrap();
        let codex = tmp.path().join("codex");
        fs::create_dir_all(&codex).unwrap();
        std::env::set_var("CODEX_HOME", &codex);
        f(&codex);
        (tmp, guard)
    }

    #[test]
    fn fresh_install_creates_config() {
        let _g = with_codex_home(|_| {});
        let report = install(&make_spec("distill")).unwrap();
        assert_eq!(report.action, Action::Created);
        let body = fs::read_to_string(&report.config_path).unwrap();
        assert!(body.contains("[mcp_servers.distill]"));
        assert!(body.contains("command = \"/usr/local/bin/mcp-distill\""));
        assert!(body.contains("MCP_DISTILL_ROOT"));
        assert!(body.contains("MCP_DISTILL_COMPRESSION"));
        assert!(!body.contains("MCP_DISTILL_KEEP_RAW")); // off by default
                                                         // Auto-approve so codex doesn't cancel our MCP calls.
        assert!(body.contains("default_tools_approval_mode = \"approve\""));
    }

    #[test]
    fn idempotent_when_unchanged() {
        let _g = with_codex_home(|_| {});
        let spec = make_spec("distill");
        assert_eq!(install(&spec).unwrap().action, Action::Created);
        assert_eq!(install(&spec).unwrap().action, Action::Unchanged);
    }

    #[test]
    fn refuses_to_clobber_without_force() {
        let _g = with_codex_home(|p| {
            fs::write(
                p.join("config.toml"),
                "[mcp_servers.distill]\ncommand = \"/old/path\"\n",
            )
            .unwrap();
        });
        let err = install(&make_spec("distill")).unwrap_err();
        assert!(err.to_string().contains("--force"));
    }

    #[test]
    fn force_overwrites_existing() {
        let _g = with_codex_home(|p| {
            fs::write(
                p.join("config.toml"),
                "model = \"gpt-5\"\n[mcp_servers.distill]\ncommand = \"/old/path\"\n",
            )
            .unwrap();
        });
        let mut spec = make_spec("distill");
        spec.force = true;
        let report = install(&spec).unwrap();
        assert_eq!(report.action, Action::Updated);
        let body = fs::read_to_string(&report.config_path).unwrap();
        assert!(body.contains("/usr/local/bin/mcp-distill"));
        assert!(!body.contains("/old/path"));
        // Other settings preserved.
        assert!(body.contains("model = \"gpt-5\""));
    }

    #[test]
    fn uninstall_removes_entry() {
        let _g = with_codex_home(|_| {});
        install(&make_spec("distill")).unwrap();
        let report = uninstall(Client::Codex, "distill").unwrap();
        assert_eq!(report.action, Action::Removed);
        let body = fs::read_to_string(&report.config_path).unwrap();
        assert!(!body.contains("[mcp_servers.distill]"));
    }

    #[test]
    fn claude_argv_includes_env_and_serve() {
        let mut spec = make_spec("distill");
        spec.client = Client::Claude;
        spec.keep_raw = true;
        let argv = claude_install_argv(&spec);
        assert_eq!(&argv[0..4], &["mcp", "add", "-s", "user"]);
        assert!(argv
            .iter()
            .any(|s| s == "--env=MCP_DISTILL_ROOT=/tmp/distill"));
        assert!(argv.iter().any(|s| s == "--env=MCP_DISTILL_KEEP_RAW=1"));
        assert!(argv
            .iter()
            .any(|s| s == "--env=MCP_DISTILL_COMPRESSION=zstd"));
        // server name, separator, command, command-arg
        let sep = argv.iter().position(|s| s == "--").unwrap();
        assert_eq!(argv[sep - 1], "distill");
        assert_eq!(argv[sep + 1], "/usr/local/bin/mcp-distill");
        assert_eq!(argv[sep + 2], "serve");
    }

    #[test]
    fn claude_argv_omits_unset_env() {
        let mut spec = make_spec("distill");
        spec.client = Client::Claude;
        spec.store_root = None;
        spec.compression = None;
        spec.keep_raw = false;
        let argv = claude_install_argv(&spec);
        assert!(!argv.iter().any(|s| s.starts_with("--env=")));
    }

    #[test]
    fn uninstall_nonexistent_is_noop() {
        let _g = with_codex_home(|_| {});
        let report = uninstall(Client::Codex, "distill").unwrap();
        assert_eq!(report.action, Action::NotPresent);
    }
}
