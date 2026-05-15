//! Wire `mcp-distill` into agent CLIs as an MCP server.
//!
//! Currently supports codex (`~/.codex/config.toml`, override via `$CODEX_HOME`).
//! Edits are made in-place with `toml_edit` to preserve other settings,
//! comments, and formatting.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use toml_edit::{value, Array, DocumentMut, Item, Table};

#[derive(Debug, Clone, Copy)]
pub enum Client {
    Codex,
}

impl Client {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "codex" => Ok(Client::Codex),
            other => Err(anyhow!("unknown client {other:?} (supported: codex)")),
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

    let mut envs: Vec<(&str, String)> = Vec::new();
    if let Some(root) = &spec.store_root {
        envs.push(("MCP_DISTILL_ROOT", root.to_string_lossy().to_string()));
    }
    if spec.keep_raw {
        envs.push(("MCP_DISTILL_KEEP_RAW", "1".to_string()));
    }
    if let Some(comp) = &spec.compression {
        envs.push(("MCP_DISTILL_COMPRESSION", comp.clone()));
    }
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

pub fn install(spec: &InstallSpec) -> Result<InstallReport> {
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
    fn uninstall_nonexistent_is_noop() {
        let _g = with_codex_home(|_| {});
        let report = uninstall(Client::Codex, "distill").unwrap();
        assert_eq!(report.action, Action::NotPresent);
    }
}
