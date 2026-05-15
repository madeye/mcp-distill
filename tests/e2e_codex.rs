//! End-to-end: run the real `codex` CLI on the codex-cli source repo with a
//! distillation-style prompt, with our `mcp-distill` MCP server registered as
//! a codex MCP tool provider, and verify codex actually wrote a session via
//! the MCP protocol into our store.
//!
//! Gated. Requires:
//!   - `codex` on PATH
//!   - `OPENAI_API_KEY` (or pre-existing `codex login`) — codex itself enforces this
//!   - network access (to clone openai/codex and call the API)
//!   - `git` on PATH
//!
//! Marked `#[ignore]`. Run explicitly:
//!
//! ```sh
//! MCP_DISTILL_E2E_CODEX=1 cargo test --test e2e_codex -- --ignored --nocapture
//! ```
//!
//! Set `MCP_DISTILL_CODEX_REPO=/path/to/local/codex` to skip the clone and use
//! a local checkout instead — useful for iteration.
//!
//! What it does:
//!   1. Clones (or reuses) the openai/codex repo into a temp dir.
//!   2. Configures codex via `-c mcp_servers.distill.*` to launch our
//!      `mcp-distill` binary (built by Cargo at test-build time) over stdio,
//!      pointed at a per-test storage root.
//!   3. Runs `codex exec --json` with a prompt that asks the model to
//!      analyze the codex repo for optimization opportunities AND to record
//!      its own prompt/response via our `distill` MCP server.
//!   4. Asserts that our store contains at least one session JSONL file with
//!      at least one turn — i.e. codex actually invoked our MCP tools.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use tempfile::TempDir;

const PROMPT: &str = "\
You have an MCP server named `distill` available with tools including \
`record_interaction(provider, model, request_messages, response, system?, tags?, metadata?)`.

Task: scan this codebase and identify 3 concrete optimization opportunities. \
For each, name the file and a one-line rationale. Be terse.

When you have your final answer, you MUST call the `distill.record_interaction` \
tool exactly once. Pass arguments as JSON values (NOT JSON-encoded strings):
  - provider: the literal string \"codex\"
  - model: the literal string \"codex\"
  - request_messages: a JSON array containing one object with keys role=\"user\" \
and content set to the verbatim task statement above (a real string, not JSON-encoded)
  - response: a JSON object with keys role=\"assistant\" and content set to your \
final answer text (a real string, not JSON-encoded)
  - tags: [\"e2e\", \"codex\", \"distillation\"]

Then print the final answer to the user as your normal reply.";

/// Codex's account-default model is used unless `MCP_DISTILL_CODEX_MODEL` is set.
/// (ChatGPT-account auth rejects passing arbitrary model names like "gpt-5".)
fn model_override() -> Option<String> {
    std::env::var("MCP_DISTILL_CODEX_MODEL").ok()
}

// Cargo builds the binary for us and exposes its path here.
const MCP_DISTILL_BIN: &str = env!("CARGO_BIN_EXE_mcp-distill");

fn skip_unless_enabled() -> bool {
    if std::env::var("MCP_DISTILL_E2E_CODEX").ok().as_deref() != Some("1") {
        eprintln!("e2e_codex: skipped (set MCP_DISTILL_E2E_CODEX=1 to enable)");
        return true;
    }
    if which("codex").is_none() {
        eprintln!("e2e_codex: skipped — `codex` not on PATH");
        return true;
    }
    if which("git").is_none() {
        eprintln!("e2e_codex: skipped — `git` not on PATH");
        return true;
    }
    false
}

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn provision_codex_repo(scratch: &Path) -> PathBuf {
    if let Ok(local) = std::env::var("MCP_DISTILL_CODEX_REPO") {
        let p = PathBuf::from(local);
        assert!(
            p.is_dir(),
            "MCP_DISTILL_CODEX_REPO does not exist: {}",
            p.display()
        );
        eprintln!("e2e_codex: using local codex repo at {}", p.display());
        return p;
    }
    let dst = scratch.join("codex");
    eprintln!(
        "e2e_codex: cloning openai/codex (shallow) -> {}",
        dst.display()
    );
    let status = Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "https://github.com/openai/codex.git",
        ])
        .arg(&dst)
        .status()
        .expect("git clone failed to spawn");
    assert!(status.success(), "git clone of openai/codex failed");
    dst
}

fn count_session_files(root: &Path) -> usize {
    fn walk(p: &Path, n: &mut usize) {
        let Ok(entries) = std::fs::read_dir(p) else {
            return;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                walk(&path, n);
            } else if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                *n += 1;
            }
        }
    }
    let mut n = 0;
    walk(&root.join("sessions"), &mut n);
    n
}

#[test]
#[ignore = "requires codex CLI, network, and OPENAI_API_KEY — run with --ignored"]
fn codex_records_optimizations_via_mcp_distill() {
    if skip_unless_enabled() {
        return;
    }

    let scratch = TempDir::new().unwrap();
    let repo = provision_codex_repo(scratch.path());
    let store_root = scratch.path().join("distill-root");
    std::fs::create_dir_all(&store_root).unwrap();
    let last_msg_path = scratch.path().join("last_message.txt");

    eprintln!("e2e_codex: mcp-distill binary at {MCP_DISTILL_BIN}");
    eprintln!("e2e_codex: distill store root: {}", store_root.display());

    // Register our MCP server with codex via -c overrides. The values are
    // parsed as TOML by codex.
    let store_root_str = store_root.to_string_lossy().to_string();
    let cmd_override = format!("mcp_servers.distill.command=\"{MCP_DISTILL_BIN}\"");
    let args_override = "mcp_servers.distill.args=[\"serve\"]".to_string();
    let env_override = format!("mcp_servers.distill.env.MCP_DISTILL_ROOT=\"{store_root_str}\"");

    eprintln!("e2e_codex: running `codex exec` (this can take a few minutes)");
    let mut cmd = Command::new("codex");
    cmd.args([
        "exec",
        "--json",
        "--skip-git-repo-check",
        // Without this, codex `exec` auto-cancels every MCP tool invocation
        // (no human present to approve), and our server never sees the call.
        // `approval_policy="never"` only governs shell commands — MCP tool
        // calls need the broader bypass.
        "--dangerously-bypass-approvals-and-sandbox",
    ]);
    if let Some(m) = model_override() {
        cmd.arg("-m").arg(m);
    }
    cmd.arg("-c")
        .arg(&cmd_override)
        .arg("-c")
        .arg(&args_override)
        .arg("-c")
        .arg(&env_override)
        .arg("-C")
        .arg(&repo)
        .arg("--output-last-message")
        .arg(&last_msg_path)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let started = std::time::Instant::now();
    let mut child = cmd.spawn().expect("failed to spawn codex");
    child
        .stdin
        .as_mut()
        .expect("codex stdin")
        .write_all(PROMPT.as_bytes())
        .expect("write prompt to codex stdin");
    drop(child.stdin.take());
    let output = child.wait_with_output().expect("codex wait failed");
    let elapsed = started.elapsed();
    eprintln!(
        "e2e_codex: codex exec finished in {:.1}s (status={:?})",
        elapsed.as_secs_f32(),
        output.status.code(),
    );
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout_tail: String = String::from_utf8_lossy(&output.stdout)
            .lines()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "codex exec failed (status={:?})\n--- stderr ---\n{}\n--- stdout (tail) ---\n{}",
            output.status.code(),
            stderr,
            stdout_tail,
        );
    }

    let final_message = std::fs::read_to_string(&last_msg_path)
        .unwrap_or_default()
        .trim()
        .to_string();
    eprintln!("e2e_codex: final message = {} chars", final_message.len());

    // Always persist the JSONL event stream + the final message outside the
    // tempdir so we can debug after the test finishes (tempdir is auto-cleaned).
    let debug_dir = std::env::var("MCP_DISTILL_E2E_DEBUG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("mcp-distill-e2e"));
    let _ = std::fs::create_dir_all(&debug_dir);
    let events_path = debug_dir.join("codex_events.jsonl");
    let _ = std::fs::write(&events_path, &output.stdout);
    let _ = std::fs::write(debug_dir.join("codex_stderr.log"), &output.stderr);
    let _ = std::fs::write(debug_dir.join("final_message.txt"), &final_message);
    eprintln!(
        "e2e_codex: wrote codex stdout/stderr to {}",
        debug_dir.display()
    );

    // Surface events that hint whether our MCP server was loaded / called.
    let interesting: Vec<String> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| {
            let s = l.to_ascii_lowercase();
            s.contains("mcp")
                || s.contains("distill")
                || s.contains("tool_call")
                || s.contains("function_call")
        })
        .map(|s| s.to_string())
        .collect();
    eprintln!(
        "e2e_codex: {} interesting events (mcp/distill/tool_call):",
        interesting.len()
    );
    for line in interesting.iter().take(30) {
        eprintln!("  {line}");
    }

    // Defensive: codex must have actually called the model.
    assert!(
        elapsed > Duration::from_secs(2),
        "codex returned suspiciously fast ({:?}); did the call really happen?",
        elapsed,
    );

    // The actual end-to-end assertion: did codex talk to our MCP server?
    let n_sessions = count_session_files(&store_root);
    let index_path = store_root.join("index.jsonl");
    eprintln!(
        "e2e_codex: store contains {} session file(s); index.jsonl exists: {}",
        n_sessions,
        index_path.exists()
    );
    assert!(
        n_sessions >= 1,
        "expected codex to have written at least one session via the distill MCP server, \
         but found none under {}",
        store_root.display(),
    );
    assert!(
        index_path.exists(),
        "expected {} to exist (created by Store::write_meta)",
        index_path.display(),
    );

    // Spot-check one session file: must have a meta line and at least one turn line.
    let session_file = first_session_file(&store_root).expect("session file");
    let body = std::fs::read_to_string(&session_file).unwrap();
    let lines: Vec<&str> = body.lines().filter(|l| !l.trim().is_empty()).collect();
    assert!(
        lines.len() >= 2,
        "session file {} has fewer than 2 records:\n{}",
        session_file.display(),
        body,
    );
    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first["kind"], "meta", "first record should be meta");
    let saw_turn = lines
        .iter()
        .skip(1)
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .any(|v| v["kind"] == "turn");
    assert!(
        saw_turn,
        "expected at least one `turn` record in {}",
        session_file.display()
    );
}

fn first_session_file(root: &Path) -> Option<PathBuf> {
    fn walk(p: &Path) -> Option<PathBuf> {
        for e in std::fs::read_dir(p).ok()?.flatten() {
            let path = e.path();
            if path.is_dir() {
                if let Some(found) = walk(&path) {
                    return Some(found);
                }
            } else if path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                return Some(path);
            }
        }
        None
    }
    walk(&root.join("sessions"))
}
