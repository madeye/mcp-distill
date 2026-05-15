# mcp-distill

An MCP server (in Rust) that records prompts, context, and responses from
Claude (Anthropic Messages API) and Codex (OpenAI Chat Completions API) for
**small-model distillation**.

Captures every turn losslessly to JSONL, then exports to the format your
trainer wants.

## Quick start

```bash
cargo build --release
./target/release/mcp-distill where        # show storage root
./target/release/mcp-distill serve        # run as MCP server over stdio
```

Wire it into **codex** automatically:

```bash
./target/release/mcp-distill install codex \
  --store-root ~/.mcp-distill --compression zstd
# adds [mcp_servers.distill] to ~/.codex/config.toml (preserves other settings)
# `install codex --force` overwrites a differing existing entry
# `uninstall codex` removes it
```

Wire it into Claude Code manually:

```jsonc
// .claude/settings.json
{
  "mcpServers": {
    "distill": { "command": "/abs/path/to/mcp-distill", "args": ["serve"] }
  }
}
```

## MCP tools exposed

| Tool                 | What it does                                                      |
| -------------------- | ----------------------------------------------------------------- |
| `start_session`      | Open a session for `claude` or `codex`. Returns `session_id`.     |
| `append_turn`        | Append a provider-native message (request or response).           |
| `append_usage`       | Record token usage for the most recent assistant turn.            |
| `end_session`        | Mark a session ended.                                             |
| `record_interaction` | One-shot: start + write all messages + response in a single call. |
| `list_sessions`      | List captured sessions.                                           |
| `export_dataset`     | Emit JSONL in `openai_chat` / `sharegpt` / `anthropic` format.    |
| `stats`              | Session/turn counts, providers seen, storage root.                |

## Storage

```
$MCP_DISTILL_ROOT (default: ~/.mcp-distill)/
  sessions/YYYY/MM/DD/<session_id>.jsonl   # raw, lossless, append-only
  index.jsonl                              # session index (latest meta wins)
  exports/                                 # generated SFT datasets
```

See [`docs/FORMAT.md`](docs/FORMAT.md) for the full schema and the rationale
behind the format choice.
