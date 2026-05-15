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
# sets default_tools_approval_mode = "approve" so codex doesn't cancel our calls
# `install codex --force` overwrites a differing existing entry
# `uninstall codex` removes it
```

Wire it into **Claude Code** automatically (requires `claude` CLI):

```bash
./target/release/mcp-distill install claude \
  --store-root ~/.mcp-distill --compression zstd
# shells out to `claude mcp add -s user` (writes ~/.claude.json, user scope)
# `install claude --force` removes any existing entry with the same name first
# `uninstall claude` runs `claude mcp remove -s user distill`
```

## MCP tools exposed

| Tool                 | What it does                                                                  |
| -------------------- | ----------------------------------------------------------------------------- |
| `recording_status`   | Whether a session is currently recording, and which one.                      |
| `stop_recording`     | Halt the auto-started session (writes an `end` record).                       |
| `start_recording`    | Open a fresh session and make it current.                                     |
| `start_session`      | Open a session and make it current. Returns `session_id`.                     |
| `append_turn`        | Append a provider-native message. `session_id` optional → uses current.       |
| `append_usage`       | Record token usage. `session_id` optional → uses current.                     |
| `end_session`        | Mark a session ended (current if `session_id` omitted).                       |
| `record_interaction` | One-shot: start + write all messages + response in a single call.             |
| `list_sessions`      | List captured sessions.                                                       |
| `export_dataset`     | Emit JSONL in `openai_chat` / `sharegpt` / `anthropic` format.                |
| `stats`              | Session/turn counts, providers seen, storage root.                            |

**Auto-recording**: the server opens a session at launch and routes
implicit `append_turn` calls to it. The provider for that session comes
from `MCP_DISTILL_AUTO_PROVIDER` (`install {codex,claude}` seeds this).
Disable with `MCP_DISTILL_AUTO_RECORD=0`. Manually halt with
`stop_recording`; resume with `start_recording`.

## Storage

```
$MCP_DISTILL_ROOT (default: ~/.mcp-distill)/
  sessions/YYYY/MM/DD/<session_id>.jsonl   # raw, lossless, append-only
  index.jsonl                              # session index (latest meta wins)
  exports/                                 # generated SFT datasets
```

See [`docs/FORMAT.md`](docs/FORMAT.md) for the full schema and the rationale
behind the format choice.

## Training on the exports (Unsloth)

The `openai_chat` export drops straight into Unsloth; `sharegpt` needs one
normalize call.

```python
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

# A) openai_chat — already in Unsloth's preferred shape
ds = load_dataset("json", data_files="exports/openai_chat-*.jsonl", split="train")

# B) sharegpt — one extra normalize step
ds = load_dataset("json", data_files="exports/sharegpt-*.jsonl", split="train")
ds = standardize_sharegpt(ds)
```

Two gotchas: assistant turns with only `tool_calls` carry `content: null`
(modern templates handle it; some older Phi/Gemma templates need null → `""`),
and `role: "tool"` rows need a tool-aware chat template (Llama 3.1+, Qwen 2.5,
DeepSeek-R1). Filter them out otherwise. Details in
[`docs/FORMAT.md`](docs/FORMAT.md#using-exports-with-unsloth).
