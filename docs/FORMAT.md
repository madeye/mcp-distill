# Storage format for distillation traces

Goal: capture *every* prompt, context block, tool call, tool result, and
response — losslessly — and make it cheap to project that into whatever
fine-tuning format the target trainer expects.

## Format choice: two-layer JSONL

We store **two artifacts**:

### 1. Raw layer — `sessions/YYYY/MM/DD/<session_id>.jsonl`

One JSON object per line. Append-only. Each line is a `TurnRecord`:

```json
{"kind":"meta","ts":"...","session_id":"...","seq":0,
 "meta":{"provider":"claude","model":"claude-opus-4-7","tags":[],"metadata":{}}}
{"kind":"turn","ts":"...","session_id":"...","seq":1,
 "turn":{"role":"user","blocks":[{"type":"text","text":"..."}],"raw":{...}}}
{"kind":"turn","ts":"...","session_id":"...","seq":2,
 "turn":{"role":"assistant",
         "blocks":[{"type":"text","text":"..."},
                   {"type":"tool_use","tool_call_id":"c1","tool_name":"x","tool_input":{...}}],
         "raw":{...}}}
{"kind":"usage","ts":"...","seq":3,"usage":{"input_tokens":..,"output_tokens":..}}
{"kind":"end","ts":"...","seq":4}
```

Why JSONL:
- streaming-friendly (append on every turn — no rewrites)
- crash-safe (a torn final line is a single bad record, not a corrupt file)
- works with `jq`, `duckdb`, `polars`, `datasets.load_dataset("json", ...)`
- diff-friendly in git/object stores

Why store **both** `blocks` (canonical) and `raw` (provider-native):
- `raw` is the ground truth — never lose a field from the upstream API
- `blocks` is the projection downstream tooling actually consumes; cheap to recompute, but pre-computing means exporters never need provider-specific code

### 2. Export layer — `exports/<format>-<timestamp>.jsonl`

Generated on demand from the raw layer via `export_dataset`. Three targets:

| Format         | Shape per line                                               | Used by                     |
| -------------- | ------------------------------------------------------------ | --------------------------- |
| `openai_chat`  | `{"messages":[{"role","content","tool_calls"?},…]}`          | OpenAI / Together / vLLM SFT |
| `sharegpt`     | `{"conversations":[{"from":"human"\|"gpt"\|"system","value"}]}` | Axolotl, LLaMA-Factory, HF datasets |
| `anthropic`    | `{"system":...,"messages":[{"role","content":[blocks]}]}`     | Anthropic-style trainers, replay |

Default recommendation for distillation into a small model: **`openai_chat`**.
It's the broadest target across open-source SFT trainers (TRL, Axolotl,
LLaMA-Factory, vLLM, Unsloth) and preserves tool-calling natively.

## Why not Parquet / Arrow / SQLite?

- Distillation pipelines almost universally consume JSON/JSONL. Parquet would
  require a conversion step and lose human-inspectability.
- SQLite gives indexed queries but mutates a single file → bad for append-only
  capture from multiple processes and bad for object-store sync.
- JSONL → Parquet is one `duckdb COPY` away when scale demands it, so we don't
  pre-pay that cost.

## Capture model

Two ergonomic paths:

1. **Streaming** — `start_session` → many `append_turn` → `append_usage` → `end_session`.
   Use this from instrumentation/middleware that sees turns as they happen.
2. **One-shot** — `record_interaction(provider, model, request_messages, response)`.
   Use this from a wrapper that already has the full request and response in hand.

Provider adapters normalize Claude (Anthropic Messages) and Codex (OpenAI Chat
Completions) on the way in, so downstream code never branches on provider.
