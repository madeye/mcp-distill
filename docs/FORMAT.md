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

### Storage knobs (size matters)

LLM traces are huge (1M-token contexts ≈ MBs of JSON). Two knobs, both
opt-out / opt-in via env or `StoreOptions`:

| knob | env var | default | purpose |
| ---- | ------- | ------- | ------- |
| `keep_raw` | `MCP_DISTILL_KEEP_RAW=1` | **off** | retain the provider-native `raw` field alongside the canonical `blocks` view |
| `compression` | `MCP_DISTILL_COMPRESSION=zstd` | `none` | per-line zstd frame; file becomes `<id>.jsonl.zst` |
| `zstd_level` | `MCP_DISTILL_ZSTD_LEVEL=3` | `3` | 1..=22 |

Measured on a synthetic corpus of 4 sessions × 40 turns × ~13 KB/turn
(~3 MB raw baseline), `cargo run --release --example storage_bench`:

| config | size | vs baseline | write time |
| ------ | ----:| -----------:| ----------:|
| raw=on,  none     | 3.09 MB | 1.00× | 50 ms |
| raw=off, none     | 1.11 MB | 0.36× | 46 ms |
| raw=on,  zstd-3   | 484 KB  | 0.15× | 71 ms |
| **raw=off, zstd-3** | **287 KB** | **0.09×** | 77 ms |
| raw=off, zstd-9   | 251 KB  | 0.08× | 128 ms |
| raw=off, zstd-19  | 236 KB  | 0.07× | 648 ms |

Recommended for production capture: `keep_raw=false` (default) +
`MCP_DISTILL_COMPRESSION=zstd` at level 3 — ~11× smaller with negligible
write overhead. Zstd frames are concatenable, so append-only writes stay
crash-safe at the line boundary. Read paths auto-detect `.jsonl` vs
`.jsonl.zst` for the same `session_id`.

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

### Using exports with Unsloth

The `openai_chat` export drops straight into Unsloth's SFT path; `sharegpt`
needs one normalize call.

```python
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

# A) openai_chat — already in Unsloth's preferred shape
ds = load_dataset("json", data_files="exports/openai_chat-*.jsonl", split="train")

# B) sharegpt — one extra normalize step
ds = load_dataset("json", data_files="exports/sharegpt-*.jsonl", split="train")
ds = standardize_sharegpt(ds)  # rewrites {from,value} -> {role,content}
```

Then pass `ds` to `SFTTrainer` and let `apply_chat_template` handle formatting.

Two gotchas to watch for:

1. **Assistant turns with only `tool_calls`** carry `content: null` in the
   `openai_chat` export. Modern templates (Llama 3.1+, Qwen 2.5, Mistral,
   DeepSeek-R1) handle this; older templates (some Phi/Gemma variants) error
   on null content. If yours does, `dataset.map(...)` to coerce null → `""`.
2. **`role: "tool"` messages** require a tool-aware chat template. If your
   target model lacks tool support, filter those rows out:
   ```python
   ds = ds.filter(lambda r: all(m["role"] != "tool" for m in r["messages"]))
   ```

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
