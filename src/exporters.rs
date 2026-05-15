//! Export canonical turns to common SFT/distillation formats.
//!
//! - `openai_chat` -> `{"messages":[{"role":...,"content":...,"tool_calls":?},...]}` per session.
//!   Used by OpenAI fine-tuning, Together, vLLM, TRL, Axolotl, LLaMA-Factory.
//! - `sharegpt`    -> `{"conversations":[{"from":"human"|"gpt"|"system","value":...}]}`
//!   Common HuggingFace SFT shape.
//! - `anthropic`   -> `{"system":?,"messages":[{"role":"user"|"assistant","content":[blocks]}]}`
//!   Native Anthropic Messages shape (lossless replay).

use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::Result;
use serde_json::{json, Map, Value};

use crate::schema::{RecordKind, Role, Turn, TurnRecord};

#[derive(Debug, Clone, Copy)]
pub enum Format {
    OpenAiChat,
    ShareGpt,
    Anthropic,
}

impl Format {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "openai_chat" | "openai" | "chatml" => Some(Format::OpenAiChat),
            "sharegpt" => Some(Format::ShareGpt),
            "anthropic" | "claude" => Some(Format::Anthropic),
            _ => None,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Format::OpenAiChat => "openai_chat",
            Format::ShareGpt => "sharegpt",
            Format::Anthropic => "anthropic",
        }
    }
}

fn turn_text(turn: &Turn) -> String {
    let mut out = String::new();
    for b in &turn.blocks {
        if b.kind == "text" {
            if let Some(t) = &b.text {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(t);
            }
        }
    }
    out
}

pub fn to_openai_chat(turns: &[Turn], system: Option<&str>) -> Value {
    let mut msgs: Vec<Value> = Vec::new();
    if let Some(s) = system {
        msgs.push(json!({"role": "system", "content": s}));
    }
    for t in turns {
        if matches!(t.role, Role::Tool) {
            for b in &t.blocks {
                if b.kind == "tool_result" {
                    let content = match &b.tool_result {
                        Some(Value::String(s)) => Value::String(s.clone()),
                        Some(other) => Value::String(other.to_string()),
                        None => Value::String(String::new()),
                    };
                    msgs.push(json!({
                        "role": "tool",
                        "tool_call_id": b.tool_call_id.clone().unwrap_or_default(),
                        "content": content,
                    }));
                }
            }
            continue;
        }

        let mut text_parts: Vec<&str> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        for b in &t.blocks {
            if b.kind == "text" {
                if let Some(t) = &b.text {
                    if !t.is_empty() {
                        text_parts.push(t.as_str());
                    }
                }
            } else if b.kind == "tool_use" {
                tool_calls.push(json!({
                    "id": b.tool_call_id.clone().unwrap_or_default(),
                    "type": "function",
                    "function": {
                        "name": b.tool_name.clone().unwrap_or_default(),
                        "arguments": serde_json::to_string(
                            b.tool_input.as_ref().unwrap_or(&Value::Object(Map::new()))
                        ).unwrap_or_else(|_| "{}".into()),
                    },
                }));
            }
        }
        let content = if text_parts.is_empty() {
            Value::Null
        } else {
            Value::String(text_parts.join("\n"))
        };
        let role = match t.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        let mut obj = serde_json::Map::new();
        obj.insert("role".into(), Value::String(role.into()));
        obj.insert("content".into(), content);
        if !tool_calls.is_empty() {
            obj.insert("tool_calls".into(), Value::Array(tool_calls));
        }
        msgs.push(Value::Object(obj));
    }
    json!({"messages": msgs})
}

pub fn to_sharegpt(turns: &[Turn], system: Option<&str>) -> Value {
    let mut convs: Vec<Value> = Vec::new();
    if let Some(s) = system {
        convs.push(json!({"from": "system", "value": s}));
    }
    for t in turns {
        let from = match t.role {
            Role::User => "human",
            Role::Assistant => "gpt",
            Role::System => "system",
            Role::Tool => "tool",
        };
        let mut parts: Vec<String> = Vec::new();
        for b in &t.blocks {
            if b.kind == "text" {
                if let Some(text) = &b.text {
                    if !text.is_empty() {
                        parts.push(text.clone());
                    }
                }
            } else if b.kind == "tool_use" {
                let args = serde_json::to_string(
                    b.tool_input.as_ref().unwrap_or(&Value::Object(Map::new())),
                )
                .unwrap_or_else(|_| "{}".into());
                parts.push(format!(
                    "<tool_call name=\"{}\">{}</tool_call>",
                    b.tool_name.clone().unwrap_or_default(),
                    args,
                ));
            } else if b.kind == "tool_result" {
                let tr = match &b.tool_result {
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => other.to_string(),
                    None => String::new(),
                };
                parts.push(format!("<tool_result>{tr}</tool_result>"));
            }
        }
        if !parts.is_empty() {
            convs.push(json!({"from": from, "value": parts.join("\n")}));
        }
    }
    json!({"conversations": convs})
}

pub fn to_anthropic(turns: &[Turn], system: Option<&str>) -> Value {
    let mut msgs: Vec<Value> = Vec::new();
    for t in turns {
        if matches!(t.role, Role::System) {
            continue;
        }
        let role = if matches!(t.role, Role::Assistant) {
            "assistant"
        } else {
            "user"
        };
        let mut content: Vec<Value> = Vec::new();
        for b in &t.blocks {
            match b.kind.as_str() {
                "text" => content.push(json!({
                    "type": "text",
                    "text": b.text.clone().unwrap_or_default(),
                })),
                "tool_use" => content.push(json!({
                    "type": "tool_use",
                    "id": b.tool_call_id.clone().unwrap_or_default(),
                    "name": b.tool_name.clone().unwrap_or_default(),
                    "input": b.tool_input.clone().unwrap_or(Value::Object(Map::new())),
                })),
                "tool_result" => content.push(json!({
                    "type": "tool_result",
                    "tool_use_id": b.tool_call_id.clone().unwrap_or_default(),
                    "content": b.tool_result.clone().unwrap_or(Value::Null),
                    "is_error": b.is_error.unwrap_or(false),
                })),
                _ => content.push(b.extra.clone().unwrap_or(Value::Null)),
            }
        }
        msgs.push(json!({"role": role, "content": content}));
    }
    let mut out = json!({"messages": msgs});
    if let Some(s) = system {
        out.as_object_mut()
            .unwrap()
            .insert("system".into(), Value::String(s.to_string()));
    }
    out
}

/// Convert a session's records into one export row.
pub fn session_to_export(records: &[TurnRecord], fmt: Format) -> Value {
    let mut system: Option<String> = None;
    let mut turns: Vec<Turn> = Vec::new();
    for r in records {
        if !matches!(r.kind, RecordKind::Turn) {
            continue;
        }
        let Some(turn) = &r.turn else { continue };
        if matches!(turn.role, Role::System) && system.is_none() {
            let t = turn_text(turn);
            if !t.is_empty() {
                system = Some(t);
            }
            continue;
        }
        turns.push(turn.clone());
    }
    match fmt {
        Format::OpenAiChat => to_openai_chat(&turns, system.as_deref()),
        Format::ShareGpt => to_sharegpt(&turns, system.as_deref()),
        Format::Anthropic => to_anthropic(&turns, system.as_deref()),
    }
}

pub fn write_jsonl<I: IntoIterator<Item = Value>>(rows: I, path: &Path) -> Result<usize> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = fs::File::create(path)?;
    let mut n = 0;
    for row in rows {
        let s = serde_json::to_string(&row)?;
        f.write_all(s.as_bytes())?;
        f.write_all(b"\n")?;
        n += 1;
    }
    Ok(n)
}
