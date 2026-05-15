//! Anthropic / Claude Messages adapter.
//!
//! Input shape:
//! ```json
//! {"role": "user"|"assistant",
//!  "content": "string" | [
//!     {"type":"text","text":"..."},
//!     {"type":"tool_use","id":"...","name":"...","input":{...}},
//!     {"type":"tool_result","tool_use_id":"...","content":...,"is_error":bool}
//!  ]}
//! ```

use serde_json::Value;

use crate::schema::{ContentBlock, Role, Turn};

fn role_from(value: &Value) -> Role {
    match value.get("role").and_then(Value::as_str).unwrap_or("user") {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

fn blocks_from_content(content: &Value) -> Vec<ContentBlock> {
    if let Some(s) = content.as_str() {
        return vec![ContentBlock::text(s)];
    }
    let Some(arr) = content.as_array() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(arr.len());
    for b in arr {
        let kind = b.get("type").and_then(Value::as_str).unwrap_or("unknown");
        match kind {
            "text" => out.push(ContentBlock {
                kind: "text".into(),
                text: b.get("text").and_then(Value::as_str).map(String::from),
                ..Default::default()
            }),
            "tool_use" => out.push(ContentBlock {
                kind: "tool_use".into(),
                tool_call_id: b.get("id").and_then(Value::as_str).map(String::from),
                tool_name: b.get("name").and_then(Value::as_str).map(String::from),
                tool_input: b.get("input").cloned(),
                ..Default::default()
            }),
            "tool_result" => out.push(ContentBlock {
                kind: "tool_result".into(),
                tool_call_id: b
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .map(String::from),
                tool_result: b.get("content").cloned(),
                is_error: b.get("is_error").and_then(Value::as_bool),
                ..Default::default()
            }),
            other => out.push(ContentBlock {
                kind: other.to_string(),
                extra: Some(b.clone()),
                ..Default::default()
            }),
        }
    }
    out
}

pub fn message_to_turn(message: &Value) -> Turn {
    let content = message.get("content").cloned().unwrap_or(Value::Null);
    Turn {
        role: role_from(message),
        blocks: blocks_from_content(&content),
        raw: Some(message.clone()),
    }
}

pub fn response_to_turn(response: &Value) -> Turn {
    let content = response.get("content").cloned().unwrap_or(Value::Null);
    Turn {
        role: Role::Assistant,
        blocks: blocks_from_content(&content),
        raw: Some(response.clone()),
    }
}
