//! Codex / OpenAI Chat Completions adapter.
//!
//! Input shape:
//! ```json
//! {"role": "system"|"user"|"assistant"|"tool",
//!  "content": "..." | [{"type":"text","text":"..."}, {"type":"image_url",...}] | null,
//!  "tool_calls": [{"id":"c1","type":"function",
//!                  "function":{"name":"...","arguments":"<json string>"}}],
//!  "tool_call_id": "c1"  // when role == "tool"
//! }
//! ```

use serde_json::Value;

use crate::schema::{ContentBlock, Role, Turn};

fn role_from_str(s: &str) -> Role {
    match s {
        "system" => Role::System,
        "assistant" => Role::Assistant,
        "tool" | "function" => Role::Tool,
        _ => Role::User,
    }
}

fn blocks_from_content(content: &Value) -> Vec<ContentBlock> {
    if content.is_null() {
        return Vec::new();
    }
    if let Some(s) = content.as_str() {
        if s.is_empty() {
            return Vec::new();
        }
        return vec![ContentBlock::text(s)];
    }
    let Some(arr) = content.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .map(|b| {
            let kind = b.get("type").and_then(Value::as_str).unwrap_or("unknown");
            match kind {
                "text" | "input_text" | "output_text" => ContentBlock {
                    kind: "text".into(),
                    text: b.get("text").and_then(Value::as_str).map(String::from),
                    ..Default::default()
                },
                "image_url" | "input_image" | "image" => ContentBlock {
                    kind: "image".into(),
                    extra: Some(b.clone()),
                    ..Default::default()
                },
                other => ContentBlock {
                    kind: other.to_string(),
                    extra: Some(b.clone()),
                    ..Default::default()
                },
            }
        })
        .collect()
}

pub fn message_to_turn(message: &Value) -> Turn {
    let role_str = message
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or("user");
    let role = role_from_str(role_str);
    let content = message.get("content").cloned().unwrap_or(Value::Null);
    let mut blocks = blocks_from_content(&content);

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tc in tool_calls {
            let id = tc.get("id").and_then(Value::as_str).map(String::from);
            let func = tc.get("function").cloned().unwrap_or(Value::Null);
            let name = func.get("name").and_then(Value::as_str).map(String::from);
            let args_val = func.get("arguments").cloned().unwrap_or(Value::Null);
            let tool_input = match args_val {
                Value::String(ref s) => serde_json::from_str::<Value>(s)
                    .unwrap_or_else(|_| serde_json::json!({"_raw": s})),
                other => other,
            };
            blocks.push(ContentBlock {
                kind: "tool_use".into(),
                tool_call_id: id,
                tool_name: name,
                tool_input: Some(tool_input),
                ..Default::default()
            });
        }
    }

    if matches!(role, Role::Tool) {
        let tool_call_id = message
            .get("tool_call_id")
            .and_then(Value::as_str)
            .map(String::from);
        let tool_result = message.get("content").cloned();
        // The tool result *is* the message content; drop the duplicate text block.
        blocks.retain(|b| b.kind != "text");
        blocks.push(ContentBlock {
            kind: "tool_result".into(),
            tool_call_id,
            tool_result,
            ..Default::default()
        });
    }

    Turn {
        role,
        blocks,
        raw: Some(message.clone()),
    }
}

pub fn response_to_turn(response: &Value) -> Turn {
    // Accept either a full Chat Completions response or a bare message.
    let msg = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .and_then(|c| c.get("message"))
        .cloned()
        .unwrap_or_else(|| response.clone());
    let mut msg_obj = msg;
    if let Some(obj) = msg_obj.as_object_mut() {
        obj.insert("role".into(), Value::String("assistant".into()));
    }
    let mut turn = message_to_turn(&msg_obj);
    turn.raw = Some(response.clone());
    turn
}
