//! Canonical record schema for distillation traces.
//!
//! Design goals:
//! - Lossless: preserve provider-native message structure (`raw`).
//! - Portable: a thin canonical view (`blocks`) is easy to project to
//!   OpenAI/ShareGPT/Anthropic export formats.
//! - Append-only: each record is a single JSON object on its own line (JSONL).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Claude,
    Codex,
    Openai,
    Anthropic,
    Other,
}

impl Provider {
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "claude" | "anthropic" => Provider::Claude,
            "codex" | "openai" => Provider::Codex,
            _ => Provider::Other,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Provider::Claude => "claude",
            Provider::Codex => "codex",
            Provider::Openai => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Other => "other",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single content block within a turn.
///
/// Mirrors Anthropic's block model (text/image/tool_use/tool_result). For
/// OpenAI-style plain-string content we emit a single `text` block.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct ContentBlock {
    /// "text" | "image" | "tool_use" | "tool_result" | "reasoning" | ...
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    // tool_use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_input: Option<Value>,
    // tool_result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    /// Catch-all passthrough for fields we don't model explicitly (image_url, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

impl ContentBlock {
    pub fn text(t: impl Into<String>) -> Self {
        Self {
            kind: "text".into(),
            text: Some(t.into()),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Turn {
    pub role: Role,
    #[serde(default)]
    pub blocks: Vec<ContentBlock>,
    /// Provider-native payload preserved verbatim for lossless replay.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct Usage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SessionMeta {
    pub session_id: String,
    pub provider: Provider,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// ISO-8601 (RFC 3339) UTC.
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum RecordKind {
    Meta,
    Turn,
    Usage,
    End,
}

/// One line in the per-session JSONL file.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TurnRecord {
    pub kind: RecordKind,
    pub ts: String,
    pub session_id: String,
    pub seq: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn: Option<Turn>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<SessionMeta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}
