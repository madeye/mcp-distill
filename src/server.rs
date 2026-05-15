//! MCP server: tools to record interactions and export distillation datasets.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use parking_lot::RwLock;
use std::future::Future;

use rmcp::{
    handler::server::tool::ToolRouter,
    model::{
        CallToolResult, Content, Implementation, ProtocolVersion, ServerCapabilities, ServerInfo,
    },
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router, Error as McpError, ServerHandler,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::adapters;
use crate::exporters::{session_to_export, write_jsonl, Format};
use crate::schema::{Provider, RecordKind, SessionMeta, Turn, TurnRecord, Usage};
use crate::storage::{now_rfc3339, Store};

#[derive(Clone)]
pub struct DistillServer {
    store: Arc<Store>,
    /// Currently-active session for implicit recording (`append_turn` without
    /// an explicit `session_id` lands here). `None` means recording is paused.
    current: Arc<RwLock<Option<String>>>,
    /// Provider used when auto-starting a session. Set from
    /// `MCP_DISTILL_AUTO_PROVIDER` (the installers seed this per client).
    auto_provider: Provider,
    /// Default model for auto-started sessions, from `MCP_DISTILL_AUTO_MODEL`.
    auto_model: Option<String>,
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StartSessionArgs {
    /// "claude" | "codex" | "openai" | "anthropic" | "other"
    pub provider: String,
    pub model: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AppendTurnArgs {
    /// Session to append to. Omit to use the currently-recording session.
    pub session_id: Option<String>,
    /// Provider hint for adapter selection. Omit to use the server's
    /// configured auto-provider.
    pub provider: Option<String>,
    /// Provider-native message (Anthropic Messages or OpenAI Chat Completions shape).
    pub message: Value,
    /// Set to true if `message` is a model *response* rather than a request message.
    #[serde(default)]
    pub is_response: bool,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct AppendUsageArgs {
    pub session_id: Option<String>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_creation_tokens: Option<u64>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EndSessionArgs {
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StartRecordingArgs {
    /// "claude" | "codex" | "openai" | "anthropic" | "other". Defaults to the
    /// server's auto-provider.
    pub provider: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
}

#[derive(Debug, Default, Deserialize, schemars::JsonSchema)]
pub struct EmptyArgs {}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecordInteractionArgs {
    pub provider: String,
    pub model: Option<String>,
    pub request_messages: Vec<Value>,
    pub response: Value,
    pub system: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, Value>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExportDatasetArgs {
    /// "openai_chat" | "sharegpt" | "anthropic"
    #[serde(default = "default_format")]
    pub format: String,
    pub session_ids: Option<Vec<String>>,
    pub output_path: Option<String>,
    pub tag_filter: Option<Vec<String>>,
}

fn default_format() -> String {
    "openai_chat".into()
}

fn ok_json(v: Value) -> Result<CallToolResult, McpError> {
    let s = serde_json::to_string(&v).unwrap_or_else(|_| "{}".into());
    Ok(CallToolResult::success(vec![Content::text(s)]))
}

fn err(msg: impl Into<String>) -> McpError {
    McpError::internal_error(msg.into(), None)
}

/// Whether the server should auto-start a recording session at launch.
fn auto_record_enabled() -> bool {
    !matches!(
        std::env::var("MCP_DISTILL_AUTO_RECORD").as_deref(),
        Ok("0") | Ok("false") | Ok("off") | Ok("no")
    )
}

#[tool_router]
impl DistillServer {
    pub fn new(root: PathBuf) -> Result<Self> {
        let store = Arc::new(Store::new(root)?);
        let auto_provider = std::env::var("MCP_DISTILL_AUTO_PROVIDER")
            .ok()
            .map(|s| Provider::from_str_loose(&s))
            .unwrap_or(Provider::Other);
        let auto_model = std::env::var("MCP_DISTILL_AUTO_MODEL").ok();

        let current = Arc::new(RwLock::new(None::<String>));
        if auto_record_enabled() {
            let auto_tags = std::env::var("MCP_DISTILL_AUTO_TAGS")
                .ok()
                .map(|s| {
                    s.split(',')
                        .map(|t| t.trim().to_string())
                        .filter(|t| !t.is_empty())
                        .collect()
                })
                .unwrap_or_default();
            let sid = uuid::Uuid::new_v4().simple().to_string();
            let meta = SessionMeta {
                session_id: sid.clone(),
                provider: auto_provider.clone(),
                model: auto_model.clone(),
                started_at: Utc::now().to_rfc3339(),
                ended_at: None,
                tags: auto_tags,
                metadata: Default::default(),
            };
            store.write_meta(&meta)?;
            *current.write() = Some(sid.clone());
            tracing::info!(session_id = %sid, provider = ?auto_provider, "auto-started recording session");
        } else {
            tracing::info!("auto-recording disabled (MCP_DISTILL_AUTO_RECORD=0)");
        }

        Ok(Self {
            store,
            current,
            auto_provider,
            auto_model,
            tool_router: Self::tool_router(),
        })
    }

    fn require_current(&self) -> Result<String, McpError> {
        self.current
            .read()
            .clone()
            .ok_or_else(|| err("recording is stopped; call start_recording first"))
    }

    fn resolve_session(&self, explicit: Option<String>) -> Result<String, McpError> {
        match explicit {
            Some(s) => Ok(s),
            None => self.require_current(),
        }
    }

    fn resolve_provider(&self, explicit: Option<String>) -> Provider {
        explicit
            .map(|s| Provider::from_str_loose(&s))
            .unwrap_or_else(|| self.auto_provider.clone())
    }

    #[tool(
        description = "Start a new recording session and make it the currently-recording one. \
                       Returns the session_id and trace file path."
    )]
    async fn start_session(
        &self,
        rmcp::handler::server::tool::Parameters(args): rmcp::handler::server::tool::Parameters<
            StartSessionArgs,
        >,
    ) -> Result<CallToolResult, McpError> {
        let sid = args
            .session_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
        let meta = SessionMeta {
            session_id: sid.clone(),
            provider: Provider::from_str_loose(&args.provider),
            model: args.model,
            started_at: Utc::now().to_rfc3339(),
            ended_at: None,
            tags: args.tags,
            metadata: args.metadata,
        };
        let path = self
            .store
            .write_meta(&meta)
            .map_err(|e| err(e.to_string()))?;
        *self.current.write() = Some(sid.clone());
        ok_json(json!({"session_id": sid, "path": path.to_string_lossy()}))
    }

    #[tool(
        description = "Status of the implicit recording session: {recording: bool, session_id?, root}."
    )]
    async fn recording_status(
        &self,
        _: rmcp::handler::server::tool::Parameters<EmptyArgs>,
    ) -> Result<CallToolResult, McpError> {
        let cur = self.current.read().clone();
        ok_json(json!({
            "recording": cur.is_some(),
            "session_id": cur,
            "root": self.store.root.to_string_lossy(),
        }))
    }

    #[tool(
        description = "Stop the currently-recording session (writes an `end` record). \
                       Subsequent append_turn calls without an explicit session_id will fail \
                       until start_recording or start_session is called."
    )]
    async fn stop_recording(
        &self,
        _: rmcp::handler::server::tool::Parameters<EmptyArgs>,
    ) -> Result<CallToolResult, McpError> {
        let prev = self.current.write().take();
        if let Some(sid) = &prev {
            let rec = TurnRecord {
                kind: RecordKind::End,
                ts: now_rfc3339(),
                session_id: sid.clone(),
                seq: self.store.next_seq(sid),
                turn: None,
                meta: None,
                usage: None,
            };
            self.store
                .write_record(sid, &rec)
                .map_err(|e| err(e.to_string()))?;
        }
        ok_json(json!({"ok": true, "stopped_session_id": prev}))
    }

    #[tool(description = "Resume recording: opens a new session and makes it the current one.")]
    async fn start_recording(
        &self,
        rmcp::handler::server::tool::Parameters(args): rmcp::handler::server::tool::Parameters<
            StartRecordingArgs,
        >,
    ) -> Result<CallToolResult, McpError> {
        let provider = self.resolve_provider(args.provider);
        let model = args.model.or_else(|| self.auto_model.clone());
        let sid = uuid::Uuid::new_v4().simple().to_string();
        let meta = SessionMeta {
            session_id: sid.clone(),
            provider,
            model,
            started_at: Utc::now().to_rfc3339(),
            ended_at: None,
            tags: args.tags,
            metadata: args.metadata,
        };
        let path = self
            .store
            .write_meta(&meta)
            .map_err(|e| err(e.to_string()))?;
        *self.current.write() = Some(sid.clone());
        ok_json(json!({"session_id": sid, "path": path.to_string_lossy()}))
    }

    #[tool(
        description = "Append one turn to a session. `message` is provider-native (Anthropic \
                       Messages or OpenAI Chat Completions). If session_id is omitted, the \
                       currently-recording session is used. Set is_response=true for model responses."
    )]
    async fn append_turn(
        &self,
        rmcp::handler::server::tool::Parameters(args): rmcp::handler::server::tool::Parameters<
            AppendTurnArgs,
        >,
    ) -> Result<CallToolResult, McpError> {
        let session_id = self.resolve_session(args.session_id)?;
        let provider = self.resolve_provider(args.provider);
        let turn = if args.is_response {
            adapters::response_to_turn(&provider, &args.message)
        } else {
            adapters::to_turn(&provider, &args.message)
        };
        let role = format!("{:?}", turn.role).to_lowercase();
        let blocks = turn.blocks.len();
        let rec = TurnRecord {
            kind: RecordKind::Turn,
            ts: now_rfc3339(),
            session_id: session_id.clone(),
            seq: self.store.next_seq(&session_id),
            turn: Some(turn),
            meta: None,
            usage: None,
        };
        self.store
            .write_record(&session_id, &rec)
            .map_err(|e| err(e.to_string()))?;
        ok_json(json!({"ok": true, "session_id": session_id, "role": role, "blocks": blocks}))
    }

    #[tool(
        description = "Record token usage for the most recent assistant turn. \
                          If session_id is omitted, uses the currently-recording session."
    )]
    async fn append_usage(
        &self,
        rmcp::handler::server::tool::Parameters(args): rmcp::handler::server::tool::Parameters<
            AppendUsageArgs,
        >,
    ) -> Result<CallToolResult, McpError> {
        let session_id = self.resolve_session(args.session_id)?;
        let usage = Usage {
            input_tokens: args.input_tokens,
            output_tokens: args.output_tokens,
            cache_read_tokens: args.cache_read_tokens,
            cache_creation_tokens: args.cache_creation_tokens,
        };
        let rec = TurnRecord {
            kind: RecordKind::Usage,
            ts: now_rfc3339(),
            session_id: session_id.clone(),
            seq: self.store.next_seq(&session_id),
            turn: None,
            meta: None,
            usage: Some(usage),
        };
        self.store
            .write_record(&session_id, &rec)
            .map_err(|e| err(e.to_string()))?;
        ok_json(json!({"ok": true, "session_id": session_id}))
    }

    #[tool(
        description = "Mark a specific session ended (or the current one if omitted). \
                          Does not change the currently-recording session unless it matches."
    )]
    async fn end_session(
        &self,
        rmcp::handler::server::tool::Parameters(args): rmcp::handler::server::tool::Parameters<
            EndSessionArgs,
        >,
    ) -> Result<CallToolResult, McpError> {
        let session_id = self.resolve_session(args.session_id)?;
        let rec = TurnRecord {
            kind: RecordKind::End,
            ts: now_rfc3339(),
            session_id: session_id.clone(),
            seq: self.store.next_seq(&session_id),
            turn: None,
            meta: None,
            usage: None,
        };
        self.store
            .write_record(&session_id, &rec)
            .map_err(|e| err(e.to_string()))?;
        let mut cur = self.current.write();
        if cur.as_deref() == Some(&session_id) {
            *cur = None;
        }
        ok_json(json!({"ok": true, "session_id": session_id}))
    }

    #[tool(
        description = "One-shot: start a session, write all request messages plus the assistant response, return session_id."
    )]
    async fn record_interaction(
        &self,
        rmcp::handler::server::tool::Parameters(args): rmcp::handler::server::tool::Parameters<
            RecordInteractionArgs,
        >,
    ) -> Result<CallToolResult, McpError> {
        let provider = Provider::from_str_loose(&args.provider);
        let sid = uuid::Uuid::new_v4().simple().to_string();
        let meta = SessionMeta {
            session_id: sid.clone(),
            provider: provider.clone(),
            model: args.model,
            started_at: Utc::now().to_rfc3339(),
            ended_at: None,
            tags: args.tags,
            metadata: args.metadata,
        };
        self.store
            .write_meta(&meta)
            .map_err(|e| err(e.to_string()))?;

        if let Some(sys) = args.system {
            let turn = Turn {
                role: crate::schema::Role::System,
                blocks: vec![crate::schema::ContentBlock::text(sys)],
                raw: None,
            };
            self.store
                .write_record(
                    &sid,
                    &TurnRecord {
                        kind: RecordKind::Turn,
                        ts: now_rfc3339(),
                        session_id: sid.clone(),
                        seq: self.store.next_seq(&sid),
                        turn: Some(turn),
                        meta: None,
                        usage: None,
                    },
                )
                .map_err(|e| err(e.to_string()))?;
        }
        for m in &args.request_messages {
            let turn = adapters::to_turn(&provider, m);
            self.store
                .write_record(
                    &sid,
                    &TurnRecord {
                        kind: RecordKind::Turn,
                        ts: now_rfc3339(),
                        session_id: sid.clone(),
                        seq: self.store.next_seq(&sid),
                        turn: Some(turn),
                        meta: None,
                        usage: None,
                    },
                )
                .map_err(|e| err(e.to_string()))?;
        }
        let resp_turn = adapters::response_to_turn(&provider, &args.response);
        self.store
            .write_record(
                &sid,
                &TurnRecord {
                    kind: RecordKind::Turn,
                    ts: now_rfc3339(),
                    session_id: sid.clone(),
                    seq: self.store.next_seq(&sid),
                    turn: Some(resp_turn),
                    meta: None,
                    usage: None,
                },
            )
            .map_err(|e| err(e.to_string()))?;
        ok_json(json!({"session_id": sid}))
    }

    #[tool(description = "List all known sessions.")]
    async fn list_sessions(&self) -> Result<CallToolResult, McpError> {
        let sessions = self.store.list_sessions().map_err(|e| err(e.to_string()))?;
        ok_json(json!({"sessions": sessions}))
    }

    #[tool(
        description = "Export captured sessions to a fine-tuning dataset (JSONL). format: openai_chat | sharegpt | anthropic."
    )]
    async fn export_dataset(
        &self,
        rmcp::handler::server::tool::Parameters(args): rmcp::handler::server::tool::Parameters<
            ExportDatasetArgs,
        >,
    ) -> Result<CallToolResult, McpError> {
        let fmt = Format::parse(&args.format)
            .ok_or_else(|| err(format!("unknown format {}", args.format)))?;
        let mut sessions = self.store.list_sessions().map_err(|e| err(e.to_string()))?;
        if let Some(tags) = args.tag_filter {
            sessions.retain(|s| s.tags.iter().any(|t| tags.contains(t)));
        }
        if let Some(ids) = args.session_ids {
            sessions.retain(|s| ids.contains(&s.session_id));
        }
        let mut rows: Vec<Value> = Vec::new();
        for s in &sessions {
            let recs = self
                .store
                .iter_session(&s.session_id)
                .map_err(|e| err(e.to_string()))?;
            if recs.is_empty() {
                continue;
            }
            rows.push(session_to_export(&recs, fmt));
        }
        let out_path = match args.output_path {
            Some(p) => PathBuf::from(p),
            None => self.store.root.join("exports").join(format!(
                "{}-{}.jsonl",
                fmt.label(),
                Utc::now().format("%Y%m%dT%H%M%S")
            )),
        };
        let n = write_jsonl(rows, &out_path).map_err(|e| err(e.to_string()))?;
        ok_json(json!({"path": out_path.to_string_lossy(), "rows": n, "format": fmt.label()}))
    }

    #[tool(description = "Quick stats: session count, turn count, providers seen.")]
    async fn stats(&self) -> Result<CallToolResult, McpError> {
        let sessions = self.store.list_sessions().map_err(|e| err(e.to_string()))?;
        let mut by_provider: std::collections::HashMap<String, u64> = Default::default();
        let mut turns: u64 = 0;
        for s in &sessions {
            *by_provider.entry(s.provider.clone()).or_insert(0) += 1;
            for r in self
                .store
                .iter_session(&s.session_id)
                .map_err(|e| err(e.to_string()))?
            {
                if matches!(r.kind, RecordKind::Turn) {
                    turns += 1;
                }
            }
        }
        ok_json(json!({
            "sessions": sessions.len(),
            "turns": turns,
            "by_provider": by_provider,
            "root": self.store.root.to_string_lossy(),
        }))
    }
}

#[tool_handler]
impl ServerHandler for DistillServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "mcp-distill".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
            instructions: Some(
                "Records prompts/context/responses from Claude (Anthropic Messages) and Codex \
                 (OpenAI Chat Completions) for small-model distillation. \
                 \n\nA recording session is auto-started at server launch — call \
                 `append_turn` (with `message` set to a provider-native message) and it lands in \
                 the active session. The user can call `stop_recording` to halt and \
                 `start_recording` to resume; check `recording_status` to see state. \
                 \n\nFor explicit control, use `start_session` (returns a session_id and makes \
                 it current) or the one-shot `record_interaction`. \
                 Use `export_dataset` to emit JSONL in openai_chat / sharegpt / anthropic format."
                    .into(),
            ),
        }
    }

    async fn initialize(
        &self,
        _request: rmcp::model::InitializeRequestParam,
        _context: RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::InitializeResult, McpError> {
        Ok(self.get_info())
    }
}
