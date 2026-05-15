use mcp_distill::adapters::{claude, codex};
use mcp_distill::exporters::{session_to_export, Format};
use mcp_distill::schema::{Provider, RecordKind, SessionMeta, TurnRecord};
use mcp_distill::storage::{now_rfc3339, Store};
use serde_json::json;
use tempfile::TempDir;

fn fresh_store() -> (TempDir, Store) {
    let tmp = TempDir::new().unwrap();
    let store = Store::new(tmp.path().to_path_buf()).unwrap();
    (tmp, store)
}

fn meta(sid: &str, provider: Provider) -> SessionMeta {
    SessionMeta {
        session_id: sid.into(),
        provider,
        model: Some("test".into()),
        started_at: now_rfc3339(),
        ended_at: None,
        tags: vec![],
        metadata: Default::default(),
    }
}

fn turn_rec(sid: &str, seq: u64, turn: mcp_distill::schema::Turn) -> TurnRecord {
    TurnRecord {
        kind: RecordKind::Turn,
        ts: now_rfc3339(),
        session_id: sid.into(),
        seq,
        turn: Some(turn),
        meta: None,
        usage: None,
    }
}

#[test]
fn store_roundtrip_claude() {
    let (_tmp, store) = fresh_store();
    let m = meta("abc", Provider::Claude);
    store.write_meta(&m).unwrap();
    store
        .write_record(
            "abc",
            &turn_rec(
                "abc",
                1,
                claude::message_to_turn(&json!({"role": "user", "content": "hi"})),
            ),
        )
        .unwrap();
    store
        .write_record(
            "abc",
            &turn_rec(
                "abc",
                2,
                claude::message_to_turn(&json!({
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello!"}],
                })),
            ),
        )
        .unwrap();

    let recs = store.iter_session("abc").unwrap();
    assert!(matches!(recs[0].kind, RecordKind::Meta));
    assert_eq!(
        recs[1].turn.as_ref().unwrap().blocks[0].text.as_deref(),
        Some("hi")
    );
    assert_eq!(
        recs[2].turn.as_ref().unwrap().blocks[0].text.as_deref(),
        Some("hello!")
    );
}

#[test]
fn openai_export_text() {
    let (_tmp, store) = fresh_store();
    store.write_meta(&meta("x", Provider::Codex)).unwrap();
    store
        .write_record(
            "x",
            &turn_rec(
                "x",
                1,
                codex::message_to_turn(&json!({"role": "system", "content": "be brief"})),
            ),
        )
        .unwrap();
    store
        .write_record(
            "x",
            &turn_rec(
                "x",
                2,
                codex::message_to_turn(&json!({"role": "user", "content": "2+2"})),
            ),
        )
        .unwrap();
    store
        .write_record(
            "x",
            &turn_rec(
                "x",
                3,
                codex::response_to_turn(&json!({
                    "choices": [{"message": {"role": "assistant", "content": "4"}}],
                })),
            ),
        )
        .unwrap();

    let recs = store.iter_session("x").unwrap();
    let out = session_to_export(&recs, Format::OpenAiChat);
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "be brief");
    assert_eq!(msgs.last().unwrap()["content"], "4");
}

#[test]
fn openai_export_preserves_tool_calls() {
    let (_tmp, store) = fresh_store();
    store.write_meta(&meta("t", Provider::Codex)).unwrap();
    store
        .write_record(
            "t",
            &turn_rec(
                "t",
                1,
                codex::message_to_turn(&json!({"role": "user", "content": "weather?"})),
            ),
        )
        .unwrap();
    store
        .write_record(
            "t",
            &turn_rec(
                "t",
                2,
                codex::message_to_turn(&json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"sf\"}"},
                    }],
                })),
            ),
        )
        .unwrap();
    store
        .write_record(
            "t",
            &turn_rec(
                "t",
                3,
                codex::message_to_turn(&json!({
                    "role": "tool", "tool_call_id": "c1", "content": "62F",
                })),
            ),
        )
        .unwrap();

    let recs = store.iter_session("t").unwrap();
    let out = session_to_export(&recs, Format::OpenAiChat);
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs[1]["tool_calls"][0]["function"]["name"], "get_weather");
    assert_eq!(msgs[2]["role"], "tool");
    assert_eq!(msgs[2]["tool_call_id"], "c1");
    assert_eq!(msgs[2]["content"], "62F");
}

#[test]
fn anthropic_export_preserves_tool_use_blocks() {
    let (_tmp, store) = fresh_store();
    store.write_meta(&meta("a", Provider::Claude)).unwrap();
    store
        .write_record(
            "a",
            &turn_rec(
                "a",
                1,
                claude::message_to_turn(&json!({"role": "user", "content": "calc"})),
            ),
        )
        .unwrap();
    store
        .write_record(
            "a",
            &turn_rec(
                "a",
                2,
                claude::message_to_turn(&json!({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "calling tool"},
                        {"type": "tool_use", "id": "t1", "name": "calc", "input": {"a": 1, "b": 2}},
                    ],
                })),
            ),
        )
        .unwrap();

    let recs = store.iter_session("a").unwrap();
    let out = session_to_export(&recs, Format::Anthropic);
    let msgs = out["messages"].as_array().unwrap();
    assert_eq!(msgs[1]["content"][1]["type"], "tool_use");
    assert_eq!(msgs[1]["content"][1]["name"], "calc");
    assert_eq!(msgs[1]["content"][1]["input"]["b"], 2);
}
